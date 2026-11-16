#!/usr/bin/env bash
# Reviewer test for chunk 1a.1 (timers SWMM_SOLVE, SWMM_LOG, DIAG_TIME; src/diag_log.h
# resume positioning and checkpoint flush; the write_times leak fix). Uncommitted; the
# coordinator's stamp commit carries it. Not a production path.
#
# Modes:
#   --static   (default) laptop-safe: python3 source checks, no compile, no solver run.
#   --compile  Rivanna only: builds a harness against src/diag_log.h with mpicxx and runs
#              the positioning oracle. Refuses to run unless $TRITON_REVIEW_ON_RIVANNA=1.
#
# Usage: test/test_timer_split_review.sh [--static|--compile] [REPO_ROOT [BASE_REV]]
#
# Every class below is stated as a quantified predicate (for every X in D, P(X)); each
# check has a must-FAIL control that varies the PROPERTY under test (a source fixture or
# a mutant implementation), so a zero from a broken matcher cannot pass as a true absence.
#
# One-time test: this file proves chunk 1a.1 once and is deleted or promoted by the
# coordinator at the landing (B16); it carries no stored blob and no absolute count.
set -u

MODE="--static"
if [ "${1:-}" = "--static" ] || [ "${1:-}" = "--compile" ]; then MODE="$1"; shift; fi
REPO="$(readlink -f "${1:-$(dirname "$0")/..}")"
BASE="${2:-02e82ce}"

if [ "$MODE" = "--static" ]; then
python3 -I - "$REPO" "$BASE" <<'PYEOF'
import re, subprocess, sys

repo, base = sys.argv[1], sys.argv[2]
fails = []


def say(ok, msg):
    """Prints one verdict line and records a failure."""
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def read(path):
    """Reads a repo file with CRLF folded to LF (three of the solver headers use CRLF)."""
    with open(repo + "/" + path, newline="") as f:
        return f.read().replace("\r\n", "\n")


def strip_comments(text):
    """Removes // and /* */ comments, keeping line numbers stable."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.split("\n"))


def code_lines(text):
    """(line number, stripped code) of every non-blank, non-comment line."""
    out = []
    for n, ln in enumerate(strip_comments(text).split("\n"), 1):
        if ln.strip():
            out.append((n, ln.strip()))
    return out


# ---------------------------------------------------------------------------------
# Class 1 (Invariant B, C1). D = every st.start(DIAG_TIME) in src/triton.h.
# P(x): the statement before x stops a category other than DIAG_TIME, nothing between
# x and its st.stop(DIAG_TIME) starts or stops a category, and the statement after
# the stop restarts the category stopped before x. (super_timer categories are
# independent accumulators, so a DIAG bracket opened inside an open category counts
# the same wall time twice.)
# Why required: per-rank Compute/IO would then carry capture time too, and the
# neutrality suite's differential (shape N10) could not attribute it. Kills: a source
# edit that drops the st.stop before the DIAG start, or the restart after it. Second
# correct implementation: a wrapper object that stops and restarts the open category
# around the DIAG work, as long as the same statements appear as a bracket.
# ---------------------------------------------------------------------------------
def diag_bracket_violations(text):
    """Returns one message per DIAG_TIME bracket that breaks the class predicate above."""
    cl = code_lines(text)
    bad = []
    starts = [i for i, (_, s) in enumerate(cl) if s == "st.start(DIAG_TIME);"]
    stops = [i for i, (_, s) in enumerate(cl) if s == "st.stop(DIAG_TIME);"]
    if len(starts) != len(stops):
        bad.append("DIAG_TIME starts %d != stops %d" % (len(starts), len(stops)))
    for i in starts:
        ln = cl[i][0]
        m = re.fullmatch(r"st\.stop\((\w+)\);", cl[i - 1][1]) if i > 0 else None
        if not m or m.group(1) == "DIAG_TIME":
            bad.append("line %d: st.start(DIAG_TIME) is not directly preceded by st.stop(<other category>)" % ln)
            continue
        cat = m.group(1)
        j = next((k for k in stops if k > i), None)
        if j is None:
            bad.append("line %d: st.start(DIAG_TIME) has no st.stop(DIAG_TIME) after it" % ln)
            continue
        inner = [cl[k][1] for k in range(i + 1, j) if re.match(r"st\.(start|stop)\(", cl[k][1])]
        if inner:
            bad.append("line %d: category timer calls inside the DIAG bracket: %s" % (ln, inner))
        nxt = cl[j + 1][1] if j + 1 < len(cl) else ""
        if nxt != "st.start(%s);" % cat:
            bad.append("line %d: DIAG bracket closed but %s is not restarted (next statement: %r)" % (ln, cat, nxt))
    return bad


tri = read("src/triton.h")
v = diag_bracket_violations(tri)
say(not v and "st.start(DIAG_TIME);" in "\n".join(s for _, s in code_lines(tri)),
    "C1: every DIAG_TIME bracket in src/triton.h closes and reopens the category it sits in %s" % v)

# must-fail controls: vary the property (a category left open; a category never reopened)
ctl1 = re.sub(r"[ \t]*st\.stop\(IO_TIME\);\n([ \t]*st\.start\(DIAG_TIME\);)", r"\1", tri, count=1)
say(ctl1 != tri and diag_bracket_violations(ctl1) != [],
    "C1 control: DIAG opened while IO_TIME stays open is refused")
ctl2 = re.sub(r"(st\.stop\(DIAG_TIME\);\n)[ \t]*st\.start\(IO_TIME\);\n", r"\1", tri, count=1)
say(ctl2 != tri and diag_bracket_violations(ctl2) != [],
    "C1 control: DIAG bracket that never reopens IO_TIME is refused")
ctl3 = re.sub(r"(st\.stop\(IO_TIME\);\n[ \t]*st\.start\(DIAG_TIME\);\n)", r"\1          st.start(COMPUTE_TIME);\n", tri, count=1)
say(ctl3 != tri and diag_bracket_violations(ctl3) != [],
    "C1 control: another category started inside the DIAG bracket is refused")


# ---------------------------------------------------------------------------------
# Class 2 (Invariant A, C3 and C8). The checkpoint flush call: it precedes the next
# out.write_output( (the checkpoint cfg writer) in the same conditional context, and
# no open #if of that context names TRITON_SWMM, and its guard names no coupling
# count.
# Why required: a flush after the cfg leaves a resumable checkpoint whose capture
# files may end before t_k; a flush inside the coupled-only region never runs in an
# uncoupled build. Kills: moving the call after out.write_output, inside the TRITON_SWMM
# region, or under the coupling-count guard. Second correct implementation: the flush
# inside a helper called from the same position.
# ---------------------------------------------------------------------------------
def flush_order_violations(text):
    """Returns one message per way the checkpoint flush call site breaks the class predicate."""
    bad = []
    stack = []  # (id, directive text)
    next_id = 0
    ctx = {}    # line number -> tuple of open conditional ids
    texts = {}
    lines = text.split("\n")
    for n, ln in enumerate(lines, 1):
        s = ln.strip()
        if re.match(r"#\s*(if|ifdef|ifndef)\b", s):
            next_id += 1
            stack.append((next_id, s))
            texts[next_id] = s
        elif re.match(r"#\s*endif\b", s):
            if stack:
                stack.pop()
        ctx[n] = tuple(i for i, _ in stack)
    cl = code_lines(text)
    flush = [n for n, s in cl if "diag_log::flush_all()" in s]
    if len(flush) != 1:
        return ["expected exactly one diag_log::flush_all() call site, found %d" % len(flush)]
    f = flush[0]
    w = next((n for n, s in cl if n > f and "out.write_output(" in s), None)
    if w is None:
        return ["no out.write_output( after the flush"]
    if ctx[f] != ctx[w]:
        bad.append("flush (line %d) and out.write_output (line %d) sit in different conditional contexts" % (f, w))
    for i in ctx[f]:
        if "TRITON_SWMM" in texts[i]:
            bad.append("flush sits inside %s" % texts[i])
    guard = [s for n, s in cl if f - 6 <= n <= f and "global_num_of_swmm_links" in s]
    if guard:
        bad.append("flush guarded by the coupling count: %s" % guard)
    return bad


v = flush_order_violations(tri)
say(not v, "C3/C8: checkpoint flush precedes the cfg write in the same context, outside TRITON_SWMM %s" % v)

m = re.search(r"([ \t]*if \(diag_log::any_open\(\)\)\n[ \t]*\{\n(?:.*\n){5}[ \t]*\}\n)", tri)
blk = m.group(1) if m else ""
say(bool(blk), "C3 control setup: the flush if-block was located for fixture construction")
without = tri.replace(blk, "", 1)
after_cfg = re.sub(r"(\tout\.write_output\([^\n]*\n)", lambda mm: mm.group(1) + blk, without, count=1)
say(after_cfg != tri and flush_order_violations(after_cfg) != [],
    "C3 control: the flush moved to after the cfg write is refused")
inside = without.replace("#ifdef TRITON_SWMM\n        // Tie exchange-replay", "#ifdef TRITON_SWMM\n" + blk + "        // Tie exchange-replay", 1)
say(inside != without and flush_order_violations(inside) != [],
    "C8 control: the flush moved inside the #ifdef TRITON_SWMM block is refused")
guarded = tri.replace("if (diag_log::any_open())", "if (swmm_model.global_num_of_swmm_links > 0 && diag_log::any_open())", 1)
say(guarded != tri and flush_order_violations(guarded) != [],
    "C8 control: the flush guarded by the coupling count is refused")


# ---------------------------------------------------------------------------------
# Class 3 (hydraulics neutrality of the default build). D = the added and removed
# code lines of src/triton.h and src/swmm_triton.h over base..HEAD. P(x): an added
# line assigns none of the loop's state variables, and every removed line reappears
# (whitespace-free, trailing punctuation trimmed) inside the added lines, i.e. the
# diff only wraps or re-expresses code and deletes none.
# Why required: the default build must change no hydraulic state; the wrap of
# swmm_step in a timer bracket is the only edit allowed to touch a solver call.
# Kills: an added global_dt/simtime update or a deleted kernel launch. Second correct
# implementation: capturing the swmm_step return code in a differently named local.
# ---------------------------------------------------------------------------------
STATE = r"\b(global_dt|simtime|it_count|print_id|average_dt|cell_size)\b\s*([+\-*/%]?=)(?!=)"


def parse_diff(diff):
    """Splits a unified diff into {path: (added lines, removed lines)}."""
    files, cur = {}, None
    for ln in diff.split("\n"):
        if ln.startswith("+++ b/"):
            cur = ln[6:]
            files[cur] = ([], [])
        elif ln.startswith("---") or ln.startswith("+++"):
            continue
        elif cur and ln.startswith("+"):
            files[cur][0].append(ln[1:])
        elif cur and ln.startswith("-"):
            files[cur][1].append(ln[1:])
    return files


def neutrality_violations(diff, paths=("src/triton.h", "src/swmm_triton.h")):
    """Returns one message per added state assignment or removed-and-not-reappearing code line."""
    bad = []
    for path, (added, removed) in parse_diff(diff).items():
        if path not in paths:
            continue
        a = [s.strip() for s in strip_comments("\n".join(added)).split("\n") if s.strip()]
        r = [s.strip() for s in strip_comments("\n".join(removed)).split("\n") if s.strip()]
        blob = re.sub(r"\s+", "", "".join(a))
        for s in a:
            if re.search(STATE, s):
                bad.append("%s: added line assigns loop state: %s" % (path, s))
        for s in r:
            key = re.sub(r"[),;{]+$", "", re.sub(r"\s+", "", s))
            if key and key not in blob:
                bad.append("%s: removed line does not reappear in the added code: %s" % (path, s))
    return bad


diff = subprocess.run(["git", "-C", repo, "diff", base, "HEAD", "--", "src/triton.h", "src/swmm_triton.h"],
                      capture_output=True, text=True).stdout.replace("\r", "")
say(bool(diff), "C-neutral setup: the base..HEAD diff of triton.h and swmm_triton.h is non-empty (%s)" % base)
v = neutrality_violations(diff)
say(not v, "hydraulics: no state assignment added, no code line removed without reappearing %s" % v)
syn_assign = "+++ b/src/triton.h\n+        global_dt = 0.5 * global_dt;\n"
say(neutrality_violations(syn_assign) != [], "hydraulics control: an added global_dt assignment is refused")
syn_removed = "+++ b/src/triton.h\n-        Kernels::flux_x(rows*cols, rows, cols);\n+        st.start(DIAG_TIME);\n"
say(neutrality_violations(syn_removed) != [], "hydraulics control: a removed kernel launch is refused")
syn_branch = "+++ b/src/swmm_triton.h\n+        simtime += 1.0;\n"
say(neutrality_violations(syn_branch) != [], "hydraulics control: an added simtime update is refused")


# ---------------------------------------------------------------------------------
# Class 4 (the write_times leak fix, erratum E-4 item L). D = every array that
# write_times allocates with new T[size_]. P(a): it is freed with delete[] exactly
# once, at the body's top level, after its last other mention, with no return, exit
# or throw between the first allocation and the last delete[].
# Why required: freeing early reads freed memory in the gathers; freeing twice or on
# one branch only leaks or crashes. Kills: a missing, doubled, early or branch-guarded
# delete[]. Second correct implementation: std::vector<T> in place of new T[size_].
# ---------------------------------------------------------------------------------
def write_times_body(text):
    """Returns the comment-free text of Output::output<T>::write_times, braces included."""
    code = strip_comments(text)
    i = code.index("output<T>::write_times(")
    j = code.index("{", i)
    depth, k = 0, j
    while True:
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return code[j:k + 1]


def delete_violations(text):
    """Returns one message per way the write_times new[]/delete[] pairing breaks the class predicate."""
    body = write_times_body(text)
    bad = []
    allocs = re.findall(r"T\s*\*\s*(\w+)\s*=\s*new\s+T\s*\[", body)
    dels = re.findall(r"delete\[\]\s*(\w+)\s*;", body)
    if not allocs:
        return ["no allocation found in write_times"]
    if set(allocs) != set(dels):
        bad.append("allocated but not freed: %s; freed but not allocated: %s" % (sorted(set(allocs) - set(dels)), sorted(set(dels) - set(allocs))))
    for a in set(dels):
        if dels.count(a) != 1:
            bad.append("%s freed %d times" % (a, dels.count(a)))
    for a in set(allocs) & set(dels):
        pos = [mm.start() for mm in re.finditer(r"\b%s\b" % a, body)]
        d0 = [mm.start() for mm in re.finditer(r"delete\[\]\s*%s\s*;" % a, body)][0]
        dn = d0 + body[d0:].index(a)
        if any(p > dn for p in pos):
            bad.append("%s is mentioned after its delete[]" % a)
        depth = body[:d0].count("{") - body[:d0].count("}")
        if depth != 1:
            bad.append("%s freed at brace depth %d, not the body's top level" % (a, depth))
    first = min(mm.start() for mm in re.finditer(r"new\s+T\s*\[", body))
    last = max([mm.start() for mm in re.finditer(r"delete\[\]", body)] or [first])
    if re.search(r"\b(return|exit|throw)\b", body[first:last]):
        bad.append("a return, exit or throw sits between the allocations and the last delete[]")
    return bad


out = read("src/output.h")
v = delete_violations(out)
say(not v, "leak fix: every write_times array is freed once, after its last use, at top level %s" % v)
dropped = re.sub(r"\n[ \t]*delete\[\] swmm_log_time_all;", "", out, count=1)
say(dropped != out and delete_violations(dropped) != [], "leak control: a missing delete[] is refused")
doubled = out.replace("delete[] diag_time_all;", "delete[] diag_time_all;\n\t\tdelete[] diag_time_all;", 1)
say(doubled != out and delete_violations(doubled) != [], "leak control: a double delete[] is refused")
early = out.replace("\t\tT *compute_time_all = new T[size_];", "\t\tT *compute_time_all = new T[size_];\n\t\tdelete[] compute_time_all;", 1)
say(early != out and delete_violations(early) != [], "leak control: a delete[] before the last use is refused")
returned = out.replace("\t\t// Every array above is read only before this point", "\t\tif (rank_ == 99) return;\n\t\t// Every array above is read only before this point", 1)
say(returned != out and delete_violations(returned) != [], "leak control: an early return between allocation and delete[] is refused")


# ---------------------------------------------------------------------------------
# Class 5 (printed layout). D = every column name of the performance.txt header.
# P(c): the per-rank row and the Average row print the variable for c, in header
# order, and section G's expected header string equals the header literal.
# Why required: readers select columns by name but the printed file is positional; a
# swapped pair prints DIAG under Other with no error. Kills: swapping two columns in
# either row printer or in section G's string. Second correct implementation: a table
# of (name, array) pairs that prints the header and both rows from one list.
# ---------------------------------------------------------------------------------
def column_var(name):
    """Maps a performance.txt column name to the gathered array that prints it."""
    special = {"Compute": "compute_time_all", "MPI": "mpi_time_all", "IO": "io_time_all",
               "Resize": "resize_time_all", "SWMM": "swmm_time_all", "DIAG": "diag_time_all",
               "Other": "other_time_all", "Simulation": "simulation_time_all",
               "Init": "init_time_all", "Total": "total_time_all"}
    return special.get(name, name.lower() + "_time_all")


def layout_violations(text, suite):
    """Returns one message per column the header, the two row printers or section G disagree on."""
    body = write_times_body(text)
    bad = []
    hm = re.search(r'output << "(%Rank[^"]*)"', body)
    if not hm:
        return ["header literal not found"]
    header = hm.group(1)
    cols = [c.strip() for c in header.lstrip("%").split(",")][1:]
    want = [column_var(c) for c in cols]
    row = re.search(r"for\(int j=0;j<size_;j\+\+\)\{(.*?)\n\t\t\t\}", body, re.S)
    got_row = re.findall(r"(\w+_all)\[j\]", row.group(1)) if row else []
    if got_row != want:
        bad.append("per-rank row prints %s, header wants %s" % (got_row, want))
    avg = re.search(r'"Average"(.*?)std::endl;', body, re.S)
    got_avg = re.findall(r"average\((\w+_all),size_\)", avg.group(1)) if avg else []
    if got_avg != want:
        bad.append("Average row prints %s, header wants %s" % (got_avg, want))
    sm = re.search(r"^G_HEADER='([^']*)'", suite, re.M)
    if not sm or sm.group(1) != header:
        bad.append("section G header string differs from the header literal")
    return bad


suite = read("test/coupled_resume_suite.sh")
v = layout_violations(out, suite)
say(not v, "layout: header, per-rank row, Average row and section G agree column by column %s" % v)
swapped = out.replace("<< swmm_solve_time_all[j] << \", \" << swmm_log_time_all[j]", "<< swmm_log_time_all[j] << \", \" << swmm_solve_time_all[j]", 1)
say(swapped != out and layout_violations(swapped, suite) != [], "layout control: two swapped per-rank columns are refused")
avgswap = out.replace("average(diag_time_all,size_) << \", \" <<  average(other_time_all,size_)", "average(other_time_all,size_) << \", \" <<  average(diag_time_all,size_)", 1)
say(avgswap != out and layout_violations(avgswap, suite) != [], "layout control: two swapped Average columns are refused")
sh_bad = suite.replace("SWMM_SOLVE, SWMM_LOG, DIAG, Other", "SWMM_LOG, SWMM_SOLVE, DIAG, Other", 1)
say(sh_bad != suite and layout_violations(out, sh_bad) != [], "layout control: a reordered section G header string is refused")

print("")
print("static: %d failure(s)" % len(fails))
sys.exit(1 if fails else 0)
PYEOF
exit $?
fi

# ------------------------------------------------------------------------------------
# --compile: the positioning oracle. Rivanna only.
# ------------------------------------------------------------------------------------
if [ "${TRITON_REVIEW_ON_RIVANNA:-0}" != "1" ]; then
  echo "refused: --compile builds and runs binaries; run it on Rivanna with TRITON_REVIEW_ON_RIVANNA=1" >&2
  exit 2
fi
command -v mpicxx >/dev/null 2>&1 || { echo "refused: mpicxx not on PATH (load gcc and openmpi first)" >&2; exit 2; }
W="${TRITON_REVIEW_WORK:-$(mktemp -d)}"
mkdir -p "$W/data"
cat > "$W/oracle.cpp" <<'CPPEOF'
// Positioning oracle for src/diag_log.h.
//
// Invariant (shape rule 3): for every capture file shape, every kill point and every
// checkpoint time t_k, positioning leaves exactly the header plus the COMPLETE records
// stamped <= t_k, as bytes; and a resumed writer that appends the remaining records
// leaves the file byte-identical to a clean run's.
//
// D = {fixed 48-byte records, the T1a variable-block shape (48 fixed bytes plus n*4),
//      an interval csv with its time column first, an interval csv with it last}
//     x {0, 1, 3 complete records} x {every cut length 0..record_len of the next record}
//     x {t_k below, at and between stamps, before the first record, after the last}.
// The oracle is computed here from the clean record list, never from diag_log.h.
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include "diag_log.h"

using namespace diag_log;
typedef std::vector<char> bytes;

static std::string dir;
// Path of a scratch file inside the harness work directory.
static std::string P(const char* n) { return dir + "/" + n; }

// One clean-run record: its end-time stamp and its exact bytes.
struct rec { double stamp; bytes data; };

// Appends n raw bytes to a byte vector.
static void put(bytes& b, const void* p, size_t n) { const char* c = (const char*)p; b.insert(b.end(), c, c + n); }

// The 16-byte binary header the writer produces, taken from diag_log::write_header.
static bytes bin_header(int32_t magic, int32_t n)
{
	std::ostringstream os;
	write_header(os, magic, n);
	const std::string s = os.str();
	return bytes(s.begin(), s.end());
}

// the T1a-shaped record of shape section 2: {f64 simtime, f64 dt, i32 trials, i32 n_nonconv,
// f64 p3_pos, f64 p3_neg, f64 link_volume} (48 bytes) then n_nonconv int32 node indices
static rec t1a_rec(int i)
{
	static const int ns[] = {0, 3, 0, 17, 1, 5, 0, 2};
	const int n = ns[i % 8];
	rec r; r.stamp = 1.0 + 0.5 * i;
	const double dt = 0.5, pp = 1.5 * i, pn = -0.25 * i, lv = 100.0 + i;
	const int32_t trials = 3 + i, nn = n;
	put(r.data, &r.stamp, 8); put(r.data, &dt, 8); put(r.data, &trials, 4); put(r.data, &nn, 4);
	put(r.data, &pp, 8); put(r.data, &pn, 8); put(r.data, &lv, 8);
	for (int k = 0; k < n; k++) { const int32_t id = 7 * i + k; put(r.data, &id, 4); }
	return r;
}

// A 48-byte fixed record: the stamp then five doubles (the dt_limiter / edge_flux stride).
static rec fixed_rec(int i)
{
	rec r; r.stamp = 1.0 + 0.5 * i;
	put(r.data, &r.stamp, 8);
	for (int k = 0; k < 5; k++) { const double v = i * 10.0 + k; put(r.data, &v, 8); }
	return r;   // 48 bytes
}

// One interval csv row, with the time column first or last.
static rec csv_rec(int i, bool time_first)
{
	char buf[64];
	const double t = 1.0 + 0.5 * i;
	if (time_first) std::snprintf(buf, sizeof(buf), "%.1f,%d,%d\n", t, 3 * i, 5 * i);
	else            std::snprintf(buf, sizeof(buf), "%d,%d,%.1f\n", 3 * i, 5 * i, t);
	rec r; r.stamp = t; put(r.data, buf, std::strlen(buf));
	return r;
}

// Replaces a file with the given bytes.
static void write_file(const std::string& p, const bytes& b)
{
	std::ofstream o(p, std::ios::binary | std::ios::trunc);
	o.write(b.data(), (std::streamsize)b.size());
}

// Reads a whole file as bytes.
static bytes read_file(const std::string& p)
{
	std::ifstream in(p, std::ios::binary);
	return bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The two file families the positioners handle.
enum kind_t { BIN, CSV };

// One capture-file shape: how to generate record i, the header, and how diag_log reads it.
struct shape_t
{
	const char* name;
	kind_t kind;
	std::function<rec(int)> make;
	bytes header;
	std::string header_row;   // csv
	int time_col;             // csv
	record_reader reader;     // bin
	int32_t magic, n_items;
};

typedef std::function<position_result(const shape_t&, const std::string&, double)> positioner_t;

// The implementation under review: diag_log::position_binary or position_text.
static position_result real_pos(const shape_t& s, const std::string& path, double t)
{
	return s.kind == BIN ? position_binary(path, s.magic, s.n_items, s.reader, t)
	                     : position_text(path, s.header_row, s.time_col, t);
}

// mutant 1: the checkpoint bound treated as exclusive (a record stamped exactly t_k is
// "after" the checkpoint). Varies the inclusive-at-equality property of the bound.
static position_result mut_exclusive(const shape_t& s, const std::string& path, double t) { return real_pos(s, path, t - 5e-4); }

// mutant 2: a partial block accepted (the block's short read is not an incomplete record).
static position_result mut_lenient_block(const shape_t& s, const std::string& path, double t)
{
	if (s.kind != BIN) return real_pos(s, path, t);
	record_reader lenient = [](std::istream& in, double& stamp) -> bool {
		std::vector<char> buf(48);
		in.read(buf.data(), 48);
		if (in.gcount() != 48) return false;
		int32_t n = 0; std::memcpy(&n, buf.data() + 20, 4);
		std::memcpy(&stamp, buf.data(), 8);
		std::vector<char> blk((size_t)n * 4 + 1);
		in.read(blk.data(), (std::streamsize)n * 4);
		in.clear();      // a short block is tolerated: the partial record is kept
		return true;
	};
	return position_binary(path, s.magic, s.n_items, lenient, t);
}

// mutant 3: the reader parameters the coder's hand-off note gives for the T1a shape
// (24 fixed bytes), applied to the 48-byte record the shape specifies.
static position_result mut_note_params(const shape_t& s, const std::string& path, double t)
{
	if (s.kind != BIN) return real_pos(s, path, t);
	// counted_block_record_reader(24, 20, 4) with one guard added by this harness: a count whose
	// block is over 1 MiB is read as an incomplete record, because a walk that is off the record
	// stride reads garbage counts of up to 8 GB and would allocate them before reading.
	record_reader note = [](std::istream& in, double& stamp) -> bool {
		std::vector<char> buf(24);
		in.read(buf.data(), 24);
		if (in.gcount() != 24) return false;
		int32_t n = 0; std::memcpy(&n, buf.data() + 20, 4);
		if (n < 0 || (size_t)n * 4 > (1u << 20)) return false;
		std::memcpy(&stamp, buf.data(), 8);
		std::vector<char> blk((size_t)n * 4);
		if (n > 0) { in.read(blk.data(), (std::streamsize)blk.size()); if (in.gcount() != (std::streamsize)blk.size()) return false; }
		return true;
	};
	return position_binary(path, s.magic, s.n_items, note, t);
}

// mutant 4 (csv): the time read from the wrong column.
static position_result mut_wrong_col(const shape_t& s, const std::string& path, double t)
{
	if (s.kind != CSV) return real_pos(s, path, t);
	return position_text(path, s.header_row, s.time_col == 0 ? 1 : 0, t);
}

// Case and mismatch counts of one oracle run.
struct tally { long cases = 0, mismatches = 0; };

// Runs the oracle for one shape against one positioner over every kill point and checkpoint time.
static void run_shape(const shape_t& s, const positioner_t& pos, tally& tl, bool verbose, bool with_resume)
{
	const int complete_counts[] = {0, 1, 3};
	for (int c : complete_counts)
	{
		std::vector<rec> all;
		for (int i = 0; i < c + 2; i++) all.push_back(s.make(i));   // c complete, record c cut, record c+1 never written
		for (size_t cut = 0; cut <= all[c].data.size(); cut++)
		{
			bytes file = s.header;
			for (int i = 0; i < c; i++) file.insert(file.end(), all[i].data.begin(), all[i].data.end());
			file.insert(file.end(), all[c].data.begin(), all[c].data.begin() + cut);
			const int complete = c + (cut == all[c].data.size() ? 1 : 0);

			std::vector<double> ts = {-1.0, 1e9};
			for (int i = 0; i < c + 2; i++) { ts.push_back(all[i].stamp - 0.25); ts.push_back(all[i].stamp); ts.push_back(all[i].stamp + 0.25); }
			for (double t : ts)
			{
				int keep = 0;
				for (int i = 0; i < complete; i++) if (all[i].stamp <= t) keep++;
				bytes expect = s.header;
				for (int i = 0; i < keep; i++) expect.insert(expect.end(), all[i].data.begin(), all[i].data.end());

				write_file(P("case.dat"), file);
				const position_result r = pos(s, P("case.dat"), t);
				const bytes got = read_file(P("case.dat"));
				bool ok = (r.status == position_status::ok) && (r.records == keep) && (got == expect);

				if (ok && with_resume)
				{
					// resume: the live segment re-generates every record the clean run holds after t_k
					write_file(P("case.dat"), file);
					capture_file cf;
					if (s.kind == BIN) cf.declare_binary(P("case.dat"), s.magic, s.n_items, s.reader);
					else cf.declare_text(P("case.dat"), s.header_row, s.time_col);
					const long kept = cf.open_resume(t);
					for (int i = keep; i < c + 2; i++) cf.append(all[i].data.data(), all[i].data.size());
					cf.close();
					bytes clean = s.header;
					for (int i = 0; i < c + 2; i++) clean.insert(clean.end(), all[i].data.begin(), all[i].data.end());
					ok = (kept == keep) && (read_file(P("case.dat")) == clean);
				}
				tl.cases++;
				if (!ok)
				{
					tl.mismatches++;
					if (verbose && tl.mismatches <= 3)
						std::printf("  mismatch: shape=%s complete=%d cut=%zu t=%g -> status=%d records=%ld (want %d) bytes=%zu (want %zu)\n",
						            s.name, c, cut, t, (int)r.status, r.records, keep, got.size(), expect.size());
				}
			}
		}
	}
}

// argv[1] is a work directory; exits 0 only when the real positioner matches the oracle on every
// case and every mutant is caught.
int main(int, char** argv)
{
	dir = argv[1];
	std::vector<shape_t> shapes;
	{
		shape_t s; s.name = "fixed48"; s.kind = BIN; s.make = fixed_rec; s.magic = MAGIC_EDGE_FLUX; s.n_items = 4;
		s.header = bin_header(s.magic, s.n_items); s.reader = fixed_record_reader(48); shapes.push_back(s);
	}
	{
		shape_t s; s.name = "t1a_block"; s.kind = BIN; s.make = t1a_rec; s.magic = MAGIC_SWMM_STEP_DIAG; s.n_items = 0;
		s.header = bin_header(s.magic, s.n_items); s.reader = counted_block_record_reader(48, 20, 4); shapes.push_back(s);
	}
	for (int tf = 1; tf >= 0; tf--)
	{
		shape_t s; s.name = tf ? "csv_time_first" : "csv_time_last"; s.kind = CSV;
		s.make = [tf](int i) { return csv_rec(i, tf != 0); };
		s.header_row = tf ? "%t_end, a, b" : "%a, b, t_end"; s.time_col = tf ? 0 : 2;
		const std::string h = s.header_row + "\n"; s.header = bytes(h.begin(), h.end());
		shapes.push_back(s);
	}

	long real_bad = 0;
	for (const shape_t& s : shapes)
	{
		tally tl; run_shape(s, real_pos, tl, true, true);
		std::printf("REAL    %-16s cases=%ld mismatches=%ld\n", s.name, tl.cases, tl.mismatches);
		real_bad += tl.mismatches;
	}

	// must-fail arms: each mutant must be caught (mismatches > 0) on the shapes it targets
	struct mut_t { const char* name; positioner_t pos; const char* shape; };
	const mut_t muts[] = {
		{"exclusive_bound", mut_exclusive, "fixed48"}, {"exclusive_bound", mut_exclusive, "t1a_block"},
		{"exclusive_bound", mut_exclusive, "csv_time_first"},
		{"lenient_partial_block", mut_lenient_block, "t1a_block"},
		{"note_params_24_fixed_bytes", mut_note_params, "t1a_block"},
		{"wrong_time_column", mut_wrong_col, "csv_time_first"}, {"wrong_time_column", mut_wrong_col, "csv_time_last"},
	};
	long uncaught = 0;
	for (const mut_t& m : muts)
		for (const shape_t& s : shapes)
			if (std::strcmp(s.name, m.shape) == 0)
			{
				tally tl; run_shape(s, m.pos, tl, false, false);
				std::printf("MUTANT  %-28s on %-16s cases=%ld caught=%ld\n", m.name, s.name, tl.cases, tl.mismatches);
				if (tl.mismatches == 0) uncaught++;
			}

	std::printf("oracle: real mismatches=%ld, uncaught mutants=%ld\n", real_bad, uncaught);
	return (real_bad == 0 && uncaught == 0) ? 0 : 1;
}
CPPEOF
mpicxx -std=c++17 -Wall -Wextra -I "$REPO/src" "$W/oracle.cpp" -o "$W/oracle" || { echo "FAIL compile of the oracle harness"; exit 1; }
"$W/oracle" "$W/data"
exit $?
