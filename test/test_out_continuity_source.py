#!/usr/bin/env python3
"""Source-structural checks for WP-1C -- full-window SWMM output continuity.

Usage:  test_out_continuity_source.py <repo-root>

WHY THIS TIER EXISTS, stated rather than assumed.  Three of WP-1C's five
properties cannot be EXECUTED without an open model: ``output_openOutFile`` has
internal linkage, and the prologue comparison and the positioning arithmetic
both read ``OutputStartPos`` / ``BytesPerPeriod``, which are file-scope statics
that only ``output_open`` sets -- and ``output_open`` needs an ``.inp``.  A
Tier-1 test that called them anyway would be exercising a zero-valued stride
and reporting it as coverage.  So the behavioural arms live in the compiled
tier (``test/snapshot/test_state_snapshot.cpp``) and in Tier 2, and the
STRUCTURAL arms live here, in the same shape the campaign already uses for
``test_swmm_column_split_source.py`` and ``test_compute_timer_fence.py``.

EVERY CHECK CARRIES A DEFECT PROBE.  A structural check reads source text, so
the way it fails is by silently matching nothing.  Each ``S*`` check below is
paired with an ``X*`` probe that mutates the source IN MEMORY into the defect
form the check exists to catch and asserts the check goes RED.  A check whose
probe does not fire is reported as VACUOUS and fails the run -- that is the
only thing separating this file from a set of greps that agree with themselves.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

failures: list[str] = []
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name.split()[0]] = ok
    if ok:
        print("ok    %s" % name)
    else:
        failures.append("%s: %s" % (name, detail))
        print("FAIL  %s   %s" % (name, detail))


# ---------------------------------------------------------------------------
# The predicates. Each takes the relevant source text and returns a bool, so
# the same function serves the live check and its defect probe -- a probe that
# re-implemented the predicate would be testing a second copy of it.
# ---------------------------------------------------------------------------

def p_open_is_existence_keyed(out_c: str) -> bool:
    """D-OUT3: the mode is chosen from the file EXISTING, and both arms are present."""
    body = _func_body(out_c, "output_openOutFile")
    return ('"r+b"' in body) and ('"w+b"' in body) and ("OutputReopened" in body)


def p_no_unconditional_truncating_open(out_c: str) -> bool:
    """The truncating open must not be the only/unguarded one it used to be."""
    body = _func_body(out_c, "output_openOutFile")
    # the surviving "w+b" must sit under a NULL-handle guard, not at top level
    m = re.search(r'if\s*\(\s*Fout\.file\s*==\s*NULL\s*\)\s*\{[^}]*"w\+b"', body, re.S)
    return m is not None


def p_signal_is_file_local(repo: Path) -> bool:
    """D-OUT3 acceptance: zero references to the resume signal outside output.c."""
    hits = _grep_tree(repo, "OutputReopened")
    return all(p.name == "output.c" for p in hits)


def p_one_arithmetic_root(repo: Path, out_c: str) -> bool:
    """D-OUT2 acceptance: BytesPerPeriod never leaves output.c, and the resume
    product is formed in exactly one function."""
    # SCOPED TO THE SOLVER, on a named ground rather than to make this pass.
    # external/swmm/src/outfile/ is the INDEPENDENT reader library, and its
    # `p_data->BytesPerPeriod` is a struct MEMBER it re-derives from the file's
    # own header (swmm_output.c:231) -- a different symbol, in a different
    # module, predating WP-1C. D-OUT2 is about the SOLVER's file-scope static
    # not escaping output.c, and that is what this measures.
    outside = [p for p in _grep_tree(repo, "BytesPerPeriod")
               if p.name != "output.c" and "outfile" not in p.parts]
    roots = re.findall(r"OutputStartPos\s*\+\s*\(F_OFF\)Nperiods\s*\*\s*BytesPerPeriod", out_c)
    return not outside and len(roots) == 1


def p_one_truncation_wrapper(repo: Path, out_c: str) -> bool:
    """D-OUT4 acceptance: one wrapper, and no call site names a platform."""
    outside = [p for p in _grep_tree(repo, "F_TRUNC") if p.name != "output.c"]
    defs = re.findall(r"^\s*#define\s+F_TRUNC\(", out_c, re.M)
    raw = re.findall(r"\b(?:ftruncate|_chsize_s)\s*\(", out_c)
    # the two raw names appear ONLY inside the two #define arms
    return (not outside) and len(defs) == 2 and len(raw) == 2


def p_truncate_is_in_output_end(out_c: str) -> bool:
    """Requirement 8: the close-time truncate lives in output_end, after the
    sixth trailer write."""
    body = _func_body(out_c, "output_end")
    if "output_truncateAt" not in body:
        return False
    # it must come after the MAGICNUMBER trailer write, not before it
    return body.index("MAGICNUMBER") < body.index("output_truncateAt")


def p_truncate_is_not_at_the_fclose_site(swmm5_c: str) -> bool:
    """Requirement 8's REFUSED relocation. swmm_report runs between output_end
    and fclose and leaves the stream deep inside the payload, so a truncate at
    the reported position there cuts the file mid-record. Its dependent is
    every external .out reader, which locates the trailer by seeking backwards
    from SEEK_END."""
    body = _func_body(swmm5_c, "swmm_close")
    return ("F_TRUNC" not in body) and ("output_truncateAt" not in body) \
        and ("ftruncate" not in body) and ("_chsize" not in body)


def p_truncation_is_flushed_first(out_c: str) -> bool:
    """output_close performs no flush and the only one in the vendored tree is
    inside fclose, so a descriptor-level truncate must flush for itself."""
    body = _func_body(out_c, "output_truncateAt")
    return "fflush" in body and body.index("fflush") < body.index("F_TRUNC")


def p_prologue_is_compared_not_rewritten(out_c: str) -> bool:
    """D-OUT1 / requirement 2: on a reopened file the prologue is compared and
    a disagreement is refused, rather than written over a payload the OLD
    prologue describes."""
    body = _func_body(out_c, "output_open")
    return ("output_comparePrologue" in body) and ("ERR_OUT_WRITE" in body) \
        and ("tmpfile" in body)


def p_positioning_is_restore_path_only(swmm_h: str) -> bool:
    """Chunk (3): the positioning call is on the restore path and NOT on the
    replay fallback, which restores no Nperiods and so has no operand for it."""
    restore = _func_body(swmm_h, "swmm_triton::try_restore_state_snapshot")
    replay = _func_body(swmm_h, "swmm_triton::replay_exchange_history")
    return ("output_positionForResume" in restore) and \
           ("output_positionForResume" not in replay)


def p_flush_precedes_the_snapshot_write(swmm_h: str) -> bool:
    """Requirement 4: the period count the snapshot records must be a claim
    about bytes that are on disk."""
    body = _func_body(swmm_h, "swmm_triton::write_state_snapshot")
    return "output_flush" in body and body.index("output_flush") < body.index("snapshot_save")


def p_caller_aborts_on_unreadable(triton_h: str) -> bool:
    """Fail-fast: ABSENT falls back, everything else aborts -- and the abort is
    MPI_Abort, because this block runs under a rank-0 guard and a rank-0-only
    exit() would hang every other rank at the next collective."""
    win = _window(triton_h, "try_restore_state_snapshot", 40)
    return ("SR::Absent" in win) and ("replay_exchange_history" in win) \
        and ("MPI_Abort" in win) and ("exit(EXIT_FAILURE);" not in win)


def p_resume_event_has_four_fields(swmm_h: str) -> bool:
    """Chunk (5): checkpoint id, resume time, path, and (replay only) reason --
    appended AFTER the `to t=` token so the existing numeric parse cannot move."""
    restore = _func_body(swmm_h, "swmm_triton::try_restore_state_snapshot")
    replay = _func_body(swmm_h, "swmm_triton::replay_exchange_history")
    ok_r = ("SWMM state restored from snapshot to t=" in restore
            and "resume-event checkpoint=" in restore
            and "path=snapshot" in restore
            and restore.index("to t=") < restore.index("resume-event"))
    ok_p = ("SWMM exchange history replayed to t=" in replay
            and "resume-event checkpoint=" in replay
            and "path=replay reason=" in replay
            and replay.index("to t=") < replay.index("resume-event"))
    return ok_r and ok_p


def p_reason_vocabulary_is_the_contract(swmm_h: str) -> bool:
    """The three classified reasons, and no fourth invented one."""
    body = _func_body(swmm_h, "swmm_triton::classify_missing_snapshot")
    returns = set(re.findall(r'return\s+"([a-z-]+)"\s*;', body))
    return returns == {"absent", "retention-collision", "pre-snapshot-checkpoint"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    """Blanks // and /* */ comments, preserving offsets and string literals.

    LOAD-BEARING, and it is the second instrument defect this file paid for.
    Every ordering predicate below asks "does X appear before Y in the code",
    and a comment ABOUT the code answers that question wrongly: the resume-event
    comment names both `to t=` and `resume-event` in the opposite order to the
    emitted statement, so S13 went red on correct source. Blanking rather than
    deleting keeps every index meaningful."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                out.append(src[i])
                if src[i] == "\\" and i + 1 < n:
                    out.append(src[i + 1]); i += 2; continue
                if src[i] == q:
                    i += 1; break
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out.append(" "); i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                out.append("\n" if src[i] == "\n" else " "); i += 1
            out.append("  "); i += 2
            continue
        out.append(c); i += 1
    return "".join(out)


def _func_body(src: str, name: str) -> str:
    """Brace-matched body of a DEFINITION, with comments blanked. Returns ''
    when absent, which every caller treats as a failure, not as an empty pass.

    THREE THINGS THIS GOT WRONG ON FIRST WRITING, each of which turned a
    correct source into a red check AND made the paired probe report CAUGHT
    vacuously -- the exact shape of a test agreeing with itself:
      (1) it matched the FORWARD DECLARATION and then brace-matched from the
          next `{` anywhere in the file, returning an unrelated body;
      (2) it anchored at column 0, so every tab-indented C++ member definition
          in swmm_triton.h was invisible;
      (3) it demanded `{` immediately after the parameter list -- but the EPA
          house style puts the whole doc comment BETWEEN the signature and the
          brace, and a trailing `const` sits there on a C++ member.
    """
    code = _strip_comments(src)
    for m in re.finditer(r"^[ \t]*[A-Za-z_][\w:<>\*\s&]*?\b" + re.escape(name)
                         + r"\s*\(", code, re.M):
        k = code.index("(", m.end() - 1)
        depth = 0
        while k < len(code):
            if code[k] == "(":
                depth += 1
            elif code[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        # Skip whitespace and the trailing qualifiers a definition may carry.
        # UNBOUNDED rather than a fixed window: _strip_comments BLANKS the EPA
        # doc comment that sits between the signature and the brace, and that
        # blank run is longer than any window worth guessing -- a 200-character
        # one left `tail` empty and silently rejected every correct definition
        # in output.c.
        mt = re.match(r"\s*(?:(?:const|noexcept|override|final)\s*)*",
                      code[k + 1:])
        tail = code[k + 1 + mt.end():]
        if not tail.startswith("{"):
            continue                      # a declaration or a prototype
        i = code.index("{", k)
        depth, j = 0, i
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    return code[i:j + 1]
            j += 1
    return ""


def _window(src: str, token: str, lines: int) -> str:
    idx = src.find(token)
    if idx < 0:
        return ""
    rest = src[idx:].splitlines()[:lines]
    return "\n".join(rest)


def _inject_into(src: str, func: str, needle: str, repl: str) -> str:
    """Mutate `needle` only INSIDE `func`'s body, so a probe cannot land in a
    same-named statement in a neighbouring function and prove nothing."""
    body = _func_body(src, func)
    if not body or needle not in body:
        return src
    i = _strip_comments(src).index(body)
    return src[:i] + body.replace(needle, repl, 1) + src[i + len(body):]


_SKIP_DIRS = {".git", "build", "external/kokkos"}


def _grep_tree(repo: Path, token: str) -> list[Path]:
    hits = []
    for base in ("src", "external/swmm/src", "test"):
        root = repo / base
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in (".c", ".h", ".cpp", ".hpp"):
                continue
            try:
                # COMMENTS ARE NOT CODE, and this check exists to find a symbol
                # ESCAPING a translation unit -- a name mentioned in prose has
                # escaped nothing. Caught by this file's own two-state probe:
                # a comment in test_state_snapshot.cpp naming
                # "OutputStartPos and BytesPerPeriod" turned S4 red, and the
                # tempting repair was to reword the comment. That would have
                # been a source edited to satisfy a broken instrument.
                if token in _strip_comments(p.read_text(errors="replace")):
                    hits.append(p)
            except OSError:
                pass
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: %s <repo-root>" % argv[0])
        return 2
    repo = Path(argv[1]).resolve()
    solver = repo / "external" / "swmm" / "src" / "solver"
    out_c = (solver / "output.c").read_text(errors="replace")
    swmm5_c = (solver / "swmm5.c").read_text(errors="replace")
    swmm_h = (repo / "src" / "swmm_triton.h").read_text(errors="replace")
    triton_h = (repo / "src" / "triton.h").read_text(errors="replace")

    # --- S: the live checks -------------------------------------------------
    check("S1  the .out open is existence-keyed and carries both modes (D-OUT3)",
          p_open_is_existence_keyed(out_c), "output_openOutFile no longer chooses its mode")
    check("S2  the truncating open survives only under a NULL-handle guard",
          p_no_unconditional_truncating_open(out_c), "an unguarded w+b is back")
    check("S3  the resume signal is file-local (D-OUT3 acceptance)",
          p_signal_is_file_local(repo), "OutputReopened is referenced outside output.c")
    check("S4  the resume arithmetic has exactly one root (D-OUT2 acceptance)",
          p_one_arithmetic_root(repo, out_c),
          "BytesPerPeriod escaped output.c, or the product is formed twice")
    check("S5  one truncation wrapper, no platform named at a call site (D-OUT4)",
          p_one_truncation_wrapper(repo, out_c), "F_TRUNC escaped, or a raw primitive is called directly")
    check("S6  the close-time truncate is in output_end, after the trailer (req 8)",
          p_truncate_is_in_output_end(out_c), "absent, or placed before the trailer write")
    check("S7  it is NOT at the fclose site, where the stream sits mid-payload",
          p_truncate_is_not_at_the_fclose_site(swmm5_c),
          "a truncate at swmm_close would cut the file mid-record (error 435 for every reader)")
    check("S8  the truncation flushes first",
          p_truncation_is_flushed_first(out_c), "it would act on unflushed buffered bytes")
    check("S9  the prologue is compared and refused, not rewritten (D-OUT1, req 2)",
          p_prologue_is_compared_not_rewritten(out_c), "the comparison is gone")
    check("S10 positioning is on the restore path ONLY (chunk 3)",
          p_positioning_is_restore_path_only(swmm_h),
          "it leaked onto the replay fallback, where the expression has no operand")
    check("S11 Fout is flushed before the snapshot write (req 4)",
          p_flush_precedes_the_snapshot_write(swmm_h), "the recorded period count is not a claim about disk")
    check("S12 the caller aborts on a non-absent refusal, via MPI_Abort",
          p_caller_aborts_on_unreadable(triton_h), "a silent fallback is back, or a rank-0-only exit() would hang")
    check("S13 the resume-event record carries four fields, after the t= token",
          p_resume_event_has_four_fields(swmm_h), "a field is missing, or it precedes the parsed numeric")
    check("S14 the reason vocabulary is exactly the pinned three",
          p_reason_vocabulary_is_the_contract(swmm_h), "a reason was invented or dropped")

    # --- X: the defect probes ------------------------------------------------
    #
    # Each mutates the source IN MEMORY into the form the paired check exists
    # to catch. A probe that does not fire means the check is VACUOUS, which is
    # reported as a failure of THIS FILE rather than of the tree.
    print("")
    probes = [
        ("X1  -> S1", lambda: p_open_is_existence_keyed(
            out_c.replace('"r+b"', '"w+b"'))),
        ("X2  -> S2", lambda: p_no_unconditional_truncating_open(
            re.sub(r"if\s*\(\s*Fout\.file\s*==\s*NULL\s*\)", "if (1)", out_c))),
        ("X4  -> S4", lambda: p_one_arithmetic_root(
            repo, out_c + "\nF_OFF dup(void){return OutputStartPos + (F_OFF)Nperiods * BytesPerPeriod;}\n")),
        ("X5  -> S5", lambda: p_one_truncation_wrapper(
            repo, out_c + "\nint raw(int fd){return ftruncate(fd, 0);}\n")),
        ("X6  -> S6", lambda: p_truncate_is_in_output_end(
            out_c.replace("output_truncateAt(output_payloadEndPos()", "no_truncate((", 1))),
        ("X7  -> S7", lambda: p_truncate_is_not_at_the_fclose_site(
            _inject_into(swmm5_c, "swmm_close", "fclose(Fout.file)",
                         "F_TRUNC(0,0); fclose(Fout.file)"))),
        ("X8  -> S8", lambda: p_truncation_is_flushed_first(
            out_c.replace("if ( fflush(Fout.file) != 0 ) return 1;\n    return F_TRUNC",
                          "return F_TRUNC", 1))),
        ("X9  -> S9", lambda: p_prologue_is_compared_not_rewritten(
            out_c.replace("output_comparePrologue", "nope"))),
        ("X10 -> S10", lambda: p_positioning_is_restore_path_only(
            swmm_h.replace("const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/true);",
                           "output_positionForResume(0,0); const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/true);", 1))),
        ("X11 -> S11", lambda: p_flush_precedes_the_snapshot_write(
            swmm_h.replace("\t\toutput_flush();\n", "", 1))),
        ("X12 -> S12", lambda: p_caller_aborts_on_unreadable(
            triton_h.replace("MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);", "exit(EXIT_FAILURE);", 1))),
        ("X13 -> S13", lambda: p_resume_event_has_four_fields(
            swmm_h.replace(' << " path=snapshot]"', ' << "]"', 1))),
        ("X14 -> S14", lambda: p_reason_vocabulary_is_the_contract(
            swmm_h.replace('return "retention-collision";', 'return "made-up";', 1))),
    ]
    # S3 HAS NO IN-MEMORY PROBE, and that is recorded rather than papered over
    # with a fake one. Its input is the FILESYSTEM (a tree grep), not a source
    # string, so a probe would have to write a file into the repo to fire. The
    # escape shape it guards -- a symbol leaking out of output.c -- is probed
    # in S5's `outside` clause via X5, which exercises the same _grep_tree
    # path. A reader should treat S3 as covered for VACUITY by X5 and not as
    # independently probed.
    # A PROBE IS TWO-STATE, and the second state is the one this file paid to
    # learn. Asserting only "red on mutated source" observes ONE bit, and two
    # different worlds produce it: the check caught the mutation, or the check
    # was ALREADY red for an unrelated reason and the mutation was irrelevant.
    # Measured on this file's own first run: 8 of 14 checks were red against
    # CORRECT source because a shared helper returned "" on a miss, and 7 of
    # the 13 probes reported CAUGHT on that same run. Every one was vacuous.
    #
    # The negative control is therefore asserted in the SAME run: the paired
    # S check must be GREEN on unmutated source before a red on mutated source
    # means anything at all.
    for name, fn in probes:
        sid = name.split("->")[1].strip()
        baseline_green = results.get(sid) is True
        still_green = fn()
        check("%s the defect form is CAUGHT" % name,
              baseline_green and not still_green,
              ("the paired %s is RED on UNMUTATED source, so this probe proves "
               "nothing -- fix the instrument before reading its verdict" % sid)
              if not baseline_green else
              "the mutated source still passes -- the paired check is VACUOUS")

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: WP-1C's structural properties hold, and every check discriminates.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
