#!/usr/bin/env python3
"""Comparator and checks of the diagnostics neutrality suite (test/diagnostics_neutrality_suite.sh).

Every subcommand prints what it measured and exits 0 on PASS, 1 on a finding, 2 on a usage or
input error, so the suite reads the verdict from the exit status itself, unpiped.

  compare          byte-compare two run output directories: the H, QX, QY and MH rasters of bin/, the
                   exchange side-file, the SWMM .out and the .rpt flow-routing continuity block; with
                   --captures also the named capture files. Exit 1 on any byte that differs.
  flip-byte        copy a run's output directory and flip one byte of one raster (case N8).
  captures-absent  no capture file exists in the output directory (every capture switched off).
  captures-check   every enabled capture file exists with 16 + records x record size bytes (or, for
                   the csv files, a header row and whole rows) and no other capture file exists.
  perf-nonzero     named performance.txt columns of one rank are above zero (SWMM_SOLVE, SWMM_LOG).
  capture-coverage which of the required coverage items a capture-on run exercised (extbc_m3,
                   swmm_exchange_m3, east and north edge flux, a non-converged T1a block, S(0)).
  perf-diff        per-rank differential of performance.txt: a capture-on run against the band of two
                   capture-off runs of the same configuration (Invariant B).
  spin-us          the busy-wait length (microseconds per step) whose total is a multiple of a band.
  first-balance    the first domain_balance.csv row's delta stored volume against S(t_end) - S(0)
                   computed from the t=0 and first-print H rasters (case N3).
  count-rows       the number of data rows of a capture csv file.
  end-drain        the T3 end-of-run arm: the exchange side-file's final cumulative dt equals the run
                   end time and SWMM stepped after the last print instant (case N12).
  steps-per-interval  TRITON steps (side-file records) per print interval of a run (case N12).
  coverage         the fixed case enters every capture branch (run-close condition R4).
  source-sites     N11: no capture call outside the T1 rows lies inside an #ifdef TRITON_SWMM region
                   or the global_num_of_swmm_links guard.
  make-n11-fixture write the N11 control: a copy of src/triton.h with the capture flush moved inside
                   the #ifdef TRITON_SWMM block (the iteration-6 position).
"""
import argparse
import glob
import math
import os
import re
import shutil
import struct
import sys

CAPTURE_FILES = [
    "swmm_step_diag.bin", "swmm_node_path3.bin", "swmm_node_path3_step.bin", "swmm_link_flow_diag.bin",
    "edge_flux.bin", "dt_limiter.bin", "wet_cells.csv", "domain_balance.csv",
    "swmm_node_diag.nodes", "swmm_link_flow_diag.links",
]
# toggle group -> the files it writes (erratum E-11's recorded mapping, E-9 for the surface files)
GROUPS = {
    "t1": ["swmm_step_diag.bin", "swmm_node_path3.bin", "swmm_node_diag.nodes"],
    "t1c": ["swmm_node_path3_step.bin", "swmm_node_diag.nodes"],
    "t1d": ["swmm_link_flow_diag.bin", "swmm_link_flow_diag.links"],
    "t2": ["edge_flux.bin", "domain_balance.csv"],
    "t5": ["wet_cells.csv"],
    "t6": ["dt_limiter.bin"],
}
MAGIC = {
    "swmm_step_diag.bin": 0x44535031, "swmm_node_path3.bin": 0x44504E33, "swmm_node_path3_step.bin": 0x44505333,
    "swmm_link_flow_diag.bin": 0x444C4631, "edge_flux.bin": 0x44454631, "dt_limiter.bin": 0x44444C31,
}
WET_HEADER = "%print_id, simtime, rank, wet_mean, wet_at_print"
BAL_HEADER = ("%t_end, d_storage_m3, runoff_m3, inflow_m3, extbc_m3, clamp_m3, swmm_exchange_m3, edge_west_m3, "
              "edge_east_m3, edge_south_m3, edge_north_m3, residual_m3")
PERF_CATEGORIES = ["Compute", "MPI", "IO", "Resize", "SWMM", "SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER",
                   "SWMM_SOLVE", "SWMM_LOG", "Other"]
SIDE_FILE = os.path.join("swmm", "hydraulics_exchange_replay.bin")
SWMM_OUT = os.path.join("swmm", "hydraulics.out")
SWMM_RPT = os.path.join("swmm", "hydraulics.rpt")


def rd(path):
    with open(path, "rb") as f:
        return f.read()


def say(msg):
    print(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------- compare

def rpt_section(path, header):
    """Lines of the .rpt section whose header line starts with `header`, up to the next banner."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(header):
            out, j = [], i + 2
            while j < len(lines) and not lines[j].strip().startswith("*****"):
                out.append(lines[j])
                j += 1
            return out
    return None


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def cmd_compare(a):
    """Hydraulic outputs byte for byte; exit 1 on any difference or on a vacuous comparison."""
    fails, compared = [], []

    def one(label, pa, pb):
        if not os.path.exists(pa) or not os.path.exists(pb):
            fails.append("%s: missing (%s %s, %s %s)" % (label, pa, os.path.exists(pa), pb, os.path.exists(pb)))
            return
        ba, bb = rd(pa), rd(pb)
        compared.append((label, len(ba)))
        if ba != bb:
            fails.append("%s: differs (first differing byte %d; sizes %d and %d)" % (label, first_diff(ba, bb), len(ba), len(bb)))

    da, db = os.path.join(a.a, a.raster_dir), os.path.join(a.b, a.raster_dir)
    names_a = sorted(n for n in os.listdir(da)) if os.path.isdir(da) else []
    names_b = sorted(n for n in os.listdir(db)) if os.path.isdir(db) else []
    wanted = re.compile(r"^(H|QX|QY|MH)_")
    ra = [n for n in names_a if wanted.match(n)]
    rb = [n for n in names_b if wanted.match(n)]
    if not ra:
        fails.append("no H/QX/QY/MH raster in %s (a vacuous comparison)" % da)
    if ra != rb:
        fails.append("raster file sets differ: only in a %s, only in b %s" % (sorted(set(ra) - set(rb))[:5], sorted(set(rb) - set(ra))[:5]))
    for n in ra:
        if n in rb:
            one(a.raster_dir + "/" + n, os.path.join(da, n), os.path.join(db, n))
    if not a.no_swmm:
        one(SIDE_FILE, os.path.join(a.a, SIDE_FILE), os.path.join(a.b, SIDE_FILE))
        one(SWMM_OUT, os.path.join(a.a, SWMM_OUT), os.path.join(a.b, SWMM_OUT))
        sa = rpt_section(os.path.join(a.a, SWMM_RPT), "Flow Routing Continuity") if os.path.exists(os.path.join(a.a, SWMM_RPT)) else None
        sb = rpt_section(os.path.join(a.b, SWMM_RPT), "Flow Routing Continuity") if os.path.exists(os.path.join(a.b, SWMM_RPT)) else None
        if sa is None or sb is None:
            fails.append(".rpt Flow Routing Continuity block missing (a %s, b %s)" % (sa is not None, sb is not None))
        else:
            compared.append((".rpt Flow Routing Continuity (%d lines)" % len(sa), len("\n".join(sa))))
            if sa != sb:
                fails.append(".rpt Flow Routing Continuity block differs")
    for n in (a.captures.split(",") if a.captures else []):
        n = n.strip()
        if not n:
            continue
        one("capture " + n, os.path.join(a.a, n), os.path.join(a.b, n))
        if os.path.exists(os.path.join(a.a, n)) and os.path.getsize(os.path.join(a.a, n)) == 0:
            fails.append("capture %s is empty (a vacuous comparison)" % n)
    say("compare a=%s b=%s: %d item(s) compared, %d byte(s) read" % (a.a, a.b, len(compared), sum(c[1] for c in compared)))
    for label, size in compared:
        say("  compared %s (%d bytes)" % (label, size))
    for f in fails:
        say("  DIFFERENCE: " + f)
    say("RESULT %s compare" % ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def cmd_flip_byte(a):
    if os.path.exists(a.dst):
        shutil.rmtree(a.dst)
    shutil.copytree(a.src, a.dst, symlinks=True)
    cands = sorted(glob.glob(os.path.join(a.dst, "bin", "H_*")))
    if not cands:
        say("no H raster to flip in %s" % a.dst)
        return 2
    p = cands[-1]
    b = bytearray(rd(p))
    k = len(b) // 2
    old = b[k]
    b[k] ^= 0x01
    with open(p, "wb") as f:
        f.write(bytes(b))
    say("flipped bit 0 of byte %d of %s (0x%02x -> 0x%02x)" % (k, p, old, b[k]))
    return 0


# ---------------------------------------------------------------- captures

def read_header(path):
    b = rd(path)
    if len(b) < 16:
        return None, b
    return struct.unpack_from("<4i", b, 0), b


def walk_t1a(b):
    """Records of swmm_step_diag.bin; returns (n_records, ends_at_eof, stamps)."""
    off, n, stamps = 16, 0, []
    while off + 48 <= len(b):
        t, dt, trials, nnc = struct.unpack_from("<ddii", b, off)
        if nnc < 0 or off + 48 + 4 * nnc > len(b):
            return n, False, stamps
        stamps.append(t)
        off += 48 + 4 * nnc
        n += 1
    return n, off == len(b), stamps


def read_lines(path):
    with open(path) as f:
        return [l for l in f.read().split("\n") if l != ""]


def csv_rows(path):
    return [l for l in read_lines(path) if not l.startswith("%")]


def cmd_captures_absent(a):
    present = [n for n in CAPTURE_FILES if os.path.exists(os.path.join(a.out, n))]
    for n in present:
        say("  capture file present with every capture off: %s" % n)
    say("checked %d capture file name(s) in %s: %d present" % (len(CAPTURE_FILES), a.out, len(present)))
    say("RESULT %s captures-absent" % ("FAIL" if present else "PASS"))
    return 1 if present else 0


def cmd_captures_check(a):
    enabled = [g.strip() for g in a.enabled.split(",") if g.strip()]
    if enabled == ["all"]:
        enabled = ["t1", "t1c", "t1d", "t2", "t5", "t6"]
    expect = set()
    for g in enabled:
        if g not in GROUPS:
            say("unknown capture group %r" % g)
            return 2
        expect.update(GROUPS[g])
    fails, counts = [], {}
    for n in CAPTURE_FILES:
        p = os.path.join(a.out, n)
        if n not in expect:
            if os.path.exists(p):
                fails.append("%s exists but its toggle is off" % n)
            continue
        if not os.path.exists(p):
            fails.append("%s is missing" % n)
    nodes = read_lines(os.path.join(a.out, "swmm_node_diag.nodes")) if os.path.exists(os.path.join(a.out, "swmm_node_diag.nodes")) else None
    links = read_lines(os.path.join(a.out, "swmm_link_flow_diag.links")) if os.path.exists(os.path.join(a.out, "swmm_link_flow_diag.links")) else None
    for n in sorted(expect):
        p = os.path.join(a.out, n)
        if not os.path.exists(p):
            continue
        size = os.path.getsize(p)
        if n in MAGIC:
            h, b = read_header(p)
            if h is None:
                fails.append("%s shorter than its 16-byte header (%d bytes)" % (n, size))
                continue
            if h[0] != MAGIC[n] or h[1] != 1 or h[2] != 8:
                fails.append("%s header {magic %#x, version %d, width %d} is not {%#x, 1, 8}" % (n, h[0], h[1], h[2], MAGIC[n]))
            if n == "swmm_step_diag.bin":
                recs, ok, _ = walk_t1a(b)
                counts[n] = recs
                if not ok:
                    fails.append("%s: the record walk does not end at the end of the file (%d bytes)" % (n, size))
                if h[3] != 0:
                    fails.append("%s n_items %d is not 0" % (n, h[3]))
            else:
                rs = {"swmm_node_path3.bin": lambda: 8 + 20 * len(nodes), "swmm_node_path3_step.bin": lambda: 16 + 8 * len(nodes),
                      "swmm_link_flow_diag.bin": lambda: 8 + 8 * len(links), "edge_flux.bin": lambda: 40, "dt_limiter.bin": lambda: 48}[n]
                if n.startswith("swmm_node") and nodes is None or n == "swmm_link_flow_diag.bin" and links is None:
                    fails.append("%s: its sidecar is missing" % n)
                    continue
                size_rec = rs()
                recs = (size - 16) // size_rec
                counts[n] = recs
                if (size - 16) % size_rec != 0 or size != 16 + recs * size_rec:
                    fails.append("%s: %d bytes is not 16 + records x %d" % (n, size, size_rec))
                want_items = {"swmm_node_path3.bin": len(nodes or []), "swmm_node_path3_step.bin": len(nodes or []),
                              "swmm_link_flow_diag.bin": len(links or []), "edge_flux.bin": 4, "dt_limiter.bin": 0}[n]
                if h[3] != want_items:
                    fails.append("%s n_items %d is not %d" % (n, h[3], want_items))
        elif n == "wet_cells.csv":
            lines = read_lines(p)
            counts[n] = len(csv_rows(p))
            if not lines or lines[0] != WET_HEADER:
                fails.append("wet_cells.csv header is %r" % (lines[0] if lines else None))
            if a.ranks and counts[n] % a.ranks != 0:
                fails.append("wet_cells.csv has %d rows, not a whole number of %d-rank intervals" % (counts[n], a.ranks))
            if counts[n] == 0:
                fails.append("wet_cells.csv has no data row")
        elif n == "domain_balance.csv":
            lines = read_lines(p)
            counts[n] = len(csv_rows(p))
            if not lines or lines[0] != BAL_HEADER:
                fails.append("domain_balance.csv header is %r" % (lines[0] if lines else None))
            if counts[n] == 0:
                fails.append("domain_balance.csv has no data row")
            if any(len(r.split(",")) != 12 for r in csv_rows(p)):
                fails.append("domain_balance.csv has a row of other than 12 fields")
    if "swmm_step_diag.bin" in counts and "swmm_node_path3_step.bin" in counts and counts["swmm_step_diag.bin"] != counts["swmm_node_path3_step.bin"]:
        fails.append("T1a has %d records, T1c %d" % (counts["swmm_step_diag.bin"], counts["swmm_node_path3_step.bin"]))
    if "swmm_step_diag.bin" in counts and "swmm_link_flow_diag.bin" in counts and counts["swmm_step_diag.bin"] != counts["swmm_link_flow_diag.bin"]:
        fails.append("T1a has %d records, T1d %d" % (counts["swmm_step_diag.bin"], counts["swmm_link_flow_diag.bin"]))
    sf = os.path.join(a.out, SIDE_FILE)
    if "swmm_step_diag.bin" in counts and os.path.exists(sf) and not a.no_side_file_count:
        nrec = side_file_records(sf)[0]
        say("  T1a records %d against exchange side-file records %d" % (counts["swmm_step_diag.bin"], nrec))
        if nrec != counts["swmm_step_diag.bin"]:
            fails.append("T1a has %d records, the exchange side-file %d (a writer that ignores stepped_this_step)" % (counts["swmm_step_diag.bin"], nrec))
    if "edge_flux.bin" in counts and "dt_limiter.bin" in counts:
        if a.fixed_dt:
            if counts["dt_limiter.bin"] != 0:
                fails.append("fixed time increment: dt_limiter.bin holds %d records, not 0" % counts["dt_limiter.bin"])
            if os.path.exists(os.path.join(a.out, "dt_limiter.bin")) and os.path.getsize(os.path.join(a.out, "dt_limiter.bin")) != 16:
                fails.append("fixed time increment: dt_limiter.bin is not the 16-byte header alone")
        elif counts["edge_flux.bin"] != counts["dt_limiter.bin"]:
            fails.append("edge_flux.bin has %d records, dt_limiter.bin %d" % (counts["edge_flux.bin"], counts["dt_limiter.bin"]))
    if "edge_flux.bin" in counts and counts["edge_flux.bin"] == 0:
        fails.append("edge_flux.bin holds no record")
    if "dt_limiter.bin" in counts and not a.fixed_dt and counts["dt_limiter.bin"] == 0:
        fails.append("dt_limiter.bin holds no record")
    if a.end_time is not None and "edge_flux.bin" in counts and counts["edge_flux.bin"] > 0:
        b = rd(os.path.join(a.out, "edge_flux.bin"))
        last = struct.unpack_from("<d", b, 16 + (counts["edge_flux.bin"] - 1) * 40)[0]
        say("  edge_flux.bin last stamp %r against run end %r" % (last, a.end_time))
        if abs(last - a.end_time) > 1e-6:
            fails.append("edge_flux.bin last stamp %r is not the run end %r" % (last, a.end_time))
    for n in sorted(expect):
        p = os.path.join(a.out, n)
        say("  %s: %s bytes, %s records" % (n, os.path.getsize(p) if os.path.exists(p) else "missing", counts.get(n, "-")))
    for f in fails:
        say("  FINDING: " + f)
    say("RESULT %s captures-check enabled=%s" % ("FAIL" if fails else "PASS", ",".join(enabled)))
    return 1 if fails else 0


# ---------------------------------------------------------------- performance

def read_perf(out):
    p = os.path.join(out, "performance.txt")
    lines = [l for l in rd(p).decode().splitlines() if l.strip()]
    hdr = [h for h in re.split(r"[,\s]+", lines[0].lstrip("%")) if h]
    rows = {}
    for ln in lines[1:]:
        row = [x for x in re.split(r"[,\s]+", ln) if x]
        if len(row) == len(hdr):
            rows[int(float(row[0]))] = dict(zip(hdr, [float(x) for x in row]))
    return rows


def parse_ranks(spec, available):
    if spec in (None, "", "all"):
        return sorted(available)
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [r for r in out if r in available]


def band_for(off_rows, rank, col, floor, rel=0.0):
    """The band of a category on a rank: the larger of the capture-off runs' spread, an absolute floor
    (the printed precision and clock noise of a short run) and a fraction of the capture-off mean."""
    vals = [o[rank][col] for o in off_rows]
    spread = max(vals) - min(vals)
    mean = sum(vals) / len(vals)
    return max(spread, floor, rel * abs(mean)), mean, spread


def cmd_perf_diff(a):
    on = read_perf(a.on)
    offs = [read_perf(o) for o in a.off]
    if len(offs) < 2:
        say("--off needs two capture-off runs (their spread is the band)")
        return 2
    cols = [c for c in (a.cols.split(",") if a.cols else PERF_CATEGORIES)]
    ranks = parse_ranks(a.ranks, set(on) & set.intersection(*[set(o) for o in offs]))
    if not ranks:
        say("no rank in common between the runs")
        return 2
    fails = []
    for r in ranks:
        for c in cols:
            if c not in on[r]:
                fails.append("rank %d has no column %s" % (r, c))
                continue
            band, mean, spread = band_for(offs, r, c, a.abs_floor, a.rel_floor)
            delta = on[r][c] - mean
            tag = "OUT" if abs(delta) > band else "in"
            say("  rank %d %-10s on %-9g off mean %-9g spread %-9g band %-9g delta %+9g %s" % (r, c, on[r][c], mean, spread, band, delta, tag))
            if abs(delta) > band:
                fails.append("rank %d %s leaves the capture-off band (delta %+g, band %g)" % (r, c, delta, band))
    if a.diag_positive:
        dr = parse_ranks(a.diag_positive, set(on))
        for r in dr:
            say("  rank %d DIAG %g (must be > 0)" % (r, on[r].get("DIAG", 0.0)))
            if not on[r].get("DIAG", 0.0) > 0.0:
                fails.append("rank %d DIAG is %g, not above 0 (the capture bracket never ran)" % (r, on[r].get("DIAG", 0.0)))
    for f in fails:
        say("  FINDING: " + f)
    say("RESULT %s perf-diff on=%s cols=%s ranks=%s" % ("FAIL" if fails else "PASS", a.on, ",".join(cols), ",".join(map(str, ranks))))
    return 1 if fails else 0


def side_file_records(path):
    b = rd(path)
    magic, nn, w = struct.unpack_from("<3i", b, 0)
    rs = w * (1 + nn)
    return (len(b) - 12) // rs, nn, w


def side_file_dts(path):
    b = rd(path)
    magic, nn, w = struct.unpack_from("<3i", b, 0)
    if w != 8:
        raise SystemExit("side-file value width %d is not 8" % w)
    rs = w * (1 + nn)
    return [struct.unpack_from("<d", b, 12 + k * rs)[0] for k in range((len(b) - 12) // rs)]


def cmd_spin_us(a):
    offs = [read_perf(o) for o in a.offs]
    ranks = set.intersection(*[set(o) for o in offs])
    band = 0.0
    for r in ranks:
        band = max(band, band_for(offs, r, a.col, a.abs_floor, a.rel_floor)[0])
    steps = side_file_records(os.path.join(a.steps_from, SIDE_FILE))[0]
    us = int(math.ceil(a.factor * band * 1e6 / steps))
    say("band of %s over ranks %s = %g s; TRITON steps (side-file records) = %d; spin = %g x band / steps" % (a.col, sorted(ranks), band, steps, a.factor))
    say("SPIN_US %d" % us)
    return 0


def cmd_perf_nonzero(a):
    rows = read_perf(a.out)
    fails = []
    for c in a.cols.split(","):
        v = rows[a.rank].get(c)
        say("  rank %d %s = %r (must be > 0)" % (a.rank, c, v))
        if not (v is not None and v > 0.0):
            fails.append("rank %d %s is %r" % (a.rank, c, v))
    for f in fails:
        say("  FINDING: " + f)
    say("RESULT %s perf-nonzero" % ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


def cmd_capture_coverage(a):
    """Which of the required coverage items a capture-on run exercised (run-close condition R4)."""
    items = []
    rows = [r.split(",") for r in csv_rows(os.path.join(a.out, "domain_balance.csv"))]
    col = {n: i for i, n in enumerate(BAL_HEADER.lstrip("%").replace(" ", "").split(","))}

    def nz(name):
        return sum(1 for r in rows if float(r[col[name]]) != 0.0)

    items.append(("extbc_m3 nonzero in domain_balance.csv rows", nz("extbc_m3"), len(rows)))
    items.append(("swmm_exchange_m3 nonzero rows", nz("swmm_exchange_m3"), len(rows)))
    items.append(("edge_east_m3 nonzero rows", nz("edge_east_m3"), len(rows)))
    items.append(("edge_north_m3 nonzero rows", nz("edge_north_m3"), len(rows)))
    items.append(("initial stored volume: first row d_storage_m3 (nonzero)", 1 if float(rows[0][col["d_storage_m3"]]) != 0.0 else 0, 1))
    b = rd(os.path.join(a.out, "edge_flux.bin"))
    n = (len(b) - 16) // 40
    ee = sum(1 for k in range(n) if struct.unpack_from("<d", b, 16 + k * 40 + 16)[0] != 0.0)
    en = sum(1 for k in range(n) if struct.unpack_from("<d", b, 16 + k * 40 + 32)[0] != 0.0)
    items.append(("edge_flux.bin steps with a nonzero east flux", ee, n))
    items.append(("edge_flux.bin steps with a nonzero north flux", en, n))
    b = rd(os.path.join(a.out, "swmm_step_diag.bin"))
    off, steps, nc, listed = 16, 0, 0, 0
    while off + 48 <= len(b):
        t0, dt, trials, nnc = struct.unpack_from("<ddii", b, off)
        steps += 1
        nc += 1 if nnc > 0 else 0
        listed += nnc
        off += 48 + 4 * nnc
    items.append(("swmm_step_diag.bin steps with a non-converged node (T1a variable block, %d ordinals listed)" % listed, nc, steps))
    fails = []
    for label, count, of in items:
        say("  %s: %d of %d" % (label, count, of))
        if count == 0:
            fails.append(label)
    for f in fails:
        say("  NOT EXERCISED: " + f)
    say("RESULT %s capture-coverage" % ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


# ---------------------------------------------------------------- balance and rasters

def read_dem_values(path, header_rows):
    with open(path) as f:
        lines = f.read().splitlines()
    vals = []
    for ln in lines[header_rows:]:
        if ln.strip():
            vals.append([float(x) for x in ln.split()])
    return vals


def read_raster(path, nrows, ncols):
    """A BIN or ASC raster written by TRITON, as a list of rows."""
    b = rd(path)
    n = nrows * ncols
    # a BIN raster is the grid of doubles after a short header (16 bytes on the fixed case)
    if len(b) < n * 8 or len(b) - n * 8 not in (0, 16):
        raise SystemExit("raster %s has %d bytes, which is not %d doubles plus a 0- or 16-byte header" % (path, len(b), n))
    vals = struct.unpack("<%dd" % n, b[len(b) - n * 8:])
    return [list(vals[i * ncols:(i + 1) * ncols]) for i in range(nrows)]


def raster_files(out, prefix):
    return sorted(glob.glob(os.path.join(out, "bin", prefix + "_*")))


def cmd_first_balance(a):
    init = read_dem_values(a.init_h, 0)
    nrows, ncols = len(init), len(init[0])
    area = a.cell * a.cell
    s0 = sum(sum(r) for r in init) * area
    hs = raster_files(a.out, "H")
    if not hs:
        say("no H raster in %s" % os.path.join(a.out, "bin"))
        return 2
    s1 = sum(sum(r) for r in read_raster(hs[0], nrows, ncols)) * area
    rows = csv_rows(os.path.join(a.out, "domain_balance.csv"))
    if not rows:
        say("domain_balance.csv has no data row")
        return 2
    f = rows[0].split(",")
    t_end, d_storage = float(f[0]), float(f[1])
    derived = s1 - s0
    say("first raster %s; S(0) = %.9g m3, S(t_end) = %.9g m3, derived delta %.9g m3; row 1: t_end %g, d_storage_m3 %.9g; S(0) must be nonzero: %s"
        % (os.path.basename(hs[0]), s0, s1, derived, t_end, d_storage, s0 != 0.0))
    tol = max(1e-9 * max(abs(s0), abs(s1), 1.0), 1e-9)
    ok = s0 != 0.0 and abs(d_storage - derived) <= tol
    say("  |row 1 - derived| = %.3g, tolerance %.3g" % (abs(d_storage - derived), tol))
    say("RESULT %s first-balance" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_count_rows(a):
    n = len(csv_rows(a.file))
    say("%s: %d data row(s)" % (a.file, n))
    say("ROWS %d" % n)
    return 0


def cmd_end_drain(a):
    dts = side_file_dts(os.path.join(a.out, SIDE_FILE))
    cum, ends = 0.0, []
    for d in dts:
        cum += d
        ends.append(cum)
    last_print = math.floor(a.end_time / a.print_interval + 1e-9) * a.print_interval
    after = [e for e in ends if e > last_print + 1e-6]
    need_after = a.end_time > last_print + 1e-6   # a run that ends on a print instant has nothing after it
    say("side-file records %d; final cumulative dt %.9f; run end %.9f; last print instant %.3f; records ending after it: %d (required: %s)"
        % (len(dts), cum, a.end_time, last_print, len(after), "at least 1" if need_after else "none, the run ends on a print instant"))
    ok = abs(cum - a.end_time) <= 1e-6 and (len(after) >= 1 or not need_after)
    say("RESULT %s end-drain" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_steps_per_interval(a):
    dts = side_file_dts(os.path.join(a.out, SIDE_FILE))
    cum, counts, k = 0.0, {}, 0
    for d in dts:
        cum += d
        k = int(math.ceil(cum / a.print_interval - 1e-9))
        counts[k] = counts.get(k, 0) + 1
    seq = [counts.get(i, 0) for i in range(1, max(counts) + 1)] if counts else []
    say("TRITON steps per print interval (side-file records, interval 1..n): %s" % seq)
    if a.pick_candidates:
        cands = [int(x) for x in a.pick_candidates.split(",")]
        idx = a.interval_index
        n = seq[idx - 1] if 0 < idx <= len(seq) else 0
        pick = next((c for c in cands if n % c != 0), None)
        say("interval %d holds %d steps; swmm_coupling_interval candidates %s" % (idx, n, cands))
        say("PICK %s" % pick)
        return 0 if pick is not None else 1
    return 0


# ---------------------------------------------------------------- coverage

def cmd_coverage(a):
    fails = []
    rpt = os.path.join(a.out, SWMM_RPT)
    txt = open(rpt, errors="replace").read()
    m = re.search(r"% of Steps Not Converging\s*:\s*([0-9.]+)", txt)
    nonconv = float(m.group(1)) if m else None
    say("  non-converged: .rpt '%% of Steps Not Converging' = %s (must be > 0)" % nonconv)
    if not nonconv:
        fails.append("no non-converged SWMM step (%% of Steps Not Converging %s)" % nonconv)
    sec = rpt_section(rpt, "Node Flooding Summary")
    flooded = [l for l in (sec or []) if re.match(r"^\s*[A-Za-z0-9_]+\s+[0-9.]+\s+[0-9.]+", l)]
    say("  flooded nodes: %d row(s) in the Node Flooding Summary %s" % (len(flooded), [l.split()[0] for l in flooded]))
    if not flooded:
        fails.append("no flooded node")
    nrows = ncols = None
    dem = read_dem_values(a.dem, 6)
    nrows, ncols = len(dem), len(dem[0])
    hs = raster_files(a.out, "H")
    qxs = raster_files(a.out, "QX")
    qys = raster_files(a.out, "QY")
    if not hs or not qxs or not qys:
        fails.append("no H/QX/QY raster to read")
    else:
        r0, r1 = a.extbc_rows
        extbc_cells = [(i, 0) for i in range(r0, r1 + 1)]
        hmax = 0.0
        for p in hs:
            g = read_raster(p, nrows, ncols)
            hmax = max(hmax, max(g[i][j] for i, j in extbc_cells))
        say("  extbc type-1 cells: west column rows %d-%d, largest depth over %d H rasters = %g (must be > 0)" % (r0, r1, len(hs), hmax))
        if not hmax > 0.0:
            fails.append("no extbc cell is wet")
        qe = qw = qn = 0.0
        for p in qxs:
            g = read_raster(p, nrows, ncols)
            qe = max(qe, max(abs(g[i][ncols - 1]) for i in range(nrows)))
            qw = max(qw, max(abs(g[i][0]) for i in range(nrows)))
        for p in qys:
            g = read_raster(p, nrows, ncols)
            qn = max(qn, max(abs(v) for v in g[0]))
        say("  edge flow: max |qx| on the east column %g, on the west column %g, max |qy| on the north row %g (east and north must be > 0)" % (qe, qw, qn))
        if not (qe > 0.0 and qn > 0.0):
            fails.append("no nonzero east-edge qx or north-edge qy")
    init = read_dem_values(a.init_h, 0)
    s0 = sum(sum(r) for r in init) * a.cell * a.cell
    say("  initial stored volume S(0) = %.9g m3 (must be > 0)" % s0)
    if not s0 > 0.0:
        fails.append("the case does not start wet")
    for f in fails:
        say("  FINDING: " + f)
    say("RESULT %s coverage" % ("FAIL" if fails else "PASS"))
    return 1 if fails else 0


# ---------------------------------------------------------------- source scan (N11)

CAPTURE_CALL = re.compile(r"\bdiag_surface\.\w+\s*\(|\bdiag_log::\w+\s*\(|\bdiag_print_row\s*\(|\bdiagnostics::\w+\s*\(|\bst\.(start|stop)\(DIAG_TIME\)")
# erratum E-9: the one swmm_exchange_m3 read of exchange_q after the coupling block closes SWMM_TIME
EXEMPT_CALLS = re.compile(r"\badd_exchange\s*\(")
# the T1 rows: statements guarded by the T1 capture predicates
T1_GUARD = re.compile(r"step_diag_due\s*\(|node_path3_enabled\s*\(")
SWMM_GUARD = re.compile(r"global_num_of_swmm_links")


def strip_comment(line):
    return re.sub(r"//.*$", "", line)


def block_lines(lines, start):
    """Line indices of the statement that starts at `start` (an if with a brace block, or one statement)."""
    depth, opened, i = 0, False, start
    while i < len(lines):
        s = strip_comment(lines[i])
        for ch in s:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return range(start, i + 1)
        if not opened and s.rstrip().endswith(";"):
            return range(start, i + 1)
        i += 1
    return range(start, len(lines))


def scan_sites(path):
    """Capture calls inside an #ifdef TRITON_SWMM region or the global_num_of_swmm_links guard,
    outside the T1 rows and the E-9 exemption. Returns (violations, exempt_count, calls_total)."""
    lines = rd(path).decode().replace("\r", "").split("\n")
    in_swmm_ifdef = [False] * len(lines)
    stack = []  # each entry: [condition_is_swmm, in_else]
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith("#"):
            d = re.match(r"#\s*(ifdef|ifndef|if|else|elif|endif)\b\s*(.*)", s)
            if d:
                kind, rest = d.group(1), d.group(2)
                if kind in ("ifdef", "ifndef", "if"):
                    is_swmm = kind == "ifdef" and rest.strip() == "TRITON_SWMM"
                    neg_swmm = kind == "ifndef" and rest.strip() == "TRITON_SWMM"
                    stack.append([is_swmm, neg_swmm, False])
                elif kind == "else" and stack:
                    stack[-1][2] = True
                elif kind == "endif" and stack:
                    stack.pop()
        inside = False
        for is_swmm, neg_swmm, in_else in stack:
            if (is_swmm and not in_else) or (neg_swmm and in_else):
                inside = True
        in_swmm_ifdef[i] = inside
    guard = [False] * len(lines)
    t1 = [False] * len(lines)
    for i, raw in enumerate(lines):
        s = strip_comment(raw)
        if re.match(r"\s*(else\s+)?if\s*\(", s):
            if T1_GUARD.search(s):
                for k in block_lines(lines, i):
                    t1[k] = True
            if SWMM_GUARD.search(s):
                for k in block_lines(lines, i):
                    guard[k] = True
    # E-9: the statement block holding the add_exchange read (its DIAG_TIME bracket included) is exempt
    ex_block = [False] * len(lines)
    for i, raw in enumerate(lines):
        if EXEMPT_CALLS.search(strip_comment(raw)):
            j = i
            while j >= 0 and not re.match(r"\s*(else\s+)?if\s*\(", strip_comment(lines[j])):
                j -= 1
            if j >= 0 and i in block_lines(lines, j):
                for k in block_lines(lines, j):
                    ex_block[k] = True
            else:
                ex_block[i] = True
    viol, exempt, total = [], 0, 0
    for i, raw in enumerate(lines):
        s = strip_comment(raw)
        if not CAPTURE_CALL.search(s) and not EXEMPT_CALLS.search(s):
            continue
        total += 1
        if ex_block[i]:
            exempt += 1
            continue
        if t1[i]:
            continue
        if in_swmm_ifdef[i] or guard[i]:
            viol.append((i + 1, "inside %s: %s" % ("#ifdef TRITON_SWMM" if in_swmm_ifdef[i] else "the global_num_of_swmm_links guard", raw.strip()[:110])))
    return viol, exempt, total


def cmd_source_sites(a):
    viol, exempt, total = scan_sites(a.file)
    say("%s: %d capture-call line(s) found, %d exempt (the E-9 add_exchange read and its DIAG_TIME bracket), %d outside every T1 row lie inside an #ifdef TRITON_SWMM region or the guard"
        % (a.file, total, exempt, len(viol)))
    for ln, why in viol:
        say("  VIOLATION line %d %s" % (ln, why))
    if total == 0:
        say("  the scan matched no capture call at all: the matcher is broken")
        say("RESULT FAIL source-sites")
        return 2
    say("RESULT %s source-sites" % ("FAIL" if viol else "PASS"))
    return 1 if viol else 0


def cmd_make_n11_fixture(a):
    """Move the capture flush block inside the #ifdef TRITON_SWMM block of the exchange-log flush."""
    lines = rd(a.src).decode().split("\n")
    flush_at = next((i for i, l in enumerate(lines) if "diag_log::any_open()" in l and re.match(r"\s*if\s*\(", l)), None)
    if flush_at is None:
        say("the capture flush block (if (diag_log::any_open())) was not found in %s" % a.src)
        return 2
    blk = list(block_lines([l.replace("\r", "") for l in lines], flush_at))
    block = lines[blk[0]:blk[-1] + 1]
    anchor = next((i for i, l in enumerate(lines) if "flush_exchange_log" in l), None)
    if anchor is None:
        say("flush_exchange_log not found in %s" % a.src)
        return 2
    endif_at = next((i for i in range(anchor, len(lines)) if re.match(r"\s*#\s*endif", lines[i])), None)
    if endif_at is None or endif_at > flush_at:
        say("no #endif between the exchange-log flush and the capture flush in %s" % a.src)
        return 2
    out = lines[:endif_at] + block + lines[endif_at:blk[0]] + lines[blk[-1] + 1:]
    with open(a.dst, "w", newline="") as f:
        f.write("\n".join(out))
    say("moved the capture flush block (source lines %d-%d) before the #endif at source line %d into %s" % (blk[0] + 1, blk[-1] + 1, endif_at + 1, a.dst))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    sp.required = True   # the keyword form needs Python 3.7; Rivanna's default python3 is 3.6
    p = sp.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--captures", default="")
    p.add_argument("--no-swmm", action="store_true", help="skip the side-file, .out and .rpt (uncoupled builds)")
    p.add_argument("--raster-dir", default="bin", help="directory of the rasters under the output folder (bin; asc for ASC output)")
    p.set_defaults(f=cmd_compare)
    p = sp.add_parser("flip-byte")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.set_defaults(f=cmd_flip_byte)
    p = sp.add_parser("captures-absent")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_captures_absent)
    p = sp.add_parser("captures-check")
    p.add_argument("--out", required=True)
    p.add_argument("--enabled", required=True, help="comma list of t1,t1c,t1d,t2,t5,t6 or all")
    p.add_argument("--ranks", type=int, default=0)
    p.add_argument("--fixed-dt", action="store_true")
    p.add_argument("--end-time", type=float, default=None)
    p.add_argument("--no-side-file-count", action="store_true")
    p.set_defaults(f=cmd_captures_check)
    p = sp.add_parser("perf-diff")
    p.add_argument("--on", required=True)
    p.add_argument("--off", nargs="+", required=True)
    p.add_argument("--cols", default="")
    p.add_argument("--ranks", default="all")
    p.add_argument("--abs-floor", type=float, default=0.05)
    p.add_argument("--rel-floor", type=float, default=0.03)
    p.add_argument("--diag-positive", default="")
    p.set_defaults(f=cmd_perf_diff)
    p = sp.add_parser("perf-nonzero")
    p.add_argument("--out", required=True)
    p.add_argument("--cols", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.set_defaults(f=cmd_perf_nonzero)
    p = sp.add_parser("capture-coverage")
    p.add_argument("--out", required=True)
    p.set_defaults(f=cmd_capture_coverage)
    p = sp.add_parser("spin-us")
    p.add_argument("--offs", nargs="+", required=True)
    p.add_argument("--col", default="Compute")
    p.add_argument("--factor", type=float, default=10.0)
    p.add_argument("--abs-floor", type=float, default=0.05)
    p.add_argument("--rel-floor", type=float, default=0.03)
    p.add_argument("--steps-from", required=True)
    p.set_defaults(f=cmd_spin_us)
    p = sp.add_parser("first-balance")
    p.add_argument("--out", required=True)
    p.add_argument("--init-h", required=True)
    p.add_argument("--cell", type=float, required=True)
    p.set_defaults(f=cmd_first_balance)
    p = sp.add_parser("count-rows")
    p.add_argument("--file", required=True)
    p.set_defaults(f=cmd_count_rows)
    p = sp.add_parser("end-drain")
    p.add_argument("--out", required=True)
    p.add_argument("--end-time", type=float, required=True)
    p.add_argument("--print-interval", type=float, required=True)
    p.set_defaults(f=cmd_end_drain)
    p = sp.add_parser("steps-per-interval")
    p.add_argument("--out", required=True)
    p.add_argument("--print-interval", type=float, required=True)
    p.add_argument("--pick-candidates", default="")
    p.add_argument("--interval-index", type=int, default=1)
    p.set_defaults(f=cmd_steps_per_interval)
    p = sp.add_parser("coverage")
    p.add_argument("--out", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--init-h", required=True)
    p.add_argument("--cell", type=float, required=True)
    p.add_argument("--extbc-rows", type=int, nargs=2, required=True)
    p.set_defaults(f=cmd_coverage)
    p = sp.add_parser("source-sites")
    p.add_argument("--file", required=True)
    p.set_defaults(f=cmd_source_sites)
    p = sp.add_parser("make-n11-fixture")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.set_defaults(f=cmd_make_n11_fixture)
    a = ap.parse_args()
    try:
        return a.f(a)
    except (OSError, struct.error, KeyError, ValueError) as e:
        say("ERROR %s: %s" % (type(e).__name__, e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
