#!/usr/bin/env python3
"""Readers and checks of the driven-stage outfall case N14 of test/diagnostics_neutrality_suite.sh.

Every subcommand prints what it read and ends with `RESULT PASS|FAIL name`; the exit status is 0 on
PASS, 1 on FAIL and 2 when an input is missing or malformed (the suite treats 2 as an ERROR, never a
pass). The suite runs it with `python -I`, so it imports nothing but the standard library.

  layout CASE NP BELOW ABOVE        the cell, DEM elevation, invert and owning rank of the two outfalls;
                                    PASS only when BELOW's cell lies below its invert, ABOVE's lies above
                                    its invert and neither is owned by rank 0 at NP ranks
  stats OUT                         counts of the records of outfall_stage.bin (discharge, backflow, wet
                                    and dry steps), for the suite's log and for sizing the case
  stage-check OUT CASE NAME PRINT HEXTRA TOL
                                    case N14 arm (f): at each print instant the stage of the record
                                    stamped with that instant is DEM[cell] + H_raster[cell] +
                                    q_back_applied * dt / dx^2 where the depth read was at least hextra,
                                    and the outfall's invert otherwise (serial run: H_NN_00.out)
  ledger OUT CASE NAME PRINT NODES [LINK]
                                    case N14 arm (e): the .rpt flow-routing continuity totals External
                                    Inflow and External Outflow against the full sums of the step
                                    records (the driven outfall's flow from outfall_stage.bin, the other
                                    outfall's from the T1d flow of LINK, the coupling nodes' from the
                                    exchange side-file); NODES are the coupling nodes in solver order
  edit-inp DIR OP ARGS...           a refusal arm's variant of the run directory's swmm/hydraulics.inp:
                                    coords NAME FROM (NAME takes FROM's coordinates), outfall-type NAME TYPE
                                    (the [OUTFALLS] row of NAME gets that type), add-conduit NAME FROM TO
                                    (a second 0.5 m circular conduit)
  fixture-no-backflow OUT CASE NAME PRINT HEXTRA DST
                                    a copy of OUT with q_back_applied 0 (the F-4 control's input)
  residual OUT                      domain_balance.csv: the residual column at round-off with the
                                    outfall_backflow_m3 term in the sum (the header has 14 columns)
"""
import math
import os
import re
import struct
import sys

CFS_M3S = 0.028316846592   # one cubic foot per second in m3/s (SWMM's own factor)
FLOW_TOL_CFS = 1.0e-5      # external_lateral flows below this are zeroed (consts.h)
SIDE_FILE = os.path.join("swmm", "hydraulics_exchange_replay.bin")
RPT_FILE = os.path.join("swmm", "hydraulics.rpt")


def die(msg):
    sys.stdout.write(msg + "\n")
    sys.exit(2)


def result(ok, name):
    sys.stdout.write("RESULT %s %s\n" % ("PASS" if ok else "FAIL", name))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------ the case

def geometry(case):
    lines = open(os.path.join(case, "elevation.dem")).read().split("\n")
    hdr = dict((l.split()[0].lower(), l.split()[1]) for l in lines[:6])
    nrows, ncols = int(hdr["nrows"]), int(hdr["ncols"])
    dem = [[float(x) for x in l.split()] for l in lines[6:] if l.strip()]
    if len(dem) != nrows or any(len(r) != ncols for r in dem):
        die("elevation.dem is not %d x %d" % (nrows, ncols))
    return nrows, ncols, float(hdr["xllcorner"]), float(hdr["yllcorner"]), float(hdr["cellsize"]), dem


def inp_section(case, name):
    out, on = [], False
    for l in open(os.path.join(case, "swmm", "hydraulics.inp")).read().split("\n"):
        s = l.strip()
        if s.startswith("["):
            on = (s.upper() == name)
            continue
        if on and s and not s.startswith(";"):
            out.append(s.split())
    return out


def cell_of(x, y, nrows, xll, yll, dx):
    """The solver's placement rule (calc_swmm_node_col and calc_swmm_node_row)."""
    return int(math.ceil((nrows - ((y - yll) / dx)) + 1e-16)) - 1, int(math.ceil(((x - xll) / dx) + 1e-16)) - 1


def node_cell(case, name):
    nrows, ncols, xll, yll, dx, dem = geometry(case)
    for f in inp_section(case, "[COORDINATES]"):
        if f[0].upper() == name.upper():
            return cell_of(float(f[1]), float(f[2]), nrows, xll, yll, dx)
    die("no [COORDINATES] entry for %s" % name)


def invert_of(case, name):
    for f in inp_section(case, "[OUTFALLS]"):
        if f[0].upper() == name.upper():
            return float(f[1])
    die("no [OUTFALLS] entry for %s" % name)


def owner_of(row, nrows, size):
    acc = 0
    for p in range(size):
        n = nrows // size + (1 if p < nrows % size else 0)
        if row < acc + n:
            return p
        acc += n
    return None


def cmd_layout(argv):
    case, np_, below, above = argv[0], int(argv[1]), argv[2], argv[3]
    nrows, ncols, xll, yll, dx, dem = geometry(case)
    ok = True
    for nm, want in ((below, "below"), (above, "above")):
        row, col = node_cell(case, nm)
        z, inv = dem[row][col], invert_of(case, nm)
        owner = owner_of(row, nrows, np_)
        good = (z < inv if want == "below" else z > inv) and owner != 0
        sys.stdout.write("OUTFALL %s row %d col %d dem %.4f invert %.4f cell_%s_invert owner_np%d %d %s\n"
                         % (nm, row, col, z, inv, want, np_, owner, "ok" if good else "NOT AS DESIGNED"))
        ok = ok and good
    return result(ok, "layout")


# ------------------------------------------------------------------------------------ the captures

def read_stage(out):
    p = os.path.join(out, "outfall_stage.bin")
    if not os.path.exists(p):
        die("%s is missing" % p)
    b = open(p, "rb").read()
    if len(b) < 16:
        die("%s is shorter than its 16-byte header" % p)
    magic, ver, width, n = struct.unpack_from("<4i", b, 0)
    if magic != 0x444F5331 or ver != 1 or width != 8:
        die("outfall_stage.bin header {%#x, %d, %d} is not {0x444f5331, 1, 8}" % (magic, ver, width))
    rec = 8 * (2 + 3 * n)
    if (len(b) - 16) % rec != 0:
        die("outfall_stage.bin: %d bytes is not 16 + records x %d" % (len(b), rec))
    names = [l.strip() for l in open(os.path.join(out, "outfall_stage.nodes")).read().split("\n") if l.strip()]
    if len(names) != n:
        die("outfall_stage.nodes names %d outfalls, the header says %d" % (len(names), n))
    recs = []
    for k in range((len(b) - 16) // rec):
        v = struct.unpack_from("<%dd" % (2 + 3 * n), b, 16 + k * rec)
        recs.append(dict(t=v[0], dt=v[1], stage=v[2:2 + n], q_out=v[2 + n:2 + 2 * n], q_back=v[2 + 2 * n:2 + 3 * n]))
    return names, recs


def read_side(out):
    p = os.path.join(out, SIDE_FILE)
    b = open(p, "rb").read()
    magic, n, w = struct.unpack_from("<3i", b, 0)
    rec = w * (1 + n)
    recs = [struct.unpack_from("<%dd" % (1 + n), b, 12 + k * rec) for k in range((len(b) - 12) // rec)]
    return n, recs


def read_t1d(out):
    b = open(os.path.join(out, "swmm_link_flow_diag.bin"), "rb").read()
    links = [l.strip() for l in open(os.path.join(out, "swmm_link_flow_diag.links")).read().split("\n") if l.strip()]
    magic, ver, width, n = struct.unpack_from("<4i", b, 0)
    if n != len(links):
        die("swmm_link_flow_diag.bin header says %d links, the sidecar names %d" % (n, len(links)))
    rec = 8 + 8 * n
    return links, [struct.unpack_from("<%dd" % (1 + n), b, 16 + k * rec) for k in range((len(b) - 16) // rec)]


def read_raster(path, nrows, ncols):
    b = open(path, "rb").read()
    n = nrows * ncols
    if len(b) < n * 8 or len(b) - n * 8 not in (0, 16):
        die("raster %s has %d bytes, which is not %d doubles plus a 0- or 16-byte header" % (path, len(b), n))
    vals = struct.unpack("<%dd" % n, b[len(b) - n * 8:])
    return [list(vals[i * ncols:(i + 1) * ncols]) for i in range(nrows)]


def cmd_stats(argv):
    names, recs = read_stage(argv[0])
    sys.stdout.write("outfall_stage.bin: %d records, outfalls %s\n" % (len(recs), names))
    for i, nm in enumerate(names):
        qo = [r["q_out"][i] for r in recs]
        qb = [r["q_back"][i] for r in recs]
        st = [r["stage"][i] for r in recs]
        sys.stdout.write("  %s: steps with discharge %d, with backflow reaching the engine %d, applied %d; stage min %.6g max %.6g; "
                         "max discharge %.6g m3/s, max backflow applied %.6g m3/s; discharge volume %.6g m3, backflow applied volume %.6g m3\n"
                         % (nm, sum(1 for q in qo if q > 0), sum(1 for q in qo if q < 0), sum(1 for q in qb if q > 0),
                            min(st), max(st), max(qo), max(qb),
                            sum(max(0.0, q) * r["dt"] for q, r in zip(qo, recs)), sum(q * r["dt"] for q, r in zip(qb, recs))))
    return 0


def cmd_stage_check(argv):
    out, case, name, print_interval, hextra, tol = argv[0], argv[1], argv[2], float(argv[3]), float(argv[4]), float(argv[5])
    nrows, ncols, xll, yll, dx, dem = geometry(case)
    names, recs = read_stage(out)
    if name not in names:
        die("outfall %s is not in outfall_stage.nodes %s" % (name, names))
    i = names.index(name)
    row, col = node_cell(case, name)
    z, inv = dem[row][col], invert_of(case, name)
    prints = sorted(f for f in os.listdir(os.path.join(out, "bin")) if re.match(r"^H_\d+_00\.out$", f))
    if not prints:
        die("no serial H raster in %s" % os.path.join(out, "bin"))
    wet = dry = 0
    removed_seen = False
    bad = []
    for f in prints:
        k = int(f.split("_")[1])
        t = k * print_interval
        hit = [r for r in recs if abs(r["t"] - t) < 1e-6]
        if len(hit) != 1:
            bad.append("print %d (t=%g): %d outfall_stage.bin records carry that stamp" % (k, t, len(hit)))
            continue
        r = hit[0]
        h_raster = read_raster(os.path.join(out, "bin", f), nrows, ncols)[row][col]
        removed = r["q_back"][i] * r["dt"] / (dx * dx)
        h_read = h_raster + removed
        if h_read >= hextra:
            want, kind = z + h_raster + removed, "wet"
            wet += 1
            removed_seen = removed_seen or r["q_back"][i] > 0.0
        else:
            want, kind = inv, "dry"
            dry += 1
        err = abs(r["stage"][i] - want)
        sys.stdout.write("  print %2d t=%6g %s: stage %.15g expected %.15g (DEM %.4f + H %.15g + removed %.6g) |diff| %.3g\n"
                         % (k, t, kind, r["stage"][i], want, z, h_raster, removed, err))
        if err > tol * max(1.0, abs(want)):
            bad.append("print %d (t=%g, %s): stage %.15g, expected %.15g" % (k, t, kind, r["stage"][i], want))
    sys.stdout.write("%s: cell (%d, %d), DEM %.4f, invert %.4f; %d print instants read, %d with the depth read at least hextra, %d below it\n"
                     % (name, row, col, z, inv, len(prints), wet, dry))
    if wet == 0:
        bad.append("no print instant has the depth read at least hextra: the stage rule's wet branch was not exercised")
    if not removed_seen:
        bad.append("no wet print instant has q_back_applied > 0: the check cannot see the backflow removal (a break that does not lower the cell or reads the wrong cell would pass)")
    for b in bad:
        sys.stdout.write("  FINDING: " + b + "\n")
    return result(not bad, "stage-check " + name)


def rpt_continuity(out):
    lines = open(os.path.join(out, RPT_FILE), errors="replace").read().split("\n")
    vals = {}
    for i, l in enumerate(lines):
        if l.strip().startswith("Flow Routing Continuity"):
            for m in lines[i + 2:i + 20]:
                mm = re.match(r"^\s*(External Inflow|External Outflow)\s*\.+\s+(\S+)\s+(\S+)\s*$", m)
                if mm:
                    vals[mm.group(1)] = (mm.group(2), mm.group(3))
            break
    if len(vals) != 2:
        die("the .rpt flow-routing continuity table has no External Inflow and External Outflow lines")
    return vals


def cmd_ledger(argv):
    out, case, name, print_interval = argv[0], argv[1], argv[2], float(argv[3])
    nodes = argv[4].split(",")
    link = argv[5] if len(argv) > 5 else ""
    names, recs = read_stage(out)
    i = names.index(name)
    ncoupled, side = read_side(out)
    if ncoupled != len(nodes):
        die("the exchange side-file has %d columns, NODES names %d" % (ncoupled, len(nodes)))
    if len(side) != len(recs):
        die("the exchange side-file has %d records, outfall_stage.bin %d" % (len(side), len(recs)))
    other = None
    if link:
        links, t1d = read_t1d(out)
        if link not in links:
            die("link %s is not in swmm_link_flow_diag.links %s" % (link, links))
        if len(t1d) != len(recs):
            die("swmm_link_flow_diag.bin has %d records, outfall_stage.bin %d" % (len(t1d), len(recs)))
        other = [r[1 + links.index(link)] * CFS_M3S for r in t1d]
    ei = eo = 0.0
    nbackflow = ndis = 0
    parts = {}   # label -> [inflow m3, outflow m3]: what each term contributed, for the log
    for k, (r, s) in enumerate(zip(recs, side)):
        dt = s[0]
        if abs(dt - r["dt"]) > 1e-12 * max(1.0, dt):
            die("step %d: side-file dt %.17g, outfall_stage.bin dt %.17g" % (k, dt, r["dt"]))
        q_o = [(name, r["q_out"][i])] + ([(link, other[k])] if other is not None else [])
        for lab, q in q_o:
            ei += dt * max(0.0, -q)
            eo += dt * max(0.0, q)
            p = parts.setdefault("outfall " + lab, [0.0, 0.0])
            p[0] += dt * max(0.0, -q)
            p[1] += dt * max(0.0, q)
        for lab, q_cfs in zip(nodes, s[1:]):
            # the side-file holds each node's exchange_q in cfs (the engine's own unit: the exchange
            # kernel writes -flow / FT3_TO_M3_FACTOR), and FLOW_TOL zeroes it in cfs
            q = 0.0 if abs(q_cfs) < FLOW_TOL_CFS else q_cfs * CFS_M3S
            ei += dt * max(0.0, q)
            eo += dt * max(0.0, -q)
            p = parts.setdefault("coupled " + lab, [0.0, 0.0])
            p[0] += dt * max(0.0, q)
            p[1] += dt * max(0.0, -q)
        nbackflow += 1 if r["q_out"][i] < 0 else 0
        ndis += 1 if r["q_out"][i] > 0 else 0
    for lab in sorted(parts):
        sys.stdout.write("  term %-16s inflow %.6f m3, outflow %.6f m3\n" % (lab, parts[lab][0], parts[lab][1]))
    vals = rpt_continuity(out)
    ok = True
    for label, mine in (("External Inflow", ei), ("External Outflow", eo)):
        printed = vals[label][1]
        decimals = len(printed.split(".")[1]) if "." in printed else 0
        mine_ml = mine / 1000.0
        diff = abs(float(printed) - mine_ml)
        # the table prints 10^6 ltr (1000 m3) to `decimals` places: one half unit of the last place
        good = diff <= 0.5 * 10.0 ** (-decimals) + 1e-12
        sys.stdout.write("  %-17s recomputed %.6f m3 = %.6f (10^6 ltr), printed %s (hectare-m %s): |diff| %.6f against a half unit %.4f -> %s\n"
                         % (label, mine, mine_ml, printed, vals[label][0], diff, 0.5 * 10.0 ** (-decimals), "ok" if good else "MISMATCH"))
        ok = ok and good
        if mine_ml < 10.0 ** (-decimals):
            sys.stdout.write("  FINDING: %s recomputed is below the table's printed precision: the case is too small for this check\n" % label)
            ok = False
    sys.stdout.write("  %d SWMM steps; %s: %d with discharge, %d with backflow; other outfall link %s\n" % (len(recs), name, ndis, nbackflow, link or "(none)"))
    return result(ok, "ledger " + name)


def cmd_residual(argv):
    out = argv[0]
    rows = [l for l in open(os.path.join(out, "domain_balance.csv")).read().split("\n") if l.strip() and not l.startswith("%")]
    hdr = [c.strip() for c in open(os.path.join(out, "domain_balance.csv")).readline().lstrip("%").split(",")]
    if "outfall_backflow_m3" not in hdr or "outfall_backflow_deficit_m3" not in hdr or len(hdr) != 14:
        die("domain_balance.csv header %s does not carry the two driven-outfall columns" % hdr)
    ci = dict((c, j) for j, c in enumerate(hdr))
    # the residual closes with every other term but the deficit in the signed sum
    worst = 0.0
    scale = 0.0
    for r in rows:
        v = [float(x) for x in r.split(",")]
        terms = sum(v[ci[c]] for c in hdr if c not in ("t_end", "d_storage_m3", "residual_m3", "outfall_backflow_deficit_m3"))
        res = v[ci["d_storage_m3"]] - terms
        worst = max(worst, abs(res - v[ci["residual_m3"]]))
        scale = max(scale, abs(v[ci["d_storage_m3"]]), abs(v[ci["outfall_backflow_m3"]]))
    resmax = max(abs(float(r.split(",")[ci["residual_m3"]])) for r in rows)
    sys.stdout.write("  %d rows; max |residual column| %.3g m3; max |d_storage - sum(terms) - residual column| %.3g m3; scale %.3g m3\n"
                     % (len(rows), resmax, worst, scale))
    ok = resmax <= 1e-9 * max(scale, 1.0) * 100 and worst <= 1e-9 * max(scale, 1.0) * 100
    nonzero = any(float(r.split(",")[ci["outfall_backflow_m3"]]) != 0.0 for r in rows)
    sys.stdout.write("  outfall_backflow_m3 nonzero in some row: %s\n" % nonzero)
    return result(ok and nonzero, "residual")


def cmd_edit_inp(argv):
    d, op, args = argv[0], argv[1], argv[2:]
    p = os.path.join(d, "swmm", "hydraulics.inp")
    lines = open(p).read().split("\n")
    sec = None
    out = []
    coords = {}
    for l in lines:
        s = l.strip()
        if s.startswith("["):
            sec = s.upper()
        elif sec == "[COORDINATES]" and s and not s.startswith(";"):
            f = s.split()
            coords[f[0].upper()] = (f[1], f[2])
    done = False
    for l in lines:
        s = l.strip()
        f = s.split()
        if s.startswith("["):
            sec = s.upper()
        if op == "coords" and sec == "[COORDINATES]" and f and f[0].upper() == args[0].upper() and not s.startswith(";"):
            x, y = coords[args[1].upper()]
            l = "%-7s %s  %s" % (f[0], x, y)
            done = True
        elif op == "outfall-type" and sec == "[OUTFALLS]" and f and f[0].upper() == args[0].upper() and not s.startswith(";"):
            l = "%-7s %s  %s" % (f[0], f[1], " ".join(args[1:]))
            done = True
        out.append(l)
        if op == "add-conduit" and s.upper() == "[CONDUITS]":
            out.append("%-7s %s  %s  40.0  0.013  0  0  0  0" % (args[0], args[1], args[2]))
            done = True
        if op == "add-conduit" and s.upper() == "[XSECTIONS]":
            out.append("%-7s CIRCULAR  0.5  0  0  0  1" % args[0])
    if not done:
        die("edit-inp %s %s: nothing to edit" % (op, " ".join(args)))
    open(p, "w").write("\n".join(out))
    sys.stdout.write("edited %s: %s %s\n" % (p, op, " ".join(args)))
    return 0


def cmd_fixture_no_backflow(argv):
    """fixture-no-backflow OUT CASE NAME PRINT HEXTRA DST: a copy of OUT whose outfall_stage.bin carries q_back_applied 0 and, at each print
    instant, the stage the rule gives without a removal, so that stage-check's only remaining finding is its must-be-nonzero control."""
    import shutil
    out, case, name, print_interval, hextra, dst = argv[0], argv[1], argv[2], float(argv[3]), float(argv[4]), argv[5]
    nrows, ncols, xll, yll, dx, dem = geometry(case)
    names, recs = read_stage(out)
    i = names.index(name)
    row, col = node_cell(case, name)
    z, inv = dem[row][col], invert_of(case, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(out, dst)
    b = bytearray(open(os.path.join(dst, "outfall_stage.bin"), "rb").read())
    n = len(names)
    rec = 8 * (2 + 3 * n)
    for k, r in enumerate(recs):
        off = 16 + k * rec
        struct.pack_into("<d", b, off + 8 * (2 + 2 * n + i), 0.0)
        pi = r["t"] / print_interval
        if abs(pi - round(pi)) < 1e-9 and round(pi) >= 1:
            h = read_raster(os.path.join(dst, "bin", "H_%02d_00.out" % round(pi)), nrows, ncols)[row][col]
            struct.pack_into("<d", b, off + 8 * (2 + i), z + h if h >= hextra else inv)
    open(os.path.join(dst, "outfall_stage.bin"), "wb").write(bytes(b))
    sys.stdout.write("wrote %s\n" % dst)
    return 0


if __name__ == "__main__":
    fn = {"layout": cmd_layout, "stats": cmd_stats, "stage-check": cmd_stage_check, "ledger": cmd_ledger,
          "residual": cmd_residual, "edit-inp": cmd_edit_inp, "fixture-no-backflow": cmd_fixture_no_backflow}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if fn is None:
        die("unknown subcommand")
    sys.exit(fn(sys.argv[2:]))
