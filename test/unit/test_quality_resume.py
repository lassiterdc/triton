#!/usr/bin/env python3
"""A snapshot resume of a coupled model that routes water quality equals the uninterrupted run.

The problem this guards: SWMM's water-quality state -- node and link
concentrations (Node/Link oldQual and newQual), link loads (Link totalLoad) and
the per-pollutant step accumulator behind the Quality Routing Continuity totals
(StepQualTotals) -- crosses every routing step. A snapshot that does not carry
it restarts every concentration at the pollutant's initial value and every link
load at zero when a run resumes from it, while the quality totals and outfall
loads it does carry continue. The resumed quality results then differ from the
uninterrupted run's.

What this does: in a fresh temporary directory it writes a 16 x 16 DEM, an .inp
and a cfg (the layout of test_zero_node_refusal.py: a symbolic link to the built
triton.exe at {tmp}/proj/bin/triton.exe, so the project directory is {tmp}/proj,
and the relative output_folder "out"). The .inp is coupled (two junctions carry
an [INFLOWS] FLOW line), uses CMS units and DYNWAVE, routes quality
(IGNORE_QUALITY NO), and carries one pollutant with a non-zero [DWF]
concentration on top of a non-zero [DWF] flow at both coupled junctions, one
street conduit with one inlet ([STREETS], [INLETS], [INLET_USAGE]) capturing
into the sewer, and one [TREATMENT] expression at a downstream sewer node. The
cfg checkpoints three times (print_interval 600 s over 1800 s).

It runs the model once, uninterrupted (U). It then resumes twice from
checkpoint 2, each resume on its own copy of U's output tree and launched on
that copy's cfg/config_2.cfg:
  arm A takes the default route and must print "path=snapshot";
  arm B adds swmm_snapshot_disable=1 to its config_2.cfg and must print
  "path=replay".
Both arms must equal U byte for byte in every swmm/*.out file (the SWMM binary
output and every per-node node*.out) and, in the .rpt, in the sections
"Quality Routing Continuity", "Link Pollutant Load Summary" and "Outfall Loading
Summary", each extracted by its header because the .rpt's timing lines differ
between runs. A missing section, a missing file or an empty node*.out set fails.

So the test is not vacuous, U's SWMM .out must carry the pollutant at a maximum
node concentration above 0: a model carrying no mass would pass every equality.
Arm B is the control showing the instrument can report equality; arm A is the
property.

--fixture-only DIR writes the fixture into DIR (no run) and prints the cfg
path; --ignore-quality writes IGNORE_QUALITY YES. Both exist for the
regression comparison that runs this model clean on two builds.

Usage:
  test_quality_resume.py --run-command "mpirun -n 4" --exe /path/to/triton.exe
"""
import argparse
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

N = 16
CELL = 10.0
CHECKPOINT = 2
SECTIONS = ("Quality Routing Continuity", "Link Pollutant Load Summary", "Outfall Loading Summary")

INP = """[TITLE]
Quality-routed coupled snapshot-resume fixture

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
IGNORE_QUALITY       {ignore_quality}
START_DATE           01/01/2000
START_TIME           00:00:00
REPORT_START_DATE    01/01/2000
REPORT_START_TIME    00:00:00
END_DATE             01/01/2000
END_TIME             01:00:00
REPORT_STEP          00:05:00
WET_STEP             00:00:10
DRY_STEP             00:01:00
ROUTING_STEP         00:00:01

[POLLUTANTS]
;;Name  Units  Crain  Cgw  Crdii  Kdecay  SnowOnly  CoPollut  CoFrac  Cdwf  Cinit
POLL    MG/L   0.0    0.0  0.0    0.0     NO        *         0.0     0.0   0.0

[JUNCTIONS]
;;Name  InvertElev  MaxDepth  InitDepth  SurchargeDepth  PondedArea
J1      0.0         1.0       0          0               0
J2      -0.1        1.1       0          0               0
S1      0.80        0.2       0          0               0
S2      0.75        0.25      0          0               0

[OUTFALLS]
;;Name  InvertElev  Type
O1      -0.3        FREE
O2      0.70        FREE

[CONDUITS]
;;Name  From  To  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow
C1      J1    J2  50.0    0.013      0         0          0         0
C2      J2    O1  50.0    0.013      0         0          0         0
ST1     S1    S2  50.0    0.016      0         0          0         0
ST2     S2    O2  50.0    0.016      0         0          0         0

[XSECTIONS]
;;Link  Shape     Geom1  Geom2  Geom3  Geom4  Barrels
C1      CIRCULAR  0.5    0      0      0      1
C2      CIRCULAR  0.5    0      0      0      1
ST1     STREET    STR
ST2     STREET    STR

[STREETS]
;;Name  Tcrown  Hcurb  Sx    nRoad  a     W    Sides  Tback  Sback  nBack
STR     5.0     0.15   0.02  0.016  0.0   0.0  2      0.0    0.0    0.0

[INLETS]
;;Name  Type   Length  Width  GrateType
GR1     GRATE  1.0     0.6    P_BAR-50

[INLET_USAGE]
;;Conduit  Inlet  Node  Number  %Clogged  Qmax  aLocal  wLocal  Placement
ST1        GR1    J2    1       0         0     0       0       ON_GRADE

[INFLOWS]
;;Node  Constituent  TimeSeries  Type  Mfactor  Sfactor  Baseline
J1      FLOW         ""          FLOW  1.0      1        0
S1      FLOW         ""          FLOW  1.0      1        0

[DWF]
;;Node  Constituent  Baseline
J1      FLOW         0.010
J1      POLL         100.0
S1      FLOW         0.020
S1      POLL         50.0

[TREATMENT]
;;Node  Pollutant  Function
J2      POLL       R = 0.5

[REPORT]
INPUT      NO
CONTROLS   NO
NODES ALL
LINKS ALL

[COORDINATES]
;;Node  X      Y
J1      25.0   25.0
J2      75.0   25.0
O1      125.0  25.0
S1      25.0   105.0
S2      75.0   105.0
O2      125.0  105.0
"""


def write_fixture(d, ignore_quality=False):
    """Write the DEM, the .inp and the cfg into directory d; return the cfg's path."""
    dem = os.path.join(d, "dem.asc")
    with open(dem, "w") as f:
        f.write("ncols         %d\nnrows         %d\nxllcorner     0.0\nyllcorner     0.0\n"
                "cellsize      %.1f\nNODATA_value  -9999\n" % (N, N, CELL))
        for _ in range(N):
            f.write(" ".join(["1.0"] * N) + "\n")
    inp = os.path.join(d, "model.inp")
    with open(inp, "w") as f:
        f.write(INP.format(ignore_quality="YES" if ignore_quality else "NO"))
    cfg = os.path.join(d, "run.cfg")
    with open(cfg, "w") as f:
        f.write('dem_filename="%s"\n' % dem)
        f.write('inp_filename="%s"\n' % inp)
        f.write('output_folder="out"\n')
        f.write('input_format=ASC\ninput_option=SEQ\noutfile_pattern="%s/%s/%s_%02d_%02d"\n'
                "output_format=BIN\noutput_option=SEQ\n"
                "manhole_diameter=1.2\nmanhole_loss=0.1\n"
                'num_sources=0\nn_infile=""\nconst_mann=0.03\nnum_runoffs=0\nnum_extbc=0\n'
                'observation_loc_file=""\ntime_series_flag=0\nprint_option=huv\n'
                "sim_start_time=0\nsim_duration=1800\ncheckpoint_id=0\n"
                "time_increment_fixed=0\ntime_step=1.0\nprint_interval=600.0\n"
                "it_count=0\ncourant=0.5\nhextra=0.001\ngpu_direct_flag=0\n"
                "domain_decomposition=static\nopen_boundaries=0\n")
    return cfg


def max_pollutant_concentration(path):
    """Return the maximum node concentration of pollutant 0 over every period of a SWMM .out."""
    with open(path, "rb") as f:
        data = f.read()
    i4 = lambda off: struct.unpack_from("<i", data, off)[0]
    n_sub, n_node, n_link, n_poll = (i4(12), i4(16), i4(20), i4(24))
    if n_poll < 1:
        return None
    end = len(data) - 24
    input_start, output_start, n_periods = i4(end + 4), i4(end + 8), i4(end + 12)
    off = input_start
    k = i4(off); off += 4 + 4 * k + 4 * n_sub          # subcatch area property
    k = i4(off); off += 4 + 4 * k + 4 * k * n_node     # node properties
    k = i4(off); off += 4 + 4 * k + 4 * k * n_link     # link properties
    n_sub_vars = i4(off); off += 4 + 4 * n_sub_vars
    n_node_vars = i4(off); off += 4 + 4 * n_node_vars
    n_link_vars = i4(off); off += 4 + 4 * n_link_vars
    n_sys_vars = i4(off)
    period = 8 + 4 * (n_sub * n_sub_vars + n_node * n_node_vars + n_link * n_link_vars + n_sys_vars)
    best = 0.0
    for p in range(n_periods):
        base = output_start + p * period + 8 + 4 * n_sub * n_sub_vars
        for j in range(n_node):
            v = struct.unpack_from("<f", data, base + 4 * (j * n_node_vars + n_node_vars - n_poll))[0]
            best = max(best, v)
    return best


def rpt_section(path, header):
    """Return the lines of one .rpt section, from its header to the next banner, or None.

    The header line is matched by prefix: SWMM writes a continuity table's
    column titles (the pollutant names) on the same line as its header.
    """
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(header):
            out = []
            j = i + 2                                   # skip the banner below the header
            while j < len(lines) and not lines[j].strip().startswith("*****"):
                out.append(lines[j])
                j += 1
            return out
    return None


def swmm_dir(out_dir):
    return os.path.join(out_dir, "swmm")


def compare(u_out, r_out, label):
    """Compare a resumed output tree against U; return a list of failure strings."""
    fails = []
    u_sw, r_sw = swmm_dir(u_out), swmm_dir(r_out)
    outs = sorted(n for n in os.listdir(u_sw) if n.endswith(".out"))
    nodes = [n for n in outs if n.startswith("node")]
    if not any(not n.startswith("node") for n in outs):
        fails.append("%s: U has no SWMM binary .out" % label)
    if not nodes:
        fails.append("%s: U has no node*.out file" % label)
    for n in outs:
        a, b = os.path.join(u_sw, n), os.path.join(r_sw, n)
        if not os.path.exists(b):
            fails.append("%s: %s is missing" % (label, n))
            continue
        with open(a, "rb") as fa, open(b, "rb") as fb:
            if fa.read() != fb.read():
                fails.append("%s: %s differs from U" % (label, n))
    rpts = sorted(n for n in os.listdir(u_sw) if n.endswith(".rpt"))
    if not rpts:
        fails.append("%s: U has no .rpt" % label)
    for n in rpts:
        for s in SECTIONS:
            a = rpt_section(os.path.join(u_sw, n), s)
            b = rpt_section(os.path.join(r_sw, n), s) if os.path.exists(os.path.join(r_sw, n)) else None
            if a is None:
                fails.append("%s: U's %s has no '%s' section" % (label, n, s))
            elif a != b:
                fails.append("%s: '%s' in %s differs from U" % (label, s, n))
    print("%s: compared %d .out file(s) (%d node*.out) and %d .rpt section(s): %s"
          % (label, len(outs), len(nodes), len(rpts) * len(SECTIONS), "EQUAL" if not fails else "DIFFER"))
    return fails


def run(cmd, cwd, tag):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print("----- %s: %s (exit %d) -----" % (tag, " ".join(cmd), r.returncode))
    print(r.stdout[-6000:])
    return r.returncode, r.stdout


def main():
    """Build the fixture, run U and the two resumes, and judge them; return 0 only on PASS."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-command", default="")
    ap.add_argument("--exe")
    ap.add_argument("--fixture-only")
    ap.add_argument("--ignore-quality", action="store_true")
    a = ap.parse_args()
    if a.fixture_only:
        os.makedirs(a.fixture_only, exist_ok=True)
        print(write_fixture(a.fixture_only, a.ignore_quality))
        return 0
    if not a.exe:
        ap.error("--exe is required unless --fixture-only is given")

    d = tempfile.mkdtemp(prefix="triton_quality_resume_")
    fails = []
    try:
        cfg = write_fixture(d, a.ignore_quality)
        proj = os.path.join(d, "proj")
        exe = os.path.join(proj, "bin", "triton.exe")
        os.makedirs(os.path.dirname(exe))
        os.symlink(os.path.abspath(a.exe), exe)
        launch = shlex.split(a.run_command) + [exe]

        rc, out = run(launch + [cfg], d, "U")
        u_out = os.path.join(proj, "out")
        if rc != 0 or "Simulation ends" not in out:
            fails.append("U did not complete")
        else:
            outs = [n for n in os.listdir(swmm_dir(u_out)) if n.endswith(".out") and not n.startswith("node")]
            cmax = max_pollutant_concentration(os.path.join(swmm_dir(u_out), outs[0])) if outs else None
            print("U: maximum node concentration of the pollutant = %r" % cmax)
            if cmax is None or not cmax > 0.0:
                fails.append("U carries no pollutant mass (maximum concentration %r): the test would be vacuous" % cmax)
            # Diagnostic only, never judged: what the inlet and the treatment did in U.
            for n in sorted(n for n in os.listdir(swmm_dir(u_out)) if n.endswith(".rpt")):
                with open(os.path.join(swmm_dir(u_out), n), errors="replace") as f:
                    rows = [l for l in f.read().splitlines() if l.split()[:1] == ["ST1"]]
                print("U %s rows for street conduit ST1 (its inlet GR1 captures into J2):" % n)
                for l in rows:
                    print("    " + l)
                print("U %s Quality Routing Continuity:" % n)
                for l in rpt_section(os.path.join(swmm_dir(u_out), n), SECTIONS[0]) or []:
                    print("    " + l)

            for arm, route, extra in (("A", "path=snapshot", ""), ("B", "path=replay", "swmm_snapshot_disable=1\n")):
                name = "out_" + arm
                r_out = os.path.join(proj, name)
                shutil.copytree(u_out, r_out, symlinks=True)
                # Non-vacuity: the copy starts out equal to U, so remove what the resume must
                # regenerate (the .rpt and every node*.out) and record the binary .out's stamp.
                rsw = swmm_dir(r_out)
                removed = [n for n in os.listdir(rsw) if n.endswith(".rpt") or (n.startswith("node") and n.endswith(".out"))]
                for n in removed:
                    os.remove(os.path.join(rsw, n))
                bins = [n for n in os.listdir(rsw) if n.endswith(".out") and not n.startswith("node")]
                before = {n: os.stat(os.path.join(rsw, n)).st_mtime_ns for n in bins}
                print("arm %s: removed %d regenerable file(s) before the resume: %s" % (arm, len(removed), removed))
                rcfg = os.path.join(r_out, "cfg", "config_%d.cfg" % CHECKPOINT)
                with open(rcfg) as f:
                    text = f.read()
                text = re.sub(r'(?m)^output_folder=.*$', 'output_folder="%s"' % name, text)
                if not text.endswith("\n"):
                    text += "\n"
                with open(rcfg, "w") as f:
                    f.write(text + extra)
                rc, out = run(launch + [rcfg], d, "arm " + arm)
                if rc != 0 or "Simulation ends" not in out:
                    fails.append("arm %s did not complete" % arm)
                    continue
                after = {n: os.stat(os.path.join(rsw, n)).st_mtime_ns for n in bins if os.path.exists(os.path.join(rsw, n))}
                untouched = [n for n in bins if after.get(n) == before[n]]
                print("arm %s: binary .out rewritten: %s" % (arm, not untouched))
                if untouched or not bins or not removed:
                    fails.append("arm %s: the resume did not rewrite its outputs (untouched %s, removed %d)" % (arm, untouched, len(removed)))
                if route not in out:
                    fails.append("arm %s did not take the expected route (%s)" % (arm, route))
                fails += compare(u_out, r_out, "arm " + arm)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    for f in fails:
        print("FAIL: " + f)
    print("PASS" if not fails else "FAIL (%d)" % len(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
