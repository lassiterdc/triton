#!/usr/bin/env python3
"""A coupled run that couples no SWMM node must stop at startup, not run.

The problem this guards: a TRITON-SWMM cfg names an .inp, but no node of that
model is a coupling node -- here, the model has no [INFLOWS] FLOW line and the
cfg gives no swmm_coupling_nodes. TRITON then opens and starts SWMM and never
steps it. Without the refusal in init_swmm the run finishes with exit code 0 as
an uncoupled surface run and leaves a fragment of a SWMM report, which reads as
a successful coupled simulation. With the refusal, it stops with an error naming
the model and the words "couples no node".

What this does: in a fresh temporary directory it writes a 16 x 16 DEM, an .inp
that SWMM opens and starts cleanly (CMS units, dynamic wave, an end date after
the start date, one junction and one unconnected outfall, both inside the DEM,
and no [INFLOWS] section) and a cfg that names them by absolute path, with the
RELATIVE output_folder "out". It places a symbolic link to the built
triton.exe at {tmp}/proj/bin/triton.exe and runs
"{run command} {tmp}/proj/bin/triton.exe {cfg}" once, with the temporary
directory as the working directory, and passes only if BOTH the exit
code is non-zero AND the combined output contains "couples no node". The two
conditions are separate on purpose: a run that stops for another reason has a
non-zero exit code and no such line, and a run that prints the line and carries
on has the line and exit code 0. The temporary directory is removed afterwards.

Why the symlink, and not an absolute output_folder: TRITON takes its project
directory to be two directories above the executable's path as invoked, without
resolving links, and on a coupled run builds SWMM's report and output folders by
joining that directory with output_folder. An absolute output_folder therefore
becomes a path under the project directory that cannot be created, and SWMM stops
at swmm_open with ERROR 305 before the refusal is reached. Launched through
{tmp}/proj/bin/triton.exe, the project directory is {tmp}/proj, so every output
-- TRITON's, SWMM's and the cid marker -- lands in the temporary directory and is
removed with it. The executable, the launcher and the path from startup to the
refusal are the ones a production run takes; only where outputs land moves. Not
a copy: the link is made at test time, so it always names the current build.

16 rows, because TRITON splits the DEM into row strips, one per MPI rank, and
nothing refuses a rank that gets no rows; 16 covers the largest rank count a
machine file launches (8).

The combined output is always printed, so a CTest log shows what the run said.

Usage:
  test_zero_node_refusal.py --run-command "mpirun -n 4" --exe /path/to/triton.exe
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

N = 16
CELL = 10.0

INP = """[TITLE]
Zero-coupling-node refusal fixture

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2000
START_TIME           00:00:00
REPORT_START_DATE    01/01/2000
REPORT_START_TIME    00:00:00
END_DATE             01/01/2000
END_TIME             01:00:00
REPORT_STEP          00:10:00
WET_STEP             00:00:10
DRY_STEP             00:01:00
ROUTING_STEP         00:00:01

[JUNCTIONS]
;;Name  InvertElev  MaxDepth  InitDepth  SurchargeDepth  PondedArea
J1      0.0         1.0       0          0               0

[OUTFALLS]
;;Name  InvertElev  Type
O1      -0.5        FREE

[COORDINATES]
;;Node  X      Y
J1      45.0   45.0
O1      115.0  115.0
"""


def write_fixture(d):
    """Write the DEM, the .inp and the cfg into directory d; return the cfg's path.

    The DEM and .inp are named by absolute path and output_folder is the relative
    "out", which TRITON resolves against its project directory.
    """
    dem = os.path.join(d, "dem.asc")
    with open(dem, "w") as f:
        f.write("ncols         %d\nnrows         %d\nxllcorner     0.0\nyllcorner     0.0\n"
                "cellsize      %.1f\nNODATA_value  -9999\n" % (N, N, CELL))
        for _ in range(N):
            f.write(" ".join(["1.0"] * N) + "\n")
    inp = os.path.join(d, "model.inp")
    with open(inp, "w") as f:
        f.write(INP)
    cfg = os.path.join(d, "run.cfg")
    with open(cfg, "w") as f:
        f.write('dem_filename="%s"\n' % dem)
        f.write('inp_filename="%s"\n' % inp)
        f.write('output_folder="out"\n')
        f.write('input_format=ASC\ninput_option=SEQ\noutfile_pattern="%s/%s/%s_%02d_%02d"\n'
                "output_format=BIN\noutput_option=SEQ\n"
                "manhole_diameter=1.2\nmanhole_loss=0.1\n"
                'num_sources=0\nn_infile=""\nconst_mann=0.03\nnum_runoffs=0\nnum_extbc=0\n'
                'observation_loc_file=""\ntime_series_flag=0\nprint_option=h\n'
                "sim_start_time=0\nsim_duration=2.0\ncheckpoint_id=0\n"
                "time_increment_fixed=0\ntime_step=0.1\nprint_interval=1.0\n"
                "it_count=0\ncourant=0.5\nhextra=0.001\ngpu_direct_flag=0\n"
                "domain_decomposition=static\nopen_boundaries=0\n")
    return cfg


def main():
    """Build the fixture, run it once through the symlinked executable, and judge it.

    Returns 0 only when the run's exit code is non-zero and its combined output
    contains "couples no node"; prints the command, the exit code and the output
    either way.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-command", default="")
    ap.add_argument("--exe", required=True)
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix="triton_zero_node_")
    try:
        cfg = write_fixture(d)
        exe = os.path.join(d, "proj", "bin", "triton.exe")
        os.makedirs(os.path.dirname(exe))
        os.symlink(os.path.abspath(a.exe), exe)
        cmd = shlex.split(a.run_command) + [exe, cfg]
        r = subprocess.run(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = r.stdout
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("command: %s" % " ".join(cmd))
    print("exit code: %d" % r.returncode)
    print("----- combined output -----")
    print(out)
    print("---------------------------")
    refused = "couples no node" in out
    ok = r.returncode != 0 and refused
    print("non-zero exit code: %s; 'couples no node' present: %s -> %s"
          % (r.returncode != 0, refused, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
