#!/usr/bin/env python3
"""Writes the fixed coupled case of the diagnostics neutrality suite (test/coupled_case/).

The committed files in this directory are this script's output; the script is kept so the case
is reproducible and its design reads in one place:

    python3 make_case.py [DIR]        (default: the directory holding this script)

The case is a small coupled TRITON-SWMM model chosen to enter every branch the diagnostics
captures add, which no other case of the repository does:

  * a 64 x 48 domain of 5 m cells, sloping down to the east and to the north, so open-edge
    water leaves through the east edge and the north edge at a nonzero rate (T2 edge reads);
    64 rows, because the DEM is split into row strips, one per rank, and the resume suite runs
    16 ranks (a strip must hold at least one row, and the rank-0 strip must hold no manhole);
  * a type-1 external boundary (water level against time) on part of the west edge, holding a
    stage above the ground there, so the boundary cells are set wet every step (T2 extbc_m3);
  * a pond on the initial depth grid near the north-east corner (the domain starts wet, so the
    stored volume S(0) is nonzero) and one inflow hydrograph source (T2 inflow term);
  * a coupled SWMM model with four junctions and two outfalls whose inflow is an external time
    series file (swmm/dats/inflow.dat, which also makes the resume suite's degraded section
    possible: without the file SWMM stops with error 361), large enough to flood J1 and J3
    (path-3 flooded branch, swmm_exchange_m3);
  * MAX_TRIALS 1 and HEAD_TOLERANCE 0.0000001 after ROUTING_STEP in the .inp, which leaves
    nodes non-converged on every step (T1a variable block); it changes the model's hydraulics,
    so every run of the suite, capture on or off, uses it.

Every SWMM junction's invert is set to its cell's DEM elevation minus its MaxDepth, so the
rim of each node is the ground at its cell.
"""
import math
import os
import sys

NR, NC = 64, 48
DX = 5.0
PRINT_INTERVAL = 60.0
SIM_DURATION = 1800.0
MAXDEPTH = 1.0


def ground(i, j):
    """Elevation (m) of the cell in row i (0 = north) and column j (0 = west)."""
    return 1.0 + 1.5 * (1.0 - j / (NC - 1.0)) + 1.0 * (i / (NR - 1.0))


def cell_of(x, y):
    """(row, col) of the cell holding the point (x, y), the grid's origin at (0, 0)."""
    return int((NR * DX - y) // DX), int(x // DX)


def xy_of(i, j):
    """Cell-centre coordinates of the cell in row i and column j."""
    return (j + 0.5) * DX, (NR - i - 0.5) * DX


# (name, x, y) of the four junctions and two outfalls, all inside the DEM.
JUNCTIONS = [("J1", 100.0, 160.0), ("J2", 140.0, 160.0), ("J3", 100.0, 200.0), ("J4", 140.0, 200.0)]
OUTFALLS = [("O1", 180.0, 160.0), ("O2", 180.0, 200.0)]
# the west-edge rows the type-1 boundary covers (row 0 is the northern edge)
EXTBC_ROWS = (14, 30)
EXTBC_LEVEL = 3.9


def dem_text():
    lines = ["ncols         %d" % NC, "nrows         %d" % NR, "xllcorner     0.0", "yllcorner     0.0",
             "cellsize      %.1f" % DX, "NODATA_value  -9999"]
    for i in range(NR):
        lines.append(" ".join("%.4f" % ground(i, j) for j in range(NC)))
    return "\n".join(lines) + "\n"


def init_depth_text():
    """Initial depth grid: no header, one row of NC values per line (the format of *.inith)."""
    rows = []
    for i in range(NR):
        vals = []
        for j in range(NC):
            pond = 2 <= i <= 8 and 36 <= j <= 44
            vals.append("0.3000" if pond else "0.0000")
        rows.append(" ".join(vals))
    return "\n".join(rows) + "\n"


def inp_text():
    node_lines, coords = [], []
    zr = {}
    for name, x, y in JUNCTIONS:
        i, j = cell_of(x, y)
        zr[name] = ground(i, j)
        node_lines.append("%-7s %-11.4f %-9.1f 0          0               0" % (name, zr[name] - MAXDEPTH, MAXDEPTH))
        coords.append("%-7s %-6.1f %-6.1f" % (name, x, y))
    out_lines = []
    for (name, x, y), up in zip(OUTFALLS, ("J2", "J4")):
        out_lines.append("%-7s %-11.4f FREE" % (name, zr[up] - MAXDEPTH - 0.3))
        coords.append("%-7s %-6.1f %-6.1f" % (name, x, y))
    return """[TITLE]
Fixed coupled case of the diagnostics neutrality suite (see make_case.py)

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
IGNORE_QUALITY       YES
START_DATE           01/01/2000
START_TIME           00:00:00
REPORT_START_DATE    01/01/2000
REPORT_START_TIME    00:00:00
END_DATE             01/01/2000
END_TIME             01:00:00
REPORT_STEP          00:01:00
WET_STEP             00:00:10
DRY_STEP             00:01:00
ROUTING_STEP         00:00:01
MAX_TRIALS           1
HEAD_TOLERANCE       0.0000001

[JUNCTIONS]
;;Name  InvertElev  MaxDepth  InitDepth  SurchargeDepth  PondedArea
%s

[OUTFALLS]
;;Name  InvertElev  Type
%s

[CONDUITS]
;;Name  From  To  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow
C1      J1    J2  40.0    0.013      0         0          0         0
C2      J2    O1  40.0    0.013      0         0          0         0
C3      J3    J4  40.0    0.013      0         0          0         0
C4      J4    O2  40.0    0.013      0         0          0         0

[XSECTIONS]
;;Link  Shape     Geom1  Geom2  Geom3  Geom4  Barrels
C1      CIRCULAR  0.5    0      0      0      1
C2      CIRCULAR  0.5    0      0      0      1
C3      CIRCULAR  0.5    0      0      0      1
C4      CIRCULAR  0.5    0      0      0      1

[TIMESERIES]
;;Name  File
TS1     FILE "dats/inflow.dat"

[INFLOWS]
;;Node  Constituent  TimeSeries  Type  Mfactor  Sfactor  Baseline
J1      FLOW         TS1         FLOW  1.0      1        0
J3      FLOW         ""          FLOW  1.0      1        0.4

[REPORT]
INPUT      NO
CONTROLS   NO
NODES ALL
LINKS ALL

[COORDINATES]
;;Node  X      Y
%s
""" % ("\n".join(node_lines), "\n".join(out_lines), "\n".join(coords))


def inflow_dat_text():
    """SWMM external time-series file: J1's inflow (cms), a flood pulse that ends at 25 minutes."""
    pts = [(0, 0.0), (2, 2.5), (20, 2.5), (25, 0.1), (60, 0.1)]
    return "".join("01/01/2000 %02d:%02d %.3f\n" % (m // 60, m % 60, q) for m, q in pts)


def hydrograph_text():
    """Surface inflow source, time in hours and discharge in m3/s."""
    return "% Hydrograph\n% Time(hr) Discharge(cms)\n0,0.0\n0.1,0.8\n0.4,0.8\n0.5,0.0\n1.0,0.0\n"


def source_loc_text():
    x, y = xy_of(30, 42)
    return "%%X-Location,Y-Location\n%.1f,%.1f\n" % (x, y)


def extbc_text():
    _, y1 = xy_of(EXTBC_ROWS[0], 0)
    _, y2 = xy_of(EXTBC_ROWS[1], 0)
    x = DX / 2.0
    return "%% BC Type, X1, Y1, X2, Y2, BC\n1,%.1f,%.1f,%.1f,%.1f,west_level.txt\n" % (x, y1, x, y2)


def extbc_level_text():
    return "%% time (hr), water level (m)\n0.0,%.2f\n1.0,%.2f\n" % (EXTBC_LEVEL, EXTBC_LEVEL)


CFG = """#---------------------------------------------------------------------------
# Fixed coupled case of the diagnostics neutrality suite (see make_case.py).
# Every path is relative to the working directory the suite runs in.
#---------------------------------------------------------------------------
dem_filename="elevation.dem"
inp_filename="swmm/hydraulics.inp"
manhole_diameter=1.2
manhole_loss=0.1
input_format=ASC
input_option=SEQ
outfile_pattern="%s/%s/%s_%02d_%02d"
output_format=BIN
output_option=SEQ
output_folder="output"
num_sources=1
hydrograph_filename="inflow.hyg"
src_loc_file="source.src"
n_infile=""
const_mann=0.03
num_runoffs=0
runoff_filename=""
runoff_map=""
num_extbc=1
extbc_dir="."
extbc_file="west.extbc"
observation_loc_file=""
time_series_flag=0
print_option=huv
max_value_print_option=h
sim_start_time=0
sim_duration=@SIM@
checkpoint_id=0
time_increment_fixed=0
time_step=1.0
print_interval=@PRINT@
h_infile="init_h.dem"
qx_infile=""
qy_infile=""
it_count=0
courant=0.5
hextra=0.001
gpu_direct_flag=0
domain_decomposition=static
factor_interval_domain_decomposition=2
open_boundaries=1
"""


def main():
    d = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(d, "swmm", "dats"), exist_ok=True)
    files = {
        "elevation.dem": dem_text(), "init_h.dem": init_depth_text(),
        "inflow.hyg": hydrograph_text(), "source.src": source_loc_text(),
        "west.extbc": extbc_text(), "west_level.txt": extbc_level_text(),
        "swmm/hydraulics.inp": inp_text(), "swmm/dats/inflow.dat": inflow_dat_text(),
        "TRITONSWMM.cfg": CFG.replace("@SIM@", "%d" % SIM_DURATION).replace("@PRINT@", "%d" % PRINT_INTERVAL),
    }
    for name, text in files.items():
        with open(os.path.join(d, name), "w", newline="\n") as f:
            f.write(text)
        print("wrote", name, len(text), "bytes")


if __name__ == "__main__":
    main()
