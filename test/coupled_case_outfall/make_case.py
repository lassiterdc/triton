#!/usr/bin/env python3
"""Writes the fixed outfall case of the neutrality suite (test/coupled_case_outfall/, cases N14).

The committed files in this directory are this script's output; the script is kept so the case
is reproducible and its design reads in one place:

    python3 make_case.py [DIR]        (default: the directory holding this script)

It is a SECOND case directory, beside test/coupled_case/, whose make_case.py is not edited because
every earlier case of the suite reads its outputs. The two cases share the grid, the time axis and
the surface forcing's shape (without the west boundary, whose stage would flood the domain) so one suite instance runs both; what this case adds is two SWMM
OUTFALL nodes of type FIXED, each on its own cell with ONE connected conduit, in the two forms of
the Norfolk model the driven-stage outfall (cfg key swmm_driven_outfalls) is built for:

  * O1: the cell's elevation lies BELOW the outfall's invert (a depression, the "bay" cell of a
    shoreline outfall that is elevated above the water); a stage driven to the cell's water
    surface reaches the invert only when the cell holds more than invert - z of water;
  * O2: the cell's elevation lies ABOVE the outfall's invert (an outfall set below the ground);
    a stage driven to the cell's water surface z + h is above the invert whenever the cell is
    wet by the solver's own test (h >= hextra).

The model carries no [INFLOWS] line, rain gauge or dry-weather flow: the two coupling nodes (J1, J3,
declared by coupling_nodes.txt) get water from the surface only, through TRITON's exchange, so the
external lateral flow of each coupling node is the exchange alone and the .rpt flow-routing
continuity totals "External Inflow" and "External Outflow" are the sums arm (e) of case N14
recomputes. Every other file of this directory is a list file or a derived input of an arm:

  coupling_nodes.txt     J1 and J3 (swmm_coupling_nodes)
  driven_below.txt       O1 (swmm_driven_outfalls): the outfall whose cell lies below its invert
  driven_above.txt       O2
  driven_empty.txt       a list that names nothing
  coupling_empty.txt     a coupling list that names nothing
  links_c2.txt, links_c4.txt  the conduit of O1 and of O2 (swmm_diag_links, the T1d flow of the OTHER
                         outfall in the ledger arm)

The refusal arms of the suite derive their variants from these files in the run directory (a
junction, a TIMESERIES outfall, a second link, a shared cell are edits of a COPY of the model).
"""
import math
import os
import sys

NR, NC = 64, 48
DX = 5.0
PRINT_INTERVAL = 60.0
SIM_DURATION = 1800.0
MAXDEPTH = 1.0

# The cell of each node (row 0 is the northern edge); the node sits at the cell's centre, so the
# solver's placement rule (col = ceil(x / dx) - 1, row = ceil(nrows - y / dx) - 1) and the cell this
# script reads the DEM at cannot disagree on a cell boundary.
NODE_CELLS = [("J1", 32, 20), ("J2", 32, 28), ("J3", 24, 20), ("J4", 24, 28), ("O1", 32, 36), ("O2", 24, 36)]


def xy_of(i, j):
    """Cell-centre coordinates of the cell in row i (0 = north) and column j (0 = west)."""
    return (j + 0.5) * DX, (NR - i - 0.5) * DX


def cell_of(x, y):
    """(row, col) of the cell holding the point (x, y), by the solver's rule (calc_swmm_node_col/row)."""
    return int(math.ceil(NR - y / DX + 1e-16)) - 1, int(math.ceil(x / DX + 1e-16)) - 1


# (name, x, y) of the four junctions and two outfalls, all inside the DEM.
JUNCTIONS = [(n,) + xy_of(i, j) for n, i, j in NODE_CELLS if n.startswith("J")]
OUTFALLS = [(n,) + xy_of(i, j) for n, i, j in NODE_CELLS if n.startswith("O")]
# O1: the DEM at its cell is lowered to Z_O1 by a bowl of radius BOWL_R cells; its invert lies above
# that. O2: the cell keeps the plane's elevation, above its invert.
Z_O1 = 0.45
INVERT_O1 = 0.75
INVERT_O2 = 0.6873
BOWL_R = 3.5
BOWL_WATER_LEVEL = 1.25       # the bowl starts full to this elevation (m)
POND_J = 0.30                 # initial depth (m) of the ponds around the two coupling nodes
SOURCE_CELL = (32, 30)       # the surface inflow source: west of the bowl, upslope of it
POND_O2 = 0.20                # initial depth (m) of the pond around O2's cell


def plane(i, j):
    """The sloping plane of test/coupled_case/: down to the east and to the north."""
    return 1.0 + 1.5 * (1.0 - j / (NC - 1.0)) + 1.0 * (i / (NR - 1.0))


O1_CELL = cell_of(*OUTFALLS[0][1:])
O2_CELL = cell_of(*OUTFALLS[1][1:])


def ground(i, j):
    """Elevation (m) of the cell in row i and column j: the plane, with the bowl around O1's cell."""
    z = plane(i, j)
    r = math.hypot(i - O1_CELL[0], j - O1_CELL[1])
    if r < BOWL_R:
        z -= (plane(*O1_CELL) - Z_O1) * (1.0 - r / BOWL_R) ** 0.5
    return z


def dem_text():
    lines = ["ncols         %d" % NC, "nrows         %d" % NR, "xllcorner     0.0", "yllcorner     0.0",
             "cellsize      %.1f" % DX, "NODATA_value  -9999"]
    for i in range(NR):
        lines.append(" ".join("%.4f" % ground(i, j) for j in range(NC)))
    return "\n".join(lines) + "\n"


def init_depth_text():
    """Initial depth grid: no header, one row of NC values per line (the format of *.inith)."""
    jc = [cell_of(x, y) for _, x, y in JUNCTIONS if _ in ("J1", "J3")]
    rows = []
    for i in range(NR):
        vals = []
        for j in range(NC):
            h = 0.0
            if any(abs(i - ci) <= 4 and abs(j - cj) <= 4 for ci, cj in jc):
                h = POND_J
            if abs(i - O2_CELL[0]) <= 2 and abs(j - O2_CELL[1]) <= 2:
                h = POND_O2
            if math.hypot(i - O1_CELL[0], j - O1_CELL[1]) < BOWL_R:
                h = max(0.0, BOWL_WATER_LEVEL - round(ground(i, j), 4))
            vals.append("%.4f" % h)
        rows.append(" ".join(vals))
    return "\n".join(rows) + "\n"


def inp_text():
    node_lines, coords = [], []
    zr = {}
    for name, x, y in JUNCTIONS:
        i, j = cell_of(x, y)
        zr[name] = round(ground(i, j), 4)
        node_lines.append("%-7s %-11.4f %-9.1f 0          0               0" % (name, zr[name] - MAXDEPTH, MAXDEPTH))
        coords.append("%-7s %-6.1f %-6.1f" % (name, x, y))
    out_lines = []
    for (name, x, y), inv in zip(OUTFALLS, (INVERT_O1, INVERT_O2)):
        # FIXED at the invert: with the key absent the outfall holds a constant stage at its invert,
        # which is the baseline every arm of N14 is compared with
        out_lines.append("%-7s %-11.4f FIXED   %.4f" % (name, inv, inv))
        coords.append("%-7s %-6.1f %-6.1f" % (name, x, y))
    return """[TITLE]
Fixed outfall case of the diagnostics neutrality suite (see make_case.py)

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

[JUNCTIONS]
;;Name  InvertElev  MaxDepth  InitDepth  SurchargeDepth  PondedArea
%s

[OUTFALLS]
;;Name  InvertElev  Type  Stage
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

[REPORT]
INPUT      NO
CONTROLS   NO
NODES ALL
LINKS ALL

[COORDINATES]
;;Node  X      Y
%s
""" % ("\n".join(node_lines), "\n".join(out_lines), "\n".join(coords))


def hydrograph_text():
    """Surface inflow source, time in hours and discharge in m3/s."""
    return "% Hydrograph\n% Time(hr) Discharge(cms)\n0,0.0\n0.1,0.8\n0.4,0.8\n0.5,0.0\n1.0,0.0\n"


def source_loc_text():
    x, y = xy_of(*SOURCE_CELL)
    return "%%X-Location,Y-Location\n%.1f,%.1f\n" % (x, y)


CFG = """#---------------------------------------------------------------------------
# Fixed outfall case of the diagnostics neutrality suite (see make_case.py).
# Every path is relative to the working directory the suite runs in.
#---------------------------------------------------------------------------
dem_filename="elevation.dem"
inp_filename="swmm/hydraulics.inp"
swmm_coupling_nodes="coupling_nodes.txt"
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
num_extbc=0
extbc_dir=""
extbc_file=""
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
    os.makedirs(os.path.join(d, "swmm"), exist_ok=True)
    files = {
        "elevation.dem": dem_text(), "init_h.dem": init_depth_text(),
        "inflow.hyg": hydrograph_text(), "source.src": source_loc_text(),
        "swmm/hydraulics.inp": inp_text(),
        "coupling_nodes.txt": "; coupling nodes (swmm_coupling_nodes)\nJ1\nJ3\n",
        "driven_below.txt": "; driven outfalls (swmm_driven_outfalls): the outfall whose cell lies below its invert\nO1\n",
        "driven_above.txt": "; driven outfalls: the outfall whose cell lies above its invert\nO2\n",
        "driven_empty.txt": "; a driven-outfall list that names nothing\n",
        "coupling_empty.txt": "; a coupling-node list that names nothing\n",
        "links_c2.txt": "; T1d links (swmm_diag_links): O1's one connected conduit\nC2\n",
        "links_c4.txt": "; T1d links (swmm_diag_links): O2's one connected conduit\nC4\n",
        "TRITONSWMM.cfg": CFG.replace("@SIM@", "%d" % SIM_DURATION).replace("@PRINT@", "%d" % PRINT_INTERVAL),
    }
    for name, text in files.items():
        with open(os.path.join(d, name), "w", newline="\n") as f:
            f.write(text)
        print("wrote", name, len(text), "bytes")


if __name__ == "__main__":
    main()
