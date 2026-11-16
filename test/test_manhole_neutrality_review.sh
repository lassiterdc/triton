#!/bin/bash
# Reviewer test for chunk 3c.1 (case N13 of test/diagnostics_neutrality_suite.sh): it drives the
# N13 helper and N13 refusal/must-differ/constants functions of the suite under test with stubs and
# synthetic side-files, so a verdict of the suite's own arm logic is read on an input whose right
# answer is known. Nothing here compiles or runs the solver; run it on Rivanna all the same.
#
#   usage: SUITE=path/to/diagnostics_neutrality_suite.sh bash test_manhole_neutrality_review.sh
#
# Each check prints  CHECK id  EXPECT x  GOT y  OK|WRONG  and the exit status is the WRONG count.
#
# Per check (B3 fields i, iii, iv; field ii, the shape row, is in the reviewer's round file):
#  R1..R12  class: every refusal-arm outcome (exit status, log text, files, processes) the arm can
#           meet. Why: a refusal arm that reads PASS on an outcome that is not a refusal makes the
#           row's "must exit non-zero before the first step" claim unfalsifiable. Killed by: an arm
#           that drops its own non-zero-exit test (R2). A second correct implementation: one that
#           tests `rc -ne 0` first, or reads the exit status of the launcher through `wait`.
#  H1..H9   class: first-differing-record readings over side-files of 2 coupling columns. Why: the
#           row says N*'s column must be the ONLY column of the first differing record.
#           Killed by: a detector that tests `expect in cols`. A second correct implementation:
#           compare the set of differing column indices with {expect}.
#  S1..S3   the arms the suite runs on the real case, replayed on synthetic records (S1 is a check);
#           S2 and S3 run two mutant detectors through the same arms and only INFORM (see F2).
#  C1..C4   class: the equal/missing/malformed constants of the case cfg.
#  D1..D4   class: the compare exit status and output n13_must_differ reads.
set -u
SUITE=${SUITE:?set SUITE to the suite file under test}
PYTHON=${PYTHON:-$(command -v python3.11 || command -v python3)}
W=$(mktemp -d "${TMPDIR:-/tmp}/n13rev.XXXXXX"); W=$(readlink -f "$W")
BAD=0; BUILD_ONLY=0; WORK=$W; BUILD_RECORD=$W/build_record.txt; LOGS=$W/logs; RESULTS=$W/results.txt
mkdir -p "$LOGS" "$W/run" "$W/stubs"; : > "$RESULTS"; : > "$BUILD_RECORD"

grab() { # grab FUNCTION : the text of a top-level shell function of the suite
  awk -v f="$1" '$0 ~ "^"f"\\(\\) \\{" {on=1} on {print} on && /^\}$/ {exit}' "$SUITE"
}
for fn in n13_write_helper n13_table n13_double n13_refusal n13_must_differ; do
  body=$(grab "$fn"); [ -n "$body" ] || { echo "cannot find $fn in $SUITE"; exit 3; }
  eval "$body"
done
rec() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RESULTS"; }
verdict_of() { awk -F'\t' -v id="$1" '$1==id {s=$2} END {print s==""?"NONE":s}' "$RESULTS"; }
check() { # check ID EXPECT GOT
  if [ "$2" = "$3" ]; then echo "CHECK $1  EXPECT $2  GOT $3  OK"; else echo "CHECK $1  EXPECT $2  GOT $3  WRONG"; BAD=$((BAD+1)); fi
}

# ---------------------------------------------------------------- refusal arm, driven by stub executables
cat > "$W/stubs/triton_stub.sh" <<'EOF'
#!/bin/bash
tbl=$(sed -n 's/^swmm_manhole_table=//p' arm.cfg)
case "$STUBMODE" in
  ok)        echo "[ERROR] Manhole table $tbl line 1: refused" >&2; exit 1;;
  exit0)     echo "[ERROR] Manhole table $tbl line 1: refused" >&2; exit 0;;
  nopath)    echo "[ERROR] a manhole table was refused" >&2; exit 1;;
  started)   echo "[OK] Simulation starts"; echo "[ERROR] Manhole table $tbl line 1: refused" >&2; exit 1;;
  raster)    mkdir -p out/bin; : > out/bin/H_01_00.out; echo "[ERROR] Manhole table $tbl line 1: refused" >&2; exit 1;;
  hang)      echo "[ERROR] Manhole table $tbl line 1: refused" >&2; sleep 30; exit 1;;
  leftover)  ( while :; do sleep 1; done ) & echo "[ERROR] Manhole table $tbl line 1: refused" >&2; exit 1;;
  completes) echo "[OK] Simulation starts"; echo "[OK] Simulation ends"; exit 0;;
  wrongline) echo "[ERROR] Manhole table $tbl line 2: refused" >&2; exit 1;;
  noline)    echo "[ERROR] Manhole table $tbl: refused" >&2; exit 1;;
esac
EOF
chmod +x "$W/stubs/triton_stub.sh"
rundir() { local d="$W/run/$1"; rm -rf "$d"; mkdir -p "$d/build"; cp "$W/stubs/triton_stub.sh" "$d/build/triton.exe"; echo "$d"; }
mkcfg() { local d=$1; : > "$d/arm.cfg"; shift 3; local kv; for kv in "$@"; do echo "$kv" >> "$d/arm.cfg"; done; }
launcher() { echo ""; }
timeout() { shift; command timeout 3 "$@"; }
refusal() { # refusal ID MODE LINE
  STUBMODE=$2 n13_refusal "$1" "$W/tbl_$1.txt" "$3" "stub $2" > /dev/null 2>&1
  verdict_of "$1"
}
check R1_refusal_with_file_and_line      PASS  "$(refusal R1 ok 1)"
check R2_exit0_names_file_no_start       FAIL  "$(refusal R2 exit0 1)"
check R3_exit1_file_not_named            FAIL  "$(refusal R3 nopath 1)"
check R4_exit1_run_started               FAIL  "$(refusal R4 started 1)"
check R5_exit1_raster_written            FAIL  "$(refusal R5 raster 1)"
check R6_timeout_124                     FAIL  "$(refusal R6 hang 1)"
check R7_process_left_running            FAIL  "$(refusal R7 leftover 1)"
check R8_run_completes_not_refused       BLIND "$(refusal R8 completes 1)"
check R9_wrong_line_number               FAIL  "$(refusal R9 wrongline 1)"
check R10_no_line_wanted_line_absent_ok  PASS  "$(refusal R10 noline 0)"
check R11_line_wanted_line_absent        FAIL  "$(refusal R11 noline 1)"

# ---------------------------------------------------------------- the helper on synthetic side-files
n13_write_helper
H="$W/n13_helper.py"
cat > "$W/mk.py" <<'EOF'
import struct, sys
def write(path, recs, n):
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", 0x4F4B5832, n, 8))
        for r in recs:
            f.write(struct.pack("<%dd" % (1 + n), *r))
base = [[0.1 * k] + [0.0, 0.0] for k in range(1, 6)]
def variant(name, edits, n=2, recs=None):
    r = [list(x) for x in (recs or base)]
    for (k, j, v) in edits:
        r[k][j] = v
    write(name, r, n)
out = sys.argv[1]
write(out + "/a.bin", base, 2)
variant(out + "/same.bin", [])
variant(out + "/col0.bin", [(2, 1, 5.0)])
variant(out + "/col1.bin", [(2, 2, 5.0)])
variant(out + "/both.bin", [(2, 1, 5.0), (2, 2, 5.0)])
variant(out + "/dt.bin", [(2, 0, 9.9)])
variant(out + "/dtcol0.bin", [(2, 0, 9.9), (2, 1, 5.0)])
variant(out + "/late_other.bin", [(2, 1, 5.0), (4, 2, 7.0)])      # first record names col 0, a later one col 1
variant(out + "/early_other_late_expected.bin", [(1, 2, 5.0), (3, 1, 5.0)])  # first record names col 1
write(out + "/short.bin", base[:3], 2)
write(out + "/three.bin", [[0.0, 0.0, 0.0, 0.0]] * 5, 3)
EOF
$PYTHON -I "$W/mk.py" "$W" || { echo "cannot build synthetic side-files"; exit 3; }
fc() { $PYTHON -I "${HELPER:-$H}" firstcols "$W/a.bin" "$W/$1.bin" "$2" > /dev/null 2>&1; echo $?; }
check H1_identical_files_expect0         1 "$(fc same 0)"
check H2_expected_column_alone           0 "$(fc col0 0)"
check H3_other_column_alone              1 "$(fc col1 0)"
check H4_two_columns_in_first_record     1 "$(fc both 0)"
check H5_two_columns_expect1             1 "$(fc both 1)"
check H6_dt_alone                        1 "$(fc dt 0)"
check H7_dt_and_expected_column          1 "$(fc dtcol0 0)"
check H8_first_record_expected_later_other 0 "$(fc late_other 0)"
check H9_first_record_other_later_expected 1 "$(fc early_other_late_expected 0)"
check H10_prefix_equal_shorter_file      1 "$(fc short 0)"
$PYTHON -I "$H" firstcols "$W/a.bin" "$W/three.bin" 0 > /dev/null 2>&1; check H11_shape_mismatch_is_an_error 2 "$?"

# the arms the suite runs on the real case, as outcomes on synthetic records: (c) expects N*, (c') expects N* and must fail,
# (c') expects N' and must pass. Two mutant detectors are then run through the same three arms.
mutant() { sed "s/ok = (cols == \[expect\])/ok = ($1)/" "$H" > "$W/helper_$2.py"; echo "$W/helper_$2.py"; }
arms() { # arms HELPER -> the three outcomes of the suite's arms: c(expect N*) c'(expect N*) c'(expect N')
  local h=$1 a b c
  $PYTHON -I "$h" firstcols "$W/a.bin" "$W/col0.bin" 0 > /dev/null 2>&1; a=$?
  $PYTHON -I "$h" firstcols "$W/a.bin" "$W/col1.bin" 0 > /dev/null 2>&1; b=$?
  $PYTHON -I "$h" firstcols "$W/a.bin" "$W/col1.bin" 1 > /dev/null 2>&1; c=$?
  echo "$a$b$c"
}
check S1_real_helper_arm_outcomes        010 "$(arms "$H")"
MUT_IN=$(mutant "expect in cols" in)
MUT_ANY=$(mutant "len(cols) == 1" any)
# INFO rows, not checks: whether a mutant detector survives the three arms the suite runs on real data. A survivor is
# a finding about the arm set (prose finding F2 of the round file), not an assertion this file can make about one
# implementation of the arm set, so the rows report and never count.
info() { echo "INFO  $1  $2"; }
info S2_mutant_expect_in_cols      "$( [ "$(arms "$MUT_IN")"  = 010 ] && echo "SURVIVES all three arms (outcomes 010)" || echo "caught by the arm set" )"
info S3_mutant_len_cols_eq_1       "$( [ "$(arms "$MUT_ANY")" = 010 ] && echo "SURVIVES all three arms (outcomes 010)" || echo "caught by the arm set" )"

# ---------------------------------------------------------------- select: N*, N' and the owners from the case geometry
cat > "$W/mkcase.py" <<'EOF2'
import os, struct, sys
out, tag, coords, ncols_nodes, firsts = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
d = os.path.join(out, "case_" + tag); os.makedirs(os.path.join(d, "swmm"), exist_ok=True)
open(os.path.join(d, "elevation.dem"), "w").write("ncols 48\nnrows 64\nxllcorner 0.0\nyllcorner 0.0\ncellsize 5.0\nNODATA_value -9999\n")
inp = "[JUNCTIONS]\n;;x\n[COORDINATES]\n;;Node X Y\n" + "".join("%s %s %s\n" % tuple(c.split(",")) for c in coords.split(";")) + "[VERTICES]\n"
open(os.path.join(d, "swmm", "hydraulics.inp"), "w").write(inp)
fr = [int(x) for x in firsts.split(",")]
recs = []
for k in range(8):
    recs.append([0.1 * k] + [(1.0 if k >= fr[j] else 0.0) for j in range(ncols_nodes)])
with open(os.path.join(out, "side_" + tag + ".bin"), "wb") as f:
    f.write(struct.pack("<3i", 0x4F4B5832, ncols_nodes, 8))
    for r in recs: f.write(struct.pack("<%dd" % (1 + ncols_nodes), *r))
EOF2
sel() { # sel TAG "NAME,X,Y;..." NP NAMES...  -> the select output, one line per record kind
  local tag=$1 coords=$2 np=$3; shift 3
  $PYTHON -I "$W/mkcase.py" "$W" "$tag" "$coords" "$#" "$SELFIRST" || return 9
  $PYTHON -I "$H" select "$W/case_$tag" "$np" "$W/side_$tag.bin" "$@" 2>&1 | grep -a '^NSTAR\|^NPRIME\|^OWNERS' | tr '\n' '|'
}
SELFIRST=3,1
check X1_two_nodes_np8_owners_3_and_2       "NSTAR J1 0 3 3|NPRIME J3 1 2 not_rank0|OWNERS 2 3|" "$(sel x1 'J1,100.0,160.0;J3,100.0,200.0' 8 J1 J3)"
SELFIRST=3,1,2
check X2_a_rank0_node_is_preferred_as_Nprime "NSTAR J1 0 3 3|NPRIME J0 2 0 rank0|OWNERS 0 2 3|" "$(sel x2 'J1,100.0,160.0;J3,100.0,200.0;J0,100.0,315.0' 8 J1 J3 J0)"
SELFIRST=3,1
check X3_two_nodes_in_one_cell_have_no_Nstar "" "$(sel x3 'J1,100.0,160.0;J3,100.0,161.0' 8 J1 J3)"
SELFIRST=3,1
check X4_same_nodes_np16_owners_change      "NSTAR J1 0 7 3|NPRIME J3 1 5 not_rank0|OWNERS 5 7|" "$(sel x4 'J1,100.0,160.0;J3,100.0,200.0' 16 J1 J3)"

# ---------------------------------------------------------------- the constants-differ block
{ echo 'constants_block() {'; awk '/^  N13_DIA=\$\(sed/ {on=1} on {print} on && /rec "N13\/constants" PASS/ {exit}' "$SUITE"; echo '}'; } > "$W/constants_block.sh"
. "$W/constants_block.sh"
constants() { # constants DIA LOSS -> the verdict row
  CASE_SRC="$W/case_$1_$2"; mkdir -p "$CASE_SRC"; { [ "$1" = ABSENT ] || echo "manhole_diameter=$1"; [ "$2" = ABSENT ] || echo "manhole_loss=$2"; } > "$CASE_SRC/TRITONSWMM.cfg"
  : > "$RESULTS"; constants_block > /dev/null 2>&1; verdict_of N13/constants
}
check C1_constants_differ                PASS  "$(constants 1.2 0.1)"
check C2_constants_equal                 ERROR "$(constants 0.5 0.5)"
check C3_constants_equal_by_value        ERROR "$(constants 0.50 0.5)"
check C4_diameter_missing                ERROR "$(constants ABSENT 0.1)"

# ---------------------------------------------------------------- n13_must_differ, over the compare exit status and text
cat > "$W/stubs/cmp_stub.sh" <<'EOF'
#!/bin/bash
case "$CMPMODE" in
  sidefile) echo "DIFFERENCE: swmm/hydraulics_exchange_replay.bin: differs at byte 12"; exit 1;;
  rasters)  echo "DIFFERENCE: bin/H_01_00.out: differs at byte 3"; exit 1;;
  same)     exit 0;;
  error)    echo "cannot read"; exit 2;;
esac
EOF
chmod +x "$W/stubs/cmp_stub.sh"
CMP="$W/stubs/cmp_stub.sh"; NEUT=ignored
md() { : > "$RESULTS"; CMPMODE=$1 n13_must_differ "MD_$1" "stub" a b > /dev/null 2>&1; verdict_of "MD_$1"; }
check D1_side_file_differs               PASS  "$(md sidefile)"
check D2_only_rasters_differ             ERROR "$(md rasters)"
check D3_byte_identical                  FAIL  "$(md same)"
check D4_compare_error                   ERROR "$(md error)"

# ---------------------------------------------------------------- the table writer
N13_NODES=(J1 J3); N13_DIA=1.2; N13_LOS=0.1
n13_table "$W/t.txt" "J3=2.4,0.1"
check T1_table_rows_and_override "J1 1.2 0.1|J3 2.4 0.1" "$(paste -sd'|' "$W/t.txt")"
check T2_double_is_exact         "2.4|0.2" "$(printf '%s|%s' "$(n13_double 1.2)" "$(n13_double 0.1)")"

echo "WRONG count: $BAD"
rm -rf "$W"
exit "$BAD"
