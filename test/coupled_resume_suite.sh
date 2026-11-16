#!/usr/bin/env bash
# Coupled TRITON-SWMM resume/bit-exactness suite.
#
# Drives the integrated coupled solver over the reference benchmark case in the
# configurations the resume machinery can reach, and byte-compares every resumed
# artifact against the uninterrupted run: the bin/ rasters (H/QX/QY/MH/GR_*),
# the TRITON->SWMM exchange side-file and SWMM's binary .out.
#
# Usage:  coupled_resume_suite.sh <path-to-triton.exe> [suite-dir]
#
# Requirements: the case is self-contained (DEM, initial depths, hydrograph, external
# boundary, SWMM model and its dats/ file are copied in by the script from the repository's
# test/coupled_case/ directory, the fixed coupled case also used by the diagnostics
# neutrality suite; see test/coupled_case/make_case.py), the
# binary must live two levels below the suite root (TRITON derives project_dir
# from argv[0]); the script arranges that itself via a build/ symlink.
#
# Checkpoint numbers are derived from the case's cfg: NCP = sim_duration / print_interval
# checkpoints, MID = NCP / 4 (the mid-run resume), 2 x MID (the chain's second hop) and NCP
# (the last checkpoint); the first checkpoint is 1.
#
# Matrix (see B4B_VERIFICATION.md for the full discussion):
#   A. serial, np1, SEQ output, live SWMM:
#      A1 clean run                      (snapshot keep-2 default)
#      A2 resume@MID replay forced       (swmm_snapshot_disable=1)
#      A3 resume@MID snapshot route      (keep_all clean leg)
#      A4 resume@1   snapshot route      (first-checkpoint edge)
#      A5 resume@NCP snapshot route      (last-checkpoint edge)
#      A6 multi-resume chain             (resume@MID then resume its cp 2xMID)
#      A7 resume@MID into a finished keep-2 output -> replay, retention-collision
#   B. MPI np8, PAR output: clean vs resume@MID snapshot + replay
#   C. OpenMP binary (1 rank): clean vs resume@MID snapshot
#   D. np16 deadlock regression (rank 0 strip holds no manhole): completes,
#      writes hydraulics.rpt, resumes
#   E. degraded SWMM (no dats/ -> engine in error state, replay-only):
#      clean rasters byte-identical to the pre-integration baseline when the
#      baseline dir is provided via $BASELINE
#   F. failure semantics: missing side-file, V1-magic side-file, corrupt
#      snapshot -- each must abort with its specific diagnostic
#   H. diagnostics captures on: clean vs resume@MID, snapshot and replay route, np1 and np8; every
#      capture file of the resumed output equals the clean run's byte for byte
#   G. performance.txt: 17 columns in the documented order, SWMM_OTHER >= 0 on
#      every row, SWMM_SOLVE + SWMM_LOG <= SWMM_STEP, DIAG == 0 with every
#      capture off, SWMM_SOLVE > 0 on rank 0, and the per-rank identity
#      Compute+MPI+IO+SWMM+Resize+DIAG+Other == Simulation as a file-integrity
#      check (it holds by the construction of Other whatever DIAG covers, so it
#      detects a malformed or mis-ordered file and does NOT detect capture work
#      charged to the wrong category -- the neutrality suite's differential does)
set -u

BIN_ABS=$(readlink -f "$1")
ROOT=$(readlink -f "${2:-$(dirname "$0")/coupled_resume_suite_run}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC="$REPO/test/coupled_case"
[ -f "$SRC/TRITONSWMM.cfg" ] || { echo "no case at $SRC"; exit 2; }
SIM=$(sed -n 's/^sim_duration=//p' "$SRC/TRITONSWMM.cfg"); PRT=$(sed -n 's/^print_interval=//p' "$SRC/TRITONSWMM.cfg")
NCP=$(( ${SIM%.*} / ${PRT%.*} )); MID=$(( NCP / 4 )); MID2=$(( 2 * MID ))
[ $MID -ge 1 ] || { echo "case has too few checkpoints ($NCP)"; exit 2; }

PASS=0; FAIL=0; FAILED_CASES=""

say()  { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); say "PASS  $*"; }
bad()  { FAIL=$((FAIL+1)); FAILED_CASES="$FAILED_CASES
  FAIL  $*"; say "FAIL  $*"; }

# run_case <name> <launcher...> -- runs triton in <root>/<name> with a local
# build/ symlink so project_dir resolves to the case dir
run_case() {
  local name=$1; shift
  local dir="$ROOT/$name"
  mkdir -p "$dir/build"
  for f in elevation.dem init_h.dem inflow.hyg source.src west.extbc west_level.txt; do
    cp -r "$SRC/$f" "$dir/"
  done
  cp -r "$SRC/swmm" "$dir/swmm"                                     # live SWMM: swmm/dats/ holds its inflow file
  ln -sf "$BIN_ABS" "$dir/build/triton.exe"
  ( cd "$dir" && "$@" > run.log 2>&1 )
  return $?
}

# fresh cfg: <dest> <outname> [extra keys...]
make_cfg() {
  local dest=$1 out=$2; shift 2
  sed -e "s|^output_folder=.*|output_folder=\"$out\"|" "$SRC/TRITONSWMM.cfg" > "$dest"
  for k in "$@"; do echo "$k" >> "$dest"; done
}

# resume <case-dir> <outname> <cpid>: clone output, point cfg at clone, resume
# RESUME_LAUNCHER: set before calling resume() for MPI cases (e.g.
# "mpirun --allow-run-as-root --oversubscribe -np 8"); a resume must run with
# the SAME rank count as the leg that wrote the checkpoints.
RESUME_LAUNCHER=""
resume() {
  local dir=$1 out=$2 cpid=$3
  rm -rf "$dir/${out}_res$cpid"
  cp -r "$dir/$out" "$dir/${out}_res$cpid"
  sed -i "s|^output_folder=.*|output_folder=\"${out}_res$cpid\"|" "$dir/${out}_res$cpid/cfg/config_$cpid.cfg"
  ( cd "$dir" && $RESUME_LAUNCHER ./build/triton.exe "${out}_res$cpid/cfg/config_$cpid.cfg" > "res$cpid.log" 2>&1 )
}

# compare_caps <clean-out> <resumed-out> <log>: every capture file the clean run wrote is byte-identical
# in the resumed output, and the clean run wrote at least one (a vacuous comparison fails)
CAP_FILES="swmm_step_diag.bin swmm_node_path3.bin swmm_node_path3_step.bin swmm_link_flow_diag.bin swmm_node_diag.nodes swmm_link_flow_diag.links edge_flux.bin dt_limiter.bin wet_cells.csv domain_balance.csv"
compare_caps() {
  local a="$ROOT/$1" b="$ROOT/$2" n=0 rc=0 f
  for f in $CAP_FILES; do
    [ -f "$a/$f" ] || continue
    n=$((n+1))
    cmp -s "$a/$f" "$b/$f" || { bad "$1 vs $2: capture file $f differs"; rc=1; }
  done
  [ $n -ge 10 ] || { bad "$1: only $n of the 10 capture files exist (a vacuous comparison)"; rc=1; }
  [ $rc -eq 0 ] && ok "$2 reproduces $1's $n capture files byte-for-byte"
  return $rc
}

# compare <clean-out> <resumed-out> <resumed-log>: bin/, side-file, .out.
# The log must show "Simulation ends" -- without that, a resume that crashed
# at startup would compare the still-clean clone against the clean run and
# pass vacuously.
compare() {
  local a="$ROOT/$1" b="$ROOT/$2" rlog="$3" rc=0
  if ! grep -aq "Simulation ends" "$rlog" 2>/dev/null; then
    bad "$2: resumed run did not complete (see $rlog); comparison skipped"
    return 1
  fi
  diff -r "$a/bin" "$b/bin" > /dev/null 2>&1 || { bad "$1 vs $2: bin/ rasters differ"; rc=1; }
  cmp -s "$a/swmm/hydraulics_exchange_replay.bin" "$b/swmm/hydraulics_exchange_replay.bin" \
    || { bad "$1 vs $2: exchange side-file differs"; rc=1; }
  cmp -s "$a/swmm/hydraulics.out" "$b/swmm/hydraulics.out" \
    || { bad "$1 vs $2: SWMM hydraulics.out differs"; rc=1; }
  [ $rc -eq 0 ] && ok "$2 reproduces $1 byte-for-byte (bin/ + side-file + .out)"
  return $rc
}

mkdir -p "$ROOT"
say "=== coupled resume suite: binary=$BIN_ABS root=$ROOT ==="

# ---------- A. serial np1, SEQ ----------
RESUME_LAUNCHER=""
say "--- A: serial, 1 rank, SEQ output, live SWMM ---"
D="$ROOT/serial"; run_case serial true
make_cfg "$D/case_ka.cfg"  out_ka  swmm_snapshot_keep_all=1
make_cfg "$D/case_dis.cfg" out_dis swmm_snapshot_disable=1
make_cfg "$D/case_k2.cfg"  out_k2

( cd "$D" && ./build/triton.exe case_ka.cfg  > ka.log  2>&1 ) || bad "A3 clean keep_all leg"
( cd "$D" && ./build/triton.exe case_dis.cfg > dis.log 2>&1 ) || bad "A2 clean disable leg"
( cd "$D" && ./build/triton.exe case_k2.cfg  > k2.log  2>&1 ) || bad "A1/A7 clean keep-2 leg"

# A2: replay forced
resume "$D" out_dis $MID
grep -aq "path=replay reason=absent" "$D/res$MID.log" \
  && ok "A2 resume@$MID disable: took replay route (reason=absent)" \
  || bad "A2 resume@$MID disable: expected replay route"
compare serial/out_dis serial/out_dis_res$MID "$ROOT/serial/res$MID.log"

# A3: snapshot route, mid-run
resume "$D" out_ka $MID
grep -aq "path=snapshot" "$D/res$MID.log" \
  && ok "A3 resume@$MID keep_all: took snapshot route" \
  || bad "A3 resume@$MID keep_all: expected snapshot route"
compare serial/out_ka serial/out_ka_res$MID "$ROOT/serial/res$MID.log"

# A4: first checkpoint
resume "$D" out_ka 1
grep -aq "path=snapshot" "$D/res1.log" \
  && ok "A4 resume@1: snapshot route from first checkpoint" \
  || bad "A4 resume@1: expected snapshot route"
compare serial/out_ka serial/out_ka_res1 "$ROOT/serial/res1.log"

# A5: last checkpoint
resume "$D" out_ka $NCP
compare serial/out_ka serial/out_ka_res$NCP "$ROOT/serial/res$NCP.log"

# A6: multi-resume chain (resume@MID, then resume ITS checkpoint 2 x MID)
rm -rf "$D/out_ka_chain" && cp -r "$D/out_ka_res$MID" "$D/out_ka_chain"
sed -i "s|^output_folder=.*|output_folder=\"out_ka_chain\"|" "$D/out_ka_chain/cfg/config_$MID2.cfg"
( cd "$D" && ./build/triton.exe out_ka_chain/cfg/config_$MID2.cfg > chain.log 2>&1 ) \
  && compare serial/out_ka serial/out_ka_chain "$ROOT/serial/chain.log" && ok "A6 multi-resume chain ($MID -> $MID2) reproduces clean" \
  || bad "A6 multi-resume chain"

# A7: keep-2 collision -> replay with classified reason
resume "$D" out_k2 $MID
grep -aq "path=replay reason=retention-collision" "$D/res$MID.log" \
  && ok "A7 keep-2 collision: replay route with reason=retention-collision" \
  || bad "A7 keep-2 collision: expected classified replay route"
compare serial/out_k2 serial/out_k2_res$MID "$ROOT/serial/res$MID.log"

# ---------- B. MPI np8 ----------
RESUME_LAUNCHER=""
say "--- B: MPI, 8 ranks, SEQ and PAR output ---"
D="$ROOT/mpi8"; run_case mpi8 true
make_cfg "$D/case_ka.cfg" out_ka8 swmm_snapshot_keep_all=1
( cd "$D" && mpirun --allow-run-as-root --oversubscribe -np 8 ./build/triton.exe case_ka.cfg > ka.log 2>&1 ) \
  || bad "B clean leg"
RESUME_LAUNCHER="mpirun --allow-run-as-root --oversubscribe -np 8"
resume "$D" out_ka8 $MID
grep -aq "path=snapshot" "$D/res$MID.log" && ok "B resume@$MID np8 SEQ: snapshot route" || bad "B route"
compare mpi8/out_ka8 mpi8/out_ka8_res$MID "$ROOT/mpi8/res$MID.log"

# B2: PAR checkpoints + resume at np>1 (resumes from per-rank rasters).
make_cfg "$D/case_par.cfg" out_par8 swmm_snapshot_keep_all=1
sed -i "s|^output_option=.*|output_option=PAR|" "$D/case_par.cfg"
( cd "$D" && mpirun --allow-run-as-root --oversubscribe -np 8 ./build/triton.exe case_par.cfg > par.log 2>&1 )
RESUME_LAUNCHER="mpirun --allow-run-as-root --oversubscribe -np 8"
resume "$D" out_par8 $MID
grep -aq "path=snapshot" "$D/res$MID.log" && ok "B2 resume@$MID np8 PAR: snapshot route" || bad "B2 route"
compare mpi8/out_par8 mpi8/out_par8_res$MID "$ROOT/mpi8/res$MID.log"

# ---------- C. OMP binary, 1 rank ----------
if ldd "$BIN_ABS" 2>/dev/null | grep -q "gomp"; then
  say "--- C: OpenMP binary, 1 rank ---"
  D="$ROOT/omp1"; run_case omp1 true
  make_cfg "$D/case_ka.cfg" out_ka_omp swmm_snapshot_keep_all=1
  ( cd "$D" && OMP_NUM_THREADS=4 ./build/triton.exe case_ka.cfg > ka.log 2>&1 ) || bad "C clean leg"
  resume "$D" out_ka_omp $MID
  compare omp1/out_ka_omp omp1/out_ka_omp_res$MID "$ROOT/omp1/res$MID.log"
else
  say "--- C: skipped (binary is not an OpenMP build) ---"
fi

# ---------- D. np16 deadlock regression ----------
RESUME_LAUNCHER=""
say "--- D: np16, rank 0 strip holds no manhole ---"
D="$ROOT/np16"; run_case np16 true
make_cfg "$D/case.cfg" out16
sed -i "s|^sim_duration=.*|sim_duration=$(( 10 * ${PRT%.*} ))|" "$D/case.cfg"
( cd "$D" && timeout 300 mpirun --allow-run-as-root --oversubscribe -np 16 ./build/triton.exe case.cfg > np16.log 2>&1 )
rc=$?
if [ $rc -eq 0 ] && grep -aq "Simulation ends" "$D/np16.log" && [ -s "$ROOT/np16/out16/swmm/hydraulics.rpt" ]; then
  ok "D np16 completes and writes hydraulics.rpt (pre-integration: deadlock)"
else
  bad "D np16 did not complete cleanly (rc=$rc)"
fi

# ---------- E. degraded SWMM (missing dats/) startup refusal ----------
RESUME_LAUNCHER=""
say "--- E: degraded SWMM (no dats/), startup refusal on SWMM error 361 ---"
D="$ROOT/degraded"; mkdir -p "$D/build"
for f in elevation.dem init_h.dem inflow.hyg source.src west.extbc west_level.txt; do cp -r "$SRC/$f" "$D/"; done
cp -r "$SRC/swmm" "$D/swmm"
rm -rf "$D/swmm/dats"                                # deliberately NO swmm/dats: the model's inflow file is missing
ln -sf "$BIN_ABS" "$D/build/triton.exe"
make_cfg "$D/case.cfg" out_degraded
( cd "$D" && ./build/triton.exe case.cfg > run.log 2>&1 )
rc=$?
if [ $rc -ne 0 ] && grep -aq "swmm_open returned error code" "$D/run.log" && grep -aq "ERROR 361" "$D/run.log" && [ -s "$ROOT/degraded/out_degraded/swmm/hydraulics.rpt" ]; then
  ok "E degraded SWMM (no dats/): refuses at swmm_open and echoes ERROR 361 from hydraulics.rpt"
else
  bad "E degraded SWMM (no dats/) did not refuse cleanly at swmm_open (rc=$rc)"
fi

# ---------- F. failure semantics ----------
RESUME_LAUNCHER=""
say "--- F: failure semantics ---"
D="$ROOT/serial"
# F1: missing side-file
rm -rf "$D/out_fail_ms" && cp -r "$D/out_k2" "$D/out_fail_ms"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_ms\"|" "$D/out_fail_ms/cfg/config_$MID.cfg"
rm -f "$D/out_fail_ms/swmm/hydraulics_exchange_replay.bin"
( cd "$D" && ./build/triton.exe out_fail_ms/cfg/config_$MID.cfg > f1.log 2>&1 )
grep -aq "requires the exchange-replay side-file" "$D/f1.log" \
  && ok "F1 missing side-file: fatal with guidance" || bad "F1 missing side-file"
# F2: V1-magic side-file (old format)
rm -rf "$D/out_fail_v1" && cp -r "$D/out_k2" "$D/out_fail_v1"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_v1\"|" "$D/out_fail_v1/cfg/config_$MID.cfg"
python3 - "$D/out_fail_v1/swmm/hydraulics_exchange_replay.bin" <<'EOF'
import struct, sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
b[0:4] = struct.pack('<i', 0x53574D4D)          # ASCII "SWMM": the V1 magic
b[4:8] = struct.pack('<i', 13)                  # node count still first
open(p,'wb').write(b)
EOF
( cd "$D" && ./build/triton.exe out_fail_v1/cfg/config_$MID.cfg > f2.log 2>&1 )
grep -aq "pre-versioning (V1) magic" "$D/f2.log" \
  && ok "F2 V1 side-file: refused with the legacy diagnostic" || bad "F2 V1 side-file"
# F3: corrupt snapshot (present but truncated -> Unreadable -> abort)
rm -rf "$D/out_fail_snap" && cp -r "$D/out_ka" "$D/out_fail_snap"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_snap\"|" "$D/out_fail_snap/cfg/config_$MID.cfg"
python3 - "$D/out_fail_snap/swmm/hydraulics_state_cp$MID.snapshot" <<'EOF'
import sys
p = sys.argv[1]
b = open(p,'rb').read()
open(p,'wb').write(b[:len(b)//2])               # truncate: header short
EOF
( cd "$D" && ./build/triton.exe out_fail_snap/cfg/config_$MID.cfg > f3.log 2>&1 )
grep -aq "UNUSABLE" "$D/f3.log" && grep -aq "Coupled resume aborted" "$D/f3.log" \
  && ok "F3 corrupt snapshot: aborted (UNREADABLE), no fallback" || bad "F3 corrupt snapshot"

# ---------- G. performance.txt shape ----------
say "--- G: performance.txt accounting ---"
P="$ROOT/serial/out_ka/performance.txt"
# The exact header string: the column order is part of the interface hhemt's
# readers were told (they select by name, but a reordered file is a different file).
G_HEADER='%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER, SWMM_SOLVE, SWMM_LOG, DIAG, Other, Simulation, Init, Total'
if [ -s "$P" ]; then
  [ "$(head -1 "$P" | tr -d '\r')" = "$G_HEADER" ] \
    && ok "G performance header: 17 columns, SWMM_SOLVE and SWMM_LOG after SWMM_OTHER, DIAG before Other" \
    || bad "G performance header differs from the documented 17-column order"
  python3 - "$P" <<'EOF' && ok "G SWMM_OTHER >= 0 on every row (residual sign)" || bad "G SWMM_OTHER negative"
import csv, math, sys
rows = [[c.strip() for c in r] for r in csv.reader(open(sys.argv[1]))]
hdr = rows[0][1:]   # drop '%Rank' so header indices line up with the numeric cells

def half_unit(x):
    """Largest rounding error of x printed with setprecision(4): half a unit in the fourth
    significant digit."""
    return 0.0 if x == 0.0 else 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - 3)

i_other = hdr.index('SWMM_OTHER'); i_x = hdr.index('SWMM_XFER'); i_m = hdr.index('SWMM_MPI')
i_s = hdr.index('SWMM_STEP'); i_p = hdr.index('SWMM')
for r in rows[1:]:
    v = [float(x) for x in r[1:]]
    assert v[i_other] >= 0.0, r
    # the file prints 4 significant digits, so the identity holds only to the sum of the five
    # printed values' rounding errors (a fixed 0.05 s fails a correct file once a time exceeds 100 s)
    tol = sum(half_unit(v[i]) for i in (i_x, i_m, i_s, i_other, i_p)) + 1e-9
    assert abs(v[i_x]+v[i_m]+v[i_s]+v[i_other] - v[i_p]) <= tol, r
EOF
  python3 - "$P" <<'EOF' && ok "G SWMM_SOLVE + SWMM_LOG <= SWMM_STEP; SWMM_SOLVE > 0 on rank 0; DIAG == 0 with captures off" || bad "G SWMM_SOLVE/SWMM_LOG/DIAG accounting"
import csv, math, sys
rows = [[c.strip() for c in r] for r in csv.reader(open(sys.argv[1]))]
hdr = rows[0][1:]
col = {n: hdr.index(n) for n in ('SWMM_STEP', 'SWMM_SOLVE', 'SWMM_LOG', 'DIAG')}

def half_unit(x):
    """Largest rounding error of x printed with setprecision(4): half a unit in
    the fourth significant digit."""
    return 0.0 if x == 0.0 else 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - 3)

for r in rows[1:]:
    v = [float(x) for x in r[1:]]
    # Children of SWMM_STEP are bracketed inside it and never open together.
    tol = half_unit(v[col['SWMM_STEP']]) + half_unit(v[col['SWMM_SOLVE']]) + half_unit(v[col['SWMM_LOG']]) + 1e-9
    assert v[col['SWMM_SOLVE']] + v[col['SWMM_LOG']] <= v[col['SWMM_STEP']] + tol, r
    # No capture is on in this suite, so the DIAG bracket is never entered.
    assert v[col['DIAG']] == 0.0, r
# Must-be-nonzero control for the two zeros-by-assumption above: the SWMM solve runs
# on rank 0 of this live-SWMM case, so a SWMM_SOLVE bracket that never runs reads 0.
rank0 = [float(x) for x in rows[1][1:]]
assert rows[1][0] == '0' and rank0[col['SWMM_SOLVE']] > 0.0, rows[1]
EOF
  python3 - "$P" <<'EOF' && ok "G per-rank identity Compute+MPI+IO+SWMM+Resize+DIAG+Other == Simulation (file integrity)" || bad "G per-rank identity does not close"
import csv, math, sys
rows = [[c.strip() for c in r] for r in csv.reader(open(sys.argv[1]))]
hdr = rows[0][1:]
terms = ['Compute', 'MPI', 'IO', 'SWMM', 'Resize', 'DIAG', 'Other']
idx = [hdr.index(n) for n in terms]
i_sim = hdr.index('Simulation')

def half_unit(x):
    return 0.0 if x == 0.0 else 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - 3)

for r in rows[1:]:
    v = [float(x) for x in r[1:]]
    # Other is Simulation minus the other terms in the solver, so the identity is exact
    # before printing; the bound is the sum of the eight printed values' rounding errors.
    tol = sum(half_unit(v[i]) for i in idx) + half_unit(v[i_sim]) + 1e-9
    assert abs(sum(v[i] for i in idx) - v[i_sim]) <= tol, r
EOF
else
  bad "G performance.txt not found"
fi

# ---------- H. diagnostics captures on: clean vs resumed, capture files byte for byte ----------
RESUME_LAUNCHER=""
say "--- H: all captures on, resume@$MID, snapshot and replay route, np1 and np8 ---"
H_KEYS="swmm_diag_steps=1 swmm_diag_node_path3_per_step=1 swmm_diag_links=ALL diag_domain_balance=1 diag_wet_cells=1 diag_dt_limiter=1 swmm_snapshot_keep_all=1"
D="$ROOT/caps1"; run_case caps1 true
make_cfg "$D/case_h.cfg" out_h $H_KEYS
( cd "$D" && ./build/triton.exe case_h.cfg > h.log 2>&1 ) || bad "H clean leg np1 (captures on)"
resume "$D" out_h $MID
grep -aq "path=snapshot" "$D/res$MID.log" && ok "H1 captures on, np1: resume@$MID took the snapshot route" || bad "H1 route"
compare caps1/out_h caps1/out_h_res$MID "$ROOT/caps1/res$MID.log" && compare_caps caps1/out_h caps1/out_h_res$MID
rm -rf "$D/out_h_res$MID"; cp -r "$D/out_h" "$D/out_h_res$MID"
sed -i "s|^output_folder=.*|output_folder=\"out_h_res$MID\"|" "$D/out_h_res$MID/cfg/config_$MID.cfg"
echo "swmm_snapshot_disable=1" >> "$D/out_h_res$MID/cfg/config_$MID.cfg"
( cd "$D" && ./build/triton.exe out_h_res$MID/cfg/config_$MID.cfg > res$MID.log 2>&1 )
grep -aq "path=replay" "$D/res$MID.log" && ok "H2 captures on, np1: resume@$MID took the replay route" || bad "H2 route"
compare caps1/out_h caps1/out_h_res$MID "$ROOT/caps1/res$MID.log" && compare_caps caps1/out_h caps1/out_h_res$MID
D="$ROOT/caps8"; run_case caps8 true
make_cfg "$D/case_h.cfg" out_h8 $H_KEYS
( cd "$D" && mpirun --allow-run-as-root --oversubscribe -np 8 ./build/triton.exe case_h.cfg > h.log 2>&1 ) || bad "H clean leg np8 (captures on)"
RESUME_LAUNCHER="mpirun --allow-run-as-root --oversubscribe -np 8"
resume "$D" out_h8 $MID
grep -aq "path=snapshot" "$D/res$MID.log" && ok "H3 captures on, np8: resume@$MID took the snapshot route" || bad "H3 route"
compare caps8/out_h8 caps8/out_h8_res$MID "$ROOT/caps8/res$MID.log" && compare_caps caps8/out_h8 caps8/out_h8_res$MID

say ""
say "=== suite result: $PASS passed, $FAIL failed ==="
[ -n "$FAILED_CASES" ] && say "$FAILED_CASES"
[ $FAIL -eq 0 ]
