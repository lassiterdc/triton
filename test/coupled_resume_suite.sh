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
# Requirements: the case is self-contained (DEM/Manning/swmm/strmflow/extbc are
# copied in by the script from the repository's benchmark/ directory), the
# binary must live two levels below the suite root (TRITON derives project_dir
# from argv[0]); the script arranges that itself via a build/ symlink.
#
# Matrix (see B4B_VERIFICATION.md for the full discussion):
#   A. serial, np1, SEQ output, live SWMM:
#      A1 clean 24h                      (snapshot keep-2 default)
#      A2 resume@36  replay forced       (swmm_snapshot_disable=1)
#      A3 resume@36  snapshot route      (keep_all clean leg)
#      A4 resume@1  snapshot route       (first-checkpoint edge)
#      A5 resume@144 snapshot route      (last-checkpoint edge)
#      A6 multi-resume chain             (resume@36 then resume its cp72)
#      A7 resume@36 into a finished keep-2 output -> replay, retention-collision
#   B. MPI np8, PAR output: clean vs resume@36 snapshot + replay
#   C. OpenMP binary (1 rank): clean vs resume@36 snapshot
#   D. np16 deadlock regression (rank 0 strip holds no manhole): completes,
#      writes hydraulics.rpt, resumes
#   E. degraded SWMM (no dats/ -> engine in error state, replay-only):
#      clean rasters byte-identical to the pre-integration baseline when the
#      baseline dir is provided via $BASELINE
#   F. failure semantics: missing side-file, V1-magic side-file, corrupt
#      snapshot -- each must abort with its specific diagnostic
#   G. performance.txt: 14 columns, SWMM_OTHER >= 0 on every row
set -u

BIN_ABS=$(readlink -f "$1")
ROOT=$(readlink -f "${2:-$(dirname "$0")/coupled_resume_suite_run}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC="$REPO/benchmark"

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
  for f in elevation_3.50m.dem mannings_3.50m.dem strmflow extbc cid; do
    cp -r "$SRC/$f" "$dir/" 2>/dev/null
  done
  cp -r "$SRC/swmm" "$dir/swmm"
  mkdir -p "$dir/swmm/dats" && cp "$SRC"/dats/* "$dir/swmm/dats/"   # live SWMM
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
resume() {
  local dir=$1 out=$2 cpid=$3
  rm -rf "$dir/${out}_res$cpid"
  cp -r "$dir/$out" "$dir/${out}_res$cpid"
  sed -i "s|^output_folder=.*|output_folder=\"${out}_res$cpid\"|" "$dir/${out}_res$cpid/cfg/config_$cpid.cfg"
  ( cd "$dir" && ./build/triton.exe "${out}_res$cpid/cfg/config_$cpid.cfg" > "res$cpid.log" 2>&1 )
}

# compare <clean-out> <resumed-out>: bin/, side-file, .out; prints verdicts
compare() {
  local a="$ROOT/$1" b="$ROOT/$2" rc=0
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
say "--- A: serial, 1 rank, SEQ output, live SWMM ---"
D="$ROOT/serial"; run_case serial true
make_cfg "$D/case_ka.cfg"  out_ka  swmm_snapshot_keep_all=1
make_cfg "$D/case_dis.cfg" out_dis swmm_snapshot_disable=1
make_cfg "$D/case_k2.cfg"  out_k2

( cd "$D" && ./build/triton.exe case_ka.cfg  > ka.log  2>&1 ) || bad "A3 clean keep_all leg"
( cd "$D" && ./build/triton.exe case_dis.cfg > dis.log 2>&1 ) || bad "A2 clean disable leg"
( cd "$D" && ./build/triton.exe case_k2.cfg  > k2.log  2>&1 ) || bad "A1/A7 clean keep-2 leg"

# A2: replay forced
resume "$D" out_dis 36
grep -aq "path=replay reason=absent" "$D/res36.log" \
  && ok "A2 resume@36 disable: took replay route (reason=absent)" \
  || bad "A2 resume@36 disable: expected replay route"
compare serial/out_dis serial/out_dis_res36

# A3: snapshot route, mid-run
resume "$D" out_ka 36
grep -aq "path=snapshot" "$D/res36.log" \
  && ok "A3 resume@36 keep_all: took snapshot route" \
  || bad "A3 resume@36 keep_all: expected snapshot route"
compare serial/out_ka serial/out_ka_res36

# A4: first checkpoint
resume "$D" out_ka 1
grep -aq "path=snapshot" "$D/res1.log" \
  && ok "A4 resume@1: snapshot route from first checkpoint" \
  || bad "A4 resume@1: expected snapshot route"
compare serial/out_ka serial/out_ka_res1

# A5: last checkpoint
resume "$D" out_ka 144
compare serial/out_ka serial/out_ka_res144

# A6: multi-resume chain (resume@36, then resume ITS checkpoint 72)
rm -rf "$D/out_ka_chain" && cp -r "$D/out_ka_res36" "$D/out_ka_chain"
sed -i "s|^output_folder=.*|output_folder=\"out_ka_chain\"|" "$D/out_ka_chain/cfg/config_72.cfg"
( cd "$D" && ./build/triton.exe out_ka_chain/cfg/config_72.cfg > chain.log 2>&1 ) \
  && compare serial/out_ka serial/out_ka_chain && ok "A6 multi-resume chain (36 -> 72) reproduces clean" \
  || bad "A6 multi-resume chain"

# A7: keep-2 collision -> replay with classified reason
resume "$D" out_k2 36
grep -aq "path=replay reason=retention-collision" "$D/res36.log" \
  && ok "A7 keep-2 collision: replay route with reason=retention-collision" \
  || bad "A7 keep-2 collision: expected classified replay route"
compare serial/out_k2 serial/out_k2_res36

# ---------- B. MPI np8, PAR ----------
say "--- B: MPI, 8 ranks, PAR output ---"
D="$ROOT/mpi8"; run_case mpi8 true
make_cfg "$D/case_ka.cfg" out_ka8 swmm_snapshot_keep_all=1
sed -i "s|^output_option=.*|output_option=PAR|" "$D/case_ka.cfg"
( cd "$D" && mpirun --allow-run-as-root --oversubscribe -np 8 ./build/triton.exe case_ka.cfg > ka.log 2>&1 ) \
  || bad "B clean leg"
resume "$D" out_ka8 36
grep -aq "path=snapshot" "$D/res36.log" && ok "B resume@36 np8: snapshot route" || bad "B route"
compare mpi8/out_ka8 mpi8/out_ka8_res36

# ---------- C. OMP binary, 1 rank ----------
if ldd "$BIN_ABS" 2>/dev/null | grep -q "gomp"; then
  say "--- C: OpenMP binary, 1 rank ---"
  D="$ROOT/omp1"; run_case omp1 true
  make_cfg "$D/case_ka.cfg" out_ka_omp swmm_snapshot_keep_all=1
  ( cd "$D" && OMP_NUM_THREADS=4 ./build/triton.exe case_ka.cfg > ka.log 2>&1 ) || bad "C clean leg"
  resume "$D" out_ka_omp 36
  compare omp1/out_ka_omp omp1/out_ka_omp_res36
else
  say "--- C: skipped (binary is not an OpenMP build) ---"
fi

# ---------- D. np16 deadlock regression ----------
say "--- D: np16, rank 0 strip holds no manhole ---"
D="$ROOT/np16"; run_case np16 true
make_cfg "$D/case.cfg" out16
sed -i "s|^sim_duration=.*|sim_duration=3600|" "$D/case.cfg"
( cd "$D" && timeout 300 mpirun --allow-run-as-root --oversubscribe -np 16 ./build/triton.exe case.cfg > np16.log 2>&1 )
rc=$?
if [ $rc -eq 0 ] && grep -aq "Simulation ends" "$D/np16.log" && [ -s "$ROOT/np16/out16/swmm/hydraulics.rpt" ]; then
  ok "D np16 completes and writes hydraulics.rpt (pre-integration: deadlock)"
else
  bad "D np16 did not complete cleanly (rc=$rc)"
fi

# ---------- E. degraded SWMM, baseline regression ----------
say "--- E: degraded SWMM (no dats/), baseline regression ---"
D="$ROOT/degraded"; mkdir -p "$D/build"
for f in elevation_3.50m.dem mannings_3.50m.dem strmflow extbc cid; do cp -r "$SRC/$f" "$D/" 2>/dev/null; done
cp -r "$SRC/swmm" "$D/swmm"                      # deliberately NO swmm/dats
ln -sf "$BIN_ABS" "$D/build/triton.exe"
make_cfg "$D/case.cfg" out_degraded
( cd "$D" && ./build/triton.exe case.cfg > run.log 2>&1 ) || bad "E clean leg"
if [ -n "${BASELINE:-}" ] && [ -d "${BASELINE:-}/bin" ]; then
  diff -r "${BASELINE}/bin" "$ROOT/degraded/out_degraded/bin" > /dev/null 2>&1 \
    && ok "E degraded rasters byte-identical to pre-integration baseline" \
    || bad "E degraded rasters DIFFER from pre-integration baseline"
else
  say "     (no BASELINE dir given; degraded regression skipped)"
fi
# degraded resume still works (snapshot refuses, replay fallback)
resume "$D" out_degraded 36
grep -aq "path=replay" "$D/res36.log" \
  && compare degraded/out_degraded degraded/out_degraded_res36 \
  && ok "E degraded resume@36: replay fallback reproduces clean" \
  || bad "E degraded resume@36"

# ---------- F. failure semantics ----------
say "--- F: failure semantics ---"
D="$ROOT/serial"
# F1: missing side-file
rm -rf "$D/out_fail_ms" && cp -r "$D/out_k2" "$D/out_fail_ms"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_ms\"|" "$D/out_fail_ms/cfg/config_36.cfg"
rm -f "$D/out_fail_ms/swmm/hydraulics_exchange_replay.bin"
( cd "$D" && ./build/triton.exe out_fail_ms/cfg/config_36.cfg > f1.log 2>&1 )
grep -aq "requires the exchange-replay side-file" "$D/f1.log" \
  && ok "F1 missing side-file: fatal with guidance" || bad "F1 missing side-file"
# F2: V1-magic side-file (old format)
rm -rf "$D/out_fail_v1" && cp -r "$D/out_k2" "$D/out_fail_v1"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_v1\"|" "$D/out_fail_v1/cfg/config_36.cfg"
python3 - "$D/out_fail_v1/swmm/hydraulics_exchange_replay.bin" <<'EOF'
import struct, sys
p = sys.argv[1]
b = bytearray(open(p,'rb').read())
b[0:4] = struct.pack('<i', 0x53574D4D)          # ASCII "SWMM": the V1 magic
b[4:8] = struct.pack('<i', 13)                  # node count still first
open(p,'wb').write(b)
EOF
( cd "$D" && ./build/triton.exe out_fail_v1/cfg/config_36.cfg > f2.log 2>&1 )
grep -aq "pre-versioning (V1) magic" "$D/f2.log" \
  && ok "F2 V1 side-file: refused with the legacy diagnostic" || bad "F2 V1 side-file"
# F3: corrupt snapshot (present but truncated -> Unreadable -> abort)
rm -rf "$D/out_fail_snap" && cp -r "$D/out_ka" "$D/out_fail_snap"
sed -i "s|^output_folder=.*|output_folder=\"out_fail_snap\"|" "$D/out_fail_snap/cfg/config_36.cfg"
python3 - "$D/out_fail_snap/swmm/hydraulics_state_cp36.snapshot" <<'EOF'
import sys
p = sys.argv[1]
b = open(p,'rb').read()
open(p,'wb').write(b[:len(b)//2])               # truncate: header short
EOF
( cd "$D" && ./build/triton.exe out_fail_snap/cfg/config_36.cfg > f3.log 2>&1 )
grep -aq "UNUSABLE" "$D/f3.log" && grep -aq "Coupled resume aborted" "$D/f3.log" \
  && ok "F3 corrupt snapshot: aborted (UNREADABLE), no fallback" || bad "F3 corrupt snapshot"

# ---------- G. performance.txt shape ----------
say "--- G: performance.txt accounting ---"
P="$ROOT/serial/out_ka/performance.txt"
if [ -s "$P" ]; then
  head -1 "$P" | grep -q "SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER" \
    && ok "G performance header carries the four SWMM child columns" \
    || bad "G performance header missing child columns"
  python3 - "$P" <<'EOF' && ok "G SWMM_OTHER >= 0 on every row (residual sign)" || bad "G SWMM_OTHER negative"
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0]
i_other = hdr.index('SWMM_OTHER'); i_x = hdr.index('SWMM_XFER'); i_m = hdr.index('SWMM_MPI')
i_s = hdr.index('SWMM_STEP'); i_p = hdr.index('SWMM')
for r in rows[1:]:
    v = [float(x) for x in r[1:]]
    assert v[i_other] >= 0.0, r
    assert abs(v[i_x]+v[i_m]+v[i_s]+v[i_other] - v[i_p]) < 1e-6, r
EOF
else
  bad "G performance.txt not found"
fi

say ""
say "=== suite result: $PASS passed, $FAIL failed ==="
[ -n "$FAILED_CASES" ] && say "$FAILED_CASES"
[ $FAIL -eq 0 ]
