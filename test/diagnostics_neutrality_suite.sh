#!/usr/bin/env bash
# Hydraulics-neutrality suite of the coupling-diagnostics captures.
#
# Builds the faba2232 baseline and the branch with the SAME flags and the SAME Kokkos tree, runs
# both on the fixed coupled case (test/coupled_case/) and compares the hydraulic outputs byte for
# byte: the H, QX, QY and MH rasters, the exchange side-file, the SWMM .out and the .rpt
# flow-routing continuity block. Each case that switches a capture on also asserts that every
# enabled capture file exists with 16 + records x record size bytes. The cases N1..N13, the arms
# each one runs, the quantity it reads and the test-only build it must fail on are the table of the
# approved shape, section 10; test/neutrality_compare.py holds every comparison. Cases N14 (the
# driven-stage open-water outfall, cfg key swmm_driven_outfalls, on its own case directory
# test/coupled_case_outfall/) and N15 (the per-node DEM-rim exemption, cfg key
# swmm_dem_rim_exempt_nodes, on test/coupled_case/ with swmm_node_rim_from_dem=1) are sections
# 17.5 and 17.6; N14's readers live in test/coupled_case_outfall/outfall_checks.py and the
# test-only breaks of N14 and N15 are source mutants the suite builds from a copy of the test tree.
#
# A byte difference in a hydraulic output at any case reclassifies the failing capture as possibly
# hydraulics-affecting: the merge stops until the developer rules. The suite reports it and never
# scopes it out.
#
# Usage:
#   diagnostics_neutrality_suite.sh --baseline-sha SHA --test-sha SHA
#       [--cmake-flags-from FILE]      cmake line of the hhemt compile script for the GPU backend
#       [--cpu-cmake-flags-from FILE]  the same for the CPU backend (default: hhemt's CPU flags,
#                                      without its conda libstdc++ fragment; recorded in the record)
#       [--repo DIR] [--work DIR] [--kokkos-dir DIR] [--cases N1,N3,...] [--np-big 8] [--no-gpu]
#       [--keep-going] [--sec-floor 0.05] [--build-only] [--n6-part small|big|all]
# Without --cmake-flags-from (or with --no-gpu) the A100 arms (N6, the A100 arm of N9) are SKIPPED
# and reported as SKIPPED, never as passed.
#
# The caller supplies the environment (modules, a Slurm allocation, mpirun on PATH). Every compile
# and every run belongs on Rivanna scratch, never on the laptop. Exit status: 0 only when no case
# FAILED or ERRORED; N9 is reported to the developer and does not gate.
set -u

THIS=$(readlink -f "$0")
REPO=$(cd "$(dirname "$THIS")/.." && pwd)
BASE_SHA=""; TEST_SHA=""; GPU_FLAGS_FILE=""; CPU_FLAGS_FILE=""; WORK=""; KOKKOS_DIR=""
CASES="N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15"; NP_BIG=8; NO_GPU=0; KEEP_GOING=1; SEC_FLOOR=0.05; BUILD_ONLY=0; N6_PART=all
while [ $# -gt 0 ]; do
  case "$1" in
    --baseline-sha) BASE_SHA=$2; shift 2;;
    --test-sha) TEST_SHA=$2; shift 2;;
    --cmake-flags-from) GPU_FLAGS_FILE=$2; shift 2;;
    --cpu-cmake-flags-from) CPU_FLAGS_FILE=$2; shift 2;;
    --repo) REPO=$(readlink -f "$2"); shift 2;;
    --work) WORK=$2; shift 2;;
    --kokkos-dir) KOKKOS_DIR=$2; shift 2;;
    --cases) CASES=$2; shift 2;;
    --np-big) NP_BIG=$2; shift 2;;
    --no-gpu) NO_GPU=1; shift;;
    --n6-part) N6_PART=$2; shift 2;;   # small: A100 np1 and np2 (1-2 GPUs); big: the np8 arms and the T-S/T-E arms (one 8-GPU node); all
    --build-only) BUILD_ONLY=1; shift;;   # build every variant the selected cases need, run nothing (a node without the GPU can check the A100 builds)
    --keep-going) KEEP_GOING=1; shift;;
    --sec-floor) SEC_FLOOR=$2; shift 2;;
    -h|--help) sed -n '2,40p' "$THIS"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$BASE_SHA" ] && [ -n "$TEST_SHA" ] || { echo "--baseline-sha and --test-sha are required" >&2; exit 2; }
[ -n "$WORK" ] || WORK="$PWD/neutrality_run"
mkdir -p "$WORK"; WORK=$(readlink -f "$WORK")
LOGS="$WORK/logs"; mkdir -p "$LOGS" "$WORK/build" "$WORK/run" "$WORK/src"
RESULTS="$WORK/results.txt"; : > "$RESULTS"
BUILD_RECORD="$WORK/build_record.txt"; : > "$BUILD_RECORD"

export OMP_NUM_THREADS=${NEUTRALITY_OMP_THREADS:-1}   # Kokkos OpenMP pool size: fixed for every arm
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
LAUNCH_MPI="mpirun --allow-run-as-root --oversubscribe"
PYTHON=${PYTHON:-$(command -v python3.11 || command -v python3)}   # Rivanna's default python3 is 3.6
CMP="$PYTHON -I"
TEST_SRC="$WORK/src/test_tree"; BASE_SRC="$WORK/src/base_tree"
NEUT="$TEST_SRC/test/neutrality_compare.py"
CASE_SRC="$TEST_SRC/test/coupled_case"

say() { printf '%s\n' "$*"; }
want() { case ",$CASES," in *",$1,"*) return 0;; *) return 1;; esac; }

# ------------------------------------------------------------------ results
HYD_FAIL=0
rec() { # rec ID STATUS detail
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$RESULTS"
  say "[$2] $1 $3"
  if [ "$2" = FAIL ] && [ "${HYD:-0}" = 1 ]; then HYD_FAIL=1; fi
}
# chk ID EXPECT DESCRIPTION cmd... : EXPECT 0 = must pass (exit 0); 1 = must-fail arm (exit exactly 1).
# The exit status is read from the command itself, unpiped.
chk() {
  local id=$1 exp=$2 desc=$3; shift 3
  "$@" > "$LOGS/$(echo "$id" | tr '/ ()' '____').log" 2>&1
  local rc=$?
  if [ "$exp" = 0 ]; then
    if [ $rc -eq 0 ]; then rec "$id" PASS "$desc"
    elif [ $rc -eq 1 ]; then rec "$id" FAIL "$desc (exit $rc; see logs)"
    else rec "$id" ERROR "$desc (exit $rc; see logs)"; fi
  else
    if [ $rc -eq 1 ]; then rec "$id" PASS "$desc (must-fail arm failed as it must)"
    elif [ $rc -eq 0 ]; then rec "$id" BLIND "$desc (must-fail arm did NOT fail: the instrument cannot detect this break on this case)"
    else rec "$id" ERROR "$desc (exit $rc; see logs)"; fi
  fi
  return $rc
}

# ------------------------------------------------------------------ sources and builds
fetch_sources() {
  rm -rf "$TEST_SRC" "$BASE_SRC"; mkdir -p "$TEST_SRC" "$BASE_SRC"
  git -C "$REPO" archive "$TEST_SHA" | tar -x -C "$TEST_SRC" || { say "cannot archive $TEST_SHA"; exit 2; }
  git -C "$REPO" archive "$BASE_SHA" | tar -x -C "$BASE_SRC" || { say "cannot archive $BASE_SHA"; exit 2; }
  [ -n "$KOKKOS_DIR" ] || KOKKOS_DIR="$REPO/external/kokkos"
  [ -f "$KOKKOS_DIR/CMakeLists.txt" ] || { say "no Kokkos tree at $KOKKOS_DIR (pass --kokkos-dir; the submodule is empty in a worktree)"; exit 2; }
  for t in "$TEST_SRC" "$BASE_SRC"; do rm -rf "$t/external/kokkos"; cp -r "$KOKKOS_DIR" "$t/external/kokkos"; done
  local h1 h2
  h1=$(cd "$TEST_SRC/external/kokkos" && find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)
  h2=$(cd "$BASE_SRC/external/kokkos" && find . -type f -not -path './.git/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)
  {
    echo "== Kokkos baseline equality =="
    echo "kokkos source dir: $KOKKOS_DIR"
    echo "revision: $(git -C "$KOKKOS_DIR" rev-parse HEAD 2>&1 | head -1)"
    echo "tracked-file modifications (git status --porcelain lines): $(git -C "$KOKKOS_DIR" status --porcelain 2>/dev/null | wc -l)"
    echo "tree hash (sha256 over sha256 of every file but .git) baseline copy: $h1"
    echo "tree hash branch copy:                                              $h2"
    [ "$h1" = "$h2" ] && echo "VERIFIED: the two trees are byte-identical files" || echo "MISMATCH: the Kokkos trees differ"
  } | tee -a "$BUILD_RECORD"
  [ "$h1" = "$h2" ] || exit 2
  echo "baseline sha $BASE_SHA; test sha $TEST_SHA" >> "$BUILD_RECORD"
}

# hhemt's CPU flags verbatim, minus the conda libstdc++ fragment that depends on a conda environment
DEFAULT_CPU_FLAGS="-DTRITON_IGNORE_MACHINE_FILES=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_HIP=OFF -DKokkos_ENABLE_CUDA=OFF -DCMAKE_CXX_FLAGS='-O3 -fopenmp' -DCMAKE_C_FLAGS='-fopenmp' -DCMAKE_SHARED_LINKER_FLAGS='-fopenmp' -DCMAKE_EXE_LINKER_FLAGS='-fopenmp'"

# the flags of the cmake line of an hhemt compile script (everything between "cmake " and the source dir)
flags_from_script() {
  local f=$1 line
  line=$(grep -E '^cmake ' "$f" | head -1)
  [ -n "$line" ] || return 1
  line=${line#cmake }
  line=${line%% \"\$\{TRITON_DIR\}\"*}
  # the suite sets the SWMM options itself
  line=${line//-DTRITON_ENABLE_SWMM=ON/}
  line=${line//-DTRITON_SWMM_FLOODING_DEBUG=ON/}
  printf '%s' "$line"
}

CPU_FLAGS=""; GPU_FLAGS=""
resolve_flags() {
  if [ -n "$CPU_FLAGS_FILE" ]; then CPU_FLAGS=$(flags_from_script "$CPU_FLAGS_FILE") || { say "no cmake line in $CPU_FLAGS_FILE"; exit 2; }
  else CPU_FLAGS=$DEFAULT_CPU_FLAGS; fi
  if [ -n "$GPU_FLAGS_FILE" ] && [ "$NO_GPU" = 0 ]; then
    GPU_FLAGS=$(flags_from_script "$GPU_FLAGS_FILE") || { say "no cmake line in $GPU_FLAGS_FILE"; exit 2; }
    export NVCC=${NVCC:-$(command -v nvcc || true)} MPICXX=${MPICXX:-$(command -v mpicxx || true)} MPICC=${MPICC:-$(command -v mpicc || true)}
  else GPU_FLAGS=""; fi
  { echo "== flags =="; echo "CPU flags: $CPU_FLAGS"; echo "GPU flags: ${GPU_FLAGS:-(none: A100 arms skipped)}"; } | tee -a "$BUILD_RECORD"
}

# build NAME SRCDIR BACKEND "cxx defines" "extra C flags" [swmm]   (swmm: on | off)
build() {
  local name=$1 src=$2 backend=$3 defs=$4 cextra=$5 swmm=${6:-on} bd="$WORK/build/$1" flags cbase
  [ -x "$bd/triton.exe" ] && [ -f "$bd/.built" ] && return 0
  [ "$backend" = gpu ] && flags=$GPU_FLAGS || flags=$CPU_FLAGS
  rm -rf "$bd"; mkdir -p "$bd"
  local cmake_cmd="cmake -DTRITON_ENABLE_SWMM=$( [ "$swmm" = on ] && echo ON || echo OFF ) -DTRITON_SWMM_FLOODING_DEBUG=ON $flags"
  [ -n "$defs" ] && cmake_cmd="$cmake_cmd -DCOMPILER_FLAGS_APPEND=\"$defs\""
  if [ -n "$cextra" ]; then
    cbase=$(printf '%s' "$flags" | sed -n "s/.*-DCMAKE_C_FLAGS='\([^']*\)'.*/\1/p")
    cmake_cmd="$cmake_cmd -DCMAKE_C_FLAGS=\"$cbase $cextra\""
  fi
  ( cd "$bd" && eval "$cmake_cmd \"$src\"" > cmake.log 2>&1 ); local crc=$?
  local mrc=-1
  [ $crc -eq 0 ] && { ( cd "$bd" && make -j"${NEUTRALITY_JOBS:-$(nproc)}" > make.log 2>&1 ); mrc=$?; }
  {
    echo "== build $name =="
    echo "source: $src ($([ "$src" = "$BASE_SRC" ] && echo "baseline $BASE_SHA" || echo "test sha $TEST_SHA"))  backend: $backend  TRITON_ENABLE_SWMM: $swmm"
    echo "C++ defines (COMPILER_FLAGS_APPEND): ${defs:-(none)}  extra C flags: ${cextra:-(none)}"
    echo "cmake line: $cmake_cmd \"$src\""
    echo "cmake exit: $crc  make exit: $mrc"
    [ -x "$bd/triton.exe" ] && echo "triton.exe sha256: $(sha256sum "$bd/triton.exe" | cut -c1-64)"
  } >> "$BUILD_RECORD"
  if [ $crc -eq 0 ] && [ $mrc -eq 0 ] && [ -x "$bd/triton.exe" ]; then touch "$bd/.built"; return 0; fi
  rec "BUILD/$name" ERROR "cmake exit $crc, make exit $mrc (see $bd/cmake.log, $bd/make.log)"
  return 1
}
exe() { echo "$WORK/build/$1/triton.exe"; }

# ------------------------------------------------------------------ running
# rundir NAME BUILD : a fresh copy of the case at $WORK/run/NAME with build/triton.exe -> the build's binary
rundir() {
  local d="$WORK/run/$1"
  rm -rf "$d"; mkdir -p "$d/build"
  cp -r "$CASE_SRC"/. "$d/"
  rm -f "$d/make_case.py"
  ln -sf "$(exe "$2")" "$d/build/triton.exe"
  echo "$d"
}
set_key() { # set_key cfg key value : replace the key's line or append it
  local cfg=$1 k=$2 v=$3
  sed -i "/^$k=/d" "$cfg"; printf '%s=%s\n' "$k" "$v" >> "$cfg"
}
mkcfg() { # mkcfg dir cfgname outname [key=value ...]
  local d=$1 cfg=$2 out=$3; shift 3
  cp "$d/TRITONSWMM.cfg" "$d/$cfg"
  set_key "$d/$cfg" output_folder "\"$out\""
  local kv
  for kv in "$@"; do set_key "$d/$cfg" "${kv%%=*}" "${kv#*=}"; done
}
CAP_T1="swmm_diag_steps=1 swmm_diag_node_path3_per_step=1 swmm_diag_links=ALL"
CAP_SURF="diag_domain_balance=1 diag_wet_cells=1 diag_dt_limiter=1"
CAP_ALL="$CAP_T1 $CAP_SURF"
ALLFILES="swmm_step_diag.bin,swmm_node_path3.bin,swmm_node_path3_step.bin,swmm_link_flow_diag.bin,swmm_node_diag.nodes,swmm_link_flow_diag.links,edge_flux.bin,dt_limiter.bin,wet_cells.csv,domain_balance.csv"
SURFFILES="edge_flux.bin,dt_limiter.bin,wet_cells.csv,domain_balance.csv"

launcher() { # launcher NP -> the command prefix
  if [ "$1" -gt 1 ]; then echo "$LAUNCH_MPI -np $1"; else echo ""; fi
}
# run_arm NAME BUILD NP KEYS... : a clean run; returns triton's exit status; sets ARM_DIR and ARM_OUT
ARM_DIR=""; ARM_OUT=""
run_arm() {
  local name=$1 b=$2 np=$3; shift 3
  [ "$BUILD_ONLY" = 1 ] && { rec "RUN/$name" SKIP "--build-only"; return 1; }
  local d; d=$(rundir "$name" "$b")
  mkcfg "$d" arm.cfg out "$@"
  ARM_DIR=$d; ARM_OUT="$d/out"
  ( cd "$d" && $(launcher "$np") ./build/triton.exe arm.cfg > run.log 2>&1 ); local rc=$?
  if [ $rc -ne 0 ] || ! grep -aq "Simulation ends" "$d/run.log"; then
    rec "RUN/$name" ERROR "exit $rc without a completed simulation (see $d/run.log)"
    return 1
  fi
  return 0
}
# resume_arm SRC_ARM_DIR NAME BUILD NP CKPT [extra key=value ...] : resume from checkpoint CKPT of SRC's out/
resume_arm() {
  local src=$1 name=$2 b=$3 np=$4 ck=$5; shift 5
  local d="$WORK/run/$name"
  rm -rf "$d"; mkdir -p "$d/build"; cp -r "$src"/. "$d/" 2>/dev/null
  rm -rf "$d/out_res"; cp -r "$src/out" "$d/out_res"
  ln -sf "$(exe "$b")" "$d/build/triton.exe"
  local cfg="$d/out_res/cfg/config_$ck.cfg"
  [ -f "$cfg" ] || { rec "RUN/$name" ERROR "no checkpoint cfg $cfg"; return 1; }
  sed -i "s|^output_folder=.*|output_folder=\"out_res\"|" "$cfg"
  sed -i '/^diag_test_pause/d' "$cfg"
  local kv; for kv in "$@"; do set_key "$cfg" "${kv%%=*}" "${kv#*=}"; done
  ( cd "$d" && $(launcher "$np") ./build/triton.exe "out_res/cfg/config_$ck.cfg" > res.log 2>&1 ); local rc=$?
  ARM_DIR=$d; ARM_OUT="$d/out_res"
  if [ $rc -ne 0 ] || ! grep -aq "Simulation ends" "$d/res.log"; then
    rec "RUN/$name" ERROR "resume exit $rc without a completed simulation (see $d/res.log)"
    return 1
  fi
  return 0
}
last_cfg() { ls "$1"/cfg/config_*.cfg 2>/dev/null | sed 's/.*config_\([0-9]*\)\.cfg/\1/' | sort -n | tail -1; }

# kill_run NAME BUILD NP WAIT_KIND ARG KEYS... : start a run in its own process group, wait for the
# state, SIGKILL the group. WAIT_KIND: cfg (config_ARG exists, then 0.2 s later), cfgsnap (config_ARG
# and the SWMM snapshot of ARG exist, then a pause of 2 s inside the T-P pause), size (file ARG1 of
# the output folder reaches ARG2 bytes, then a pause of 2 s), none.
kill_run() {
  local name=$1 b=$2 np=$3 kind=$4 arg=$5; shift 5
  [ "$BUILD_ONLY" = 1 ] && { rec "RUN/$name" SKIP "--build-only"; return 1; }
  local d; d=$(rundir "$name" "$b")
  mkcfg "$d" arm.cfg out "$@"
  ARM_DIR=$d; ARM_OUT="$d/out"
  ( cd "$d" && exec setsid $(launcher "$np") ./build/triton.exe arm.cfg > run.log 2>&1 ) &
  local bgpid=$! t=0 ok=0 pgid
  sleep 0.5
  pgid=$(ps -o pgid= -p "$bgpid" 2>/dev/null | tr -d ' ')
  # the run must lead its own process group; the suite's own group is never signalled
  [ -n "$pgid" ] && [ "$pgid" = "$bgpid" ] || { kill -9 "$bgpid" 2>/dev/null; rec "RUN/$name" ERROR "the run is not the leader of its own process group (pgid '$pgid', pid $bgpid)"; return 1; }
  while [ $t -lt ${KILL_TIMEOUT:-1200} ]; do
    case "$kind" in
      cfg) [ -f "$d/out/cfg/config_$arg.cfg" ] && ok=1;;
      cfgsnap) [ -f "$d/out/cfg/config_$arg.cfg" ] && ls "$d"/out/swmm/*cp"$arg".snapshot > /dev/null 2>&1 && ok=1;;
      size) set -- $arg; [ -f "$d/out/$1" ] && [ "$(stat -c %s "$d/out/$1")" -ge "$2" ] && ok=1;;
      csvrows) set -- $arg; [ -f "$d/out/$1" ] && [ "$(grep -vc '^%' "$d/out/$1")" -ge "$2" ] && ok=1;;
    esac
    [ $ok = 1 ] && break
    kill -0 "$bgpid" 2>/dev/null || break
    sleep 0.05; t=$((t+1))
  done
  if [ $ok = 1 ]; then
    case "$kind" in cfg) sleep 0.2;; *) sleep 2;; esac
  fi
  local alive=0; kill -0 "$bgpid" 2>/dev/null && alive=1
  kill -9 -- "-$pgid" 2>/dev/null
  kill -9 "$bgpid" 2>/dev/null; wait "$bgpid" 2>/dev/null
  if [ $ok = 0 ] || [ $alive = 0 ]; then
    rec "RUN/$name" ERROR "the kill state was not reached while the run was alive (state reached: $ok, alive at kill: $alive; see $d/run.log)"
    return 1
  fi
  return 0
}

# a resumed arm against its clean arm: hydraulics and every capture file of the list
cmp_arm() { # cmp_arm ID EXPECT DESC CLEAN_OUT RESUMED_OUT [files]
  HYD=1 chk "$1" "$2" "$3" $CMP "$NEUT" compare --a "$4" --b "$5" ${6:+--captures "$6"}
}

# ------------------------------------------------------------------ baselines (memoised)
BASE_DIR_cpu_1=""; BASE_DIR_cpu_8=""
base_run() { # base_run BACKEND NP [extra keys] -> sets BASE_OUT
  local be=$1 np=$2 tag=$3; shift 3
  local name="base_${be}_np${np}${tag}"
  if [ -f "$WORK/run/$name/.done" ]; then BASE_OUT="$WORK/run/$name/out"; return 0; fi
  run_arm "$name" "${be}_base" "$np" "$@" || return 1
  touch "$WORK/run/$name/.done"; BASE_OUT="$WORK/run/$name/out"
}
branch_run() { # branch_run NAME BACKEND NP KEYS...
  local name=$1 be=$2 np=$3; shift 3
  run_arm "$name" "${be}_branch" "$np" "$@"
}

# the checkpoint the resume and kill arms use, derived from the case's own cfg
fetch_sources
SIM=$(sed -n 's/^sim_duration=//p' "$CASE_SRC/TRITONSWMM.cfg"); PRINT=$(sed -n 's/^print_interval=//p' "$CASE_SRC/TRITONSWMM.cfg")
NCP=$(( ${SIM%.*} / ${PRINT%.*} )); CK=$(( NCP / 4 )); [ $CK -lt 1 ] && CK=1
CELL=$(python3 -c "import re,sys; print([l.split()[1] for l in open('$CASE_SRC/elevation.dem') if l.lower().startswith('cellsize')][0])")
say "case: sim_duration=$SIM print_interval=$PRINT checkpoints=$NCP resume checkpoint=$CK cellsize=$CELL"
resolve_flags
{ echo "case: sim_duration=$SIM print_interval=$PRINT checkpoints=$NCP resume checkpoint=$CK"; } >> "$BUILD_RECORD"

# builds needed by the selected cases
need_cpu() { build cpu_base "$BASE_SRC" cpu "" "" on; build cpu_branch "$TEST_SRC" cpu "" "" on; }
GPU_OK=0; [ -n "$GPU_FLAGS" ] && GPU_OK=1

# ------------------------------------------------------------------ the case
# coverage of the fixed case on the baseline (R4): entered before anything is read from it
case_coverage() {
  need_cpu || return 1
  base_run cpu 1 "" || return 1
  HYD=0 chk "COVER/baseline" 0 "the fixed case on the faba2232 baseline enters every capture branch (non-converged step, flooded node, extbc type-1 cell, nonzero east/north edge flow, S(0) > 0)" \
    $CMP "$NEUT" coverage --out "$BASE_OUT" --dem "$CASE_SRC/elevation.dem" --init-h "$CASE_SRC/init_h.dem" --cell "$CELL" --extbc-rows 14 30
}

# ------------------------------------------------------------------ N1 / N4: hydraulics unchanged, no capture file with every capture off
case_N1() {
  need_cpu || return 1
  local np
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "" || continue
    branch_run "n1_off_np$np" cpu "$np" || continue
    cmp_arm "N1/cpu_np$np" 0 "all captures off, CPU np$np: branch against the faba2232 baseline, byte for byte" "$BASE_OUT" "$ARM_OUT"
    chk "N1/cpu_np$np/absent" 0 "no capture file exists with every capture off (np$np)" $CMP "$NEUT" captures-absent --out "$ARM_OUT"
    [ "$np" = 1 ] && N1_OUT=$ARM_OUT
    # N4: the timers alone; the 17-column performance header of the resume suite's section G
    local hdr; hdr=$(head -1 "$ARM_OUT/performance.txt" | tr -d '\r')
    local want='%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER, SWMM_SOLVE, SWMM_LOG, DIAG, Other, Simulation, Init, Total'
    [ "$hdr" = "$want" ] && rec "N4/cpu_np$np" PASS "timers only: hydraulics identical to the baseline (N1 arm) and performance.txt carries the 17 columns" || rec "N4/cpu_np$np" FAIL "performance.txt header is '$hdr'"
    cmp_arm "N4/cpu_np$np/bytes" 0 "timers only, CPU np$np: branch against the baseline, byte for byte" "$BASE_OUT" "$ARM_OUT"
  done
}

# ------------------------------------------------------------------ N8: the comparator itself
case_N8() {
  [ -d "${N1_OUT:-}" ] || { case_N1 || return 1; }
  chk "N8/flip" 0 "flip one bit of one H raster in a copy of N1's branch output" $CMP "$NEUT" flip-byte --src "$N1_OUT" --dst "$WORK/run/n8_flipped_out"
  # the comparator must exit non-zero on the flipped copy (exit exactly 1)
  chk "N8" 1 "comparator on the flipped copy" $CMP "$NEUT" compare --a "$N1_OUT" --b "$WORK/run/n8_flipped_out"
  chk "N8/control" 0 "the comparator exits 0 on an unflipped copy (must-pass control)" $CMP "$NEUT" compare --a "$N1_OUT" --b "$N1_OUT"
  # the comparator's own self-test of the per-rank MPI allowance (erratum E-18); no solver run
  chk "N8/perf-diff-selftest" 0 "test_neutrality_perf_diff.py: the MPI allowance is net of the rank's own DIAG and reaches no other column" $PYTHON "$TEST_SRC/test/test_neutrality_perf_diff.py"
}

# ------------------------------------------------------------------ band arms: the capture-off runs, twice
off_band() { # off_band BACKEND NP GPU_DIRECT -> sets OFF1 OFF2 OFF3 and OFFS
  local be=$1 np=$2 gd=${3:-0}
  local tag="${be}_np${np}_gd${gd}"
  branch_run "band_${tag}_1" "$be" "$np" gpu_direct_flag=$gd || return 1; OFF1=$ARM_OUT
  branch_run "band_${tag}_2" "$be" "$np" gpu_direct_flag=$gd || return 1; OFF2=$ARM_OUT
  branch_run "band_${tag}_3" "$be" "$np" gpu_direct_flag=$gd || return 1; OFF3=$ARM_OUT
  OFFS="$OFF1 $OFF2 $OFF3"   # three capture-off runs: their spread, floored, is the band
}

# ------------------------------------------------------------------ N2
case_N2() {
  need_cpu || return 1
  local np
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "" || continue
    off_band cpu "$np" 0 || continue
    branch_run "n2_t1_np$np" cpu "$np" $CAP_T1 || continue
    cmp_arm "N2/cpu_np$np" 0 "T1 on (ALL links, T1c on), CPU np$np: hydraulics against the baseline" "$BASE_OUT" "$ARM_OUT"
    chk "N2/cpu_np$np/files" 0 "every T1 file exists with 16 + records x record size bytes (np$np)" $CMP "$NEUT" captures-check --out "$ARM_OUT" --enabled t1,t1c,t1d --end-time "$SIM"
    chk "N2/cpu_np$np/perf" 0 "per-rank differential of the timers against the capture-off band, DIAG > 0 on rank 0 (np$np)" \
      $CMP "$NEUT" perf-diff --on "$ARM_OUT" --off $OFFS --cols Compute,MPI,IO,Resize,SWMM,SWMM_XFER,SWMM_MPI,SWMM_STEP,SWMM_OTHER,SWMM_SOLVE,SWMM_LOG,Other --abs-floor "$SEC_FLOOR" --diag-positive 0
    # must-be-nonzero control for N1's absence check: the same check on a capture-ON run must fail
    chk "N2/cpu_np$np/absent-control" 1 "captures-absent on a T1-on run (the absence check must see the files)" $CMP "$NEUT" captures-absent --out "$ARM_OUT"
    # the reviewer's end-to-end consistency script of chunk 2a.2, called rather than re-implemented
    local w="$WORK/run/n2_consistency_np$np"; rm -rf "$w"; mkdir -p "$w"
    ( cd "$w" && $PYTHON "$TEST_SRC/test/test_t1_capture_consistency.py" "$TEST_SRC" "$(exe cpu_branch)" "$np" $( [ "$np" = 1 ] && echo full || echo quick ) ) > "$LOGS/N2_consistency_np$np.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && grep -q "RESULT PASS" "$LOGS/N2_consistency_np$np.log" && rec "N2/cpu_np$np/t1-consistency" PASS "test_t1_capture_consistency.py $( [ "$np" = 1 ] && echo full || echo quick ) at np $np (exit $rc)" || rec "N2/cpu_np$np/t1-consistency" FAIL "test_t1_capture_consistency.py exit $rc"
  done
}

# ------------------------------------------------------------------ N3
case_N3() {
  need_cpu || return 1
  build cpu_tu "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_NO_SEED" "" on
  build cpu_tc "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_NO_TAIL" "" on
  local np
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "" || continue
    off_band cpu "$np" 0 || continue
    branch_run "n3_surf_np$np" cpu "$np" $CAP_SURF || continue
    local on=$ARM_OUT
    cmp_arm "N3/cpu_np$np" 0 "T2, T5, T6 on, CPU np$np: hydraulics against the baseline" "$BASE_OUT" "$on"
    chk "N3/cpu_np$np/files" 0 "every surface capture file exists with 16 + records x record size bytes (np$np)" $CMP "$NEUT" captures-check --out "$on" --enabled t2,t5,t6 --ranks "$np" --end-time "$SIM"
    chk "N3/cpu_np$np/perf" 0 "per-rank differential against the capture-off band, DIAG > 0 on every rank (np$np)" \
      $CMP "$NEUT" perf-diff --on "$on" --off $OFFS --cols Compute,MPI,IO,Resize,SWMM,SWMM_XFER,SWMM_MPI,SWMM_STEP,SWMM_OTHER,SWMM_SOLVE,SWMM_LOG,Other --abs-floor "$SEC_FLOOR" --diag-positive all
    chk "N3/cpu_np$np/first-balance" 0 "row 1's delta stored volume equals S(t_end) - S(0) from the t=0 and first-print H rasters (np$np)" $CMP "$NEUT" first-balance --out "$on" --init-h "$CASE_SRC/init_h.dem" --cell "$CELL"
    if [ "$np" = 1 ]; then
      # T-U: the unseeded build must fail the first-balance check
      run_arm "n3_tu_np1" cpu_tu 1 $CAP_SURF && chk "N3/T-U" 1 "first-balance on a T-U (no seed) build" $CMP "$NEUT" first-balance --out "$ARM_OUT" --init-h "$CASE_SRC/init_h.dem" --cell "$CELL"
    fi
  done
  # fixed-time-increment arm: T6 writes its header and no record
  local fixed="time_increment_fixed=1 time_step=0.1 sim_duration=300"
  base_run cpu 1 "_fixed" $fixed && local bf=$BASE_OUT
  branch_run "n3_fixed" cpu 1 $CAP_SURF $fixed && {
    [ -n "${bf:-}" ] && cmp_arm "N3/fixed" 0 "fixed time increment 0.1 s, T2/T5/T6 on: hydraulics against the baseline at the same increment" "$bf" "$ARM_OUT"
    chk "N3/fixed/files" 0 "dt_limiter.bin is the 16-byte header alone and edge_flux.bin holds records" $CMP "$NEUT" captures-check --out "$ARM_OUT" --enabled t2,t5,t6 --ranks 1 --fixed-dt
  }
  # half-interval arm: sim_duration = a whole number of print intervals plus half of one
  local full=$(( NCP / 2 )); local half=$(( full * ${PRINT%.*} + ${PRINT%.*} / 2 ))
  branch_run "n3_half_clean" cpu 1 $CAP_SURF sim_duration=$half && {
    local clean_rows; clean_rows=$($CMP "$NEUT" count-rows --file "$ARM_OUT/domain_balance.csv" | sed -n 's/^ROWS //p')
    local want_rows=$((full + 1))
    [ "$clean_rows" = "$want_rows" ] && rec "N3/half/clean" PASS "half-interval arm (sim_duration $half): $clean_rows domain_balance rows = $full whole intervals + the partial one" || rec "N3/half/clean" FAIL "half-interval arm has $clean_rows rows, expected $want_rows"
    run_arm "n3_half_tc" cpu_tc 1 $CAP_SURF sim_duration=$half && {
      local tc_rows; tc_rows=$($CMP "$NEUT" count-rows --file "$ARM_OUT/domain_balance.csv" | sed -n 's/^ROWS //p')
      [ "$tc_rows" = "$want_rows" ] && rec "N3/T-C" BLIND "T-C build still wrote $tc_rows rows (the partial interval was written: the break is not detected)" || rec "N3/T-C" PASS "must-fail arm failed as it must: the T-C build wrote $tc_rows rows against the clean arm's $clean_rows (one row fewer)"
    }
  }
}

# ------------------------------------------------------------------ N5 and N12 helpers: resume and kill arms
# resume_suite_arms TAG BACKEND_BUILD_SET NP EXTRA_KEYS : arms (a) (b) (c) (e) on the branch; prints through rec
n5_core() { # n5_core TAG NP KEYS...   (branch build)
  local tag=$1 np=$2; shift 2
  branch_run "${tag}_a_clean" cpu "$np" $CAP_ALL swmm_snapshot_keep_all=1 "$@" || return 1
  N5_CLEAN_DIR=$ARM_DIR; N5_CLEAN=$ARM_OUT
  chk "${tag}/a/files" 0 "(a) clean run: every capture file exists with 16 + records x record size bytes" $CMP "$NEUT" captures-check --out "$N5_CLEAN" --enabled all --ranks "$np"
  chk "${tag}/a/coverage" 0 "(a) clean run exercised extbc_m3, swmm_exchange_m3, east and north edge flux, a non-converged T1a block and S(0) > 0" $CMP "$NEUT" capture-coverage --out "$N5_CLEAN"
  resume_arm "$N5_CLEAN_DIR" "${tag}_b_snapshot" cpu_branch "$np" "$CK" && cmp_arm "${tag}/b" 0 "(b) resume from checkpoint $CK, snapshot route, equals the clean run (hydraulics and every capture file)" "$N5_CLEAN" "$ARM_OUT" "$ALLFILES"
  grep -aq "path=snapshot" "$ARM_DIR/res.log" && rec "${tag}/b/route" PASS "(b) took the snapshot route" || rec "${tag}/b/route" FAIL "(b) did not take the snapshot route (see $ARM_DIR/res.log)"
  resume_arm "$N5_CLEAN_DIR" "${tag}_c_replay" cpu_branch "$np" "$CK" swmm_snapshot_disable=1 && cmp_arm "${tag}/c" 0 "(c) resume from checkpoint $CK, replay route, equals the clean run" "$N5_CLEAN" "$ARM_OUT" "$ALLFILES"
  grep -aq "path=replay" "$ARM_DIR/res.log" && rec "${tag}/c/route" PASS "(c) took the replay route" || rec "${tag}/c/route" FAIL "(c) did not take the replay route"
}

# ------------------------------------------------------------------ N5
case_N5() {
  need_cpu || return 1
  build cpu_tp "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_PAUSE_AFTER_FLUSH" "" on
  build cpu_tl "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_LATE_INTERVAL" "" on
  build cpu_tltp "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_LATE_INTERVAL -DTRITON_DIAG_TEST_PAUSE_AFTER_FLUSH" "" on
  build cpu_tu "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_NO_SEED" "" on
  build nos_tp "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_PAUSE_AFTER_FLUSH" "" off
  local np tag
  for np in 1 "$NP_BIG"; do
    tag="N5_np$np"
    n5_core "$tag" "$np" || continue
    local clean=$N5_CLEAN clean_dir=$N5_CLEAN_DIR
    local nnodes; nnodes=$(wc -l < "$clean/swmm_node_diag.nodes")
    # (d) T-P build, pause at after_cfg, SIGKILL inside the pause, resume from checkpoint CK
    if kill_run "${tag}_d_kill" cpu_tp "$np" cfgsnap "$CK" $CAP_ALL swmm_snapshot_keep_all=1 diag_test_pause_id=$CK diag_test_pause_s=600 diag_test_pause_at=after_cfg; then
      local kd=$ARM_DIR
      resume_arm "$kd" "${tag}_d_resume" cpu_branch "$np" "$CK" && cmp_arm "${tag}/d" 0 "(d) killed inside the after_cfg pause, resumed from checkpoint $CK, equals the clean run" "$clean" "$ARM_OUT" "$ALLFILES"
    fi
    # (e) SIGKILL mid-interval after checkpoint CK
    if kill_run "${tag}_e_kill" cpu_branch "$np" cfg "$CK" $CAP_ALL swmm_snapshot_keep_all=1; then
      local ke=$ARM_DIR
      resume_arm "$ke" "${tag}_e_resume" cpu_branch "$np" "$CK" && cmp_arm "${tag}/e" 0 "(e) killed mid-interval after checkpoint $CK, resumed from it, equals the clean run" "$clean" "$ARM_OUT" "$ALLFILES"
    fi
    # (f) T-P build, pause at flush_to_cfg (between the capture flush and the cfg), SIGKILL; resume from the last cfg
    local t1b_bytes=$(( 16 + CK * (8 + 20 * nnodes) ))
    if kill_run "${tag}_f_kill" cpu_tp "$np" size "swmm_node_path3.bin $t1b_bytes" $CAP_ALL swmm_snapshot_keep_all=1 diag_test_pause_id=$CK diag_test_pause_s=600 diag_test_pause_at=flush_to_cfg; then
      local kf=$ARM_DIR lc; lc=$(last_cfg "$kf/out")
      if [ -f "$kf/out/cfg/config_$CK.cfg" ]; then rec "${tag}/f/state" FAIL "(f) config_$CK.cfg exists: the kill did not land between the flush and the cfg"; else rec "${tag}/f/state" PASS "(f) killed between the capture flush and the cfg (config_$CK.cfg absent, last cfg $lc)"; fi
      resume_arm "$kf" "${tag}_f_resume" cpu_branch "$np" "$lc" && cmp_arm "${tag}/f" 0 "(f) killed between flush and cfg, resumed from checkpoint $lc, equals the clean run" "$clean" "$ARM_OUT" "$ALLFILES"
    fi
    # controls and mutants
    build cpu_tl "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_LATE_INTERVAL" "" on
    # the clean arm of the T-L build is run with that build
    run_arm "${tag}_tl_clean" cpu_tl "$np" $CAP_ALL swmm_snapshot_keep_all=1 && {
      local tlc=$ARM_OUT tld=$ARM_DIR
      resume_arm "$tld" "${tag}_tl_b" cpu_tl "$np" "$CK" && cmp_arm "${tag}/T-L/b-control" 0 "control: (b) on a T-L build passes (the finished run holds the late row)" "$tlc" "$ARM_OUT" "$ALLFILES"
      if kill_run "${tag}_tltp_kill" cpu_tltp "$np" cfgsnap "$CK" $CAP_ALL swmm_snapshot_keep_all=1 diag_test_pause_id=$CK diag_test_pause_s=600 diag_test_pause_at=after_cfg; then
        resume_arm "$ARM_DIR" "${tag}_tltp_resume" cpu_tl "$np" "$CK" && cmp_arm "${tag}/T-L/d-mutant" 1 "mutant: (d) on a T-L+T-P build at after_cfg (the late row is never written before the kill)" "$tlc" "$ARM_OUT" "$ALLFILES"
      fi
    }
    resume_arm "$clean_dir" "${tag}_tu_b" cpu_tu "$np" "$CK" && cmp_arm "${tag}/T-U/b-mutant" 1 "mutant: (b) resume on a T-U build (the first row after the resume reads S(t_end) - 0)" "$clean" "$ARM_OUT" "$ALLFILES"
  done
  # (g) the coupled build on the uncoupled paraboloid cfg (the coupling guard false); (h) a build without TRITON_SWMM
  for np in 1 "$NP_BIG"; do
    n5_paraboloid g cpu_tp "$np"
    n5_paraboloid h nos_tp "$np"
  done
}

# n5_paraboloid LABEL BUILD NP : clean arm and arm (f) of the uncoupled paraboloid case with T2, T5, T6 on
n5_paraboloid() {
  local lab=$1 b=$2 np=$3 tag="N5_np${3}_$1"
  [ "$BUILD_ONLY" = 1 ] && { rec "RUN/$tag" SKIP "--build-only"; return 1; }
  local d="$WORK/run/${tag}_clean"; rm -rf "$d"; mkdir -p "$d/test" "$d/input"
  cp -r "$TEST_SRC/input/paraboloid" "$d/input/"; cp "$TEST_SRC/test/paraboloid.cfg" "$d/test/"
  ln -sf "$(exe "$b")" "$d/triton.exe"
  local cfg="$d/test/paraboloid.cfg"
  set_key "$cfg" output_folder out; local kv; for kv in $CAP_SURF; do set_key "$cfg" "${kv%%=*}" "${kv#*=}"; done
  ( cd "$d/test" && $(launcher "$np") ../triton.exe ./paraboloid.cfg > run.log 2>&1 ); local rc=$?
  if [ $rc -ne 0 ] || ! grep -aq "Simulation ends" "$d/test/run.log"; then rec "RUN/$tag" ERROR "paraboloid clean arm exit $rc (see $d/test/run.log)"; return 1; fi
  local clean="$d/test/out"
  chk "${tag}/clean/files" 0 "($lab) paraboloid clean arm: every T2/T5/T6 file exists with whole records" $CMP "$NEUT" captures-check --out "$clean" --enabled t2,t5,t6 --ranks "$np"
  # arm (f): T-P at flush_to_cfg, kill, resume from the last cfg
  local k=6 df="$WORK/run/${tag}_f"; rm -rf "$df"; mkdir -p "$df/test" "$df/input"
  cp -r "$TEST_SRC/input/paraboloid" "$df/input/"; cp "$TEST_SRC/test/paraboloid.cfg" "$df/test/"; ln -sf "$(exe "$b")" "$df/triton.exe"
  local cf="$df/test/paraboloid.cfg"; set_key "$cf" output_folder out
  for kv in $CAP_SURF diag_test_pause_id=$k diag_test_pause_s=600 diag_test_pause_at=flush_to_cfg; do set_key "$cf" "${kv%%=*}" "${kv#*=}"; done
  ( cd "$df/test" && exec setsid $(launcher "$np") ../triton.exe ./paraboloid.cfg > run.log 2>&1 ) &
  local bg=$! t=0 pg ok=0; sleep 0.5; pg=$(ps -o pgid= -p "$bg" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && [ "$pg" = "$bg" ] || { kill -9 "$bg" 2>/dev/null; rec "RUN/${tag}_f" ERROR "the run is not the leader of its own process group"; return 1; }
  while [ $t -lt 6000 ]; do
    [ -f "$df/test/out/domain_balance.csv" ] && [ "$(grep -vc '^%' "$df/test/out/domain_balance.csv")" -ge "$k" ] && { ok=1; break; }
    kill -0 "$bg" 2>/dev/null || break
    sleep 0.05; t=$((t+1))
  done
  [ $ok = 1 ] && sleep 2
  local alive=0; kill -0 "$bg" 2>/dev/null && alive=1
  kill -9 -- "-$pg" 2>/dev/null; kill -9 "$bg" 2>/dev/null; wait "$bg" 2>/dev/null
  if [ $ok = 0 ] || [ $alive = 0 ]; then rec "RUN/${tag}_f" ERROR "kill state not reached (state $ok, alive $alive)"; return 1; fi
  local lc; lc=$(last_cfg "$df/test/out")
  [ -n "$lc" ] || { rec "RUN/${tag}_f" ERROR "no checkpoint cfg written before the kill"; return 1; }
  rm -rf "$df/test/out_res"; cp -r "$df/test/out" "$df/test/out_res"
  sed -i "s|^output_folder=.*|output_folder=out_res|" "$df/test/out_res/cfg/config_$lc.cfg"; sed -i '/^diag_test_pause/d' "$df/test/out_res/cfg/config_$lc.cfg"
  ( cd "$df/test" && $(launcher "$np") ../triton.exe ./out_res/cfg/config_$lc.cfg > res.log 2>&1 ); rc=$?
  if [ $rc -ne 0 ] || ! grep -aq "Simulation ends" "$df/test/res.log"; then rec "RUN/${tag}_f" ERROR "resume exit $rc (see $df/test/res.log)"; return 1; fi
  HYD=1 chk "${tag}/f" 0 "($lab) paraboloid arm (f): killed between flush and cfg, resumed from checkpoint $lc, equals the clean arm (T2/T5/T6 files and the ASC rasters)" \
    $CMP "$NEUT" compare --a "$clean" --b "$df/test/out_res" --captures "$SURFFILES" --no-swmm --raster-dir asc
}

# ------------------------------------------------------------------ N6
case_N6() {
  [ $GPU_OK = 1 ] || { rec "N6" SKIP "no --cmake-flags-from (GPU flags): the A100 arms did not run"; return 0; }
  build gpu_base "$BASE_SRC" gpu "" "" on; build gpu_branch "$TEST_SRC" gpu "" "" on
  build gpu_ts "$TEST_SRC" gpu "-DTRITON_DIAG_TEST_SKEW" "" on
  build gpu_tets "$TEST_SRC" gpu "-DTRITON_DIAG_TEST_SKEW -DTRITON_DIAG_TEST_EARLY_MINLOC" "" on
  local np gd
  local nps="1 2 $NP_BIG"; [ "$N6_PART" = small ] && nps="1 2"; [ "$N6_PART" = big ] && nps="$NP_BIG"
  for np in $nps; do
    for gd in 0 1; do
      [ "$np" = 1 ] && [ "$gd" = 1 ] && continue
      local suf="gpu_np${np}_gd${gd}"
      base_run gpu "$np" "_gd$gd" gpu_direct_flag=$gd || continue
      local bo=$BASE_OUT
      off_band gpu "$np" "$gd" || continue
      branch_run "n6_all_$suf" gpu "$np" $CAP_ALL gpu_direct_flag=$gd || continue
      cmp_arm "N6/$suf" 0 "A100 np$np gpu_aware_mpi=$gd, all captures on: hydraulics against the baseline" "$bo" "$ARM_OUT"
      chk "N6/$suf/files" 0 "every capture file exists with whole records" $CMP "$NEUT" captures-check --out "$ARM_OUT" --enabled all --ranks "$np"
      chk "N6/$suf/perf" 0 "per-rank differential against the capture-off band, DIAG > 0 on rank 0" \
        $CMP "$NEUT" perf-diff --on "$ARM_OUT" --off $OFFS --cols Compute,MPI,IO,Resize,SWMM,SWMM_SOLVE,SWMM_LOG,Other --abs-floor "$SEC_FLOOR" --diag-positive 0
    done
  done
  [ "$N6_PART" = small ] && return 0
  # np8, gpu_aware_mpi on, T6 only: clean+T-S twice, T-E+T-S mutant
  local gd=1 skew
  local nsteps; nsteps=$(python3 -c "import struct,sys; b=open('$OFF1/swmm/hydraulics_exchange_replay.bin','rb').read(); m,n,w=struct.unpack_from('<3i',b,0); print((len(b)-12)//(w*(1+n)))")
  skew=$(python3 -c "import math; print(int(math.ceil(10*$SEC_FLOOR*1e6/$nsteps)))")
  say "N6: $nsteps steps; diag_test_skew_us = $skew (ten times the band floor over the run)"
  run_arm "n6_ts_1" gpu_ts "$NP_BIG" diag_dt_limiter=1 gpu_direct_flag=$gd diag_test_skew_us=$skew && local ts1=$ARM_OUT
  run_arm "n6_ts_2" gpu_ts "$NP_BIG" diag_dt_limiter=1 gpu_direct_flag=$gd diag_test_skew_us=$skew && local ts2=$ARM_OUT
  run_arm "n6_ts_3" gpu_ts "$NP_BIG" diag_dt_limiter=1 gpu_direct_flag=$gd diag_test_skew_us=$skew && local ts3=$ARM_OUT
  [ -n "${ts1:-}" ] && [ -n "${ts2:-}" ] && [ -n "${ts3:-}" ] && chk "N6/T-S/clean" 0 "clean+T-S arm: DIAG of ranks 1-$((NP_BIG-1)) within the band of the two earlier clean+T-S runs" \
    $CMP "$NEUT" perf-diff --on "$ts3" --off "$ts1" "$ts2" --cols DIAG --ranks "1-$((NP_BIG-1))" --abs-floor "$SEC_FLOOR"
  run_arm "n6_tets" gpu_tets "$NP_BIG" diag_dt_limiter=1 gpu_direct_flag=$gd diag_test_skew_us=$skew && [ -n "${ts1:-}" ] && [ -n "${ts2:-}" ] && \
    chk "N6/T-E" 1 "T-E+T-S mutant: DIAG of ranks 1-$((NP_BIG-1)) rises with the skew (the collective issued before MPI_MIN)" \
      $CMP "$NEUT" perf-diff --on "$ARM_OUT" --off "$ts1" "$ts2" --cols DIAG --ranks "1-$((NP_BIG-1))" --abs-floor "$SEC_FLOOR"
}

# ------------------------------------------------------------------ N7
set_threads() { # set_threads DIR N : SWMM THREADS option of the case's .inp
  sed -i '/^THREADS /d' "$1/swmm/hydraulics.inp"; sed -i "/^\[OPTIONS\]/a THREADS               $2" "$1/swmm/hydraulics.inp"
}
run_threads() { # run_threads NAME BUILD THREADS KEYS... -> ARM_OUT
  local name=$1 b=$2 th=$3; shift 3
  [ "$BUILD_ONLY" = 1 ] && { rec "RUN/$name" SKIP "--build-only"; return 1; }
  local d; d=$(rundir "$name" "$b"); set_threads "$d" "$th"
  mkcfg "$d" arm.cfg out "$@"
  ARM_DIR=$d; ARM_OUT="$d/out"
  ( cd "$d" && ./build/triton.exe arm.cfg > run.log 2>&1 ); local rc=$?
  if [ $rc -ne 0 ] || ! grep -aq "Simulation ends" "$d/run.log"; then rec "RUN/$name" ERROR "exit $rc (see $d/run.log)"; return 1; fi
}
case_N7() {
  need_cpu || return 1
  build cpu_tr "$TEST_SRC" cpu "" "-DTRITON_DIAG_TEST_OMP_REDUCTION" on
  local th
  for th in 1 8; do
    run_threads "n7_base_t$th" cpu_base "$th" && local bo=$ARM_OUT
    run_threads "n7_off1_t$th" cpu_branch "$th"; local o1=$ARM_OUT
    run_threads "n7_off2_t$th" cpu_branch "$th"; local o2=$ARM_OUT
    run_threads "n7_on_t$th" cpu_branch "$th" $CAP_ALL && {
      eval "N7_ON_$th=$ARM_OUT"
      [ -n "${bo:-}" ] && cmp_arm "N7/threads$th" 0 "all captures on, SWMM THREADS $th: hydraulics against the baseline at THREADS $th" "$bo" "$ARM_OUT"
      chk "N7/threads$th/files" 0 "every capture file exists with whole records (THREADS $th)" $CMP "$NEUT" captures-check --out "$ARM_OUT" --enabled all --ranks 1 --end-time "$SIM"
      chk "N7/threads$th/perf" 0 "per-rank differential against the capture-off band (THREADS $th)" \
        $CMP "$NEUT" perf-diff --on "$ARM_OUT" --off "$o1" "$o2" --cols Compute,MPI,IO,Resize,SWMM,SWMM_SOLVE,SWMM_LOG,Other --abs-floor "$SEC_FLOOR" --diag-positive 0
    }
    run_threads "n7_tr_t$th" cpu_tr "$th" $CAP_ALL && eval "N7_TR_$th=$ARM_OUT"
  done
  [ -n "${N7_ON_1:-}" ] && [ -n "${N7_ON_8:-}" ] && cmp_arm "N7/1-vs-8" 0 "capture files and hydraulics of THREADS 1 and THREADS 8 equal each other (branch)" "$N7_ON_1" "$N7_ON_8" "$ALLFILES"
  [ -n "${N7_TR_1:-}" ] && [ -n "${N7_TR_8:-}" ] && cmp_arm "N7/T-R" 1 "T-R build: the capture files of THREADS 1 and THREADS 8 differ (domain sums in thread order)" "$N7_TR_1" "$N7_TR_8" "swmm_step_diag.bin"
}

# ------------------------------------------------------------------ N9 (the developer's result, not a gate)
case_N9() {
  need_cpu || return 1
  build cpu_o3 "$TEST_SRC" cpu "" "-O3" on
  branch_run "n9_cpu_default" cpu 1; local d1=$ARM_OUT
  run_arm "n9_cpu_o3" cpu_o3 1 && {
    $CMP "$NEUT" compare --a "$d1" --b "$ARM_OUT" > "$LOGS/N9_cpu.log" 2>&1; local rc=$?
    [ $rc -eq 0 ] && rec "N9/cpu" PASS "DEVELOPER RESULT: -O3 and default C builds, all captures off, CPU serial: byte-identical" || rec "N9/cpu" REPORT "DEVELOPER RESULT: -O3 and default C builds differ (compare exit $rc; see $LOGS/N9_cpu.log): the developer rules on the use of -O3"
  }
  if [ $GPU_OK = 1 ]; then
    build gpu_branch "$TEST_SRC" gpu "" "" on; build gpu_o3 "$TEST_SRC" gpu "" "-O3" on
    branch_run "n9_gpu_default" gpu 1; local g1=$ARM_OUT
    run_arm "n9_gpu_o3" gpu_o3 1 && {
      $CMP "$NEUT" compare --a "$g1" --b "$ARM_OUT" > "$LOGS/N9_gpu.log" 2>&1; local rc=$?
      [ $rc -eq 0 ] && rec "N9/gpu" PASS "DEVELOPER RESULT: -O3 and default C builds, A100 np1: byte-identical" || rec "N9/gpu" REPORT "DEVELOPER RESULT: -O3 and default C builds differ on A100 np1 (compare exit $rc; see $LOGS/N9_gpu.log)"
    }
  else rec "N9/gpu" SKIP "no --cmake-flags-from: the A100 arm did not run"; fi
}

# ------------------------------------------------------------------ N10
case_N10() {
  need_cpu || return 1
  build cpu_tb "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_BUSYWAIT" "" on
  build cpu_tmtb "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_BUSYWAIT -DTRITON_DIAG_TIMER_MUTANT" "" on
  off_band cpu 1 0 || return 1
  local spin; spin=$($CMP "$NEUT" spin-us --offs $OFFS --col Compute --factor 10 --abs-floor "$SEC_FLOOR" --steps-from "$OFF1" | sed -n 's/^SPIN_US //p')
  say "N10: diag_test_busywait_us = $spin"
  run_arm "n10_tb_1" cpu_tb 1 diag_domain_balance=1 diag_test_busywait_us=$spin && local c1=$ARM_OUT
  run_arm "n10_tb_2" cpu_tb 1 diag_domain_balance=1 diag_test_busywait_us=$spin && local c2=$ARM_OUT
  [ -n "${c1:-}" ] && chk "N10/clean" 0 "clean+T-B arm: the spin lands in DIAG, Compute stays inside the capture-off band, DIAG > 0 (run 1)" \
    $CMP "$NEUT" perf-diff --on "$c1" --off $OFFS --cols Compute --abs-floor "$SEC_FLOOR" --diag-positive 0
  [ -n "${c2:-}" ] && chk "N10/clean2" 0 "clean+T-B arm, second run" $CMP "$NEUT" perf-diff --on "$c2" --off $OFFS --cols Compute --abs-floor "$SEC_FLOOR" --diag-positive 0
  run_arm "n10_tmtb" cpu_tmtb 1 diag_domain_balance=1 diag_test_busywait_us=$spin && \
    chk "N10/T-M" 1 "T-M+T-B mutant: the spin runs while COMPUTE_TIME is open, so Compute leaves the band" \
      $CMP "$NEUT" perf-diff --on "$ARM_OUT" --off $OFFS --cols Compute --abs-floor "$SEC_FLOOR"
}

# ------------------------------------------------------------------ N11
case_N11() {
  local tri="$TEST_SRC/src/triton.h"
  chk "N11" 0 "no capture call outside the T1 rows lies inside an #ifdef TRITON_SWMM region or the global_num_of_swmm_links guard (the E-9 add_exchange read is exempt)" $CMP "$NEUT" source-sites --file "$tri"
  chk "N11/fixture-make" 0 "the control: triton.h with the capture flush moved inside the #ifdef TRITON_SWMM block (iteration-6 position)" $CMP "$NEUT" make-n11-fixture --src "$tri" --dst "$WORK/run/triton_n11_fixture.h"
  chk "N11/fixture" 1 "the scan on the fixture copy" $CMP "$NEUT" source-sites --file "$WORK/run/triton_n11_fixture.h"
}

# ------------------------------------------------------------------ N12
case_N12() {
  need_cpu || return 1
  build cpu_td "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_NO_DRAIN" "" on
  build cpu_td2 "$TEST_SRC" cpu "-DTRITON_DIAG_TEST_NO_END_DRAIN" "" on
  local np
  # off: the T3 keys set explicitly to their defaults are byte-identical to the baseline
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "" || continue
    branch_run "n12_defaults_np$np" cpu "$np" swmm_coupling_interval=1 swmm_coupling_lag=0 && \
      cmp_arm "N12/off/np$np" 0 "T3 keys set explicitly to 1/0, CPU np$np: byte-identical to the baseline" "$BASE_OUT" "$ARM_OUT"
  done
  # which interval does the arm use: 4, unless 4 divides the steps run by the resume checkpoint (the accumulator then is zero there)
  base_run cpu 1 "" || return 1
  local steps_log="$LOGS/N12_steps.log"
  $CMP "$NEUT" steps-per-interval --out "$BASE_OUT" --print-interval "${PRINT%.*}" --pick-candidates 4,5,3,7 --interval-index "$CK" > "$steps_log" 2>&1
  local N; N=$(sed -n 's/^PICK //p' "$steps_log")
  [ -n "$N" ] && [ "$N" != None ] || { rec "N12" ERROR "no swmm_coupling_interval candidate leaves a nonzero remainder at checkpoint $CK (see $steps_log)"; return 1; }
  say "N12: $(grep 'steps per print interval' "$steps_log"); $(grep 'steps have run' "$steps_log"); swmm_coupling_interval = $N"
  local mode keys
  for mode in interval lag; do
    [ $mode = interval ] && keys="swmm_coupling_interval=$N" || keys="swmm_coupling_lag=1"
    for np in 1 "$NP_BIG"; do
      local tag="N12_${mode}_np$np"
      branch_run "${tag}_clean1" cpu "$np" $CAP_ALL swmm_snapshot_keep_all=1 $keys || continue
      local c1=$ARM_OUT c1d=$ARM_DIR
      branch_run "${tag}_clean2" cpu "$np" $CAP_ALL swmm_snapshot_keep_all=1 $keys && cmp_arm "N12/on/$mode/np$np/determinism" 0 "T3 on ($keys), CPU np$np: two clean runs equal each other" "$c1" "$ARM_OUT" "$ALLFILES"
      chk "N12/on/$mode/np$np/files" 0 "T3 on ($keys): capture files whole; T1a records equal the exchange side-file's (a writer that ignores stepped_this_step shows here)" $CMP "$NEUT" captures-check --out "$c1" --enabled all --ranks "$np"
      chk "N12/on/$mode/np$np/timers" 0 "SWMM_SOLVE and SWMM_LOG are nonzero on rank 0 in a T3-on run (reviewer finding F-4)" $CMP "$NEUT" perf-nonzero --out "$c1" --cols SWMM_SOLVE,SWMM_LOG --rank 0
      resume_arm "$c1d" "${tag}_b" cpu_branch "$np" "$CK" && cmp_arm "N12/on/$mode/np$np/b" 0 "T3 on, resume arm (b) snapshot route equals its clean run" "$c1" "$ARM_OUT" "$ALLFILES"
      resume_arm "$c1d" "${tag}_c" cpu_branch "$np" "$CK" swmm_snapshot_disable=1 && cmp_arm "N12/on/$mode/np$np/c" 0 "T3 on, resume arm (c) replay route equals its clean run" "$c1" "$ARM_OUT" "$ALLFILES"
      if kill_run "${tag}_e_kill" cpu_branch "$np" cfg "$CK" $CAP_ALL swmm_snapshot_keep_all=1 $keys; then
        resume_arm "$ARM_DIR" "${tag}_e" cpu_branch "$np" "$CK" && cmp_arm "N12/on/$mode/np$np/e" 0 "T3 on, resume arm (e) after a mid-interval kill equals its clean run" "$c1" "$ARM_OUT" "$ALLFILES"
      fi
    done
  done
  # mutant: resume arm (b) on a T-D build with swmm_coupling_interval=N
  run_arm "N12_td_clean" cpu_td 1 $CAP_ALL swmm_snapshot_keep_all=1 swmm_coupling_interval=$N && {
    local tdc=$ARM_OUT tdd=$ARM_DIR
    resume_arm "$tdd" "N12_td_b" cpu_td 1 "$CK" && cmp_arm "N12/T-D" 1 "mutant: resume arm (b) on a T-D build, swmm_coupling_interval=$N (the checkpoint carries a nonzero accumulator)" "$tdc" "$ARM_OUT" "$ALLFILES"
  }
  # end-of-run arms (E-3/E-6): sim_duration = whole intervals + half of one; T3 on, CPU serial
  local half=$(( (NCP / 2) * ${PRINT%.*} + ${PRINT%.*} / 2 )) whole=$(( (NCP / 2) * ${PRINT%.*} ))
  for keys in "swmm_coupling_interval=$N" "swmm_coupling_lag=1"; do
    local lab; lab=$(echo "$keys" | tr '=' '_')
    branch_run "N12_end_$lab" cpu 1 diag_domain_balance=1 $keys sim_duration=$half && chk "N12/end/$lab" 0 "end-of-run arm ($keys, duration $half): the side-file's final cumulative dt equals the run end and SWMM stepped after the last print instant" $CMP "$NEUT" end-drain --out "$ARM_OUT" --sim-duration "$half" --print-interval "${PRINT%.*}"
    run_arm "N12_end_td2_$lab" cpu_td2 1 diag_domain_balance=1 $keys sim_duration=$half && chk "N12/end/$lab/T-D2" 1 "mutant: the same arm on a T-D2 build (no end-of-run drain)" $CMP "$NEUT" end-drain --out "$ARM_OUT" --sim-duration "$half" --print-interval "${PRINT%.*}"
    run_arm "N12_whole_td2_$lab" cpu_td2 1 diag_domain_balance=1 $keys sim_duration=$whole && chk "N12/whole/$lab/T-D2" 0 "control: a whole-interval duration ($whole) passes on a T-D2 build" $CMP "$NEUT" end-drain --out "$ARM_OUT" --sim-duration "$whole" --print-interval "${PRINT%.*}"
  done
}

# ------------------------------------------------------------------ N13
# Per-node manhole table (swmm_manhole_table; packages 1f and 3c; approved shape section 10 row N13).
#
# The fixed case's two constants are manhole_diameter=1.2 and manhole_loss=0.1 (TRITONSWMM.cfg);
# they differ, and the case refuses to run when they are equal (a table that swaps its two columns
# would then equal the constants and pass).
#
# Measured on Rivanna (job 23042105: the faba2232 baseline binary, CPU, this case at np1 and np8;
# the same two nodes are logged by the solver's "SWMM coupled node" lines and read from the [INFLOWS]
# FLOW lines of swmm/hydraulics.inp):
#   coupling nodes, in the solver's global order, 2:
#     J1  global index 0, cell row 31 col 19, owner rank 3 at np8, first nonzero exchange at side-file record 42 of (a)
#     J3  global index 1, cell row 23 col 19, owner rank 2 at np8, first nonzero exchange at side-file record 16 of (a)
#   the np8 row strips are eight rows each (create_local_dims, mpi_utils.h), so rank 3 owns rows 24-31 and rank 2 rows 16-23;
#   the owners are corroborated by the baseline np8 run's performance.txt: SWMM_XFER is above zero on ranks 2 and 3 only.
#   The two nodes sit in two different cells, so each is alone in its TRITON cell.
#   N* = J1 (a coupling node alone in its cell, in a strip of a rank other than 0, with a nonzero exchange in (a)).
#   N' = J3: NO coupling node of this case is owned by rank 0 (rank 0's strip is rows 0-7 and make_case.py keeps it
#   free of manholes), so N' cannot be rank 0 here; the detector's control needs only a node other than N*.
#   (a) at np1 and np8 is NOT byte-identical (the rasters, the side-file and the .out all differ between the two rank
#   counts on the baseline), so the np1-against-np8 readings of (c) and (d) are reported as an instrument unable to
#   detect that break on this case (R4) and the first-record check carries the placement verdict alone.
# The suite re-derives all of this at run time (the select step below) and prints it.
#
# N13 reads the exchange side-file through an embedded helper (written to $WORK at run time)
# because neutrality_compare.py, which chunk 3c.1 may not edit, has no subcommand that names the
# columns of the first differing side-file record.
n13_write_helper() {
  cat > "$WORK/n13_helper.py" <<'PYEOF'
"""N13 helper: coupling nodes, owners and the first differing side-file record's columns."""
import math, os, struct, sys


def die(msg):
    sys.stderr.write(msg + "\n")
    sys.exit(2)


def case_geometry(case):
    dem = open(os.path.join(case, "elevation.dem")).read().split("\n")
    hdr = dict((l.split()[0].lower(), l.split()[1]) for l in dem[:6])
    return int(hdr["nrows"]), float(hdr["xllcorner"]), float(hdr["yllcorner"]), float(hdr["cellsize"])


def inp_coordinates(case):
    out, on = {}, False
    for l in open(os.path.join(case, "swmm", "hydraulics.inp")).read().split("\n"):
        s = l.strip()
        if s.startswith("["):
            on = (s.upper() == "[COORDINATES]")
            continue
        if on and s and not s.startswith(";"):
            f = s.split()
            out[f[0].upper()] = (float(f[1]), float(f[2]))
    return out


def strips(nrows, size):
    return [nrows // size + (1 if (p < nrows % size) else 0) for p in range(size)]


def owner_of(row, nrows, size):
    acc = 0
    for p, n in enumerate(strips(nrows, size)):
        if row < acc + n:
            return p
        acc += n
    return None


def sidefile(path):
    b = open(path, "rb").read()
    m, n, w = struct.unpack_from("<3i", b, 0)
    rec = w * (1 + n)
    return b, n, w, rec, (len(b) - 12) // rec


def first_nonzero(path, j):
    b, n, w, rec, nrec = sidefile(path)
    for k in range(nrec):
        v = struct.unpack_from("<d", b, 12 + k * rec + w * (1 + j))[0]
        if v != 0.0:
            return k
    return None


def cmd_select(argv):
    # select CASE NP SIDEFILE NAME...  (names in the solver's coupling order)
    case, np_, side, names = argv[0], int(argv[1]), argv[2], argv[3:]
    nrows, xll, yll, dx = case_geometry(case)
    coords = inp_coordinates(case)
    b, n, w, rec, nrec = sidefile(side)
    if n != len(names):
        die("side-file has %d exchange columns but the solver logged %d coupling nodes" % (n, len(names)))
    info = []
    for i, nm in enumerate(names):
        x, y = coords[nm.upper()]
        row = int(math.ceil((nrows - ((y - yll) / dx)) + 1e-16)) - 1
        col = int(math.ceil(((x - xll) / dx) + 1e-16)) - 1
        info.append(dict(name=nm, idx=i, row=row, col=col, owner=owner_of(row, nrows, np_), first=first_nonzero(side, i)))
    for d in info:
        sys.stdout.write("NODE %s idx %d row %d col %d owner_np%d %d first_nonzero_record %s\n" % (d["name"], d["idx"], d["row"], d["col"], np_, d["owner"], d["first"]))
    cells = [(d["row"], d["col"]) for d in info]
    cand = [d for d in info if cells.count((d["row"], d["col"])) == 1 and d["owner"] != 0 and d["first"] is not None]
    if not cand:
        die("no coupling node is alone in its cell, owned by a rank other than 0 and exchanging a nonzero flow in (a)")
    star = cand[0]
    others = [d for d in info if d["idx"] != star["idx"]]
    if not others:
        die("the case has one coupling node: no control node N'")
    rank0 = [d for d in others if d["owner"] == 0]
    prime = rank0[0] if rank0 else others[0]
    sys.stdout.write("NSTAR %s %d %d %d\n" % (star["name"], star["idx"], star["owner"], star["first"]))
    sys.stdout.write("NPRIME %s %d %d %s\n" % (prime["name"], prime["idx"], prime["owner"], "rank0" if rank0 else "not_rank0"))
    sys.stdout.write("OWNERS %s\n" % " ".join(str(o) for o in sorted(set(d["owner"] for d in info))))
    return 0


def cmd_firstcols(argv):
    # firstcols A B EXPECT_NODE_INDEX : exit 0 only when the first differing record differs in exactly that node's column
    pa, pb, expect = argv[0], argv[1], int(argv[2])
    ba, na, wa, ra, ka = sidefile(pa)
    bb, nb, wb, rb, kb = sidefile(pb)
    if (na, wa) != (nb, wb):
        die("side-files have different shapes: %s and %s" % ((na, wa), (nb, wb)))
    for k in range(min(ka, kb)):
        ra_ = ba[12 + k * ra: 12 + (k + 1) * ra]
        rb_ = bb[12 + k * ra: 12 + (k + 1) * ra]
        if ra_ != rb_:
            va = struct.unpack("<%dd" % (1 + na), ra_)
            vb = struct.unpack("<%dd" % (1 + na), rb_)
            cols = [("dt" if j == 0 else j - 1) for j in range(1 + na) if ra_[j * wa:(j + 1) * wa] != rb_[j * wa:(j + 1) * wa]]
            sys.stdout.write("first differing record %d of %d and %d; columns that differ: %s; expected: [%d]\n" % (k, ka, kb, cols, expect))
            for c in cols:
                j = 0 if c == "dt" else c + 1
                sys.stdout.write("  column %s: %r against %r\n" % (c, va[j], vb[j]))
            ok = (cols == [expect])
            sys.stdout.write("RESULT %s firstcols\n" % ("PASS" if ok else "FAIL"))
            return 0 if ok else 1
    sys.stdout.write("no record of the first %d differs (sizes %d and %d bytes); expected a difference in column [%d]\n" % (min(ka, kb), len(ba), len(bb), expect))
    sys.stdout.write("RESULT FAIL firstcols\n")
    return 1


def cmd_xfer(argv):
    # xfer OUT : ranks whose SWMM_XFER is above zero (corroborates the owners)
    lines = open(os.path.join(argv[0], "performance.txt")).read().split("\n")
    names = [x.strip() for x in lines[0].lstrip("%").split(",")]
    ranks = []
    for l in lines[1:]:
        if l.strip() and l.replace(",", " ").split()[0].isdigit():   # the per-rank rows; the "Average" row is skipped
            row = dict(zip(names, l.replace(",", " ").split()))
            if float(row["SWMM_XFER"]) > 0.0:
                ranks.append(int(row["Rank"]))
    sys.stdout.write("RANKS_WITH_SWMM_XFER %s\n" % " ".join(str(r) for r in sorted(ranks)))
    return 0


if __name__ == "__main__":
    fn = {"select": cmd_select, "firstcols": cmd_firstcols, "xfer": cmd_xfer}.get(sys.argv[1])
    if fn is None:
        die("unknown subcommand")
    sys.exit(fn(sys.argv[2:]))
PYEOF
}

N13_NODES=(); N13_DIA=""; N13_LOS=""
# n13_table FILE [NAME=DIAMETER,LOSS ...] : one row per coupling node, the case's two constants unless overridden
n13_table() {
  local f=$1; shift
  : > "$f"
  local nm ov row
  for nm in "${N13_NODES[@]}"; do
    row="$nm $N13_DIA $N13_LOS"
    for ov in "$@"; do
      [ "${ov%%=*}" = "$nm" ] && { local v=${ov#*=}; row="$nm ${v%%,*} ${v#*,}"; }
    done
    printf '%s\n' "$row" >> "$f"
  done
}
n13_double() { $PYTHON -I -c "print(repr(2*float('$1')))"; }

# n13_refusal ID TABLEPATH LINE DESC : an np8 run that must stop before the first step, naming the file (and the line when LINE > 0),
# with no rank left running
n13_refusal() {
  local id=$1 tbl=$2 ln=$3 desc=$4
  [ "$BUILD_ONLY" = 1 ] && { rec "$id" SKIP "--build-only"; return 0; }
  local name="n13_ref_$(echo "$id" | tr '/ ' '__')" d rc
  d=$(rundir "$name" cpu_branch)
  mkcfg "$d" arm.cfg out "swmm_manhole_table=$tbl"
  ( cd "$d" && timeout 300 $(launcher "$NP_BIG") ./build/triton.exe arm.cfg > run.log 2>&1 ); rc=$?
  local left=0 pid
  for pid in $(pgrep -u "$(id -u)" -f "[b]uild/triton.exe arm.cfg" 2>/dev/null); do
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$d" ] && { left=$((left+1)); kill -9 "$pid" 2>/dev/null; }
  done
  local rasters; rasters=$(find "$d/out/bin" -name 'H_*' 2>/dev/null | wc -l)
  if [ $rc -eq 0 ] && grep -aq "Simulation ends" "$d/run.log"; then
    rec "$id" BLIND "$desc: the run was NOT refused (exit 0, simulation completed; see $d/run.log): the instrument cannot detect this break"
  elif [ $rc -eq 0 ]; then
    rec "$id" FAIL "$desc: exit 0 without a completed simulation: a refusal must exit non-zero (see $d/run.log)"
  elif [ $rc -eq 124 ] || [ $left -gt 0 ]; then
    rec "$id" FAIL "$desc: exit $rc, $left rank process(es) left running (see $d/run.log)"
  elif grep -aq "Simulation starts" "$d/run.log" || [ "$rasters" -gt 0 ]; then
    rec "$id" FAIL "$desc: exit $rc but the run got past initialisation (Simulation starts printed or $rasters raster file(s) written; see $d/run.log)"
  elif ! grep -aqF "$tbl" "$d/run.log"; then
    rec "$id" FAIL "$desc: exit $rc but stderr does not name the file $tbl (see $d/run.log)"
  elif [ "$ln" -gt 0 ] && ! grep -aqE "line $ln([^0-9]|$)" "$d/run.log"; then
    rec "$id" FAIL "$desc: exit $rc, the file is named but not 'line $ln' (see $d/run.log)"
  else
    rec "$id" PASS "$desc: exit $rc before the first step, the file named$( [ "$ln" -gt 0 ] && echo " with line $ln"), no rank left running"
  fi
}

# n13_must_differ ID DESC A B : the two outputs must differ (a table that changes the hydraulics by design); never scoped out
n13_must_differ() {
  local id=$1 desc=$2 a=$3 b=$4
  $CMP "$NEUT" compare --a "$a" --b "$b" > "$LOGS/$(echo "$id" | tr '/ ()' '____').log" 2>&1; local rc=$?
  if [ $rc -eq 1 ] && grep -aq "DIFFERENCE: swmm/hydraulics_exchange_replay.bin: differs" "$LOGS/$(echo "$id" | tr '/ ()' '____').log"; then rec "$id" PASS "$desc (differs, as it must)"
  elif [ $rc -eq 0 ]; then rec "$id" FAIL "$desc: byte-identical to (a): the key or the column was ignored"
  else rec "$id" ERROR "$desc (compare exit $rc without a side-file difference; see logs)"; fi
}

case_N13() {
  need_cpu || return 1
  N13_DIA=$(sed -n 's/^manhole_diameter=//p' "$CASE_SRC/TRITONSWMM.cfg"); N13_LOS=$(sed -n 's/^manhole_loss=//p' "$CASE_SRC/TRITONSWMM.cfg")
  if [ -z "$N13_DIA" ] || [ -z "$N13_LOS" ] || [ "$($PYTHON -I -c "print(float('$N13_DIA') == float('$N13_LOS'))")" != False ]; then
    rec "N13/constants" ERROR "the case's manhole_diameter ('$N13_DIA') and manhole_loss ('$N13_LOS') are missing or equal: N13 refuses to run (a table with its columns swapped would equal the constants)"
    return 1
  fi
  rec "N13/constants" PASS "the case's two constants differ: manhole_diameter=$N13_DIA, manhole_loss=$N13_LOS"
  n13_write_helper
  local H="$WORK/n13_helper.py" TB="$WORK/n13_tables" np
  mkdir -p "$TB"
  local -A BASEO AO
  local c1="" c8a="" c8b="" d1="" d8=""
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "" || return 1; BASEO[$np]=$BASE_OUT
    branch_run "n13_a_np$np" cpu "$np" || return 1; AO[$np]=$ARM_OUT
    cmp_arm "N13/a/np$np" 0 "(a) swmm_manhole_table absent, CPU np$np: branch against the faba2232 baseline, byte for byte" "${BASEO[$np]}" "${AO[$np]}"
  done
  # the coupling nodes as the solver logged them, in its order
  N13_NODES=($(sed -n 's/.*SWMM coupled node \([^:]*\):.*/\1/p' "${BASEO[1]}/../run.log"))
  [ ${#N13_NODES[@]} -ge 2 ] || { rec "N13/nodes" ERROR "fewer than two 'SWMM coupled node' lines in ${BASEO[1]}/../run.log"; return 1; }
  $PYTHON -I "$H" select "$CASE_SRC" "$NP_BIG" "${BASEO[1]}/swmm/hydraulics_exchange_replay.bin" "${N13_NODES[@]}" > "$LOGS/N13_select.log" 2>&1; local src=$?
  cat "$LOGS/N13_select.log"
  [ $src -eq 0 ] || { rec "N13/select" ERROR "no node N* fits (exit $src; see $LOGS/N13_select.log)"; return 1; }
  local NS_NAME NS_IDX NS_OWN NS_FIRST NP_NAME NP_IDX NP_OWN NP_R0
  read -r _ NS_NAME NS_IDX NS_OWN NS_FIRST <<< "$(grep '^NSTAR ' "$LOGS/N13_select.log")"
  read -r _ NP_NAME NP_IDX NP_OWN NP_R0 <<< "$(grep '^NPRIME ' "$LOGS/N13_select.log")"
  rec "N13/select" PASS "coupling nodes (global order) ${N13_NODES[*]}; N*=$NS_NAME (global index $NS_IDX, owner rank $NS_OWN at np$NP_BIG, first nonzero exchange at side-file record $NS_FIRST of (a)); N'=$NP_NAME (global index $NP_IDX, owner rank $NP_OWN at np$NP_BIG, $( [ "$NP_R0" = rank0 ] && echo "owned by rank 0" || echo "NO coupling node of the case is owned by rank 0: N' is the other node and is not rank 0")); both constants and the two nodes are the comment block above"
  # corroboration of the owners from the baseline np8 run: the ranks that did any SWMM transfer work
  $PYTHON -I "$H" xfer "${BASEO[$NP_BIG]}" > "$LOGS/N13_xfer.log" 2>&1; cat "$LOGS/N13_xfer.log"
  local own_set xfer_set
  own_set=$(sed -n 's/^OWNERS //p' "$LOGS/N13_select.log"); xfer_set=$(sed -n 's/^RANKS_WITH_SWMM_XFER //p' "$LOGS/N13_xfer.log")
  [ -n "$own_set" ] && [ "$own_set" = "$xfer_set" ] && rec "N13/owners" PASS "the owners computed from the row-strip split (ranks $own_set) are the ranks with SWMM_XFER above zero in the baseline np$NP_BIG run" || rec "N13/owners" FAIL "the owners computed from the row-strip split (ranks '$own_set') differ from the ranks with SWMM_XFER above zero ('$xfer_set')"
  # does (a) at np1 equal (a) at np8 byte for byte (R4: when it does not, the np1-against-np8 readings are blind)
  $CMP "$NEUT" compare --a "${AO[1]}" --b "${AO[$NP_BIG]}" > "$LOGS/N13_a_np1_vs_np8.log" 2>&1; local arc=$?
  if [ $arc -eq 0 ]; then rec "N13/a/np1-vs-np$NP_BIG" PASS "(a) at np1 and np$NP_BIG are byte-identical: the np1-against-np$NP_BIG readings of (c) and (d) are read"
  else rec "N13/a/np1-vs-np$NP_BIG" REPORT "(a) at np1 and np$NP_BIG are NOT byte-identical (compare exit $arc; see $LOGS/N13_a_np1_vs_np8.log): the np1-against-np$NP_BIG readings of (c) and (d) are an instrument unable to detect that break on this case (R4), and the first-record check carries N13's placement verdict alone"; fi

  local dbl_d dbl_l; dbl_d=$(n13_double "$N13_DIA"); dbl_l=$(n13_double "$N13_LOS")
  n13_table "$TB/b.txt"
  n13_table "$TB/c.txt" "$NS_NAME=$dbl_d,$N13_LOS"
  n13_table "$TB/d.txt" "$NS_NAME=$N13_DIA,$dbl_l"
  n13_table "$TB/cprime.txt" "$NP_NAME=$dbl_d,$N13_LOS"
  { echo "== N13 tables =="; for f in b c d cprime; do echo "-- $f.txt"; cat "$TB/$f.txt"; done; } | tee -a "$BUILD_RECORD"

  # (b) every coupling node at the constants: byte-identical to (a), plus N5's arm (b) on it
  for np in 1 "$NP_BIG"; do
    branch_run "n13_b_np$np" cpu "$np" swmm_manhole_table="$TB/b.txt" && \
      cmp_arm "N13/b/np$np" 0 "(b) a table listing every coupling node at the constants, CPU np$np: byte-identical to (a)" "${AO[$np]}" "$ARM_OUT"
    # N5 arm (b): resume from checkpoint CK on the snapshot route equals the table's own clean run, every capture file included
    branch_run "n13_b5_np${np}_clean" cpu "$np" $CAP_ALL swmm_snapshot_keep_all=1 swmm_manhole_table="$TB/b.txt" || continue
    local b5=$ARM_OUT b5d=$ARM_DIR
    resume_arm "$b5d" "n13_b5_np${np}_resume" cpu_branch "$np" "$CK" && {
      cmp_arm "N13/b/np$np/N5b" 0 "N5 arm (b) on the table (b): resume from checkpoint $CK, snapshot route, equals its clean run (hydraulics and every capture file)" "$b5" "$ARM_OUT" "$ALLFILES"
      grep -aq "path=snapshot" "$ARM_DIR/res.log" && rec "N13/b/np$np/N5b/route" PASS "the resume took the snapshot route" || rec "N13/b/np$np/N5b/route" FAIL "the resume did not take the snapshot route (see $ARM_DIR/res.log)"
      grep -aqF "swmm_manhole_table" "$ARM_DIR/out_res/cfg/config_$CK.cfg" && rec "N13/b/np$np/N5b/key" PASS "the checkpoint cfg carries swmm_manhole_table (the resumed run reads the table again)" || rec "N13/b/np$np/N5b/key" FAIL "the checkpoint cfg config_$CK.cfg does not carry swmm_manhole_table"
    }
  done

  # (c) N* at twice the constant diameter: serial once, np8 twice
  branch_run "n13_c_np1" cpu 1 swmm_manhole_table="$TB/c.txt" && c1=$ARM_OUT
  branch_run "n13_c_np${NP_BIG}_1" cpu "$NP_BIG" swmm_manhole_table="$TB/c.txt" && c8a=$ARM_OUT
  branch_run "n13_c_np${NP_BIG}_2" cpu "$NP_BIG" swmm_manhole_table="$TB/c.txt" && c8b=$ARM_OUT
  # (d) N*'s loss coefficient doubled, diameter at the constant: serial once, np8 once
  branch_run "n13_d_np1" cpu 1 swmm_manhole_table="$TB/d.txt" && d1=$ARM_OUT
  branch_run "n13_d_np${NP_BIG}" cpu "$NP_BIG" swmm_manhole_table="$TB/d.txt" && d8=$ARM_OUT
  local arm var
  for arm in "c:$c1:1" "c:$c8a:$NP_BIG" "d:$d1:1" "d:$d8:$NP_BIG"; do
    local lab=${arm%%:*} rest=${arm#*:}; local out=${rest%%:*} anp=${rest#*:}
    [ -n "$out" ] || continue
    n13_must_differ "N13/$lab/np$anp/differs" "($lab) N*=$NS_NAME $( [ $lab = c ] && echo "diameter doubled" || echo "loss coefficient doubled"), CPU np$anp, against (a) at np$anp" "${AO[$anp]}" "$out"
    HYD=0 chk "N13/$lab/np$anp/column" 0 "($lab) np$anp against (a): the first differing side-file record differs in N*'s column ($NS_NAME, index $NS_IDX) alone" $PYTHON -I "$H" firstcols "${AO[$anp]}/swmm/hydraulics_exchange_replay.bin" "$out/swmm/hydraulics_exchange_replay.bin" "$NS_IDX"
  done
  [ -n "$c8a" ] && [ -n "$c8b" ] && HYD=0 chk "N13/c/np$NP_BIG/determinism" 0 "(c) two np$NP_BIG runs of the same table are byte-identical" $CMP "$NEUT" compare --a "$c8a" --b "$c8b"
  # np1 against np8, read only where (a) at np1 and (a) at np8 are byte-identical
  local pair
  for pair in "c:$c1:$c8a" "d:$d1:$d8"; do
    local lab=${pair%%:*} r=${pair#*:}; local o1=${r%%:*} o8=${r#*:}
    [ -n "$o1" ] && [ -n "$o8" ] || continue
    if [ $arc -eq 0 ]; then
      HYD=0 chk "N13/$lab/np1-vs-np$NP_BIG" 0 "($lab) np1 against np$NP_BIG: rasters, side-file and .out byte-identical ((a) is byte-identical across the two rank counts)" $CMP "$NEUT" compare --a "$o1" --b "$o8"
    else
      rec "N13/$lab/np1-vs-np$NP_BIG" REPORT "($lab) np1 against np$NP_BIG not read: (a) is not byte-identical across the two rank counts, so the reading is an instrument unable to detect a mis-split value on this case (R4)"
    fi
  done
  # the detector refuses a first differing record that names two columns: a control the real case cannot supply, because its two nodes first exchange at different records
  $PYTHON -I -c 'import struct,sys; d=sys.argv[1]; h=struct.pack("<3i",0x4F4B5832,2,8)
a=[[0.1,0.0,0.0],[0.2,0.0,0.0]]; b=[[0.1,0.0,0.0],[0.2,5.0,5.0]]
for n,r in (("two_a.bin",a),("two_b.bin",b)): open(d+"/"+n,"wb").write(h+b"".join(struct.pack("<3d",*x) for x in r))' "$TB"
  HYD=0 chk "N13/detector/two-columns" 1 "(control) the detector expecting column 0 on a first differing record that differs in columns 0 and 1: must FAIL" $PYTHON -I "$H" firstcols "$TB/two_a.bin" "$TB/two_b.bin" 0
  # (c') a different node's diameter doubled, serial: the detector expecting N* must FAIL, and its control (expecting N') must pass
  branch_run "n13_cprime_np1" cpu 1 swmm_manhole_table="$TB/cprime.txt" && {
    n13_must_differ "N13/cprime/np1/differs" "(c') N'=$NP_NAME diameter doubled, CPU np1, against (a)" "${AO[1]}" "$ARM_OUT"
    HYD=0 chk "N13/cprime/np1/detector" 1 "(c') the detector with N*=$NS_NAME expected on a run whose first differing record names N'=$NP_NAME: must FAIL" $PYTHON -I "$H" firstcols "${AO[1]}/swmm/hydraulics_exchange_replay.bin" "$ARM_OUT/swmm/hydraulics_exchange_replay.bin" "$NS_IDX"
    HYD=0 chk "N13/cprime/np1/detector-control" 0 "(c') the same detector expecting N'=$NP_NAME passes (it names the right node on known input)" $PYTHON -I "$H" firstcols "${AO[1]}/swmm/hydraulics_exchange_replay.bin" "$ARM_OUT/swmm/hydraulics_exchange_replay.bin" "$NP_IDX"
  }

  # refusals at np8: each must stop before the first step, name the file (and the line), leave no rank running
  local first=${N13_NODES[0]} rest_rows
  { echo "$first $N13_DIA $N13_LOS"; } > "$TB/r_missing.txt"
  n13_table "$TB/r_unknown.txt"; echo "ZZ_NOT_A_NODE $N13_DIA $N13_LOS" >> "$TB/r_unknown.txt"
  n13_table "$TB/r_repeat.txt"; echo "$(echo "$first" | tr 'A-Z' 'a-z') $N13_DIA $N13_LOS" >> "$TB/r_repeat.txt"
  n13_table "$TB/r_zerodia.txt" "$first=0,$N13_LOS"
  n13_table "$TB/r_negloss.txt" "$first=$N13_DIA,-$N13_LOS"
  n13_table "$TB/r_nolosscol.txt"; sed -i "1s/.*/$first $N13_DIA/" "$TB/r_nolosscol.txt"
  local nrows=${#N13_NODES[@]}
  n13_refusal "N13/refuse/missing-row" "$TB/r_missing.txt" 0 "a table with no row for coupling node(s) ${N13_NODES[*]:1}"
  n13_refusal "N13/refuse/unknown-node" "$TB/r_unknown.txt" $((nrows + 1)) "a row naming no coupling node (line $((nrows + 1)))"
  n13_refusal "N13/refuse/repeated-name" "$TB/r_repeat.txt" $((nrows + 1)) "a name repeated in another case (line $((nrows + 1)))"
  n13_refusal "N13/refuse/zero-diameter" "$TB/r_zerodia.txt" 1 "a zero diameter (line 1)"
  n13_refusal "N13/refuse/negative-loss" "$TB/r_negloss.txt" 1 "a negative loss coefficient (line 1)"
  n13_refusal "N13/refuse/no-loss-column" "$TB/r_nolosscol.txt" 1 "a row with a diameter and no loss coefficient (line 1)"
  n13_refusal "N13/refuse/unreadable" "$TB/does_not_exist.txt" 0 "an unreadable path"
}

# ------------------------------------------------------------------ N14 and N15: the driven-stage outfall and the DEM-rim exemption
# N14 runs on its own case directory (test/coupled_case_outfall/, built by its make_case.py, read by
# its outfall_checks.py); N15 runs on test/coupled_case/ with swmm_node_rim_from_dem=1. The test-only
# breaks are SOURCE MUTANTS: a copy of the test tree with one statement changed, built like the branch
# (the solver carries no TRITON_DIAG_TEST define for these features, and src/ is not this suite's to
# edit); each mutant's diff is recorded in the build record, and each must make the arm it targets fail.
OFL_SRC="$TEST_SRC/test/coupled_case_outfall"; OFL_CK="$OFL_SRC/outfall_checks.py"
FSTOL=${NEUTRALITY_STAGE_TOL:-1e-12}   # arm (f)'s relative tolerance: 1e-12 for a double build, 1e-6 for a single build

# ofl_differ A B : exit 0 when a hydraulic output (a raster, the side-file, the .out, the .rpt block) differs between the two
# outputs, 1 when they are byte-identical, 2 when the comparison itself did not work (a missing file is not a difference)
ofl_differ() {
  local lg="$LOGS/ofl_differ_$$.tmp"
  $CMP "$NEUT" compare --a "$1" --b "$2" > "$lg" 2>&1; local rc=$?
  cat "$lg"
  if [ $rc -eq 0 ]; then rm -f "$lg"; return 1; fi
  if [ $rc -eq 1 ] && grep -aq "DIFFERENCE: .*differs (first differing byte" "$lg"; then rm -f "$lg"; return 0; fi
  rm -f "$lg"; return 2
}

# mutate NAME FILE REGEX REPLACEMENT : a copy of the test tree with exactly one match of REGEX in FILE replaced (backreferences
# allowed); the unified diff goes to the build record
mutate() {
  local name=$1 file=$2 old=$3 new=$4 d="$WORK/src/mut_$1"
  [ -d "$d" ] || cp -r "$TEST_SRC" "$d"
  $PYTHON -I - "$d/$file" "$old" "$new" > "$LOGS/mutate_$name.log" 2>&1 <<'PYEOF'
import re, sys
p, old, new = sys.argv[1:4]
t = open(p).read()
n = len(re.findall(old, t))
if n != 1:
    sys.exit("the mutation matches %d times in %s (it must match exactly once): %s" % (n, p, old))
open(p, "w").write(re.sub(old, new, t))
PYEOF
  local rc=$?
  { echo "== source mutant $name: $file =="; diff -u "$TEST_SRC/$file" "$d/$file"; } >> "$BUILD_RECORD"
  return $rc
}
# build_mut NAME FILE REGEX REPLACEMENT : mutate, then build the mutant like the branch (CPU, SWMM on)
build_mut() {
  mutate "$1" "$2" "$3" "$4" || { rec "BUILD/$1" ERROR "the source mutation did not apply (see $LOGS/mutate_$1.log)"; return 1; }
  build "$1" "$WORK/src/mut_$1" cpu "" "" on
}

# run_arm_edit EDIT NAME BUILD NP KEYS... : run_arm on a run directory that the function EDIT (called with the directory) has edited;
# returns 1 when the run did not complete, so a refusal arm can classify it, and records nothing itself
run_arm_edit() {
  local edit=$1 name=$2 b=$3 np=$4; shift 4
  [ "$BUILD_ONLY" = 1 ] && { rec "RUN/$name" SKIP "--build-only"; return 2; }
  local d; d=$(rundir "$name" "$b")
  "$edit" "$d" > "$LOGS/edit_$name.log" 2>&1 || { rec "RUN/$name" ERROR "the run directory edit $edit failed (see $LOGS/edit_$name.log)"; return 2; }
  mkcfg "$d" arm.cfg out "$@"
  ARM_DIR=$d; ARM_OUT="$d/out"
  ( cd "$d" && $(launcher "$np") ./build/triton.exe arm.cfg > run.log 2>&1 ); local rc=$?
  [ $rc -eq 0 ] && grep -aq "Simulation ends" "$d/run.log" && return 0
  return 1
}
no_edit() { return 0; }

# refusal_arm ID DESC NEEDLE EDIT KEYS... : an np8 run that must stop before the first step, with NEEDLE in its log, no raster
# written and no rank left running; EDIT is a function called with the run directory
# refrec ID STATUS DETAIL : rec, except that a named kill (REFUSAL_INVERT=1) reads the refusal arm the other way round: the arm must FAIL on the
# broken build, and a PASS there means the arm cannot tell the break from the refusal it is meant to test
refrec() {
  if [ "${REFUSAL_INVERT:-0}" = 1 ]; then
    case "$2" in
      FAIL) rec "$1" PASS "$3 (must-fail arm failed as it must)";;
      PASS) rec "$1" BLIND "$3 (must-fail arm did NOT fail: the arm cannot detect this break)";;
      *) rec "$1" "$2" "$3";;
    esac
  else rec "$1" "$2" "$3"; fi
}
refusal_arm() {
  local id=$1 desc=$2 needle=$3 edit=$4; shift 4
  [ "$BUILD_ONLY" = 1 ] && { rec "$id" SKIP "--build-only"; return 0; }
  local name="ref_$(echo "$id" | tr '/ ()' '____')" d rc
  d=$(rundir "$name" "${REFUSAL_BUILD:-cpu_branch}")
  "$edit" "$d" > "$LOGS/edit_$name.log" 2>&1 || { refrec "$id" ERROR "the run directory edit $edit failed (see $LOGS/edit_$name.log)"; return 1; }
  mkcfg "$d" arm.cfg out "$@"
  ( cd "$d" && timeout 300 $(launcher "$NP_BIG") ./build/triton.exe arm.cfg > run.log 2>&1 ); rc=$?
  local left=0 pid
  for pid in $(pgrep -u "$(id -u)" -f "[b]uild/triton.exe arm.cfg" 2>/dev/null); do
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$d" ] && { left=$((left+1)); kill -9 "$pid" 2>/dev/null; }
  done
  local rasters; rasters=$(find "$d/out/bin" -name 'H_*' 2>/dev/null | wc -l)
  if [ $rc -eq 0 ] && grep -aq "Simulation ends" "$d/run.log"; then
    refrec "$id" BLIND "$desc: the run was NOT refused (exit 0, simulation completed; see $d/run.log): the instrument cannot detect this break"
  elif [ $rc -eq 0 ]; then
    refrec "$id" FAIL "$desc: exit 0 without a completed simulation: a refusal must exit non-zero (see $d/run.log)"
  elif [ $rc -eq 124 ] || [ $left -gt 0 ]; then
    refrec "$id" FAIL "$desc: exit $rc, $left rank process(es) left running (see $d/run.log)"
  elif grep -aq "Simulation starts" "$d/run.log" || [ "$rasters" -gt 0 ]; then
    refrec "$id" FAIL "$desc: exit $rc but the run got past initialisation (Simulation starts printed or $rasters raster file(s) written; see $d/run.log)"
  elif ! grep -aqF -- "$needle" "$d/run.log"; then
    refrec "$id" FAIL "$desc: exit $rc before the first step but the log does not carry '$needle' (see $d/run.log)"
  else
    refrec "$id" PASS "$desc: exit $rc before the first step, the log carries '$needle', no rank left running"
  fi
}

# the edits of the N14 refusal arms (each a variant of a COPY of the model in the run directory)
ofl_edit_timeseries() { $PYTHON -I "$OFL_CK" edit-inp "$1" outfall-type O1 TIMESERIES TS1; }
ofl_edit_o1_on_j1()   { $PYTHON -I "$OFL_CK" edit-inp "$1" coords O1 J1; }
ofl_edit_o2_on_o1()   { $PYTHON -I "$OFL_CK" edit-inp "$1" coords O2 O1; }
ofl_edit_two_links()  { $PYTHON -I "$OFL_CK" edit-inp "$1" add-conduit C5 J3 O1; }

# n14_resume TAG NP : a clean run with every capture on and the driven key, then arms (b) snapshot route, (c) replay route and (e)
# a mid-interval kill, each resumed from checkpoint CK and compared with the clean run, outfall_stage.bin and its sidecar included
n14_resume() {
  local tag=$1 np=$2 files="$ALLFILES,outfall_stage.bin,outfall_stage.nodes" clean clean_dir
  branch_run "${tag}_clean" cpu "$np" $CAP_ALL swmm_snapshot_keep_all=1 swmm_driven_outfalls=driven_below.txt || return 1
  clean=$ARM_OUT; clean_dir=$ARM_DIR
  resume_arm "$clean_dir" "${tag}_b" cpu_branch "$np" "$CK" && {
    cmp_arm "N14/d/np$np/b" 0 "(d) (c) resumed from checkpoint $CK, snapshot route, equals its clean run (hydraulics, every capture file and outfall_stage.bin)" "$clean" "$ARM_OUT" "$files"
    grep -aq "path=snapshot" "$ARM_DIR/res.log" && rec "N14/d/np$np/b/route" PASS "(d)(b) took the snapshot route" || rec "N14/d/np$np/b/route" FAIL "(d)(b) did not take the snapshot route (see $ARM_DIR/res.log)"
  }
  resume_arm "$clean_dir" "${tag}_c" cpu_branch "$np" "$CK" swmm_snapshot_disable=1 && {
    cmp_arm "N14/d/np$np/c" 0 "(d) (c) resumed from checkpoint $CK, replay route, equals its clean run (the replay sets the logged stage before each replayed SWMM step)" "$clean" "$ARM_OUT" "$files"
    grep -aq "path=replay" "$ARM_DIR/res.log" && rec "N14/d/np$np/c/route" PASS "(d)(c) took the replay route" || rec "N14/d/np$np/c/route" FAIL "(d)(c) did not take the replay route (see $ARM_DIR/res.log)"
  }
  if kill_run "${tag}_e_kill" cpu_branch "$np" cfg "$CK" $CAP_ALL swmm_snapshot_keep_all=1 swmm_driven_outfalls=driven_below.txt; then
    resume_arm "$ARM_DIR" "${tag}_e_resume" cpu_branch "$np" "$CK" && cmp_arm "N14/d/np$np/e" 0 "(d) (c) killed mid-interval after checkpoint $CK, resumed from it, equals its clean run" "$clean" "$ARM_OUT" "$files"
  fi
}

case_N14() {
  need_cpu || return 1
  local saved=$CASE_SRC; CASE_SRC=$OFL_SRC
  n14_body; local rc=$?
  CASE_SRC=$saved
  return $rc
}
n14_body() {
  local np osim oprint ocell
  osim=$(sed -n 's/^sim_duration=//p' "$OFL_SRC/TRITONSWMM.cfg"); oprint=$(sed -n 's/^print_interval=//p' "$OFL_SRC/TRITONSWMM.cfg")
  ocell=$(python3 -c "import sys; print([l.split()[1] for l in open('$OFL_SRC/elevation.dem') if l.lower().startswith('cellsize')][0])")
  if [ "$osim" != "$SIM" ] || [ "$oprint" != "$PRINT" ] || [ "$ocell" != "$CELL" ]; then
    rec "N14/case" ERROR "the outfall case's sim_duration/print_interval/cellsize ($osim/$oprint/$ocell) differ from the fixed case's ($SIM/$PRINT/$CELL), which the suite's checkpoint and print arithmetic is derived from"
    return 1
  fi
  local hextra; hextra=$(sed -n 's/^hextra=//p' "$OFL_SRC/TRITONSWMM.cfg")
  HYD=0 chk "N14/layout" 0 "the outfall case: O1's cell lies below its invert, O2's cell lies above its invert, neither outfall is owned by rank 0 at np$NP_BIG" $PYTHON -I "$OFL_CK" layout "$OFL_SRC" "$NP_BIG" O1 O2

  # (a) key absent, against the outfall case's own baseline built with the same flags
  local -A BASEO AO
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "_ofl" || return 1; BASEO[$np]=$BASE_OUT
    branch_run "n14_a_np$np" cpu "$np" || return 1; AO[$np]=$ARM_OUT
    cmp_arm "N14/a/np$np" 0 "(a) swmm_driven_outfalls absent, CPU np$np: branch against the baseline on the outfall case, byte for byte" "${BASEO[$np]}" "${AO[$np]}"
  done
  $CMP "$NEUT" compare --a "${AO[1]}" --b "${AO[$NP_BIG]}" > "$LOGS/N14_a_np1_vs_np8.log" 2>&1; local arc=$?
  if [ $arc -eq 0 ]; then rec "N14/a/np1-vs-np$NP_BIG" PASS "(a) at np1 and np$NP_BIG are byte-identical: the np1-against-np$NP_BIG reading of (c) is read"
  else rec "N14/a/np1-vs-np$NP_BIG" REPORT "(a) at np1 and np$NP_BIG are NOT byte-identical (see $LOGS/N14_a_np1_vs_np8.log): the np1-against-np$NP_BIG reading of (c) is an instrument unable to detect that break on this case"; fi

  # (b) the key set to a list that names nothing
  for np in 1 "$NP_BIG"; do
    branch_run "n14_b_np$np" cpu "$np" swmm_driven_outfalls=driven_empty.txt && \
      cmp_arm "N14/b/np$np" 0 "(b) swmm_driven_outfalls set to an empty list file, CPU np$np: byte-identical to (a)" "${AO[$np]}" "$ARM_OUT"
  done

  # (c) the below-invert outfall listed: serial once, np8 twice
  local c1="" c1d="" c8a="" c8b=""
  branch_run n14_c_np1 cpu 1 swmm_driven_outfalls=driven_below.txt && { c1=$ARM_OUT; c1d=$ARM_DIR; }
  branch_run "n14_c_np${NP_BIG}_1" cpu "$NP_BIG" swmm_driven_outfalls=driven_below.txt && c8a=$ARM_OUT
  branch_run "n14_c_np${NP_BIG}_2" cpu "$NP_BIG" swmm_driven_outfalls=driven_below.txt && c8b=$ARM_OUT
  [ -n "$c1" ] && chk "N14/c/np1/differs" 0 "(c) O1 (cell below its invert) listed, CPU np1, differs from (a) at np1" ofl_differ "${AO[1]}" "$c1"
  [ -n "$c8a" ] && chk "N14/c/np$NP_BIG/differs" 0 "(c) O1 listed, CPU np$NP_BIG, differs from (a) at np$NP_BIG" ofl_differ "${AO[$NP_BIG]}" "$c8a"
  [ -n "$c8a" ] && [ -n "$c8b" ] && HYD=0 chk "N14/c/np$NP_BIG/determinism" 0 "(c) two np$NP_BIG runs of the same list are byte-identical (outfall_stage.bin included)" $CMP "$NEUT" compare --a "$c8a" --b "$c8b" --captures outfall_stage.bin,outfall_stage.nodes
  if [ -n "$c1" ] && [ -n "$c8a" ]; then
    if [ $arc -eq 0 ]; then
      HYD=0 chk "N14/c/np1-vs-np$NP_BIG" 0 "(c) np1 against np$NP_BIG: rasters, side-file, .out, .rpt block and outfall_stage.bin byte-identical ((a) is byte-identical across the two rank counts)" $CMP "$NEUT" compare --a "$c1" --b "$c8a" --captures outfall_stage.bin,outfall_stage.nodes
    else
      rec "N14/c/np1-vs-np$NP_BIG" REPORT "(c) np1 against np$NP_BIG not read: (a) is not byte-identical across the two rank counts, so the reading cannot detect a mis-split value on this case"
    fi
  fi
  [ -n "$c1" ] && $PYTHON -I "$OFL_CK" stats "$c1" > "$LOGS/N14_c_stats.log" 2>&1 && rec "N14/c/stats" REPORT "$(tr '\n' ' ' < "$LOGS/N14_c_stats.log" | cut -c1-400)"

  # (c') the above-invert outfall listed
  local cp1=""
  branch_run n14_cp_np1 cpu 1 swmm_driven_outfalls=driven_above.txt && cp1=$ARM_OUT
  [ -n "$cp1" ] && chk "N14/cprime/np1/differs-a" 0 "(c') O2 (cell above its invert) listed, CPU np1, differs from (a)" ofl_differ "${AO[1]}" "$cp1"
  [ -n "$cp1" ] && [ -n "$c1" ] && chk "N14/cprime/np1/differs-c" 0 "(c') differs from (c): the two outfall forms are different arms" ofl_differ "$c1" "$cp1"
  local cp8a="" cp8b=""
  branch_run "n14_cp_np${NP_BIG}_1" cpu "$NP_BIG" swmm_driven_outfalls=driven_above.txt && cp8a=$ARM_OUT
  branch_run "n14_cp_np${NP_BIG}_2" cpu "$NP_BIG" swmm_driven_outfalls=driven_above.txt && cp8b=$ARM_OUT
  [ -n "$cp8a" ] && chk "N14/cprime/np$NP_BIG/differs" 0 "(c') O2 listed, CPU np$NP_BIG, differs from (a) at np$NP_BIG" ofl_differ "${AO[$NP_BIG]}" "$cp8a"
  [ -n "$cp8a" ] && [ -n "$cp8b" ] && HYD=0 chk "N14/cprime/np$NP_BIG/determinism" 0 "(c') two np$NP_BIG runs of the same list are byte-identical (outfall_stage.bin included)" $CMP "$NEUT" compare --a "$cp8a" --b "$cp8b" --captures outfall_stage.bin,outfall_stage.nodes
  if [ -n "$cp1" ] && [ -n "$cp8a" ]; then
    if [ $arc -eq 0 ]; then
      HYD=0 chk "N14/cprime/np1-vs-np$NP_BIG" 0 "(c') np1 against np$NP_BIG byte-identical ((a) is byte-identical across the two rank counts)" $CMP "$NEUT" compare --a "$cp1" --b "$cp8a" --captures outfall_stage.bin,outfall_stage.nodes
    else rec "N14/cprime/np1-vs-np$NP_BIG" REPORT "(c') np1 against np$NP_BIG not read: (a) is not byte-identical across the two rank counts"; fi
  fi
  HYD=0 chk "N14/cprime/owner" 0 "(c') the above-invert outfall O2 is owned by a rank other than 0 at np$NP_BIG (the layout check, which reads both outfalls' owners by the solver's strip rule)" $PYTHON -I "$OFL_CK" layout "$OFL_SRC" "$NP_BIG" O1 O2
  [ -n "$cp1" ] && $PYTHON -I "$OFL_CK" stats "$cp1" > "$LOGS/N14_cprime_stats.log" 2>&1 && rec "N14/cprime/stats" REPORT "$(tr '\n' ' ' < "$LOGS/N14_cprime_stats.log" | cut -c1-400)"

  # (f) the stage check, on (c) and (c'), with the coupling interval 1 and lag 0 (the case's defaults)
  [ -n "$c1" ] && HYD=0 chk "N14/f/c" 0 "(f) (c): at each print instant stage_m = DEM[cell] + H_raster[cell] + q_back_applied*dt/dx^2 where the depth read was at least hextra, the invert otherwise" $PYTHON -I "$OFL_CK" stage-check "$c1" "$OFL_SRC" O1 "$PRINT" "$hextra" "$FSTOL"
  [ -n "$cp1" ] && HYD=0 chk "N14/f/cprime" 0 "(f) (c'): the same stage check on the above-invert outfall O2" $PYTHON -I "$OFL_CK" stage-check "$cp1" "$OFL_SRC" O2 "$PRINT" "$hextra" "$FSTOL"

  # the F-4 control of the stage check: a copy of (c)'s output whose backflow is zero (stages made consistent with it) must FAIL the check on its
  # must-be-nonzero control alone
  if [ -n "$c1" ]; then
    chk "N14/f/control-make" 0 "the control: (c)'s output with q_back_applied zero and the stages rewritten to match" $PYTHON -I "$OFL_CK" fixture-no-backflow "$c1" "$OFL_SRC" O1 "$PRINT" "$hextra" "$WORK/run/n14_f_nobackflow"
    HYD=0 chk "N14/f/control" 1 "(f) on the zero-backflow copy: fails (no wet print instant has q_back_applied > 0)" $PYTHON -I "$OFL_CK" stage-check "$WORK/run/n14_f_nobackflow" "$OFL_SRC" O1 "$PRINT" "$hextra" "$FSTOL"
  fi

  # (e) the ledger: the domain balance closes with the new column, and the .rpt continuity totals equal the full sums
  if branch_run n14_e_np1 cpu 1 swmm_driven_outfalls=driven_below.txt diag_domain_balance=1 swmm_diag_steps=1 swmm_diag_links=links_c4.txt; then
    local nodes; nodes=$(sed -n 's/.*SWMM coupled node \([^:]*\): rim .*/\1/p' "$ARM_DIR/run.log" | paste -sd, -)
    HYD=0 chk "N14/e/balance" 0 "(e) domain_balance.csv carries the two driven-outfall columns, outfall_backflow_m3 is in the signed sum and the residual is at round-off" $PYTHON -I "$OFL_CK" residual "$ARM_OUT"
    HYD=0 chk "N14/e/rpt" 0 "(e) the .rpt continuity External Inflow and External Outflow equal the full sums of the step records (coupling nodes $nodes; the other outfall's flow from the T1d record of C4)" $PYTHON -I "$OFL_CK" ledger "$ARM_OUT" "$OFL_SRC" O1 "$PRINT" "$nodes" C4
  fi

  # (d) resume, serial and np8
  for np in 1 "$NP_BIG"; do n14_resume "n14_d_np$np" "$np"; done

  # the refusal arms, np8: each stops before the first step, names the node (or the key) and leaves no rank running
  local TB="$WORK/n14_lists"; mkdir -p "$TB"
  printf 'J1\n' > "$TB/r_junction.txt"
  printf 'J1\nJ3\nO1\n' > "$TB/r_coupling_o1.txt"
  printf 'O1\nO2\n' > "$TB/r_both.txt"
  printf 'O1\no1\n' > "$TB/r_repeat.txt"
  refusal_arm "N14/refuse/junction" "a junction listed (J1)" "node J1 is not an OUTFALL node" no_edit swmm_driven_outfalls="$TB/r_junction.txt"
  refusal_arm "N14/refuse/timeseries" "a TIMESERIES outfall listed (O1 retyped in a copy of the model)" "type TIMESERIES" ofl_edit_timeseries swmm_driven_outfalls=driven_below.txt
  refusal_arm "N14/refuse/coupling-node" "an outfall that is also a coupling node (O1 added to swmm_coupling_nodes; the refusal of check_driven_outfalls, not the older junctions-only one of init_swmm)" "is also a coupling node" no_edit swmm_driven_outfalls=driven_below.txt swmm_coupling_nodes="$TB/r_coupling_o1.txt"
  refusal_arm "N14/refuse/cell-holds-coupling-node" "an outfall whose cell holds a coupling node (O1 moved onto J1's coordinates in a copy of the model)" "shares its cell with coupling node J1" ofl_edit_o1_on_j1 swmm_driven_outfalls=driven_below.txt
  refusal_arm "N14/refuse/two-outfalls-one-cell" "two outfalls in one cell (O2 moved onto O1's coordinates in a copy of the model)" "shares its cell with driven outfall" ofl_edit_o2_on_o1 swmm_driven_outfalls="$TB/r_both.txt"
  refusal_arm "N14/refuse/two-links" "an outfall with two connected links (a conduit C5 joined to O1 in a copy of the model; the engine's own ERROR 141 stops the run at swmm_start, before the solver's connected-link count)" "Outfall O1 has more than 1 inlet link" ofl_edit_two_links swmm_driven_outfalls=driven_below.txt
  refusal_arm "N14/refuse/repeated-name" "a repeated name (O1, then o1)" "repeats" no_edit swmm_driven_outfalls="$TB/r_repeat.txt"
  refusal_arm "N14/refuse/unreadable" "an unreadable path" "$TB/does_not_exist.txt" no_edit swmm_driven_outfalls="$TB/does_not_exist.txt"
  refusal_arm "N14/refuse/no-coupling-node" "a nonempty list with an empty coupling list (the refusal names the key)" "swmm_driven_outfalls" no_edit swmm_driven_outfalls=driven_below.txt swmm_coupling_nodes=coupling_empty.txt

  # the test-only breaks: a source mutant per arm, each must make its arm fail
  if build_mut n14_m1 src/config_utils.h 'arglist\.swmm_driven_outfalls = argsd\("swmm_driven_outfalls", argmap, ""\);' 'arglist.swmm_driven_outfalls = "";'; then
    run_arm n14_m1_c n14_m1 1 swmm_driven_outfalls=driven_below.txt && \
      chk "N14/M1/c" 1 "break M1 (a build that ignores the key): (c) is byte-identical to (a), so (c)'s must-differ arm fails" ofl_differ "${AO[1]}" "$ARM_OUT"
  fi
  if build_mut n14_m2 src/triton.h 'device_vec\[H\] \+ d\.pos\[driven_owned_i_\[k\]\], sizeof\(T\)' 'device_vec[H] + d.pos[driven_owned_i_[k]] + 1, sizeof(T)'; then
    run_arm n14_m2_c n14_m2 1 swmm_driven_outfalls=driven_below.txt && \
      HYD=0 chk "N14/M2/f" 1 "break M2 (the stage is read at the neighbouring cell): (f) fails" $PYTHON -I "$OFL_CK" stage-check "$ARM_OUT" "$OFL_SRC" O1 "$PRINT" "$hextra" "$FSTOL"
  fi
  if build_mut n14_m3 src/swmm_triton.h 'h_arr\[sid\] = hij;(\s+qx_arr\[sid\] = qx;\s+qy_arr\[sid\] = qy;\s+applied\[k\] = ap;)' '/* test-only break: the cell is not lowered */\1'; then
    run_arm n14_m3_c n14_m3 1 swmm_driven_outfalls=driven_below.txt diag_domain_balance=1 && {
      HYD=0 chk "N14/M3/f" 1 "break M3 (the stage is applied and the backflow is booked but the cell is not lowered): (f) fails" $PYTHON -I "$OFL_CK" stage-check "$ARM_OUT" "$OFL_SRC" O1 "$PRINT" "$hextra" "$FSTOL"
      HYD=0 chk "N14/M3/e" 1 "break M3: (e)'s balance fails (the residual no longer closes with outfall_backflow_m3 in the sum)" $PYTHON -I "$OFL_CK" residual "$ARM_OUT"
    }
  fi
  # named kill of the F-1 arm: the check "is also a coupling node" (check_driven_outfalls) deleted in a copy; the arm must fail on it
  if build_mut n14_m7 src/swmm_triton.h 'for \(const std::string& c : coupling_names\)\s*if \(swmm_upper\(c\) == up\) \{[^}]*is also a coupling node[^}]*\}' '/* test-only break: the coupling-node check is deleted */'; then
    REFUSAL_BUILD=n14_m7 REFUSAL_INVERT=1 refusal_arm "N14/M7/coupling-node" "break M7 (the new coupling-node refusal deleted, the older junctions-only refusal of init_swmm still stops the run naming O1): the arm N14/refuse/coupling-node" "is also a coupling node" no_edit swmm_driven_outfalls=driven_below.txt swmm_coupling_nodes="$TB/r_coupling_o1.txt"
  fi
  if build_mut n14_m4 src/swmm_triton.h 'encode_outfall_stage_record\(step_end_time, swmm_dt, driven\.stage_m\.data\(\)' 'encode_outfall_stage_record(step_end_time, swmm_dt, driven.q_back_applied_m3s.data()'; then
    run_arm n14_m4_clean n14_m4 1 $CAP_ALL swmm_snapshot_keep_all=1 swmm_driven_outfalls=driven_below.txt && {
      local m4=$ARM_OUT m4d=$ARM_DIR
      resume_arm "$m4d" n14_m4_replay n14_m4 1 "$CK" swmm_snapshot_disable=1 && \
        cmp_arm "N14/M4/d" 1 "break M4 (a build that does not log the stage the replay sets): (d) on the replay route fails" "$m4" "$ARM_OUT" "$ALLFILES,outfall_stage.bin"
    }
  fi
}

# n15_edit_lower DIR : the DEM cell of the listed coupling node is set below the node's invert in a copy of the case (N15 (d))
N15_ROW=""; N15_COL=""; N15_VAL=""
n15_edit_lower() {
  $PYTHON -I - "$1/elevation.dem" "$N15_ROW" "$N15_COL" "$N15_VAL" <<'PYEOF'
import sys
p, row, col, val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
lines = open(p).read().split("\n")
r = lines[6 + row].split()
r[col] = val
lines[6 + row] = " ".join(r)
open(p, "w").write("\n".join(lines))
sys.stdout.write("DEM cell (%d, %d) of %s set to %s\n" % (row, col, p, val))
PYEOF
}
# n15_rim_count LOG : the number the rank-0 rim line reports, "SWMM coupled-node rims set from the DEM: N nodes"
n15_rim_count() { sed -n 's/.*SWMM coupled-node rims set from the DEM: \([0-9]*\) nodes.*/\1/p' "$1" | head -1; }

case_N15() {
  need_cpu || return 1
  n13_write_helper
  local H="$WORK/n13_helper.py" TB="$WORK/n15_lists" np RIM="swmm_node_rim_from_dem=1"
  mkdir -p "$TB"
  local -A BASEO AO
  for np in 1 "$NP_BIG"; do
    base_run cpu "$np" "_rim" $RIM || return 1; BASEO[$np]=$BASE_OUT
    branch_run "n15_a_np$np" cpu "$np" $RIM || return 1; AO[$np]=$ARM_OUT
    cmp_arm "N15/a/np$np" 0 "(a) swmm_dem_rim_exempt_nodes absent with swmm_node_rim_from_dem=1, CPU np$np: branch against the baseline run of the same configuration, byte for byte" "${BASEO[$np]}" "${AO[$np]}"
  done
  local nodes; nodes=($(sed -n 's/.*SWMM coupled node \([^:]*\): rim .*/\1/p' "${BASEO[1]}/../run.log"))
  [ ${#nodes[@]} -ge 2 ] || { rec "N15/nodes" ERROR "fewer than two 'SWMM coupled node ... rim' lines in ${BASEO[1]}/../run.log"; return 1; }
  $PYTHON -I "$H" select "$CASE_SRC" "$NP_BIG" "${BASEO[1]}/swmm/hydraulics_exchange_replay.bin" "${nodes[@]}" > "$LOGS/N15_select.log" 2>&1; local src=$?
  cat "$LOGS/N15_select.log"
  [ $src -eq 0 ] || { rec "N15/select" ERROR "no coupling node fits (exit $src; see $LOGS/N15_select.log)"; return 1; }
  local NS NS_IDX NS_OWN NS_FIRST
  read -r _ NS NS_IDX NS_OWN NS_FIRST <<< "$(grep '^NSTAR ' "$LOGS/N15_select.log")"
  # the listed node must be a node whose DEM rim differs from SWMM's rim, or the exemption has nothing to change
  local rimline demline rlog="${AO[1]}/../run.log"
  rimline=$(sed -n "s/.*SWMM coupled node $NS: rim \([^,]*\), invert \([^ ]*\).*/\1 \2/p" "$rlog" | head -1)
  demline=$(sed -n "s/.*SWMM coupled node $NS: DEM rim \([^ ]*\) (cell elevation \([^)]*\)).*/\1 \2/p" "$rlog" | head -1)
  if [ -z "$rimline" ] || [ -z "$demline" ] || [ "${rimline%% *}" = "${demline%% *}" ]; then
    rec "N15/select" ERROR "the listed node $NS has SWMM rim '${rimline%% *}' and DEM rim '${demline%% *}' in (a)'s log: a node whose DEM rim equals SWMM's rim gives the exemption nothing to change"
    return 1
  fi
  rec "N15/select" PASS "N15 lists $NS (global index $NS_IDX, owner rank $NS_OWN at np$NP_BIG): SWMM rim ${rimline%% *}, invert ${rimline##* }, DEM rim ${demline%% *} (cell elevation ${demline##* }); it is alone in its cell and its first nonzero exchange is at side-file record $NS_FIRST of (a)"
  printf '; exempt: one coupling node\n%s\n' "$NS" > "$TB/exempt_one.txt"
  printf '; exempt: none\n' > "$TB/exempt_empty.txt"
  printf '%s\n%s\n' "$NS" "$(echo "$NS" | tr 'A-Z' 'a-z')" > "$TB/exempt_repeat.txt"
  printf 'O1\n' > "$TB/exempt_noncoupling.txt"

  # (b) the key set to a list that names nothing
  for np in 1 "$NP_BIG"; do
    branch_run "n15_b_np$np" cpu "$np" $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_empty.txt" && \
      cmp_arm "N15/b/np$np" 0 "(b) swmm_dem_rim_exempt_nodes set to an empty list file, CPU np$np: byte-identical to (a)" "${AO[$np]}" "$ARM_OUT"
  done
  # (c) one coupled node listed: serial and np8
  local c1="" c8="" c1d="" c8d=""
  branch_run n15_c_np1 cpu 1 $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt" && { c1=$ARM_OUT; c1d=$ARM_DIR; }
  branch_run "n15_c_np$NP_BIG" cpu "$NP_BIG" $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt" && { c8=$ARM_OUT; c8d=$ARM_DIR; }
  local arm anp out dir
  for arm in "1:$c1:$c1d" "$NP_BIG:$c8:$c8d"; do
    anp=${arm%%:*}; out=${arm#*:}; dir=${out#*:}; out=${out%%:*}
    [ -n "$out" ] || continue
    n13_must_differ "N15/c/np$anp/differs" "(c) $NS listed under swmm_node_rim_from_dem=1, CPU np$anp, against (a) at np$anp" "${AO[$anp]}" "$out"
    HYD=0 chk "N15/c/np$anp/column" 0 "(c) np$anp against (a): the first differing side-file record differs in $NS's column ($NS_IDX) alone" $PYTHON -I "$H" firstcols "${AO[$anp]}/swmm/hydraulics_exchange_replay.bin" "$out/swmm/hydraulics_exchange_replay.bin" "$NS_IDX"
    local na nc; na=$(n15_rim_count "${AO[$anp]}/../run.log"); nc=$(n15_rim_count "$dir/run.log")
    [ -n "$na" ] && [ -n "$nc" ] && [ "$nc" = "$((na - 1))" ] && rec "N15/c/np$anp/rim-line" PASS "(c) the rank-0 rim line counts $nc nodes against (a)'s $na: one node fewer" || rec "N15/c/np$anp/rim-line" FAIL "(c) the rank-0 rim line counts '$nc' nodes against (a)'s '$na': expected one fewer"
    local nex; nex=$(grep -ac "DEM rim exempt" "$dir/run.log")
    [ "$nex" = 1 ] && grep -aq "SWMM coupled node $NS: DEM rim exempt" "$dir/run.log" && rec "N15/c/np$anp/exempt-lines" PASS "(c) exactly one 'DEM rim exempt' line, naming $NS" || rec "N15/c/np$anp/exempt-lines" FAIL "(c) $nex 'DEM rim exempt' lines (want exactly one, naming $NS; see $dir/run.log)"
  done
  if [ -n "$c1" ] && [ -n "$c8" ]; then
    $CMP "$NEUT" compare --a "${AO[1]}" --b "${AO[$NP_BIG]}" > "$LOGS/N15_a_np1_vs_np8.log" 2>&1
    if [ $? -eq 0 ]; then HYD=0 chk "N15/c/np1-vs-np$NP_BIG" 0 "(c) np1 against np$NP_BIG: byte-identical ((a) is byte-identical across the two rank counts)" $CMP "$NEUT" compare --a "$c1" --b "$c8"
    else rec "N15/c/np1-vs-np$NP_BIG" REPORT "(c) np1 against np$NP_BIG not read: (a) is not byte-identical across the two rank counts"; fi
  fi
  # (d) a listed node whose cell lies at or below its invert runs: the DEM cell of $NS is set below its invert in a copy of the case
  N15_ROW=$(awk -v n="$NS" '$1=="NODE" && $2==n {print $6}' "$LOGS/N15_select.log"); N15_COL=$(awk -v n="$NS" '$1=="NODE" && $2==n {print $8}' "$LOGS/N15_select.log")
  N15_VAL=$($PYTHON -I -c "print(repr(float('${rimline##* }') - 0.05))")
  if run_arm_edit n15_edit_lower n15_d_ctl cpu_branch 1 $RIM; then
    rec "N15/d/control" BLIND "(d) control: the variant (cell of $NS at $N15_VAL, below its invert ${rimline##* }) ran WITHOUT the exemption: the variant does not produce the refused condition"
  elif grep -aq "its DEM cell cannot give a rim" "$ARM_DIR/run.log"; then
    rec "N15/d/control" PASS "(d) control: the variant (cell of $NS at $N15_VAL, below its invert ${rimline##* }) is refused without the exemption, 'its DEM cell cannot give a rim'"
  else rec "N15/d/control" ERROR "(d) control: the variant run failed for another reason (see $ARM_DIR/run.log)"; fi
  if run_arm_edit n15_edit_lower n15_d cpu_branch 1 $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt"; then
    grep -aq "SWMM coupled node $NS: DEM rim exempt" "$ARM_DIR/run.log" && rec "N15/d" PASS "(d) $NS listed, its cell at $N15_VAL below its invert ${rimline##* }: the run completes, $NS exempt (not refused at the DEM-rim route)" || rec "N15/d" FAIL "(d) the run completed but did not log $NS as exempt (see $ARM_DIR/run.log)"
  else rec "N15/d" FAIL "(d) $NS listed, its cell below its invert: the run did not complete (see $ARM_DIR/run.log)"; fi
  # the refusal arms, np8
  refusal_arm "N15/refuse/no-coupling-node" "a listed name that is no coupling node (O1)" "$TB/exempt_noncoupling.txt" no_edit $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_noncoupling.txt"
  refusal_arm "N15/refuse/repeated-name" "a repeated name ($NS, then its lower-case spelling)" "$TB/exempt_repeat.txt" no_edit $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_repeat.txt"
  refusal_arm "N15/refuse/unreadable" "an unreadable path" "$TB/does_not_exist.txt" no_edit $RIM swmm_dem_rim_exempt_nodes="$TB/does_not_exist.txt"
  refusal_arm "N15/refuse/rim-key-off" "a nonempty list while swmm_node_rim_from_dem is off" "swmm_node_rim_from_dem is off" no_edit swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt"
  # the test-only breaks
  if build_mut n15_m5 src/config_utils.h 'arglist\.swmm_dem_rim_exempt_nodes = argsd\("swmm_dem_rim_exempt_nodes", argmap, ""\);' 'arglist.swmm_dem_rim_exempt_nodes = "";'; then
    run_arm n15_m5_c n15_m5 1 $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt" && \
      chk "N15/M5/c" 1 "break M5 (a build that ignores the key): (c) is byte-identical to (a), so (c)'s must-differ arm fails" ofl_differ "${AO[1]}" "$ARM_OUT"
  fi
  if build_mut n15_m6 src/triton.h 'if \(!rim_exempt\.empty\(\) && rim_exempt\.count\(SWMM_triton::swmm_upper\(swmm_model\.nodeID\[k\]\)\) > 0\) \{' 'if (!rim_exempt.empty()) {'; then
    run_arm n15_m6_c n15_m6 1 $RIM swmm_dem_rim_exempt_nodes="$TB/exempt_one.txt" && {
      local nex6; nex6=$(grep -ac "DEM rim exempt" "$ARM_DIR/run.log")
      [ "$nex6" = 1 ] && rec "N15/M6/exempt-lines" BLIND "break M6 (a build that exempts every node once the list is nonempty): still exactly one 'DEM rim exempt' line, the instrument cannot detect it" || rec "N15/M6/exempt-lines" PASS "break M6 (a build that exempts every node once the list is nonempty): $nex6 'DEM rim exempt' lines against the one listed node (must-fail arm failed as it must)"
    }
  fi
}

# ------------------------------------------------------------------ run
case_coverage || true
for c in N1 N2 N3 N5 N6 N7 N8 N9 N10 N11 N12 N13 N14 N15; do
  if want "$c"; then say "=============== $c ==============="; "case_$c"; fi
done

# ------------------------------------------------------------------ summary
say ""
say "=== neutrality suite result (baseline $BASE_SHA, test $TEST_SHA; Kokkos tree: see $BUILD_RECORD) ==="
awk -F'\t' '{n[$2]++} END {for (s in n) printf "%s %d\n", s, n[s]}' "$RESULTS" | sort
NBAD=$(awk -F'\t' '$2=="FAIL" || $2=="ERROR" || $2=="BLIND"' "$RESULTS" | wc -l)
awk -F'\t' '$2=="FAIL" || $2=="ERROR" || $2=="BLIND" {printf "  %s %s %s\n", $2, $1, $3}' "$RESULTS"
if [ "$HYD_FAIL" = 1 ]; then
  say "HYDRAULICS: a hydraulic-output arm FAILED. The failing capture is reclassified as possibly hydraulics-affecting and its merge stops until the developer rules (for N13: a difference at (a) or (b), whose table must not change the hydraulics, is held and reported)."
fi
say "records: $RESULTS  build record: $BUILD_RECORD  logs: $LOGS"
[ "$NBAD" -eq 0 ]
