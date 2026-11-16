#!/usr/bin/env bash
# Hydraulics-neutrality suite of the coupling-diagnostics captures.
#
# Builds the faba2232 baseline and the branch with the SAME flags and the SAME Kokkos tree, runs
# both on the fixed coupled case (test/coupled_case/) and compares the hydraulic outputs byte for
# byte: the H, QX, QY and MH rasters, the exchange side-file, the SWMM .out and the .rpt
# flow-routing continuity block. Each case that switches a capture on also asserts that every
# enabled capture file exists with 16 + records x record size bytes. The cases N1..N12, the arms
# each one runs, the quantity it reads and the test-only build it must fail on are the table of the
# approved shape, section 10; test/neutrality_compare.py holds every comparison.
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
CASES="N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12"; NP_BIG=8; NO_GPU=0; KEEP_GOING=1; SEC_FLOOR=0.05; BUILD_ONLY=0; N6_PART=all
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
    -h|--help) sed -n '2,34p' "$THIS"; exit 0;;
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

# ------------------------------------------------------------------ run
case_coverage || true
for c in N1 N2 N3 N5 N6 N7 N8 N9 N10 N11 N12; do
  if want "$c"; then say "=============== $c ==============="; "case_$c"; fi
done

# ------------------------------------------------------------------ summary
say ""
say "=== neutrality suite result (baseline $BASE_SHA, test $TEST_SHA; Kokkos tree: see $BUILD_RECORD) ==="
awk -F'\t' '{n[$2]++} END {for (s in n) printf "%s %d\n", s, n[s]}' "$RESULTS" | sort
NBAD=$(awk -F'\t' '$2=="FAIL" || $2=="ERROR" || $2=="BLIND"' "$RESULTS" | wc -l)
awk -F'\t' '$2=="FAIL" || $2=="ERROR" || $2=="BLIND" {printf "  %s %s %s\n", $2, $1, $3}' "$RESULTS"
if [ "$HYD_FAIL" = 1 ]; then
  say "HYDRAULICS: a hydraulic-output arm FAILED. The failing capture is reclassified as possibly hydraulics-affecting and its merge stops until the developer rules."
fi
say "records: $RESULTS  build record: $BUILD_RECORD  logs: $LOGS"
[ "$NBAD" -eq 0 ]
