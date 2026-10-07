#!/bin/bash
# WP-0A REVIEW harness: the stored-width guard across a REAL precision boundary.
#
# Builds test_exchange_log_cross_precision.cpp TWICE against an existing coupled
# TRITON build tree -- once at the tree's own value_t, once with
# -DUSE_SINGLE_PRECISION -- then crosses the artifacts:
#
#   double-written file  read by the float  binary  -> must be refused as `width`
#   float-written  file  read by the double binary  -> must be refused as `width`
#   each file read by its OWN binary                -> must be accepted as `ok`
#
# The two same-precision legs are the control: without them a guard that refused
# EVERY file would pass the crossed legs and read as correct.
#
# This is deliberately NOT registered in test/unit/CMakeLists.txt. That file belongs
# to the implementation under review and a reviewer does not edit it; registering
# this driver as CTest node ids is a one-line change for the maintainer (see the
# review account). It runs standalone in the meantime.
#
# usage: run_cross_precision_check.sh <existing-coupled-build-dir>
set -u
BUILD="${1:?usage: $0 <coupled build dir>}"
SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$SRC_ROOT/test/unit_cross_precision"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CXX="${CXX:-mpic++}"
NODES=14
RECORDS=5
rc=0

# Derive the include/define set from the coupled build's OWN generated flags, rather
# than restating it here: a hand-listed set silently drifts from what the build uses
# (and omits the GENERATED desul/Kokkos config headers, which live under the build
# tree and are not discoverable from the source tree at all).
FLAGS_MAKE="$BUILD/test/unit/CMakeFiles/test_exchange_log_header.dir/flags.make"
if [ ! -f "$FLAGS_MAKE" ]; then
  echo "ERROR: expected generated flags at $FLAGS_MAKE"
  echo "       configure the build with -DTRITON_ENABLE_SWMM=ON -DBUILD_TESTS=ON first."
  exit 4
fi
INC="$(sed -n 's/^CXX_INCLUDES = //p' "$FLAGS_MAKE")"
DEFS="$(sed -n 's/^CXX_DEFINES = //p' "$FLAGS_MAKE")"

# Same reasoning for the link line: take it from the build's own generated link.txt,
# stripping the leading compiler invocation and the -o/output words, so this harness
# tracks whatever the coupled target links rather than restating it.
LINK_TXT="$BUILD/test/unit/CMakeFiles/test_exchange_log_header.dir/link.txt"
if [ ! -f "$LINK_TXT" ]; then echo "ERROR: expected generated link line at $LINK_TXT"; exit 4; fi
LIBS="$(cd "$BUILD/test/unit" && awk '{f=0; for(i=1;i<=NF;i++){ if(f==2) printf "%s ", $i; if($i=="-o" && f==0) f=1; else if(f==1) f=2 }}' "$LINK_TXT" \
        | tr " " "\n" | grep -v "^$" \
        | while read -r w; do case "$w" in -*) echo "$w";; *) readlink -f "$w" 2>/dev/null || echo "$w";; esac; done \
        | tr "\n" " ")"
RPATH="-Wl,-rpath,$BUILD/external/swmm/src/solver"

build_one() {  # $1 = out binary, $2... = extra flags
  local out="$1"; shift
  $CXX -std=c++17 -O1 -DTRITON_SWMM $DEFS "$@" $INC \
       "$HERE/test_exchange_log_cross_precision.cpp" -o "$out" $LIBS $RPATH 2> "$out.buildlog"
  local e=$?
  if [ $e -ne 0 ]; then echo "BUILD FAILED ($out), tail of log:"; tail -25 "$out.buildlog"; fi
  return $e
}

echo "=== leg 1: the shipped build's own precision ==="
build_one "$WORK/xp_native" || exit 3
echo "  built xp_native (value_t width = this tree's)"

echo "=== leg 2: the OTHER precision, via -DUSE_SINGLE_PRECISION ==="
if build_one "$WORK/xp_other" -DUSE_SINGLE_PRECISION; then
  OTHER_BUILT=1; echo "  built xp_other"
else
  OTHER_BUILT=0
  echo "  *** xp_other DID NOT BUILD -- see the finding printed at the end ***"
fi

run() { echo "--- $*"; "$@" || rc=1; }

echo
echo "=== leg A (always runs): real bytes, shipped writer -> shipped reader ==="
run "$WORK/xp_native" write "$WORK/native.bin" $NODES $RECORDS
run "$WORK/xp_native" read  "$WORK/native.bin" $NODES ok

echo
echo "=== leg B (always runs): a byte-level foreign-precision file ==="
# Ask the binary what width it is, so the forged width is genuinely the OTHER one.
NATIVE_W="$("$WORK/xp_native" read "$WORK/native.bin" $NODES ok | sed -n 's/.*this_build_width=\([0-9]*\).*/\1/p')"
if [ "$NATIVE_W" = "8" ]; then OTHER_W=4; else OTHER_W=8; fi
cp "$WORK/native.bin" "$WORK/forged.bin"
run "$WORK/xp_native" forge-width "$WORK/forged.bin" $OTHER_W
run "$WORK/xp_native" read "$WORK/forged.bin" $NODES width

echo
if [ "$OTHER_BUILT" = "1" ]; then
  echo "=== leg C: the TRUE two-binary cross ==="
  run "$WORK/xp_other" write "$WORK/other.bin" $NODES $RECORDS
  run "$WORK/xp_other" read  "$WORK/other.bin"  $NODES ok
  run "$WORK/xp_other" read  "$WORK/native.bin" $NODES width
  run "$WORK/xp_native" read "$WORK/other.bin"  $NODES width
fi

echo
if [ $rc -ne 0 ]; then
  echo "CROSS-PRECISION CHECK: FAIL -- a leg that ran did not behave as the guard requires."
  exit 1
fi
if [ "$OTHER_BUILT" = "1" ]; then
  echo "CROSS-PRECISION CHECK: PASS (legs A, B and C)"
  exit 0
fi

cat <<'MSG'
CROSS-PRECISION CHECK: INCOMPLETE -- legs A and B PASSED; leg C COULD NOT BE BUILT.

FINDING (this is the result, not a harness defect):
  The coupled solver does not compile under -DUSE_SINGLE_PRECISION. replay_exchange_history()
  passes std::vector<value_t>::data() into swmm5.h's swmm_step(double*, double*, double*, double),
  which is a hard type error when value_t is float. Measured at HEAD and, identically, at the
  package's base commit -- so it is PRE-EXISTING and WP-0A neither introduced nor was asked to
  fix it.

  Consequence for WP-0A: the scenario the guard defends against -- a side-file WRITTEN by a
  single-precision coupled build -- cannot presently be produced by this build system, because
  that build does not exist. The guard is correct and is worth keeping as defense-in-depth; leg B
  shows it refusing a genuine foreign-width file on disk. But its headline property is not
  reachable end-to-end today, and no single-binary test can show that.

  Exit 5 marks exactly this: the guard is verified as far as it can be, and the remaining
  distance is blocked by a defect outside WP-0A's declared files.
MSG
exit 5
