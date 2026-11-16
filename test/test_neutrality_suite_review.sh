#!/bin/bash
# Reviewer test of the Invariant B per-rank restatement (erratum E-18) in test/neutrality_compare.py perf-diff.
# Written by the 3a.1 reviewing seat; uncommitted. Venue: Rivanna scratch only (it runs python3 3.6 and no solver).
#
# Usage:  bash test/test_neutrality_suite_review.sh WORK_DIR [R4_DIR]
#   WORK_DIR  a NEW directory (the script refuses an existing one)
#   R4_DIR    the coder's round-4 repeat-run directory (/scratch/dcl3nd/3a1/r4), read only; enables part D
#
# Parts:  A  the coder's unit test passes on the tree's comparator (baseline)
#         B  mutants of the comparator: every mutant that is not an equivalent one must be KILLED by the coder's unit test
#         C  synthetic probes of the restated check (boundaries, three ranks, a header without DIAG)
#         D  replay of the round-4 per-run data (each capture-on run against off_1..off_3, the suite's three-off form)
# Exit status: 0 only when part A passes, every probe behaves as stated, and no mutant outside the stated
# survivors list survives; the stated survivors are printed as SURVIVED-EXPECTED with the reason.
set -u
T=$(cd "$(dirname "$0")" && pwd)
WORK=${1:?usage: test_neutrality_suite_review.sh WORK_DIR [R4_DIR]}
R4=${2:-}
[ -e "$WORK" ] && { echo "refusing: $WORK exists"; exit 2; }
mkdir -p "$WORK/base" || exit 2
cp "$T/neutrality_compare.py" "$T/test_neutrality_perf_diff.py" "$WORK/base/" || exit 2
BAD=0

echo "=== A. coder's unit test on the tree's comparator"
( cd "$WORK/base" && python3 -I test_neutrality_perf_diff.py > "$WORK/A.log" 2>&1 ); rc=$?
tail -3 "$WORK/A.log"
if [ $rc -eq 0 ]; then echo "A PASS (exit 0)"; else echo "A FAIL (exit $rc)"; BAD=$((BAD+1)); fi

cat > "$WORK/probe.py" <<'PYEOF'
import os, shutil, subprocess, sys, tempfile
COMPARE = sys.argv[1]
bad = []

def write_run(d, header, rows):
    os.makedirs(d)
    with open(os.path.join(d, "performance.txt"), "w") as f:
        f.write(header + "\n")
        for r in sorted(rows):
            f.write("%d, %s\n" % (r, ", ".join("%g" % v for v in rows[r])))

def run(header, offs, on, cols="Compute,MPI,IO,Other"):
    tmp = tempfile.mkdtemp(prefix="probe_")
    try:
        od = []
        for i, rows in enumerate(offs):
            d = os.path.join(tmp, "off%d" % i); write_run(d, header, rows); od.append(d)
        on_d = os.path.join(tmp, "on"); write_run(on_d, header, on)
        p = subprocess.Popen([sys.executable, "-I", COMPARE, "perf-diff", "--on", on_d, "--off"] + od + ["--cols", cols],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.communicate()[0].decode()
        return p.returncode, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

H = "%Rank, Compute, MPI, IO, Other, DIAG"
def offs3(n):
    return [dict((r, (3.0, 0.5, 1.0, 0.05, 0.0)) for r in range(n)) for _ in range(3)]

def expect(name, got, want):
    rc, out = got
    ok = (rc == want) and ("Traceback" not in out)
    print("%-70s exit %d (wanted %d) %s" % (name, rc, want, "ok" if ok else "WRONG"))
    if not ok:
        bad.append(name); print(out)

# MPI band here: spread 0 -> max(0, 0.05, 0.5 x 0.5) = 0.25
# P1 three ranks; the maximum DIAG is on rank 2 (1.0), rank 0 has 0.9, rank 1 has 0.2
def three(d0, d1):
    return {0: (3.0, 0.5 + d0, 1.0, 0.05, 0.9), 1: (3.0, 0.5 + d1, 1.0, 0.05, 0.2), 2: (3.0, 0.5, 1.0, 0.05, 1.0)}
expect("P1a 3 ranks: rank 1 delta 0.9 <= gap 0.8 + band 0.25 passes", run(H, offs3(3), three(0.0, 0.9)), 0)
expect("P1b 3 ranks: rank 1 delta 1.1 > 1.05 FAILS", run(H, offs3(3), three(0.0, 1.1)), 1)
expect("P1c 3 ranks: rank 0 delta 0.3 <= gap 0.1 + band 0.25 passes", run(H, offs3(3), three(0.3, 0.0)), 0)
expect("P1d 3 ranks: rank 0 delta 0.4 > 0.35 FAILS (the allowance is per rank, not the largest gap)", run(H, offs3(3), three(0.4, 0.0)), 1)
# P2 boundary on two ranks, gap 0.6, band 0.25: threshold 0.85
def two(d1):
    return {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.5 + d1, 1.0, 0.05, 0.4)}
expect("P2a delta 0.84 passes (threshold 0.85)", run(H, offs3(2), two(0.84)), 0)
expect("P2b delta 0.86 FAILS (threshold 0.85)", run(H, offs3(2), two(0.86)), 1)
# P3 a header without a DIAG column: no allowance, no crash
H0 = "%Rank, Compute, MPI, IO, Other"
on_nodiag = lambda d1: {0: (3.0, 0.5, 1.0, 0.05), 1: (3.0, 0.5 + d1, 1.0, 0.05)}
offs_nodiag = [dict((r, (3.0, 0.5, 1.0, 0.05)) for r in range(2)) for _ in range(3)]
expect("P3a no DIAG column, MPI delta 0.2 in band passes", run(H0, offs_nodiag, on_nodiag(0.2)), 0)
expect("P3b no DIAG column, MPI delta 0.4 > band 0.25 FAILS (no allowance without DIAG)", run(H0, offs_nodiag, on_nodiag(0.4)), 1)
# P4 equal DIAG on every rank: no allowance
eq = {0: (3.0, 0.5, 1.0, 0.05, 0.7), 1: (3.0, 0.8, 1.0, 0.05, 0.7)}
expect("P4 equal DIAG, MPI delta 0.3 > band 0.25 FAILS", run(H, offs3(2), eq), 1)
# P5 DIAG zero on every rank (a capture-off run presented as capture-on): no allowance
zero = {0: (3.0, 0.5, 1.0, 0.05, 0.0), 1: (3.0, 0.8, 1.0, 0.05, 0.0)}
expect("P5 DIAG zero everywhere, MPI delta 0.3 FAILS", run(H, offs3(2), zero), 1)
# P6 a rank with a gap but a Compute excursion: still fails (Compute never takes the allowance), sized between band and band + gap
# Compute band = 0.15 x 3.0 = 0.45; delta 0.8 lies between 0.45 and 0.45 + 0.6
cx = {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.8, 0.5, 1.0, 0.05, 0.4)}
expect("P6 Compute delta 0.8 (band 0.45, gap 0.6) FAILS", run(H, offs3(2), cx), 1)
# P7 same for IO: IO band = 0.5 x 1.0 = 0.5; delta 0.8 lies between 0.5 and 1.1
iox = {0: (3.0, 0.5, 1.0, 0.05, 1.0), 1: (3.0, 0.5, 1.8, 0.05, 0.4)}
expect("P7 IO delta 0.8 (band 0.5, gap 0.6) FAILS", run(H, offs3(2), iox), 1)
# P8 the case the restatement cannot separate: a NON-capture MPI cost on a non-busiest rank below gap + band passes
expect("P8 non-capture MPI cost 0.84 on rank 1 (below gap + band) PASSES: a stated blind region", run(H, offs3(2), two(0.84)), 0)

# P9 SWMM_MPI (a column the suite's N2 and N3 rows pass): an excursion between band and band + gap still FAILS
HS = "%Rank, Compute, MPI, IO, Other, SWMM_MPI, DIAG"
offs_s = [dict((r, (3.0, 0.5, 1.0, 0.05, 0.3, 0.0)) for r in range(2)) for _ in range(3)]
on_s = {0: (3.0, 0.5, 1.0, 0.05, 0.3, 1.0), 1: (3.0, 0.5, 1.0, 0.05, 0.7, 0.4)}
expect("P9 SWMM_MPI delta 0.4 (band 0.05, gap 0.6) FAILS", run(HS, offs_s, on_s, cols="SWMM_MPI"), 1)

print("PROBES " + ("PASS" if not bad else "FAIL: %s" % bad))
sys.exit(0 if not bad else 1)
PYEOF

echo "=== B. mutants of the comparator, each run through the coder's unit test"
cat > "$WORK/mutate.py" <<'PYEOF'
import sys
src, dst, old, new = sys.argv[1:5]
t = open(src).read()
n = t.count(old)
if n != 1:
    sys.stderr.write("pattern found %d times (needs exactly 1): %r\n" % (n, old))
    sys.exit(3)
open(dst, "w").write(t.replace(old, new))
PYEOF
S_OUT='out = delta > band + allow or delta < -band'
S_ALLOW='allow = max(diags) - on[r]["DIAG"]'
S_IF='if c == "MPI":'
S_DIAGS='diags = [on[k]["DIAG"] for k in on if "DIAG" in on[k]]'
# id | pattern | replacement | EXPECT (kill or survive-equivalent or survive-testgap) | note
run_mutant() {
  local id=$1 old=$2 new=$3 expect=$4 note=$5
  mkdir -p "$WORK/$id" && cp "$T/test_neutrality_perf_diff.py" "$WORK/$id/"
  python3 -I "$WORK/mutate.py" "$T/neutrality_compare.py" "$WORK/$id/neutrality_compare.py" "$old" "$new" || { echo "$id MUTATOR-ERROR"; BAD=$((BAD+1)); return; }
  ( cd "$WORK/$id" && python3 -I test_neutrality_perf_diff.py > "$WORK/$id.log" 2>&1 ); local rc=$?
  local failed; failed=$(grep -c "WRONG" "$WORK/$id.log")
  python3 -I "$WORK/probe.py" "$WORK/$id/neutrality_compare.py" > "$WORK/$id.probe.log" 2>&1; local prc=$?
  local pk="probes survive"; [ $prc -ne 0 ] && pk="probes KILL ($(grep -c WRONG "$WORK/$id.probe.log") WRONG rows)"
  if [ $rc -ne 0 ]; then
    if [ "$expect" = kill ]; then echo "$id KILLED ($failed WRONG rows)  $note"
    else echo "$id KILLED-UNEXPECTEDLY ($failed WRONG rows)  $note"; BAD=$((BAD+1)); fi
  else
    if [ "$expect" = kill ]; then echo "$id SURVIVED  $note"; BAD=$((BAD+1))
    elif [ "$expect" = survive-testgap ] && [ $prc -eq 0 ]; then echo "$id SURVIVED both the unit test and the probes  $note"; BAD=$((BAD+1))
    else echo "$id SURVIVED-EXPECTED ($expect; $pk)  $note"; fi
  fi
}
run_mutant M01 "$S_OUT" 'out = abs(delta) > band' kill "allowance dropped (the unrestated check)"
run_mutant M02 "$S_IF" 'if True:' kill "allowance on every column"
run_mutant M03 "$S_ALLOW" 'allow = max(diags)' kill "allowance not net of the rank's own DIAG"
run_mutant M04 "$S_ALLOW" 'allow = max(diags) - min(diags)' kill "allowance is the DIAG spread, nonzero on the busiest rank"
run_mutant M05 "$S_ALLOW" 'allow = sum(diags) - on[r]["DIAG"]' kill "allowance from the sum of the other ranks' DIAG"
run_mutant M06 "$S_OUT" 'out = delta > band + allow' kill "lower side of the band dropped"
run_mutant M07 "$S_OUT" 'out = abs(delta) > band + allow' kill "allowance applied to the lower side too"
run_mutant M08 "$S_ALLOW" 'allow = 2.0 * (max(diags) - on[r]["DIAG"])' kill "allowance doubled"
run_mutant M09 "$S_DIAGS" 'diags = [on[k]["DIAG"] for k in on if k != 0 and "DIAG" in on[k]]' kill "rank 0 left out of the maximum"
run_mutant M10 "$S_ALLOW" 'allow = min(diags) - on[r]["DIAG"]' kill "minimum in place of the maximum"
run_mutant M11 "$S_OUT" 'out = delta >= band + allow or delta < -band' survive-equivalent "boundary equality: a measured float never lands on it"
run_mutant M12 "$S_IF" 'if c in ("MPI", "IO"):' survive-testgap "allowance leaks to IO: the unit test's IO excursion is sized above band + gap, so it cannot see the leak"
run_mutant M13 "$S_IF" 'if c in ("MPI", "Compute"):' survive-testgap "allowance leaks to Compute: the unit test's Compute excursion is sized above band + gap"
run_mutant M14 "$S_IF" 'if c in ("MPI", "Other"):' kill "allowance leaks to Other (the unit test's Other excursion is inside band + gap)"
run_mutant M15 "$S_IF" 'if c in ("MPI", "SWMM_MPI"):' survive-testgap "allowance leaks to SWMM_MPI: the unit test never passes that column, though the suite's N2 and N3 rows do"

echo "=== C. synthetic probes of the restated check"
python3 -I "$WORK/probe.py" "$T/neutrality_compare.py" > "$WORK/C.log" 2>&1; rc=$?
cat "$WORK/C.log"
[ $rc -eq 0 ] || BAD=$((BAD+1))

echo "=== D. replay of the round-4 per-run data (each capture-on run against off_1 off_2 off_3)"
if [ -z "$R4" ]; then
  echo "D not run: no R4_DIR argument"
elif [ ! -d "$R4" ]; then
  echo "D FAIL: $R4 is not a directory"; BAD=$((BAD+1))
else
  CPU_COLS=Compute,MPI,IO,Resize,SWMM,SWMM_XFER,SWMM_MPI,SWMM_STEP,SWMM_OTHER,SWMM_SOLVE,SWMM_LOG,Other
  GPU_COLS=Compute,MPI,IO,Resize,SWMM,SWMM_SOLVE,SWMM_LOG,Other
  for cs in n2np8 n3np8 n6np2gd0 n6np2gd1; do
    cols=$CPU_COLS; case $cs in n6*) cols=$GPU_COLS;; esac
    okr=0; okm=0; n=0
    for k in 1 2 3 4 5 6; do
      d="$R4/$cs/on_$k/out"
      [ -f "$d/performance.txt" ] || { echo "D $cs on_$k: no performance.txt"; BAD=$((BAD+1)); continue; }
      n=$((n+1))
      python3 -I "$T/neutrality_compare.py" perf-diff --on "$d" --off "$R4/$cs/off_1/out" "$R4/$cs/off_2/out" "$R4/$cs/off_3/out" --cols "$cols" --abs-floor 0.05 > "$WORK/D_${cs}_${k}.restated.log" 2>&1; rcr=$?
      python3 -I "$WORK/M01/neutrality_compare.py" perf-diff --on "$d" --off "$R4/$cs/off_1/out" "$R4/$cs/off_2/out" "$R4/$cs/off_3/out" --cols "$cols" --abs-floor 0.05 > "$WORK/D_${cs}_${k}.unrestated.log" 2>&1; rcu=$?
      [ $rcr -eq 0 ] && okr=$((okr+1)); [ $rcu -eq 0 ] && okm=$((okm+1))
      line=$(grep -E "^  rank [0-9]+ MPI " "$WORK/D_${cs}_${k}.restated.log" | grep -E "OUT|allowance" | head -2 | tr '\n' ';')
      echo "D $cs on_$k: restated exit $rcr, unrestated exit $rcu   $line"
    done
    echo "D SUMMARY $cs: $n capture-on runs; restated passes $okr, unrestated passes $okm"
  done
fi

echo "REVIEW_TEST_EXIT=$BAD"
[ $BAD -eq 0 ]
