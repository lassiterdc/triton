#!/usr/bin/env python3
"""T1a REVIEW findings, authored by the Build reviewing seat as assertions.

Three findings, each measured before it was written and each red on the tree as
landed.  None of them touches the coder's three paths; this file is the
reviewer's whole finding channel.

WHY THESE THREE AND NOT A PROSE NOTE
------------------------------------
All three are the same shape of defect at three depths: an instrument that
returns a confident verdict about a property it cannot observe.  R1 is a
reviewer PATH that cannot observe the change it is the declared path for; R2 is
a POST-CONDITION that cannot observe the text it certifies absent; R3 is a code
COMMENT that describes live behaviour the source contradicts.  A prose note
about any of them is discharged by being read.  An assertion is discharged only
by the property becoming true.

R1 -- THE RELEVANCE PROPERTY (axis 8)
-------------------------------------
A declared verification path is discharged only when it is REACHABLE,
INDEPENDENT, and RELEVANT.  The first two were checked by prior parties; the
third is measured here for the first time.

``test/test_swmm_guard_predicate_closure.py`` is T1a's declared reviewer path.
It is reachable (exit 0) and independent (absent from the coder's write set).
It is NOT relevant: it reads only ``src/triton.h``, plus ``src/*.h`` for G4a's
stray scan, and never opens ``external/swmm/src/solver/snapshot.c`` or
``src/swmm_triton.h`` -- which between them carry every line of T1a.1, T1a.2,
and the whole of T1a.3's and T1a.4's solver halves.

Measured: its verdict set is byte-identical on the pre-fix base and the landed
tree -- 10 verdict lines, same sha256, exit 0 on both.  The only differences in
its output are the advisory line numbers in its own informational listings,
which move because the coder inserted lines above them.

This assertion ranges over EVERY ``test/test_*.py``, not over that one file,
because the finding is a COVERAGE finding and the remedy is a relevant path
rather than a repair to an existing one.  It greens the moment any test under
``test/`` returns a different verdict set on the two trees.

NOTE ON OWNERSHIP, because this one may not be the coder's to green.  The shape
assigns the regression tests for the payload length and the killed-write
consultation to item 4, which is ``T2a.1``, gated behind ``T1a.4``.  So the
coverage that WOULD discriminate T1a arrives one work package after T1a is
committed.  If that is the intended sequencing then this red is a property of
the partition rather than of the package, and it belongs to the coordinator as
a finding against the criterion -- which is what the reviewer's report says.
Either way the red is the honest state and is not satisfied by reading about it.

R2 -- THE POST-CONDITION INSTRUMENT (axis 5)
--------------------------------------------
T1a.3 certifies its four-site refusal repair with a single-sided post-condition
over the wrong shape, which is the correct FORM.  The INSTRUMENT is a
line-anchored ``grep -c``, and every long message in ``snapshot.c`` is written
as adjacent string literals wrapped across source lines, so a line-anchored
grep cannot see any message in full.

Measured, re-running the recorded command verbatim:

    $ grep -c "short or unreadable" external/swmm/src/solver/snapshot.c
    0

and the phrase is in the file, at :1435-1436, wrapped across two comment lines.
The same instrument returns 0 for
``the vendored EPA structs changed between the writing and reading builds``
on a LIVE ``snprintf`` message at :1425-1427 that says exactly that.

So the certifying instrument returns a clean zero on a surviving misattribution.
The verdicts it recorded are nevertheless CORRECT -- the live payload-arm message
was genuinely replaced and the surviving occurrence is a retrospective quote in
a comment.  What is unfounded is the evidence, and the consequence is forward:
the fifth site is routed to a later round carrying a post-condition that cannot
verify the repair it is routed for.

This assertion is deliberately scoped to the INSTRUMENT and says nothing about
whether the fifth site should be repaired.  That routing was declined on scope
and accepted, and widening a settled enumeration from inside the implementing
stage is the one-way amendment the loop rules prevent.

R3 -- TWO COMMENT CLAIMS THE SOURCE CONTRADICTS (axis 3)
--------------------------------------------------------
The ``SNAP_COUNT`` enum comment asserts two properties of live behaviour:

  (i)  SNAP_COUNT "touches no file and no model memory".
  (ii) ``snap_d``/``snap_i``/``snap_l``/``snap_c`` "increments c->count
       unconditionally and then branches".

Both are false at live source.

(i)  ``SNAP_P``'s non-MANIFEST branch is ``&((arr)[i])``, so a COUNT pass walks
     the real arrays; and the re-predicated Outfall guard, gated only on
     ``mode != SNAP_MANIFEST``, LOADS ``s->routeTo`` and ``s->wRouted`` from
     ``Outfall[i]`` in every non-MANIFEST mode including SNAP_COUNT.  That is a
     model-memory read introduced by T1a.1 into the mode T1a.2 added.
     It is NOT a live crash: the pre-loop ``Nnodes[OUTFALL] > 0 && !Outfall``
     guard also fires in COUNT mode, so the element address is never formed on
     a null array.  The cost is in the record, not in the run -- and the record
     matters here because this exact sentence is the shape's own stated ground
     for closing its open item on whether the count mode forces a guard edit.
     The conclusion survives; the reason does not.  The guards protect COUNT
     mode because they FIRE in COUNT mode, not because COUNT dereferences
     nothing.

(ii) ``snap_d`` opens ``if ( c->error ) return;`` immediately before
     ``c->count++``, so the increment is error-gated.  The same file states the
     true form 530 lines below, in the Outfall guard's own comment -- "snap_d
     returns immediately on it" -- so the file asserts both halves of a
     contradiction and the source settles it against the SNAP_COUNT comment.

Each sub-assertion pairs a CLAIM match with a SOURCE match and fires only when
both are present, so correcting either side greens it.  Neither can be greened
by rewording this test.

USAGE
-----
    python3 test/test_review_t1a_relevance_and_instruments.py [repo-root]

Exit 0 when every assertion holds, 1 otherwise.  No compilation, no solver run:
every instrument reads source text or runs a sibling source-text test.
"""

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_REF = "031ea42b"

SNAPSHOT_C = Path("external/swmm/src/solver/snapshot.c")
SWMM_TRITON_H = Path("src/swmm_triton.h")
TRITON_H = Path("src/triton.h")

# The three paths T1a declares. R1 asks which of them any test under test/ reads.
T1A_WRITE_SET = (SNAPSHOT_C, SWMM_TRITON_H, TRITON_H)

# The tier uses two verdict dialects: `ok   `/`FAIL ` plus a terminal `PASS:`,
# and `PASS <name>`/`FAIL <name>` per subtest. Both are matched, because a
# regex covering only the first silently reports "no verdict line" on four
# files and a reader would take that for a measurement.
VERDICT_RE = re.compile(r"^(?:ok   |FAIL |PASS:|FAIL:|PASS |SKIP )", re.M)


# --------------------------------------------------------------------------
# shared text helpers
# --------------------------------------------------------------------------

def join_text(src):
    """Collapse C line-wrapping so a phrase split across source lines is
    findable.  Comment markers are blanked rather than removed so a wrapped
    comment reads as running prose; whitespace collapses to single spaces.

    This is the instrument R2 says the post-condition needed and did not use."""
    return re.sub(r"\s+", " ", src.replace("//", " "))


def strip_comments(src):
    """Return the source with every // tail and every /* */ block blanked, so
    what remains is code -- string literals included.  Newlines are preserved
    so line numbers survive."""
    out = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                 src, flags=re.S)
    return "\n".join(line.split("//")[0] for line in out.split("\n"))


def live_message_text(src):
    """The concatenated contents of every double-quoted string literal in the
    CODE (comments already blanked), joined so a message wrapped across
    adjacent literals reads as one string."""
    code = strip_comments(src)
    return re.sub(r"\s+", " ", " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', code)))


def line_grep_count(path, phrase):
    """Exactly what `grep -c -- phrase path` counts: LINES carrying the phrase."""
    return sum(1 for ln in path.read_text().split("\n") if phrase in ln)


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------

def export_base(repo, dest):
    """Materialise BASE_REF into dest via git archive.  Returns True on success."""
    try:
        archive = subprocess.run(["git", "-C", str(repo), "archive", BASE_REF],
                                 stdout=subprocess.PIPE, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    try:
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def verdict_digest(test_path, tree):
    """Run a sibling source-text test against `tree` and digest its VERDICT
    lines only.  Advisory line numbers live on non-verdict lines, so they do
    not enter the digest -- which is the whole point: a test whose verdicts are
    invariant under the change has not observed the change."""
    try:
        p = subprocess.run([sys.executable, str(test_path), str(tree)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, "could not run (%s)" % type(exc).__name__
    text = p.stdout.decode("utf-8", "replace")
    verdicts = [m.group(0) + text[m.end():].split("\n")[0]
                for m in VERDICT_RE.finditer(text)]
    if not verdicts:
        return None, "emitted no verdict line"
    digest = hashlib.sha256("\n".join(verdicts).encode()).hexdigest()
    return (digest, "%d verdict line(s), exit %d" % (len(verdicts), p.returncode))


def check_r1(repo, check):
    tests = sorted(p for p in (repo / "test").glob("test_*.py")
                   if p.name != Path(__file__).name)

    # R1a: the declared path's READ SURFACE. Stated as an assertion because the
    # relevance failure's cause is this and nothing else: a test that never
    # opens a file cannot discriminate a change confined to it.
    declared = repo / "test" / "test_swmm_guard_predicate_closure.py"
    if declared.is_file():
        body = declared.read_text()
        # Scoped to the two paths that carry T1a.1 and T1a.2. src/triton.h is
        # DELIBERATELY excluded from this operand set: it is in T1a's write set,
        # so including it makes the assertion pass on a path that names triton.h
        # and nothing else -- which is exactly the state being measured. An
        # assertion satisfied by the condition it is testing for is not one.
        carriers = (SNAPSHOT_C, SWMM_TRITON_H)
        named = [p.name for p in carriers
                 if p.name in body or str(p) in body]
        check("R1a T1a's declared reviewer path names at least one of the two "
              "paths carrying T1a.1 and T1a.2 (%d of 2 named)" % len(named),
              bool(named),
              "names neither %s -- it reads src/triton.h only, so T1a.1 "
              "entirely, T1a.2 entirely, and the solver halves of T1a.3 and "
              "T1a.4 are outside everything it can observe"
              % " nor ".join(p.name for p in carriers))
    else:
        check("R1a T1a's declared reviewer path exists", False,
              "%s not found" % declared)

    # R1b: the discrimination measurement itself, over the WHOLE test tier.
    tmp = Path(tempfile.mkdtemp(prefix="t1a-base-"))
    try:
        if not export_base(repo, tmp):
            check("R1b at least one test under test/ returns a different "
                  "verdict set on the pre-fix base than on this tree", False,
                  "could not export %s -- measurement not performed, and an "
                  "unperformed measurement is not a pass" % BASE_REF)
            return
        rows, discriminating, unmeasurable = [], [], []
        for t in tests:
            d_now, n_now = verdict_digest(t, repo)
            d_base, n_base = verdict_digest(t, tmp)
            if d_now is None or d_base is None:
                # NOT-MEASURABLE is its own state and must not render as
                # "differs". A file this harness cannot run is evidence about
                # the harness, and showing it as a difference would manufacture
                # the discrimination the assertion is looking for.
                state = "NOT MEASURABLE"
                unmeasurable.append(t.name)
            elif d_now == d_base:
                state = "INVARIANT"
            else:
                state = "DISCRIMINATES"
                discriminating.append(t.name)
            rows.append((t.name, n_now, n_base, state))
        print("")
        print("R1b per-test discrimination across %s -> this tree:" % BASE_REF)
        for name, n_now, n_base, state in rows:
            print("        %-62s %s" % (name, state))
            print("              landed: %-28s base: %s" % (n_now, n_base))
        measured = len(rows) - len(unmeasurable)
        print("")
        print("        measured %d of %d test file(s); %d not measurable by "
              "this harness" % (measured, len(rows), len(unmeasurable)))
        print("")
        check("R1b at least one test under test/ discriminates the T1a change "
              "(%d of %d measured do)" % (len(discriminating), measured),
              bool(discriminating),
              "every MEASURED test in the tier returns an identical verdict "
              "set on the pre-fix base and on this tree, so no runnable "
              "instrument in this repository can tell the two apart "
              "(%d file(s) not measurable: %s)"
              % (len(unmeasurable), ", ".join(unmeasurable) or "none"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------

# Each phrase T1a.3's post-condition certifies absent, with the file it was
# certified against. The assertion is that a line-anchored count and a
# concatenation-aware count AGREE -- not that either reaches any value.
POSTCONDITION_PHRASES = (
    (SNAPSHOT_C, "short or unreadable"),
    (SNAPSHOT_C, "to resume via the exchange"),
    (SWMM_TRITON_H, "to resume via the exchange"),
    (TRITON_H, "to resume via the exchange"),
    (SNAPSHOT_C, "written by a different build or is corrupt"),
    (SWMM_TRITON_H, "written by a different build or is corrupt"),
    (TRITON_H, "written by a different build or is corrupt"),
)


def check_r2(repo, check):
    disagree = []
    print("R2 post-condition instrument -- line-anchored count vs "
          "concatenation-aware count:")
    for rel, phrase in POSTCONDITION_PHRASES:
        path = repo / rel
        if not path.is_file():
            disagree.append("%s missing" % rel)
            continue
        n_line = line_grep_count(path, phrase)
        n_join = join_text(path.read_text()).count(phrase)
        flag = "" if n_line == n_join else "   <-- DISAGREE"
        print("        %-34s %-46s grep -c=%d  joined=%d%s"
              % (rel.name, '"%s"' % phrase, n_line, n_join, flag))
        if n_line != n_join:
            disagree.append('%s: "%s" grep -c=%d but joined=%d'
                            % (rel.name, phrase, n_line, n_join))
    print("")
    check("R2  every phrase T1a.3's post-condition certifies absent is counted "
          "identically by a line-anchored grep and by a concatenation-aware "
          "scan (%d disagreement(s))" % len(disagree),
          not disagree,
          "; ".join(disagree) + " -- the recorded post-condition cannot "
          "observe a phrase wrapped across source lines, which is how every "
          "long message in this file is written")


# --------------------------------------------------------------------------
# R3
# --------------------------------------------------------------------------

def check_r3(repo, check):
    path = repo / SNAPSHOT_C
    if not path.is_file():
        check("R3  snapshot.c is readable", False, "%s not found" % SNAPSHOT_C)
        return
    src = path.read_text()
    joined = join_text(src)
    code = strip_comments(src)

    # R3a -- the model-memory claim.
    claim_no_memory = "touches no file and no model memory" in joined
    snap_p_real = re.search(r"define\s+SNAP_P\(.*?&\(\(arr\)\[i\]\)", joined) is not None
    guard_loads_element = re.search(
        r"mode\s*!=\s*SNAP_MANIFEST[^;{]*?s->routeTo", code, re.S) is not None
    print("R3a  claim 'touches no file and no model memory' present: %s"
          % claim_no_memory)
    print("     SNAP_P non-MANIFEST branch yields &((arr)[i]):        %s"
          % snap_p_real)
    print("     a mode != SNAP_MANIFEST predicate loads s->routeTo:   %s"
          % guard_loads_element)
    check("R3a no comment claims a non-MANIFEST traversal mode touches no model "
          "memory while a mode != SNAP_MANIFEST predicate loads a field of the "
          "per-element pointer",
          not (claim_no_memory and snap_p_real and guard_loads_element),
          "SNAP_P's non-MANIFEST branch is &((arr)[i]) and the re-predicated "
          "Outfall guard loads s->routeTo and s->wRouted in every non-MANIFEST "
          "mode, SNAP_COUNT included -- so the SNAP_COUNT comment's "
          "no-model-memory claim is false, and it is the shape's own recorded "
          "ground for closing its count-mode open item")

    # R3b -- the unconditional-increment claim.
    claim_uncond = re.search(r"increments c->count unconditionally", joined) is not None
    snap_d_body = re.search(
        r"static void snap_d\s*\([^)]*\)\s*\{(.*?)\n\}", code, re.S)
    gated = False
    if snap_d_body:
        head = snap_d_body.group(1)
        i_ret = head.find("if ( c->error ) return;")
        i_inc = head.find("c->count++")
        gated = 0 <= i_ret < i_inc
    print("R3b  claim 'increments c->count unconditionally' present:  %s"
          % claim_uncond)
    print("     snap_d's error-gated return precedes c->count++:      %s" % gated)
    check("R3b no comment claims the snap_* family increments c->count "
          "unconditionally while snap_d's body returns on c->error before the "
          "increment",
          not (claim_uncond and gated),
          "snap_d opens 'if ( c->error ) return;' immediately before "
          "'c->count++', so the increment is error-gated; the Outfall guard's "
          "own comment states the true form ('snap_d returns immediately on "
          "it'), so the file asserts both halves of a contradiction")


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# R4 -- round 3. ITEM 1's DECLARING TEST COVERS TWO OF ITS THREE DELIVERABLES
# --------------------------------------------------------------------------
#
# Item 1's partition row lands THREE things in snapshot.c: the three-way
# pointer selection, the re-predicated-and-repositioned guard, and
# SNAPSHOT_VERSION 5 -> 6. The declared detecting test
# `test/test_snapshot_traversal_guard_source.py` asserts the SECOND and the
# THIRD. It names the FIRST nowhere -- measured: zero references to
# `wRoutedPlaceholder`, `slot`, `_snapDummyScalar`, `s->wRouted[k]` or
# "placeholder" anywhere in the file.
#
# That is faithful to the criterion: the live shape enumerates exactly three
# assertions under "WHAT THE TEST MUST DETECT" and the three-way selection is
# not among them. So this is a finding against the ENUMERATION, not against
# the coder, who implemented all three plus two vacuity guards.
#
# WHY IT MATTERS RATHER THAN BEING A TIDINESS POINT. Reverting the three-way
# selection ALONE -- back to the pre-fix two-way ternary, guard untouched,
# version untouched -- is the shape's OPTION A, which the shape declined in
# these words: "it crashes ... it kills the process. Not admissible."
# And it is strictly WORSE than the pre-fix tree:
#
#   pre-fix   guard `Nobjects[POLLUT] > 0 && !s->wRouted` FIRES on a
#             route-less outfall -> c->error, return -> short file, refused.
#   mutant    guard carries `s->routeTo >= 0`, so on a route-less outfall the
#             conjunct is FALSE, the guard does NOT fire, and the loop
#             dereferences &s->wRouted[k] with wRouted == NULL.
#
# REACHABILITY, by artifact rather than by argument: node.c:163 sets
# `wRouted = NULL` and :166 allocates only under `routeTo >= 0`, and the
# campaign model is the input class -- the shape's own DoD 2 names Norfolk
# 0.35 m as carrying `Nobjects[POLLUT] >= 1` and ONE ROUTE-LESS OUTFALL.
#
# So the two states the declared test cannot tell apart are "the repair as
# landed" and "the one option the shape refused for crashing".
#
# NOT-APPLICABLE IS NOT GREEN. On a tree with no three-way block (a pre-fix
# export) the mutant cannot be built, and this reports NOT APPLICABLE rather
# than passing -- a silent green there would be exactly the vacuous pass R1
# was filed about.

_THREE_WAY_BLOCK = """            double* slot;
            snap_name(c, "Outfall", "wRouted");
            if      ( c->mode == SNAP_MANIFEST ) slot = &_snapDummyScalar;
            else if ( s->wRouted )               slot = &s->wRouted[k];
            else                                 slot = &wRoutedPlaceholder;
            snap_d(c, slot);"""

_TWO_WAY_TERNARY = """            snap_name(c, "Outfall", "wRouted");
            snap_d(c, (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar
                                                 : &s->wRouted[k]);"""

DECLARED_ITEM1_TEST = Path("test/test_snapshot_traversal_guard_source.py")


def check_r4(repo, check):
    declared = repo / DECLARED_ITEM1_TEST
    if not declared.is_file():
        check("R4  item 1's declared detecting test exists", False,
              "%s not found" % DECLARED_ITEM1_TEST)
        return

    # R4a -- the read-surface half, stated as an assertion rather than left as
    # the mutant's explanation. Any ONE of these tokens would put the
    # three-way selection inside the test's subject set.
    body = declared.read_text()
    tokens = ("wRoutedPlaceholder", "slot", "_snapDummyScalar",
              "s->wRouted[k]", "placeholder")
    named = [tok for tok in tokens if tok in body]
    check("R4a item 1's declared test names at least one operand of the "
          "three-way pointer selection (%d of %d named)"
          % (len(named), len(tokens)),
          bool(named),
          "names none of %s, so the three-way selection is outside everything "
          "the declared test can observe" % ", ".join(tokens))

    target = repo / SNAPSHOT_C
    src = target.read_text() if target.is_file() else ""
    if _THREE_WAY_BLOCK not in src:
        print("R4b NOT APPLICABLE -- this tree carries no three-way selection "
              "block, so the mutant cannot be constructed here. Reported as "
              "not-applicable rather than green.")
        return

    tmp = Path(tempfile.mkdtemp(prefix="t1a-m1-"))
    try:
        for sub in ("external", "test"):
            if (repo / sub).is_dir():
                shutil.copytree(repo / sub, tmp / sub, symlinks=True)
        mutant_c = tmp / SNAPSHOT_C
        mutant_c.write_text(src.replace(_THREE_WAY_BLOCK, _TWO_WAY_TERNARY, 1))
        try:
            proc = subprocess.run(
                [sys.executable, str(declared), str(tmp)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            rc = proc.returncode
            _out = proc.stdout.decode("utf-8", "replace").strip()
            detail_tail = _out.splitlines()[-1] if _out else "(no output)"
        except (subprocess.TimeoutExpired, OSError) as exc:
            check("R4b item 1's declared test discriminates a revert of the "
                  "three-way pointer selection", False,
                  "could not run the declared test on the mutant (%s)"
                  % type(exc).__name__)
            return
        print("")
        print("R4b OPTION-A MUTANT: three-way selection -> pre-fix two-way "
              "ternary; guard and SNAPSHOT_VERSION untouched")
        print("        declared test on mutant: exit %d" % rc)
        print("        terminal line          : %s" % detail_tail)
        print("")
        check("R4b item 1's declared test discriminates a revert of the "
              "three-way pointer selection (mutant exit %d)" % rc,
              rc != 0,
              "the mutant passes every assertion and exits 0, so the declared "
              "test cannot tell the landed repair from the shape's Option A, "
              "which the shape declined because it crashes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv):
    repo = Path(argv[1] if len(argv) > 1 else ".").resolve()
    failures = []

    def check(label, ok, detail=""):
        print("%s %s%s" % ("ok   " if ok else "FAIL ", label,
                           "" if ok else ("  -- %s" % detail)))
        if not ok:
            failures.append(label)

    if not (repo / SNAPSHOT_C).is_file() and not (repo / TRITON_H).is_file():
        print("FAIL: %s does not look like the TRITON fork" % repo)
        return 1

    check_r1(repo, check)
    check_r2(repo, check)
    check_r3(repo, check)
    check_r4(repo, check)
    print("")

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: every T1a review assertion holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
