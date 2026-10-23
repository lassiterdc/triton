#!/usr/bin/env python3
"""Source-text check over external/swmm/src/solver/snapshot.c.

THE PROBLEM THIS FILE SOLVES
----------------------------
Work-package item 1 repaired the snapshot traversal's Outfall guards, and
NOTHING IN THE SUITE COULD DETECT A REVERT OF IT.  Measured before this file
existed: the two nearest candidates, ``test_snapshot_inventory_criterion_p.py``
and ``test_snapshot_inventory_closure_modes.py``, carry zero references to
``wRouted``, ``routeTo``, ``SNAP_MANIFEST`` or ``Nobjects[POLLUT]``, and their
verdict output is byte-identical on the pre-fix and post-fix trees.  A test
that never names a predicate cannot discriminate a change to it.

The repair is not expressible as a behavioural unit test from this tree: the
guards sit inside a vendored EPA solver traversal whose failure mode on a
narrowed guard is a null dereference, and exercising it needs a compiled
coupled solver on an HPC venue.  So the check is SOURCE TEXT -- it reads
snapshot.c as text, compiles nothing and runs no solver, which is what lets it
sit in the fast tier and run on every edit.

THE FOUR ASSERTIONS, AND THEY ARE NOT ALL THE SAME KIND
-------------------------------------------------------
The framing distinction is the triton reviewing seat's, carried in its words:

    "assertions 1 and 3 are detectors keyed on properties item 1 changed;
     assertion 2 is an anti-narrowing invariant keyed on a property item 1
     deliberately did not change, and is correctly green on both trees."

Holding the retired single framing -- that all three redden on a pre-fix tree
-- predicts that A2 reddens pre-fix.  It does not, and reading that as a
broken test is the misdiagnosis this paragraph exists to prevent.

  A1  DETECTOR.  The per-element Outfall guard GAINS ``s->routeTo >= 0`` and
      RETAINS the pollutant-count conjunct.  0 pre-fix, 1 post-fix.
  A2  ANTI-NARROWING INVARIANT.  The pre-loop guard still admits the counting
      mode.  Green on both trees; it fails only if someone narrows it.
  A3  DETECTOR.  ``SNAPSHOT_VERSION`` is 6 where pre-fix carries 5.
  A4  DETECTOR.  The three-way pointer selection that chooses ``slot`` is the
      landed three-arm ``if / else if / else`` chain, not the pre-fix two-way
      ternary.  1 post-fix, 0 pre-fix.

WHY A4 EXISTS AT ALL -- IT CLOSES A GAP IN THE ENUMERATION, NOT A DEVIATION
---------------------------------------------------------------------------
Item 1's work-package row lands THREE deliverables: the three-way pointer
selection, the re-predicated and repositioned guard, and ``SNAPSHOT_VERSION``
5 to 6.  A1/A2/A3 cover the second and third.  MEASURED, bare token counts over
the 268-line instrument as it stood before this assertion existed, UNIT =
occurrences in the test file: ``wRoutedPlaceholder`` 0, ``slot`` 0,
``_snapDummyScalar`` 0, ``s->wRouted[k]`` 0, ``placeholder`` 0.  No assertion
named any operand of the first deliverable, so a revert of the selection ALONE
-- guard untouched, version untouched -- ran this instrument to exit 0 with
every assertion green.  That mutant is the shape's declined Option A: on a
route-less outfall ``Outfall[k].wRouted`` is NULL (node.c:163, allocated only
under ``if ( Outfall[k].routeTo >= 0 )`` at :164-167), so the two-way form
takes the address of ``NULL[k]`` -- and the pre-loop guard that would have
caught it is disarmed by the very ``s->routeTo >= 0`` conjunct A1 asserts.  The
campaign model has one route-less outfall.  So the instrument certified a
segfaulting tree as item 1's repair.

WHY A2 IS LOAD-BEARING RATHER THAN DECORATIVE
---------------------------------------------
Two guards at that block do two different jobs.  The PRE-LOOP guard returns
before the loop so the element address is never formed on a null array; the
PER-ELEMENT guard's ``s->routeTo`` and ``s->wRouted`` loads are safe ONLY
BECAUSE the pre-loop guard already fired.  Narrowing the pre-loop guard to
``SNAP_WRITE || SNAP_READ``, or giving the count pass a fast path around it,
reads as a no-op and is a null dereference: the observable is a segfault
forming an element address at ``(nil)``, exit 139.

FIVE INSTRUMENT HAZARDS, EACH MEASURED ON THIS TREE RATHER THAN ASSUMED
-----------------------------------------------------------------------
Each one, left unhandled, ships a PERMANENTLY GREEN or PERMANENTLY RED
assertion -- a test that reports on nothing while reading as coverage.

(a) THE LANDED PREDICATE WRAPS ACROSS TWO SOURCE LINES.  Measured: a
    line-anchored match on the full conjunction returns 0 LINES on the FIXED
    tree, while the same conjunction over whitespace-collapsed text returns 1
    MATCH.  So every operand here is matched against ``_collapse()`` output,
    never against a raw line.

(b) DO NOT ASSERT THE PRE-FIX FORM ABSENT.  The literal
    ``Nobjects[POLLUT] > 0 && !s->wRouted`` is LIVE on the fixed tree inside
    the repair's own explanatory comment (1 match, on a ``//`` line).  An
    "absent" assertion is RED on the tree it exists to protect.

    MEASURED EXTENSION, and it is this file's own finding rather than a
    carried one: the SAME trap sits at a SECOND site.
    ``SNAP_WRITE || SNAP_READ`` -- the narrowing A2 forbids -- is also LIVE,
    1 match, inside the comment at snapshot.c:342 reading "Deliberately NOT
    narrowed to `SNAP_WRITE || SNAP_READ`".  So A2 is stated POSITIVELY, on
    the presence of the un-narrowed form, and this file asserts NO absence of
    any narrowing literal.  Deliberately NOT the absence form: it would be red
    on the fixed tree for the same reason (b) names.

(c) ``!Outfall`` IS A PREFIX OF ``!OutfallStats``, and the two blocks'
    pre-loop guards are otherwise character-identical.  Measured collapsed on
    the fixed tree: the unbounded operand returns 2 MATCHES and the
    ``!OutfallStats`` form returns 1 -- so one of the unbounded operand's two
    hits IS the wrong guard.  An unbounded match stays GREEN after exactly the
    narrowing A2 forbids: delete the ``!Outfall`` guard and the
    ``!OutfallStats`` line alone still satisfies it.
    THIS FILE USES THE CLOSING-PARENTHESIS BOUND -- it matches ``!Outfall )``
    -- which returns 1 MATCH.  Of the three admissible bounds (this one; an
    operand-count-2-with-OutfallStats-1 pair; matching the whole block) it is
    the one whose failure mode is a plain miss rather than an arithmetic
    coincidence between two counts.

(d) ``grep -c`` COUNTS LINES, NOT MATCHES, and the long messages in this file
    are adjacent string literals wrapped across source lines.  A post-condition
    built on a line count is permanently green.  Every figure this file prints
    names its UNIT, and the unit is MATCHES over collapsed text except where
    stated otherwise.

(e) THE BARE ``wRoutedPlaceholder`` TOKEN IS INADMISSIBLE AS A4'S OPERAND, and
    this is hazard (b) in a new place rather than a new kind of trap.  Measured
    on the fixed tree, UNIT = matches over collapsed text: the bare identifier
    returns 3, and ``grep -n`` localises them to THREE DIFFERENT KINDS of line
    -- the declaration at snapshot.c:513, the repair's own explanatory comment
    at :968 ("Nothing writes wRoutedPlaceholder in WRITE mode"), and the one
    EXECUTABLE use at :974.  The two-way revert rewrites only :972-974, so the
    declaration and the comment survive it and a ``> 0`` assertion on the bare
    token is PERMANENTLY GREEN against exactly the revert A4 exists to catch.
    A4 therefore matches the WHOLE THREE-ARM CHAIN in executable form, which
    returns 1 on the fixed tree and 0 on the revert.  Its failure mode is a
    plain miss on a reformat -- the same cost hazard (c) accepted, and accepted
    here for the same reason: a miss is loud and an arithmetic coincidence
    between two token counts is not.

A SIXTH HAZARD THIS FILE HIT WHILE BEING WRITTEN, recorded because the
measurement is the only reason it is not in the shipped instrument: the define
is written ``#define SNAPSHOT_VERSION  6`` with TWO spaces, so a literal
two-space match over collapsed text returns 0 -- collapse folds it to one.
A3 therefore CAPTURES THE VALUE with a regex tolerant of the spacing, rather
than matching a formatting accident.  Deliberately NOT a literal match on the
define text: its failure mode is a silent 0 that reads as a missing define.

WHAT THIS DOES *NOT* COVER, stated so the green is not over-read
----------------------------------------------------------------
  * It is a TEXT check.  It cannot show the guards are correct, only that the
    landed predicates are still the ones that were reviewed.  A semantically
    equivalent rewrite that preserves behaviour will redden it; that is the
    cost of a text instrument and it is accepted here because the behavioural
    alternative needs an HPC venue.
  * Its subject is snapshot.c and nothing else.  Items 2, 3 and 6 touch
    src/swmm_triton.h and src/triton.h and are outside it.
  * It says nothing about the write-side disposition, the refusal messages, or
    the killed-write consultation.
"""

import re
import sys
from pathlib import Path

# A1's landed per-element conjunction.  Wraps across two source lines in the
# file, which is why every operand is matched against collapsed text (a).
PER_ELEMENT_GUARD = (
    "c->mode != SNAP_MANIFEST && Nobjects[POLLUT] > 0 "
    "&& s->routeTo >= 0 && !s->wRouted"
)

# A2's pre-loop guard, BOUNDED by the closing parenthesis (c).  The unbounded
# form matches the OutfallStats sibling too and survives the narrowing this
# asserts against.
PRE_LOOP_GUARD_BOUNDED = (
    "c->mode != SNAP_MANIFEST && Nnodes[OUTFALL] > 0 && !Outfall )"
)

# The sibling whose operand the bounded form exists to exclude.  Used ONLY as a
# vacuity guard: if the collision ever stops existing, the bound stops doing
# work and a reader should be told rather than left with a quietly weaker test.
PRE_LOOP_GUARD_SIBLING = (
    "c->mode != SNAP_MANIFEST && Nnodes[OUTFALL] > 0 && !OutfallStats"
)

# A4's landed three-way pointer selection, in EXECUTABLE form and as ONE
# operand spanning all three arms.  Matching the whole chain rather than the
# bare `wRoutedPlaceholder` identifier is hazard (e): that identifier also
# appears in its own declaration and in the repair's explanatory comment, both
# of which survive the two-way revert.
SELECTION_THREE_WAY = (
    "if ( c->mode == SNAP_MANIFEST ) slot = &_snapDummyScalar; "
    "else if ( s->wRouted ) slot = &s->wRouted[k]; "
    "else slot = &wRoutedPlaceholder;"
)

# Used ONLY as A4's vacuity guard, in the same role PRE_LOOP_GUARD_SIBLING
# plays for A2: it reports whether the non-executable occurrences that
# disqualify the bare token still exist.  If they ever stop existing, the
# executable-form bound has stopped doing work and a reader should be told
# rather than inheriting a quietly weaker assertion.
SELECTION_PLACEHOLDER_TOKEN = "wRoutedPlaceholder"

# A3 captures the VALUE rather than matching the define's spacing (the sixth
# hazard in the module docstring).
VERSION_RE = re.compile(r"#\s*define\s+SNAPSHOT_VERSION\s+(\d+)")

EXPECTED_VERSION = 6


def _collapse(text):
    """Join continuation lines by folding every whitespace run to one space.

    This is the whole answer to hazard (a): the landed predicate spans two
    source lines, so any line-anchored instrument returns 0 on the fixed tree.
    Collapsing makes the file one logical unit and the UNIT of every count
    taken against it MATCHES, never lines.
    """
    return re.sub(r"\s+", " ", text)


def main(argv):
    """Run the four assertions against one source tree and return an exit code.

    ``argv[1]`` is the repository root, which is how the sibling SOURCE-* nodes
    are invoked (``${CMAKE_SOURCE_DIR}`` is passed as the single argument), and
    it defaults to the working directory so the file is runnable by hand.
    Taking the root as an argument rather than deriving it from ``__file__`` is
    what lets the same script be pointed at a pre-fix export or an out-of-tree
    mutant, which is how its own differential was taken.

    Returns 0 when every assertion holds and 1 otherwise; it prints one
    ``ok``/``FAIL`` line per assertion and a terminal ``PASS``/``FAIL`` line, so
    the verdict prefixes match the family's ``^(ok|FAIL|SKIP|PASS)`` capture
    rule.
    """
    repo = Path(argv[1] if len(argv) > 1 else ".").resolve()
    target = repo / "external" / "swmm" / "src" / "solver" / "snapshot.c"
    if not target.is_file():
        print("FAIL: %s not found" % target)
        return 1

    raw = target.read_text()
    flat = _collapse(raw)
    failures = []

    def check(label, ok, detail=""):
        """Emit one verdict line and record a failure, closing over `failures`."""
        print("%s %s%s" % ("ok   " if ok else "FAIL ", label,
                           "" if ok else ("  -- %s" % detail)))
        if not ok:
            failures.append(label)

    print("subject   %s" % target)
    print("UNIT      MATCHES over whitespace-collapsed file text, except A3 "
          "(regex matches over raw text)")
    print("")

    # --- A1: DETECTOR -- the re-predicated per-element guard ----------------
    n_per_element = len(re.findall(re.escape(PER_ELEMENT_GUARD), flat))
    print("A1  per-element Outfall guard, joined across its two source lines:")
    print("        operand  %s" % PER_ELEMENT_GUARD)
    print("        matches  %d" % n_per_element)
    print("")
    check("A1  the per-element guard carries `s->routeTo >= 0` AND retains the "
          "pollutant-count conjunct (%d match)" % n_per_element,
          n_per_element == 1,
          "expected exactly 1; 0 means the re-predication was reverted or "
          "reformatted, >1 means the guard was duplicated")

    # A1 vacuity guard.  Without it a rename of SNAP_MANIFEST or of the fields
    # empties the population and turns A1 green by deleting its subject.
    n_mode_tests = len(re.findall(re.escape("c->mode != SNAP_MANIFEST"), flat))
    check("A1b the `c->mode != SNAP_MANIFEST` population is non-empty "
          "(%d match) -- without this A1 can pass by rename" % n_mode_tests,
          n_mode_tests > 0,
          "no mode test found at all; the traversal's mode vocabulary was "
          "renamed and A1 lost its subject")

    # --- A2: ANTI-NARROWING INVARIANT -- green on BOTH trees ----------------
    n_bounded = len(re.findall(re.escape(PRE_LOOP_GUARD_BOUNDED), flat))
    n_sibling = len(re.findall(re.escape(PRE_LOOP_GUARD_SIBLING), flat))
    print("")
    print("A2  pre-loop guard, CLOSING-PARENTHESIS bound (hazard (c)):")
    print("        operand  %s" % PRE_LOOP_GUARD_BOUNDED)
    print("        matches  %d" % n_bounded)
    print("        sibling  !OutfallStats form -> %d match(es)" % n_sibling)
    print("")
    check("A2  the Outfall[] PRE-LOOP guard still admits the counting mode and "
          "is NOT narrowed (%d match)" % n_bounded,
          n_bounded == 1,
          "expected exactly 1; 0 means the pre-loop guard was narrowed or "
          "removed, which makes the per-element s->routeTo load a null "
          "dereference (observable: element address at (nil), exit 139)")

    # A2 vacuity guard -- NOT a second form of A2.  It reports whether the
    # prefix collision the bound exists to exclude still exists.  If the
    # sibling ever disappears, the bound stops doing work; a reader should be
    # told rather than inheriting a quietly weaker assertion.
    check("A2b the !OutfallStats sibling still exists (%d match), so the "
          "closing-parenthesis bound is still load-bearing" % n_sibling,
          n_sibling == 1,
          "the sibling guard moved; re-derive whether A2's bound is still "
          "needed rather than assuming it")

    # --- A3: DETECTOR -- the version bump -----------------------------------
    versions = VERSION_RE.findall(raw)
    print("")
    print("A3  SNAPSHOT_VERSION define(s) found: %r" % (versions,))
    print("")
    check("A3  exactly one SNAPSHOT_VERSION define (%d found)" % len(versions),
          len(versions) == 1,
          "expected exactly 1; the value assertion below is ambiguous "
          "otherwise")
    if len(versions) == 1:
        check("A3  SNAPSHOT_VERSION is %d (found %s)"
              % (EXPECTED_VERSION, versions[0]),
              int(versions[0]) == EXPECTED_VERSION,
              "pre-fix carries 5; a value other than %d means the bump was "
              "reverted or moved again" % EXPECTED_VERSION)

    # --- A4: DETECTOR -- the three-way pointer selection --------------------
    n_selection = len(re.findall(re.escape(SELECTION_THREE_WAY), flat))
    n_placeholder_token = len(
        re.findall(re.escape(SELECTION_PLACEHOLDER_TOKEN), flat))
    print("")
    print("A4  three-way `slot` selection, EXECUTABLE form, all three arms as "
          "one operand:")
    print("        operand  %s" % SELECTION_THREE_WAY)
    print("        matches  %d" % n_selection)
    print("        bare `%s` token -> %d match(es) "
          "(declaration + comment + use; see hazard (e))"
          % (SELECTION_PLACEHOLDER_TOKEN, n_placeholder_token))
    print("")
    check("A4  `slot` is chosen by the landed THREE-ARM chain naming "
          "`_snapDummyScalar`, `s->wRouted[k]` and `wRoutedPlaceholder` "
          "(%d match)" % n_selection,
          n_selection == 1,
          "expected exactly 1; 0 means the selection was reverted to the "
          "pre-fix two-way ternary `slot = ( c->mode == SNAP_MANIFEST ) ? "
          "&_snapDummyScalar : &s->wRouted[k];`, which forms &NULL[k] on a "
          "route-less outfall (node.c:163 leaves wRouted NULL unless "
          "routeTo >= 0) -- and A1's own `s->routeTo >= 0` conjunct disarms "
          "the guard that would have caught it. Observable: segfault, exit 139")

    # A4 vacuity guard -- NOT a second form of A4.  See hazard (e): the bare
    # token's non-executable occurrences are precisely what makes a `> 0`
    # assertion on it permanently green against the revert.  This reports that
    # those occurrences still exist, so the executable-form bound is still the
    # one doing the work.
    check("A4b the bare `%s` token still occurs OUTSIDE the selection "
          "(%d match, >1 expected), so A4's executable-form bound is still "
          "load-bearing" % (SELECTION_PLACEHOLDER_TOKEN, n_placeholder_token),
          n_placeholder_token > 1,
          "the declaration and/or the explanatory comment moved; re-derive "
          "whether A4 still needs the whole-chain bound rather than assuming "
          "it")
    print("")

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: the snapshot traversal guards, the three-way `slot` "
          "selection and the version bump are the ones item 1 landed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
