#!/usr/bin/env python3
"""Source-structural checks for the cfg-gated STEM SUPPRESSION mechanism.

Usage:  test_snapshot_stem_suppression_source.py {repo-root}

WHAT THIS MECHANISM IS.  ``swmm_snapshot_disable=1`` leaves
``snapshot_path_stem`` EMPTY at its single assignment site in
``swmm_triton::init_swmm``.  An empty stem takes
``classify_missing_snapshot``'s FOURTH EXIT -- the one that fires BEFORE any
directory scan -- so the recorded resume reason is ``absent`` whatever the
snapshot directory holds.  That is how the re-run's replay-fallback arm forces
the OLD route deterministically instead of racing the retention collision for
it.

WHY THIS TIER IS STRUCTURAL RATHER THAN BEHAVIOURAL, stated plainly because it
is this file's one real limitation.  ``classify_missing_snapshot``,
``write_state_snapshot`` and ``try_restore_state_snapshot`` are all PUBLIC, so a
compiled Tier-1 test can CALL them.  Both inputs the fourth exit reads are
PRIVATE -- ``snapshot_path_stem`` and ``rank_`` -- and the stem's only
assignment site is inside ``init_swmm``, which calls ``swmm_open`` and therefore
needs a real ``.inp``.  So a Tier-1 test can call the function but cannot
ARRANGE it.  A default-constructed ``swmm_triton`` leaves ``rank_``
INDETERMINATE (an ``int`` member with no initializer), so classifying on a fresh
object returns ``absent`` through an undefined-behaviour disjunct -- which is
not an arrangement and is not used here.

THREE SEAMS WERE AVAILABLE AND NONE WAS TAKEN: a ``friend`` declaration for the
test, a test-only public setter, and ``#define private public`` before the
include.  Each adds permanent production surface, or UB, for a test.  The
behavioural two-arm differential is discharged at the admissible venue on a real
resumed coupled member, not here.  **Do not read a green run of this file as
behavioural coverage of the route.**  It covers the SOURCE properties that make
the route exist; the venue covers the route.

EVERY CHECK CARRIES A DEFECT PROBE, and every probe is TWO-STATE.  A structural
check reads source text, so the way it fails is by silently matching nothing.
Each ``S*`` below is paired with an ``X*`` that mutates the source IN MEMORY
into the defect form the check exists to catch and asserts the check goes RED
--  AND asserts the same check was GREEN on unmutated source in the same run.
Asserting only the red observes one bit, and two different worlds produce it:
the check caught the mutation, or the check was already red for an unrelated
reason.  A check whose probe does not discriminate is reported VACUOUS and
fails the run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KEY = "swmm_snapshot_disable"

failures: list[str] = []
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name.split()[0]] = ok
    if ok:
        print("ok    %s" % name)
    else:
        failures.append("%s: %s" % (name, detail))
        print("FAIL  %s   %s" % (name, detail))


def _code(text: str) -> str:
    """Strip C/C++ comments, preserving line structure.

    A SOURCE WALKER THAT READS COMMENTS AS CODE IS A MEASURED FAILURE CLASS IN
    THIS CAMPAIGN, and it fired on this file's own first run.  The declaration
    this file guards is preceded by a doc comment that names both the TYPE
    (`bool`) and the KEY in prose, with no intervening `;` -- so S1's
    `\\bbool\\b[^;]*?KEY` matched the COMMENT and reported the declaration
    present after the declaration's type had been mutated away.  The paired
    probe is the only reason that was visible.

    Newlines are preserved so any position- or line-anchored predicate keeps
    its meaning.  LIMITATION, stated rather than left implicit: this is a
    lexer-free strip, so a `//` inside a string literal would be treated as a
    comment start.  No such literal exists in the regions these predicates
    read; a future one would surface as a predicate going red with its probe
    still firing, not as a silent pass.
    """
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _swmm_ifdef_regions(text: str) -> list[str]:
    """Every `#ifdef TRITON_SWMM` ... `#endif` region, innermost-nesting-free.

    The struct's SWMM fields and the get_args SWMM reads each live in such a
    region, and a declaration that drifted OUT of one would compile in a
    coupled build and break a non-coupled one -- a configuration-dependent
    failure no coupled-only check can see.
    """
    out = []
    for m in re.finditer(r"#ifdef\s+TRITON_SWMM(.*?)#endif", text, re.S):
        out.append(m.group(1))
    return out


# ---------------------------------------------------------------------------
# The predicates.  Each takes source text and returns a bool, so the same
# function serves the live check and its probe -- a probe that re-implemented
# the predicate would be testing a second copy of it.
# ---------------------------------------------------------------------------

def p_declared_bool_under_swmm_ifdef(cfg_h: str) -> bool:
    """S1: declared, as a `bool`, inside a `#ifdef TRITON_SWMM` region."""
    for region in _swmm_ifdef_regions(_code(cfg_h)):
        if KEY not in region:
            continue
        # the declaration must be reached by a `bool` type-group opener with no
        # intervening `;` that would close a different group
        m = re.search(r"\bbool\b([^;]*?)\b%s\b" % KEY, region, re.S)
        if m:
            return True
    return False


def p_read_is_additive_not_mandatory(cfg_h: str) -> bool:
    """S2: read via argsd with default "0" -- NOT via args.

    Additive is required: every cfg omitting this key must stay byte-equivalent,
    because the arm this gates exists to leave the baseline measurement
    unperturbed.  `args` is the mandatory form used by the two manhole siblings
    beside it, and a drift to `args` is exactly the plausible-looking edit this
    check exists to catch.
    """
    m = re.search(r"%s\s*=\s*([^;]+);" % re.escape("arglist." + KEY), _code(cfg_h), re.S)
    if not m:
        return False
    rhs = m.group(1)
    if "args(" in rhs.replace("argsd(", ""):
        return False
    return bool(re.search(r'argsd\(\s*"%s"\s*,\s*argmap\s*,\s*"0"\s*\)' % KEY, rhs))


def p_threaded_through_every_signature(swmm_h: str) -> bool:
    """S3: present in all FOUR solver-side signature/call sites.

    initialize's declaration and definition, init_swmm's declaration and
    definition.  A parameter added to the definition but not the declaration is
    a compile error, but a parameter threaded into `initialize` and then DROPPED
    at the `init_swmm` call compiles cleanly and silently disables the
    mechanism -- the stem is then always assigned and the arm never engages.
    """
    swmm_h = _code(swmm_h)
    sites = 0
    for m in re.finditer(r"(?:void\s+(?:swmm_triton::)?(?:initialize|init_swmm)\s*\([^;{]*)", swmm_h):
        if KEY in m.group(0):
            sites += 1
    # Plus the inner call that FORWARDS it, and the anchor is load-bearing.  An
    # unanchored `init_swmm\s*\(...KEY` also matches the DECLARATION and the
    # DEFINITION, both of which legitimately name the parameter -- so dropping
    # the argument at the call would leave two matches and the check would stay
    # green.  Measured: it did, and the paired probe is what surfaced it.  The
    # call statement is the only one of the three that opens a line with
    # whitespace and no return type.
    forwarded = bool(
        re.search(r"^[ \t]*init_swmm\s*\([^;]*\b%s\b" % KEY, swmm_h, re.M))
    return sites >= 4 and forwarded


def p_call_site_passes_the_arglist_field(triton_h: str) -> bool:
    """S3b: triton.h's sole `initialize` call passes the arglist field."""
    m = re.search(r"swmm_model\.initialize\s*\((.*?)\);", _code(triton_h), re.S)
    if not m:
        return False
    return ("arglist." + KEY) in m.group(1)


def p_stem_assignment_is_guarded_by_the_negated_flag(swmm_h: str) -> bool:
    """S4: the stem assignment sits under `if (!swmm_snapshot_disable)`.

    The SENSE is the whole property.  A non-negated guard assigns the stem only
    when suppression IS requested, which inverts the mechanism while leaving
    every signature, every read and every call site correct -- and leaves a
    coupled run that never snapshots at all.
    """
    m = re.search(
        r"if\s*\(\s*!\s*%s\s*\)\s*\{\s*snapshot_path_stem\s*=" % KEY, _code(swmm_h), re.S)
    return bool(m)


def p_stem_has_exactly_one_assignment_site(swmm_h: str) -> bool:
    """S5: exactly ONE assignment to snapshot_path_stem in the whole file.

    One assignment site is what makes the three `.empty()` readers
    (write_state_snapshot, classify_missing_snapshot,
    try_restore_state_snapshot) unable to disagree.  A second assignment --
    however reasonable in isolation -- reintroduces the possibility that the
    stem is empty at one reader and set at another.
    """
    return len(re.findall(r"\bsnapshot_path_stem\s*=\s*[^=]", _code(swmm_h))) == 1


def p_fourth_exit_precedes_the_directory_scan(swmm_h: str) -> bool:
    """S6: the `.empty()` exit is positioned BEFORE the directory iteration.

    This is the ROUTE property.  The mechanism's whole value is that an empty
    stem decides the reason without reading the filesystem, so the recorded
    reason cannot be perturbed by whatever snapshots happen to survive
    retention.  Move the exit below the scan and an empty stem would reach
    `directory_iterator("")`, set the error code, leave `any` false and yield
    `pre-snapshot-checkpoint` -- a DIFFERENT recorded value, on the other side
    of the re-run's acceptance.
    """
    body = _classify_body(_code(swmm_h))
    if not body:
        return False
    exit_at = body.find("snapshot_path_stem.empty()")
    scan_at = body.find("directory_iterator")
    return exit_at != -1 and scan_at != -1 and exit_at < scan_at


def p_ladder_higher_test_is_strict(swmm_h: str) -> bool:
    """S7: `higher` is set by a STRICT greater-than, and the three returns are
    in their documented order.

    THIS IS THE PROPERTY THAT MAKES THE DISCRIMINATOR WORK, and it is recorded
    here because it is counter-intuitive and a reviewer needs it pinned.  The
    strictness is why a conforming snapshot at the RESUME ID or LOWER leaves
    `higher` false and sends the ladder to its tail, which returns `absent` --
    the SAME string the fourth exit returns.  So an arrangement that places a
    snapshot at the resume id CANNOT distinguish the suppressed route from the
    ladder tail, and a test built on it passes whether suppression engaged or
    not.

    Only a snapshot at a STRICTLY HIGHER id makes the ladder return something
    else (`retention-collision`), so that is the one arrangement on which the
    assertion can fail if suppression did not engage.  Relaxing `>` to `>=`
    here would silently change which arrangements discriminate.
    """
    body = _classify_body(_code(swmm_h))
    if not body:
        return False
    if not re.search(r"\)\s*>\s*\(long\)\s*checkpoint_id", body):
        return False
    order = [body.find('return "pre-snapshot-checkpoint"'),
             body.find('return "retention-collision"'),
             body.find('return "absent"', body.find("directory_iterator"))]
    return all(i != -1 for i in order) and order == sorted(order)


def p_defensive_only_no_longer_covers_the_empty_refusal(swmm_h: str) -> bool:
    """S8: the restore path's comment does not call the `.empty()` refusal defensive.

    The opening used to read `Defensive only:` and sit above BOTH refusals while
    its reasoning covered only the `rank_` one.  This chunk promotes the
    `.empty()` refusal from a defensive accident to the mechanism's INTENDED
    production route, so a comment still characterising it as defensive is now
    false -- and a false comment above a deliberate branch trains the next
    reader to treat a configured decline as a defect.
    """
    body = _restore_prologue(swmm_h)
    if not body:
        return False
    # The word may still appear, but it must be SCOPED to the rank_ refusal and
    # the empty-stem refusal must be positively characterised as intended.
    scoped = bool(re.search(r"rank_\s+refusal\s+below\s+IS\s+defensive", body))
    intended = bool(re.search(r"snapshot_path_stem\s+refusal\s+below\s+is\s+NOT\s+defensive",
                              body))
    return scoped and intended


def p_guard_comment_agrees_with_its_branch_structure(swmm_h: str) -> bool:
    """S9: the stem guard's own comment block does not contradict the branch below it.

    WHY THIS IS A SEPARATE PROPERTY AND NOT COVERED BY S4.  S4 is a guard-SENSE
    property: its regex terminates at the assignment and never reads past the
    closing brace, so it cannot see whether an `else` follows.  S1-S7 all run on
    `_code()`, which STRIPS comments -- correct hygiene, and precisely what made a
    false comment written beside the guarded statement in the same commit
    invisible to every one of them.  S8 reads RAW text but a DIFFERENT region (the
    restore prologue), so it could not see this one either.

    MEASURED, because the scope of the blindness decides what the fix has to be:
    deleting this guard's entire 290-byte `else` block changes the verdict of NONE
    of the nine checks above.  The blindness is therefore to the branch STRUCTURE,
    not merely to the comment -- which is also why DELETING the else (the obvious
    way to make a "there is no else branch" comment true) would make the comment
    true exactly once and guard nothing: the next `else` added under it would be
    as invisible as the first.

    Two conjuncts over the comment block immediately preceding the guard:
      (a) it does NOT assert the else away while an else is in fact present;
      (b) when an else IS present, it POSITIVELY names one.
    Both are load-bearing.  (a) alone passes a comment that is merely SILENT about
    a branch it ought to describe, and silence is how this defect started -- the
    original comment was written when there was no else, and the else arrived
    underneath it.
    """
    block = _guard_comment_block(swmm_h)
    if not block:
        return False
    has_else = _guard_has_else(swmm_h)
    # the denial form, in the shape a maintainer collapsing the block would write
    denies = bool(re.search(r"\b(?:is|are|has|have)\s+no\s+else\b", block, re.I))
    # a positive mention that SURVIVES removal of every denial phrase, so the
    # denial's own occurrence of the word cannot satisfy conjunct (b)
    residue = re.sub(r"\b(?:is|are|has|have)\s+no\s+else\b", "", block, flags=re.I)
    names = bool(re.search(r"\belse\b", residue, re.I))
    if has_else:
        return (not denies) and names
    # no else present: a comment claiming one is equally a disagreement
    return not names


# ---------------------------------------------------------------------------
# Body extractors.  Returning "" on a miss would make every predicate above
# red for the WRONG reason, which is the measured failure mode the paired
# baseline-green assertion exists to surface -- so these are kept narrow and
# their misses are reported by the checks that consume them.
# ---------------------------------------------------------------------------

def _classify_body(swmm_h: str) -> str:
    m = re.search(r"classify_missing_snapshot\s*\(int\s+checkpoint_id\s*\)\s*const\s*\{(.*?)\n\t\}",
                  swmm_h, re.S)
    if m:
        return m.group(1)
    i = swmm_h.find("swmm_triton::classify_missing_snapshot")
    if i == -1:
        return ""
    return swmm_h[i:i + 2500]


def _guard_comment_block(swmm_h: str) -> str:
    """The contiguous `//` lines immediately above the stem guard, RAW.

    RAW is the whole point: this is the one extractor in this file whose consumer
    is asserting ON comment text rather than reading THROUGH it, so passing it
    `_code()` would hand S9 an empty string and make it vacuously red.  `_code()`
    blanks comment bodies in place, so the blank-line skip below would then walk
    the entire preceding function.
    """
    i = swmm_h.find("if (!%s)" % KEY)
    if i == -1:
        return ""
    # drop the partial line carrying the guard's own indentation, then walk up
    # while the lines are comment lines.  The block uses bare `//` separators, so
    # it contains no blank lines and the first non-comment line ends the block.
    out = []
    for ln in reversed(swmm_h[:i].split("\n")[:-1]):
        if ln.strip().startswith("//"):
            out.append(ln)
        else:
            break
    return "\n".join(reversed(out))


def _guard_has_else(swmm_h: str) -> bool:
    """Whether the stem guard carries an `else` arm.  Read from STRIPPED source.

    The inverse of the extractor above, and deliberately so: this half is a
    question about CODE, and an `else` named in a comment must not answer it --
    that would let the comment corroborate itself and collapse S9 into a
    tautology.  `[^{}]*` keeps the match inside the guard's own single-statement
    block rather than running on to some later `else` in the file.
    """
    return bool(re.search(r"if\s*\(\s*!\s*%s\s*\)\s*\{[^{}]*\}\s*else\b" % KEY,
                          _code(swmm_h), re.S))


def _restore_prologue(swmm_h: str) -> str:
    i = swmm_h.find("swmm_triton::try_restore_state_snapshot")
    if i == -1:
        return ""
    j = swmm_h.find("snapshot_path_stem.empty()) return SnapshotRestore::Absent;", i)
    if j == -1:
        return ""
    return swmm_h[i:j]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: %s {repo-root}" % argv[0])
        return 2
    root = Path(argv[1])
    cfg_h = (root / "src" / "config_utils.h").read_text()
    swmm_h = (root / "src" / "swmm_triton.h").read_text()
    triton_h = (root / "src" / "triton.h").read_text()

    # L0 VACUITY GUARDS.  A structural check fails by matching nothing, so the
    # inputs are asserted non-empty before any property is read from them.
    check("L0a the three source files are non-empty",
          all(len(t) > 1000 for t in (cfg_h, swmm_h, triton_h)),
          "a source read came back short -- every property below would be "
          "vacuously red")
    check("L0b classify_missing_snapshot's body is locatable",
          len(_classify_body(swmm_h)) > 200,
          "the extractor missed; S6/S7 would be red for the wrong reason")
    check("L0c the restore prologue is locatable",
          len(_restore_prologue(swmm_h)) > 200,
          "the extractor missed; S8 would be red for the wrong reason")
    check("L0d at least one `#ifdef TRITON_SWMM` region exists in config_utils.h",
          len(_swmm_ifdef_regions(cfg_h)) >= 1,
          "S1 would be vacuously red")
    check("L0e the stem guard's comment block is locatable",
          len(_guard_comment_block(swmm_h)) > 200,
          "the extractor missed; S9 would be red for the wrong reason -- and S9 "
          "is the one check whose input is comment text, so it is the one most "
          "easily made vacuous by handing it stripped source")

    check("S1 the key is a bool under `#ifdef TRITON_SWMM`",
          p_declared_bool_under_swmm_ifdef(cfg_h),
          "not declared, not a bool, or outside the coupled-only region")
    check("S2 the read is additive (argsd default \"0\"), not mandatory",
          p_read_is_additive_not_mandatory(cfg_h),
          "the read is absent, uses args(), or carries a different default")
    check("S3 the parameter is threaded through all four solver signatures",
          p_threaded_through_every_signature(swmm_h),
          "a signature or the inner init_swmm call does not carry it")
    check("S3b triton.h's initialize call passes the arglist field",
          p_call_site_passes_the_arglist_field(triton_h),
          "the sole call site does not forward it")
    check("S4 the stem assignment is guarded by the NEGATED flag",
          p_stem_assignment_is_guarded_by_the_negated_flag(swmm_h),
          "the guard is absent or its sense is inverted")
    check("S5 the stem has exactly one assignment site",
          p_stem_has_exactly_one_assignment_site(swmm_h),
          "zero or more than one assignment -- the three .empty() readers can "
          "then disagree")
    check("S6 the empty-stem exit precedes the directory scan",
          p_fourth_exit_precedes_the_directory_scan(swmm_h),
          "the route property is broken: an empty stem would reach the scan "
          "and yield a different recorded reason")
    check("S7 the ladder's `higher` test is strict and its returns are ordered",
          p_ladder_higher_test_is_strict(swmm_h),
          "a relaxed comparison or a reordered ladder changes which "
          "arrangements can discriminate the two absent-returning routes")
    check("S8 `Defensive only` no longer covers the empty-stem refusal",
          p_defensive_only_no_longer_covers_the_empty_refusal(swmm_h),
          "the comment still characterises the mechanism's intended production "
          "route as defensive")
    check("S9 the guard's comment block agrees with the branch below it",
          p_guard_comment_agrees_with_its_branch_structure(swmm_h),
          "the comment denies an else that is present, or is silent about one "
          "-- the exact defect a reviewer found in this chunk, which every "
          "check above was structurally unable to see")

    # -----------------------------------------------------------------------
    # DEFECT PROBES.  Each mutates source in memory into the form its paired
    # check exists to catch.  The mutation is written to be PLAUSIBLE -- the
    # edit a later maintainer might actually make -- rather than merely
    # destructive, because a check that only catches vandalism catches nothing
    # that will happen.
    # -----------------------------------------------------------------------
    probes = [
        # the type drifts from bool to int: compiles, works, but leaves the
        # struct's flag group and the field's two-valued semantics disagreeing
        # NOTE the regex rather than a literal str.replace: these sources carry
        # CRLF line endings, so a mutation written with a bare "\n" matches
        # NOTHING and str.replace no-ops SILENTLY -- leaving the probe reporting
        # "not caught" for a check that is in fact fine.  Measured on this
        # file's first run against exactly this probe.
        ("X1 -> S1", lambda: p_declared_bool_under_swmm_ifdef(
            re.sub(r"bool(\s*)%s;" % KEY, r"int\g<1>%s;" % KEY, cfg_h, count=1))),
        # the read drifts to the mandatory form its two siblings use
        ("X2 -> S2", lambda: p_read_is_additive_not_mandatory(
            cfg_h.replace('argsd("%s", argmap, "0")' % KEY,
                          'args("%s", argmap)' % KEY, 1))),
        # the parameter is accepted by initialize and dropped at the forward
        ("X3 -> S3", lambda: p_threaded_through_every_signature(
            swmm_h.replace("init_swmm(project_dir, inp_filename, %s);" % KEY,
                           "init_swmm(project_dir, inp_filename);", 1))),
        ("X3b -> S3b", lambda: p_call_site_passes_the_arglist_field(
            triton_h.replace("arglist." + KEY, "false", 1))),
        # THE INVERTED GUARD: the single highest-value probe in this file
        ("X4 -> S4", lambda: p_stem_assignment_is_guarded_by_the_negated_flag(
            swmm_h.replace("if (!%s)" % KEY, "if (%s)" % KEY, 1))),
        # a second, locally-reasonable assignment appears
        ("X5 -> S5", lambda: p_stem_has_exactly_one_assignment_site(
            swmm_h.replace("\t\t\tswmm_open(", "\t\t\tsnapshot_path_stem = \"\";\n\t\t\tswmm_open(", 1))),
        # the exit is moved below the scan -- the route-breaking refactor
        ("X6 -> S6", lambda: p_fourth_exit_precedes_the_directory_scan(
            swmm_h.replace(
                'if (rank_ != 0 || snapshot_path_stem.empty()) return "absent";',
                'if (rank_ != 0) return "absent";', 1))),
        # `>` relaxed to `>=`: silently changes which arrangements discriminate
        ("X7 -> S7", lambda: p_ladder_higher_test_is_strict(
            swmm_h.replace("> (long)checkpoint_id", ">= (long)checkpoint_id", 1))),
        # the re-scoped opening regresses to the single unscoped characterisation
        ("X8 -> S8", lambda: p_defensive_only_no_longer_covers_the_empty_refusal(
            swmm_h.replace("The rank_ refusal below IS defensive only:",
                           "Defensive only:", 1))),
        # THE HISTORICAL DEFECT, re-inserted verbatim: a maintainer collapses the
        # block back to the terse claim while the else stays put.  This is not a
        # hypothetical mutation -- it is the text that shipped at e6a4f76 and that
        # every check above passed.
        ("X9 -> S9", lambda: p_guard_comment_agrees_with_its_branch_structure(
            swmm_h.replace(
                "There IS an else, four lines below,",
                "There is no else branch,", 1))),
        # THE SILENCE FORM, which conjunct (a) alone would pass: no denial is
        # re-added, the comment simply stops describing the branch.  This is how
        # the defect ORIGINATED rather than how it would regress -- the comment
        # predated the else and was never updated -- so a check carrying only
        # conjunct (a) would catch only the second half of the class.
        #
        # The mutation drops every COMMENT line mentioning `else` and touches no
        # code, which is exactly the state "the comment no longer describes the
        # branch" and nothing more.  A narrower edit to one clause is NOT a
        # silence mutation while a second clause still names the branch, and
        # writing one was this probe's own first-run failure: it reported
        # "not caught" against a check that was fine.
        ("X9b -> S9", lambda: p_guard_comment_agrees_with_its_branch_structure(
            "\n".join(
                ln for ln in swmm_h.split("\n")
                if not (ln.lstrip().startswith("//")
                        and re.search(r"\belse\b", ln, re.I))))),
    ]

    for name, fn in probes:
        sid = name.split("->")[1].strip()
        baseline_green = results.get(sid) is True
        still_green = fn()
        check("%s the defect form is CAUGHT" % name,
              baseline_green and not still_green,
              ("the paired %s is RED on UNMUTATED source, so this probe proves "
               "nothing -- fix the instrument before reading its verdict" % sid)
              if not baseline_green else
              "the mutated source still passes -- the paired check is VACUOUS")

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: stem suppression's source properties hold, and every check "
          "discriminates.")
    print("NOTE: this is STRUCTURAL coverage. The behavioural two-arm "
          "differential -- a resumed coupled member recording `absent` with "
          "the key set, and `retention-collision` without it under a "
          "higher-id snapshot -- is owed at the venue and is NOT discharged "
          "by a green here.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
