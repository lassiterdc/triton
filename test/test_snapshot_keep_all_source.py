#!/usr/bin/env python3
"""Source-structural checks for the cfg-gated SNAPSHOT RETENTION OVERRIDE.

Usage:  test_snapshot_keep_all_source.py {repo-root}

WHAT THIS MECHANISM IS.  ``swmm_snapshot_keep_all=1`` makes
``swmm_triton::write_state_snapshot`` SKIP its prune, so every per-checkpoint
SWMM state snapshot survives for the life of the run instead of only the
current and the immediately previous one.  Default OFF; keep-2 is unchanged.

WHY IT EXISTS, because a reader will otherwise look for the bug it fixes and
find none.  There is no bug.  keep-2 is correct and its fallback is correct: a
resume that finds no snapshot for its checkpoint id is classified
``retention-collision`` and replays the exchange history, which reaches the
same state by a slower route.  What keep-2 costs is that WHICH ROUTE a resume
takes is a WALL-CLOCK property -- whether the wanted snapshot survived depends
on how far the run advanced before the interruption fell.  Measured over 30
runs that all resumed at checkpoint 108: the split was 22 snapshot / 8 replay,
and four configurations split between their own two repeats on identical
inputs, so the route is not even a function of the configuration.  The key lets
a caller ELECT uniform route coverage, removing that variable when one run is
compared against another.

WHY THIS TIER IS STRUCTURAL, stated plainly because it is this file's one real
limitation.  The three facts that would make a behavioural assertion possible --
that a snapshot file for ``checkpoint_id - 2`` still exists after the (N)th
write, that it does not when the key is unset, and that a resume consequently
records ``snapshot`` rather than ``retention-collision`` -- all require a
COMPILED coupled solver writing real snapshot files from a real ``.inp``.
``write_state_snapshot`` is public and a Tier-1 test could CALL it, but
``snapshot_path_stem`` is private and its only assignment site is inside
``init_swmm``, which calls ``swmm_open``; so as with the sibling stem-suppression
node, a compiled test can call the function and cannot ARRANGE it.  No ``friend``
declaration, test-only setter, or ``#define private public`` was added, for the
same reason the sibling declined them.  **Do not read a green run of this file
as behavioural coverage.**  It covers the SOURCE properties that make the
mechanism exist; the two-arm behavioural differential needs a compiled run and
this file's last line says so on every run.

EVERY CHECK CARRIES A DEFECT PROBE, and every probe is TWO-STATE: it mutates the
source IN MEMORY into the defect form the check exists to catch, asserts the
check goes RED, **and** asserts the same check was GREEN on unmutated source in
the same run.  Asserting only the red observes one bit, and two different worlds
produce it: the check caught the mutation, or the check was already red for an
unrelated reason.  A check whose probe does not discriminate is reported VACUOUS
and fails the run.

TWO INSTRUMENT HAZARDS ARE INHERITED FROM THE SIBLING NODE AND HANDLED HERE.
(1) ``src/config_utils.h`` and ``src/swmm_triton.h`` carry CRLF on disk while
``src/triton.h`` does not -- AND THAT ASYMMETRY DOES NOT REACH THIS FILE, WHICH
IS THE OPPOSITE OF WHAT THIS PARAGRAPH SAID UNTIL IT WAS MEASURED.  Every source
here is read through ``Path.read_text()``, whose universal-newline translation
converts ``"\\r\\n"`` to ``"\\n"`` before any predicate sees it: measured on this
tree, ``read_text().count("\\r")`` is 0 on both CRLF files against
``read_bytes().count(b"\\r")`` of 707 and 1342.  So a bare ``"\\n"`` is the
CORRECT anchor everywhere, and the INERT one is ``"\\r\\n"`` -- the exact reverse
of the earlier claim.  That direction is the dangerous one: an author who
believes this file sees raw CRLF keys a mutation on ``"\\r\\n"``, it matches
nothing, ``str.replace`` no-ops SILENTLY, and the probe then prints "the mutated
source still passes -- the paired check is VACUOUS" against a check that is
fine.  A FALSE VACUITY report sends the next reader to rewrite a correct
predicate.  The hazard was neutralised by the I/O call, never by mutation
discipline, so discipline is not what this file relies on: ``_perturbed()``
asserts, as its own named check, that each new probe's mutation CHANGED the
string before any verdict is read.  Two of the mutations below are same-length
substitutions, so a length comparison would report them inert -- string
inequality is the predicate that works.  The 15 pre-existing probes are not yet
routed through ``_perturbed()``; all 15 were independently measured to perturb
at this commit, so the gap is recorded rather than papered over, and wrapping
them is a mechanical follow-up.  (2) A source walker that reads
comments as code is a measured failure class in this work: the declaration
this file guards is preceded by a doc comment naming both the type and the key
in prose, so the predicates strip comments.  The one predicate that
deliberately READS comment text is exempt by construction and says so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KEY = "swmm_snapshot_keep_all"
MEMBER = "snapshot_keep_all"

# The four keys output_cfg rewrites.  Enumerated here so S9 can assert that the
# new key is neither one of them nor a substring of one of them.
REWRITTEN_CFG_KEYS = ("sim_start_time=", "checkpoint_id=", "time_step=", "it_count=")

def _drift_read_below_endif(cfg_h: str) -> str:
    """X8b's mutation: relocate the cfg read to just BELOW its region's ``#endif``.

    BALANCED BY CONSTRUCTION.  It moves a statement rather than inserting a
    directive, so the ``#ifdef``/``#endif`` count is unchanged and the byte
    length is unchanged -- which is precisely what makes this the drift no other
    instrument in this tree sees.  MEASURED: the coupled
    ``SOURCE-SYNTAX-PARSE-ALL-TUS`` pass stays GREEN on this shape, while a
    define-free ``-fsyntax-only`` pass over ``src/main.cpp`` raises ``'struct
    ConfigUtils::arguments<double>' has no member named swmm_snapshot_keep_all``.
    The alternative shape -- INSERTING an ``#endif`` above the read -- is
    deliberately NOT used: it unbalances the directives and the already-
    registered parse node catches it today as ``#endif without #if``, so a probe
    written that way would be exercising a defect this file does not own.

    IT IS POSITION-INDEPENDENT ON PURPOSE.  An earlier form anchored on the
    literal ``read-line`` + ``#endif`` pair, which is correct only while the read
    is the LAST statement in its region.  Moving the read to a different,
    equally-correct position inside the region made that anchor match nothing,
    and the probe's own perturbation check then went red -- an accurate
    instrument message, but a node failure on a legitimate rearrangement.  This
    form locates the read wherever it sits and uses the first ``#endif`` after
    it, so S8b's probe survives any correct position.

    The ``"\\n"`` anchors are correct and are NOT a CRLF hazard: ``config_utils.h``
    is CRLF on disk, but every predicate here reads it through
    ``Path.read_text()``, whose universal-newline translation yields ``"\\n"`` --
    measured, ``read_text().count("\\r")`` is 0 against ``read_bytes()``'s 707.
    An anchor written as ``"\\r\\n"`` is the INERT one.  See hazard (1) above.

    On any miss it returns the input UNCHANGED rather than raising, so the
    failure surfaces through ``_perturbed()``'s named check instead of as a
    traceback with no ``FAIL`` line.
    """
    m = re.search(r"^[ \t]*arglist\.%s\b[^\n]*\n" % re.escape(KEY), cfg_h, re.M)
    if not m:
        return cfg_h
    read_line = m.group(0)
    endif = re.compile(r"^#endif[^\n]*\n", re.M).search(cfg_h, m.end())
    if not endif:
        return cfg_h
    return (cfg_h[:m.start()]
            + cfg_h[m.end():endif.end()]
            + read_line
            + cfg_h[endif.end():])

failures: list[str] = []
results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name.split()[0]] = ok
    if ok:
        print("ok    %s" % name)
    else:
        failures.append("%s: %s" % (name, detail))
        print("FAIL  %s   %s" % (name, detail))


def _perturbed(probe: str, before: str, after: str) -> str:
    """Assert a probe's mutation actually CHANGED the source, as its own check.

    AN INERT MUTATION AND A VACUOUS CHECK ARE INDISTINGUISHABLE AT THE PROBE'S
    OWN VERDICT, and that is why this is a separate named assertion rather than
    a condition folded into the probe.  A ``str.replace`` whose pattern misses
    returns the input unchanged and raises nothing, so the probe then evaluates
    its predicate against UNMUTATED source, finds it green, and prints "the
    mutated source still passes -- the paired check is VACUOUS" against a check
    that is in fact fine.  The report is then a FALSE VACUITY claim, which is
    worse than a missed probe because it sends the next reader to rewrite a
    correct predicate.

    A length comparison is NOT sufficient and is named here because it is the
    plausible wrong form: two of this file's mutations (the ``"0"`` -> ``"1"``
    default flip and the ``- 2`` -> ``- 3`` offset drift) are same-length
    substitutions, so ``len(after) != len(before)`` reports them inert.  String
    inequality is the predicate that distinguishes a same-length substitution
    from a miss.

    It returns ``after`` so it composes inline inside a probe lambda, and it
    does NOT raise: a raise would exit non-zero with a traceback and no ``FAIL``
    line, which is an exit code with an empty failure set -- a shape that has
    already been mistaken for a verdict once here.
    """
    check("%s the mutation PERTURBS the source" % probe,
          after != before,
          "the mutation matched NOTHING and the source is unchanged, so the "
          "paired probe's verdict below is about UNMUTATED source and proves "
          "nothing -- fix the mutation's anchor, not the predicate")
    return after


def _code(text: str) -> str:
    """Strip C/C++ comments, preserving line structure.

    Newlines are preserved so any line- or position-anchored predicate keeps its
    meaning.  LIMITATION, stated rather than left implicit: this is a lexer-free
    strip, so a ``//`` inside a string literal would be read as a comment start.
    No such literal exists in the regions these predicates read; a future one
    would surface as a predicate going red with its probe still firing, not as a
    silent pass.
    """
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _swmm_ifdef_regions(text: str) -> list[str]:
    """Every ``#ifdef TRITON_SWMM`` ... ``#endif`` region.

    The struct's SWMM fields and the get_args SWMM reads each live in such a
    region, and a declaration that drifted OUT of one would compile in a coupled
    build and break a non-coupled one -- a configuration-dependent failure no
    coupled-only check can see.
    """
    return [m.group(1) for m in
            re.finditer(r"#ifdef\s+TRITON_SWMM(.*?)#endif", text, re.S)]


def _write_snapshot_body(swmm_h: str) -> str:
    """The body of ``swmm_triton::write_state_snapshot``, braces balanced.

    Extracted by brace counting rather than by a terminator regex because the
    body contains nested blocks, and a regex stopping at the first ``}`` would
    hand every predicate below a truncated prefix that happens to contain the
    early returns and not the prune -- a silent-pass shape.
    """
    i = swmm_h.find("void swmm_triton::write_state_snapshot(")
    if i == -1:
        return ""
    j = swmm_h.find("{", i)
    if j == -1:
        return ""
    depth = 0
    for k in range(j, len(swmm_h)):
        if swmm_h[k] == "{":
            depth += 1
        elif swmm_h[k] == "}":
            depth -= 1
            if depth == 0:
                return swmm_h[j:k + 1]
    return ""


def _output_cfg_body(output_h: str) -> str:
    """The body of ``output<T>::output_cfg``, braces balanced."""
    i = output_h.find("void output<T>::output_cfg(")
    if i == -1:
        return ""
    j = output_h.find("{", i)
    if j == -1:
        return ""
    depth = 0
    for k in range(j, len(output_h)):
        if output_h[k] == "{":
            depth += 1
        elif output_h[k] == "}":
            depth -= 1
            if depth == 0:
                return output_h[j:k + 1]
    return ""


def _prune_guard_comment_block(swmm_h: str) -> str:
    """The RAW (uncommented-stripped) comment run immediately above the prune.

    This is the one predicate input that is comment text, so it is handed raw
    source deliberately.  Handing it ``_code()`` output would make S11 vacuous.
    """
    body = _write_snapshot_body(swmm_h)
    if not body:
        return ""
    g = body.find("if (!%s && checkpoint_id >= 2)" % MEMBER)
    if g == -1:
        # the guard may be in a defect form; fall back to the unguarded shape so
        # the extractor still locates a block and S11 fails LOUDLY rather than
        # vacuously
        g = body.find("checkpoint_id >= 2")
    if g == -1:
        return ""
    # Back up to the START of the guard's own line.  The fallback anchor above
    # lands MID-LINE (inside `if (checkpoint_id >= 2)`), so slicing at it leaves
    # a partial code line as the last element, the walk-back breaks on it
    # immediately, and the extractor returns "" -- reporting "no comment block"
    # about source that has one.  Measured against pre-change source, where it
    # made L0e red for the wrong reason: exactly the vacuity class L0e exists to
    # announce, arriving in L0e's own extractor.
    g = body.rfind("\n", 0, g) + 1
    lines = body[:g].split("\n")
    out: list[str] = []
    for ln in reversed(lines):
        if ln.strip().startswith("//") or ln.strip() == "":
            out.append(ln)
            if ln.strip() == "" and out and len([o for o in out if o.strip()]) > 0:
                # a blank line inside the comment run is allowed; keep walking
                pass
        else:
            break
    return "\n".join(reversed(out))


# ---------------------------------------------------------------------------
# The predicates.  Each takes source text and returns a bool, so the same
# function serves the live check and its probe -- a probe that re-implemented
# the predicate would be testing a second copy of it.
# ---------------------------------------------------------------------------

def p_declared_bool_under_swmm_ifdef(cfg_h: str) -> bool:
    """S1: declared, as a ``bool``, inside a ``#ifdef TRITON_SWMM`` region.

    The type is load-bearing twice over: the struct's flag group is bool, and a
    drift to ``int`` would leave the field's two-valued semantics and its
    declared type disagreeing while compiling and working.
    """
    for region in _swmm_ifdef_regions(_code(cfg_h)):
        if KEY not in region:
            continue
        if re.search(r"\bbool\b([^;]*?)\b%s\b" % KEY, region, re.S):
            return True
    return False


def p_read_is_additive_not_mandatory(cfg_h: str) -> bool:
    """S2: read via ``argsd`` with default ``"0"`` -- NOT via ``args``.

    THIS IS THE PROPERTY THAT MAKES THE CHANGE ADDITIVE RATHER THAN BREAKING.
    ``argsd`` returns its default when the key is absent from the map (verified
    at ``config_utils.h``'s own definition: ``if (y.find(x) != y.end()) return
    ...; else return d;``), so a cfg omitting the key parses to ``atoi("0") ==
    0`` and keeps loading.  ``args`` is ``argsd(x, y, "")`` -- it would yield
    ``atoi("") == 0`` too, so the FALSE value would survive a drift to ``args``
    and this check would look like a style preference.  It is not.  The two
    mandatory manhole siblings beside this read use ``args`` precisely because a
    coupled run cannot proceed without them, and a reviewer who sees ``args``
    here will reasonably infer the same about this key -- which is the opposite
    of its contract.  The default string is additionally checked to be ``"0"``
    and not ``"1"``: a flipped default is the one drift here that silently
    changes every existing cfg's behaviour.
    """
    m = re.search(r"%s\s*=\s*([^;]+);" % re.escape("arglist." + KEY),
                  _code(cfg_h), re.S)
    if not m:
        return False
    rhs = m.group(1)
    if "args(" in rhs.replace("argsd(", ""):
        return False
    return bool(re.search(r'argsd\(\s*"%s"\s*,\s*argmap\s*,\s*"0"\s*\)' % KEY, rhs))


def p_threaded_through_initialize(swmm_h: str) -> bool:
    """S3: present in BOTH ``initialize`` signature sites, as a ``const bool``.

    Declaration and definition.  A parameter present in one and not the other is
    a compile error, so this check's value is not catching that -- it is
    catching a drift to a non-const or non-bool parameter, and pinning that the
    value arrives by THREADING rather than by any other route.
    """
    swmm_h = _code(swmm_h)
    sites = 0
    for m in re.finditer(r"void\s+(?:swmm_triton::)?initialize\s*\([^;{]*", swmm_h):
        if re.search(r"const\s+bool\s+%s\b" % KEY, m.group(0)):
            sites += 1
    return sites == 2


def p_not_threaded_into_init_swmm(swmm_h: str) -> bool:
    """S3c: the key is NOT threaded into ``init_swmm``.

    A DELIBERATE ASYMMETRY WITH THE SIBLING, pinned so a later maintainer
    "completing the pattern" has to argue with a red test rather than with a
    comment.  ``swmm_snapshot_disable``'s consumer IS ``init_swmm`` -- it decides
    whether the stem is assigned, inside the same call -- so it is threaded there
    and needs no member.  This key's consumer is ``write_state_snapshot``, called
    once per checkpoint long after ``initialize`` returns, so it is latched to a
    member instead.  Threading it into ``init_swmm`` as well would add a
    parameter that function has no use for, and an unused parameter in a
    signature is read by the next reader as a dependency that exists.
    """
    for m in re.finditer(r"(?:void\s+(?:swmm_triton::)?init_swmm\s*\([^;{]*)",
                         _code(swmm_h)):
        if KEY in m.group(0):
            return False
    return True


def p_call_site_passes_the_arglist_field(triton_h: str) -> bool:
    """S3b: triton.h's sole ``initialize`` call passes the arglist field."""
    m = re.search(r"swmm_model\.initialize\s*\((.*?)\);", _code(triton_h), re.S)
    if not m:
        return False
    return ("arglist." + KEY) in m.group(1)


def p_call_site_is_inside_a_swmm_ifdef(triton_h: str) -> bool:
    """S8: triton.h's forwarded argument sits inside a ``#ifdef TRITON_SWMM`` region.

    THE NON-COUPLED BUILD IS COVERED BY NO EXECUTABLE CHECK IN THIS TREE, which
    is the whole reason this predicate exists.  The repo's one no-build parse
    node (``SOURCE-SYNTAX-PARSE-ALL-TUS``) passes ``-DTRITON_SWMM``
    unconditionally and its own source says the define is REQUIRED, so a green
    parse says nothing about the configuration where ``arguments<T>`` does not
    carry this field at all.  An argument that drifted out of the guarded region
    -- during a reindent, a merge, or a reflow of that call's continuation lines
    -- would compile in every coupled configuration anyone tests and break the
    non-coupled one, which is the asymmetric-visibility failure the sibling
    node's S1 records for the DECLARATION side.  This is its CALL-SITE mirror.

    The region scan is run on comment-stripped text so a commented-out mention
    inside an ifdef cannot satisfy it.
    """
    regions = _swmm_ifdef_regions(_code(triton_h))
    return any(("arglist." + KEY) in r for r in regions)


def p_read_is_inside_a_swmm_ifdef(cfg_h: str) -> bool:
    """S8b: the cfg READ sits inside a ``#ifdef TRITON_SWMM`` region.

    THE THIRD MEMBER OF S1's AND S8's OWN CLASS, AND IT HAD NO CHECK.  This
    commit introduces three sites whose correctness depends on the coupled
    define: the DECLARATION (covered by S1), the cfg READ (this check), and the
    FORWARDED ARGUMENT in ``triton.h`` (covered by S8).  All three are dependent
    constructs over ``arguments<T>``; the enumeration in S8's own docstring
    named two of them.  Neither pre-existing predicate reaches this one: S1 is
    region-scoped but keys on ``bool ... KEY`` and is satisfied by the
    DECLARATION region, and S2 runs over whole-file comment-stripped text with
    no region constraint at all.

    MEASURED, NOT ASSERTED, and the measurement is what sets the probe's form
    below.  Two drift shapes put this read outside the region.  One inserts an
    ``#endif`` above it, which UNBALANCES the directives -- and that shape is
    already caught by ``SOURCE-SYNTAX-PARSE-ALL-TUS`` today, with the define, as
    ``error: #endif without #if``.  The other RELOCATES the read below the
    existing ``#endif``, leaving the directives balanced, the byte length
    unchanged, and the coupled parse GREEN.  The second shape -- the reflow or
    merge that actually happens -- is the one no instrument in this tree saw,
    and it is the one X8b mutates.  A probe written in the unbalanced form would
    be testing a defect the parse node already owns.

    WHY THE PREDICATE IS UNIVERSAL AND NOT EXISTENTIAL, which is where it
    diverges from S8.  S8 asks ``any(... in r for r in regions)``: a check that
    SOME occurrence is guarded, not that EVERY one is.  The two coincide only
    because there is exactly one occurrence today; a second, unguarded read
    would satisfy the existential form.  This predicate asserts instead that the
    in-region occurrence count EQUALS the whole-file count, so an added
    unguarded read goes red.  The ``>= 1`` floor is the vacuity half: without it
    a key that vanished entirely would give ``0 == 0`` and pass, and a
    structural check that passes by matching nothing is the failure shape this
    work has now hit in several separate instruments.

    The region scan is run on comment-stripped text so a commented-out mention
    inside an ifdef cannot satisfy it.
    """
    code = _code(cfg_h)
    anchor = "arglist." + KEY
    whole = code.count(anchor)
    inside = sum(r.count(anchor) for r in _swmm_ifdef_regions(code))
    return whole >= 1 and inside == whole


def p_latched_to_member_exactly_once(swmm_h: str) -> bool:
    """S4: the parameter is latched to the member exactly once, in ``initialize``.

    One latch site is what makes the member and the cfg value unable to
    disagree.  A second assignment -- however reasonable in isolation --
    reintroduces the possibility that the gate reads a value the cfg never set.

    THE IN-CLASS INITIALISER IS REMOVED FROM THIS PREDICATE'S INPUT FIRST, and
    that is not a convenience.  ``bool snapshot_keep_all = false;`` is
    syntactically an assignment to the same name, so counting raw ``MEMBER =``
    occurrences returns 2 on correct source and this check was RED on its own
    first run -- caught by its paired X4 probe, which reported that the check it
    was validating was already red.  The declaration's presence is S4b's
    property; the number of RUNTIME latch sites is this one's.  Measuring them
    with one regex conflated two properties and made the stricter of them
    unsatisfiable.
    """
    code = _code(swmm_h)
    code = re.sub(r"\bbool\s+%s\s*=\s*[^;]*;" % MEMBER, "", code)
    assigns = re.findall(r"\b(?:this->)?%s\s*=\s*[^=]" % MEMBER, code)
    if len(assigns) != 1:
        return False
    return bool(re.search(
        r"(?:this->)?%s\s*=\s*%s\s*;" % (MEMBER, KEY), code))


def p_member_defaults_false(swmm_h: str) -> bool:
    """S4b: the member carries an in-class ``= false`` initialiser.

    Without it the member is an indeterminate ``bool`` on any object whose
    ``initialize`` did not run, and an indeterminate gate would prune or not
    prune unpredictably -- the one failure mode here that is not reproducible.
    The sibling node records exactly this hazard about ``rank_``, which has no
    initialiser; this member does not repeat it.
    """
    return bool(re.search(r"\bbool\s+%s\s*=\s*false\s*;" % MEMBER, _code(swmm_h)))


def p_gate_sense_and_threshold(swmm_h: str) -> bool:
    """S5: the prune is gated by the NEGATED member, conjoined with ``>= 2``.

    THE SENSE IS THE WHOLE PROPERTY, and this is the single highest-value check
    in this file.  A non-negated guard prunes only when preservation was
    REQUESTED -- which inverts the mechanism while leaving the declaration, the
    read, both signatures, the call site and the latch all correct, and leaves an
    experiment that asked for every snapshot holding two.  The ``>= 2`` threshold
    and the ``checkpoint_id - 2`` argument are checked in the same predicate
    because together they ARE "keep-2": a drift in either silently changes what
    the OFF path does, which is the path the developer required be untouched.
    """
    body = _code(_write_snapshot_body(swmm_h))
    if not body:
        return False
    return bool(re.search(
        r"if\s*\(\s*!\s*%s\s*&&\s*checkpoint_id\s*>=\s*2\s*\)" % MEMBER, body)) and \
        bool(re.search(
            r"filesystem::remove\s*\(\s*snapshot_path_for\s*\(\s*checkpoint_id\s*-\s*2\s*\)",
            body))


def p_exactly_one_removal_site_in_the_writer(swmm_h: str) -> bool:
    """S6: ``write_state_snapshot`` contains exactly ONE removal call.

    THIS IS THE ON-PATH PROPERTY, and it is not implied by S5.  S5 establishes
    that the known prune is gated; it says nothing about a SECOND removal that is
    not.  "ON retains every snapshot" is a claim about the whole writer, and the
    only way to hold it structurally is to pin that the gated call is the only
    call -- otherwise a later unguarded cleanup would defeat the key while every
    other check in this file stayed green.
    """
    body = _code(_write_snapshot_body(swmm_h))
    if not body:
        return False
    return len(re.findall(r"filesystem::remove\s*\(", body)) == 1


def p_writer_does_not_reparse_the_flag(swmm_h: str) -> bool:
    """S7: the writer reads the latched member and parses nothing.

    The flag is read ONCE, at cfg-parse time, and threaded.  A re-parse at the
    prune site -- ``argsd``, ``args``, ``atoi``, ``getenv``, or a second file
    read -- would make the gate's value depend on state the cfg no longer
    controls, and would be invisible to every signature and call-site check.
    """
    body = _code(_write_snapshot_body(swmm_h))
    if not body:
        return False
    for tok in ("argsd(", "args(", "atoi(", "getenv("):
        if tok in body:
            return False
    return MEMBER in body


def p_rewriter_cannot_touch_the_key(output_h: str) -> bool:
    """S9: ``output_cfg`` rewrites a closed four-key set, and the new key is not
    in it and is not a substring of any member of it.

    THIS IS THE RESUME-SURVIVAL PROPERTY.  ``output_cfg`` seeds its output from
    the verbatim parent cfg text and replaces exactly four ``key=`` spans by
    substring search, copying every other line -- including every key it does not
    know -- into ``config_{id}.cfg`` unchanged.  So the new key is carried forward
    across every resume by the SAME passthrough that already carries
    ``swmm_snapshot_disable``, and nothing has to be added for it.

    What this check therefore guards is the one way that could stop being true:
    the search is ``str2.find("key=")``, an UNANCHORED substring match over the
    whole buffer, so a future key whose text contains one of the four would be
    rewritten by that branch.  The assertion is that the four-key set is still
    exactly these four AND that the new key collides with none of them.  A fifth
    rewritten key is not itself a defect -- it is a change that has to be checked
    against this key, and this is the check.
    """
    body = _output_cfg_body(_code(output_h))
    if not body:
        return False
    found = set(re.findall(r'line\.find\("([A-Za-z0-9_]+=)"\)', body))
    if found != set(REWRITTEN_CFG_KEYS):
        return False
    for k in REWRITTEN_CFG_KEYS:
        if k.rstrip("=") in KEY or KEY in k:
            return False
    return True


def p_retention_comment_describes_both_paths(swmm_h: str) -> bool:
    """S11: the prune's own comment block describes the OVERRIDE, not just keep-2.

    WHY THIS IS A PROPERTY AND NOT A STYLE NOTE.  The comment that shipped above
    this prune read "two is enough for the resume that is about to happen", which
    was TRUE of the resume and became FALSE of the EXPERIMENT the moment this key
    existed -- an experiment needs one nominated checkpoint to survive on EVERY
    member regardless of when each was interrupted.  A false comment above a
    deliberate branch trains the next reader to treat the gate as dead weight and
    delete it.  Every other check in this file runs on ``_code()``, which strips
    comments, so none of them can see this; this predicate is handed RAW text
    deliberately and is the only one that is.

    Two conjuncts, both load-bearing: the block must NAME the key (so a reader
    arriving at the gate can find its contract) and must say the default is
    UNCHANGED (so a reader does not conclude the retention policy moved).
    """
    block = _prune_guard_comment_block(swmm_h)
    if not block:
        return False
    names_key = KEY in block
    names_default = bool(re.search(r"\bBY DEFAULT\b|\bdefault is unchanged\b",
                                   block))
    return names_key and names_default


def gate(keep_all: bool, checkpoint_id: int) -> bool:
    """The NEW gate's condition, as a two-input predicate.

    Used by the D* differential below.  This is the one place in this file where
    a condition from the source is restated rather than read, and the restatement
    is the point: S5 asserts the source carries this exact condition, and D1/D2
    assert this condition has the intended truth table.  Neither establishes the
    other, and together they are the strongest claim available without a
    compiler.
    """
    return (not keep_all) and checkpoint_id >= 2


def old_gate(checkpoint_id: int) -> bool:
    """The condition as it stood before this change: ``checkpoint_id >= 2``."""
    return checkpoint_id >= 2


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: %s {repo-root}" % argv[0])
        return 2
    root = Path(argv[1])
    cfg_h = (root / "src" / "config_utils.h").read_text()
    swmm_h = (root / "src" / "swmm_triton.h").read_text()
    triton_h = (root / "src" / "triton.h").read_text()
    output_h = (root / "src" / "output.h").read_text()

    # ---------------------------------------------------------------------
    # L0 VACUITY GUARDS.  A structural check fails by matching nothing, so the
    # inputs are asserted non-empty and the extractors asserted to have landed
    # before any property is read from them.
    # ---------------------------------------------------------------------
    check("L0a the four source files are non-empty",
          all(len(t) > 1000 for t in (cfg_h, swmm_h, triton_h, output_h)),
          "a source read came back short -- every property below would be "
          "vacuously red")
    check("L0b write_state_snapshot's body is locatable and brace-balanced",
          len(_write_snapshot_body(swmm_h)) > 400,
          "the extractor missed; S5/S6/S7 would be red for the wrong reason")
    check("L0c output_cfg's body is locatable",
          len(_output_cfg_body(output_h)) > 400,
          "the extractor missed; S9 would be red for the wrong reason")
    check("L0d at least one `#ifdef TRITON_SWMM` region exists in config_utils.h",
          len(_swmm_ifdef_regions(cfg_h)) >= 1,
          "S1 would be vacuously red")
    check("L0e the prune's comment block is locatable in RAW source",
          len(_prune_guard_comment_block(swmm_h)) > 200,
          "the extractor missed; S11 would be red for the wrong reason -- and "
          "S11 is the one check whose input is comment text, so it is the one "
          "most easily made vacuous by handing it stripped source")

    # ---------------------------------------------------------------------
    # The properties.
    # ---------------------------------------------------------------------
    check("S1 the key is a bool under `#ifdef TRITON_SWMM`",
          p_declared_bool_under_swmm_ifdef(cfg_h),
          "not declared, not a bool, or outside the coupled-only region")
    check("S2 the read is additive (argsd default \"0\"), not mandatory",
          p_read_is_additive_not_mandatory(cfg_h),
          "the read is absent, uses args(), or carries a different default -- a "
          "flipped default changes every existing cfg's behaviour")
    check("S3 the parameter is a const bool in both initialize signatures",
          p_threaded_through_initialize(swmm_h),
          "a signature does not carry it, or carries it non-const / non-bool")
    check("S3b triton.h's initialize call passes the arglist field",
          p_call_site_passes_the_arglist_field(triton_h),
          "the sole call site does not forward it")
    check("S8 triton.h's forwarded argument is inside a `#ifdef TRITON_SWMM` region",
          p_call_site_is_inside_a_swmm_ifdef(triton_h),
          "the argument drifted outside the coupled-only region -- it would "
          "compile in every coupled configuration and break the non-coupled "
          "one, and no executable check in this tree parses that configuration")
    check("S8b config_utils.h's cfg read is inside a `#ifdef TRITON_SWMM` region",
          p_read_is_inside_a_swmm_ifdef(cfg_h),
          "the read drifted outside the coupled-only region, or a second "
          "unguarded read was added -- the read is a dependent member access "
          "on `arguments<T>`, so outside the region it breaks exactly the "
          "configuration nothing in this tree parses, and it does so with the "
          "directives balanced and the coupled parse still green")
    check("S3c the key is NOT threaded into init_swmm",
          p_not_threaded_into_init_swmm(swmm_h),
          "an unused parameter was added to init_swmm -- the deliberate "
          "asymmetry with swmm_snapshot_disable has been 'completed' away")
    check("S4 the parameter is latched to the member exactly once",
          p_latched_to_member_exactly_once(swmm_h),
          "zero, more than one, or a latch from something other than the "
          "parameter -- the gate could then read a value the cfg never set")
    check("S4b the member carries an in-class `= false` initialiser",
          p_member_defaults_false(swmm_h),
          "the member is indeterminate on an object whose initialize did not "
          "run, which is the one failure mode here that is not reproducible")
    check("S5 the prune is gated by the NEGATED member and keeps `>= 2` / `- 2`",
          p_gate_sense_and_threshold(swmm_h),
          "the guard is absent, its sense is inverted, or the keep-2 threshold "
          "or offset drifted -- the OFF path is the one the developer required "
          "be untouched")
    check("S6 the writer contains exactly one removal call",
          p_exactly_one_removal_site_in_the_writer(swmm_h),
          "a second removal site exists; the ON path would not retain every "
          "snapshot and no other check in this file could see it")
    check("S7 the writer reads the latched member and parses nothing",
          p_writer_does_not_reparse_the_flag(swmm_h),
          "the flag is re-parsed at the prune site, or the member is not read "
          "there at all")
    check("S9 output_cfg's rewritten-key set is the closed four and excludes this key",
          p_rewriter_cannot_touch_the_key(output_h),
          "the rewritten-key set changed, or the new key collides with one of "
          "them -- output_cfg matches `key=` by unanchored substring, so a "
          "collision would have the rewriter overwrite this key on every "
          "per-checkpoint cfg dump")
    check("S11 the prune's comment describes the override and the unchanged default",
          p_retention_comment_describes_both_paths(swmm_h),
          "the comment still presents keep-2 as the whole retention story, "
          "which is the state that trains the next reader to delete the gate")

    # ---------------------------------------------------------------------
    # D* TRUTH-TABLE DIFFERENTIAL over the gate condition.  Two arms, neither
    # substituting for the other: D1 is the SATISFYING arm (OFF must reproduce
    # the pre-change condition at every checkpoint id, which is the additive
    # property), D2 is the VIOLATING arm (ON must suppress the prune at every id
    # where the old condition fired, which is the capability).  D1 alone passes
    # a gate that never fires; D2 alone passes a gate that always fires.
    # ---------------------------------------------------------------------
    ids = list(range(0, 200))
    check("D1 OFF reproduces the pre-change condition at every checkpoint id",
          all(gate(False, c) == old_gate(c) for c in ids),
          "the OFF arm diverges from `checkpoint_id >= 2` at "
          + repr([c for c in ids if gate(False, c) != old_gate(c)][:8]))
    pruned_before = [c for c in ids if old_gate(c)]
    check("D2 ON suppresses the prune at every id where it previously fired",
          len(pruned_before) > 0 and not any(gate(True, c) for c in pruned_before),
          "the ON arm still prunes at "
          + repr([c for c in pruned_before if gate(True, c)][:8])
          + " (or the violating arm is empty, which would make D2 vacuous)")

    # S10: the corpus arm of the additive property.  Every cfg shipped in this
    # tree omits the key, so every one of them takes the argsd default and keeps
    # the retention it already had.  This is an ARTIFACT check rather than a
    # source check, and it is the closest thing to a before/after on a real cfg
    # that is available without a compiler.
    cfgs = sorted(p for p in root.rglob("*.cfg") if ".git" not in p.parts)
    carriers = [p for p in cfgs if KEY in p.read_text(errors="replace")]
    check("S10 no in-tree cfg carries the key, so all %d take the default" % len(cfgs),
          len(cfgs) > 0 and not carriers,
          "these cfgs carry the key and so are NOT evidence of default-inertness: "
          + repr([str(p) for p in carriers][:8])
          + " (or zero cfgs were found, which would make S10 vacuous)")

    # ---------------------------------------------------------------------
    # DEFECT PROBES.  Each mutates source in memory into the form its paired
    # check exists to catch.  Each mutation is written to be PLAUSIBLE -- the
    # edit a later maintainer might actually make -- rather than merely
    # destructive, because a check that only catches vandalism catches nothing
    # that will happen.  A bare "\n" is the CORRECT anchor in every mutation
    # below, including against the two files that are CRLF on disk, because
    # Path.read_text() has already translated them -- see hazard (1) in the
    # module docstring, which asserted the reverse until it was measured.
    # ---------------------------------------------------------------------
    probes = [
        # the type drifts from bool to int: compiles, works, but leaves the
        # struct's flag group and the field's two-valued semantics disagreeing
        ("X1 -> S1", lambda: p_declared_bool_under_swmm_ifdef(
            re.sub(r"bool(\s*)%s;" % KEY, r"int\g<1>%s;" % KEY, cfg_h, count=1))),
        # the read drifts to the mandatory form its two manhole siblings use
        ("X2 -> S2", lambda: p_read_is_additive_not_mandatory(
            cfg_h.replace('argsd("%s", argmap, "0")' % KEY,
                          'args("%s", argmap)' % KEY, 1))),
        # THE FLIPPED DEFAULT: the one drift in the read that silently changes
        # every pre-existing cfg's behaviour while staying additive
        ("X2b -> S2", lambda: p_read_is_additive_not_mandatory(
            cfg_h.replace('argsd("%s", argmap, "0")' % KEY,
                          'argsd("%s", argmap, "1")' % KEY, 1))),
        # the parameter loses its const: compiles, and the next reader reads a
        # threaded value as mutable in-function
        ("X3 -> S3", lambda: p_threaded_through_initialize(
            swmm_h.replace("const bool %s)" % KEY, "bool %s)" % KEY, 1))),
        ("X3b -> S3b", lambda: p_call_site_passes_the_arglist_field(
            triton_h.replace("arglist." + KEY, "false", 1))),
        # the guarded region closes BEFORE the forwarded argument: the shape a
        # reflow or a merge of that call's continuation lines would produce,
        # which breaks only the configuration nothing here parses
        ("X8 -> S8", lambda: p_call_site_is_inside_a_swmm_ifdef(
            triton_h.replace("                          arglist." + KEY + ");",
                             "#endif" + "\n"
                             "                          arglist." + KEY + ");", 1))),
        # THE BALANCED DRIFT, which is the one no other instrument in this tree
        # sees.  The read is RELOCATED below the region's existing `#endif`
        # rather than having an `#endif` inserted above it: the directives stay
        # balanced, the byte length is unchanged, and -- MEASURED -- the coupled
        # `SOURCE-SYNTAX-PARSE-ALL-TUS` pass stays GREEN while a define-free
        # pass over src/main.cpp raises `'struct ConfigUtils::arguments<double>'
        # has no member named swmm_snapshot_keep_all`.  The unbalanced form is
        # NOT used here on purpose: it produces `#endif without #if`, which the
        # already-registered parse node catches today, so a probe written that
        # way would be testing a defect this file does not own.
        ("X8b -> S8b", lambda: p_read_is_inside_a_swmm_ifdef(_perturbed(
            "X8b", cfg_h, _drift_read_below_endif(cfg_h)))),
        # "completing the pattern": the key is threaded into init_swmm, which
        # has no use for it
        ("X3c -> S3c", lambda: p_not_threaded_into_init_swmm(
            swmm_h.replace(
                "void init_swmm(std::string project_dir, std::string inp_filename, "
                "const bool swmm_snapshot_disable);",
                "void init_swmm(std::string project_dir, std::string inp_filename, "
                "const bool swmm_snapshot_disable, const bool %s);" % KEY, 1))),
        # THE DROPPED LATCH: initialize accepts the parameter and never stores
        # it, so the member keeps its false default and the key does nothing.
        # This compiles, and every signature and call-site check stays green.
        ("X4 -> S4", lambda: p_latched_to_member_exactly_once(
            swmm_h.replace("this->%s = %s;" % (MEMBER, KEY), "", 1))),
        # the in-class initialiser is dropped during a tidy-up
        ("X4b -> S4b", lambda: p_member_defaults_false(
            swmm_h.replace("bool %s = false;" % MEMBER, "bool %s;" % MEMBER, 1))),
        # THE INVERTED GUARD: the single highest-value probe in this file
        ("X5 -> S5", lambda: p_gate_sense_and_threshold(
            swmm_h.replace("if (!%s && checkpoint_id >= 2)" % MEMBER,
                           "if (%s && checkpoint_id >= 2)" % MEMBER, 1))),
        # the keep-2 offset drifts to keep-3: the OFF path stops reproducing the
        # behaviour the developer required be untouched
        ("X5b -> S5", lambda: p_gate_sense_and_threshold(
            swmm_h.replace("snapshot_path_for(checkpoint_id - 2)",
                           "snapshot_path_for(checkpoint_id - 3)", 1))),
        # a second, locally-reasonable cleanup appears in the writer and is NOT
        # gated -- the ON path then silently fails to retain everything
        ("X6 -> S6", lambda: p_exactly_one_removal_site_in_the_writer(
            swmm_h.replace(
                "\t\tif (!%s && checkpoint_id >= 2)" % MEMBER,
                "\t\t{ std::error_code ec2; "
                "std::filesystem::remove(snapshot_path_for(checkpoint_id - 3), ec2); }"
                "\t\tif (!%s && checkpoint_id >= 2)" % MEMBER, 1))),
        # the gate is re-derived at the prune site from the cfg map instead of
        # read from the latched member
        ("X7 -> S7", lambda: p_writer_does_not_reparse_the_flag(
            swmm_h.replace("if (!%s && checkpoint_id >= 2)" % MEMBER,
                           "if (!atoi(\"0\") && checkpoint_id >= 2)", 1))),
        # output_cfg gains a fifth rewritten key: not a defect in itself, but it
        # is the change that has to be re-checked against this key, and S9 is
        # what forces that re-check
        ("X9 -> S9", lambda: p_rewriter_cannot_touch_the_key(
            output_h.replace('line.find("it_count=")',
                             'line.find("courant=")', 1))),
        # THE REGRESSION FORM: a maintainer collapses the retention comment back
        # to the pre-change claim.  This is not hypothetical -- it is the text
        # that shipped before this change and that said "two is enough",
        # which is true of the resume and false of the experiment.
        ("X11 -> S11", lambda: p_retention_comment_describes_both_paths(
            "\n".join(
                ln for ln in swmm_h.split("\n")
                if not (ln.lstrip().startswith("//")
                        and (KEY in ln or "BY DEFAULT" in ln))))),
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
    print("PASS: the retention override's source properties hold, and every "
          "check discriminates.")
    print("NOTE: this is STRUCTURAL coverage plus a truth-table differential "
          "over the gate CONDITION. The behavioural two-arm differential -- a "
          "coupled run with swmm_snapshot_keep_all=1 still holding the snapshot "
          "for checkpoint N-2 after writing N, and NOT holding it with the key "
          "unset -- requires a compiled coupled solver and is NOT discharged "
          "here. A green here does NOT discharge it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
