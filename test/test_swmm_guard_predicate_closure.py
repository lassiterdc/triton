#!/usr/bin/env python3
"""Guard-predicate closure check over src/triton.h (WP-1B chunk 13).

THE PROPERTY
------------
*Every rank-0-guarded call into ``swmm_model`` guards on the same predicate.*

Sec 4.6.1 states the operation verbatim::

    grep -n "rank == 0 && swmm_model\\." src/triton.h

and the check is that the predicate is IDENTICAL on every returned line.

WHY IT IS A CHECK AND NOT A CONVENTION
--------------------------------------
``num_of_swmm_links`` is rank 0's LOCAL (subdomain) count; ``global_num_of_swmm_links``
is the whole-domain count.  On a row-striped decomposition where rank 0's top
strip holds no manhole they differ, and a site that tests the local count where
the operation is global does nothing -- SILENTLY, on a run that did nothing
wrong.  There is no error, no warning, and no artifact that shows the skip.  The
two predicates are one token apart and read identically at a glance, which is
exactly why a reviewer does not catch the divergence and a grep does.

At the time chunk (12) was written this check returned 3 lines carrying 2
distinct predicates and FAILED: ``:453`` used the global count and its own
comment stated the rule, while the snapshot-write and ``flush_exchange_log``
sites used the local one.  Chunk (12) made all three agree.

WHAT THIS DOES *NOT* COVER, stated so the green is not over-read
---------------------------------------------------------------
The operation is keyed on the literal ``rank == 0 && swmm_model.`` prefix, so it
reaches only the sites that spell the rank test and the member access together
in one condition.  It does NOT reach:

  * ``if (swmm_model.num_of_swmm_links > 0)`` with no ``rank ==`` in the same
    condition.  Two such sites were real defects on the rank-0-owns-no-manhole
    decomposition; both are repaired, and subtests G3a and G3b below now ASSERT
    rather than observe, so neither can silently return.
  * a rank test hoisted into an enclosing block.

  * an index site outside src/triton.h.  G4a asserts there is none, rather
    than leaving the walk's single-file extent as an unstated assumption.

A check that names its own boundary is worth more than one that implies
coverage it does not have.

WHY G4 EXISTS: G1-G3 COVER THE PREDICATE, NOT THE INDEX SITE
------------------------------------------------------------
G3a and G3b are EXCLUSION assertions -- nothing of a named kind may sit inside a
local-count-guarded block.  They are silent on the converse, and the repair has
a second half that lives there.  Measured: removing the INNER local-count guard
from the per-step coupling block (guard count 3 -> 2) leaves G1-G3c returning
REAL EXIT 0, so a tree on which `host_vec[SWMM_NEWD]` is an out-of-range
`operator[]` on a rank owning no sewer node passes this file clean.  Nothing
moved that G3a or G3b looks for; the guard that made the index legal simply
stopped being there.

G4 is the inverse walk: every `SWMM_*`-indexed access to the flat host/device
vectors must BE enclosed by a local-count guard.  The predicate has to be the
LOCAL count -- a global-count guard is not a weaker form of the same check but
the specific bug triton.h's own comment describes, because the SWMM_LOSS..SWMM_Q
entries are appended by a block that is itself local-count-guarded, so only a
predicate matching the APPEND's condition establishes that what is being indexed
exists.

WHY G3 IS TWO ASSERTIONS AND NOT ONE
------------------------------------
The two repaired sites belong to DIFFERENT classes, and one predicate cannot
reach both.

G3a asserts that no MPI collective is lexically enclosed by a local-count guard.
That is the per-step coupling block's defect exactly: it enclosed an
``MPI_Gatherv`` and an ``MPI_Scatterv`` over ``ENSIFY_COMM_WORLD``, so a rank
whose strip held no manhole skipped two collectives every other rank entered and
the run DEADLOCKED.  G3a is correctly SILENT on the host/device-vector append
block in ``initialize()``, which is legitimately local-count-guarded because it
appends per-rank buffers and enters no collective, and on the per-rank device
transfers inside the repaired coupling block for the same reason.

G3a IS BLIND TO THE SECOND SITE, and shipping it alone would have left that site
with no static guard at all.  ``end_swmm`` encloses NO collective -- it is
internally ``rank_ == 0``-guarded around swmm_end/swmm_report/swmm_close -- so it
could never deadlock, and G3a cannot see it.  Its defect is different in kind:
``init_swmm`` opens SWMM on rank 0 whenever the MODEL has any node, guarding
``swmm_open``/``swmm_start`` on ``rank_ == 0`` and nothing else, so a
local-count-guarded finalizer left an opened engine never closed and never
reported -- no hydraulics.rpt, silently.  Repairing only the collective site
would have converted a loud hang into a missing report file on exactly the runs
the repair rescues, which is why the two land together and why the guard needs
two assertions rather than one.

G3b therefore asserts a different property: no ``swmm_model.<method>(`` CALL is
lexically enclosed by a local-count guard.  A call into the coupling object is an
operation on the MODEL, whose scope is global; a member-data access
(``swmm_model.loss.data()``, ``swmm_model.num_of_swmm_links``) is not a call and
does not match, which is what keeps G3b silent on the append block.

LIMITATION OF THE BRACE WALK, stated so a green is not over-read: the enclosure
scan strips ``//`` line comments and counts braces.  It does NOT parse block
comments, string literals containing braces, or preprocessor conditionals.  It
is sound on this file's style and would need a real parser to be sound in
general.  Both subtests print the block extent they walked, so a reader can
check the walk rather than trust it.
"""

import re
import sys
from pathlib import Path

PREDICATE_RE = re.compile(r"rank == 0 && swmm_model\.(\w+)")
LOCAL_ONLY_RE = re.compile(r"if \(swmm_model\.(\w+) > 0\)")

# G3a's subject: any MPI entry point. Collectives are the deadlocking class, but
# the pattern is deliberately wider than MPI_Gatherv/MPI_Scatterv -- a future
# edit that moves an MPI_Allreduce or an MPI_Barrier under a local-count guard is
# the same defect, and enumerating today's two calls would not catch it.
MPI_CALL_RE = re.compile(r"\bMPI_[A-Z]\w*\s*\(")

# G3b's subject: a CALL into the coupling object. The `(` immediately after the
# member name is what separates `swmm_model.end_swmm(` from `swmm_model.loss` and
# from `swmm_model.loss.data()`, whose `(` follows `data`, not `loss`.
SWMM_MODEL_CALL_RE = re.compile(r"\bswmm_model\.(\w+)\s*\(")


def strip_line_comment(text):
    """Drop a `//` tail. Sound on this file's style; see the docstring's stated
    limitation -- block comments and brace-bearing string literals are not
    parsed."""
    cut = text.find("//")
    return text if cut < 0 else text[:cut]


def enclosed_block(lines, start_idx):
    """Return (first_idx, last_idx) of the brace-delimited block opened at or
    after `start_idx`, inclusive, or None if no brace opens on the guard line.

    `start_idx` is 0-based. The walk begins on the guard line itself, so a guard
    written `if (...) {` is handled; a braceless single-statement guard returns
    None and is reported by the caller rather than silently skipped."""
    depth = 0
    opened = False
    for i in range(start_idx, len(lines)):
        for ch in strip_line_comment(lines[i]):
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return (start_idx, i)
        if not opened and i > start_idx:
            # The guard line carried no `{` and neither did the next line: a
            # braceless single-statement guard.
            return None
    return None


def local_count_guards(lines):
    """Every `if (swmm_model.num_of_swmm_links > 0)` site, as
    (line_no_1based, block_first_idx, block_last_idx, text)."""
    out = []
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("//") or PREDICATE_RE.search(line):
            continue
        m = LOCAL_ONLY_RE.search(line)
        if not m or m.group(1) != "num_of_swmm_links":
            continue
        block = enclosed_block(lines, n - 1)
        if block is None:
            out.append((n, None, None, line.strip()))
        else:
            out.append((n, block[0], block[1], line.strip()))
    return out


def scan_guarded_blocks(lines, guards, pattern):
    """Return [(guard_line, hit_line, hit_text), ...] for every `pattern` match
    lexically inside a local-count-guarded block, comments excluded."""
    hits = []
    for guard_line, first, last, _ in guards:
        if first is None:
            continue
        for i in range(first, last + 1):
            raw = lines[i]
            if raw.lstrip().startswith("//"):
                continue
            body = strip_line_comment(raw)
            if pattern.search(body):
                hits.append((guard_line, i + 1, raw.strip()))
    return hits


def main(argv):
    repo = Path(argv[1] if len(argv) > 1 else ".").resolve()
    triton_h = repo / "src" / "triton.h"
    if not triton_h.is_file():
        print("FAIL: %s not found" % triton_h)
        return 1

    lines = triton_h.read_text().split("\n")
    failures = []

    def check(label, ok, detail=""):
        print("%s %s%s" % ("ok   " if ok else "FAIL ", label,
                           "" if ok else ("  -- %s" % detail)))
        if not ok:
            failures.append(label)

    # --- G1: the operation returns at least one site -------------------------
    #
    # Without this the whole file is vacuous: a grep that matches nothing
    # satisfies "every returned line carries the same predicate" trivially, and
    # a rename of the member would turn this test green by deleting its subject.
    hits = []
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("//"):
            continue          # a predicate quoted in a comment is not a call site
        m = PREDICATE_RE.search(line)
        if m:
            hits.append((n, m.group(1), line.strip()))

    check("G1  the operation returns at least one rank-0-guarded site",
          len(hits) > 0,
          "grep for 'rank == 0 && swmm_model.' matched nothing -- the member "
          "was renamed, or this check lost its subject")
    if not hits:
        return 1

    # --- G2: every returned line carries the SAME predicate ------------------
    predicates = sorted({p for _, p, _ in hits})
    detail = "; ".join("%d: %s" % (n, t) for n, _, t in hits)
    check("G2  every rank-0-guarded swmm_model site guards on ONE predicate "
          "(%d site(s), %d distinct)" % (len(hits), len(predicates)),
          len(predicates) == 1, detail)

    # --- G2b: and that predicate is the GLOBAL count -------------------------
    #
    # G2 alone is satisfied by making all three agree on the LOCAL count, which
    # would be uniformly wrong. Agreement is necessary; it is not sufficient.
    check("G2b that one predicate is global_num_of_swmm_links, not the local count",
          predicates == ["global_num_of_swmm_links"],
          "predicate(s) found: %s" % ", ".join(predicates))

    # --- G3: ASSERT over the sites the G1/G2 operation cannot reach ----------
    #
    # These are `swmm_model.<count> > 0` conditions with no rank test in the same
    # condition, so the published operation is blind to them by construction.
    # They were OBSERVE-only while the two defects below were open; both are now
    # repaired and both assertions are live. The block extents are printed so the
    # brace walk is checkable rather than trusted.
    guards = local_count_guards(lines)
    print("")
    print("G3  %d local-count guard site(s) carry no rank test in the same "
          "condition:" % len(guards))
    for n, first, last, t in guards:
        extent = ("braceless single statement" if first is None
                  else "block :%d-:%d" % (first + 1, last + 1))
        print("        :%d  %-58s  [%s]" % (n, t[:58], extent))
    print("")

    # --- G3a: no MPI call inside a local-count-guarded block -----------------
    #
    # The deadlocking class. A local-count guard is a PER-RANK predicate and an
    # MPI collective is a WHOLE-COMMUNICATOR operation, so the two can never be
    # correctly composed: the decomposition that makes the counts differ is
    # exactly the one on which some ranks enter and others do not.
    mpi_hits = scan_guarded_blocks(lines, guards, MPI_CALL_RE)
    check("G3a no MPI call is lexically enclosed by a local-count guard "
          "(%d hit(s))" % len(mpi_hits),
          not mpi_hits,
          "; ".join(":%d (guard at :%d) %s" % (h, g, t) for g, h, t in mpi_hits))

    # --- G3b: no swmm_model method CALL inside a local-count-guarded block ---
    #
    # The silently-lost-artifact class, which G3a cannot see because its member
    # of it -- end_swmm -- encloses no collective at all. A call into the
    # coupling object operates on the MODEL, whose scope is global; gating one on
    # this rank's subdomain count skips whole-model work on a rank that happens
    # to own no node. Member-data access does not match: the `(` must follow the
    # member name, so `swmm_model.loss.data()` is not a hit.
    call_hits = scan_guarded_blocks(lines, guards, SWMM_MODEL_CALL_RE)
    check("G3b no swmm_model method call is lexically enclosed by a local-count "
          "guard (%d hit(s))" % len(call_hits),
          not call_hits,
          "; ".join(":%d (guard at :%d) %s" % (h, g, t) for g, h, t in call_hits))

    # --- G3c: the walk reached a block on every guard ------------------------
    #
    # Without this, a guard rewritten into a braceless single statement makes
    # G3a and G3b vacuously green on it -- they scan an empty range and find
    # nothing, which is byte-identical at the exit code to finding nothing in a
    # block that was genuinely clean.
    braceless = [(n, t) for n, first, _, t in guards if first is None]
    check("G3c the brace walk reached a block on every local-count guard "
          "(%d unwalked)" % len(braceless),
          not braceless,
          "; ".join(":%d %s" % (n, t) for n, t in braceless))
    # --- G4: the INVERSE walk -- every SWMM_* index site IS guarded ----------
    #
    # G3a and G3b are EXCLUSION assertions: nothing of a named kind may appear
    # inside a local-count-guarded block.  They say nothing about what must
    # appear inside one, and the repair has a second half they therefore cannot
    # reach.  Removing the inner local-count guard at the per-step coupling
    # block leaves this file returning 0: a tree on which `host_vec[SWMM_NEWD]`
    # is an out-of-range `operator[]` on a rank owning no sewer node passes
    # clean, because no MPI call and no swmm_model method call moved.
    #
    # G4 asserts the converse.  Every `SWMM_*`-indexed access to the flat
    # host/device vectors MUST be lexically enclosed by a LOCAL-count guard.
    #
    # THE PREDICATE MUST BE THE LOCAL COUNT, and a global-count guard is not a
    # weaker version of the same thing -- it is the specific bug triton.h's own
    # comment at the coupling block describes.  The SWMM_LOSS..SWMM_Q entries
    # are appended to host_vec/device_vec by a block in initialize() that is
    # itself guarded on the LOCAL count, so on a zero-local-link rank those
    # indices are past the end of a 23-entry vector with no bounds check.  Only
    # a predicate matching the APPEND's condition establishes that the entries
    # being indexed exist.
    #
    # The vector names are matched as a set rather than enumerated one by one,
    # and the index as any `SWMM`-prefixed macro, because naming today's five
    # macros would leave a sixth uncovered on the day it is added -- the same
    # reason G3a matches any `MPI_*` rather than today's two collectives.
    site_pat = re.compile(
        r"\b(?:host_vec|device_vec|host_vec_int|device_vec_int)\s*\[\s*(SWMM\w*)\s*\]")

    # G4a: the population is confined to the file this walk covers.  Stated as
    # an assertion rather than assumed, because the walk reads only triton.h:
    # an index site added in another header would be invisible here and G4
    # would stay green while covering less than it claims.
    stray = []
    for path in sorted((repo / "src").glob("*.h")):
        if path.name == "triton.h":
            continue
        for n, raw in enumerate(path.read_text().split("\n"), 1):
            if raw.lstrip().startswith("//"):
                continue
            if site_pat.search(strip_line_comment(raw)):
                stray.append("%s:%d" % (path.name, n))
    check("G4a every SWMM_* index site lives in triton.h, which is the file "
          "this walk covers (%d elsewhere)" % len(stray),
          not stray, "; ".join(stray))

    sites = []
    for n, raw in enumerate(lines, 1):
        if raw.lstrip().startswith("//"):
            continue          # an index spelled in a comment is not an access
        m = site_pat.search(strip_line_comment(raw))
        if m:
            sites.append((n, m.group(1), raw.strip()))

    # G4b: without this the assertion is vacuous.  A rename of the SWMM_*
    # macros, or of the vectors, would empty the population and turn G4 green
    # by deleting its subject -- the same failure G1 exists to stop above.
    check("G4b the SWMM_* index-site population is non-empty (%d site(s))"
          % len(sites),
          len(sites) > 0,
          "no host_vec/device_vec SWMM_* index site found -- the vectors or the "
          "index macros were renamed, and this assertion lost its subject")

    if sites:
        local_ranges = [(g[1], g[2], g[0]) for g in guards if g[1] is not None]
        unguarded = []
        for n, macro, text in sites:
            idx = n - 1
            if not any(first <= idx <= last for first, last, _ in local_ranges):
                unguarded.append((n, macro, text))
        print("G4  %d SWMM_* index site(s); enclosing local-count guard per site:"
              % len(sites))
        for n, macro, text in sites:
            idx = n - 1
            owner = next((gl for first, last, gl in local_ranges
                          if first <= idx <= last), None)
            print("        :%-5d %-16s %s" % (
                n, macro,
                "guarded by :%d" % owner if owner else "UNGUARDED"))
        print("")
        check("G4  every SWMM_* index site is lexically enclosed by a "
              "LOCAL-count guard (%d unguarded)" % len(unguarded),
              not unguarded,
              "; ".join(":%d %s -- %s" % (n, macro, t[:60])
                        for n, macro, t in unguarded))
    print("")

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: the guard-predicate closure holds on the tree as landed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
