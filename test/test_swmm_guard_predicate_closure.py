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
    condition -- including the per-step coupling block, which is gated on the
    LOCAL count and encloses an ``MPI_Gatherv`` and an ``MPI_Scatterv`` over
    ``ENSIFY_COMM_WORLD``.  On the same decomposition this file is about, rank 0
    skips both while every manhole-owning rank enters them and the run
    DEADLOCKS.  That site is a real and separate defect; subtest G3 below
    OBSERVES it and reports it, deliberately without failing on it, so the
    number is visible rather than implied.
  * a rank test hoisted into an enclosing block.

A check that names its own boundary is worth more than one that implies
coverage it does not have.
"""

import re
import sys
from pathlib import Path

PREDICATE_RE = re.compile(r"rank == 0 && swmm_model\.(\w+)")
LOCAL_ONLY_RE = re.compile(r"if \(swmm_model\.(\w+) > 0\)")


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

    # --- G3: OBSERVE the sites this operation cannot reach -------------------
    #
    # Reported, not asserted. These are `swmm_model.<count> > 0` conditions with
    # no rank test in the same condition, so the published operation is blind to
    # them by construction. Printing the count keeps the boundary visible in the
    # artifact instead of only in this docstring.
    unreached = []
    for n, line in enumerate(lines, 1):
        if line.lstrip().startswith("//") or PREDICATE_RE.search(line):
            continue
        m = LOCAL_ONLY_RE.search(line)
        if m and m.group(1) == "num_of_swmm_links":
            unreached.append((n, line.strip()))
    print("")
    print("NOTE: %d local-count site(s) carry no rank test in the same condition "
          "and are OUTSIDE this operation:" % len(unreached))
    for n, t in unreached:
        print("        :%d  %s" % (n, t))
    print("      The per-step coupling block among them encloses two MPI "
          "collectives; on a rank-0-owns-no-manhole")
    print("      decomposition rank 0 skips them while other ranks enter, and "
          "the run deadlocks. Not this check's")
    print("      subject, and not fixed by chunk (12).")
    print("")

    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: the guard-predicate closure holds on the tree as landed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
