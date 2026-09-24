#!/usr/bin/env python3
"""Guard-of-the-guard for the SWMM column split (WP-1A).

WHY THIS EXISTS
---------------
``test_swmm_column_split_source.py`` returns 0 on the landed tree.  So would a
guard whose every predicate was inverted, mis-anchored, or matching nothing at
all -- and the difference is invisible from a green run.  A guard with no guard
of its own can be weakened by a later edit and stay green, which is precisely the
class of defect the split was dispatched to remove from the timer accounting.

WHAT IT DOES
------------
For each named DEFECT below it materialises a throwaway copy of the source tree,
applies a single surgical mutation that reintroduces that defect, and asserts the
source guard goes RED **and names the right subtest**.  A mutation that leaves
the guard green is reported with the subtest that should have caught it.

The mutations are the placements a builder could plausibly reach for -- each one
is either a reading of the design that a careless reviewer would accept, or the
obvious "simplification" of the landed form:

  M1  declare SWMM_OTHER as a macro           -> S1  (the three-not-four ground)
  M2  hoist SWMM_STEP's bracket outside the
      rank-0 guard                            -> S3  (the placement decision)
  M3  collapse SWMM_MPI's two disjoint
      brackets into one spanning pair         -> S2  (swallows SWMM_STEP)
  M4  move a child bracket outside the parent -> S2  (breaks closure only)
  M5  subtract the children from other_time
      as well as the parent                   -> S4  (double-counts)
  M6  read the residual via get_custom_time
      on a bare literal                       -> S4/S5 (returns 0.0 silently)
  M7  drop SWMM_STEP's average(...) term      -> S4  (breaks Average arity)
  M8  drop a column from the header literal   -> S4
  M9  overlap two child brackets so a region
      is counted twice                        -> S6  (the general form of M3;
                                                      added on review, and the
                                                      one mutation no runtime
                                                      instrument can see)

USAGE
-----
    test_swmm_column_split_guard_forms.py <repo-root>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

GUARD = "test_swmm_column_split_source.py"


def _sub_in(path, pattern, repl, count=1):
    """Apply one regex substitution to a file; raise if it did not bite."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    new, n = re.subn(pattern, repl, text, count=count)
    if n != count:
        raise RuntimeError("mutation did not apply to %s: %r matched %d time(s), "
                           "wanted %d" % (path, pattern, n, count))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)


# ---------------------------------------------------------------------------
# the mutations
# ---------------------------------------------------------------------------

def M1_declare_swmm_other_macro(root):
    _sub_in(os.path.join(root, "src", "constants.h"),
            r'(#define SWMM_STEP "swmm_step")',
            r'\1\n#define SWMM_OTHER "swmm_other"')


def M2_hoist_step_bracket_outside_rank0_guard(root):
    """Pull the start above the guard and the stop below its closing brace.

    Done line-wise rather than by regex because a comment block sits between the
    guard line and the bracket, so no adjacency-assuming pattern survives an edit
    to that comment -- and a mutation that silently stops matching is exactly the
    failure this whole file exists to make loud.
    """
    p = os.path.join(root, "src", "triton.h")
    with open(p, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    start_i = [i for i, ln in enumerate(lines) if "st.start(SWMM_STEP);" in ln]
    stop_i = [i for i, ln in enumerate(lines) if "st.stop(SWMM_STEP);" in ln]
    guard_i = [i for i, ln in enumerate(lines)
               if re.search(r'if\s*\(\s*rank\s*==\s*0\s*\)', ln)]
    if len(start_i) != 1 or len(stop_i) != 1 or not guard_i:
        raise RuntimeError("expected one SWMM_STEP pair and a rank-0 guard; "
                           "found %d/%d/%d" % (len(start_i), len(stop_i),
                                               len(guard_i)))
    g = max(i for i in guard_i if i < start_i[0])

    # The closing brace of the guarded block, by brace balance from the guard.
    depth = 0
    close = None
    for i in range(g, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if i > g and depth == 0:
            close = i
            break
    if close is None:
        raise RuntimeError("could not find the rank-0 block's closing brace")

    start_txt, stop_txt = lines[start_i[0]], lines[stop_i[0]]
    out = []
    for i, ln in enumerate(lines):
        if i in (start_i[0], stop_i[0]):
            continue
        if i == g:
            out.append(start_txt)
        out.append(ln)
        if i == close:
            out.append(stop_txt)
    with open(p, "w", encoding="utf-8") as fh:
        fh.writelines(out)


def M3_collapse_swmm_mpi_to_one_bracket(root):
    p = os.path.join(root, "src", "triton.h")
    _sub_in(p, r' *st\.stop\(SWMM_MPI\);\n(?=\n* *// Step SWMM forward)', "")
    _sub_in(p, r' *st\.start\(SWMM_MPI\);\n(?= *// Scatter new_depth)', "")


def M4_move_child_bracket_outside_parent(root):
    p = os.path.join(root, "src", "triton.h")
    _sub_in(p, r'( *)st\.start\(SWMM_TIME\);\n', r'\1st.start(SWMM_XFER);\n\1st.start(SWMM_TIME);\n')
    _sub_in(p, r' *st\.start\(SWMM_XFER\);\n(?= *\n* *// Copy new_depth)', "")


def M5_subtract_children_from_other_time(root):
    _sub_in(os.path.join(root, "src", "output.h"),
            r'(T other_time = simulation_time - compute_time - mpi_time - io_time'
            r' - resize_time - swmm_time);',
            r'\1 - swmm_xfer_time;')


def M6_read_residual_via_bare_literal(root):
    _sub_in(os.path.join(root, "src", "output.h"),
            r'T swmm_other_time = swmm_time - swmm_xfer_time - swmm_mpi_time'
            r' - swmm_step_time;',
            r'T swmm_other_time = st.get_custom_time("swmm_other");')


def M7_drop_step_average_term(root):
    _sub_in(os.path.join(root, "src", "output.h"),
            r'average\(swmm_step_time_all,size_\) << ", " << ', "")


def M8_drop_a_header_column(root):
    _sub_in(os.path.join(root, "src", "output.h"),
            r'SWMM_STEP, SWMM_OTHER, Other', r'SWMM_OTHER, Other')


def M9_overlap_two_child_brackets(root):
    """Move st.stop(SWMM_XFER) after st.start(SWMM_MPI) so the Gatherv region
    falls inside BOTH children and is counted twice.

    This is the GENERAL form of the defect M3 covers in one special case, and it
    is the reviewer's own mutation. Every predicate S2 evaluates still holds --
    both brackets are inside the parent and SWMM_MPI keeps its two pairs -- so
    only S6's pairwise-disjointness check rejects it. The runtime instruments
    cannot: the derived residual keeps the emitted sum exact and goes NEGATIVE.
    """
    p = os.path.join(root, "src", "triton.h")
    with open(p, "r", encoding="utf-8") as fh:
        text = fh.read()
    stop_xfer = "      st.stop(SWMM_XFER);\n"
    anchor = "      st.start(SWMM_MPI);\n      // Gather exchange_q"
    if text.count(stop_xfer) != 1 or text.count(anchor) != 1:
        raise RuntimeError("overlap mutation could not be anchored "
                           "(stop_xfer=%d anchor=%d)"
                           % (text.count(stop_xfer), text.count(anchor)))
    text = text.replace(stop_xfer, "", 1)
    text = text.replace(
        anchor, "      st.start(SWMM_MPI);\n" + stop_xfer + "      // Gather exchange_q", 1)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


MUTATIONS = (
    ("M1_declare_swmm_other_macro", M1_declare_swmm_other_macro,
     "S1_three_child_macros_and_no_swmm_other"),
    ("M2_hoist_step_bracket_outside_rank0_guard", M2_hoist_step_bracket_outside_rank0_guard,
     "S3_swmm_step_inside_rank0_guard"),
    ("M3_collapse_swmm_mpi_to_one_bracket", M3_collapse_swmm_mpi_to_one_bracket,
     "S2_children_inside_unchanged_parent"),
    ("M4_move_child_bracket_outside_parent", M4_move_child_bracket_outside_parent,
     "S2_children_inside_unchanged_parent"),
    ("M5_subtract_children_from_other_time", M5_subtract_children_from_other_time,
     "S4_write_times_derivation_and_columns"),
    ("M6_read_residual_via_bare_literal", M6_read_residual_via_bare_literal,
     "S4_write_times_derivation_and_columns"),
    ("M7_drop_step_average_term", M7_drop_step_average_term,
     "S4_write_times_derivation_and_columns"),
    ("M8_drop_a_header_column", M8_drop_a_header_column,
     "S4_write_times_derivation_and_columns"),
    ("M9_overlap_two_child_brackets", M9_overlap_two_child_brackets,
     "S6_child_spans_are_pairwise_disjoint"),
)


def run_guard(root, subtest=None):
    guard = os.path.join(root, "test", GUARD)
    cmd = [sys.executable, guard] + ([subtest] if subtest else []) + [root]
    # subprocess.PIPE rather than capture_output=: the login nodes this runs on
    # carry python 3.6 as /usr/bin/python3, where capture_output does not exist.
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    return p.returncode, out.decode("utf-8", "replace")


def main(argv):
    if len(argv) != 2:
        print("usage: %s <repo-root>" % os.path.basename(argv[0]))
        return 2
    src_root = os.path.abspath(argv[1])

    # Baseline: the guard must be GREEN on the unmutated tree, or every red
    # below is uninformative.
    rc, out = run_guard(src_root)
    if rc != 0:
        print("FAIL: the guard is already RED on the unmutated tree, so no "
              "mutation result below means anything.\n%s" % out)
        return 1
    print("baseline          : guard GREEN on the unmutated tree")

    failures = []
    for name, mutate, expect in MUTATIONS:
        tmp = tempfile.mkdtemp(prefix="wp1a-mut-")
        try:
            for sub in ("src", "test"):
                shutil.copytree(os.path.join(src_root, sub),
                                os.path.join(tmp, sub))
            try:
                mutate(tmp)
            except RuntimeError as exc:
                failures.append(
                    "%s: could not be applied -- %s. A mutation that no longer "
                    "matches means the SOURCE moved and this guard-of-the-guard "
                    "is now measuring nothing." % (name, exc))
                print("ERROR %-44s (mutation stale)" % name)
                continue

            rc, out = run_guard(tmp, expect)
            if rc == 0:
                failures.append(
                    "%s: the guard stayed GREEN. %s was supposed to catch it.\n"
                    "%s" % (name, expect, out))
                print("FAIL  %-44s -> %s stayed green" % (name, expect))
            else:
                print("ok    %-44s -> %s went red" % (name, expect))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if failures:
        for f in failures:
            print("FAIL: %s" % f)
        return 1
    print("PASS: all %d defect forms are caught by the named subtest."
          % len(MUTATIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
