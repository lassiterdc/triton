/** @file test_coupling_declaration.cpp
 *  @brief Tier-1 tests for how the coupling nodes are declared.
 *
 *  No solver run, no cluster, no SWMM call. A cfg selects the nodes that
 *  exchange water with the surface by one key, swmm_coupling_nodes: a list file
 *  (only its names), ALL_JUNCTIONS (only the junctions), or nothing (the nodes
 *  SWMM gives a FLOW inflow, as every earlier build derived them). The three are
 *  exclusive: a real FLOW inflow -- a baseflow in a sewer or a stream -- must not
 *  make a node a coupling node when a declaration is given, or a model's inputs
 *  would silently move where the surface and the sewer trade water. These tests
 *  exercise the pure functions in swmm_triton.h that carry the declaration:
 *
 *    resolve_coupling_nodes -- the three exclusive branches over one parsed model;
 *    read_coupling_list     -- the list file's format;
 *    find_repeated_name     -- the duplicate refusal's predicate.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one
 *  node id per property; with no argument every test runs.
 */

// The same include prefix as test_coupled_list.cpp, for the reasons
// test_exchange_log_header.cpp states: swmm_triton.h compiles only after
// main.cpp's include chain.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace SWMM_triton;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

/** One parsed model for every arm: junctions J3, J1, J2 in that file order, and
 *  [INFLOWS] FLOW lines on J1 (a junction no list below declares) and on ST1 (a
 *  storage node, not a junction), plus a pollutant line on J2 that is not a FLOW
 *  inflow. So the legacy derivation selects {J1, ST1} and nothing else. */
static inp_coupled_list_t shared_model()
{
    const std::string text =
        "[TITLE]\nmodel\n\n"
        "[JUNCTIONS]\n;;Name Elev MaxDepth\nJ3 3.0 2.0\nJ1 1.0 0.0\nJ2 2.0 1.5\n\n"
        "[STORAGE]\nST1 0.0 2.0 0 FUNCTIONAL 1000 0 0\n\n"
        "[POLLUTANTS]\nTSS MG/L 0 0 0 0 0\n\n"
        "[INFLOWS]\n;;Node Constituent Series\nJ1 FLOW base_ts\nST1 FLOW st_ts\nJ2 TSS tss_ts\n\n"
        "[COORDINATES]\nJ3 1 1\nJ1 2 2\nJ2 3 3\nST1 4 4\n";
    std::istringstream is(text);
    return read_inp_sections(is);
}

static coupling_declaration_t list_of(const std::vector<std::string>& names)
{
    coupling_declaration_t d;
    d.kind = COUPLING_LIST;
    d.path = "nodes.txt";
    d.names = names;
    return d;
}

// ---------------------------------------------------------------------------
// Property: under a list, the coupling nodes are exactly the list, in [JUNCTIONS]
// order -- no [INFLOWS] line adds, removes or reorders one.
// Class: lists over a model whose FLOW lines sit on an undeclared junction and on a
// non-junction. Why: a baseflow FLOW series on J1 or ST1 must not make either a
// coupling node, or the surface exchanges water at a node the user did not choose.
// Killed by: adding the derived [INFLOWS] set to the list's result (the union
// mutant), or returning the list unsorted. Second correct implementation: filter
// [JUNCTIONS] in file order by membership in the list (same result for any list
// of junctions).
// ---------------------------------------------------------------------------
static void D1_list_is_exactly_the_list()
{
    const inp_coupled_list_t m = shared_model();
    CHECK(resolve_coupling_nodes(list_of({"J2", "J3"}), m) == std::vector<std::string>({"J3", "J2"}),
          "a list {J2, J3} gives exactly J3, J2 (junction order), not J1 or ST1 from [INFLOWS]");
    CHECK(resolve_coupling_nodes(list_of({"J2"}), m) == std::vector<std::string>({"J2"}),
          "a one-name list gives that name only, although J2 carries a pollutant inflow line");
}

// Property: ALL_JUNCTIONS gives exactly the junctions, in file order.
// Class: the same model, whose FLOW lines include a non-junction. Why: a storage
// node with a FLOW inflow must not be coupled (and so must not be refused) under
// ALL_JUNCTIONS. Killed by: adding the derived set, or dropping a junction with no
// FLOW line (J2, J3). Second correct implementation: collect [JUNCTIONS] names in
// a second pass of the file.
static void D2_all_junctions_is_exactly_the_junctions()
{
    coupling_declaration_t d;
    d.kind = COUPLING_ALL_JUNCTIONS;
    CHECK(resolve_coupling_nodes(d, shared_model()) == std::vector<std::string>({"J3", "J1", "J2"}),
          "ALL_JUNCTIONS gives J3, J1, J2 and not ST1");
}

// Property: with no declaration the coupling nodes are exactly the FLOW set.
// Class: the same model. Why: a cfg without swmm_coupling_nodes must keep the
// coupling every earlier build derived. Killed by: dropping the pollutant test (J2
// joins), or by not deriving (empty list). Second correct implementation: the
// pre-item-16 parse_inp_coupled_list, which returns {J1, ST1} on this model.
static void D3_legacy_is_exactly_the_flow_set()
{
    CHECK(resolve_coupling_nodes(coupling_declaration_t(), shared_model()) == std::vector<std::string>({"J1", "ST1"}),
          "no declaration gives J1 (a junction) then ST1 (not a junction, after them)");
}

// Property: the list file's format -- one name per line, surrounding whitespace
// stripped, blank and ';' lines skipped, each name's line number kept.
// Class: list files with indentation, CRLF, comments and blank lines. Why: a
// comment or a trailing space read as a name stops a valid run or couples a node
// nobody named. Killed by: keeping a ';' line, or keeping the trailing '\r'.
// Second correct implementation: a tokenizer splitting on whitespace and taking
// the first token of each non-comment line (same result for one-word names).
static void D4_list_file_format()
{
    std::istringstream in(";; coupling nodes\n  J1  \r\n\n; J9 is not coupled\nJ2\r\n");
    std::vector<std::string> names;
    std::vector<int> lines;
    read_coupling_list(in, names, lines);
    CHECK(names == std::vector<std::string>({"J1", "J2"}), "two names, stripped, comments and blanks skipped");
    CHECK(lines == std::vector<int>({2, 5}), "each name keeps its 1-based line number");
}

// Property: a repeated name is found after SWMM's case fold, naming both entries.
// Class: lists with an exact repeat, a case-only repeat, and no repeat. Why: SWMM
// resolves J1 and j1 to one node, so both would claim the same exchange slot.
// Killed by: a case-sensitive comparison. Second correct implementation: a set of
// upper-cased names checked on insertion.
static void D5_repeated_name_after_case_fold()
{
    CHECK(find_repeated_name({"J1", "J2", "j1"}) == std::make_pair(0, 2), "J1 and j1 repeat: entries 0 and 2");
    CHECK(find_repeated_name({"J1", "J2"}) == std::make_pair(-1, -1), "distinct names: no repeat");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"D1_list_is_exactly_the_list", D1_list_is_exactly_the_list},
    {"D2_all_junctions_is_exactly_the_junctions", D2_all_junctions_is_exactly_the_junctions},
    {"D3_legacy_is_exactly_the_flow_set", D3_legacy_is_exactly_the_flow_set},
    {"D4_list_file_format", D4_list_file_format},
    {"D5_repeated_name_after_case_fold", D5_repeated_name_after_case_fold},
};

/** Runs the test named in argv[1], or every test; exits 1 on any failure. */
int main(int argc, char** argv)
{
    const char* want = (argc > 1) ? argv[1] : nullptr;
    int ran = 0;
    for (const test_entry& t : kTests) {
        if (want && std::strcmp(want, t.name) != 0) continue;
        std::fprintf(stderr, "RUN  %s\n", t.name);
        const int before = g_failures;
        t.fn();
        std::fprintf(stderr, "%s %s\n", (g_failures == before) ? "PASS" : "FAIL", t.name);
        ++ran;
    }
    if (want && ran == 0) {
        std::fprintf(stderr, "ERROR: no test named '%s'\n", want);
        return 2;
    }
    std::fprintf(stderr, "%d test(s), %d check(s), %d failure(s)\n", ran, g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
