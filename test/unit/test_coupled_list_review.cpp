/** @file test_coupled_list_review.cpp
 *  @brief Reviewer test for the coupled-node parse: SWMM's section headers.
 *
 *  TRITON pairs its coupled nodes with SWMM's FLOW-inflow nodes by POSITION, so
 *  parse_inp_coupled_list must find the [JUNCTIONS], [COORDINATES], [INFLOWS] and
 *  [POLLUTANTS] sections exactly where SWMM finds them. SWMM does not compare a
 *  header with the full section name. It asks whether the header token BEGINS,
 *  case-insensitively, with a short section keyword -- "[JUNC", "[COORDINATE",
 *  "[INFLOW", "[POLLUT" (SectWords in external/swmm/src/solver/keywords.c, the
 *  ws_* strings in text.h, compared by findmatch and match in input.c). So SWMM
 *  reads a file whose header is written "[INFLOW]" or "[JUNCTION]" as those
 *  sections, and a parse that requires the full plural name reads them as some
 *  other section.
 *
 *  Same build arrangement and include prefix as test_coupled_list.cpp.
 */

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

static int g_failures = 0;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

/** A three-junction model whose four section headers are passed in, so each
 *  header can be written in a spelling SWMM accepts. Junctions are listed J3, J1,
 *  J2, so the coupled list's order shows whether [JUNCTIONS] was read at all. */
static inp_coupled_list_t parse_with(const char* junc, const char* coord,
                                     const char* infl, const char* poll)
{
    std::string m = "[TITLE]\nmodel\n\n";
    m += std::string(junc)  + "\nJ3 3.0 2.0\nJ1 1.0 0.0\nJ2 2.0 1.5\n\n";
    m += std::string(coord) + "\nJ1 10.0 20.0\nJ2 30.0 40.0\nJ3 50.0 60.0\n\n";
    m += std::string(poll)  + "\nFLOW MG/L 0.0\n\n";
    m += std::string(infl)  + "\nJ1 FLOW ts1\nJ2 flowrate ts2\nJ3 flowrate ts3\n\n";
    m += "[DWF]\n";
    std::istringstream is(m);
    return parse_inp_coupled_list(is);
}

/** Property: every header SWMM's own section match maps to [JUNCTIONS],
 *  [COORDINATES], [INFLOWS] or [POLLUTANTS] opens that section in the parse.
 *
 *  Class: header spellings SWMM accepts for those four sections. The domain is
 *  closed: it is the four keyword prefixes, each followed by any text. Positioned
 *  here at each keyword's shortest accepted form, the keyword plus "]", and at the
 *  singular names a person writes ("[JUNCTION]", "[INFLOW]", "[COORDINATE]",
 *  "[POLLUTANT]").
 *
 *  Why it is required: a section the parse misses changes the run silently or
 *  stops it. A missed [INFLOWS] leaves TRITON with no coupled node while SWMM
 *  has several, so init_swmm's count check stops a model SWMM accepts. A missed
 *  [POLLUTANTS] couples a "flowrate" line for a pollutant SWMM gives no FLOW
 *  inflow. A missed [JUNCTIONS] loses the node order that sorts the list.
 *
 *  Expected list in every arm: the pollutant named FLOW makes J1's "FLOW" line a
 *  pollutant line, so the coupled nodes are J2 and J3 ("flowrate" is a FLOW
 *  inflow), in [JUNCTIONS] order: J3, J2.
 *
 *  Killed by: matching each header against the full plural name ("[INFLOWS]",
 *  "[JUNCTIONS]", ...), which is how the parse matches today. Passes under any
 *  match that tests SWMM's keyword prefixes -- the four prefixes alone, or a
 *  loop over SWMM's whole SectWords list in SWMM's order.
 */
static void RV1_section_headers_by_swmm_keyword_prefix()
{
    struct arm { const char* j; const char* c; const char* i; const char* p; const char* label; };
    const arm arms[] = {
        {"[JUNCTIONS]", "[COORDINATES]", "[INFLOWS]", "[POLLUTANTS]", "plural names (control)"},
        {"[JUNCTION]",  "[COORDINATE]",  "[INFLOW]",  "[POLLUTANT]",  "singular names"},
        {"[JUNC]",      "[COORDINATE]",  "[INFLOW]",  "[POLLUT]",     "SWMM's keywords plus ]"},
        {"[junction]",  "[coordinate]",  "[inflow]",  "[pollutant]",  "singular, lower case"},
    };
    for (const arm& a : arms) {
        const inp_coupled_list_t r = parse_with(a.j, a.c, a.i, a.p);
        const bool ok = r.names == std::vector<std::string>{"J3", "J2"}
                        && r.junction_names.size() == 3 && r.coord_names.size() == 3;
        if (!ok) {
            std::fprintf(stderr, "  arm '%s': names=", a.label);
            for (const auto& n : r.names) std::fprintf(stderr, "%s ", n.c_str());
            std::fprintf(stderr, " junctions=%zu coords=%zu\n", r.junction_names.size(), r.coord_names.size());
        }
        CHECK(ok, a.label);
    }
}

/** Runs the one test; exits 1 when any arm fails, so CTest reads the exit code. */
int main(int argc, char** argv)
{
    (void)argc; (void)argv;
    std::fprintf(stderr, "RUN  RV1_section_headers_by_swmm_keyword_prefix\n");
    RV1_section_headers_by_swmm_keyword_prefix();
    std::fprintf(stderr, "%s RV1_section_headers_by_swmm_keyword_prefix\n", g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
