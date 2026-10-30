/** @file test_coupled_list.cpp
 *  @brief Tier-1 tests for the coupled-node list and the coupled-node rim rule.
 *
 *  No solver run, no cluster, no SWMM call. TRITON and SWMM pair coupled nodes by
 *  POSITION: SWMM walks its FLOW-inflow nodes in node order and consumes the
 *  exchange array, and fills the depth array, in that order. So TRITON's coupled
 *  list must be exactly SWMM's FLOW-inflow set in SWMM's node order, or every
 *  surface cell silently trades water with another node's sewer. These tests
 *  exercise the three free functions in swmm_triton.h that carry that contract:
 *
 *    parse_inp_coupled_list          -- reads the list as SWMM reads its input;
 *    check_coupled_list_against_swmm -- the startup comparison with the engine,
 *                                       which no valid model can make fail, so it
 *                                       is driven here by crafted inputs;
 *    coupled_node_rim                -- SWMM's full depth, or the parsed token
 *                                       where the two agree to a relative 1e-12;
 *    echo_swmm_report_errors         -- item 10's amended stop: which report
 *                                       lines reach stderr after a failed start.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one
 *  node id per property; with no argument every test runs.
 */

// The same include prefix as test_exchange_log_header.cpp, for the reasons that
// file states: swmm_triton.h compiles only after main.cpp's include chain.
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A three-junction model; `inflows` is the whole [INFLOWS] section text and
 *  `extra` is appended after it (a [POLLUTANTS] section, say). */
static std::string model(const std::string& inflows, const std::string& extra = "",
                         const std::string& junctions = "[JUNCTIONS]\n;;Name Elev MaxDepth\n;;---\nJ1 1.0 0.0\nJ2 2.0 1.5\nJ3 3.0 2.0\n")
{
    return "[TITLE]\nmodel\n\n" + junctions + "\n[COORDINATES]\n;;Node X Y\n;;---\n"
           "J1 10.0 20.0\nJ2 30.0 40.0\nJ3 50.0 60.0\n\n" + inflows + "\n" + extra +
           "[DWF]\n;;Node Constituent Baseline\n";
}

static std::vector<std::string> list_of(const std::string& text)
{
    std::istringstream is(text);
    return parse_inp_coupled_list(is).names;
}

static bool names_are(const std::vector<std::string>& got, const std::vector<std::string>& want)
{
    return got == want;
}

static const std::string kStdInflows =
    "[INFLOWS]\n;;Node Constituent TimeSeries\n;;---\nJ1 FLOW ts1\nJ2 FLOW ts2\nJ3 FLOW ts3\n";

// ---------------------------------------------------------------------------
// Parse: the standard input, which must give today's list.
// Class: the healthy-run baseline. A parse that drops or reorders on the input
// every existing model uses fails here. Killed by keeping today's fixed two-line
// header skip on an [INFLOWS] block with no comment lines (P2), not here; here it
// is killed by any reordering of a sorted list.
// ---------------------------------------------------------------------------
static void P1_standard_input_gives_todays_list()
{
    CHECK(names_are(list_of(model(kStdInflows)), {"J1", "J2", "J3"}),
          "two comment lines, every line FLOW, nodes in [JUNCTIONS] order: list J1 J2 J3");
}

// Class: [INFLOWS] layouts that today's two-line skip and first-blank-line stop
// misread. Killed by reinstating either rule.
static void P2_header_layouts()
{
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 FLOW ts1\nJ2 FLOW ts2\nJ3 FLOW ts3\n")), {"J1", "J2", "J3"}),
          "zero comment lines keeps all three nodes");
    CHECK(names_are(list_of(model("[INFLOWS]\n;;c\nJ1 FLOW ts1\nJ2 FLOW ts2\nJ3 FLOW ts3\n")), {"J1", "J2", "J3"}),
          "one comment line keeps all three nodes");
    CHECK(names_are(list_of(model("[INFLOWS]\n;;c\n;;c\nJ1 FLOW ts1\n\nJ2 FLOW ts2\nJ3 FLOW ts3\n")), {"J1", "J2", "J3"}),
          "a blank line inside the section does not end it");
    CHECK(names_are(list_of(model("   [inflows]   ; comment\nJ1 FLOW ts1\nJ2 FLOW ts2\nJ3 FLOW ts3\n")), {"J1", "J2", "J3"}),
          "an indented, lower-case header followed by a comment opens the section");
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 FLOW ts1\n   [DWF]\nJ2 FLOW ts2\n")), {"J1"}),
          "an indented header ends the section");
}

// Class: [INFLOWS] lines in an order other than [JUNCTIONS]. Killed by ordering
// the list by [INFLOWS] position instead of node position.
static void P3_order_follows_junctions()
{
    CHECK(names_are(list_of(model("[INFLOWS]\nJ3 FLOW ts3\nJ1 FLOW ts1\nJ2 FLOW ts2\n")), {"J1", "J2", "J3"}),
          "FLOW lines in reverse order still give the [JUNCTIONS] order");
    CHECK(names_are(list_of(model(kStdInflows, "",
                     "[JUNCTIONS]\n;;c\nJ3 3.0 2.0\n\nJ1 1.0 0.0\nJ2 2.0 1.5\n")), {"J3", "J1", "J2"}),
          "the order is [JUNCTIONS]' own, including a comment and a blank line inside it");
    CHECK(names_are(list_of(model("[INFLOWS]\nOUT1 FLOW t\nJ2 FLOW ts2\n")), {"J2", "OUT1"}),
          "a node absent from [JUNCTIONS] sorts after the junctions");
}

// Class: constituent spellings SWMM reads as FLOW, and lines SWMM does not count
// as a FLOW inflow. Killed by an exact-match "FLOW" test, or by coupling every line.
static void P4_flow_test_is_swmms()
{
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 flow ts1\nJ2 FLOWRATE ts2\nJ3 \"FLOW\" ts3\n")), {"J1", "J2", "J3"}),
          "flow, FLOWRATE and \"FLOW\" each give a FLOW inflow");
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 FLOW ;text\nJ2 FLOW ts2\n")), {"J1", "J2"}),
          "a FLOW token followed by a comment still couples");
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 FLOW ts1\nJ1 TSS tss\nJ2 FLOW ts2\nJ2 FLOW ts2b\n",
                     "[POLLUTANTS]\nTSS MG/L 0.0\n\n")), {"J1", "J2"}),
          "a pollutant line and a repeated FLOW line add no entry");
    CHECK(names_are(list_of(model("[INFLOWS]\nJ1 FLOW ts1\nJ2 flowrate ts2\n",
                     "[POLLUTANTS]\nFLOW MG/L 0.0\n\n")), {"J2"}),
          "with a pollutant named FLOW, a FLOW line is a pollutant line and flowrate is not");
}

// Class: node names that differ only in case. Killed by a case-sensitive compare.
static void P5_names_case_insensitive()
{
    CHECK(names_are(list_of(model("[INFLOWS]\nj2 FLOW ts2\nj1 FLOW ts1\nJ1 FLOW again\n")), {"j1", "j2"}),
          "lower-case names in [INFLOWS] match [JUNCTIONS] and dedupe against upper case");
    std::istringstream is(model(kStdInflows));
    const inp_coupled_list_t r = parse_inp_coupled_list(is);
    CHECK(r.junction_names.size() == 3 && r.junction_max_depth.size() == 3 &&
          r.junction_max_depth[1] == 1.5 && r.coord_names.size() == 3 && r.coord_x[2] == 50.0,
          "junction depths and coordinates are read in file order");
}

// Class: the tokenizer's quote and comment rules. Killed by keeping text after ';'
// inside quotes, or by keeping the quote characters.
static void P6_tokenizer()
{
    const std::vector<std::string> t = swmm_inp_tokens("  \"A B\"\tC ;\"x y\"\r");
    CHECK(t.size() == 2 && t[0] == "A B" && t[1] == "C", "quoted token keeps its space and drops its quotes; ';' ends the line");
    CHECK(swmm_inp_tokens(";;only a comment").empty(), "a comment-only line has no tokens");
}

// ---------------------------------------------------------------------------
// The startup comparison with the engine. It cannot fail on a valid model once the
// type refusal has run, so every failing arm is a crafted input.
// ---------------------------------------------------------------------------
static void C1_valid_list_passes()
{
    const coupled_list_check_t r = check_coupled_list_against_swmm({2, 5, 9}, {1, 1, 1}, 3);
    CHECK(r.ok, "an increasing list of FLOW-inflow nodes of the right length passes");
}

// Class: an index list not strictly increasing. Killed by dropping the order test.
static void C2_out_of_order_refused()
{
    const coupled_list_check_t r = check_coupled_list_against_swmm({2, 9, 5}, {1, 1, 1}, 3);
    CHECK(!r.ok && r.entry == 2 && r.reason.find("out of order") != std::string::npos,
          "entry 2 (index 5 after 9) is refused as out of order");
    const coupled_list_check_t d = check_coupled_list_against_swmm({4, 4}, {1, 1}, 2);
    CHECK(!d.ok && d.entry == 1, "a repeated index is out of order (strictly increasing)");
}

// Class: an entry SWMM gives no FLOW inflow. Killed by dropping the per-entry test.
static void C3_entry_without_flow_refused()
{
    const coupled_list_check_t r = check_coupled_list_against_swmm({1, 3}, {1, 0}, 2);
    CHECK(!r.ok && r.entry == 1 && r.reason.find("no FLOW inflow") != std::string::npos,
          "entry 1 is refused because SWMM gives it no FLOW inflow");
}

// Class: one wrong entry plus one missed node, counts equal. Killed by checking the
// count alone, which this pair satisfies.
static void C4_compensating_pair_refused()
{
    const coupled_list_check_t r = check_coupled_list_against_swmm({1, 3, 7}, {1, 0, 1}, 3);
    CHECK(!r.ok && r.entry == 1, "a non-FLOW entry is caught even when the counts agree");
}

// Class: the list and SWMM's FLOW-inflow set differ in size. Killed by dropping the
// count test (a missed node at the end passes every per-entry test).
static void C5_unequal_counts_refused()
{
    const coupled_list_check_t r = check_coupled_list_against_swmm({1, 3}, {1, 1}, 3);
    CHECK(!r.ok && r.entry == -1 && r.reason.find("couples 2 nodes but SWMM gives 3") != std::string::npos,
          "a list one shorter than SWMM's FLOW-inflow set is refused by the count");
}

// Class: a coupled name SWMM does not know (swmm_getIndex -1, no FLOW inflow) --
// the divergence class the engine-side unknown-name guard exists for. Killed by a
// comparison that skips entries it cannot index. At the first position it is refused
// for having no FLOW inflow; after a valid entry its -1 index is out of order.
static void C6_unknown_name_refused()
{
    const coupled_list_check_t a = check_coupled_list_against_swmm({-1, 3}, {0, 1}, 1);
    CHECK(!a.ok && a.entry == 0 && a.reason.find("no FLOW inflow") != std::string::npos,
          "an unknown name at the first position is refused: no FLOW inflow");
    const coupled_list_check_t b = check_coupled_list_against_swmm({2, -1}, {1, 0}, 1);
    CHECK(!b.ok && b.entry == 1 && b.reason.find("out of order") != std::string::npos,
          "an unknown name after a valid entry is refused: out of order");
}

// ---------------------------------------------------------------------------
// The stop's report echo (item 10, amended).
// Class: report lines that contain the text ERROR but are not SWMM error lines.
// Killed by a substring matcher, which accepts the title and the WARNING line.
// ---------------------------------------------------------------------------
static void E1_error_line_matcher()
{
    CHECK(is_swmm_error_line("  ERROR 209: undefined object NOSUCH at line 119 of [INFLOW] section:"),
          "an indented ERROR line with a code matches");
    CHECK(is_swmm_error_line("ERROR 331: cannot open hot start interface file x."), "an unindented one matches");
    CHECK(!is_swmm_error_line("  ERROR handling check"), "a title with ERROR not followed by a code does not match");
    CHECK(!is_swmm_error_line("  WARNING 02: maximum depth increased for Node ERROR_J1"), "a WARNING naming an object does not");
    CHECK(!is_swmm_error_line("  Continuity Error (%) ....."), "a continuity line does not");
    CHECK(swmm_error_line_names_input_line("  ERROR 209: undefined object X at line 1 of [INFLOW] section:"),
          "a section error names an input line");
    CHECK(swmm_error_line_names_input_line("  ERROR 200: x at line 3 of input file:"), "an input-file error does");
    CHECK(!swmm_error_line_names_input_line("  ERROR 331: cannot open hot start interface file x."), "a file error does not");
}

// Class: reports mixing input errors, file errors, titles and warnings. Killed by
// printing the next line after every error (a blank line after ERROR 331), by
// never printing it (the rejected input line is lost), or by a substring matcher.
static void E2_echo_over_a_report()
{
    const char* path = "test_coupled_list_report.rpt";
    {
        std::FILE* f = std::fopen(path, "w");
        std::fputs("\n  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2\n\n  ERROR handling check\n"
                   "  WARNING 02: maximum depth increased for Node ERROR_J1\n\n"
                   "  ERROR 209: undefined object NOSUCH at line 119 of [INFLOW] section:\n"
                   "  NOSUCH    FLOW   \"\"   FLOW   1.0   1   0\n\n"
                   "  ERROR 331: cannot open hot start interface file h.hsf.\n"
                   "  Analysis begun on:  Mon Oct 19 20:42:33 2026\n", f);
        std::fclose(f);
    }
    std::ostringstream out;
    const int n = echo_swmm_report_errors(path, out);
    const std::string s = out.str();
    CHECK(n == 3, "three lines echoed: ERROR 209, its input line, ERROR 331");
    CHECK(s.find("NOSUCH    FLOW") != std::string::npos, "the rejected input line follows ERROR 209");
    CHECK(s.find("ERROR handling check") == std::string::npos, "the title is not echoed");
    CHECK(s.find("WARNING") == std::string::npos, "the WARNING is not echoed");
    CHECK(s.find("Analysis begun") == std::string::npos, "nothing after ERROR 331 is echoed");
    std::remove(path);
    std::ostringstream miss;
    CHECK(echo_swmm_report_errors("no/such/report.rpt", miss) == 0 &&
          miss.str().find("could not be read") != std::string::npos,
          "an unreadable report prints one line saying so");
}

// ---------------------------------------------------------------------------
// The rim rule.
// Class: tokens SWMM leaves unraised, raised, and moved in the last bit. Killed by
// always keeping the token (the pre-item-11 rim), or always taking SWMM's value
// (which can change a healthy model's last bit).
// ---------------------------------------------------------------------------
static void R1_rim_rule()
{
    CHECK(coupled_node_rim(0.0, 1.2) == 1.2, "a recorded 0 under a 1.2 m crown takes SWMM's 1.2");
    CHECK(coupled_node_rim(2.0, 2.0) == 2.0, "an equal token is kept");
    const double moved = 1.5000000000000002;
    CHECK(coupled_node_rim(1.5, moved) == 1.5, "a token SWMM moved by one ulp is kept");
    CHECK(coupled_node_rim(1.5, 1.6) == 1.6, "a token below SWMM's depth takes SWMM's");
    CHECK(coupled_node_rim(0.0, 0.0) == 0.0, "a pump-only junction recorded 0 stays 0");
}

// ---------------------------------------------------------------------------
// US-unit models (item 12).
// Class: SWMM's six flow units. SWMM's length unit follows its flow unit, so the
// three US flow units (CFS, GPM, MGD) mean feet. Killed by recognising CFS only
// (the old FLOW UNITS check's reach), or by a range that admits CMS.
// Second correct implementation: a switch over the six swmm_FlowUnitsProperty values.
// ---------------------------------------------------------------------------
static void U1_us_flow_units_mean_feet()
{
    CHECK(swmm_flow_units_are_us(swmm_CFS) && swmm_flow_units_are_us(swmm_GPM) && swmm_flow_units_are_us(swmm_MGD),
          "CFS, GPM and MGD are US units");
    CHECK(!swmm_flow_units_are_us(swmm_CMS) && !swmm_flow_units_are_us(swmm_LPS) && !swmm_flow_units_are_us(swmm_MLD),
          "CMS, LPS and MLD are SI units");
}

// Class: engine lengths in a US model and in an SI model. A US length is multiplied by
// 0.3048, the factor SWMM's getters apply to every length of an SI model, so the result
// is bit for bit what the SI twin's getter returns; an SI length is unchanged. Killed by
// no conversion, by any other feet-to-metres factor, or by converting SI lengths too.
// Second correct implementation: multiplying by SWMM's own MperFT macro.
static void U2_engine_length_in_metres()
{
    const double ft = 1.6404199475065615;   // 0.5 m written in feet at %.17g
    CHECK(swmm_engine_length_m(ft, true) == ft * 0.3048, "a US length is its SI twin's getter value, ft * 0.3048");
    CHECK(std::fabs(swmm_engine_length_m(ft, true) - 0.5) < 1e-15, "0.5 m written in feet comes back as 0.5 m");
    CHECK(swmm_engine_length_m(0.0, true) == 0.0, "zero stays zero");
    CHECK(swmm_engine_length_m(1.5, false) == 1.5, "an SI length is unchanged");
}

// Class: a coupled node's rim in a US model, where the [JUNCTIONS] token is in feet.
// Killed by keeping the token when it equals SWMM's value (the SI rule applied to a US
// model gives a rim in feet), or by converting only one branch. Second correct
// implementation: convert the engine depth first, then take it unconditionally in US.
static void U3_us_rim_never_keeps_the_token()
{
    const double ft = 1.6404199475065615;   // 0.5 m in feet
    CHECK(coupled_node_rim_m(ft, ft, true) == ft * 0.3048, "a US token equal to SWMM's depth is not kept: the rim is in metres");
    const double crown_ft = 1.0 / 0.3048;   // a 1.0 m crown in feet
    CHECK(coupled_node_rim_m(0.0, crown_ft, true) == crown_ft * 0.3048, "a US node raised to its crown has the crown in metres");
    CHECK(coupled_node_rim_m(1.5, 1.5000000000000002, false) == 1.5, "the SI token-keep rule is unchanged");
    CHECK(coupled_node_rim_m(0.0, 1.2, false) == 1.2, "an SI recorded 0 still takes SWMM's depth");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"P1_standard_input_gives_todays_list", P1_standard_input_gives_todays_list},
    {"P2_header_layouts", P2_header_layouts},
    {"P3_order_follows_junctions", P3_order_follows_junctions},
    {"P4_flow_test_is_swmms", P4_flow_test_is_swmms},
    {"P5_names_case_insensitive", P5_names_case_insensitive},
    {"P6_tokenizer", P6_tokenizer},
    {"C1_valid_list_passes", C1_valid_list_passes},
    {"C2_out_of_order_refused", C2_out_of_order_refused},
    {"C3_entry_without_flow_refused", C3_entry_without_flow_refused},
    {"C4_compensating_pair_refused", C4_compensating_pair_refused},
    {"C5_unequal_counts_refused", C5_unequal_counts_refused},
    {"C6_unknown_name_refused", C6_unknown_name_refused},
    {"E1_error_line_matcher", E1_error_line_matcher},
    {"E2_echo_over_a_report", E2_echo_over_a_report},
    {"R1_rim_rule", R1_rim_rule},
    {"U1_us_flow_units_mean_feet", U1_us_flow_units_mean_feet},
    {"U2_engine_length_in_metres", U2_engine_length_in_metres},
    {"U3_us_rim_never_keeps_the_token", U3_us_rim_never_keeps_the_token},
};

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
