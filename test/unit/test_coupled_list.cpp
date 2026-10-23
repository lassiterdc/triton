/** @file test_coupled_list.cpp
 *  @brief Tier-1 tests for the stop on a failed SWMM startup: which report lines
 *         reach stderr (echo_swmm_report_errors and its two line matchers).
 *
 *  No solver run, no cluster, no SWMM call. Each test function is selectable by
 *  name from argv[1] so CTest registers one node id per property; with no
 *  argument every test runs.
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

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"E1_error_line_matcher", E1_error_line_matcher},
    {"E2_echo_over_a_report", E2_echo_over_a_report},
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
