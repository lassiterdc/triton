/** @file test_dem_rim_exemption.cpp
 *  @brief Source-structure tests for the per-node exemption of the DEM-rim loop
 *         (cfg swmm_dem_rim_exempt_nodes), chunk 1m.1.
 *
 *  Why a source-structure test. The exemption is decided inline in the DEM-rim block
 *  of src/triton.h, which no unit test includes (it needs Kokkos, MPI and a DEM), so
 *  the properties below are read from the source text: which statements sit inside
 *  the rim loop, in which order, and which refusals exit. The behaviour on a real
 *  model (the listed node's column the only one that differs, the rim line counting
 *  one node fewer, a node whose cell is at or below its invert running) is case N15
 *  of the neutrality suite, built by chunk 3d.1; this file does not run the solver.
 *
 *  Each property runs twice: on the real source, where it must hold, and on a copy
 *  the test breaks on purpose (the control), where it must FAIL. A property that
 *  holds on the broken copy as well reads nothing and is reported as a failure.
 *
 *  Usage: test_dem_rim_exemption <path to src/triton.h> <path to src/config_utils.h> [test name]
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

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

static std::string g_triton;
static std::string g_config;

/** The whole file as bytes, or an empty string if it cannot be read. */
static std::string slurp(const std::string& path)
{
    std::ifstream f(path.c_str(), std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/** Removes // and slash-star comments so a commented-out statement cannot satisfy a property. */
static std::string strip_comments(const std::string& s)
{
    std::string out;
    size_t i = 0;
    while (i < s.size())
    {
        if (s.compare(i, 2, "//") == 0) { while (i < s.size() && s[i] != '\n') ++i; continue; }
        if (s.compare(i, 2, "/*") == 0)
        {
            size_t j = s.find("*/", i + 2);
            i = (j == std::string::npos) ? s.size() : j + 2;
            continue;
        }
        out += s[i++];
    }
    return out;
}

/** The text of the key's block: from the declaration of rim_exempt to the end of the rim
 *  loop's summary line. Empty when rim_exempt is not declared. */
static std::string exemption_region(const std::string& src)
{
    const size_t b = src.find("std::set<std::string> rim_exempt");
    if (b == std::string::npos) return std::string();
    const size_t e = src.find("SWMM coupled-node rims set from the DEM", b);
    if (e == std::string::npos) return std::string();
    return src.substr(b, e - b);
}

/** The rim loop's text: from its opening `for` to the summary line. */
static std::string rim_loop(const std::string& region)
{
    const size_t b = region.find("for (int k = 0; k < swmm_model.num_of_swmm_links; k++)");
    return b == std::string::npos ? std::string() : region.substr(b);
}

/** Replaces the first occurrence of `from` with `to`; returns the text unchanged if absent. */
static std::string replace_first(std::string s, const std::string& from, const std::string& to)
{
    const size_t a = s.find(from);
    if (a != std::string::npos) s.replace(a, from.size(), to);
    return s;
}

/** Position of the first occurrence of needle in hay, or npos. */
static size_t pos(const std::string& hay, const std::string& needle)
{
    return hay.find(needle);
}

/** Property: the key is a std::string field read with the default "" (absent = no exemption). */
static bool prop_key_declared_default_empty(const std::string& cfg)
{
    const std::string c = strip_comments(cfg);
    const std::regex field(R"(std::string\s+swmm_dem_rim_exempt_nodes\s*;)");
    const std::regex reader(R"re(arglist\.swmm_dem_rim_exempt_nodes\s*=\s*argsd\(\s*"swmm_dem_rim_exempt_nodes"\s*,\s*argmap\s*,\s*""\s*\))re");
    return std::regex_search(c, field) && std::regex_search(c, reader);
}

/** Position of the first membership test on rim_exempt (count, find or contains), or npos.
 *  A second correct implementation may spell the test any of these ways. */
static size_t membership_pos(const std::string& hay, size_t from = 0)
{
    static const std::regex member(R"(rim_exempt\.(count|find|contains)\()");
    std::smatch m;
    const std::string tail = hay.substr(from);
    if (!std::regex_search(tail, m, member)) return std::string::npos;
    return from + static_cast<size_t>(m.position(0));
}

/** Property: a listed node is skipped by `continue` inside the rim loop, before the loop's
 *  own refusal of a cell at or below the invert, and the skip tests membership in rim_exempt. */
static bool prop_skip_precedes_the_refusal(const std::string& triton)
{
    const std::string loop = rim_loop(exemption_region(strip_comments(triton)));
    if (loop.empty()) return false;
    const size_t skip = membership_pos(loop);
    const size_t cont = pos(loop, "continue;");
    const size_t refuse = pos(loop, "z_cell <= invert");
    return skip != std::string::npos && cont != std::string::npos && refuse != std::string::npos
        && skip < cont && cont < refuse;
}

/** Property: the set is filled only inside the branch entered when the key is nonempty, and the
 *  loop's skip is guarded by !rim_exempt.empty(), so an absent key reaches no new statement. */
static bool prop_absent_key_reaches_nothing(const std::string& triton)
{
    const std::string region = exemption_region(strip_comments(triton));
    if (region.empty()) return false;
    const size_t gate = pos(region, "if (!arglist.swmm_dem_rim_exempt_nodes.empty())");
    const size_t insert = pos(region, "rim_exempt.insert(");
    const size_t loop_guard = pos(region, "if (!rim_exempt.empty() &&");
    return gate != std::string::npos && insert != std::string::npos && loop_guard != std::string::npos
        && gate < insert && insert < loop_guard
        && region.find("rim_exempt.insert(") == region.rfind("rim_exempt.insert(");
}

/** Count of non-overlapping occurrences of needle in hay. */
static int count_of(const std::string& hay, const std::string& needle)
{
    int n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + needle.size())) ++n;
    return n;
}

/** Property: the four refusals each end in exit(EXIT_FAILURE), and the one for the key with the
 *  DEM-rim route off names the key and tests both swmm_node_rim_from_dem and an empty inp. */
static bool prop_refusals_exit_and_name_the_key(const std::string& triton)
{
    const std::string region = exemption_region(strip_comments(triton));
    const size_t gate = pos(region, "if (!arglist.swmm_dem_rim_exempt_nodes.empty())");
    const size_t gate_end = pos(region, "if (!arglist.inp_filename.empty() && arglist.swmm_node_rim_from_dem)");
    if (gate == std::string::npos || gate_end == std::string::npos || gate_end < gate) return false;
    const std::string block = region.substr(gate, gate_end - gate);
    if (count_of(block, "exit(EXIT_FAILURE)") != 4) return false;
    const size_t off = pos(block, "!arglist.swmm_node_rim_from_dem");
    const size_t named = pos(block, "swmm_dem_rim_exempt_nodes lists");
    return off != std::string::npos && named != std::string::npos
        && block.find("arglist.inp_filename.empty()") != std::string::npos
        && block.find("!ex_names.empty()") != std::string::npos;
}

/** Property: the exemption block holds no MPI call, so each rank decides its refusals alone
 *  and every rank exits together (the file is the same on every rank). */
static bool prop_no_collective_in_the_key_block(const std::string& triton)
{
    const std::string region = exemption_region(strip_comments(triton));
    const size_t gate = pos(region, "if (!arglist.swmm_dem_rim_exempt_nodes.empty())");
    const size_t gate_end = pos(region, "if (!arglist.inp_filename.empty() && arglist.swmm_node_rim_from_dem)");
    if (gate == std::string::npos || gate_end == std::string::npos || gate_end < gate) return false;
    return region.substr(gate, gate_end - gate).find("MPI_") == std::string::npos;
}

/** Property: the rank-0 rim line counts the coupled nodes less the exempt ones. */
static bool prop_summary_counts_one_fewer(const std::string& triton)
{
    const std::string c = strip_comments(triton);
    const size_t at = c.find("SWMM coupled-node rims set from the DEM");
    if (at == std::string::npos) return false;
    const std::string tail = c.substr(at, 400);
    return tail.find("global_num_of_swmm_links - static_cast<int>(rim_exempt.size())") != std::string::npos;
}

/** Breaks a copy of the source for the control of one property. */
/** Control for E1: the exempt node is no longer skipped (the continue becomes a no-op). */
static std::string break_skip(std::string s)
{
    // The exempt node is no longer skipped: the continue becomes a no-op statement.
    const size_t a = membership_pos(s);
    if (a == std::string::npos) return s;
    const size_t c = s.find("continue;", a);
    if (c != std::string::npos) s.replace(c, 9, "(void)0;");
    return s;
}
/** Control for E1: the skip no longer comes before the loop's own refusal of a cell at or below the invert. */
static std::string break_refusal_order(std::string s)
{
    // The skip no longer comes before the refusal of a cell at or below the invert.
    const size_t a = membership_pos(s);
    const size_t r = s.find("z_cell <= invert", a == std::string::npos ? 0 : a);
    if (a == std::string::npos || r == std::string::npos) return s;
    const size_t c = s.find("continue;", a);
    if (c == std::string::npos) return s;
    s.replace(c, 9, "(void)0;");
    return s;
}
/** Control for E2: the key block is entered whether or not the key is set. */
static std::string break_guard(std::string s)
{
    return replace_first(s, "if (!arglist.swmm_dem_rim_exempt_nodes.empty())", "if (true)");
}
/** Control for R1: the refusal of the key with the DEM-rim route off no longer exits. */
static std::string break_exit(std::string s)
{
    const size_t a = s.find("swmm_dem_rim_exempt_nodes lists");
    const size_t e = a == std::string::npos ? a : s.find("exit(EXIT_FAILURE)", a);
    if (e != std::string::npos) s.replace(e, 18, "(void)0");
    return s;
}
/** Control for M1: a collective is added to the key block. */
static std::string break_mpi(std::string s)
{
    const size_t a = s.find("std::set<std::string> coupled_names;");
    if (a != std::string::npos) s.insert(a, "MPI_Barrier(ENSIFY_COMM_WORLD);\n");
    return s;
}
/** Control for L1: the rim line counts every coupled node, exempt ones included. */
static std::string break_summary(std::string s)
{
    return replace_first(s, "global_num_of_swmm_links - static_cast<int>(rim_exempt.size())",
                         "global_num_of_swmm_links");
}

/** Runs one property on the real triton.h (must hold) and on a copy the breaker damages (must fail). */
static void run_triton_prop(const char* name, bool (*prop)(const std::string&), std::string (*breaker)(std::string))
{
    const std::string src = slurp(g_triton);
    CHECK(!src.empty(), "triton.h readable");
    std::fprintf(stderr, "%s\n", name);
    CHECK(prop(src), "the property holds on the real source");
    const std::string broken = breaker(src);
    CHECK(broken != src, "the control changed the source (the matcher found its anchor)");
    CHECK(!prop(broken), "the property FAILS on the deliberately broken copy (it reads something)");
}

/** K1: the key is a std::string read with the default "" (class: the cfg reader). Wrong implementation caught: a nonempty default. */
static void K1_key_is_read_with_an_empty_default()
{
    const std::string cfg = slurp(g_config);
    CHECK(!cfg.empty(), "config_utils.h readable");
    std::fprintf(stderr, "K1_key_is_read_with_an_empty_default\n");
    CHECK(prop_key_declared_default_empty(cfg), "field declared and read with default \"\"");
    const std::string broken = replace_first(cfg, "\"swmm_dem_rim_exempt_nodes\", argmap, \"\"",
                                             "\"swmm_dem_rim_exempt_nodes\", argmap, \"x\"");
    CHECK(broken != cfg, "the control changed the source");
    CHECK(!prop_key_declared_default_empty(broken), "a nonempty default is caught");
}
/** E1: a listed node is skipped before the refusal of a cell at or below the invert. Wrong implementations caught: no skip; skip after the refusal. */
static void E1_a_listed_node_is_skipped_before_the_refusal()
{
    run_triton_prop("E1_a_listed_node_is_skipped_before_the_refusal", prop_skip_precedes_the_refusal, break_skip);
    const std::string src = slurp(g_triton);
    CHECK(!prop_skip_precedes_the_refusal(break_refusal_order(src)), "a skip placed after the refusal is caught");
}
/** E2: an absent key reaches no new statement. Wrong implementation caught: the key block entered unconditionally. */
static void E2_an_absent_key_reaches_no_new_statement()
{
    run_triton_prop("E2_an_absent_key_reaches_no_new_statement", prop_absent_key_reaches_nothing, break_guard);
}
/** R1: the four refusals exit and the inert-key refusal names the key. Wrong implementation caught: that refusal logs and continues. */
static void R1_every_refusal_exits_and_the_inert_key_is_named()
{
    run_triton_prop("R1_every_refusal_exits_and_the_inert_key_is_named", prop_refusals_exit_and_name_the_key, break_exit);
}
/** M1: no MPI call in the key block, so every rank decides alone. Wrong implementation caught: a barrier in the block. */
static void M1_no_collective_in_the_key_block()
{
    run_triton_prop("M1_no_collective_in_the_key_block", prop_no_collective_in_the_key_block, break_mpi);
}
/** L1: the rank-0 rim line counts the coupled nodes less the exempt ones. Wrong implementation caught: the unchanged count. */
static void L1_the_rim_line_counts_one_node_fewer()
{
    run_triton_prop("L1_the_rim_line_counts_one_node_fewer", prop_summary_counts_one_fewer, break_summary);
}

/** Runs every test, or the one named by argv[3]; exits nonzero on any failed check. */
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <src/triton.h> <src/config_utils.h> [test name]\n", argv[0]);
        return 2;
    }
    g_triton = argv[1];
    g_config = argv[2];
    const char* which = argc > 3 ? argv[3] : nullptr;
    struct { const char* name; void (*fn)(); } tests[] = {
        {"K1_key_is_read_with_an_empty_default", K1_key_is_read_with_an_empty_default},
        {"E1_a_listed_node_is_skipped_before_the_refusal", E1_a_listed_node_is_skipped_before_the_refusal},
        {"E2_an_absent_key_reaches_no_new_statement", E2_an_absent_key_reaches_no_new_statement},
        {"R1_every_refusal_exits_and_the_inert_key_is_named", R1_every_refusal_exits_and_the_inert_key_is_named},
        {"M1_no_collective_in_the_key_block", M1_no_collective_in_the_key_block},
        {"L1_the_rim_line_counts_one_node_fewer", L1_the_rim_line_counts_one_node_fewer},
    };
    bool ran = false;
    for (auto& t : tests)
    {
        if (which && std::strcmp(which, t.name) != 0) continue;
        ran = true;
        t.fn();
    }
    if (!ran) { std::fprintf(stderr, "no test named %s\n", which); return 2; }
    std::fprintf(stderr, "%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
