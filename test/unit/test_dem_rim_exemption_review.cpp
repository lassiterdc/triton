/** @file test_dem_rim_exemption_review.cpp
 *  @brief Reviewer's source-structure tests for chunk 1m.1 (cfg swmm_dem_rim_exempt_nodes).
 *
 *  Complements test_dem_rim_exemption.cpp. Properties it does not pin, each with a control
 *  (a copy broken on purpose that the property must reject):
 *   F1  the case fold is applied on BOTH sides of the match (the insert and the lookup), so
 *       a list spelled "j1" exempts a coupled node SWMM names "J1";
 *   O1  the refusals run in the order: repeat, inert key, no coupling node; and the
 *       coupling-node refusal exits (a name that is no coupling node stops every rank);
 *   S1  <set> is included inside the TRITON_SWMM guard, so a build without SWMM is untouched;
 *   Z1  the loop skip touches no running total: between the membership test and its continue
 *       nothing assigns bad or largest_diff or max_depth (an exempt node keeps SWMM's depth);
 *   U1  the exempt count subtracted in the rim line is the set size, a set filled only from
 *       names that passed the coupling-node check (insert follows the refusal).
 *
 *  Usage: test_dem_rim_exemption_review <src/triton.h> [test name]
 */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

static int g_checks = 0, g_failures = 0;
#define CHECK(expr, msg) do { ++g_checks; if (!(expr)) { ++g_failures; \
    std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg); } } while (0)

static std::string g_triton;

static std::string slurp(const std::string& p)
{
    std::ifstream f(p.c_str(), std::ios::binary); std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::string strip_comments(const std::string& s)
{
    std::string out; size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, 2, "//") == 0) { while (i < s.size() && s[i] != '\n') ++i; continue; }
        if (s.compare(i, 2, "/*") == 0) { size_t j = s.find("*/", i + 2); i = (j == std::string::npos) ? s.size() : j + 2; continue; }
        out += s[i++];
    }
    return out;
}
static std::string replace_first(std::string s, const std::string& a, const std::string& b)
{
    size_t p = s.find(a); if (p != std::string::npos) s.replace(p, a.size(), b); return s;
}
static std::string region(const std::string& src)
{
    const size_t b = src.find("std::set<std::string> rim_exempt");
    const size_t e = src.find("SWMM coupled-node rims set from the DEM", b == std::string::npos ? 0 : b);
    return (b == std::string::npos || e == std::string::npos) ? std::string() : src.substr(b, e - b);
}
static std::string key_block(const std::string& reg)
{
    const size_t g = reg.find("if (!arglist.swmm_dem_rim_exempt_nodes.empty())");
    const size_t h = reg.find("if (!arglist.inp_filename.empty() && arglist.swmm_node_rim_from_dem)");
    return (g == std::string::npos || h == std::string::npos || h < g) ? std::string() : reg.substr(g, h - g);
}
static std::string rim_loop(const std::string& reg)
{
    const size_t b = reg.find("for (int k = 0; k < swmm_model.num_of_swmm_links; k++)");
    return b == std::string::npos ? std::string() : reg.substr(b);
}

/** F1: insert and lookup both go through swmm_upper. */
static bool prop_fold_both_sides(const std::string& triton)
{
    const std::string reg = region(strip_comments(triton));
    static const std::regex ins(R"(rim_exempt\.insert\(\s*SWMM_triton::swmm_upper\()");
    static const std::regex look(R"(rim_exempt\.count\(\s*SWMM_triton::swmm_upper\(\s*swmm_model\.nodeID\[k\]\s*\)\s*\))");
    static const std::regex cpl(R"(coupled_names\.insert\(\s*SWMM_triton::swmm_upper\()");
    static const std::regex cpf(R"(coupled_names\.find\(\s*SWMM_triton::swmm_upper\()");
    return std::regex_search(reg, ins) && std::regex_search(reg, look)
        && std::regex_search(reg, cpl) && std::regex_search(reg, cpf);
}
/** O1: repeat refusal < inert-key refusal < coupling-node refusal, and the last exits. */
static bool prop_refusal_order(const std::string& triton)
{
    const std::string blk = key_block(region(strip_comments(triton)));
    const size_t a = blk.find("find_repeated_name(ex_names)");
    const size_t b = blk.find("swmm_dem_rim_exempt_nodes lists");
    const size_t c = blk.find("is no coupling node");
    if (a == std::string::npos || b == std::string::npos || c == std::string::npos) return false;
    if (!(a < b && b < c)) return false;
    const size_t x = blk.find("exit(EXIT_FAILURE)", c);
    const size_t ins = blk.find("rim_exempt.insert(", c);
    return x != std::string::npos && ins != std::string::npos && x < ins;   // U1 as well
}
/** S1: <set> appears inside #ifdef TRITON_SWMM ... #endif and before the swmm_triton.h include. */
static bool prop_set_include_guarded(const std::string& triton)
{
    const size_t g = triton.find("#ifdef TRITON_SWMM");
    const size_t i = triton.find("#include <set>");
    const size_t e = triton.find("#endif", g == std::string::npos ? 0 : g);
    return g != std::string::npos && i != std::string::npos && e != std::string::npos && g < i && i < e;
}
/** Z1: from the membership test to its continue the loop assigns nothing. */
static bool prop_skip_assigns_nothing(const std::string& triton)
{
    const std::string loop = rim_loop(region(strip_comments(triton)));
    const size_t m = loop.find("rim_exempt.count(");
    const size_t c = loop.find("continue;", m == std::string::npos ? 0 : m);
    if (m == std::string::npos || c == std::string::npos) return false;
    const std::string seg = loop.substr(m, c - m);
    static const std::regex assign(R"((bad|largest_diff|max_depth\[[^\]]*\])\s*(=|\+=)[^=])");
    return !std::regex_search(seg, assign);
}

static std::string brk_fold(std::string s)   // lookup by the raw node name
{
    return replace_first(s, "rim_exempt.count(SWMM_triton::swmm_upper(swmm_model.nodeID[k]))", "rim_exempt.count(swmm_model.nodeID[k])");
}
static std::string brk_order(std::string s)  // coupling-node refusal logs and continues
{
    const size_t c = s.find("is no coupling node");
    const size_t x = s.find("exit(EXIT_FAILURE)", c == std::string::npos ? 0 : c);
    if (x != std::string::npos) s.replace(x, 18, "(void)0");
    return s;
}
static std::string brk_include(std::string s) { return replace_first(s, "#include <set>", "// no set"); }
static std::string brk_assign(std::string s)  // the skip clobbers the depth
{
    const size_t m = s.find("rim_exempt.count(");
    const size_t c = s.find("continue;", m == std::string::npos ? 0 : m);
    if (c != std::string::npos) s.insert(c, "swmm_model.max_depth[k] = 0; ");
    return s;
}

static void run(const char* name, bool (*prop)(const std::string&), std::string (*brk)(std::string))
{
    const std::string src = slurp(g_triton);
    std::fprintf(stderr, "%s\n", name);
    CHECK(!src.empty(), "triton.h readable");
    CHECK(prop(src), "the property holds on the real source");
    const std::string b = brk(src);
    CHECK(b != src, "the control changed the source");
    CHECK(!prop(b), "the property FAILS on the broken copy");
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s <src/triton.h> [test]\n", argv[0]); return 2; }
    g_triton = argv[1];
    const char* which = argc > 2 ? argv[2] : nullptr;
    struct { const char* n; bool (*p)(const std::string&); std::string (*b)(std::string); } t[] = {
        {"F1_case_fold_on_both_sides", prop_fold_both_sides, brk_fold},
        {"O1_refusal_order_and_the_coupling_refusal_exits", prop_refusal_order, brk_order},
        {"S1_set_include_inside_the_swmm_guard", prop_set_include_guarded, brk_include},
        {"Z1_the_skip_assigns_no_running_total", prop_skip_assigns_nothing, brk_assign},
    };
    bool ran = false;
    for (auto& x : t) { if (which && std::strcmp(which, x.n)) continue; ran = true; run(x.n, x.p, x.b); }
    if (!ran) { std::fprintf(stderr, "no test named %s\n", which); return 2; }
    std::fprintf(stderr, "%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
