/** @file test_coupling_interval_review.cpp
 *  @brief Reviewer test for chunk 1d.1: which TRITON steps drain the SWMM coupling
 *         interval, read from the source of src/triton.h.
 *
 *  Why this file exists. The nine UNIT-COUPLING-INTERVAL nodes drive the pure
 *  bookkeeping struct coupling_interval_t and hand it `drain` as an argument. The
 *  decision of WHICH step is a drain (the step that ends a print interval, and,
 *  by erratum E-3, the last step of the run) is made inline in the T3 block of
 *  triton.h, which no unit test includes. Removing the end-of-run condition from
 *  triton.h therefore leaves all nine nodes green, and the only other cover is
 *  the neutrality suite on Rivanna, which is built later. This test reads the T3
 *  block of triton.h and asks the property directly: does the boolean that
 *  decides whether SWMM steps depend on the run's duration, and (in a default
 *  build) on the print interval?
 *
 *  It is a source-structure test and says so. It does not run the solver; it
 *  resolves, per compile-time arm of TRITON_DIAG_TEST_NO_DRAIN, the identifiers
 *  that the argument of step_due( ... ) is computed from, and requires the
 *  duration and print-interval quantities among them. It is written to survive
 *  a rename of the local booleans and the inlining of an expression, but not the
 *  moving of the predicate into a function that is handed `arglist` whole.
 *
 *  Not a unit test of arithmetic: the arithmetic of the drain predicate
 *  (`simtime + global_dt >= ...`) is checked by the neutrality case N12, not here.
 *
 *  Usage: test_coupling_interval_review <path to src/triton.h> [test name]
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

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

static std::string g_triton_path;

/** Everything the T3 block's drain decision is built from, per compile-time arm. */
struct arm_info
{
    bool found = false;                         ///< the arm exists in the T3 block
    std::string step_due_arg;                   ///< text of the argument of step_due( ... )
    std::string settle_last_arg;                ///< text of the last argument of settle( ... )
    std::set<std::string> leaves;               ///< identifiers the step_due argument resolves to
};

/** Removes // and slash-star comments so a commented-out drain cannot satisfy a property. */
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

/** Leading and trailing whitespace removed. */
static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

/** Arguments of the first call `name(` in text, split at top-level commas. */
static std::vector<std::string> call_args(const std::string& text, const std::string& name)
{
    std::vector<std::string> args;
    size_t p = text.find(name + "(");
    if (p == std::string::npos) return args;
    size_t i = p + name.size() + 1;
    int depth = 1;
    std::string cur;
    for (; i < text.size() && depth > 0; ++i)
    {
        const char c = text[i];
        if (c == '(') { ++depth; cur += c; }
        else if (c == ')') { if (--depth > 0) cur += c; }
        else if (c == ',' && depth == 1) { args.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    args.push_back(trim(cur));
    return args;
}

/** Identifier tokens of an expression; a member chain `a.b.c` is one token. */
static std::vector<std::string> tokens_of(const std::string& expr)
{
    static const std::regex id("[A-Za-z_][A-Za-z_0-9]*(?:\\s*\\.\\s*[A-Za-z_][A-Za-z_0-9]*)*");
    std::vector<std::string> out;
    for (std::sregex_iterator it(expr.begin(), expr.end(), id), end; it != end; ++it)
    {
        std::string t = it->str();
        t.erase(std::remove_if(t.begin(), t.end(),
                               [](char c) { return c == ' ' || c == '\t'; }),
                t.end());
        out.push_back(t);
    }
    return out;
}

/** Adds to `leaves` every identifier `expr` depends on, following each local boolean
 *  that `defs` defines to its own initializer (once per name, so a cycle ends). */
static void resolve(const std::string& expr, const std::map<std::string, std::string>& defs,
                    std::set<std::string>& visiting, std::set<std::string>& leaves)
{
    for (const std::string& t : tokens_of(expr))
    {
        auto d = defs.find(t);
        if (d != defs.end())
        {
            if (visiting.insert(t).second) resolve(d->second, defs, visiting, leaves);
        }
        else
            leaves.insert(t);
    }
}

/** True when some identifier in `leaves`, or the last member of a member chain such
 *  as arglist.sim_duration, equals `last_component`. */
static bool has_leaf_ending(const std::set<std::string>& leaves, const std::string& last_component)
{
    for (const std::string& t : leaves)
    {
        const size_t dot = t.rfind('.');
        const std::string tail = (dot == std::string::npos) ? t : t.substr(dot + 1);
        if (tail == last_component) return true;
    }
    return false;
}

/** Reads the T3 block of triton.h and returns {default arm, no-drain arm}. The block
 *  runs from the line carrying `coupling_interval.active()` and an opening brace to
 *  the first later line closing SWMM_TIME; its only compile-time split is
 *  TRITON_DIAG_TEST_NO_DRAIN. */
static bool read_arms(arm_info& dflt, arm_info& nodrain)
{
    std::ifstream in(g_triton_path);
    if (!in.is_open())
    {
        std::fprintf(stderr, "  cannot open %s\n", g_triton_path.c_str());
        return false;
    }
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);

    size_t first = lines.size();
    for (size_t i = 0; i < lines.size(); ++i)
        if (lines[i].find("coupling_interval.active()") != std::string::npos
            && lines[i].find('{') != std::string::npos)
        { first = i; break; }
    if (first == lines.size())
    {
        std::fprintf(stderr, "  no T3 block opening found\n");
        return false;
    }
    size_t last = lines.size();
    for (size_t i = first + 1; i < lines.size(); ++i)
        if (lines[i].find("st.stop(SWMM_TIME)") != std::string::npos) { last = i; break; }
    if (last == lines.size())
    {
        std::fprintf(stderr, "  no end of the T3 block found\n");
        return false;
    }

    // Split into arms by the one preprocessor conditional the block carries.
    enum mode_t { COMMON, ND_ONLY, DEFAULT_ONLY } mode = COMMON;
    std::string text_default, text_nodrain;
    bool saw_ifdef = false;
    for (size_t i = first; i <= last; ++i)
    {
        const std::string t = trim(lines[i]);
        if (t.rfind("#ifdef", 0) == 0 && t.find("TRITON_DIAG_TEST_NO_DRAIN") != std::string::npos)
        { mode = ND_ONLY; saw_ifdef = true; continue; }
        if (t.rfind("#ifndef", 0) == 0 && t.find("TRITON_DIAG_TEST_NO_DRAIN") != std::string::npos)
        { mode = DEFAULT_ONLY; saw_ifdef = true; continue; }
        if (t.rfind("#else", 0) == 0 && mode != COMMON)
        { mode = (mode == ND_ONLY) ? DEFAULT_ONLY : ND_ONLY; continue; }
        if (t.rfind("#endif", 0) == 0 && mode != COMMON) { mode = COMMON; continue; }
        const std::string l = lines[i] + "\n";
        if (mode != ND_ONLY) text_default += l;
        if (mode != DEFAULT_ONLY) text_nodrain += l;
    }
    if (!saw_ifdef)
    {
        std::fprintf(stderr, "  no TRITON_DIAG_TEST_NO_DRAIN conditional in the T3 block\n");
        return false;
    }

    arm_info* arms[2] = {&dflt, &nodrain};
    const std::string* texts[2] = {&text_default, &text_nodrain};
    for (int a = 0; a < 2; ++a)
    {
        const std::string body = strip_comments(*texts[a]);
        std::map<std::string, std::string> defs;
        static const std::regex decl("const\\s+bool\\s+([A-Za-z_][A-Za-z_0-9]*)\\s*=\\s*([^;]*);");
        for (std::sregex_iterator it(body.begin(), body.end(), decl), end; it != end; ++it)
            defs[(*it)[1].str()] = (*it)[2].str();
        const std::vector<std::string> sd = call_args(body, "step_due");
        const std::vector<std::string> st = call_args(body, "settle");
        if (sd.empty() || st.empty())
        {
            std::fprintf(stderr, "  arm %d: step_due( or settle( not found\n", a);
            return false;
        }
        arms[a]->step_due_arg = sd.front();
        arms[a]->settle_last_arg = st.back();
        std::set<std::string> visiting;
        resolve(arms[a]->step_due_arg, defs, visiting, arms[a]->leaves);
        arms[a]->found = true;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Property: in a default build, whether SWMM steps depends on the end of the run,
// so the last step of a run whose duration is not a whole number of print
// intervals still hands the pending exchange to SWMM.
// Class: every way the step_due argument can be built from the run duration
// (a local boolean, an inlined comparison, a helper taking the duration).
// Why: E-3 requires SWMM to step at the last TRITON step when T3 is on. Without
// it a partial final interval's exchange volume is summed and never delivered, the
// side-file ends short of the run's end time, and SWMM's continuity totals miss it.
// Killed by: dropping the end-of-run term from the drain (control: the
// scratch mutant that does so is run by the review; this test then fails).
// Second correct implementation, under which this passes: the term renamed
// (`is_last_step`), or inlined as `simtime + global_dt >= arglist.sim_duration`.
// Not established: the arithmetic of the comparison; that is N12's.
static void R1_default_drain_depends_on_run_end()
{
    arm_info d, n;
    CHECK(read_arms(d, n) && d.found && n.found,
          "the T3 block and both compile-time arms were found");
    if (!d.found) return;
    CHECK(has_leaf_ending(d.leaves, "sim_duration"),
          "default build: the argument of step_due depends on sim_duration (the end-of-run drain)");
    CHECK(!d.leaves.empty(), "control: the default arm resolved to a nonempty set of identifiers");
}

// ---------------------------------------------------------------------------
// Property: in a default build, whether SWMM steps also depends on the print
// interval, so every checkpoint holds a complete coupled state.
// Class: as R1, for the print-interval quantity.
// Why: a checkpoint that falls inside an interval carries an accumulator the
// resume does not have; that is what the drain exists to prevent.
// Killed by: dropping the print-instant term. Second correct implementation: the
// term computed in a helper handed `arglist.print_interval`.
static void R2_default_drain_depends_on_print_interval()
{
    arm_info d, n;
    CHECK(read_arms(d, n) && d.found, "the T3 block was found");
    if (!d.found) return;
    CHECK(has_leaf_ending(d.leaves, "print_interval"),
          "default build: step_due depends on print_interval (the print-instant drain)");
}

// ---------------------------------------------------------------------------
// Property: in a TRITON_DIAG_TEST_NO_DRAIN build the print-instant drain is gone and
// the end-of-run drain remains (E-3), so the test-only break is the one the shape
// names and not a larger one.
// Class: the no-drain arm of the drain definition.
// Why: if the switch also dropped the end-of-run drain, case N12's mutant arm
// could fail for the wrong reason and a missing end-of-run drain would look like
// the intended break.
// Killed by: a no-drain arm that keeps the print term, or drops the run-end term.
// Second correct implementation: `const bool drain = arglist.sim_duration <= ...;`
// written inline.
static void R3_no_drain_build_keeps_only_the_run_end()
{
    arm_info d, n;
    CHECK(read_arms(d, n) && n.found, "the T3 block and the no-drain arm were found");
    if (!n.found) return;
    CHECK(has_leaf_ending(n.leaves, "sim_duration"),
          "no-drain build: the end-of-run drain remains");
    CHECK(!has_leaf_ending(n.leaves, "print_interval"),
          "no-drain build: the print-instant drain is gone");
}

// ---------------------------------------------------------------------------
// Property: the boolean that makes SWMM step is the boolean that makes the lag
// pipeline apply the freshest depths, in both arms.
// Class: the last argument of settle( ... ) against the argument of step_due( ... ).
// Why: a drain that steps SWMM but leaves a lagged depth pending would write a
// checkpoint whose held depths a resume cannot reproduce.
// Killed by: settle( ... , false ) beside step_due(drain). Second correct
// implementation: both calls given the same inlined expression.
static void R4_one_drain_boolean_feeds_step_and_settle()
{
    arm_info d, n;
    CHECK(read_arms(d, n) && d.found && n.found, "the T3 block and both arms were found");
    if (!d.found) return;
    CHECK(d.step_due_arg == d.settle_last_arg,
          "default build: step_due and settle take the same drain");
    CHECK(n.step_due_arg == n.settle_last_arg,
          "no-drain build: step_due and settle take the same drain");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"R1_default_drain_depends_on_run_end", R1_default_drain_depends_on_run_end},
    {"R2_default_drain_depends_on_print_interval", R2_default_drain_depends_on_print_interval},
    {"R3_no_drain_build_keeps_only_the_run_end", R3_no_drain_build_keeps_only_the_run_end},
    {"R4_one_drain_boolean_feeds_step_and_settle", R4_one_drain_boolean_feeds_step_and_settle},
};

/** Runs the test named in argv[2], or every test; exits 1 on any failure. */
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <path to src/triton.h> [test name]\n", argv[0]);
        return 2;
    }
    g_triton_path = argv[1];
    const char* want = (argc > 2) ? argv[2] : nullptr;
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
