/** @file test_surface_diag_review.cpp
 *  @brief Reviewer test for chunk 2b.1 (T6, T2, T5): the source-level properties of the
 *         surface captures that the approved shape states as rule 6 and its two
 *         invariants, read from src/triton.h and src/diagnostics.h.
 *
 *  Why this file exists. The ctest selection builds the tip and runs the hydraulic
 *  cases with every capture OFF, so it cannot see a capture that is wrongly placed,
 *  wrongly bracketed or wrongly guarded: those fail only in the neutrality suite
 *  (chunk 3a.1), which does not exist yet. Each property below is one statement of
 *  rule 6 that is decidable from the text, and each is checked together with the
 *  test-only switch (or synthetic text) that violates it, so a green result carries a
 *  must-fail control that varies the PROPERTY and not the token.
 *
 *  It is a source-structure test and says so. It does not run the solver, link Kokkos
 *  or MPI, and it reads the files as text with comments, literals and inactive
 *  preprocessor arms blanked. Not established by it: that a bracket's duration lands
 *  in the right timer (case N10), that a collective is entered by every rank at run
 *  time (case N6), that a file is byte-identical after a resume (case N5) and that
 *  the balance closes (case N3). The checker is a hand-written scanner: a capture
 *  statement moved behind a macro or a function pointer is outside what it can see.
 *
 *  One-time chunk test of 2b.1: it pins the rule-6 table of this run's shape, so it is
 *  retired at the run's final landing with its passing evidence kept in the run record.
 *
 *  Usage: test_surface_diag_review <path to src/triton.h> [test name]
 *         (src/diagnostics.h and src/kernels.h are read from the same directory)
 */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(expr, msg)                                                           \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(expr)) {                                                             \
            ++g_failures;                                                          \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg); \
        }                                                                          \
    } while (0)

typedef std::set<std::string> strset;

/** The whole file with carriage returns removed (output.h and config_utils.h are CRLF). */
static std::string read_file(const std::string& path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());
    return s;
}

/** Replaces comments and the insides of string and character literals with spaces. The
 *  length and every newline are kept, so a position in the result is a position in the
 *  input and a brace inside a literal cannot unbalance a block. */
static std::string blank_noise(const std::string& s)
{
    std::string out = s;
    size_t i = 0;
    while (i < s.size())
    {
        if (s.compare(i, 2, "//") == 0)
        {
            while (i < s.size() && s[i] != '\n') out[i++] = ' ';
        }
        else if (s.compare(i, 2, "/*") == 0)
        {
            size_t j = s.find("*/", i + 2);
            size_t end = (j == std::string::npos) ? s.size() : j + 2;
            for (; i < end; ++i)
                if (s[i] != '\n') out[i] = ' ';
        }
        else if (s[i] == '"' || s[i] == '\'')
        {
            const char q = s[i];
            ++i;
            while (i < s.size() && s[i] != q && s[i] != '\n')
            {
                if (s[i] == '\\' && i + 1 < s.size()) out[i++] = ' ';
                out[i++] = ' ';
            }
            if (i < s.size() && s[i] == q) ++i;
        }
        else
        {
            ++i;
        }
    }
    return out;
}

/** Keeps the lines that are active when exactly the macros in `defined` are defined;
 *  inactive lines and the directive lines become spaces. A condition that is not
 *  ifdef/ifndef (for example `#if WRITE_PERFORMANCE`) counts as true. */
static std::string apply_arm(const std::string& text, const strset& defined)
{
    static const std::regex dir("^\\s*#\\s*(ifdef|ifndef|if|else|elif|endif)\\b\\s*(.*)$");
    std::string out;
    std::vector<std::pair<bool, bool> > stack;  // (active before the block, branch taken so far)
    bool active = true;
    std::istringstream in(text);
    std::string line;
    bool first = true;
    while (std::getline(in, line))
    {
        std::smatch m;
        std::string shown = line;
        if (std::regex_match(line, m, dir))
        {
            const std::string kind = m[1].str();
            std::string arg = m[2].str();
            arg.erase(arg.find_last_not_of(" \t") == std::string::npos ? 0 : arg.find_last_not_of(" \t") + 1);
            if (kind == "ifdef" || kind == "ifndef")
            {
                const bool cond = (kind == "ifdef") ? (defined.count(arg) > 0) : (defined.count(arg) == 0);
                stack.push_back(std::make_pair(active, cond));
                active = active && cond;
            }
            else if (kind == "if")
            {
                stack.push_back(std::make_pair(active, true));
            }
            else if (kind == "else" && !stack.empty())
            {
                active = stack.back().first && !stack.back().second;
                stack.back().second = !stack.back().second;
            }
            else if (kind == "elif" && !stack.empty())
            {
                active = stack.back().first && !stack.back().second;
            }
            else if (kind == "endif" && !stack.empty())
            {
                active = stack.back().first;
                stack.pop_back();
            }
            shown.assign(line.size(), ' ');
        }
        else if (!active)
        {
            shown.assign(line.size(), ' ');
        }
        if (!first) out += '\n';
        out += shown;
        first = false;
    }
    return out;
}

/** The text with comments, literals and the inactive arms of `defined` blanked. */
static std::string view(const std::string& raw, const strset& defined)
{
    return apply_arm(blank_noise(raw), defined);
}

/** Index of the brace closing the one at `open`, or npos. */
static size_t match_brace(const std::string& s, size_t open)
{
    int depth = 0;
    for (size_t j = open; j < s.size(); ++j)
    {
        if (s[j] == '{') ++depth;
        else if (s[j] == '}' && --depth == 0) return j;
    }
    return std::string::npos;
}

/** Index of the parenthesis closing the one at `open`, or npos. */
static size_t match_paren(const std::string& s, size_t open)
{
    int depth = 0;
    for (size_t j = open; j < s.size(); ++j)
    {
        if (s[j] == '(') ++depth;
        else if (s[j] == ')' && --depth == 0) return j;
    }
    return std::string::npos;
}

struct span_t
{
    std::string name;
    std::string params;
    size_t open;   ///< index of the function body's opening brace
    size_t close;  ///< index of its closing brace
};

/** True for a character that can be part of a C++ identifier. */
static bool ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

/** Every out-of-class definition `triton<T>::name(...) {...}` in the text. */
static std::vector<span_t> triton_functions(const std::string& t)
{
    std::vector<span_t> out;
    const std::string key = "triton<T>::";
    size_t p = 0;
    while ((p = t.find(key, p)) != std::string::npos)
    {
        const bool whole = (p == 0) || !ident_char(t[p - 1]);
        size_t q = p + key.size();
        p = q;
        if (!whole) continue;
        size_t e = q;
        while (e < t.size() && ident_char(t[e])) ++e;
        if (e == q || e >= t.size() || t[e] != '(') continue;
        const size_t pc = match_paren(t, e);
        if (pc == std::string::npos) continue;
        size_t k = pc + 1;
        while (k < t.size() && t[k] != '{' && t[k] != ';') ++k;
        if (k >= t.size() || t[k] == ';') continue;
        const size_t close = match_brace(t, k);
        if (close == std::string::npos) continue;
        span_t s;
        s.name = t.substr(q, e - q);
        s.params = t.substr(e + 1, pc - e - 1);
        s.open = k;
        s.close = close;
        out.push_back(s);
    }
    return out;
}

/** Every function with a body whose signature starts a line, in the style diagnostics.h
 *  and kernels.h use (optional template line, optional static/inline, a simple return type). */
static std::vector<span_t> header_functions(const std::string& t)
{
    static const std::regex sig(
        "\n[ \t]*(?:template<typename T>\n[ \t]*)?(?:static inline |inline )?"
        "(?:void|double|long|bool|T) (\\w+)\\(([^)]*)\\)[^{;]*\\{");
    std::vector<span_t> out;
    for (std::sregex_iterator it(t.begin(), t.end(), sig), end; it != end; ++it)
    {
        const size_t open = static_cast<size_t>(it->position(0) + it->length(0)) - 1;
        const size_t close = match_brace(t, open);
        if (close == std::string::npos) continue;
        span_t s;
        s.name = (*it)[1].str();
        s.params = (*it)[2].str();
        s.open = open;
        s.close = close;
        out.push_back(s);
    }
    return out;
}

/** The first function called `name`, or NULL when the file defines none. */
static const span_t* find_fn(const std::vector<span_t>& fns, const std::string& name)
{
    for (size_t i = 0; i < fns.size(); ++i)
        if (fns[i].name == name) return &fns[i];
    return NULL;
}

/** The text of a function from its opening brace to its closing brace. */
static std::string body_of(const std::string& t, const span_t& f) { return t.substr(f.open, f.close - f.open + 1); }

/** The identifier text immediately before position `pos`, skipping blanks. */
static std::string previous_token(const std::string& t, size_t pos)
{
    size_t e = pos;
    while (e > 0 && (t[e - 1] == ' ' || t[e - 1] == '\t' || t[e - 1] == '\n')) --e;
    size_t b = e;
    while (b > 0 && (ident_char(t[b - 1]) || t[b - 1] == ':')) --b;
    return t.substr(b, e - b);
}

/** True when the identifier at `pos` is being declared or defined, not called. */
static bool is_declaration(const std::string& t, size_t pos)
{
    const std::string prev = previous_token(t, pos);
    if (prev == "void" || prev == "int" || prev == "T") return true;
    return prev.size() >= 2 && prev.compare(prev.size() - 2, 2, "::") == 0;
}

/** Conditions of every `if (...)` in `body` whose block holds `pos`, plus the conditions of
 *  single-statement `if (...) return;` guards that end before `pos`. */
static std::vector<std::string> guarding_conditions(const std::string& body, size_t pos)
{
    static const std::regex iff("\\bif\\s*\\(");
    std::vector<std::string> out;
    for (std::sregex_iterator it(body.begin(), body.end(), iff), end; it != end; ++it)
    {
        const size_t op = static_cast<size_t>(it->position(0) + it->length(0)) - 1;
        const size_t cp = match_paren(body, op);
        if (cp == std::string::npos) continue;
        const std::string cond = body.substr(op + 1, cp - op - 1);
        size_t k = cp + 1;
        while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
        if (k >= body.size()) continue;
        if (body[k] == '{')
        {
            const size_t e = match_brace(body, k);
            if (e != std::string::npos && k < pos && pos < e) out.push_back(cond);
        }
        else
        {
            const size_t e = body.find(';', k);
            if (e != std::string::npos && e < pos && body.substr(k, e - k).find("return") != std::string::npos)
                out.push_back(cond);
        }
    }
    return out;
}

/** True when a condition names the identifier `rank`, which differs between ranks. */
static bool mentions_rank(const std::string& cond)
{
    static const std::regex r("\\brank\\b");
    return std::regex_search(cond, r);
}

// ---------------------------------------------------------------------------------------
// S1: no indexed write through a pointer parameter (read-only reductions).
// ---------------------------------------------------------------------------------------

/** Pointer parameter names of a parameter list. */
static strset pointer_params(const std::string& params)
{
    static const std::regex ptr("\\b(?:int|T|double|value_t|long)\\s*(?:const\\s*)?\\*\\s*(?:KOKKOS_RESTRICT\\s+)?(\\w+)");
    strset names;
    for (std::sregex_iterator it(params.begin(), params.end(), ptr), end; it != end; ++it)
        names.insert((*it)[1].str());
    return names;
}

/** Writes through pointer parameters in the functions of `raw`, as "function:parameter".
 *  `n_params` receives the number of pointer parameters examined. */
static std::vector<std::string> pointer_writes(const std::string& raw, size_t& n_params)
{
    const std::string t = view(raw, strset());
    std::vector<std::string> out;
    n_params = 0;
    const std::vector<span_t> fns = header_functions(t);
    for (size_t i = 0; i < fns.size(); ++i)
    {
        const strset names = pointer_params(fns[i].params);
        n_params += names.size();
        const std::string body = body_of(t, fns[i]);
        for (strset::const_iterator n = names.begin(); n != names.end(); ++n)
        {
            const std::regex wr("\\b" + *n + "\\s*\\[[^\\]]*\\]\\s*(=|\\+=|-=|\\*=|/=)([^=]|$)");
            if (std::regex_search(body, wr)) out.push_back(fns[i].name + ":" + *n);
        }
    }
    return out;
}

/** S1. Class: every pointer parameter of every function of src/diagnostics.h. Property: no
 *  indexed write goes through it. Why: the neutrality argument of T2, T5 and T6 is that
 *  they read solver state and write only memory they own; a reduction that stores into
 *  h, qx, qy or a right-hand-side array changes the hydraulics of a default-flag run
 *  only when the capture is on, which the all-off ctest selection never exercises.
 *  What kills it: a capture reducer storing into a solver array (for example
 *  `h[id] = 0.0;` in place of `acc += h[id] * area;`). A second correct implementation
 *  that still passes: the same reductions over `Kokkos::View<const T*>` wrappers. */
static void test_S1_capture_code_never_writes_through_a_solver_pointer(const std::string& src_dir)
{
    size_t n = 0;
    const std::string diag = read_file(src_dir + "/diagnostics.h");
    const std::vector<std::string> live = pointer_writes(diag, n);
    CHECK(live.empty(), "diagnostics.h writes through a pointer parameter");
    CHECK(n >= 5, "S1 saw fewer than 5 pointer parameters in diagnostics.h: the scan is vacuous");
    CHECK(view(diag, strset()).find("KOKKOS_LAMBDA") != std::string::npos, "diagnostics.h holds no device reduction");
    for (size_t i = 0; i < live.size(); ++i) std::fprintf(stderr, "    write: %s\n", live[i].c_str());

    // Must-fail control 1: the same scanner over a population that writes through its pointer
    // parameters (the solver's own kernels).
    size_t nk = 0;
    const std::vector<std::string> kern = pointer_writes(read_file(src_dir + "/kernels.h"), nk);
    CHECK(!kern.empty(), "control: the scanner found no write in kernels.h, which writes through rhsh0 and h_arr");

    // Must-fail control 2: a synthetic reduction that stores into its input, and a clean one.
    size_t ns = 0;
    const std::string bad = "\n  template<typename T>\n  double bad(int n, T const* h)\n  {\n    h[0] = 1.0;\n    return 0.0;\n  }\n";
    const std::string ok = "\n  template<typename T>\n  double ok(int n, T const* h)\n  {\n    return (double)h[0];\n  }\n";
    CHECK(pointer_writes(bad, ns).size() == 1, "control: a synthetic store through a parameter was not flagged");
    CHECK(pointer_writes(ok, ns).empty(), "control: a synthetic read-only reduction was flagged");
}

// ---------------------------------------------------------------------------------------
// S2: no collective sits under a rank-dependent condition.
// ---------------------------------------------------------------------------------------

static const char* const COLLECTIVE = "\\bMPI_(Reduce|Gather|Allreduce|Bcast|Barrier|Gatherv|Scatterv|Allgather)\\s*\\(";

/** "function: condition" for every collective whose guarding conditions name `rank`. */
static std::vector<std::string> rank_guarded_collectives(const std::string& raw, size_t& n_collectives)
{
    const std::string t = view(raw, strset());
    const std::regex coll(COLLECTIVE);
    std::vector<std::string> out;
    n_collectives = 0;
    const std::vector<span_t> fns = header_functions(t);
    for (size_t i = 0; i < fns.size(); ++i)
    {
        const std::string body = body_of(t, fns[i]);
        for (std::sregex_iterator it(body.begin(), body.end(), coll), end; it != end; ++it)
        {
            ++n_collectives;
            const std::vector<std::string> conds = guarding_conditions(body, static_cast<size_t>(it->position(0)));
            for (size_t c = 0; c < conds.size(); ++c)
                if (mentions_rank(conds[c])) out.push_back(fns[i].name + ": " + conds[c]);
        }
    }
    return out;
}

/** Call sites in triton.h of the capture functions that hold a collective, whose guarding
 *  conditions name `rank`. */
static std::vector<std::string> rank_guarded_collective_callers(const std::string& raw, const strset& defined)
{
    static const std::regex call("\\b(seed_storage|minloc_collective|print_instant|diag_print_row)\\s*\\(");
    const std::string t = view(raw, defined);
    std::vector<std::string> out;
    const std::vector<span_t> fns = triton_functions(t);
    for (size_t i = 0; i < fns.size(); ++i)
    {
        const std::string body = body_of(t, fns[i]);
        for (std::sregex_iterator it(body.begin(), body.end(), call), end; it != end; ++it)
        {
            if (is_declaration(body, static_cast<size_t>(it->position(0)))) continue;
            const std::vector<std::string> conds = guarding_conditions(body, static_cast<size_t>(it->position(0)));
            for (size_t c = 0; c < conds.size(); ++c)
                if (mentions_rank(conds[c])) out.push_back(fns[i].name + ": " + (*it)[1].str() + " under " + conds[c]);
        }
    }
    return out;
}

/** S2. Class: every MPI collective issued by src/diagnostics.h and every call in src/triton.h
 *  of a capture function that issues one. Property: none is guarded by a condition that
 *  mentions `rank`. Why: rule 6's C2 and Invariant B require a collective to be entered by
 *  every rank under one global condition; a rank-dependent guard hangs the job at the
 *  collective. What kills it: `if (rank == 0 && domain_balance && ...)` around the
 *  MPI_Reduce of the print-instant row. A second correct implementation that still passes:
 *  the same guard spelled with another rank-identical name (a bool member set from the cfg). */
static void test_S2_collectives_are_not_under_a_rank_dependent_condition(const std::string& src_dir)
{
    size_t n = 0;
    const std::string diag = read_file(src_dir + "/diagnostics.h");
    const std::vector<std::string> live = rank_guarded_collectives(diag, n);
    CHECK(live.empty(), "a collective of diagnostics.h is guarded by a condition that names rank");
    CHECK(n >= 4, "S2 saw fewer than 4 collectives in diagnostics.h: the scan is vacuous");
    for (size_t i = 0; i < live.size(); ++i) std::fprintf(stderr, "    guarded: %s\n", live[i].c_str());

    const std::string triton = read_file(src_dir + "/triton.h");
    const strset arms[2] = { strset(), [] { strset s; s.insert("TRITON_SWMM"); return s; }() };
    for (int a = 0; a < 2; ++a)
    {
        const std::vector<std::string> callers = rank_guarded_collective_callers(triton, arms[a]);
        CHECK(callers.empty(), "a call of a collective-bearing capture function sits under a rank condition");
        for (size_t i = 0; i < callers.size(); ++i) std::fprintf(stderr, "    caller: %s\n", callers[i].c_str());
    }

    // Must-fail controls: the same scanners over synthetic text that violates the property.
    size_t ns = 0;
    const std::string bad_in =
        "\n  void print_instant(int x)\n  {\n    if (rank == 0 && x > 0)\n    {\n      MPI_Reduce(a, b, 1, MPI_DOUBLE, MPI_SUM, 0, c);\n    }\n  }\n";
    const std::string bad_early =
        "\n  void print_instant(int x)\n  {\n    if (rank != 0) return;\n    MPI_Gather(a, 1, MPI_DOUBLE, b, 1, MPI_DOUBLE, 0, c);\n  }\n";
    const std::string ok_in =
        "\n  void print_instant(int x)\n  {\n    if (domain_balance && x > 0)\n    {\n      MPI_Reduce(a, b, 1, MPI_DOUBLE, MPI_SUM, 0, c);\n    }\n  }\n";
    CHECK(rank_guarded_collectives(bad_in, ns).size() == 1, "control: a collective inside if (rank == 0) was not flagged");
    CHECK(rank_guarded_collectives(bad_early, ns).size() == 1, "control: a collective after if (rank != 0) return was not flagged");
    CHECK(rank_guarded_collectives(ok_in, ns).empty(), "control: a collective under a rank-free guard was flagged");
    const std::string bad_call =
        "\n  void triton<T>::simulate()\n  {\n    if (rank == 0)\n    {\n      diag_surface.print_instant(a);\n    }\n  }\n";
    CHECK(rank_guarded_collective_callers(bad_call, strset()).size() == 1, "control: a collective-bearing call under if (rank == 0) was not flagged");
}

// ---------------------------------------------------------------------------------------
// S3: the domain of capture-call positions is closed (rule 6).
// ---------------------------------------------------------------------------------------

/** The functions of triton.h that may host a capture call, and for each the callees rule 6
 *  lists there. A `diag_`-prefixed helper defined in triton.h hosts calls freely, and a
 *  call to such a helper is listed where it is made. */
static std::map<std::string, strset> rule6_hosts()
{
    std::map<std::string, strset> h;
    const char* sim[] = { "diag_surface.place", "diag_surface.enable_dt_limiter", "diag_surface.enable_domain_balance",
                          "diag_surface.enable_wet_cells", "diag_surface.any", "diag_surface.init_files",
                          "diag_surface.seed_storage", "diag_surface.write_t6_record", "diag_surface.pending",
                          "diag_surface.print_instant", "diag_surface.release", "diag_print_row" };
    const char* loc[] = { "diag_surface.local_row", "diag_surface.minloc_collective", "diag_surface.sample_wet", "diag_row_offset" };
    const char* glo[] = { "diag_surface.minloc_collective" };
    const char* nst[] = { "diag_surface.read_west_east", "diag_surface.read_north_south_and_clamp", "diag_surface.ensure_extbc_mask",
                          "diag_surface.extbc_before_read", "diag_surface.extbc_after_read", "diag_surface.add_exchange",
                          "diagnostics::runoff_volume", "diagnostics::inflow_volume", "diagnostics::spin_us", "diag_open", "diag_close" };
    h["simulate"] = strset(sim, sim + sizeof(sim) / sizeof(sim[0]));
    h["compute_local_dt"] = strset(loc, loc + sizeof(loc) / sizeof(loc[0]));
    h["compute_global_dt"] = strset(glo, glo + sizeof(glo) / sizeof(glo[0]));
    h["compute_new_state"] = strset(nst, nst + sizeof(nst) / sizeof(nst[0]));
    return h;
}

/** "function: callee" for every capture call that rule 6's table does not list at that position. */
static std::vector<std::string> unlisted_capture_calls(const std::string& raw, const strset& defined, size_t& n_calls)
{
    static const std::regex call("\\b(diag_surface\\.\\w+|diagnostics::\\w+|diag_print_row|diag_row_offset|diag_open|diag_close)\\s*\\(");
    const std::string t = view(raw, defined);
    const std::map<std::string, strset> hosts = rule6_hosts();
    std::vector<std::string> out;
    n_calls = 0;
    const std::vector<span_t> fns = triton_functions(t);
    for (std::sregex_iterator it(t.begin(), t.end(), call), end; it != end; ++it)
    {
        const size_t pos = static_cast<size_t>(it->position(0));
        if (is_declaration(t, pos)) continue;
        const std::string callee = (*it)[1].str();
        std::string where = "<outside any triton<T>:: function>";
        for (size_t i = 0; i < fns.size(); ++i)
            if (fns[i].open < pos && pos < fns[i].close) { where = fns[i].name; break; }
        ++n_calls;
        if (where.compare(0, 5, "diag_") == 0) continue;  // a diag_ helper hosts calls freely
        std::map<std::string, strset>::const_iterator h = hosts.find(where);
        if (h == hosts.end() || h->second.count(callee) == 0) out.push_back(where + ": " + callee);
    }
    return out;
}

/** S3. Class: every call of a surface-capture function in src/triton.h. Property: it sits in
 *  one of the four step-loop functions, at a callee rule 6's table lists for that function
 *  (or inside a `diag_`-prefixed helper). Why: rule 6 closes the domain of positions and
 *  calls a capture or test call at any position the table does not list a defect, because
 *  an unlisted position has no C1 to C8 verdict. What kills it: a capture read added to
 *  new_domain_decomposition or to the observation block. A second correct implementation
 *  that still passes: the same statements moved into a `diag_`-prefixed helper that the
 *  listed host calls. */
static void test_S3_capture_calls_sit_only_at_rule_6_positions(const std::string& src_dir)
{
    const std::string triton = read_file(src_dir + "/triton.h");
    const strset swmm = [] { strset s; s.insert("TRITON_SWMM"); return s; }();
    size_t n = 0;
    const std::vector<std::string> a = unlisted_capture_calls(triton, strset(), n);
    CHECK(a.empty(), "an unlisted capture call sits in triton.h (build without TRITON_SWMM)");
    CHECK(n >= 20, "S3 saw fewer than 20 capture calls in the default arm: the scan is vacuous");
    for (size_t i = 0; i < a.size(); ++i) std::fprintf(stderr, "    unlisted: %s\n", a[i].c_str());
    size_t n2 = 0;
    const std::vector<std::string> b = unlisted_capture_calls(triton, swmm, n2);
    CHECK(b.empty(), "an unlisted capture call sits in triton.h (build with TRITON_SWMM)");
    CHECK(n2 > n, "S3: the TRITON_SWMM arm shows no extra capture call (the exchange reads)");

    // Must-fail controls over synthetic text.
    size_t ns = 0;
    const std::string elsewhere =
        "\n  void triton<T>::new_domain_decomposition()\n  {\n    diag_surface.sample_wet(rows, cols, device_vec[H], 0.0);\n  }\n";
    const std::string wrong_host =
        "\n  void triton<T>::compute_global_dt(int print_id)\n  {\n    diag_surface.print_instant(rows, cols, device_vec[H], 0.0, 0, 0.0);\n  }\n";
    const std::string right_host =
        "\n  void triton<T>::compute_global_dt(int print_id)\n  {\n    diag_surface.minloc_collective();\n  }\n";
    const std::string helper =
        "\n  void triton<T>::diag_anything()\n  {\n    diag_surface.print_instant(rows, cols, device_vec[H], 0.0, 0, 0.0);\n  }\n";
    CHECK(unlisted_capture_calls(elsewhere, strset(), ns).size() == 1, "control: a capture call in another function was not flagged");
    CHECK(unlisted_capture_calls(wrong_host, strset(), ns).size() == 1, "control: an unlisted callee in a listed host was not flagged");
    CHECK(unlisted_capture_calls(right_host, strset(), ns).empty(), "control: a listed callee at its listed host was flagged");
    CHECK(unlisted_capture_calls(helper, strset(), ns).empty(), "control: a diag_ helper was not allowed to host calls");
}

// ---------------------------------------------------------------------------------------
// S4: every DIAG_TIME bracket opens while no other category is open (C1).
// ---------------------------------------------------------------------------------------

/** The lambda definitions `auto diag_open = [&]() {...};` blanked, so that their bodies are
 *  read as the calls `diag_open()` and `diag_close()` and not as brackets in place. */
static std::string blank_diag_lambdas(const std::string& t)
{
    static const std::regex def("\\bauto\\s+diag_(open|close)\\s*=\\s*\\[[^\\]]*\\]\\s*\\(\\)\\s*\\{");
    std::string out = t;
    std::smatch m;
    std::string rest = t;
    size_t base = 0;
    while (std::regex_search(rest, m, def))
    {
        const size_t open = base + static_cast<size_t>(m.position(0) + m.length(0)) - 1;
        const size_t close = match_brace(out, open);
        if (close == std::string::npos) break;
        for (size_t i = base + static_cast<size_t>(m.position(0)); i <= close; ++i)
            if (out[i] != '\n') out[i] = ' ';
        base = close + 1;
        rest = out.substr(base);
    }
    return out;
}

/** Categories other than the two parents open at each DIAG_TIME opening, as "context: list".
 *  A stop of a category that is not open and a start of one that already is are ignored, so
 *  the two branches of an if/else that each close and reopen a category read correctly.
 *  `n_opens` receives the number of DIAG openings examined. */
static std::vector<std::string> diag_opens_with_other_open(const std::string& body, const std::string& fname, size_t& n_opens)
{
    static const std::regex ev("\\bst\\.(start|stop)\\((\\w+)\\)|\\b(diag_open|diag_close)\\(\\)|\\b(diag_print_row)\\(");
    const std::string b = blank_diag_lambdas(body);
    strset open_;
    std::vector<std::string> bad;
    for (std::sregex_iterator it(b.begin(), b.end(), ev), end; it != end; ++it)
    {
        const std::smatch& m = *it;
        bool opens_diag = false;
        if (m[1].matched)
        {
            if (m[1].str() == "stop") { open_.erase(m[2].str()); continue; }
            if (m[2].str() == "DIAG_TIME") opens_diag = true;
            else { open_.insert(m[2].str()); continue; }
        }
        else if (m[3].matched && m[3].str() == "diag_close") { open_.erase("DIAG_TIME"); open_.insert("COMPUTE_TIME"); continue; }
        else if (m[3].matched) { open_.erase("COMPUTE_TIME"); opens_diag = true; }
        else if (m[4].matched) { opens_diag = true; }
        if (!opens_diag) continue;
        ++n_opens;
        std::string others;
        for (strset::const_iterator c = open_.begin(); c != open_.end(); ++c)
            if (*c != "SIMULATION_TIME" && *c != "TOTAL_TIME") others += (others.empty() ? "" : ",") + *c;
        if (!others.empty()) bad.push_back(fname + ": DIAG opens while " + others + " is open");
        if (m[4].matched) continue;  // a call into diag_print_row opens and closes its own bracket
        open_.insert("DIAG_TIME");
    }
    return bad;
}

/** Violations over the five functions that open DIAG_TIME, in one arm. */
static std::vector<std::string> bracket_violations(const std::string& raw, const strset& defined, size_t& n_opens)
{
    const std::string t = view(raw, defined);
    const char* names[] = { "simulate", "compute_local_dt", "compute_global_dt", "compute_new_state", "diag_print_row" };
    const std::vector<span_t> fns = triton_functions(t);
    std::vector<std::string> bad;
    n_opens = 0;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
    {
        const span_t* f = find_fn(fns, names[i]);
        if (f == NULL) { bad.push_back(std::string(names[i]) + ": function not found"); continue; }
        const std::vector<std::string> v = diag_opens_with_other_open(body_of(t, *f), names[i], n_opens);
        bad.insert(bad.end(), v.begin(), v.end());
    }
    return bad;
}

/** S4. Class: every point of src/triton.h that opens DIAG_TIME (an `st.start(DIAG_TIME)`, a
 *  `diag_open()` or a call of diag_print_row), in the default arm with and without
 *  TRITON_SWMM. Property: no category other than the two parents is open there, because
 *  super_timer categories are independent accumulators and an overlap double-counts. Why:
 *  Invariant B and C1 require capture time to accrue to DIAG and to no other category.
 *  What kills it: `TRITON_DIAG_TIMER_MUTANT`, which drops the COMPUTE_TIME stop of the
 *  first T2 row; the test must see it. A second correct implementation that still passes: a
 *  different but still balanced stop/start pairing around the same read. */
static void test_S4_every_diag_bracket_opens_with_no_other_category_open(const std::string& src_dir)
{
    const std::string triton = read_file(src_dir + "/triton.h");
    const strset swmm = [] { strset s; s.insert("TRITON_SWMM"); return s; }();
    const strset mutant = [] { strset s; s.insert("TRITON_DIAG_TIMER_MUTANT"); return s; }();
    size_t n0 = 0, n1 = 0, nm = 0;
    const std::vector<std::string> v0 = bracket_violations(triton, strset(), n0);
    const std::vector<std::string> v1 = bracket_violations(triton, swmm, n1);
    CHECK(v0.empty(), "a DIAG_TIME opening finds another category open (default arm)");
    CHECK(v1.empty(), "a DIAG_TIME opening finds another category open (TRITON_SWMM arm)");
    CHECK(n0 >= 12, "S4 saw fewer than 12 DIAG openings in the default arm: the scan is vacuous");
    CHECK(n1 > n0, "S4: the TRITON_SWMM arm shows no extra DIAG opening (the exchange reads)");
    for (size_t i = 0; i < v0.size(); ++i) std::fprintf(stderr, "    default: %s\n", v0[i].c_str());
    for (size_t i = 0; i < v1.size(); ++i) std::fprintf(stderr, "    swmm: %s\n", v1[i].c_str());

    // Must-fail control 1: the shape's own mis-bracket build.
    const std::vector<std::string> vm = bracket_violations(triton, mutant, nm);
    CHECK(!vm.empty(), "control: TRITON_DIAG_TIMER_MUTANT leaves no DIAG opening under an open category");

    // Must-fail control 2: synthetic text, and its balanced twin.
    size_t ns = 0;
    const std::string bad = "{ st.start(COMPUTE_TIME); st.start(DIAG_TIME); st.stop(DIAG_TIME); st.stop(COMPUTE_TIME); }";
    const std::string ok = "{ st.start(COMPUTE_TIME); st.stop(COMPUTE_TIME); st.start(DIAG_TIME); st.stop(DIAG_TIME); }";
    CHECK(diag_opens_with_other_open(bad, "bad", ns).size() == 1, "control: DIAG opened under COMPUTE_TIME was not flagged");
    CHECK(diag_opens_with_other_open(ok, "ok", ns).empty(), "control: a balanced swap was flagged");
}

// ---------------------------------------------------------------------------------------
// S5 and S7: the order of the print-instant row, the flush, the cfg and the close (C3).
// ---------------------------------------------------------------------------------------

/** Index of the first occurrence of `token` in `body`, or npos. */
static size_t pos_of(const std::string& body, const std::string& token)
{
    return body.find(token);
}

/** True when `a` occurs, then `b`, then `c`, each in `body`. */
static bool in_order(const std::string& body, const std::string& a, const std::string& b, const std::string& c)
{
    const size_t pa = pos_of(body, a), pb = pos_of(body, b), pc = pos_of(body, c);
    return pa != std::string::npos && pb != std::string::npos && pc != std::string::npos && pa < pb && pb < pc;
}

/** The body of triton<T>::simulate in the arm `defined`, or an empty string. */
static std::string simulate_body(const std::string& raw, const strset& defined)
{
    const std::string t = view(raw, defined);
    const std::vector<span_t> fns = triton_functions(t);
    const span_t* f = find_fn(fns, "simulate");
    return f == NULL ? std::string() : body_of(t, *f);
}

/** S5. Class: the position of the print-instant row against the capture flush and the
 *  checkpoint cfg, over the builds the shape names. Property: in the default build the row
 *  is written before the flush and the flush before `out.write_output`; in the T-L build
 *  the row comes after the cfg; with T-P defined the two pauses sit between the flush and
 *  the cfg and after the cfg. Why: C3. A row written after the flush or the cfg is lost
 *  when the run is killed in that state, and the resumed file lacks it. What kills it:
 *  moving diag_print_row after `st.stop(IO_TIME)` in the default build. A second correct
 *  implementation that still passes: the row issued from a helper called at the same place. */
static void test_S5_print_instant_row_precedes_the_flush_and_the_cfg(const std::string& src_dir)
{
    const std::string triton = read_file(src_dir + "/triton.h");
    const std::string d = simulate_body(triton, strset());
    CHECK(in_order(d, "diag_print_row(", "diag_log::flush_all(", "out.write_output("),
          "default build: the print-instant row is not before the capture flush and the cfg");
    CHECK(d.find("sleep_for(") == std::string::npos, "default build: a test-only pause is compiled in");

    // Must-fail control 1: the shape's late-row build must violate the default ordering.
    strset late;
    late.insert("TRITON_DIAG_TEST_LATE_INTERVAL");
    const std::string l = simulate_body(triton, late);
    CHECK(!in_order(l, "diag_print_row(", "diag_log::flush_all(", "out.write_output(") &&
              pos_of(l, "diag_print_row(") != std::string::npos &&
              pos_of(l, "diag_print_row(") > pos_of(l, "out.write_output("),
          "control: the T-L build does not put the row after the cfg");

    // The T-P pauses fix the two kill states of N5 (d) and (f).
    strset pause;
    pause.insert("TRITON_DIAG_TEST_PAUSE_AFTER_FLUSH");
    const std::string p = simulate_body(triton, pause);
    const size_t s1 = p.find("sleep_for(");
    const size_t s2 = (s1 == std::string::npos) ? s1 : p.find("sleep_for(", s1 + 1);
    const size_t fl = p.find("diag_log::flush_all(");
    const size_t wo = p.find("out.write_output(");
    CHECK(s1 != std::string::npos && s2 != std::string::npos && p.find("sleep_for(", s2 + 1) == std::string::npos,
          "T-P build: the pause is not compiled in exactly twice");
    CHECK(s1 != std::string::npos && s2 != std::string::npos && fl < s1 && s1 < wo && wo < s2,
          "T-P build: the pauses are not between flush and cfg, and after the cfg");
    CHECK(p.find("diag_print_row(") != std::string::npos && p.find("diag_print_row(") < fl, "T-P build: the row moved from before the flush");

    // Must-fail control 2: synthetic swapped order.
    CHECK(!in_order("out.write_output(); diag_log::flush_all(); diag_print_row();", "diag_print_row(", "diag_log::flush_all(", "out.write_output("),
          "control: a swapped ordering was accepted");
}

/** S7. Class: the end-of-run close row against the file close. Property: in the default
 *  build the pending-rows call precedes `diag_log::close_all`, which runs once; under T-C
 *  the pending-rows call is absent and the close remains. Why: a row appended after the
 *  close aborts the run at the first write, and a close that is skipped loses the tail.
 *  What kills it: `close_all()` moved above the print_instant call. A second correct
 *  implementation that still passes: the pending rows written by a helper called before
 *  close_all. */
static void test_S7_close_row_writes_pending_rows_before_closing_the_files(const std::string& src_dir)
{
    const std::string triton = read_file(src_dir + "/triton.h");
    const std::string d = simulate_body(triton, strset());
    const size_t pi = d.find("diag_surface.print_instant(");
    const size_t ca = d.find("diag_log::close_all(");
    CHECK(pi != std::string::npos && ca != std::string::npos && pi < ca, "default build: pending rows are not written before close_all");
    CHECK(ca != std::string::npos && d.find("diag_log::close_all(", ca + 1) == std::string::npos, "default build: close_all is not called exactly once");

    // Must-fail control: the shape's no-tail build must drop the pending-rows call and keep the close.
    strset notail;
    notail.insert("TRITON_DIAG_TEST_NO_TAIL");
    const std::string t = simulate_body(triton, notail);
    CHECK(t.find("diag_surface.print_instant(") == std::string::npos && t.find("diag_log::close_all(") != std::string::npos,
          "control: the T-C build still writes pending rows, or lost the close");
}

// ---------------------------------------------------------------------------------------
// S6: the T6 collective follows the step's MPI_MIN allreduce (C2).
// ---------------------------------------------------------------------------------------

/** Number of minloc_collective calls in compute_local_dt and compute_global_dt, and whether
 *  each follows the MPI_MIN allreduce of its own function. */
static void collective_order(const std::string& raw, const strset& defined, size_t& calls, bool& all_after)
{
    const std::string t = view(raw, defined);
    const std::vector<span_t> fns = triton_functions(t);
    const char* names[] = { "compute_local_dt", "compute_global_dt" };
    calls = 0;
    all_after = true;
    for (size_t i = 0; i < 2; ++i)
    {
        const span_t* f = find_fn(fns, names[i]);
        if (f == NULL) { all_after = false; continue; }
        const std::string body = body_of(t, *f);
        const size_t mn = body.find("MPI_Allreduce(&local_dt, &global_dt");
        size_t at = 0;
        while ((at = body.find("minloc_collective(", at)) != std::string::npos)
        {
            ++calls;
            if (mn == std::string::npos || mn > at) all_after = false;
            ++at;
        }
    }
}

/** S6. Class: every call of minloc_collective in the two dt functions. Property: it follows
 *  the step's existing `MPI_Allreduce(&local_dt, &global_dt ...)` of the same function.
 *  Why: C2 requires that a capture collective is not issued before the step's synchronising
 *  collective, where it would move load-imbalance wait into DIAG. What kills it: the T-E
 *  build, which issues it from compute_local_dt. A second correct implementation that still
 *  passes: the collective issued from a `diag_` helper called after the allreduce. */
static void test_S6_t6_collective_follows_the_min_allreduce(const std::string& src_dir)
{
    const std::string triton = read_file(src_dir + "/triton.h");
    size_t calls = 0;
    bool after = false;
    collective_order(triton, strset(), calls, after);
    CHECK(calls == 1 && after, "default build: the T6 collective is not issued exactly once, after the MPI_MIN allreduce");

    // Must-fail control: the shape's early-collective build.
    strset early;
    early.insert("TRITON_DIAG_TEST_EARLY_MINLOC");
    size_t ec = 0;
    bool eafter = true;
    collective_order(triton, early, ec, eafter);
    CHECK(ec == 1 && !eafter, "control: the T-E build does not issue the T6 collective before the MPI_MIN allreduce");
}

// ---------------------------------------------------------------------------------------

struct named_test
{
    const char* name;
    void (*fn)(const std::string&);
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <path to src/triton.h> [test name]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const size_t slash = path.find_last_of('/');
    const std::string src_dir = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
    const std::string only = (argc > 2) ? argv[2] : "";

    const named_test tests[] = {
        { "S1_capture_code_never_writes_through_a_solver_pointer", test_S1_capture_code_never_writes_through_a_solver_pointer },
        { "S2_collectives_are_not_under_a_rank_dependent_condition", test_S2_collectives_are_not_under_a_rank_dependent_condition },
        { "S3_capture_calls_sit_only_at_rule_6_positions", test_S3_capture_calls_sit_only_at_rule_6_positions },
        { "S4_every_diag_bracket_opens_with_no_other_category_open", test_S4_every_diag_bracket_opens_with_no_other_category_open },
        { "S5_print_instant_row_precedes_the_flush_and_the_cfg", test_S5_print_instant_row_precedes_the_flush_and_the_cfg },
        { "S6_t6_collective_follows_the_min_allreduce", test_S6_t6_collective_follows_the_min_allreduce },
        { "S7_close_row_writes_pending_rows_before_closing_the_files", test_S7_close_row_writes_pending_rows_before_closing_the_files },
    };

    bool ran = false;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i)
    {
        if (!only.empty() && only != tests[i].name) continue;
        ran = true;
        const int before = g_failures;
        tests[i].fn(src_dir);
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", tests[i].name);
    }
    if (!ran)
    {
        std::fprintf(stderr, "no test named %s\n", only.c_str());
        return 2;
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
