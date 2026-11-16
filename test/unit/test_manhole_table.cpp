/** @file test_manhole_table.cpp
 *  @brief Tier-1 tests for the per-node manhole table (cfg key swmm_manhole_table).
 *
 *  No solver run, no cluster, no SWMM call. With the key set, each coupling node
 *  takes the diameter and the loss coefficient of its table row instead of the
 *  two constants manhole_diameter and manhole_loss. The code under test is the
 *  group of functions beside read_coupling_list in swmm_triton.h:
 *
 *    read_manhole_table         -- parses the file, refusing a malformed row;
 *    check_manhole_table_names  -- a repeated name, a name that is no coupling node;
 *    assign_manhole_table       -- both columns by case-folded name, in coupling
 *                                  order, and the first coupling node with no row;
 *    load_manhole_table         -- the three above plus the file open, the warning
 *                                  and exit(EXIT_FAILURE), run in a forked child so
 *                                  an exit is observed as a status.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one
 *  node id per property; with no argument every test runs. In every fixture the
 *  two values of a row differ numerically, and the two constants of the
 *  constant-equal case differ from each other, so a reader that swaps the
 *  columns fails an assignment check. These are the tests of package 1f.
 */

// The same include prefix as test_coupling_interval.cpp, for the reasons
// test_exchange_log_header.cpp states: swmm_triton.h compiles only after
// main.cpp's include chain.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
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

struct parsed_t
{
    std::vector<std::string> names;
    std::vector<value_t> diameters, losses;
    std::vector<int> lines;
    manhole_table_error_t err;
};

static parsed_t parse(const std::string& text)
{
    parsed_t p;
    std::istringstream in(text);
    p.err = read_manhole_table(in, p.names, p.diameters, p.losses, p.lines);
    return p;
}

/** True when the refusal names `line` and its text contains `needle`. */
static bool refuses_at(const std::string& text, int line, const char* needle)
{
    const parsed_t p = parse(text);
    return !p.err.what.empty() && p.err.line == line && p.err.what.find(needle) != std::string::npos;
}

/** Writes `text` to a file of its own under the temp directory and returns the path. */
static std::string write_table(const std::string& tag, const std::string& text)
{
    const std::string path = (std::filesystem::temp_directory_path()
        / ("test_manhole_table_" + std::to_string(static_cast<long>(::getpid())) + "_" + tag + ".txt")).string();
    std::ofstream out(path.c_str());
    out << text;
    return path;
}

struct child_result_t { int status = -1; std::string err; };

/** Runs `fn` in a forked child with stderr piped back, so an exit(EXIT_FAILURE) is a status. */
static child_result_t in_child(const std::function<void()>& fn)
{
    child_result_t r;
    int fd[2];
    if (::pipe(fd) != 0) return r;
    std::cerr.flush();
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(fd[0]);
        ::dup2(fd[1], 2);
        ::close(fd[1]);
        fn();
        std::cerr.flush();
        ::_exit(0);
    }
    ::close(fd[1]);
    char buf[512];
    ssize_t n;
    while ((n = ::read(fd[0], buf, sizeof buf)) > 0) r.err.append(buf, static_cast<std::size_t>(n));
    ::close(fd[0]);
    int st = 0;
    ::waitpid(pid, &st, 0);
    r.status = WIFEXITED(st) ? WEXITSTATUS(st) : 255;
    return r;
}

/** load_manhole_table over `text` in a child, on rank `rank`, with the coupling nodes `coupling`. */
static child_result_t load(const std::string& tag, const std::string& text, const std::vector<std::string>& coupling, value_t dx, int rank, std::string* path_out = nullptr)
{
    const std::string path = write_table(tag, text);
    if (path_out) *path_out = path;
    const child_result_t r = in_child([&]() {
        std::vector<value_t> diameter(coupling.size(), 0), loss(coupling.size(), 0);
        load_manhole_table(path, coupling, dx, rank, diameter, loss);
    });
    std::remove(path.c_str());
    return r;
}

// P1. A table with comments, blank lines, indentation, CRLF and mixed case parses.
// Killed by: a reader that treats ';' lines or blank lines as rows, keeps the line
// ending in a token, or lower-cases or drops names. Second correct implementation:
// splitting on any whitespace run rather than on single spaces.
static void P1_table_with_comments_blank_lines_and_mixed_case_parses()
{
    const parsed_t p = parse(
        "; manhole table\n"
        "\n"
        "   J1 0.6 0.35   \n"
        "\t;indented comment\n"
        "Mixed_Case\t2.5\t0.8\r\n"
        "   \n"
        "j3 1.8e0 0.05\n");
    CHECK(p.err.what.empty(), "a well-formed table is not refused");
    CHECK(p.names.size() == 3, "three rows");
    if (p.names.size() != 3) return;
    CHECK(p.names[0] == "J1" && p.names[1] == "Mixed_Case" && p.names[2] == "j3", "names keep their spelling, in file order");
    CHECK(p.lines[0] == 3 && p.lines[1] == 5 && p.lines[2] == 7, "each row keeps its 1-based line number");
    CHECK(p.diameters[0] == static_cast<value_t>(0.6) && p.diameters[1] == static_cast<value_t>(2.5) && p.diameters[2] == static_cast<value_t>(1.8),
          "the second column is the diameter");
    CHECK(p.losses[0] == static_cast<value_t>(0.35) && p.losses[1] == static_cast<value_t>(0.8) && p.losses[2] == static_cast<value_t>(0.05),
          "the third column is the loss coefficient");
    CHECK(parse("").err.what.empty() && parse("").names.empty(), "control: an empty table parses to no rows");
}

// F1. A line that is not exactly a name and two numbers is refused, with its line number.
// Killed by: accepting a short row (one value), a long row, or a token strtod only
// partly consumes. Second correct implementation: sscanf of three fields, which
// accepts a trailing token and "1.2x"; this test rejects both.
static void F1_a_line_that_is_not_a_name_and_two_numbers_is_refused_with_its_line()
{
    CHECK(refuses_at("A 1.0 0.1\nB 1.0\n", 2, "expected"), "a row with only one of the two values is refused at line 2");
    CHECK(refuses_at("A 1.0 0.1\n\n\nB\n", 4, "expected"), "a bare name is refused at line 4");
    CHECK(refuses_at("; c\nA 1.0 0.1 0.2\n", 2, "expected"), "a fourth token is refused at line 2");
    CHECK(refuses_at("A 1.0x 0.1\n", 1, "expected"), "a diameter strtod consumes only in part is refused");
    CHECK(refuses_at("A 1.0 0.1y\n", 1, "expected"), "a loss coefficient strtod consumes only in part is refused");
    CHECK(refuses_at("A abc 0.1\n", 1, "expected"), "a diameter that is no number is refused");
    CHECK(refuses_at("A 1.0 0.1 ; trailing comment\n", 1, "expected"), "a trailing comment is a further token and is refused");
    CHECK(parse("A 1.0 0.1\n").err.what.empty(), "control: the same shape with exactly three tokens passes");
}

// F2. A diameter that is not finite or not above zero is refused, with its line number.
// Killed by: testing only d <= 0 (nan passes it), or only isfinite (zero passes).
static void F2_a_diameter_not_finite_or_not_above_zero_is_refused_with_its_line()
{
    CHECK(refuses_at("A 1.0 0.1\nB 0 0.1\n", 2, "diameter"), "a zero diameter is refused at line 2");
    CHECK(refuses_at("A -0.5 0.1\n", 1, "diameter"), "a negative diameter is refused");
    CHECK(refuses_at("A 1.0 0.1\nB 1.0 0.1\nC nan 0.1\n", 3, "diameter"), "a nan diameter is refused at line 3");
    CHECK(refuses_at("A inf 0.1\n", 1, "diameter"), "an infinite diameter is refused");
    CHECK(refuses_at("A 1e999 0.1\n", 1, "diameter"), "a diameter that overflows to infinity is refused");
    CHECK(parse("A 1e-6 0.1\n").err.what.empty(), "control: a tiny positive diameter passes");
}

// F3. A loss coefficient that is not finite or is negative is refused; zero passes.
// Killed by: refusing zero (a legitimate loss), or accepting a negative or nan.
static void F3_a_loss_coefficient_not_finite_or_negative_is_refused_with_its_line()
{
    CHECK(refuses_at("A 1.0 0.1\nB 1.0 -0.1\n", 2, "loss"), "a negative loss coefficient is refused at line 2");
    CHECK(refuses_at("A 1.0 nan\n", 1, "loss"), "a nan loss coefficient is refused");
    CHECK(refuses_at("A 1.0 inf\n", 1, "loss"), "an infinite loss coefficient is refused");
    const parsed_t z = parse("A 1.0 0\n");
    CHECK(z.err.what.empty() && z.losses.size() == 1 && z.losses[0] == 0, "control: a zero loss coefficient is a valid row");
}

// F4. A name repeated after SWMM's case fold is refused, naming the second line.
// Killed by: a byte-exact comparison (J1 and j1 pass it). Second correct
// implementation: sorting the folded names; the repeat is still reported at the
// later line.
static void F4_a_name_repeated_after_the_case_fold_is_refused_with_its_line()
{
    const parsed_t p = parse("J1 1.0 0.1\nJ2 1.5 0.2\nj1 2.0 0.3\n");
    const manhole_table_error_t e = check_manhole_table_names(p.names, p.lines, {"J1", "J2"});
    CHECK(e.line == 3 && e.what.find("repeats J1") != std::string::npos && e.what.find("line 1") != std::string::npos,
          "j1 on line 3 repeats J1 on line 1");
    const parsed_t q = parse("J1 1.0 0.1\nJ2 1.5 0.2\n");
    CHECK(check_manhole_table_names(q.names, q.lines, {"J1", "J2"}).what.empty(), "control: distinct names pass");
}

// F5. A name that is no coupling node is refused, naming its line.
// Killed by: ignoring surplus rows (a misspelt name would leave its node at no value).
static void F5_a_name_that_is_no_coupling_node_is_refused_with_its_line()
{
    const parsed_t p = parse("J1 1.0 0.1\nJ9 1.5 0.2\n");
    const manhole_table_error_t e = check_manhole_table_names(p.names, p.lines, {"J1", "J2"});
    CHECK(e.line == 2 && e.what.find("J9") != std::string::npos && e.what.find("not a coupling node") != std::string::npos,
          "J9 on line 2 is no coupling node");
    const parsed_t q = parse("j2 1.0 0.1\n");
    CHECK(check_manhole_table_names(q.names, q.lines, {"J1", "J2"}).what.empty(), "control: a name equal after the case fold is a coupling node");
}

// F6. A coupling node with no row is reported by assign_manhole_table (the first such node).
// Killed by: filling the node with a constant or with a neighbour's row. The solver
// fills nothing: the table is 1:1 with the coupling nodes or the run stops.
static void F6_a_coupling_node_with_no_row_is_reported()
{
    const parsed_t p = parse("J1 1.0 0.1\nJ3 1.5 0.2\n");
    std::vector<value_t> d(3, -7), l(3, -7);
    const int missing = assign_manhole_table({"J1", "J2", "J3"}, p.names, p.diameters, p.losses, d, l);
    CHECK(missing == 1, "J2 (index 1) is the first coupling node with no row");
    CHECK(d[1] == -7 && l[1] == -7, "the node with no row is left untouched, not filled");
    const parsed_t q = parse("j1 1.0 0.1\nJ2 1.5 0.2\nJ3 2.0 0.3\n");
    CHECK(assign_manhole_table({"J1", "J2", "J3"}, q.names, q.diameters, q.losses, d, l) == -1, "control: a full table reports no missing node");
}

// A1. Both columns follow the case-folded name, in coupling order, whatever the row order.
// Killed by: assignment by row position, by name without the fold, or with the two
// columns swapped. Every row's two values differ, so a swap changes an asserted value.
static void A1_assignment_of_both_columns_follows_the_case_folded_name_in_coupling_order()
{
    const std::vector<std::string> coupling = {"J3", "j1", "Mixed_Case", "J2"};
    const parsed_t p = parse("J1 0.6 0.35\nj2 1.5 0.25\nJ3 2.5 0.8\nmixed_case 3.5 0.05\n");
    CHECK(p.err.what.empty() && check_manhole_table_names(p.names, p.lines, coupling).what.empty(), "the fixture is a valid 1:1 table");
    std::vector<value_t> d(4, -1), l(4, -1);
    CHECK(assign_manhole_table(coupling, p.names, p.diameters, p.losses, d, l) == -1, "every coupling node finds its row");
    const value_t wd[4] = {2.5, 0.6, 3.5, 1.5};
    const value_t wl[4] = {0.8, 0.35, 0.05, 0.25};
    for (int i = 0; i < 4; ++i) {
        CHECK(d[i] == wd[i], "the diameter of coupling node i is its own row's second column");
        CHECK(l[i] == wl[i], "the loss coefficient of coupling node i is its own row's third column");
    }
    CHECK(d[0] != l[0] && d[1] != l[1] && d[2] != l[2] && d[3] != l[3], "control: every fixture row's two values differ, so a column swap is visible");
}

// A2. A table whose every row equals the two constants gives diameter[] and loss[]
// bit-identical to the constant loop's. The constant loop is reproduced here from
// read_inp_file's own statements (loss[i]=manhole_loss; diameter[i]=manhole_diameter),
// the constants parsed by atof as the cfg reader does, and the table text printed
// with 17 significant digits as a round-trip text.
// Killed by: a parse that loses digits (float strtof, a fixed precision), or by
// either column taking the other constant. The two constants differ (1.2 and 0.1).
static void A2_a_table_equal_to_both_constants_is_bit_identical_to_the_constant_loop()
{
    const value_t manhole_diameter = static_cast<value_t>(std::atof("1.2"));
    const value_t manhole_loss = static_cast<value_t>(std::atof("0.1"));
    CHECK(manhole_diameter != manhole_loss, "control: the two constants differ");
    const std::vector<std::string> coupling = {"J1", "J2", "J3", "J4", "J5"};
    std::vector<value_t> cd(5), cl(5);
    for (int i = 0; i < 5; ++i) {   // the constant loop of read_inp_file
        cl[i] = manhole_loss;
        cd[i] = manhole_diameter;
    }
    std::string text;
    for (const std::string& n : coupling) {
        char row[128];
        std::snprintf(row, sizeof row, "%s %.17g %.17g\n", n.c_str(), static_cast<double>(manhole_diameter), static_cast<double>(manhole_loss));
        text += row;
    }
    const parsed_t p = parse(text);
    CHECK(p.err.what.empty(), "the constant table parses");
    std::vector<value_t> td(5, 0), tl(5, 0);
    CHECK(assign_manhole_table(coupling, p.names, p.diameters, p.losses, td, tl) == -1, "every node has a row");
    CHECK(std::memcmp(td.data(), cd.data(), 5 * sizeof(value_t)) == 0, "diameter[] is byte-identical to the constant loop's");
    CHECK(std::memcmp(tl.data(), cl.data(), 5 * sizeof(value_t)) == 0, "loss[] is byte-identical to the constant loop's");
    std::vector<value_t> other(5, static_cast<value_t>(std::atof("1.3")));
    CHECK(std::memcmp(td.data(), other.data(), 5 * sizeof(value_t)) != 0, "control: the comparison does fail for a different diameter");
}

// L1. Through load_manhole_table every refusal stops the process with a failure
// status and names the file and, where it has one, the line; a good table does not.
// Killed by: a refusal that returns instead of exiting (the run would continue at
// the constants), or a message with no file or no line.
static void L1_every_refusal_stops_the_run_naming_the_file_and_the_line()
{
    const std::vector<std::string> coupling = {"J1", "J2"};
    struct bad_t { const char* tag; const char* text; const char* line; const char* needle; };
    const bad_t bad[] = {
        {"short",   "J1 1.0 0.1\nJ2 1.5\n",             "line 2:", "expected"},
        {"diam",    "J1 0 0.1\nJ2 1.5 0.2\n",           "line 1:", "diameter"},
        {"loss",    "J1 1.0 0.1\nJ2 1.5 -1\n",          "line 2:", "loss"},
        {"repeat",  "J1 1.0 0.1\nj1 1.5 0.2\nJ2 1 1\n", "line 2:", "repeats"},
        {"unknown", "J1 1.0 0.1\nJ2 1.5 0.2\nJ7 1 1\n", "line 3:", "not a coupling node"},
    };
    for (const bad_t& b : bad) {
        std::string path;
        const child_result_t r = load(b.tag, b.text, coupling, 10, 0, &path);
        CHECK(r.status == EXIT_FAILURE, b.tag);
        CHECK(r.err.find(path) != std::string::npos, "the refusal names the file");
        CHECK(r.err.find(b.line) != std::string::npos, "the refusal names the line");
        CHECK(r.err.find(b.needle) != std::string::npos, "the refusal says what is wrong");
    }
    std::string path;
    const child_result_t norow = load("norow", "J1 1.0 0.1\n", coupling, 10, 0, &path);
    CHECK(norow.status == EXIT_FAILURE && norow.err.find(path) != std::string::npos && norow.err.find("no row for coupling node J2") != std::string::npos,
          "a coupling node with no row stops the run, naming the file and the node");
    const child_result_t unreadable = in_child([&]() {
        std::vector<value_t> d(2), l(2);
        load_manhole_table("/nonexistent/manhole_table_for_test.txt", coupling, 10, 0, d, l);
    });
    CHECK(unreadable.status == EXIT_FAILURE && unreadable.err.find("/nonexistent/manhole_table_for_test.txt") != std::string::npos,
          "an unreadable file stops the run, naming the file");
    const child_result_t good = load("good", "J1 1.0 0.1\nj2 1.5 0.2\n", coupling, 10, 0);
    CHECK(good.status == 0 && good.err.empty(), "control: a good table neither exits nor warns");
}

// W1. With a table, rank 0 warns how many table diameters exceed dx and the largest;
// other ranks stay silent; no diameter above dx gives no warning.
// Killed by: warning on every rank, counting rows at or below dx, or naming the
// first rather than the largest.
static void W1_rank_0_warns_how_many_diameters_exceed_dx_and_the_largest()
{
    const std::vector<std::string> coupling = {"J1", "J2", "J3"};
    const std::string text = "J1 0.6 0.1\nJ2 3.5 0.2\nJ3 2.5 0.3\n";
    const child_result_t r0 = load("w0", text, coupling, 1.0, 0);
    CHECK(r0.status == 0, "a table above dx is not refused");
    CHECK(r0.err.find("2 of 3") != std::string::npos, "rank 0 counts the 2 diameters above dx of the 3 rows");
    CHECK(r0.err.find("3.5") != std::string::npos, "rank 0 names the largest diameter");
    const child_result_t r1 = load("w1", text, coupling, 1.0, 1);
    CHECK(r1.status == 0 && r1.err.empty(), "another rank does not warn");
    const child_result_t none = load("w2", text, coupling, 4.0, 0);
    CHECK(none.status == 0 && none.err.empty(), "control: no diameter above dx gives no warning");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"P1_table_with_comments_blank_lines_and_mixed_case_parses", P1_table_with_comments_blank_lines_and_mixed_case_parses},
    {"F1_a_line_that_is_not_a_name_and_two_numbers_is_refused_with_its_line", F1_a_line_that_is_not_a_name_and_two_numbers_is_refused_with_its_line},
    {"F2_a_diameter_not_finite_or_not_above_zero_is_refused_with_its_line", F2_a_diameter_not_finite_or_not_above_zero_is_refused_with_its_line},
    {"F3_a_loss_coefficient_not_finite_or_negative_is_refused_with_its_line", F3_a_loss_coefficient_not_finite_or_negative_is_refused_with_its_line},
    {"F4_a_name_repeated_after_the_case_fold_is_refused_with_its_line", F4_a_name_repeated_after_the_case_fold_is_refused_with_its_line},
    {"F5_a_name_that_is_no_coupling_node_is_refused_with_its_line", F5_a_name_that_is_no_coupling_node_is_refused_with_its_line},
    {"F6_a_coupling_node_with_no_row_is_reported", F6_a_coupling_node_with_no_row_is_reported},
    {"A1_assignment_of_both_columns_follows_the_case_folded_name_in_coupling_order", A1_assignment_of_both_columns_follows_the_case_folded_name_in_coupling_order},
    {"A2_a_table_equal_to_both_constants_is_bit_identical_to_the_constant_loop", A2_a_table_equal_to_both_constants_is_bit_identical_to_the_constant_loop},
    {"L1_every_refusal_stops_the_run_naming_the_file_and_the_line", L1_every_refusal_stops_the_run_naming_the_file_and_the_line},
    {"W1_rank_0_warns_how_many_diameters_exceed_dx_and_the_largest", W1_rank_0_warns_how_many_diameters_exceed_dx_and_the_largest},
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
