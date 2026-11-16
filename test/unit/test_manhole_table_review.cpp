/** @file test_manhole_table_review.cpp
 *  @brief Reviewer test for package 1f: the per-node manhole table reaches the
 *         right node on the right rank, through read_inp_file and
 *         process_swmm_node_locations, not through the helper functions alone.
 *
 *  The author's test (test_manhole_table.cpp) reaches read_manhole_table,
 *  check_manhole_table_names, assign_manhole_table and load_manhole_table. None of
 *  those four sees the order of the table's application relative to the split of
 *  the nodes over the ranks, which is what the approved shape's N13 tests at np8.
 *  This test runs the two member functions in the order initialize() runs them,
 *  once per simulated rank (rank_, size_ and the partition set by hand, no MPI
 *  collective), and checks, for every node a rank owns, that its diameter and its
 *  loss coefficient are the values of the table row bearing that node's name.
 *
 *  `private` is redefined to `public` after the standard headers so the two
 *  private member functions are callable; the code under test is unchanged.
 *
 *  Each test is selectable by name from argv[1]; with no argument all run.
 *  Same build arrangement and include prefix as test_manhole_table.cpp.
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <limits>
#include <memory>
#include <list>
#include <deque>
#include <queue>
#include <stack>
#include <array>
#include <tuple>
#include <utility>
#include <numeric>
#include <iomanip>
#include <iterator>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#define private public
#include "swmm_triton.h"
#undef private

#include <sys/wait.h>
#include <unistd.h>

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

static std::string tmpdir()
{
    const std::string d = (std::filesystem::temp_directory_path() / ("rv_mh_" + std::to_string(static_cast<long>(::getpid())))).string();
    std::filesystem::create_directories(d);
    return d;
}

static std::string write_file(const std::string& name, const std::string& text)
{
    const std::string p = tmpdir() + "/" + name;
    std::ofstream out(p.c_str(), std::ios::binary);
    out << text;
    return p;
}

// Six junctions J1..J6 on a 30 x 10 grid of 10 m cells split over three ranks of ten
// rows each (rank 0: rows 0-9, rank 1: rows 10-19, rank 2: rows 20-29). Row = 29 - k
// for y = 5 + 10 k. J1 row 3 (rank 0), J2 row 12 (rank 1), J3 row 25 (rank 2),
// J4 row 9 (rank 0), J5 row 14 (rank 1), J6 row 29 (rank 2).
static const int kRows = 30, kCols = 10, kNp = 3;
static const double kDx = 10.0;

static std::string make_inp()
{
    const int krow[6] = {3, 12, 25, 9, 14, 29};
    std::string m = "[TITLE]\nreview model\n\n[JUNCTIONS]\n";
    for (int i = 0; i < 6; ++i) m += "J" + std::to_string(i + 1) + " 1.0 2.0\n";
    m += "\n[COORDINATES]\n";
    for (int i = 0; i < 6; ++i) {
        const int k = 29 - krow[i];
        m += "J" + std::to_string(i + 1) + " " + std::to_string(15 + 10 * i) + " " + std::to_string(5 + 10 * k) + "\n";
    }
    m += "\n";
    return write_file("model.inp", m);
}

// The coupling list in an order and spelling that differ from the table's.
static std::string make_coupling_list()
{
    return write_file("coupling.txt", "j5\nJ2\nJ6\nj1\nJ4\nJ3\n");
}

static int node_number(const std::string& name)
{
    std::string u = swmm_upper(name);
    return std::atoi(u.c_str() + 1);   // "J<n>"
}

static double want_d(int n) { return 0.5 + 0.1 * n; }       // 0.6 .. 1.1
static double want_l(int n) { return 0.05 + 0.01 * n; }     // 0.06 .. 0.11 (never equal to the diameter)

static MpiUtils::partition_data_t make_pd()
{
    MpiUtils::partition_data_t pd;
    pd.size = kNp;
    pd.rows = kRows;
    pd.cols = kCols;
    for (int i = 0; i < kNp; ++i) pd.part_dims.push_back(Constants::dims_t(10 + 2 * GHOST_CELL_PADDING, kCols + 2 * GHOST_CELL_PADDING));
    return pd;
}

struct rank_state_t
{
    std::vector<std::string> names;
    std::vector<value_t> d, l;
    std::vector<int> global_index;
};

/** initialize()'s two steps for `rank` of kNp, with `table` ("" = key empty). */
static rank_state_t run_rank(int rank, const std::string& inp, const std::string& coupling, const std::string& table, value_t md, value_t ml)
{
    swmm_triton m;
    m.rank_ = rank;
    m.size_ = kNp;
    m.coupling_nodes_key = coupling;
    m.read_inp_file(inp, static_cast<value_t>(kDx), md, ml, table);
    m.process_swmm_node_locations(0, 0, static_cast<value_t>(kDx), kRows, kCols, make_pd());
    rank_state_t s;
    s.names = m.nodeID;
    s.d = m.diameter;
    s.l = m.loss;
    s.global_index = m.relative_swmm_node_index;
    // the object's destructor and counts/displs members are left as default (never allocated here)
    return s;
}

static std::string full_table_text(const char* eol, bool final_eol)
{
    // order and case differ from the coupling list and the inp
    const int order[6] = {3, 1, 6, 4, 2, 5};
    const char* spell[6] = {"j3", "J1", "J6", "j4", "J2", "J5"};
    std::string t = std::string("; table") + eol + eol;
    for (int i = 0; i < 6; ++i) {
        char row[128];
        std::snprintf(row, sizeof row, "%s %.17g %.17g", spell[i], want_d(order[i]), want_l(order[i]));
        t += row;
        if (i < 5 || final_eol) t += eol;
    }
    return t;
}

static void check_values(const rank_state_t& s, const char* what)
{
    CHECK(!s.names.empty(), what);
    for (std::size_t k = 0; k < s.names.size(); ++k) {
        const int n = node_number(s.names[k]);
        CHECK(s.d[k] == static_cast<value_t>(want_d(n)), "a node's diameter is its own table row's second column");
        CHECK(s.l[k] == static_cast<value_t>(want_l(n)), "a node's loss coefficient is its own table row's third column");
    }
}

// R1. Each rank's local arrays hold, for each node it owns, the values of the table
// row bearing that node's name. Killed by: assignment after the split by the GLOBAL
// index into the LOCAL arrays; the two columns swapped; the loss column ignored (left
// at the constant); only the diameter read.
// Class: every (rank, node) pair of a six-node model over three ranks, with the table's
// row order, spelling case and the coupling list's order all different from one another.
// Required because the arrays are indexed by the global coupling index until
// process_swmm_node_locations hands each node to one rank; an assignment made after that
// split by position gives a node another node's row and still runs deterministically.
// A second correct implementation: assigning by name inside process_swmm_node_locations,
// or through a name-keyed map in read_inp_file; both pass.
static void R1_values_travel_with_their_node_to_the_owning_rank()
{
    const std::string inp = make_inp(), cl = make_coupling_list();
    const std::string tab = write_file("table.txt", full_table_text("\n", true));
    std::size_t total = 0;
    for (int r = 0; r < kNp; ++r) {
        const rank_state_t s = run_rank(r, inp, cl, tab, 1.2, 0.1);
        CHECK(!s.names.empty(), "control: every rank owns at least one node, so the test spans ranks");
        check_values(s, "rank owns nodes");
        total += s.names.size();
    }
    CHECK(total == 6, "control: the six nodes are shared out exactly once");
    // The constants are not read for any node under a table.
    const rank_state_t c = run_rank(1, inp, cl, tab, 7.7, 0.77);
    check_values(c, "constants differing, rank 1");
    for (std::size_t k = 0; k < c.names.size(); ++k)
        CHECK(c.d[k] != static_cast<value_t>(7.7) && c.l[k] != static_cast<value_t>(0.77), "control: the constants did not leak into a table run");
}

// R2. With the key empty every node of every rank carries the two constants, and a
// table equal to the constants gives byte-identical local arrays on every rank.
static void R2_empty_key_and_constant_equal_table_are_byte_identical()
{
    const std::string inp = make_inp(), cl = make_coupling_list();
    const value_t md = static_cast<value_t>(std::atof("1.2")), ml = static_cast<value_t>(std::atof("0.1"));
    std::string text;
    {
        const char* nm[6] = {"J1", "j2", "J3", "J4", "j5", "J6"};
        for (int i = 0; i < 6; ++i) {
            char row[128];
            std::snprintf(row, sizeof row, "%s %.17g %.17g\n", nm[i], static_cast<double>(md), static_cast<double>(ml));
            text += row;
        }
    }
    const std::string tab = write_file("const_table.txt", text);
    for (int r = 0; r < kNp; ++r) {
        const rank_state_t a = run_rank(r, inp, cl, "", md, ml);
        const rank_state_t b = run_rank(r, inp, cl, tab, md, ml);
        CHECK(!a.names.empty() && a.names == b.names, "control: same nodes in the same order");
        for (std::size_t k = 0; k < a.d.size(); ++k) CHECK(a.d[k] == md && a.l[k] == ml, "key empty: both constants on every node");
        CHECK(std::memcmp(a.d.data(), b.d.data(), a.d.size() * sizeof(value_t)) == 0, "diameter arrays byte-identical");
        CHECK(std::memcmp(a.l.data(), b.l.data(), a.l.size() * sizeof(value_t)) == 0, "loss arrays byte-identical");
    }
    const rank_state_t diff = run_rank(0, inp, cl, write_file("t2.txt", full_table_text("\n", true)), md, ml);
    const rank_state_t base = run_rank(0, inp, cl, "", md, ml);
    CHECK(std::memcmp(diff.d.data(), base.d.data(), base.d.size() * sizeof(value_t)) != 0, "control: a table with other values does change the array");
}

// R3. CRLF line endings and a final line with no newline read the same table.
static void R3_crlf_and_missing_final_newline()
{
    const std::string inp = make_inp(), cl = make_coupling_list();
    const std::string crlf = write_file("crlf.txt", full_table_text("\r\n", true));
    const std::string nofinal = write_file("nofinal.txt", full_table_text("\r\n", false));
    const std::string lf_nofinal = write_file("lfnofinal.txt", full_table_text("\n", false));
    for (const std::string& t : {crlf, nofinal, lf_nofinal})
        for (int r = 0; r < kNp; ++r) check_values(run_rank(r, inp, cl, t, 1.2, 0.1), "crlf / no final newline");
    // A final row whose last character is the last digit of the loss coefficient (no eol at all) with a CR kept:
    const std::string cr_only_tail = write_file("crtail.txt", full_table_text("\n", false) + "\r");
    for (int r = 0; r < kNp; ++r) check_values(run_rank(r, inp, cl, cr_only_tail, 1.2, 0.1), "a lone CR after the last row");
}

// R4. strtod consumption of each token.
static void R4_token_consumption()
{
    auto err_of = [](const std::string& t) {
        std::istringstream in(t);
        std::vector<std::string> n; std::vector<value_t> d, l; std::vector<int> ln;
        return read_manhole_table(in, n, d, l, ln);
    };
    CHECK(!err_of("A 1.0 0.1x\n").what.empty(), "trailing junk on the loss is refused");
    CHECK(!err_of("A 1.0e 0.1\n").what.empty(), "a dangling exponent is refused");
    CHECK(!err_of("A 1,5 0.1\n").what.empty(), "a decimal comma is refused");
    CHECK(!err_of("A 1.0 0.1 0.2\n").what.empty(), "a fourth token is refused");
    CHECK(!err_of("A 1.0\n").what.empty(), "a one-value row is refused");
    CHECK(err_of("A +1.0 .5\n").what.empty(), "control: a signed and a leading-dot number pass");
    CHECK(err_of("A 1e0 0E0\n").what.empty(), "control: exponent forms and a zero loss pass");
    CHECK(!err_of("A 1.0 -0.001\n").what.empty(), "a negative loss is refused");
    CHECK(!err_of("A 1.0 nan\n").what.empty() && !err_of("A 1.0 infinity\n").what.empty(), "nan and infinity are refused");
    // an observation, asserted so a change is seen: a hexadecimal token is consumed by strtod
    CHECK(err_of("A 0x10 0.1\n").what.empty(), "observation: a hexadecimal diameter parses (strtod accepts it)");
}

// R5. Every refusal through read_inp_file exits on every rank with a failure status
// and prints the file; none returns. The exits are checked per rank.
// Class: the three refusal families (malformed row, coupling node with no row, unreadable
// file) at each of ranks 0, 1 and 2. Required because every rank reads the same file and a
// rank that returns from a refusal runs on at the constants while another has exited, the
// state that leaves a collective unmatched. Killed by: a refusal that exits on rank 0 only.
// A second correct implementation: MPI_Abort on every rank instead of exit; the status
// would differ, so the test pins exit(EXIT_FAILURE), the coupling list's own form.
static int run_child_status(const std::function<void()>& fn, std::string& err)
{
    int fd[2];
    if (::pipe(fd) != 0) return -1;
    std::cerr.flush();
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::close(fd[0]); ::dup2(fd[1], 2); ::close(fd[1]);
        fn();
        std::cerr.flush();
        ::_exit(0);
    }
    ::close(fd[1]);
    char buf[1024]; ssize_t n;
    while ((n = ::read(fd[0], buf, sizeof buf)) > 0) err.append(buf, static_cast<std::size_t>(n));
    ::close(fd[0]);
    int st = 0; ::waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 255;
}

static void R5_every_rank_exits_on_a_refusal()
{
    const std::string inp = make_inp(), cl = make_coupling_list();
    const std::string bad = write_file("bad.txt", "J1 1.0 0.1\nJ2 1.0\n");
    const std::string miss = write_file("miss.txt", "J1 1.0 0.1\nJ2 1.0 0.1\n");
    for (int r = 0; r < kNp; ++r) {
        std::string e1, e2, e3;
        CHECK(run_child_status([&]() { run_rank(r, inp, cl, bad, 1.2, 0.1); }, e1) == EXIT_FAILURE, "malformed row: this rank exits");
        CHECK(e1.find(bad) != std::string::npos && e1.find("line 2") != std::string::npos, "names the file and the line");
        CHECK(run_child_status([&]() { run_rank(r, inp, cl, miss, 1.2, 0.1); }, e2) == EXIT_FAILURE, "missing rows: this rank exits");
        CHECK(e2.find(miss) != std::string::npos && e2.find("no row for coupling node") != std::string::npos, "names the file and a node");
        CHECK(run_child_status([&]() { run_rank(r, inp, cl, "/nonexistent/rv_table.txt", 1.2, 0.1); }, e3) == EXIT_FAILURE, "unreadable: this rank exits");
    }
    // a directory as the path: observe the outcome
    std::string ed;
    const int sd = run_child_status([&]() { run_rank(0, inp, cl, tmpdir(), 1.2, 0.1); }, ed);
    std::fprintf(stderr, "  observation: a directory as the table path exits %d; stderr: %s\n", sd, ed.c_str());
    CHECK(sd == EXIT_FAILURE, "a directory as the table path stops the run");
}

// R6. The key-empty path keeps the constant fill's warning, on rank 0 only, and a
// table run does not print it.
// Class: key empty or set, rank 0 or 1, constant above dx. Required because with the key
// empty the only change of behaviour the package may make is none, and the warning text is
// the one output of that loop. The assertion turns on a string the original code and the
// changed code both carry. A second correct implementation: the warning moved into a helper
// that the loop calls; it passes.
static void R6_constant_warning_unchanged_and_absent_under_a_table()
{
    const std::string inp = make_inp(), cl = make_coupling_list();
    const std::string tab = write_file("w.txt", full_table_text("\n", true));
    const char* w = "Manhole diameter is greater than grid resolution. This may cause numerical instabilities";
    std::string e0, e1, e2;
    CHECK(run_child_status([&]() { run_rank(0, inp, cl, "", 25.0, 0.1); }, e0) == 0 && e0.find(w) != std::string::npos, "key empty, constant above dx: rank 0 warns");
    CHECK(run_child_status([&]() { run_rank(1, inp, cl, "", 25.0, 0.1); }, e1) == 0 && e1.find(w) == std::string::npos, "key empty: rank 1 is silent");
    CHECK(run_child_status([&]() { run_rank(0, inp, cl, tab, 25.0, 0.1); }, e2) == 0 && e2.find(w) == std::string::npos, "table set: the constant's warning is not printed");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"R1_values_travel_with_their_node_to_the_owning_rank", R1_values_travel_with_their_node_to_the_owning_rank},
    {"R2_empty_key_and_constant_equal_table_are_byte_identical", R2_empty_key_and_constant_equal_table_are_byte_identical},
    {"R3_crlf_and_missing_final_newline", R3_crlf_and_missing_final_newline},
    {"R4_token_consumption", R4_token_consumption},
    {"R5_every_rank_exits_on_a_refusal", R5_every_rank_exits_on_a_refusal},
    {"R6_constant_warning_unchanged_and_absent_under_a_table", R6_constant_warning_unchanged_and_absent_under_a_table},
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
    if (want && ran == 0) { std::fprintf(stderr, "ERROR: no test named '%s'\n", want); return 2; }
    std::fprintf(stderr, "%d test(s), %d check(s), %d failure(s)\n", ran, g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
