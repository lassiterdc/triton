/** @file test_perf_identity.cpp
 *  @brief Runtime construction check for the SWMM timer column split (WP-1A
 *         chunk 4): XFER + MPI + STEP + OTHER == SWMM on every emitted per-rank
 *         row, and SWMM_STEP is EXACTLY 0.0 on every rank that is not rank 0.
 *
 *  No solver run, no cluster, no SWMM call, no GPU.  These drive the REAL
 *  `SuperTimer::super_timer` and the REAL `Output::output<T>::write_times`, then
 *  parse the file write_times actually emitted.  That matters: a test that
 *  re-implemented the arithmetic would agree with itself no matter what
 *  write_times did, and the whole defect this package repairs is a number that
 *  looked right and measured the wrong span.
 *
 *  THE TIMER PATTERN UNDER TEST is the one triton.h performs, reproduced here in
 *  the same order and with the same nesting:
 *
 *      st.start(SWMM_TIME);
 *        <residual work -- the two sizeof-scaled assignments>
 *        st.start(SWMM_XFER);  <transfers + kernel>   st.stop(SWMM_XFER);
 *        st.start(SWMM_MPI);   <MPI_Gatherv>          st.stop(SWMM_MPI);
 *        if (rank == 0) {                             <- the bracket is INSIDE
 *          st.start(SWMM_STEP); <serial solve>        st.stop(SWMM_STEP);
 *        }
 *        st.start(SWMM_MPI);   <MPI_Scatterv>         st.stop(SWMM_MPI);
 *      st.stop(SWMM_TIME);
 *
 *  WHY EXACTLY 0.0 RATHER THAN MERELY SMALL, on a non-rank-0 rank.  The claim is
 *  not that the solve is fast there; it is that nothing is measured at all.
 *  `super_timer::get_cat_index_` REGISTERS an absent category rather than
 *  failing, and `add_new_timer` pushes `Constants::ull newcat_time = 0`.  A rank
 *  that never executes the bracket therefore reads a value that was never
 *  incremented -- bitwise zero, not a rounded-down small number.  P4 asserts
 *  that as exact equality against 0.0 and P5 re-asserts it on the EMITTED text,
 *  because the double and the printed cell are two different artifacts and a
 *  reader of performance.txt only ever sees the second.
 *
 *  THE TOLERANCE IS DERIVED, NOT CHOSEN.  write_times emits through
 *  `std::setprecision(4)` in the default float format -- four SIGNIFICANT
 *  digits.  A printed value differs from its double by at most half a unit in
 *  the 4th significant digit, i.e. a RELATIVE error of 5e-4.  The identity is
 *  exact in the emitting doubles (SWMM_OTHER is derived by subtraction), so what
 *  the file can show is the identity plus five roundings:
 *
 *      |sum(children) - parent| <= 5e-4 * (sum|children| + |parent|)
 *
 *  which is 1e-3 * parent -- 0.1 % -- whenever the children are non-negative and
 *  sum to the parent.  An ABSOLUTE epsilon is wrong here on a measured ground:
 *  at four significant digits the representable error scales with the value, so
 *  a fixed epsilon is vacuous on a long run and false-failing on a short one.
 *
 *  Each test function is selectable by name from argv[1] so CTest can register
 *  one node id per property; with no argument every test runs.
 */

// Neither supertimer.h nor output.h is independently includable at this commit,
// and the gaps are MEASURED from compile failures rather than assumed:
//   <sys/time.h>  -- supertimer.h calls gettimeofday without including it.
//   <unistd.h>    -- supertimer.h calls gethostname without including it.
//   <fstream>     -- matrix.h (reached via output.h) declares std::ifstream
//                    members and calls std::getline without including it.
//   <string>      -- matrix.h names std::string likewise.
//   <iostream>    -- matrix.h uses std::cerr/std::endl.
//   "mpi.h"       -- constants.h and Ensify.h name MPI_DOUBLE / MPI_COMM_WORLD.
//   "mpi_utils.h" -- output.h declares write_domain_decomposition taking a
//                    MpiUtils::partition_data_t without including it.
// This prefix reproduces main.cpp's include chain so those gaps stay OUT of this
// test's scope: none of supertimer.h, matrix.h or their include sites is WP-1A's
// to edit, and they are reported as findings instead. The same shape and the
// same rationale appear at the head of test/unit/test_exchange_log_header.cpp.
#include <iostream>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"
#include "mpi_utils.h"

#include "supertimer.h"
#include "output.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal harness (same shape as test/unit and test/snapshot)
// ---------------------------------------------------------------------------
static int g_checks = 0;
static int g_failures = 0;
static int g_rank = 0;
static int g_size = 1;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

// Half a unit in the 4th significant digit, as a relative bound. Derived from
// std::setprecision(4) in write_times; see the file header.
static const double kRelPrintError = 5.0e-4;

// ---------------------------------------------------------------------------
// Emitted-file model
// ---------------------------------------------------------------------------
struct perf_file {
    std::vector<std::string> cols;
    std::vector<std::vector<std::string> > rows;   // includes the Average row

    int col(const std::string& name) const {
        for (std::size_t i = 0; i < cols.size(); ++i)
            if (cols[i] == name) return (int)i;
        return -1;
    }
};

static std::string trim(const std::string& s)
{
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(trim(cell));
    return out;
}

static bool read_perf_file(const std::string& path, perf_file& pf, std::string& err)
{
    std::ifstream in(path.c_str());
    if (!in) { err = "cannot open " + path; return false; }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        if (first) {
            std::string h = trim(line);
            if (h.size() && h[0] == '%') h = h.substr(1);
            pf.cols = split_csv(h);
            first = false;
            continue;
        }
        pf.rows.push_back(split_csv(line));
    }
    if (first) { err = path + " is empty"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Driving the timer in triton.h's exact pattern, then emitting through the real
// write_times. Returns the emitted file's path (valid on rank 0).
// ---------------------------------------------------------------------------
static void busy_us(long micros)
{
    // usleep rather than a spin: this is measuring WALL time through
    // gettimeofday, which is what super_timer uses, so sleeping is a faithful
    // stand-in for blocked work such as an MPI collective or a device sync.
    if (micros > 0) usleep((useconds_t)micros);
}

static std::string emit_performance_file(SuperTimer::super_timer& st,
                                         const std::string& dir)
{
    // The three enclosing timers, so the emitted row is well formed and the
    // pre-existing Simulation/Total levels still close.
    st.start(TOTAL_TIME);
    st.start(SIMULATION_TIME);
    st.start(COMPUTE_TIME);
    busy_us(2000);
    st.stop(COMPUTE_TIME);

    // ---- the coupled block, in triton.h's order -------------------------
    st.start(SWMM_TIME);

    // Residual work: the two sizeof-scaled assignments at the top of the block,
    // plus the branch test below. Given a measurable duration on purpose, so a
    // residual column that silently collapsed to zero would be visible.
    busy_us(1500);

    st.start(SWMM_XFER);
    busy_us(3000);
    st.stop(SWMM_XFER);

    st.start(SWMM_MPI);
    busy_us(1000);
    st.stop(SWMM_MPI);

    if (g_rank == 0) {
        st.start(SWMM_STEP);
        busy_us(5000);          // the serial solve -- large, so its absence on
        st.stop(SWMM_STEP);     // other ranks is unmistakable
    }

    st.start(SWMM_MPI);
    busy_us(1000);
    st.stop(SWMM_MPI);

    st.stop(SWMM_TIME);
    // ---------------------------------------------------------------------

    st.start(IO_TIME);
    busy_us(500);
    st.stop(IO_TIME);

    st.stop(SIMULATION_TIME);
    st.stop(TOTAL_TIME);

    Output::output<double> out;
    // 4x4 interior grid: write_times touches none of it, but init allocates the
    // gather buffers and sets rank_/size_/project_dir_/output_folder_, which
    // write_times does read.
    out.init(4, 4, 0.0, 0.0, 1.0, g_rank, g_size, dir, "", "%s/%s/%s_%02d_%02d",
             0, "", "SEQ");
    out.write_times(st, -1);          // -1 => the final-state performance.txt
    MPI_Barrier(MPI_COMM_WORLD);
    return dir + "//performance.txt";
}

static std::string make_tmpdir()
{
    // One directory for the whole communicator: rank 0 writes the file and every
    // rank must agree on the path.
    char buf[256];
    if (g_rank == 0)
        std::snprintf(buf, sizeof(buf), "/tmp/wp1a_perf_%d", (int)getpid());
    MPI_Bcast(buf, sizeof(buf), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (g_rank == 0) {
        std::string cmd = std::string("mkdir -p ") + buf;
        if (std::system(cmd.c_str()) != 0)
            std::fprintf(stderr, "  WARN: could not create %s\n", buf);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return std::string(buf);
}

static void cleanup(const std::string& dir)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (g_rank == 0) {
        std::string cmd = std::string("rm -rf ") + dir;
        if (std::system(cmd.c_str()) != 0) { /* best effort */ }
    }
}

// ---------------------------------------------------------------------------
// P1 -- the emitted header carries the parent followed by its four children
// ---------------------------------------------------------------------------
static void P1_header_carries_four_child_columns()
{
    SuperTimer::super_timer st;
    const std::string dir = make_tmpdir();
    const std::string path = emit_performance_file(st, dir);

    if (g_rank == 0) {
        perf_file pf; std::string err;
        if (!read_perf_file(path, pf, err)) { CHECK(false, err.c_str()); cleanup(dir); return; }

        const char* want[] = { "SWMM", "SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER" };
        for (int i = 0; i < 5; ++i) {
            const bool have = pf.col(want[i]) >= 0;
            if (!have) std::fprintf(stderr, "  missing column: %s\n", want[i]);
            CHECK(have, "the emitted header must carry the parent and all four children");
        }
        const int p = pf.col("SWMM");
        CHECK(p >= 0 && pf.col("SWMM_XFER")   == p + 1 &&
                        pf.col("SWMM_MPI")    == p + 2 &&
                        pf.col("SWMM_STEP")   == p + 3 &&
                        pf.col("SWMM_OTHER")  == p + 4,
              "the four children must immediately follow SWMM, so a reader meets "
              "the whole before the parts");
        // Arity is fixed and load-bearing downstream: the consumer's
        // Average-presence detector keys on the row shape.
        for (std::size_t r = 0; r < pf.rows.size(); ++r)
            CHECK(pf.rows[r].size() == pf.cols.size(),
                  "every row must have the header's arity");
    }
    cleanup(dir);
}

// ---------------------------------------------------------------------------
// P2 -- THE CONSTRUCTION CHECK, on every emitted per-rank row
// ---------------------------------------------------------------------------
static void P2_identity_closes_on_every_per_rank_row()
{
    SuperTimer::super_timer st;
    const std::string dir = make_tmpdir();
    const std::string path = emit_performance_file(st, dir);

    if (g_rank == 0) {
        perf_file pf; std::string err;
        if (!read_perf_file(path, pf, err)) { CHECK(false, err.c_str()); cleanup(dir); return; }

        const int cP = pf.col("SWMM");
        const int cX = pf.col("SWMM_XFER");
        const int cM = pf.col("SWMM_MPI");
        const int cS = pf.col("SWMM_STEP");
        const int cO = pf.col("SWMM_OTHER");
        if (cP < 0 || cX < 0 || cM < 0 || cS < 0 || cO < 0) {
            CHECK(false, "columns missing; P1 explains");
            cleanup(dir); return;
        }

        int per_rank = 0, nondegenerate = 0;
        for (std::size_t r = 0; r < pf.rows.size(); ++r) {
            if (pf.rows[r][0] == "Average") continue;   // OUTSIDE the claim
            ++per_rank;

            const double parent = std::atof(pf.rows[r][cP].c_str());
            const double kids[4] = { std::atof(pf.rows[r][cX].c_str()),
                                     std::atof(pf.rows[r][cM].c_str()),
                                     std::atof(pf.rows[r][cS].c_str()),
                                     std::atof(pf.rows[r][cO].c_str()) };
            double sum = 0.0, absum = 0.0;
            for (int k = 0; k < 4; ++k) { sum += kids[k]; absum += std::fabs(kids[k]); }
            const double residual = sum - parent;
            const double bound = kRelPrintError * (absum + std::fabs(parent));
            if (parent != 0.0) ++nondegenerate;

            if (std::fabs(residual) > bound) {
                std::fprintf(stderr,
                    "  rank %s: SWMM=%.6g children sum=%.6g residual=%+.6g "
                    "bound=%.6g\n", pf.rows[r][0].c_str(), parent, sum,
                    residual, bound);
            }
            CHECK(std::fabs(residual) <= bound,
                  "XFER + MPI + STEP + OTHER must equal SWMM on this per-rank "
                  "row, within the bound derived from setprecision(4)");
        }
        CHECK(per_rank == g_size, "one per-rank row per rank");
        // Disclosed denominator: a row whose SWMM fields are all zero passes the
        // identity on nothing, and "examined N, all trivial" must not read the
        // same as "examined N, all meaningful".
        std::fprintf(stderr, "  per-rank rows: %d examined, %d non-degenerate\n",
                     per_rank, nondegenerate);
        CHECK(nondegenerate == per_rank,
              "every row must be non-degenerate here; this harness drives real "
              "time into every span");
    }
    cleanup(dir);
}

// ---------------------------------------------------------------------------
// P3 -- the Average row is PRESENT with full arity, and is outside the claim
// ---------------------------------------------------------------------------
static void P3_average_row_present_with_full_arity()
{
    SuperTimer::super_timer st;
    const std::string dir = make_tmpdir();
    const std::string path = emit_performance_file(st, dir);

    if (g_rank == 0) {
        perf_file pf; std::string err;
        if (!read_perf_file(path, pf, err)) { CHECK(false, err.c_str()); cleanup(dir); return; }

        int n_avg = 0; std::size_t avg = 0;
        for (std::size_t r = 0; r < pf.rows.size(); ++r)
            if (pf.rows[r][0] == "Average") { ++n_avg; avg = r; }

        CHECK(n_avg == 1, "exactly one Average row");
        if (n_avg != 1) { cleanup(dir); return; }
        CHECK(pf.rows[avg].size() == pf.cols.size(),
              "the Average row's arity is FIXED -- suppressing one column's "
              "average was rejected by name, because the downstream parser's "
              "Average-presence detector keys on the row shape");
        for (std::size_t c = 0; c < pf.rows[avg].size(); ++c)
            CHECK(!pf.rows[avg][c].empty(), "no empty Average cell");

        // Asserted so the property is recorded rather than merely believed: the
        // Average of SWMM_STEP is rank0/N and is NOT the serial-solve cost. On
        // size_ > 1 it is strictly below the rank-0 value; that is the shape the
        // design accepts, and the reason the identity claim is scoped to the
        // per-rank rows.
        const int cS = pf.col("SWMM_STEP");
        if (cS >= 0 && g_size > 1) {
            double rank0 = 0.0;
            for (std::size_t r = 0; r < pf.rows.size(); ++r)
                if (pf.rows[r][0] == "0") rank0 = std::atof(pf.rows[r][cS].c_str());
            const double mean = std::atof(pf.rows[avg][cS].c_str());
            std::fprintf(stderr,
                "  Average(SWMM_STEP)=%.6g vs rank0=%.6g over %d ranks "
                "(mean is rank0/N by construction, NOT the serial-solve cost)\n",
                mean, rank0, g_size);
            CHECK(mean < rank0,
                  "Average(SWMM_STEP) is rank0/N and must be strictly below the "
                  "rank-0 value on a multi-rank run");
        }
    }
    cleanup(dir);
}

// ---------------------------------------------------------------------------
// P4 -- SWMM_STEP is EXACTLY 0.0 on a non-rank-0 rank, in the timer
// ---------------------------------------------------------------------------
static void P4_step_is_exactly_zero_off_rank0_in_the_timer()
{
    SuperTimer::super_timer st;
    const std::string dir = make_tmpdir();
    emit_performance_file(st, dir);

    const double step = st.get_custom_time(SWMM_STEP);
    if (g_rank != 0) {
        // Exact, not approximate. add_new_timer pushes a zero-initialised
        // accumulator and nothing on this rank ever increments it, so the value
        // is bitwise zero rather than a small measured time. The `== 0.0`
        // comparison is deliberate here for that reason.
        std::fprintf(stderr, "  rank %d: SWMM_STEP = %.17g\n", g_rank, step);
        CHECK(step == 0.0,
              "a rank that never enters the rank-0 guard must report EXACTLY "
              "0.0 for SWMM_STEP -- not a small time for a branch test it "
              "evaluated and a solve it never performed");
    } else {
        CHECK(step > 0.0, "rank 0 must have measured the serial solve");
    }

    // The parent must still contain the child everywhere, including on the ranks
    // where the child is zero.
    const double parent = st.get_custom_time(SWMM_TIME);
    const double xfer = st.get_custom_time(SWMM_XFER);
    const double mpi = st.get_custom_time(SWMM_MPI);
    const double other = parent - xfer - mpi - step;
    CHECK(other >= 0.0,
          "the derived residual must be non-negative: every child bracket is "
          "nested inside the parent, so their sum cannot exceed it");
    std::fprintf(stderr,
        "  rank %d: SWMM=%.6g XFER=%.6g MPI=%.6g STEP=%.6g OTHER=%.6g\n",
        g_rank, parent, xfer, mpi, step, other);

    cleanup(dir);
}

// ---------------------------------------------------------------------------
// P5 -- SWMM_STEP prints as exact zero on every non-rank-0 emitted row
// ---------------------------------------------------------------------------
static void P5_step_prints_as_zero_off_rank0()
{
    SuperTimer::super_timer st;
    const std::string dir = make_tmpdir();
    const std::string path = emit_performance_file(st, dir);

    if (g_rank == 0) {
        perf_file pf; std::string err;
        if (!read_perf_file(path, pf, err)) { CHECK(false, err.c_str()); cleanup(dir); return; }
        const int cS = pf.col("SWMM_STEP");
        if (cS < 0) { CHECK(false, "no SWMM_STEP column"); cleanup(dir); return; }

        // The double and the printed cell are two artifacts and a reader of
        // performance.txt only ever sees the second, so P4's assertion is
        // re-made here on the text. setprecision(4) renders exact 0.0 as "0".
        for (std::size_t r = 0; r < pf.rows.size(); ++r) {
            if (pf.rows[r][0] == "Average" || pf.rows[r][0] == "0") continue;
            std::fprintf(stderr, "  emitted rank %s SWMM_STEP cell = %s\n",
                         pf.rows[r][0].c_str(), pf.rows[r][cS].c_str());
            CHECK(pf.rows[r][cS] == "0",
                  "a non-rank-0 row must print SWMM_STEP as exact zero");
        }
    }
    cleanup(dir);
}

// ---------------------------------------------------------------------------
// P6 -- the parent's value is unchanged by the split
// ---------------------------------------------------------------------------
static void P6_parent_equals_an_unsplit_measurement()
{
    // R4's architectural claim is that SWMM keeps reporting the reading it
    // reported before the split, because :2425/:2464 are untouched. Expressed
    // executably: a parent bracket around the SAME work, with and without the
    // three inner brackets, must agree to within the instrumentation cost --
    // which is bounded here well below the work itself.
    SuperTimer::super_timer a, b;

    a.start(SWMM_TIME);
    busy_us(1500);
    busy_us(3000);
    busy_us(1000);
    if (g_rank == 0) busy_us(5000);
    busy_us(1000);
    a.stop(SWMM_TIME);

    b.start(SWMM_TIME);
    busy_us(1500);
    b.start(SWMM_XFER); busy_us(3000); b.stop(SWMM_XFER);
    b.start(SWMM_MPI);  busy_us(1000); b.stop(SWMM_MPI);
    if (g_rank == 0) { b.start(SWMM_STEP); busy_us(5000); b.stop(SWMM_STEP); }
    b.start(SWMM_MPI);  busy_us(1000); b.stop(SWMM_MPI);
    b.stop(SWMM_TIME);

    const double unsplit = a.get_custom_time(SWMM_TIME);
    const double split = b.get_custom_time(SWMM_TIME);
    const double rel = std::fabs(split - unsplit) / (unsplit > 0 ? unsplit : 1.0);
    std::fprintf(stderr, "  rank %d: unsplit SWMM=%.6g split SWMM=%.6g rel=%.4f\n",
                 g_rank, unsplit, split, rel);
    // 10 %: generous against scheduler jitter on a shared login node, and still
    // two orders of magnitude tighter than the defect this package repairs,
    // where the SWMM column absorbed a whole timestep of GPU work.
    CHECK(rel < 0.10,
          "adding the inner brackets must not move the parent's reading beyond "
          "the instrumentation's own cost");
}

// ---------------------------------------------------------------------------
// Registry / driver
// ---------------------------------------------------------------------------
struct test_entry { const char* name; void (*fn)(); };

static const test_entry kTests[] = {
    { "P1_header_carries_four_child_columns",        &P1_header_carries_four_child_columns },
    { "P2_identity_closes_on_every_per_rank_row",    &P2_identity_closes_on_every_per_rank_row },
    { "P3_average_row_present_with_full_arity",      &P3_average_row_present_with_full_arity },
    { "P4_step_is_exactly_zero_off_rank0_in_the_timer", &P4_step_is_exactly_zero_off_rank0_in_the_timer },
    { "P5_step_prints_as_zero_off_rank0",            &P5_step_prints_as_zero_off_rank0 },
    { "P6_parent_equals_an_unsplit_measurement",     &P6_parent_equals_an_unsplit_measurement },
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    const char* want = (argc > 1) ? argv[1] : NULL;
    int ran = 0;

    for (std::size_t i = 0; i < sizeof(kTests) / sizeof(kTests[0]); ++i) {
        if (want && std::strcmp(want, kTests[i].name) != 0) continue;
        if (g_rank == 0) std::fprintf(stderr, "RUN  %s\n", kTests[i].name);
        const int before = g_failures;
        kTests[i].fn();
        if (g_rank == 0)
            std::fprintf(stderr, "%s %s\n",
                         (g_failures == before) ? "PASS" : "FAIL", kTests[i].name);
        ++ran;
    }

    if (want && ran == 0) {
        if (g_rank == 0) std::fprintf(stderr, "ERROR: no test named '%s'\n", want);
        MPI_Finalize();
        return 2;
    }

    // Any rank's failure fails the node id: a property that holds only on rank 0
    // is not the property, and P4/P5 are specifically about the other ranks.
    int local = g_failures, total = 0;
    MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0)
        std::fprintf(stderr, "%d test(s), %d check(s) on rank 0, %d failure(s) "
                             "across %d rank(s)\n", ran, g_checks, total, g_size);
    MPI_Finalize();
    return (total == 0) ? 0 : 1;
}
