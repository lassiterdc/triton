/** @file test_coupling_interval.cpp
 *  @brief Tier-1 tests for the coupling interval and lag bookkeeping.
 *
 *  No solver run, no cluster, no SWMM call. Two cfg keys, swmm_coupling_interval
 *  (TRITON steps per SWMM step) and swmm_coupling_lag (0 or 1), change how often
 *  SWMM is stepped and when its depths reach the surface. The bookkeeping that
 *  decides this is the pure struct coupling_interval_t in swmm_triton.h:
 *
 *    accumulate / interval_mean / interval_dt -- what SWMM is handed per step;
 *    step_due / clear_interval                -- when SWMM steps;
 *    settle                                   -- when the returned depths apply;
 *    carries_state                            -- what a checkpoint would lose;
 *    coupling_interval_cfg_error              -- the cfg range refusal.
 *
 *  triton.h owns the collectives and the swmm_step call; this file does not reach
 *  them. Each test function is selectable by name from argv[1] so CTest registers
 *  one node id per property; with no argument every test runs. These are chunk
 *  tests of the coupling interval chunk (package 1d); the module whose behaviour
 *  they guard is swmm_triton.h's coupling_interval_t.
 */

// The same include prefix as test_coupling_declaration.cpp, for the reasons
// test_exchange_log_header.cpp states: swmm_triton.h compiles only after
// main.cpp's include chain.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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

/** Drives one coupled node through `steps` TRITON steps of dt 1, in the order the
 *  coupling block of triton.h does (accumulate, decide, maybe step, settle), with
 *  SWMM's depth after its k-th step equal to k. Returns the depth the held
 *  new_depth reads at the START of each TRITON step, which is the depth that step's
 *  exchange sees. drain_at is the 1-based step that ends a print interval (0 for none). */
static std::vector<value_t> depths_seen(int interval, int lag, int steps, int drain_at)
{
    coupling_interval_t c;
    c.configure(interval, lag);
    value_t held = (value_t)0.0;
    int swmm_steps = 0;
    std::vector<value_t> seen;
    for (int s = 1; s <= steps; ++s)
    {
        seen.push_back(held);
        const value_t q = (value_t)1.0;
        c.accumulate(&q, 1, (value_t)1.0);
        const bool drain = (s == drain_at);
        const bool stepped = c.step_due(drain);
        if (stepped)
        {
            ++swmm_steps;
            c.staged_depths(1)[0] = (value_t)swmm_steps;
            c.clear_interval();
        }
        c.settle(&held, 1, stepped, drain);
    }
    return seen;
}

// ---------------------------------------------------------------------------
// Property: only a key that departs from its default selects the interval/lag
// block; the defaults leave triton.h on its unchanged block.
// Class: (interval, lag) pairs at the default, each key alone off its default, and
// both off. Why: active() is the one test that routes a step to the new block, so a
// false positive at the defaults would change every coupled run.
// Killed by: active() returning true for (1, 0), or ignoring lag. Second correct
// implementation: comparing the pair to the defaults as a std::pair.
static void I1_defaults_are_inactive()
{
    coupling_interval_t c;
    CHECK(!c.active(), "a default-constructed state is inactive");
    c.configure(1, 0);
    CHECK(!c.active(), "(1, 0) is inactive");
    c.configure(2, 0);
    CHECK(c.active(), "interval 2 alone is active");
    c.configure(1, 1);
    CHECK(c.active(), "lag 1 alone is active");
    c.configure(4, 1);
    CHECK(c.active(), "both keys off their defaults is active");
}

// ---------------------------------------------------------------------------
// Property: SWMM is handed the dt-weighted mean exchange and the summed dt, so the
// volume it receives equals the sum of the volumes the surface applied.
// Class: intervals of unequal dt on two nodes with different q per step. Why: the
// surface applies each step's q*dt; a SWMM step that received an unweighted mean or
// the wrong dt would receive a different volume than the surface gave up.
// Killed by: an unweighted mean (control below: it differs by more than 0.1 here).
// Second correct implementation: accumulating the volume and dividing by the
// summed dt at the step, rather than keeping a running weighted mean.
static void I2_interval_mean_is_dt_weighted()
{
    coupling_interval_t c;
    c.configure(3, 0);
    const value_t q1[2] = {(value_t)2.0, (value_t)10.0};
    const value_t q2[2] = {(value_t)4.0, (value_t)0.0};
    const value_t q3[2] = {(value_t)6.0, (value_t)20.0};
    c.accumulate(q1, 2, (value_t)1.0);
    c.accumulate(q2, 2, (value_t)3.0);
    c.accumulate(q3, 2, (value_t)2.0);
    CHECK(std::fabs((double)c.interval_dt() - 6.0) < 1e-6, "summed dt is 1 + 3 + 2");
    const value_t* m = c.interval_mean(2);
    const double want0 = (2.0 * 1.0 + 4.0 * 3.0 + 6.0 * 2.0) / 6.0;
    const double want1 = (10.0 * 1.0 + 0.0 * 3.0 + 20.0 * 2.0) / 6.0;
    CHECK(std::fabs((double)m[0] - want0) < 1e-5, "node 0 mean is the dt-weighted mean");
    CHECK(std::fabs((double)m[1] - want1) < 1e-5, "node 1 mean is the dt-weighted mean");
    CHECK(std::fabs((double)m[0] - (2.0 + 4.0 + 6.0) / 3.0) > 0.1,
          "control: the unweighted mean of node 0 differs from the weighted one on this input");
    CHECK(std::fabs((double)m[0] * 6.0 - (2.0 * 1.0 + 4.0 * 3.0 + 6.0 * 2.0)) < 1e-4,
          "mean times summed dt equals the volume the surface applied");
}

// ---------------------------------------------------------------------------
// Property: an interval of one TRITON step hands SWMM that step's exchange_q
// unchanged, so interval 1 with lag 1 differs from the default coupling only by
// the lag.
// Class: (q, dt) pairs, including one on which (q*dt)/dt != q in floating point.
// Why: a mean formed as (q*dt)/dt would move SWMM's input by one ulp on such pairs.
// Killed by: forming the mean by division for a one-step interval. Second correct
// implementation: special-casing a one-step interval by returning the stored q.
static void I3_one_step_interval_passes_q_unchanged()
{
    bool found_inexact = false;
    for (int k = 1; k < 400 && !found_inexact; ++k)
    {
        const value_t dt = (value_t)(0.0173 * k);
        const value_t q = (value_t)0.1;
        coupling_interval_t c;
        c.configure(1, 1);
        c.accumulate(&q, 1, dt);
        const value_t m = c.interval_mean(1)[0];
        CHECK(m == q, "a one-step interval returns q bit-for-bit");
        const value_t divided = (value_t)(((double)q * (double)dt) / (double)dt);
        if (divided != q) found_inexact = true;
    }
    // Control: in double precision the search must find a pair on which dividing
    // would have changed q, or the loop above never separated the two designs.
    if (sizeof(value_t) == sizeof(double))
        CHECK(found_inexact, "control: some (q, dt) pair makes (q*dt)/dt differ from q");
}

// ---------------------------------------------------------------------------
// Property: SWMM steps when N TRITON steps have accumulated or a drain forces it,
// and a cleared interval starts from nothing.
// Class: N = 3 with no drain, a drain on the first step, and N = 1.
// Why: this decision picks the steps that run the gather and scatter; it must be
// the same on every rank, and it is the only thing that moves a SWMM step.
// Killed by: off-by-one on N, ignoring the drain, or a clear that keeps the sum.
// Second correct implementation: counting down from N instead of up to N.
static void I4_step_due_every_n_or_on_drain()
{
    coupling_interval_t c;
    c.configure(3, 0);
    const value_t q = (value_t)1.0;
    c.accumulate(&q, 1, (value_t)1.0);
    CHECK(!c.step_due(false), "one step of three: not due");
    CHECK(c.step_due(true), "one step of three with a drain: due");
    c.accumulate(&q, 1, (value_t)1.0);
    CHECK(!c.step_due(false), "two steps of three: not due");
    c.accumulate(&q, 1, (value_t)1.0);
    CHECK(c.step_due(false), "three steps of three: due");
    c.clear_interval();
    CHECK(c.steps == 0 && c.acc_dt == 0.0, "clear_interval empties the count and the dt sum");
    CHECK(!c.step_due(false), "a cleared interval is not due");

    coupling_interval_t one;
    one.configure(1, 1);
    one.accumulate(&q, 1, (value_t)1.0);
    CHECK(one.step_due(false), "interval 1: due every step");
}

// ---------------------------------------------------------------------------
// Property: lag 1 applies the depths SWMM returned after TRITON step k from step
// k + 2 rather than k + 1; lag 0 applies them from k + 1.
// Class: interval 1, five steps, SWMM's depth after its k-th step equal to k.
// Why: the experiment's counterfactual is an exchange that reads SWMM one coupling
// interval behind; a wrong delay changes what the experiment measures.
// Killed by: no delay (lag 1 reads like lag 0) or a two-step delay. Second correct
// implementation: a two-slot ring holding the last two SWMM results.
static void I5_lag_applies_depths_from_the_second_step_after()
{
    const std::vector<value_t> lag0 = depths_seen(1, 0, 5, 0);
    const std::vector<value_t> lag1 = depths_seen(1, 1, 5, 0);
    const double want0[5] = {0, 1, 2, 3, 4};
    const double want1[5] = {0, 0, 1, 2, 3};
    for (int i = 0; i < 5; ++i)
    {
        CHECK(std::fabs((double)lag0[i] - want0[i]) < 1e-9, "lag 0: depth k applies from step k + 1");
        CHECK(std::fabs((double)lag1[i] - want1[i]) < 1e-9, "lag 1: depth k applies from step k + 2");
    }
}

// ---------------------------------------------------------------------------
// Property: a drain applies SWMM's freshest depths at once under lag 1, breaking
// the one-step delay at that step, and leaves lag 0 untouched.
// Class: interval 1, five steps, a drain on step 3.
// Why: every checkpoint must hold the newest depths so a resume (which restores
// SWMM's newest state) starts from the same depths the clean run has.
// Killed by: a drain that leaves the pending depths in place. Second correct
// implementation: promoting the pending depths and then overwriting them with the
// staged ones in two copies.
static void I6_drain_applies_freshest_depths_under_lag()
{
    const std::vector<value_t> lag1 = depths_seen(1, 1, 5, 3);
    const double want1[5] = {0, 0, 1, 3, 3};
    for (int i = 0; i < 5; ++i)
        CHECK(std::fabs((double)lag1[i] - want1[i]) < 1e-9, "lag 1 with a drain on step 3: step 4 sees depth 3");
    const std::vector<value_t> lag0 = depths_seen(1, 0, 5, 3);
    const double want0[5] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; ++i)
        CHECK(std::fabs((double)lag0[i] - want0[i]) < 1e-9, "lag 0: a drain changes nothing when SWMM steps every step");
}

// ---------------------------------------------------------------------------
// Property: after a drain step nothing is carried (no accumulated steps, no
// pending depths), and without the drain something is.
// Class: interval 4 lag 0 over six steps, and interval 1 lag 1 over one step, each
// with and without a drain on the last step.
// Why: this is the state a checkpoint at a print instant must have; a carried
// accumulator is lost at a resume, which then steps SWMM on different input.
// Killed by: a drain that leaves the counter or the pending flag set. Second
// correct implementation: resetting a separate "dirty" flag in the drain path.
static void I7_drain_leaves_nothing_carried()
{
    for (int with_drain = 0; with_drain <= 1; ++with_drain)
    {
        coupling_interval_t c;
        c.configure(4, 0);
        value_t held = (value_t)0.0;
        for (int s = 1; s <= 6; ++s)
        {
            const value_t q = (value_t)1.0;
            c.accumulate(&q, 1, (value_t)1.0);
            const bool drain = (with_drain == 1 && s == 6);
            const bool stepped = c.step_due(drain);
            if (stepped)
            {
                c.staged_depths(1)[0] = (value_t)s;
                c.clear_interval();
            }
            c.settle(&held, 1, stepped, drain);
        }
        if (with_drain)
            CHECK(!c.carries_state(), "interval 4, drain on step 6: nothing carried");
        else
            CHECK(c.carries_state(), "control: interval 4, no drain on step 6: steps 5 and 6 are carried");
    }

    for (int with_drain = 0; with_drain <= 1; ++with_drain)
    {
        coupling_interval_t c;
        c.configure(1, 1);
        value_t held = (value_t)0.0;
        const value_t q = (value_t)1.0;
        c.accumulate(&q, 1, (value_t)1.0);
        const bool drain = (with_drain == 1);
        const bool stepped = c.step_due(drain);
        if (stepped)
        {
            c.staged_depths(1)[0] = (value_t)1.0;
            c.clear_interval();
        }
        c.settle(&held, 1, stepped, drain);
        if (with_drain)
            CHECK(!c.carries_state(), "lag 1, drain: no pending depths");
        else
            CHECK(c.carries_state(), "control: lag 1, no drain: the SWMM result is pending");
    }
}

// ---------------------------------------------------------------------------
// Property: between SWMM steps the held depths do not change, and a SWMM result
// lands at the next TRITON step.
// Class: interval 3, lag 0, seven steps, SWMM's depth after its k-th step 10 * k.
// Why: "the depths SWMM returns are held for the next N steps" is the interval's
// definition on the surface side.
// Killed by: settling on a non-stepping step, or never applying the result. Second
// correct implementation: scattering straight into the held array at the step.
static void I8_held_depths_between_steps()
{
    coupling_interval_t c;
    c.configure(3, 0);
    value_t held = (value_t)0.0;
    int swmm_steps = 0;
    std::vector<value_t> seen;
    for (int s = 1; s <= 7; ++s)
    {
        seen.push_back(held);
        const value_t q = (value_t)1.0;
        c.accumulate(&q, 1, (value_t)1.0);
        const bool stepped = c.step_due(false);
        if (stepped)
        {
            ++swmm_steps;
            c.staged_depths(1)[0] = (value_t)(10 * swmm_steps);
            c.clear_interval();
        }
        c.settle(&held, 1, stepped, false);
    }
    const double want[7] = {0, 0, 0, 10, 10, 10, 20};
    for (int i = 0; i < 7; ++i)
        CHECK(std::fabs((double)seen[i] - want[i]) < 1e-9, "depths change only after steps 3 and 6");
}

// ---------------------------------------------------------------------------
// Property: the cfg refusal rejects exactly the values outside interval >= 1 and
// lag in {0, 1}, and accepts every value inside.
// Class: members of the legal set and of each illegal side (zero, negative, two).
// Why: a run that read interval 0 would divide by it or never step SWMM.
// Killed by: a check on one key only, or lag < 2 admitting a negative lag. Second
// correct implementation: a table lookup of legal lags.
static void I9_cfg_range()
{
    CHECK(coupling_interval_cfg_error(1, 0).empty(), "(1, 0) is legal");
    CHECK(coupling_interval_cfg_error(4, 0).empty(), "(4, 0) is legal");
    CHECK(coupling_interval_cfg_error(1, 1).empty(), "(1, 1) is legal");
    CHECK(coupling_interval_cfg_error(4, 1).empty(), "(4, 1) is legal");
    CHECK(!coupling_interval_cfg_error(0, 0).empty(), "interval 0 is refused");
    CHECK(!coupling_interval_cfg_error(-3, 0).empty(), "a negative interval is refused");
    CHECK(!coupling_interval_cfg_error(1, 2).empty(), "lag 2 is refused");
    CHECK(!coupling_interval_cfg_error(1, -1).empty(), "a negative lag is refused");
    CHECK(coupling_interval_cfg_error(0, 0).find("swmm_coupling_interval") != std::string::npos,
          "the interval refusal names its key");
    CHECK(coupling_interval_cfg_error(1, 2).find("swmm_coupling_lag") != std::string::npos,
          "the lag refusal names its key");
}

// ---------------------------------------------------------------------------
// Property: src/triton.h carries the test-only row T-D2: a compile switch
// TRITON_DIAG_TEST_NO_END_DRAIN, off by default, that removes only the end-of-run
// SWMM step, wired into the end-of-run term of the T3 block's drain.
// Class: the three places the row lives (the switch arm, the default arm, the
// end-of-run term); a source-structure check, because the switch is a compile-time
// choice no unit build of this file can flip.
// Why: without the row the case that shows the end-of-run step is needed cannot fail.
// Killed by: deleting the switch, leaving the default arm off, or leaving the
// end-of-run term unconnected to it. Second correct implementation: an #ifdef
// around the end-of-run term inside the block instead of a macro, which this test
// would reject only if it also dropped the macro name; the check is on the macro.
// Control: the existing TRITON_DIAG_TEST_NO_DRAIN switch must be found by the same
// reader, or a zero here would mean a broken reader.
static void I10_t_d2_switch_is_wired()
{
    std::ifstream in(TRITON_H_PATH);
    CHECK(in.is_open(), "src/triton.h opens");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string t = ss.str();
    CHECK(t.find("#ifdef TRITON_DIAG_TEST_NO_DRAIN") != std::string::npos, "control: the T-D switch is found by this reader");
    const size_t sw = t.find("#ifdef TRITON_DIAG_TEST_NO_END_DRAIN");
    CHECK(sw != std::string::npos, "the T-D2 switch exists");
    if (sw == std::string::npos) return;
    const size_t off = t.find("#define TRITON_T3_END_DRAIN 0", sw);
    const size_t els = t.find("#else", sw);
    const size_t on = t.find("#define TRITON_T3_END_DRAIN 1", sw);
    CHECK(off != std::string::npos && els != std::string::npos && on != std::string::npos && off < els && els < on,
          "the switch arm turns the end-of-run drain off and the default arm leaves it on");
    const size_t er = t.find("const bool ends_run");
    CHECK(er != std::string::npos, "the end-of-run term exists");
    if (er == std::string::npos) return;
    const std::string line = t.substr(er, t.find(';', er) - er);
    CHECK(line.find("TRITON_T3_END_DRAIN") != std::string::npos && line.find("sim_duration") != std::string::npos,
          "the end-of-run term is gated by the switch's macro and still tests the run duration");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"I1_defaults_are_inactive", I1_defaults_are_inactive},
    {"I2_interval_mean_is_dt_weighted", I2_interval_mean_is_dt_weighted},
    {"I3_one_step_interval_passes_q_unchanged", I3_one_step_interval_passes_q_unchanged},
    {"I4_step_due_every_n_or_on_drain", I4_step_due_every_n_or_on_drain},
    {"I5_lag_applies_depths_from_the_second_step_after", I5_lag_applies_depths_from_the_second_step_after},
    {"I6_drain_applies_freshest_depths_under_lag", I6_drain_applies_freshest_depths_under_lag},
    {"I7_drain_leaves_nothing_carried", I7_drain_leaves_nothing_carried},
    {"I8_held_depths_between_steps", I8_held_depths_between_steps},
    {"I9_cfg_range", I9_cfg_range},
    {"I10_t_d2_switch_is_wired", I10_t_d2_switch_is_wired},
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
