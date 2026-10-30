/** @file test_coupled_cells.cpp
 *  @brief Tier-1 tests for the per-cell exchange of coupling nodes that share a cell.
 *
 *  No solver run, no cluster, no SWMM call. Several coupling nodes may sit in one
 *  TRITON cell; the exchange kernel then runs one thread per cell, so no two
 *  threads write one cell, and every node's flow is computed from the cell's depth
 *  before any is applied. A cell with one node must run the single-node arithmetic
 *  unchanged, or every existing model changes its bytes; and a draining cell must
 *  never go below zero, or the surface invents water it then sends to the sewer.
 *  These tests call the two pure functions in swmm_triton.h on the CPU:
 *
 *    exchange_one_coupled_cell  -- one cell's exchange, applied to the cell;
 *    build_coupled_cell_index   -- the grouping of nodes into cells.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one
 *  node id per property; with no argument every test runs.
 */

// The same include prefix as test_coupled_list.cpp, for the reasons
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

/** One cell (index 0 in a 1-cell grid) holding the given nodes; runs the cell.
 *
 *  Every run also checks that the step conserves water between the cell and the
 *  sewer: the cell's volume change plus the volume the step books to SWMM is zero,
 *  (h_new - h_old)*dx^2 + sum_k exchange_q_k*FT3_TO_M3_FACTOR*dt = 0, to within
 *  1e-12 of h_old*dx^2 + sum_k |exchange_q_k|*FT3_TO_M3_FACTOR*dt (exchange_q is
 *  positive into the sewer, in ft3/s). Checking it here rather than in one test
 *  is what makes it hold for every case the suite builds -- one, two and three
 *  nodes, filling, draining, mixed, and the limiter. A kernel that moves water in
 *  the cell and books a different amount to SWMM fails it: the surface and the
 *  sewer then disagree about where the water went, and the global balance drifts
 *  by that amount every step.
 */
struct cell_case
{
    double h, qx = 0.3, qy = -0.2;
    std::vector<double> loss, diam, max_depth, new_depth, exq;
    std::vector<int> pos, start, nodes;
    void run(double dx, double dt, double hextra)
    {
        const double h_old = h;
        exq.assign(loss.size(), 0.0);
        pos.assign(loss.size(), 0);
        build_coupled_cell_index(pos, start, nodes);
        exchange_one_coupled_cell<double>(0, start.data(), nodes.data(), dx, dt, &h, &qx, &qy, hextra, pos.data(),
                                          loss.data(), diam.data(), max_depth.data(), new_depth.data(), exq.data());
        double booked = 0.0, moved = 0.0;
        for (double q : exq) { booked += q * FT3_TO_M3_FACTOR * dt; moved += std::fabs(q) * FT3_TO_M3_FACTOR * dt; }
        CHECK(std::fabs((h - h_old) * dx * dx + booked) <= 1e-12 * (h_old * dx * dx + moved),
              "conservation: the cell's volume change plus the volume booked to SWMM is zero");
    }
};

/** The single-node kernel body as it stood before cells existed, copied verbatim. */
static void reference_single_node(double dx, double dt, double& h, double& qx, double& qy, double hextra,
                                  double loss, double diam, double max_depth, double new_depth, double& exq)
{
    double flow = 0.0;
    double hij = h;
    double hold = hij;
    double areaM = PI_*0.25*diam*diam;
    if(hij>0.0 && new_depth <= max_depth){
        flow=-(2.0/3.0)*loss*PI_*diam*sqrt(2.0*_G_*hij)*hij;
    }
    else if(hij>0.0 && new_depth <= max_depth + hij){
        flow=-loss*areaM*sqrt(2.0*_G_*(hij + max_depth - new_depth));
    }
    else if(new_depth > max_depth + hij){
        flow=loss*areaM*sqrt(2.0*_G_*(new_depth - max_depth - hij ) );
    }
    double h_src = (flow * dt) / (dx * dx);
    hij += h_src;
    if (hij < hextra)
    {
        if (hij < EPS12)
        {
            hij = 0.0;
            flow=-hold*dx*dx/dt;
        }
        qx = 0.0;
        qy = 0.0;
    }
    h = hij;
    exq=-flow/FT3_TO_M3_FACTOR;
}

// ---------------------------------------------------------------------------
// Property: a cell with one node runs the single-node arithmetic bit for bit.
// Class: one node in each of the three exchange cases, below hextra, and drained
// past zero (the limiter). Why: every model with one coupling node per cell must
// keep its bytes. Killed by: routing a one-node cell through the several-node path
// -- for one node its sums are exact, so the two paths differ only where the
// limiter fires: the old limiter writes -hold*dx*dx/dt, the scaled path writes
// f*((-hold*dx*dx/dt)/f), and at h = 0.00096 and 0.00098 the exchanges those give
// differ in the last bit (at 0.00059 the flows differ but /FT3_TO_M3_FACTOR
// rounds them to one exchange, so that input could not see the change) -- or by
// any change to the three cases. Second correct implementation: the per-node
// kernel this one replaces.
// ---------------------------------------------------------------------------
static void K1_one_node_cell_is_the_single_node_kernel()
{
    struct { double h, nd, md, loss, dx, dt; } cases[] = {
        {0.30, 0.10, 1.0, 0.1, 3.5, 1.0}, {0.30, 1.10, 1.0, 0.1, 3.5, 1.0}, {0.05, 2.50, 1.0, 0.1, 3.5, 1.0},
        {0.0004, 0.0, 1.0, 0.1, 3.5, 1.0}, {0.00096, 0.0, 1.0, 0.5, 1.0, 10.0}, {0.00098, 0.0, 1.0, 0.5, 1.0, 10.0}};
    for (const auto& k : cases) {
        cell_case c; c.h = k.h; c.loss = {k.loss}; c.diam = {1.2}; c.max_depth = {k.md}; c.new_depth = {k.nd};
        c.run(k.dx, k.dt, 0.001);
        double h = k.h, qx = 0.3, qy = -0.2, exq = 0.0;
        reference_single_node(k.dx, k.dt, h, qx, qy, 0.001, k.loss, 1.2, k.md, k.nd, exq);
        CHECK(c.h == h && c.qx == qx && c.qy == qy && c.exq[0] == exq, "one node: depth, velocities and exchange bit-identical");
    }
}

// Property: several nodes see one water surface and their flows add.
// Class: two and three identical draining nodes, no limiter. Why: a node's flow
// must not depend on another node's having been applied first. Killed by: applying
// each node's flow to the cell before computing the next (sequential update), or by
// writing one node's exchange for all. Second correct implementation: compute all
// flows into a scratch array, then apply their sum.
static void K2_nodes_share_one_surface()
{
    for (int n = 2; n <= 3; ++n) {
        cell_case c; c.h = 0.30;
        c.loss.assign(n, 0.1); c.diam.assign(n, 1.2); c.max_depth.assign(n, 1.0); c.new_depth.assign(n, 0.1);
        c.run(10.0, 1.0, 0.001);
        double h = 0.30, qx = 0.3, qy = -0.2, exq1 = 0.0;
        reference_single_node(10.0, 1.0, h, qx, qy, 0.001, 0.1, 1.2, 1.0, 0.1, exq1);
        bool same = true;
        for (int k = 0; k < n; ++k) same = same && std::fabs(c.exq[k] - exq1) <= 1e-12 * std::fabs(exq1);
        const double flow1 = -exq1 * FT3_TO_M3_FACTOR;
        CHECK(same, "every node's exchange equals the single-node exchange at the cell's prior depth");
        CHECK(std::fabs(c.h - (0.30 + n * flow1 * 1.0 / 100.0)) <= 1e-12, "the cell loses the sum of the nodes' flows");
    }
}

// Property: the limiter -- a draining cell ends at exactly 0, never below, and the
// nodes' outflows sum to the cell's prior volume (item 13 DoD (iv)).
// Class: three nodes, each of which alone would drain more than the cell holds.
// Why: without it each node takes the whole cell and the surface creates water.
// Killed by: limiting each node separately (each takes the full cell), or by
// clipping depth without scaling the exchange. Second correct implementation:
// give each node the cell volume times its share of the unlimited outflow.
static void K3_limiter_drains_to_exactly_zero()
{
    cell_case c; c.h = 0.002;
    c.loss = {0.5, 0.3, 0.2}; c.diam = {1.2, 1.0, 0.8}; c.max_depth = {1.0, 1.0, 1.0}; c.new_depth = {0.0, 0.0, 0.0};
    const double dx = 1.0, dt = 10.0;
    c.run(dx, dt, 0.001);
    double out = 0.0;
    for (double q : c.exq) { CHECK(q >= 0.0, "every node drains (exchange into the sewer is positive)"); out += q * FT3_TO_M3_FACTOR * dt; }
    CHECK(c.h == 0.0, "the cell ends at exactly 0");
    CHECK(std::fabs(out - 0.002 * dx * dx) <= 1e-12 * 0.002, "the outflows sum to the prior volume");
    CHECK(c.qx == 0.0 && c.qy == 0.0, "velocities are removed in a dry cell");
}

// Property: inflows from several nodes add, and the limiter never touches an inflow.
// Class: two surcharged nodes (case 3) into a dry cell; and a mixed cell -- one
// surcharged inflow beside one drain that alone would empty the cell -- where the
// limiter fires (scale about 0.22); and a deep mixed cell where it does not. Why:
// the limiter shares out only the water the
// cell holds plus what flows in this step; an inflow is not the cell's to ration.
// Killed by: scaling inflows with outflows (the mixed case; in a filling cell scale
// is 1, so that case alone cannot see it). Second correct implementation: the sum
// of two single-node inflows, and in the mixed case giving the drain exactly the
// prior volume plus the inflow, computed without a scale factor.
static void K4_filling_cell_sums_inflows()
{
    cell_case c; c.h = 0.0;
    c.loss = {0.1, 0.1}; c.diam = {1.2, 1.2}; c.max_depth = {1.0, 1.0}; c.new_depth = {2.0, 1.5};
    c.run(5.0, 1.0, 0.001);
    double sum = 0.0;
    for (double q : c.exq) { CHECK(q < 0.0, "both nodes fill the cell"); sum += -q * FT3_TO_M3_FACTOR; }
    CHECK(std::fabs(c.h - sum / 25.0) <= 1e-12, "the cell gains the sum of the inflows");

    cell_case m; m.h = 0.05;
    m.loss = {0.5, 0.1}; m.diam = {1.2, 0.5}; m.max_depth = {1.0, 1.0}; m.new_depth = {0.0, 1.06};
    const double dx = 1.0, dt = 10.0;
    m.run(dx, dt, 0.001);
    double h = 0.05, qx = 0.3, qy = -0.2, exq_in = 0.0;
    reference_single_node(dx, dt, h, qx, qy, 0.001, 0.1, 0.5, 1.0, 1.06, exq_in);
    CHECK(m.exq[1] < 0.0 && std::fabs(m.exq[1] - exq_in) <= 1e-12 * std::fabs(exq_in),
          "mixed cell: the inflow's exchange is its single-node value, unscaled");
    const double in_vol = -m.exq[1] * FT3_TO_M3_FACTOR * dt, out_vol = m.exq[0] * FT3_TO_M3_FACTOR * dt;
    CHECK(m.h == 0.0, "mixed cell: the cell ends at exactly 0");
    CHECK(std::fabs(out_vol - (0.05 * dx * dx + in_vol)) <= 1e-12 * out_vol,
          "mixed cell: the drain takes the prior volume plus the inflow, no more");

    // A mixed cell the limiter does not touch: one weir drain and one surcharged
    // return in a deep cell. Both exchanges are booked whole and the cell keeps water.
    cell_case w; w.h = 0.30;
    w.loss = {0.1, 0.1}; w.diam = {1.2, 1.2}; w.max_depth = {1.0, 1.0}; w.new_depth = {0.1, 1.8};
    w.run(3.5, 1.0, 0.001);
    CHECK(w.exq[0] > 0.0 && w.exq[1] < 0.0 && w.h > 0.0, "unlimited mixed cell: one node drains, one returns, the cell stays wet");
}

// Property: nodes are grouped by cell, cells in first-appearance order, nodes in
// index order. Class: a position list with repeats. Why: the kernel's one thread per
// cell must see every node of its cell exactly once. Killed by: one group per node,
// or nodes out of order. Second correct implementation: a stable sort by (first
// appearance, index).
static void K5_cell_index_groups_by_cell()
{
    std::vector<int> start, nodes;
    const int n = build_coupled_cell_index({7, 3, 7, 9, 3}, start, nodes);
    CHECK(n == 3, "three distinct cells");
    CHECK(start == std::vector<int>({0, 2, 4, 5}), "offsets: two, two and one node");
    CHECK(nodes == std::vector<int>({0, 2, 1, 4, 3}), "cell 7 holds 0, 2; cell 3 holds 1, 4; cell 9 holds 3");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"K1_one_node_cell_is_the_single_node_kernel", K1_one_node_cell_is_the_single_node_kernel},
    {"K2_nodes_share_one_surface", K2_nodes_share_one_surface},
    {"K3_limiter_drains_to_exactly_zero", K3_limiter_drains_to_exactly_zero},
    {"K4_filling_cell_sums_inflows", K4_filling_cell_sums_inflows},
    {"K5_cell_index_groups_by_cell", K5_cell_index_groups_by_cell},
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
