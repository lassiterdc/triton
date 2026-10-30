/** @file test_coupled_cells_review.cpp
 *  @brief Reviewer test for the several-node exchange kernel: conservation where the
 *  limiter fires and the cell would end just above zero.
 *
 *  When several coupling nodes share a TRITON cell, the exchange kernel computes
 *  every node's flow from the cell's depth, sums them, and fires its limiter when
 *  the would-be depth is below EPS12: the cell is set to 0 and the outflows are
 *  scaled so the volume booked to SWMM equals the volume the cell lost. The scale
 *  is clamped to at most 1. A would-be depth in [0, EPS12) is the one place in the
 *  limiter's range where the exact scale is ABOVE 1, so the clamp books the
 *  unscaled outflows while the cell still loses its whole remaining depth, and the
 *  difference -- the would-be depth times dx^2 -- leaves the model. The single-node
 *  kernel has no such gap: in the same band it books exactly -hold*dx*dx/dt.
 *
 *  Same build arrangement and include prefix as test_coupled_cells.cpp.
 */

#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace SWMM_triton;

static int g_failures = 0;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

/** The weir-type (case 1) and surcharge (case 3) flows of the exchange law, written
 *  out here so the test's inputs do not depend on the kernel's own helper. */
static double weir_flow(double h, double loss, double d)
{
    return -(2.0 / 3.0) * loss * PI_ * d * std::sqrt(2.0 * _G_ * h) * h;
}
static double surcharge_flow(double h, double nd, double md, double loss, double d)
{
    return loss * (PI_ * 0.25 * d * d) * std::sqrt(2.0 * _G_ * (nd - md - h));
}

/** Runs one several-node cell whose time step is chosen so the would-be depth is
 *  5e-13 m (inside [0, EPS12)), and checks the per-cell identity of item 13 DoD (iv):
 *  (h_new - h_old)*dx^2 + sum_k exchange_q_k*FT3_TO_M3_FACTOR*dt = 0, to within 1e-12
 *  of h_old*dx^2 + sum_k |exchange_q_k|*FT3_TO_M3_FACTOR*dt. */
static void run_band_case(const char* label, double hold, double dx, double net_flow,
                          std::vector<double> loss, std::vector<double> diam,
                          std::vector<double> max_depth, std::vector<double> new_depth)
{
    const double target = 5e-13;
    const double dt = (target - hold) * dx * dx / net_flow;
    const double would_be = hold + (net_flow * dt) / (dx * dx);
    // Precondition, not the property: the input really lands in the band.
    if (!(would_be >= 0.0 && would_be < EPS12)) {
        ++g_failures;
        std::fprintf(stderr, "  PRECONDITION (%s): would-be depth %.17g is not in [0, EPS12)\n", label, would_be);
        return;
    }
    const int n = static_cast<int>(loss.size());
    double h = hold, qx = 0.3, qy = -0.2;
    std::vector<double> exq(n, 0.0);
    std::vector<int> pos(n, 0), start, nodes;
    build_coupled_cell_index(pos, start, nodes);
    exchange_one_coupled_cell<double>(0, start.data(), nodes.data(), dx, dt, &h, &qx, &qy, 0.001, pos.data(),
                                      loss.data(), diam.data(), max_depth.data(), new_depth.data(), exq.data());
    double booked = 0.0, moved = 0.0;
    for (double q : exq) { booked += q * FT3_TO_M3_FACTOR * dt; moved += std::fabs(q) * FT3_TO_M3_FACTOR * dt; }
    const double residual = (h - hold) * dx * dx + booked;
    const double tol = 1e-12 * (hold * dx * dx + moved);
    if (!(std::fabs(residual) <= tol))
        std::fprintf(stderr, "  %s: would-be %.3g m, h_new %.3g, residual %.6g m3, tolerance %.6g m3\n",
                     label, would_be, h, residual, tol);
    CHECK(std::fabs(residual) <= tol, label);
}

/** Property: the several-node kernel conserves water between the cell and SWMM
 *  (item 13 DoD (iv)'s identity, at its fixed tolerance) on limiter-firing inputs
 *  whose would-be depth lies in [0, EPS12).
 *
 *  Class: several-node cells where the limiter fires with the would-be depth at or
 *  just above zero -- here two weir drains, and one weir drain beside one surcharged
 *  return (mixed). Positioned at 5e-13 m, the middle of the band.
 *
 *  Why it is required: in that band the kernel sets the cell to 0 but books the
 *  outflows unscaled, so the would-be depth times dx^2 is removed from the surface
 *  and booked nowhere. A reader of the global balance then sees water vanish at
 *  every cell that lands in the band, and the single-node kernel, which books
 *  exactly the cell's prior volume there, does not.
 *
 *  Killed by: clamping the limiter's scale to at most 1, which is the code as it
 *  stands. Passes under: dropping that clamp so the scale may exceed 1 by at most
 *  EPS12 relative (the outflows then take exactly the prior volume plus inflows);
 *  or firing the several-node limiter only below 0 (the cell then keeps its 5e-13 m);
 *  or booking -hold*dx*dx/dt less the inflows as the one-node path does.
 */
static void RV2_several_node_limiter_conserves_at_the_band_edge()
{
    const double hold = 0.01, dx = 1.0;
    const double drain = weir_flow(hold, 0.5, 1.2);
    run_band_case("two weir drains, would-be depth in [0, EPS12)", hold, dx, 2.0 * drain,
                  {0.5, 0.5}, {1.2, 1.2}, {1.0, 1.0}, {0.0, 0.0});
    const double inflow = surcharge_flow(hold, 1.0105, 1.0, 0.1, 0.5);
    run_band_case("one drain and one surcharged return, would-be depth in [0, EPS12)", hold, dx, inflow + drain,
                  {0.5, 0.1}, {1.2, 0.5}, {1.0, 1.0}, {0.0, 1.0105});
}

/** Runs the one test; exits 1 when any case fails, so CTest reads the exit code. */
int main(int argc, char** argv)
{
    (void)argc; (void)argv;
    std::fprintf(stderr, "RUN  RV2_several_node_limiter_conserves_at_the_band_edge\n");
    RV2_several_node_limiter_conserves_at_the_band_edge();
    std::fprintf(stderr, "%s RV2_several_node_limiter_conserves_at_the_band_edge\n", g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
