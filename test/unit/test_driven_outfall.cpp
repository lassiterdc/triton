/** @file test_driven_outfall.cpp
 *  @brief Tier-1 tests for the driven-stage open-water outfalls (cfg swmm_driven_outfalls).
 *
 *  No solver run, no cluster, no SWMM call. A SWMM OUTFALL named in the list stays an
 *  outfall; TRITON sets its fixed stage from the water level of the cell that holds it,
 *  and takes the backflow the sewer pushes out through it from that cell. Everything the
 *  feature decides without MPI, an engine or a file is a pure function in swmm_triton.h,
 *  and these tests call those functions on the CPU:
 *
 *    read_coupling_list, check_driven_outfalls, read_inp_outfall_types  -- the list and its refusals;
 *    read_inp_conduit_ends, driven_outfall_link_sign                     -- the sign of a link's flow;
 *    strip_owner                                                         -- the owning rank of a cell;
 *    driven_outfall_stage, driven_stage_to_model_units, swmm_flow_units_m3s_per_unit;
 *    driven_outfall_backflow_cell, compute_driven_outfall_backflow       -- the one-way application;
 *    encode/decode_outfall_stage_record, outfall_stage_count_reader      -- outfall_stage.bin and
 *                                                                           its resume positioning.
 *
 *  The last group of tests reads src/triton.h: every call site of the feature sits under the
 *  one condition swmm_model.driven_active(), which is what keeps the key-absent path the one
 *  every earlier build ran.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one node id per
 *  property; with no argument every test runs.
 */

// The same include prefix as test_coupled_cells.cpp, for the reasons
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
#include <unistd.h>

#ifndef TRITON_H_PATH
#error "TRITON_H_PATH must name src/triton.h (target_compile_definitions in test/unit/CMakeLists.txt)"
#endif

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

// ---------------------------------------------------------------------------
// The list file and its refusals.
// ---------------------------------------------------------------------------

/** Property: the list file has the coupling list's rules (one name per line, `;`
 *  comments, no header, whitespace stripped, a CRLF line end and a missing final
 *  newline read the same), and each name keeps its 1-based line.
 *  Killed by: a reader that keeps the `;` lines, keeps a trailing `\r`, or counts a
 *  skipped line out of the numbering. Second correct implementation: a reader
 *  that tokenises with `>>`; it passes everything but a name with an inner space,
 *  which this list never has. */
static void L1_list_file_format()
{
    std::istringstream in("; the Norfolk outfalls\r\n  OUT_A  \r\n\r\n\tout_b\n;skip\nOUT_C");
    std::vector<std::string> names;
    std::vector<int> lines;
    read_coupling_list(in, names, lines);
    CHECK(names == std::vector<std::string>({"OUT_A", "out_b", "OUT_C"}), "three names, stripped, comments and blanks skipped");
    CHECK(lines == std::vector<int>({2, 4, 6}), "each name keeps its 1-based file line");

    std::istringstream empty("; nothing listed\n\n");
    names.clear(); lines.clear();
    read_coupling_list(empty, names, lines);
    CHECK(names.empty(), "a list that names nothing lists no outfall");
}

/** Property: the .inp's [OUTFALLS] section is read with SWMM's rule (a section starts at
 *  any first token beginning with `[` and matches by keyword prefix, case-insensitively,
 *  tokens split as getTokens does), and the third token is the type, upper case.
 *  Killed by: reading another section's third token, or a case-sensitive type. */
static void L2_outfall_types_read_as_swmm_reads_them()
{
    std::istringstream in(
        "[JUNCTIONS]\nJ1 10 3 0 0 0\n\n"
        "  [outfalls]  ; any indent, any case\n"
        "OUT_A 8.5 FIXED 9.0 NO\n"
        "out_b 8.0 free no ; comment\n"
        "\"OUT C\" 7 Fixed 5\n"
        "[CONDUITS]\nC1 J1 OUT_A 100 0.013\n");
    std::vector<std::string> names, types;
    read_inp_outfall_types(in, names, types);
    CHECK(names == std::vector<std::string>({"OUT_A", "out_b", "OUT C"}), "three outfalls, the quoted name without its quotes");
    CHECK(types == std::vector<std::string>({"FIXED", "FREE", "FIXED"}), "types upper-cased; no other section contributes");
}

/** Property: every refusal of check_driven_outfalls fires on its own input, names the node and
 *  the list line, and a list that is fine returns no refusal.
 *  Why required: each refusal is a startup stop on every rank; one that never fires lets a run
 *  start with a node the kernel cannot serve.
 *  Killed by: dropping any one test. Second correct implementation: the same tests in another
 *  order; the arms below hold one defect at a time, so the order does not change a result. */
static void L3_every_refusal_names_the_node()
{
    const std::vector<std::string> outs   = {"OUT_A", "OUT_B", "OUT_T"};
    const std::vector<std::string> types  = {"FIXED", "FIXED", "TIDAL"};
    const std::vector<std::string> coupl  = {"J1", "J2"};
    const std::vector<std::string> coords = {"J1", "J2", "OUT_A", "OUT_B", "OUT_T"};

    driven_outfall_error_t e = check_driven_outfalls({"OUT_A", "out_b"}, {1, 2}, outs, types, coupl, coords);
    CHECK(e.what.empty(), "a listed pair of FIXED outfalls with coordinates passes (names fold case)");

    e = check_driven_outfalls({"OUT_A", "OUT_B", "out_a"}, {3, 4, 9}, outs, types, coupl, coords);
    CHECK(e.line == 9 && e.what.find("out_a") != std::string::npos && e.what.find("line 3") != std::string::npos,
          "a name repeated after the case fold is refused with both lines");

    e = check_driven_outfalls({"OUT_A", "J1"}, {1, 2}, outs, types, coupl, coords);
    CHECK(e.line == 2 && e.what.find("J1") != std::string::npos && e.what.find("not an OUTFALL") != std::string::npos,
          "a junction is no OUTFALL node");

    e = check_driven_outfalls({"NO_SUCH"}, {5}, outs, types, coupl, coords);
    CHECK(e.line == 5 && e.what.find("NO_SUCH") != std::string::npos, "a name that is no node is refused");

    e = check_driven_outfalls({"OUT_T"}, {1}, outs, types, coupl, coords);
    CHECK(e.what.find("OUT_T") != std::string::npos && e.what.find("TIDAL") != std::string::npos
          && e.what.find("FIXED") != std::string::npos, "a TIDAL outfall is refused with its type");

    // An outfall that is also a coupling node: only reachable by a model whose coupling list
    // names an outfall; the refusal is the list's own, ahead of the engine's junction-only one.
    e = check_driven_outfalls({"OUT_A"}, {1}, outs, types, {"OUT_A"}, coords);
    CHECK(e.what.find("OUT_A") != std::string::npos && e.what.find("coupling node") != std::string::npos,
          "an outfall that is also a coupling node is refused");

    e = check_driven_outfalls({"OUT_B"}, {7}, outs, types, coupl, {"J1", "J2", "OUT_A"});
    CHECK(e.line == 7 && e.what.find("[COORDINATES]") != std::string::npos, "an outfall with no coordinates is refused");
}

/** Property: the sign of a conduit's API flow comes from the .inp's from/to names, because SWMM
 *  swaps a reversed conduit's end nodes while the API flow keeps the .inp orientation.
 *  +1 when the .inp joins the conduit INTO the outfall, -1 when it leaves it, 0 when it joins
 *  neither end to it, and the "not a conduit" sentinel for a link the .inp's [CONDUITS] lacks.
 *  Killed by: deciding the sign from the engine's end nodes (wrong exactly for a reversed
 *  conduit, the case the stage API test builds). */
static void L4_link_sign_follows_the_inp_orientation()
{
    std::istringstream in("[CONDUITS]\nC1 J1 OUT_A 100 0.013 0 0\nc2 out_b J2 80 0.013\n[PUMPS]\nP1 J1 OUT_A\n");
    const inp_conduit_ends_t ends = read_inp_conduit_ends(in);
    CHECK(ends.names.size() == 2, "two conduits, no pump");
    CHECK(driven_outfall_link_sign(ends, "C1", "OUT_A") == 1, "C1 runs J1 -> OUT_A: a positive flow enters the outfall");
    CHECK(driven_outfall_link_sign(ends, "C2", "OUT_B") == -1, "C2 runs OUT_B -> J2 (names fold): a positive flow leaves the outfall");
    CHECK(driven_outfall_link_sign(ends, "C1", "OUT_B") == 0, "C1 does not touch OUT_B");
    CHECK(driven_outfall_link_sign(ends, "P1", "OUT_A") == 2, "a pump is no [CONDUITS] line: the caller takes the engine's ends");
}

/** Property: strip_owner is the rule process_swmm_node_locations applies, in one place: the
 *  rank whose strip holds the row, with the first global row of that strip, and -1 outside
 *  the grid. Cases: one rank; strips of unequal height; a strip of height 0 skipped; the
 *  first and last row of every strip; a row above and below the grid.
 *  Killed by: an off-by-one at a strip boundary. */
static void L5_strip_owner_is_the_row_stripe_rule()
{
    int first = -9;
    CHECK(strip_owner(0, {10}, first) == 0 && first == 0, "one rank holds row 0");
    CHECK(strip_owner(9, {10}, first) == 0 && first == 0, "one rank holds its last row");
    CHECK(strip_owner(10, {10}, first) == -1, "a row below the grid has no owner");
    CHECK(strip_owner(-1, {10}, first) == -1, "a row above the grid has no owner");

    const std::vector<int> rows = {4, 3, 0, 5};
    const int expect_owner[12] = {0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3};
    const int expect_first[12] = {0, 0, 0, 0, 4, 4, 4, 7, 7, 7, 7, 7};
    bool all = true;
    for (int r = 0; r < 12; ++r) {
        int f = -9;
        const int o = strip_owner(r, rows, f);
        if (o != expect_owner[r] || f != expect_first[r]) {
            all = false;
            std::fprintf(stderr, "    row %d: owner %d first %d, expected %d %d\n", r, o, f, expect_owner[r], expect_first[r]);
        }
    }
    CHECK(all, "every row of strips 4,3,0,5 lies in its strip (the empty strip owns none)");
}

// ---------------------------------------------------------------------------
// The stage rule and the unit conversions.
// ---------------------------------------------------------------------------

/** Property: the stage is z_cell + h where the cell is wet by the surface solver's own test
 *  (h >= hextra, so a cell at exactly hextra is wet), and the invert where it is not;
 *  a cell below the invert elevation still gives z_cell + h when wet, and is not clamped.
 *  Killed by: `>` for `>=`, a clamp to the invert, or using h alone. */
static void S1_stage_rule_at_hextra()
{
    const double hextra = 1e-3, z = -0.6, invert = -1.4;
    CHECK(driven_outfall_stage(0.25, hextra, z, invert) == z + 0.25, "a wet cell gives its water-surface elevation");
    CHECK(driven_outfall_stage(hextra, hextra, z, invert) == z + hextra, "a cell at exactly hextra is wet");
    CHECK(driven_outfall_stage(std::nextafter(hextra, 0.0), hextra, z, invert) == invert, "a cell just under hextra is dry: the invert");
    CHECK(driven_outfall_stage(0.0, hextra, z, invert) == invert, "a dry cell gives the invert");
    CHECK(driven_outfall_stage(0.25, hextra, 12.0, 5.0) == 12.25, "a cell above the invert (the other Norfolk form) gives z + h");
    // The wet stage is below the invert when z + h < invert: the rule does not clamp it.
    CHECK(driven_outfall_stage(0.1, hextra, -2.0, -1.4) == -1.9, "a wet stage below the invert is passed through, not clamped");
}

/** Property: the engine's setter takes the model's length unit, so a US-flow model's stage is
 *  metres over 0.3048 and a metric model's is the metres unchanged; the flow table is exact
 *  for the unit it names (cfs 0.028316846592, cms identity) and 0 outside the enum.
 *  Killed by: a stage converted twice, or the truncated 0.0283168 of the cfs table the T1d
 *  writer uses applied to a cms flow (a 1.7e-7 relative error in every ledger term). */
static void S2_unit_conversions()
{
    CHECK(driven_stage_to_model_units(3.048, true) == 3.048 / 0.3048, "a US model's stage is in feet");
    CHECK(driven_stage_to_model_units(3.048, false) == 3.048, "a metric model's stage is in metres");
    CHECK(std::fabs(swmm_engine_length_m(driven_stage_to_model_units(7.25, true), true) - 7.25) <= 1e-15 * 7.25,
          "feet back to metres is the inverse");
    CHECK(swmm_flow_units_m3s_per_unit(swmm_CMS) == 1.0, "cms is the identity");
    CHECK(swmm_flow_units_m3s_per_unit(swmm_CFS) == 0.028316846592, "cfs");
    CHECK(std::fabs(swmm_flow_units_m3s_per_unit(swmm_LPS) - 1e-3) < 1e-18, "lps");
    CHECK(std::fabs(swmm_flow_units_m3s_per_unit(swmm_MLD) * 86400.0 - 1000.0) < 1e-9, "mld: 1e6 L/day is 1000 m3 over 86400 s");
    CHECK(std::fabs(swmm_flow_units_m3s_per_unit(swmm_MGD) * 86400.0 - 3785.411784) < 1e-9, "mgd");
    CHECK(std::fabs(swmm_flow_units_m3s_per_unit(swmm_GPM) * 60.0 - 3.785411784e-3) < 1e-15, "gpm");
    CHECK(swmm_flow_units_m3s_per_unit(99) == 0.0, "a value outside the enum converts to nothing");
}

// ---------------------------------------------------------------------------
// The one-way application.
// ---------------------------------------------------------------------------

namespace {
/** One cell run through the backflow function; returns the applied rate. */
struct bf_cell
{
    double h, qx = 0.3, qy = -0.2, applied = -1.0;
    void run(double q_back, double dx, double dt, double hextra)
    {
        driven_outfall_backflow_cell<double>(q_back, dx, dt, hextra, h, qx, qy, applied);
    }
};
}

/** Property: an ordinary backflow removes q_back*dt/dx^2 from the cell, applies exactly the rate
 *  asked for, and leaves the momentum alone while the cell stays at or above hextra.
 *  Killed by: a kernel that scales the rate, or zeroes momentum on every call. */
static void B1_backflow_removes_the_requested_depth()
{
    bf_cell c; c.h = 0.5;
    c.run(2.0, 10.0, 0.5, 1e-3);
    CHECK(std::fabs(c.h - (0.5 - 2.0 * 0.5 / 100.0)) <= 1e-15, "the cell loses q*dt/dx^2 of depth");
    CHECK(c.applied == 2.0, "the applied rate is the requested one");
    CHECK(c.qx == 0.3 && c.qy == -0.2, "momentum is untouched while the cell stays wet");
}

/** Property: a cell the removal leaves below hextra (but not below EPS12) keeps its depth and
 *  the requested rate and loses its momentum, as the one-node manhole kernel does.
 *  Killed by: leaving the momentum of a nearly dry cell, which would carry water out of it. */
static void B2_backflow_below_hextra_zeroes_the_momentum()
{
    const double hextra = 1e-2;
    bf_cell c; c.h = 2e-2;
    c.run(1.0, 10.0, 0.5, hextra);   // removes 5e-3 of depth: 2e-2 - 5e-3 = 1.5e-2, still above hextra
    CHECK(c.qx == 0.3, "control: a cell that stays at or above hextra keeps its momentum");
    bf_cell d; d.h = 1.2e-2;
    d.run(1.0, 10.0, 0.5, hextra);   // 1.2e-2 - 5e-3 = 7e-3 < hextra, above EPS12
    CHECK(std::fabs(d.h - 7e-3) <= 1e-15 && d.applied == 1.0, "ends below hextra with its depth and the requested rate");
    CHECK(d.qx == 0.0 && d.qy == 0.0, "momentum zeroed below hextra");
}

/** Property: a removal that would take the cell below EPS12 sets it to exactly zero and applies
 *  what the cell had, hold*dx^2/dt, which is less than the requested rate when the cell cannot
 *  give it: water is never created, and the whole of what the cell held leaves it.
 *  Killed by: a negative depth, or an applied rate equal to the request. */
static void B3_backflow_near_zero_takes_what_the_cell_had()
{
    bf_cell c; c.h = 4e-3;
    const double dx = 10.0, dt = 0.5;
    c.run(5.0, dx, dt, 1e-3);        // asks for 5e-3 of depth, the cell holds 4e-3
    CHECK(c.h == 0.0, "the cell ends at exactly zero, never below");
    CHECK(c.qx == 0.0 && c.qy == 0.0, "momentum zeroed");
    CHECK(c.applied < 5.0, "less than the request was applied");
    CHECK(std::fabs(c.applied * dt - 4e-3 * dx * dx) <= 1e-15, "the whole of the held volume left the cell: applied*dt == hold*dx^2");
    // A request of exactly the held depth leaves nothing and conserves.
    bf_cell e; e.h = 4e-3;
    e.run(4e-3 * dx * dx / dt, dx, dt, 1e-3);
    CHECK(e.h == 0.0 && std::fabs(e.applied * dt - 4e-3 * dx * dx) <= 1e-15, "an exact request leaves a dry cell and conserves");
}

/** Property: the backflow function is the one-node manhole kernel's arithmetic, statement for
 *  statement: for a manhole whose exchange law gives a flow out of the surface (Case 1, the
 *  weir), exchange_one_coupled_cell and driven_outfall_backflow_cell given that flow as the
 *  request leave the cell with the same depth and momentum, in the three regimes (stays wet,
 *  ends between EPS12 and hextra, ends below EPS12), and the manhole's booked exchange equals
 *  the applied rate.
 *  Why required: chunk 1j.1 does not edit the manhole kernel; the one-way backflow is a second
 *  copy of its arithmetic, and two copies can only be trusted to agree if a test holds them to it.
 *  Killed by: any edit of one copy's order of operations or its near-zero rule. */
static void B4_backflow_is_the_one_node_kernel_for_the_same_flow()
{
    const double dx = 10.0, hextra = 1e-3;
    struct regime { double h, dt; const char* name; };
    const regime regimes[3] = {
        {0.8, 0.5, "stays wet"},
        {1.001e-3, 0.5, "ends between EPS12 and hextra"},   // the weir flow removes about 2.4e-6 of depth in 0.5 s
        {2e-3, 300.0, "ends below EPS12"},                  // 300 s of the weir flow would remove about 2.7e-3
    };
    for (const regime& rg : regimes) {
        // The manhole: new_depth <= max_depth, so Case 1 (weir): flow = -(2/3)*loss*PI*diam*sqrt(2 g h)*h.
        const double loss = 0.9, diam = 1.2, max_depth = 3.0, new_depth = 1.0;
        std::vector<double> l = {loss}, d = {diam}, md = {max_depth}, nd = {new_depth}, q = {0.0};
        std::vector<int> pos = {0}, start, nodes;
        build_coupled_cell_index(pos, start, nodes);
        double h1 = rg.h, qx1 = 0.3, qy1 = -0.2;
        exchange_one_coupled_cell<double>(0, start.data(), nodes.data(), dx, rg.dt, &h1, &qx1, &qy1, hextra, pos.data(),
                                          l.data(), d.data(), md.data(), nd.data(), q.data());
        const double flow = -(2.0 / 3.0) * loss * PI_ * diam * std::sqrt(2.0 * _G_ * rg.h) * rg.h;   // m3/s, negative: out of the surface
        bf_cell c; c.h = rg.h;
        c.run(-flow, dx, rg.dt, hextra);
        std::fprintf(stderr, "    regime '%s': manhole h %.17g, backflow h %.17g\n", rg.name, h1, c.h);
        // Equal to the last bit in the usual build; the tolerance (a few ulp of the starting depth)
        // is for a build whose compiler contracts the two copies' multiply-adds differently.
        CHECK(std::fabs(h1 - c.h) <= 4.0 * DBL_EPSILON * rg.h, "same depth after the step");
        CHECK(qx1 == c.qx && qy1 == c.qy, "same momentum after the step");
        CHECK(std::fabs(q[0] * FT3_TO_M3_FACTOR - c.applied) <= 1e-12 * c.applied,
              "the manhole's booked exchange (ft3/s into the sewer) is the applied rate (m3/s)");
    }
}

/** Property: compute_driven_outfall_backflow runs the function above once per listed cell on the
 *  execution space, writes the applied rate of each, and leaves a cell whose request is not above
 *  zero untouched with applied 0. Three cells in a row: a normal removal, a zero request, a
 *  request the cell cannot meet.
 *  Killed by: a launcher that applies a zero request (it would zero the momentum of a quiet cell),
 *  or one that writes the wrong slot. */
static void B5_launcher_applies_each_cell_and_skips_a_zero_request()
{
    Kokkos::ScopeGuard guard;
    const int n = 3;
    const double dx = 10.0, dt = 0.5, hextra = 1e-3;
    std::vector<double> h = {0.5, 0.4, 4e-3, 0.9}, qx = {0.3, 0.3, 0.3, 0.3}, qy = {-0.2, -0.2, -0.2, -0.2};
    std::vector<int> pos = {0, 1, 2};
    std::vector<double> q = {2.0, 0.0, 5.0}, applied(n, -1.0);
    const std::vector<double> h0 = h;

    double *dh, *dqx, *dqy, *dq, *dap;
    int* dpos;
    gpuMalloc((void**)&dh, sizeof(double) * h.size());
    gpuMalloc((void**)&dqx, sizeof(double) * h.size());
    gpuMalloc((void**)&dqy, sizeof(double) * h.size());
    gpuMalloc((void**)&dq, sizeof(double) * n);
    gpuMalloc((void**)&dap, sizeof(double) * n);
    gpuMalloc((void**)&dpos, sizeof(int) * n);
    gpuMemcpyAsync(dh, h.data(), sizeof(double) * h.size(), gpuMemcpyHostToDevice, 0);
    gpuMemcpyAsync(dqx, qx.data(), sizeof(double) * h.size(), gpuMemcpyHostToDevice, 0);
    gpuMemcpyAsync(dqy, qy.data(), sizeof(double) * h.size(), gpuMemcpyHostToDevice, 0);
    gpuMemcpyAsync(dq, q.data(), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
    gpuMemcpyAsync(dpos, pos.data(), sizeof(int) * n, gpuMemcpyHostToDevice, 0);
    Kokkos::fence();

    compute_driven_outfall_backflow<double>(n, dpos, dx, dt, dh, dqx, dqy, hextra, dq, dap);
    Kokkos::fence();

    gpuMemcpyAsync(h.data(), dh, sizeof(double) * h.size(), gpuMemcpyDeviceToHost, 0);
    gpuMemcpyAsync(qx.data(), dqx, sizeof(double) * h.size(), gpuMemcpyDeviceToHost, 0);
    gpuMemcpyAsync(qy.data(), dqy, sizeof(double) * h.size(), gpuMemcpyDeviceToHost, 0);
    gpuMemcpyAsync(applied.data(), dap, sizeof(double) * n, gpuMemcpyDeviceToHost, 0);
    Kokkos::fence();
    gpuFree(dh); gpuFree(dqx); gpuFree(dqy); gpuFree(dq); gpuFree(dap); gpuFree(dpos);

    CHECK(std::fabs(h[0] - (0.5 - 2.0 * 0.5 / 100.0)) <= 1e-15 && applied[0] == 2.0, "cell 0: the requested removal");
    CHECK(h[1] == 0.4 && qx[1] == 0.3 && qy[1] == -0.2 && applied[1] == 0.0, "cell 1: a zero request touches nothing");
    CHECK(h[2] == 0.0 && applied[2] < 5.0 && qx[2] == 0.0, "cell 2: the limiter takes what the cell had");
    CHECK(h[3] == h0[3], "a cell not listed is not written");
}

// ---------------------------------------------------------------------------
// outfall_stage.bin: header, record layout and resume positioning.
// ---------------------------------------------------------------------------

/** Property: the file starts with the 16-byte diag_log header {magic "DOS1", version, value width 8,
 *  n outfalls}, and a record is {f64 simtime, f64 dt, f64 stage[n], f64 q_out[n], f64 q_back[n]}:
 *  16 + 24n bytes with the stamp first, which decodes back to the same numbers.
 *  Killed by: any column reordered (the shape's layout is the reader's), or a stamp that is not
 *  the first 8 bytes (diag_log's positioners read it there). */
static void R1_header_and_record_layout()
{
    CHECK(MAGIC_OUTFALL_STAGE == 0x444F5331, "magic is ASCII DOS1");
    std::ostringstream hdr;
    diag_log::write_header(hdr, MAGIC_OUTFALL_STAGE, 2);
    const std::string hs = hdr.str();
    CHECK(hs.size() == diag_log::HEADER_BYTES, "a 16-byte header");
    int32_t f[4];
    std::memcpy(f, hs.data(), 16);
    CHECK(f[0] == 0x444F5331 && f[1] == diag_log::FORMAT_VERSION && f[2] == 8 && f[3] == 2,
          "magic, version, value width 8, n_items = 2");
    CHECK(std::memcmp(hs.data(), "1SOD", 4) == 0, "the magic reads DOS1 most-significant byte first, so '1SOD' in the file");

    const double stage[2] = {1.5, -2.25}, qout[2] = {3.0, -4.0}, qback[2] = {0.0, 4.0};
    const std::vector<char> rec = encode_outfall_stage_record(12.5, 0.25, stage, qout, qback, 2);
    CHECK(rec.size() == outfall_stage_record_bytes(2) && rec.size() == 16 + 24 * 2, "16 + 24n bytes");
    double v[8];
    std::memcpy(v, rec.data(), sizeof(v));
    CHECK(v[0] == 12.5 && v[1] == 0.25, "simtime first, then dt");
    CHECK(v[2] == 1.5 && v[3] == -2.25, "then the n stages");
    CHECK(v[4] == 3.0 && v[5] == -4.0, "then the n discharges (positive to the open water)");
    CHECK(v[6] == 0.0 && v[7] == 4.0, "then the n applied backflow rates");
    double s2[2], q2[2], b2[2], dt2 = 0.0;
    const double t = decode_outfall_stage_record(rec.data(), 2, s2, q2, b2, &dt2);
    CHECK(t == 12.5 && dt2 == 0.25 && s2[0] == 1.5 && s2[1] == -2.25 && q2[0] == 3.0 && q2[1] == -4.0 && b2[0] == 0.0 && b2[1] == 4.0,
          "decode is the inverse of encode");
}

namespace {
std::string stage_path()
{
    const char* tmp = std::getenv("TMPDIR");
    return std::string(tmp ? tmp : "/tmp") + "/test_driven_outfall_" + std::to_string((long)getpid()) + ".bin";
}

/** A clean file: the header and `records` records (stamps 1..records, content a function of the stamp). */
std::string clean_file_bytes(int n, int records)
{
    std::ostringstream out;
    diag_log::write_header(out, MAGIC_OUTFALL_STAGE, n);
    for (int k = 1; k <= records; ++k) {
        std::vector<double> s(n), q(n), b(n);
        for (int i = 0; i < n; ++i) { s[i] = k + 0.1 * i; q[i] = -k * 0.5 + i; b[i] = k * 0.25; }
        const std::vector<char> rec = encode_outfall_stage_record((double)k, 1.0, s.data(), q.data(), b.data(), n);
        out.write(rec.data(), (std::streamsize)rec.size());
    }
    return out.str();
}

void write_bytes(const std::string& path, const std::string& bytes)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), (std::streamsize)bytes.size());
}

std::string read_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}
}

/** Property: positioning by the side-file's record count truncates the file to exactly that many
 *  complete records, whatever the file's tail holds: a tail of whole records past the count (a
 *  longer earlier run), a partial record cut at EVERY byte offset, and no tail; and then appending
 *  the live segment's records gives the bytes of a clean run, the resume property itself.
 *  Why required: the stage file is a resume input as well as a capture: a replay reads it step by
 *  step, so a file left longer or shorter than the side-file shifts every later stage.
 *  Killed by: truncating by stamp in a build whose stamps drift from the side-file's count, a
 *  kept partial record, or a record dropped from before the count. */
static void R2_positioning_by_count_cut_at_every_offset()
{
    const int n = 2, total = 6, keep = 4;
    const std::string clean = clean_file_bytes(n, total);
    const std::size_t rec = outfall_stage_record_bytes(n);
    const std::string kept_prefix = clean.substr(0, diag_log::HEADER_BYTES + keep * rec);
    const std::string path = stage_path();

    bool all = true;
    // A tail that is the first `cut` bytes of the fifth record, for every cut including 0 and the whole record;
    // then also a tail of all the remaining whole records.
    for (std::size_t cut = 0; cut <= 2 * rec; ++cut) {
        write_bytes(path, clean.substr(0, diag_log::HEADER_BYTES + keep * rec + cut));
        const diag_log::position_result r = diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, n,
                                                                      outfall_stage_count_reader(n, keep), DBL_MAX);
        if (r.status != diag_log::position_status::ok || r.records != keep || read_bytes(path) != kept_prefix) {
            all = false;
            std::fprintf(stderr, "    cut %zu: status %d records %ld bytes %zu (expected %d, %zu)\n", cut, (int)r.status, r.records,
                         read_bytes(path).size(), keep, kept_prefix.size());
        }
    }
    CHECK(all, "truncated to exactly the first 4 records at every tail length 0..2 records");

    // The resume property: positioned file + the live segment's records == the clean run.
    write_bytes(path, clean.substr(0, diag_log::HEADER_BYTES + keep * rec + rec / 2));
    diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, n, outfall_stage_count_reader(n, keep), DBL_MAX);
    {
        std::ofstream app(path, std::ios::binary | std::ios::app);
        const std::string live = clean.substr(diag_log::HEADER_BYTES + keep * rec);
        app.write(live.data(), (std::streamsize)live.size());
    }
    CHECK(read_bytes(path) == clean, "positioned prefix + the live segment equals the clean run, byte for byte");

    // A file shorter than the count: fewer complete records than asked is reported by the record count.
    write_bytes(path, clean.substr(0, diag_log::HEADER_BYTES + 2 * rec + 5));
    const diag_log::position_result s = diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, n,
                                                                  outfall_stage_count_reader(n, keep), DBL_MAX);
    CHECK(s.status == diag_log::position_status::ok && s.records == 2, "a file with 2 complete records reports 2, which the caller refuses against a count of 4");
    std::remove(path.c_str());
}

/** Property: the positioner refuses a file written under another layout: another outfall count, a
 *  wrong magic, a short header, a missing file; never walks it with the wrong record stride.
 *  Killed by: a header check that is skipped when the count reader is used. */
static void R3_positioning_refuses_a_foreign_file()
{
    const std::string path = stage_path();
    write_bytes(path, clean_file_bytes(3, 4));
    CHECK(diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, 2, outfall_stage_count_reader(2, 4), DBL_MAX).status
              == diag_log::position_status::n_items_mismatch, "another outfall count");
    CHECK(diag_log::position_binary(path, diag_log::MAGIC_EDGE_FLUX, 3, outfall_stage_count_reader(3, 4), DBL_MAX).status
              == diag_log::position_status::bad_header, "another capture's magic");
    write_bytes(path, clean_file_bytes(3, 0).substr(0, 9));
    CHECK(diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, 3, outfall_stage_count_reader(3, 0), DBL_MAX).status
              == diag_log::position_status::bad_header, "a short header");
    std::remove(path.c_str());
    CHECK(diag_log::position_binary(path, MAGIC_OUTFALL_STAGE, 3, outfall_stage_count_reader(3, 0), DBL_MAX).status
              == diag_log::position_status::missing_file, "a missing file");
}

// ---------------------------------------------------------------------------
// Source structure: nothing of the feature runs with the key absent.
// ---------------------------------------------------------------------------

namespace {
std::vector<std::string> read_lines(const std::string& path)
{
    std::ifstream in(path);
    std::vector<std::string> v;
    std::string line;
    while (std::getline(in, line)) v.push_back(line);
    return v;
}
}

/** Property: every call site of the feature in src/triton.h is under the one condition
 *  swmm_model.driven_active() (the same on every rank), on its own line or on one of the three
 *  lines above it (a braced block of two calls), and the definitions and declarations are the
 *  only other occurrences. A call site outside the condition runs with the key absent and breaks
 *  the byte-identity of the default path; a collective outside it is entered by some ranks only.
 *  Control: the scan is vacuous unless it sees at least the eleven call sites of the two coupling
 *  blocks plus the ledger sites, and a synthetic unguarded call is flagged.
 *  Killed by: an unguarded driven_outfall_* call in either block. A second correct implementation:
 *  the guard moved inside the helper (each returns at once); it would pass the key-absent runs
 *  but not this scan, which is the intent: the guard stays at the call site, where S2/S3 of the
 *  surface-capture review also look for it. */
static void G1_every_call_site_is_guarded_by_driven_active()
{
    static const char* const calls[] = {"driven_outfall_read_stage(", "driven_outfall_set_stage(", "driven_outfall_scatter_backflow(",
                                        "driven_outfall_apply_backflow(", "log_driven_flows(", "diag_driven_outfall_ledger(",
                                        "driven_outfall_init("};
    const std::vector<std::string> lines = read_lines(TRITON_H_PATH);
    CHECK(!lines.empty(), "src/triton.h was read");

    auto unguarded = [&](const std::vector<std::string>& ls, std::size_t* sites) {
        std::vector<std::string> bad;
        *sites = 0;
        for (std::size_t i = 0; i < ls.size(); ++i) {
            for (const char* c : calls) {
                const std::size_t at = ls[i].find(c);
                if (at == std::string::npos) continue;
                const std::string before = ls[i].substr(0, at);
                if (ls[i].find("triton<T>::") != std::string::npos) continue;                // a definition
                const std::size_t first = ls[i].find_first_not_of(" \t");
                if (ls[i].compare(first, 2, "//") == 0) continue;                           // a comment
                if (before.find("void ") != std::string::npos) continue;                    // a declaration
                ++*sites;
                bool guarded = ls[i].find("driven_active()") != std::string::npos;
                for (std::size_t k = 1; k <= 3 && !guarded && k <= i; ++k)
                    guarded = ls[i - k].find("driven_active()") != std::string::npos;
                if (!guarded) bad.push_back("line " + std::to_string(i + 1) + ": " + ls[i]);
            }
        }
        return bad;
    };

    std::size_t sites = 0;
    const std::vector<std::string> bad = unguarded(lines, &sites);
    for (const std::string& b : bad) std::fprintf(stderr, "    unguarded: %s\n", b.c_str());
    CHECK(bad.empty(), "a driven-outfall call site in triton.h sits outside swmm_model.driven_active()");
    CHECK(sites >= 11, "the scan saw the call sites of both coupling blocks (vacuous below 11)");

    std::size_t ns = 0;
    const std::vector<std::string> synth = {
        "    // a comment naming driven_outfall_set_stage(",
        "    if (swmm_model.global_num_of_swmm_links > 0) {",
        "      driven_outfall_set_stage();",
        "    }"};
    CHECK(unguarded(synth, &ns).size() == 1 && ns == 1, "control: an unguarded call is flagged, a comment is not");
    const std::vector<std::string> synth_ok = {
        "      if (swmm_model.driven_active()) {",
        "        driven_outfall_scatter_backflow();",
        "        driven_outfall_apply_backflow();",
        "      }"};
    CHECK(unguarded(synth_ok, &ns).empty(), "control: a braced block of two calls under the condition passes");
}

/** Property: the feature leaves the existing exchange path's arithmetic where it is: the one-node
 *  branch of exchange_one_coupled_cell and coupled_node_flow still carry the manhole law's three
 *  cases verbatim, and swmm_triton.h still has exactly the two swmm_step call sites of the
 *  default and interval blocks plus the replay's, none of them new. A reader's check of the
 *  key-absent neutrality argument (no edit of the kernel the manhole nodes use), not its proof:
 *  N14 (a) and (b) on Rivanna are the proof.
 *  Killed by: an edit of the weir expression or an extra swmm_step call. */
static void G2_the_manhole_path_is_not_edited()
{
    const std::vector<std::string> lines = read_lines(std::string(TRITON_H_PATH).substr(0, std::string(TRITON_H_PATH).rfind('/')) + "/swmm_triton.h");
    std::string all;
    for (const std::string& l : lines) all += l + "\n";
    CHECK(all.find("flow=-(2.0/3.0)*loss*PI_*diam*sqrt(2.0*_G_*hij)*hij;") != std::string::npos, "Case 1 of the manhole law");
    CHECK(all.find("flow=-loss*areaM*sqrt(2.0*_G_*(hij + max_depth - new_depth));") != std::string::npos, "Case 2");
    CHECK(all.find("flow=loss*areaM*sqrt(2.0*_G_*(new_depth - max_depth - hij ) );") != std::string::npos, "Case 3");
    std::size_t steps = 0, at = 0;
    while ((at = all.find("swmm_step(&elapsed", at)) != std::string::npos) { ++steps; at += 5; }
    CHECK(steps == 1, "one swmm_step call in swmm_triton.h, the replay's");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"L1_list_file_format", L1_list_file_format},
    {"L2_outfall_types_read_as_swmm_reads_them", L2_outfall_types_read_as_swmm_reads_them},
    {"L3_every_refusal_names_the_node", L3_every_refusal_names_the_node},
    {"L4_link_sign_follows_the_inp_orientation", L4_link_sign_follows_the_inp_orientation},
    {"L5_strip_owner_is_the_row_stripe_rule", L5_strip_owner_is_the_row_stripe_rule},
    {"S1_stage_rule_at_hextra", S1_stage_rule_at_hextra},
    {"S2_unit_conversions", S2_unit_conversions},
    {"B1_backflow_removes_the_requested_depth", B1_backflow_removes_the_requested_depth},
    {"B2_backflow_below_hextra_zeroes_the_momentum", B2_backflow_below_hextra_zeroes_the_momentum},
    {"B3_backflow_near_zero_takes_what_the_cell_had", B3_backflow_near_zero_takes_what_the_cell_had},
    {"B4_backflow_is_the_one_node_kernel_for_the_same_flow", B4_backflow_is_the_one_node_kernel_for_the_same_flow},
    {"B5_launcher_applies_each_cell_and_skips_a_zero_request", B5_launcher_applies_each_cell_and_skips_a_zero_request},
    {"R1_header_and_record_layout", R1_header_and_record_layout},
    {"R2_positioning_by_count_cut_at_every_offset", R2_positioning_by_count_cut_at_every_offset},
    {"R3_positioning_refuses_a_foreign_file", R3_positioning_refuses_a_foreign_file},
    {"G1_every_call_site_is_guarded_by_driven_active", G1_every_call_site_is_guarded_by_driven_active},
    {"G2_the_manhole_path_is_not_edited", G2_the_manhole_path_is_not_edited},
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
