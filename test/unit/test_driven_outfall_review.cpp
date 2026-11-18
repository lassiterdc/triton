/** @file test_driven_outfall_review.cpp
 *  @brief Reviewer tests for chunk 1j.1 (driven-stage open-water outfalls): the two properties
 *         the coder's test_driven_outfall.cpp does not state, one of the call ORDER of the
 *         feature inside the two coupling blocks of src/triton.h, one of the VOLUME the backflow
 *         kernel takes against what the outfall_stage.bin record and the T2 ledger book.
 *
 *  Approved-shape rows enforced (shape sections 17.2 item 7 and 17.3):
 *    RV1 -> the position table of 17.2 item 7 (stage read before any exchange kernel, stage
 *           collective after the coupling gather and before swmm_step, flow read after swmm_step,
 *           backflow kernel before st.stop(SWMM_TIME) and so before wet_dry, ledger after it);
 *    RV2 -> 17.3 (outfall_backflow_m3 is the volume the cells gave up, so the record's
 *           q_back_applied, the kernel's applied rate and the cell's depth change agree) and the
 *           rule 17.2 item 3 (the one-node kernel's near-zero rule: never more than the cell had).
 *
 *  Usage: test_driven_outfall_review [test name [path of a triton.h to read in place of TRITON_H_PATH]]
 */

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

#ifndef TRITON_H_PATH
#error "TRITON_H_PATH must name src/triton.h (target_compile_definitions in test/unit/CMakeLists.txt)"
#endif

using namespace SWMM_triton;

static int g_checks = 0;
static int g_failures = 0;
/** The file RV1 reads: src/triton.h, or argv[2] (a mutated copy, for the control run). */
static std::string g_triton_path = TRITON_H_PATH;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

static std::string read_text(const std::string& path)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string t = ss.str();
    std::string out;
    out.reserve(t.size());
    for (char c : t) if (c != '\r') out.push_back(c);
    return out;
}

/** The text with every // comment line removed, so a call named in prose is not a call. */
static std::string strip_line_comments(const std::string& t)
{
    std::istringstream in(t);
    std::string line, out;
    while (std::getline(in, line)) {
        const std::size_t first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line.compare(first, 2, "//") == 0) { out += "\n"; continue; }
        out += line + "\n";
    }
    return out;
}

// ---------------------------------------------------------------------------
// RV1: the order of the feature's calls inside each coupling block.
// ---------------------------------------------------------------------------

/** The two coupling blocks of compute_new_state, cut from the text: the interval/lag block runs
 *  from its guard to the first ledger call, the default block from there to the second. */
static bool cut_blocks(const std::string& src, std::string& t3, std::string& dflt)
{
    const std::string guard = "coupling_interval.active()) {";
    const std::string ledger = "diag_driven_outfall_ledger();";
    const std::size_t g = src.find(guard);
    if (g == std::string::npos) return false;
    const std::size_t l1 = src.find(ledger, g);
    if (l1 == std::string::npos) return false;
    const std::size_t l2 = src.find(ledger, l1 + ledger.size());
    if (l2 == std::string::npos) return false;
    t3 = src.substr(g, l1 + ledger.size() - g);
    dflt = src.substr(l1 + ledger.size(), l2 + ledger.size() - (l1 + ledger.size()));
    return true;
}

/** True when the tokens occur in the block in the order given, each searched after the previous.
 *  `where` receives the first token that is missing or out of order. */
static bool in_order(const std::string& block, const std::vector<std::string>& toks, std::string& where)
{
    std::size_t at = 0;
    for (const std::string& t : toks) {
        const std::size_t p = block.find(t, at);
        if (p == std::string::npos) { where = t; return false; }
        at = p + t.size();
    }
    return true;
}

static const std::vector<std::string>& position_table()
{
    // 17.2 item 7, in the order the table lists the work. The exchange kernel and the coupling
    // gather are the existing statements the new ones are positioned against.
    static const std::vector<std::string> toks = {
        "driven_outfall_read_stage();",            // before any exchange kernel of the step
        "compute_swmm_triton_exchange(",           // the existing exchange kernel
        "MPI_Gatherv(",                            // the existing coupling gather
        "driven_outfall_set_stage();",             // after that gather, before swmm_step
        "swmm_step(&swmm_local_elapsedTime",       // the existing step
        "log_driven_flows(",                       // the flow read and the record, after it
        "driven_outfall_scatter_backflow();",      // after the existing depth scatter
        "driven_outfall_apply_backflow();",        // before st.stop(SWMM_TIME), so before wet_dry
        "st.stop(SWMM_TIME);",
        "diag_driven_outfall_ledger();"            // after the bracket closes: DIAG_TIME, not SWMM_TIME
    };
    return toks;
}

/** Property: in BOTH coupling blocks the feature's calls and the existing statements they are
 *  positioned against occur in the order of the shape's position table (17.2 item 7).
 *  Input class: the text of src/triton.h between the interval-block guard and each ledger call.
 *  Why required: the stage must be set before the step that reads it (a stage set after
 *  swmm_step is applied one step late and the record would no longer pair with the flow); the
 *  backflow must leave the cell before wet_dry (the exchange-before-wet_dry rule of the file's own
 *  comment) or the post-backflow depth is not the depth wet_dry enforces; the ledger call opens
 *  DIAG_TIME and must follow st.stop(SWMM_TIME) so no two categories are open.
 *  Killed by: moving driven_outfall_set_stage() below swmm_step, or the backflow kernel below
 *  st.stop(SWMM_TIME), in either block. Second correct implementation: the same calls wrapped in
 *  one helper per phase, as long as the helpers are called in this order; this test reads the
 *  call sites, not the bodies. A FITNESS FUNCTION: it states an ordering over statements, not
 *  the arithmetic of any one of them. */
static void RV1_feature_calls_follow_the_position_table_in_both_blocks()
{
    const std::string src = strip_line_comments(read_text(g_triton_path));
    std::string t3, dflt;
    CHECK(cut_blocks(src, t3, dflt), "both coupling blocks were found (two ledger calls follow the interval guard)");
    if (t3.empty() || dflt.empty()) return;
    std::string where;
    CHECK(in_order(t3, position_table(), where), (std::string("interval block out of order near: ") + where).c_str());
    CHECK(in_order(dflt, position_table(), where), (std::string("default block out of order near: ") + where).c_str());

    // Controls that vary the PROPERTY: a block with set_stage moved below swmm_step, and one with
    // the backflow kernel moved below st.stop(SWMM_TIME), must be refused; the unmutated block passes.
    auto mutate_swap = [](std::string b, const std::string& a, const std::string& z) {
        const std::size_t pa = b.find(a), pz = b.find(z);
        if (pa == std::string::npos || pz == std::string::npos) return std::string();
        b.erase(pa, a.size());
        const std::size_t pz2 = b.find(z);
        b.insert(pz2 + z.size(), a);
        return b;
    };
    const std::string late_stage = mutate_swap(dflt, "driven_outfall_set_stage();", "swmm_step(&swmm_local_elapsedTime");
    CHECK(!late_stage.empty() && !in_order(late_stage, position_table(), where),
          "control: set_stage after swmm_step is refused");
    const std::string late_apply = mutate_swap(t3, "driven_outfall_apply_backflow();", "st.stop(SWMM_TIME);");
    CHECK(!late_apply.empty() && !in_order(late_apply, position_table(), where),
          "control: the backflow kernel after st.stop(SWMM_TIME) is refused");
    // Counted unit: call sites (not lines), per block, of each of the five new calls; both blocks carry each once.
    const char* once[] = {"driven_outfall_read_stage();", "driven_outfall_set_stage();", "log_driven_flows(",
                          "driven_outfall_scatter_backflow();", "driven_outfall_apply_backflow();"};
    for (const char* o : once) {
        std::size_t n3 = 0, nd = 0, at = 0;
        while ((at = t3.find(o, at)) != std::string::npos) { ++n3; at += 1; }
        at = 0;
        while ((at = dflt.find(o, at)) != std::string::npos) { ++nd; at += 1; }
        CHECK(n3 == 1 && nd == 1, (std::string("one call site of ") + o + " per block").c_str());
    }
}

// ---------------------------------------------------------------------------
// RV2: the volume the backflow takes, three ways.
// ---------------------------------------------------------------------------

/** Property: for every (depth, request, cell size, dt) the backflow kernel, the function that
 *  log_driven_flows runs on the depth the stage was read from, and the cell's own depth change
 *  agree on the volume: applied (the rate in the record) equals the launcher's applied bit for
 *  bit, (h_before - h_after)*dx^2 equals applied*dt to round-off, applied never exceeds the
 *  request except by the zeroed cell's sub-EPS12 residual, a cell that ends below EPS12 gives up exactly what it had, and a request of zero
 *  leaves depth and momentum alone.
 *  Input class: depths straddling EPS12 and hextra (0, 1e-13, 1e-11, 5e-4, hextra, 2e-3, 0.5, 3.7 m),
 *  requests from 1e-9 to 40 m3/s, cells of 5 m and 0.35 m, steps of 0.002 s to 1 s: 8*5*2*3 = 240 inputs.
 *  Why required: outfall_backflow_m3 of the T2 balance and the record's q_back_applied are booked
 *  from these numbers; any disagreement is a false residual (17.3).
 *  Killed by: a launcher that applies the request instead of the limited rate, or a record that
 *  books the request. Second correct implementation: a kernel that clamps by min(q, hold*dx^2/dt)
 *  before the update instead of after; it satisfies every assertion here except the bitwise
 *  equality to the one-node rule, which is asserted only against driven_outfall_backflow_cell
 *  itself (the same function both call sites use). */
static void RV2_backflow_volume_agrees_between_kernel_record_and_cell()
{
    Kokkos::ScopeGuard guard;
    const double hs[] = {0.0, 1e-13, 1e-11, 5e-4, 1e-3, 2e-3, 0.5, 3.7};
    const double qs[] = {1e-9, 1e-4, 0.3, 5.0, 40.0};
    const double dxs[] = {5.0, 0.35};
    const double dts[] = {0.002, 0.11, 1.0};
    const double hextra = 1e-3;
    int inputs = 0, bad = 0, positive = 0, clipped = 0;
    for (double dx : dxs) for (double dt : dts) for (double q : qs) {
        const int n = 8;
        std::vector<double> h(hs, hs + n), qx(n, 0.3), qy(n, -0.2), req(n, q), applied(n, -1.0);
        std::vector<int> pos(n);
        for (int i = 0; i < n; ++i) pos[i] = i;
        const std::vector<double> h0 = h;
        double *dh, *dqx, *dqy, *dq, *dap; int* dpos;
        gpuMalloc((void**)&dh, sizeof(double) * n);  gpuMalloc((void**)&dqx, sizeof(double) * n);
        gpuMalloc((void**)&dqy, sizeof(double) * n); gpuMalloc((void**)&dq, sizeof(double) * n);
        gpuMalloc((void**)&dap, sizeof(double) * n); gpuMalloc((void**)&dpos, sizeof(int) * n);
        gpuMemcpyAsync(dh, h.data(), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
        gpuMemcpyAsync(dqx, qx.data(), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
        gpuMemcpyAsync(dqy, qy.data(), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
        gpuMemcpyAsync(dq, req.data(), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
        gpuMemcpyAsync(dpos, pos.data(), sizeof(int) * n, gpuMemcpyHostToDevice, 0);
        Kokkos::fence();
        compute_driven_outfall_backflow<double>(n, dpos, dx, dt, dh, dqx, dqy, hextra, dq, dap);
        Kokkos::fence();
        gpuMemcpyAsync(h.data(), dh, sizeof(double) * n, gpuMemcpyDeviceToHost, 0);
        gpuMemcpyAsync(qx.data(), dqx, sizeof(double) * n, gpuMemcpyDeviceToHost, 0);
        gpuMemcpyAsync(applied.data(), dap, sizeof(double) * n, gpuMemcpyDeviceToHost, 0);
        Kokkos::fence();
        gpuFree(dh); gpuFree(dqx); gpuFree(dqy); gpuFree(dq); gpuFree(dap); gpuFree(dpos);

        for (int i = 0; i < n; ++i) {
            ++inputs;
            // the record's path: log_driven_flows runs the cell function on the depth the stage was read from
            double hij = h0[i], rqx = 0.0, rqy = 0.0, rec = 0.0;
            driven_outfall_backflow_cell<double>(q, dx, dt, hextra, hij, rqx, rqy, rec);
            const double removed = (h0[i] - h[i]) * dx * dx;
            const double booked = applied[i] * dt;
            const bool eq_record = (rec == applied[i]) && (hij == h[i]);
            // the tolerance is relative to the volume the cell held (h0*dx^2): (h0 - h1) cancels, so a tolerance relative to the difference itself would refuse a correct removal of 1e-9 m3/s from a 3.7 m cell
            const bool vol = std::fabs(removed - booked) <= 4e-15 * std::max(h0[i] * dx * dx, booked) + 1e-300;
            // A cell left at depth > 0 gave up the request exactly. A cell the rule zeroed gave up what it had, which
            // exceeds the request only by the sub-EPS12 residual the one-node kernel's rule carries by design
            // (statement for statement, shape 17.2 item 3): hold <= q*dt/dx^2 + EPS12.
            const bool cap = applied[i] >= 0.0 && (h[i] > 0.0 ? applied[i] <= q * (1.0 + 1e-15)
                                                              : h0[i] <= q * dt / (dx * dx) + EPS12 * (1.0 + 1e-9));
            const bool dry_all = !(h[i] == 0.0 && h0[i] > 0.0) || std::fabs(booked - h0[i] * dx * dx) <= 4e-15 * h0[i] * dx * dx + 1e-300;
            const bool mom = (h[i] < hextra) ? (qx[i] == 0.0) : (qx[i] == 0.3);
            if (applied[i] > 0.0) ++positive;
            if (h[i] == 0.0 && h0[i] > 0.0) ++clipped;
            if (!(eq_record && vol && cap && dry_all && mom)) {
                ++bad;
                std::fprintf(stderr, "    dx %g dt %g q %g h0 %g: rec %.17g launcher %.17g h1 %.17g rec_h1 %.17g removed %.17g booked %.17g\n",
                             dx, dt, q, h0[i], rec, applied[i], h[i], hij, removed, booked);
            }
        }
    }
    CHECK(inputs == 240, "240 inputs were swept (8 depths x 5 requests x 2 cell sizes x 3 steps)");
    CHECK(bad == 0, "kernel, record function and the cell's depth change agree on the volume at every input");
    CHECK(positive > 150 && clipped > 20, "the sweep reached both a limited and an unlimited removal (must-be-nonzero controls)");

    // Control that varies the property: a record that books the REQUEST instead of the applied rate
    // disagrees with the cell's depth change at the dry-limited inputs.
    double hij = 1e-11, qx_ = 0.3, qy_ = 0.0, ap = 0.0;
    driven_outfall_backflow_cell<double>(0.3, 5.0, 1.0, 1e-3, hij, qx_, qy_, ap);
    CHECK(std::fabs(0.3 * 1.0 - (1e-11 - hij) * 25.0) > 1e-3, "control: booking the request at a dry cell would not match the depth change");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"RV1_feature_calls_follow_the_position_table_in_both_blocks", RV1_feature_calls_follow_the_position_table_in_both_blocks},
    {"RV2_backflow_volume_agrees_between_kernel_record_and_cell", RV2_backflow_volume_agrees_between_kernel_record_and_cell},
};

int main(int argc, char** argv)
{
    const char* want = (argc > 1) ? argv[1] : nullptr;
    if (argc > 2) g_triton_path = argv[2];
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
