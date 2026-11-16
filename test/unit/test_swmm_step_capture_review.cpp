/** @file test_swmm_step_capture_review.cpp
 *  @brief Reviewer test for chunk 2a.1: the SWMM step-capture getters
 *         (swmm_getStepDiagnostics, swmm_getNonOutfallNodeCount,
 *         swmm_getNodeStepDiagnostics, swmm_getNonConvergedNodes), driven through
 *         the public swmm5.h API on a generated dynamic-wave network.
 *
 *  Why this file exists. The chunk adds four exported getters over statics that
 *  live in dynwave.c, and no unit test of the coder's calls them: the chunk's
 *  coder wrote its checks as a standalone harness outside the repository. The
 *  four properties below are the ones a TRITON-side writer (chunk 2a.2) will
 *  silently depend on, and each fails by writing a plausible but wrong file
 *  rather than by crashing:
 *    - the numbering of per-node values (non-outfall ordinal, not SWMM node
 *      index) shifts every later column if an outfall is counted;
 *    - a domain sum taken in thread order differs between THREADS 1 and 8 in its
 *      last bits, so the two runs write different bytes;
 *    - a step that skipped dynamic wave routing must read zero, not the
 *      previous step's values;
 *    - a refused call must not write its buffer.
 *
 *  Venue. This program runs the solver, so it is built and run on Rivanna only.
 *  It uses only symbols exported by swmm5.h, and links the swmm5 target alone
 *  (no Kokkos, no MPI). It is not registered in test/unit/CMakeLists.txt by this
 *  review; the registration is the coordinator's (see the reviewer's return).
 *
 *  Usage: test_swmm_step_capture_review [case name]   (no argument: every case)
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "swmm5.h"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, ...)                                                \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(cond)) {                                                      \
      ++g_failures;                                                     \
      std::printf("  FAIL %s:%d  %s  --  ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                         \
      std::printf("\n");                                                \
    }                                                                   \
  } while (0)

// Deterministic generator so that every run, and every thread count, sees the
// same model.
struct Lcg {
  unsigned long long s = 1442695040888963407ULL;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return double(s >> 11) / 9007199254740992.0;
  }
};

// Where the outfalls sit in SWMM's node-index order. "Interleaved" puts one
// outfall before the first junction, one inside the junction run and one between
// the two storage nodes; "trailing" puts all three after every other node. The
// non-outfall nodes keep one relative order in both.
enum class Layout { Interleaved, Trailing };

/** Writes a branching dynamic-wave network of 358 junctions-and-storage nodes
 *  plus 3 outfalls. Three chains of 120 nodes each drain to one outfall (SWMM allows
 *  one inlet link per outfall);
 *  every chain slopes downstream so the topology is the same under both
 *  layouts, and only the node-index order differs. Conduits and inflows are
 *  written in an order that does not depend on the layout, so the two layouts
 *  describe one physical model. */
void write_model(const std::string& path, Layout layout, int threads) {
  const int M = 360;  // non-outfall nodes: M-2 junctions J*, storages S0 and S1
  const int len = 120;
  Lcg rng;
  std::vector<std::string> nm(M);
  for (int i = 0; i < M - 2; ++i) nm[i] = "J" + std::to_string(i);
  nm[M - 2] = "S0";
  nm[M - 1] = "S1";
  std::vector<int> perm(M), chain(M), pos(M);
  for (int i = 0; i < M; ++i) perm[i] = i;
  for (int i = M - 1; i > 0; --i) {
    int j = int(rng.next() * (i + 1));
    std::swap(perm[i], perm[j]);
  }
  for (int k = 0; k < M; ++k) {
    chain[perm[k]] = k / len;
    pos[perm[k]] = k % len;
  }
  auto elev = [&](int i) { return 60.0 - 0.5 * pos[i]; };
  const double outElev = 60.0 - 0.5 * len - 2.0;
  std::FILE* f = std::fopen(path.c_str(), "w");
  std::fprintf(f,
               "[TITLE]\nstep capture review\n\n[OPTIONS]\nFLOW_UNITS CFS\n"
               "INFILTRATION HORTON\nFLOW_ROUTING DYNWAVE\nSTART_DATE 01/01/2020\n"
               "START_TIME 00:00:00\nREPORT_START_DATE 01/01/2020\n"
               "REPORT_START_TIME 00:00:00\nEND_DATE 01/01/2020\nEND_TIME 00:30:00\n"
               "SWEEP_START 01/01\nSWEEP_END 12/31\nDRY_DAYS 0\nREPORT_STEP 00:01:00\n"
               "WET_STEP 00:01:00\nDRY_STEP 00:01:00\nROUTING_STEP 0:00:01\n"
               "ALLOW_PONDING YES\nINERTIAL_DAMPING PARTIAL\nNORMAL_FLOW_LIMITED BOTH\n"
               "FORCE_MAIN_EQUATION H-W\nSURCHARGE_METHOD EXTRAN\n"
               "VARIABLE_STEP 0\nLENGTHENING_STEP 0\nMIN_SURFAREA 12.557\n"
               "MAX_TRIALS 4\nHEAD_TOLERANCE 0.005\nSYS_FLOW_TOL 5\nLAT_FLOW_TOL 5\n"
               "MINIMUM_STEP 0.5\nTHREADS %d\n\n",
               threads);
  auto outfall = [&](const char* id) {
    std::fprintf(f, "[OUTFALLS]\n%s %.3f FREE NO\n\n", id, outElev);
  };
  auto junctions = [&](int lo, int hi) {
    std::fprintf(f, "[JUNCTIONS]\n");
    for (int i = lo; i < hi; ++i) std::fprintf(f, "%s %.3f 5 0 0 0\n", nm[i].c_str(), elev(i));
    std::fprintf(f, "\n");
  };
  auto storage = [&](int i) {
    std::fprintf(f, "[STORAGE]\n%s %.3f 6 0 FUNCTIONAL 800 0 0 0 0 0\n\n", nm[i].c_str(), elev(i));
  };
  if (layout == Layout::Interleaved) {
    outfall("OA");
    junctions(0, 90);
    outfall("OB");
    junctions(90, M - 2);
    storage(M - 2);
    outfall("OC");
    storage(M - 1);
  } else {
    junctions(0, M - 2);
    storage(M - 2);
    storage(M - 1);
    outfall("OA");
    outfall("OB");
    outfall("OC");
  }
  const char* outs[3] = {"OA", "OB", "OC"};
  std::vector<std::vector<int>> byChain(M / len + 1);
  for (int k = 0; k < M; ++k) byChain[chain[perm[k]]].push_back(perm[k]);
  std::fprintf(f, "[CONDUITS]\n");
  for (size_t c = 0; c < byChain.size(); ++c) {
    std::vector<int> v(byChain[c].size());
    for (int p = 0; p < int(v.size()); ++p) v[pos[byChain[c][p]]] = byChain[c][p];
    for (int p = 0; p < int(v.size()); ++p) {
      const char* to = (p + 1 == int(v.size())) ? outs[c % 3] : nm[v[p + 1]].c_str();
      std::fprintf(f, "C%zu_%d %s %s 200 0.013 0 0 0 0\n", c, p, nm[v[p]].c_str(), to);
    }
  }
  std::fprintf(f, "\n[XSECTIONS]\n");
  for (size_t c = 0; c < byChain.size(); ++c)
    for (int p = 0; p < int(byChain[c].size()); ++p)
      std::fprintf(f, "C%zu_%d CIRCULAR %.1f 0 0 0 1\n", c, p, (c % 2) ? 1.5 : 1.0);
  std::fprintf(f, "\n[DWF]\n");
  for (int i = 0; i < M - 2; ++i)
    std::fprintf(f, "%s FLOW %.6f\n", nm[i].c_str(),
                 (i % 17 == 5) ? 14.0 + 6.0 * rng.next() : 4.0 * rng.next());
  std::fprintf(f, "\n[COORDINATES]\n");
  std::fclose(f);
}

struct StepRead {
  int rc = -99;
  int trials = -99;
  int nNonConverged = -99;
  int nodeRc = -99;  // what swmm_getNodeStepDiagnostics returned for a buffer of exactly the count
  double p3Pos = -99, p3Neg = -99, linkVolume = -99;
  std::vector<double> p3;         // one per non-outfall ordinal
  std::vector<int> nonConverged;  // ascending non-outfall ordinals
};

struct Run {
  std::vector<StepRead> steps;  // steps that routed (elapsed time non-zero)
  StepRead past;                // the step that returned elapsed time 0
  int nodeCount = 0, nonOutfall = 0;
  std::vector<std::string> nonOutfallNames;  // swmm_getName order, outfalls skipped
  std::vector<int> outfallIndex;
  bool opened = false;
};

StepRead read_getters() {
  StepRead r;
  r.rc = swmm_getStepDiagnostics(&r.trials, &r.nNonConverged, &r.p3Pos, &r.p3Neg, &r.linkVolume);
  int n = swmm_getNonOutfallNodeCount();
  r.p3.assign(n > 0 ? n : 0, -77.0);
  r.nodeRc = n > 0 ? swmm_getNodeStepDiagnostics(r.p3.data(), n) : n;
  std::vector<int> ord(r.nNonConverged > 0 ? r.nNonConverged : 1, -1);
  int m = swmm_getNonConvergedNodes(ord.data(), int(ord.size()));
  if (m > 0) r.nonConverged.assign(ord.begin(), ord.begin() + m);
  return r;
}

/** Runs the model to the end of its duration, reading the getters after every
 *  step. The dt pattern varies per step so a getter that cached the previous
 *  step's length would show. */
Run run_model(Layout layout, int threads, const char* tag) {
  Run run;
  const char* tmp = std::getenv("TMPDIR");
  std::string base = std::string(tmp ? tmp : "/tmp") + "/swmm_capture_review_" + tag;
  std::string inp = base + ".inp", rpt = base + ".rpt", out = base + ".out";
  write_model(inp, layout, threads);
  if (swmm_open(inp.c_str(), rpt.c_str(), out.c_str()) != 0) return run;
  run.opened = true;
  run.nodeCount = swmm_getCount(swmm_NODE);
  for (int i = 0; i < run.nodeCount; ++i) {
    char name[64];
    swmm_getName(swmm_NODE, i, name, 64);
    if (int(swmm_getValue(swmm_NODE_TYPE, i)) == swmm_OUTFALL)
      run.outfallIndex.push_back(i);
    else
      run.nonOutfallNames.push_back(name);
  }
  run.nonOutfall = int(run.nonOutfallNames.size());
  if (swmm_start(0) != 0) return run;
  static const double dts[5] = {2.0, 3.0, 5.0, 4.0, 2.5};
  double t = 0.0;
  for (int s = 0;; ++s) {
    if (swmm_step(&t, nullptr, nullptr, dts[s % 5]) != 0) break;
    run.steps.push_back(read_getters());
    if (t == 0.0) {
      // The step that returns elapsed time 0 is the LAST routed step: routing runs
      // first and the clock then passes the duration. Only the step after it is
      // past the routing duration and skips dynamic wave routing.
      swmm_step(&t, nullptr, nullptr, dts[s % 5]);
      run.past = read_getters();
      break;
    }
  }
  swmm_end();
  swmm_close();
  return run;
}

// ---------------------------------------------------------------- the cases --

/** Property: outside a solve the getters refuse or read zero, never garbage.
 *  Class: every lifecycle point of one project (not open, open, started with no
 *  step taken, ended), with output slots pre-filled with a sentinel.
 *  Why required: the writer in chunk 2a.2 calls these on rank 0 only; a wrong
 *  return at start-up or after the end writes a record of the sentinel.
 *  Killed by: dropping the IsOpenFlag/IsStartedFlag guards in swmm5.c, or
 *  returning 0 before the dynamic wave state exists.
 *  A second correct implementation: deciding rc 1 from the dynamic wave state
 *  pointers (Xnode == NULL) instead of IsOpenFlag/IsStartedFlag returns the same
 *  value at all four lifecycle points and passes; the case reads returns, not the
 *  guard. */
void case_contracts_outside_a_solve() {
  int tr = -7, nn = -7;
  double a = -7, b = -7, c = -7;
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, &b, &c) == 1 && tr == 0 && nn == 0 && a == 0.0 &&
            b == 0.0 && c == 0.0,
        "before open: rc 1 and every output zero");
  CHECK(swmm_getNonOutfallNodeCount() == 0, "before open: count 0");
  double buf[2] = {-7, -7};
  int ord[2] = {-7, -7};
  CHECK(swmm_getNodeStepDiagnostics(buf, 2) == -1 && buf[0] == -7,
        "before open: per-node refused, buffer untouched");
  CHECK(swmm_getNonConvergedNodes(ord, 2) == -1 && ord[0] == -7,
        "before open: ordinals refused, buffer untouched");

  std::string base = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                     "/swmm_capture_review_contract";
  write_model(base + ".inp", Layout::Interleaved, 1);
  CHECK(swmm_open((base + ".inp").c_str(), (base + ".rpt").c_str(), (base + ".out").c_str()) == 0,
        "model opens");
  tr = nn = -7;
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, &b, &c) == 1 && tr == 0, "open, not started: rc 1");
  CHECK(swmm_start(0) == 0, "model starts");
  tr = nn = -7;
  a = b = -7;
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, &b, &c) == 0 && tr == 0 && nn == 0 && a == 0.0 &&
            b == 0.0,
        "started, no step taken: rc 0 and zeros (no solve has run)");
  swmm_end();
  tr = nn = -7;
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, &b, &c) == 1 && tr == 0, "after swmm_end: rc 1");
  swmm_close();
}

/** Property: a refused call writes nothing, a satisfied call writes exactly the
 *  count it returns, and a NULL argument is refused.
 *  Class: every buffer length from 0 to count+1 for both per-node getters, and
 *  every one of the five output pointers of swmm_getStepDiagnostics set NULL in
 *  turn, on a step that is both non-converged (list non-empty) and flooded.
 *  Why required: a short buffer overrun in the writer corrupts the record
 *  that follows; capacity is the only protection a C caller has.
 *  Killed by: replacing `capacity < k` with `capacity <= k` or removing it.
 *  A second correct implementation: checking capacity once up front and then
 *  filling the caller's buffer with a single memcpy from a compact array passes
 *  every length; the case reads what the caller's buffer holds, not how it was
 *  filled. */
void case_argument_and_capacity_refusals() {
  Run run = run_model(Layout::Interleaved, 1, "refusals");
  CHECK(run.opened && !run.steps.empty(), "model ran");
  // Re-open to hold a project at a chosen step: the refusal checks need a live
  // project, which run_model closes. Step to the first non-converged step again.
  std::string base = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                     "/swmm_capture_review_refusals";
  write_model(base + ".inp", Layout::Interleaved, 1);
  CHECK(swmm_open((base + ".inp").c_str(), (base + ".rpt").c_str(), (base + ".out").c_str()) == 0,
        "reopen");
  CHECK(swmm_start(0) == 0, "restart");
  static const double dts[5] = {2.0, 3.0, 5.0, 4.0, 2.5};
  double t = 0.0;
  int s = 0;
  StepRead r;
  for (;; ++s) {
    if (swmm_step(&t, nullptr, nullptr, dts[s % 5]) != 0 || t == 0.0) break;
    r = read_getters();
    if (r.nNonConverged > 0) break;
  }
  CHECK(r.nNonConverged > 0, "reached a non-converged step to test refusals on");
  int n = swmm_getNonOutfallNodeCount();
  CHECK(n == run.nonOutfall && n > 0, "count equals the non-outfall node count (%d vs %d)", n,
        run.nonOutfall);
  for (int cap = 0; cap <= n + 1; ++cap) {
    std::vector<double> v(n + 2, -9.0);
    int rc = swmm_getNodeStepDiagnostics(v.data(), cap);
    int written = 0;
    for (int i = 0; i < n + 2; ++i) written += (v[i] != -9.0) ? 1 : 0;
    if (cap < n) {
      CHECK(rc == -1 && written == 0,
            "capacity %d < %d: refused, nothing written (rc %d, wrote %d)", cap, n, rc, written);
    } else {
      CHECK(rc == n && v[n] == -9.0 && v[n + 1] == -9.0,
            "capacity %d >= %d: wrote exactly %d (rc %d)", cap, n, n, rc);
    }
  }
  CHECK(swmm_getNodeStepDiagnostics(nullptr, n) == -1, "NULL path3 refused");
  int m = r.nNonConverged;
  for (int cap = 0; cap <= m + 1; ++cap) {
    std::vector<int> v(m + 2, -9);
    int rc = swmm_getNonConvergedNodes(v.data(), cap);
    int written = 0;
    for (int i = 0; i < m + 2; ++i) written += (v[i] != -9) ? 1 : 0;
    if (cap < m) {
      CHECK(rc == -1 && written == 0, "ordinal capacity %d < %d: refused, nothing written", cap, m);
    } else {
      CHECK(rc == m && written == m,
            "ordinal capacity %d >= %d: wrote exactly %d (rc %d, wrote %d)", cap, m, m, rc,
            written);
    }
  }
  CHECK(swmm_getNonConvergedNodes(nullptr, n) == -1, "NULL ordinals refused");
  int tr;
  double a, b, c;
  int nn;
  CHECK(swmm_getStepDiagnostics(nullptr, &nn, &a, &b, &c) == 2, "NULL trials -> 2");
  CHECK(swmm_getStepDiagnostics(&tr, nullptr, &a, &b, &c) == 2, "NULL nNonConverged -> 2");
  CHECK(swmm_getStepDiagnostics(&tr, &nn, nullptr, &b, &c) == 2, "NULL p3Pos -> 2");
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, nullptr, &c) == 2, "NULL p3Neg -> 2");
  CHECK(swmm_getStepDiagnostics(&tr, &nn, &a, &b, nullptr) == 2, "NULL linkVolume -> 2");
  swmm_end();
  swmm_close();
}

/** Property: the step that did not run dynamic wave routing (the step taken past
 *  the routing duration) reads zero trials, zero non-converged, zero path-3,
 *  after a run in which the previous step read non-zero values.
 *  Class: the first swmm_step whose returned elapsed time is 0, on a model whose
 *  previous step is verified non-zero in the same run (the control that the
 *  zero is the guard's doing and not an absence of data).
 *  Why required: without the guard the getter returns the last routed step's
 *  values again; the T1a writer would append a duplicate record at the end of a run.
 *  Killed by: deleting the dynwave_resetStepDiagnostics() call at the top of
 *  swmm_step, or the DiagStepRan reads in the dynwave_get* functions.
 *  A second correct implementation: resetting the capture arrays (not a flag) at
 *  the top of swmm_step would still pass; the case asserts on what the getters
 *  return and says nothing about how the guard is kept. */
void case_no_solve_step_reads_zero() {
  Run run = run_model(Layout::Interleaved, 1, "nosolve");
  CHECK(run.opened && run.steps.size() > 100, "ran %zu routed steps", run.steps.size());
  const StepRead& last = run.steps.back();
  bool lastNonZero = last.trials > 0;
  CHECK(lastNonZero, "control: the last routed step read trials %d (non-zero)", last.trials);
  const StepRead& p = run.past;
  CHECK(p.rc == 0 && p.trials == 0 && p.nNonConverged == 0 && p.p3Pos == 0.0 && p.p3Neg == 0.0,
        "past-the-end step: rc %d trials %d nnc %d pos %g neg %g", p.rc, p.trials, p.nNonConverged,
        p.p3Pos, p.p3Neg);
  bool allZero = int(p.p3.size()) == run.nonOutfall;
  for (double v : p.p3) allZero = allZero && v == 0.0;
  CHECK(allZero, "past-the-end per-node values all zero and sized %d", run.nonOutfall);
  CHECK(p.nonConverged.empty(), "past-the-end ordinal list empty");
}

/** Property: p3Pos and p3Neg are the serial sums, in non-outfall ordinal order,
 *  of the per-node values the same step returned, bit for bit.
 *  Class: every routed step of the run, at THREADS 1 and 8.
 *  Why required: the domain sums are written to swmm_step_diag.bin; if they are
 *  taken in thread order the file differs between THREADS 1 and 8 in its last
 *  bits (shape case N7), although the hydraulics are identical.
 *  Killed by: building with -DCMAKE_C_FLAGS=-DTRITON_DIAG_TEST_OMP_REDUCTION
 *  (T-R) or replacing the serial loop in dynwave_getStepDiagnostics by a
 *  `#pragma omp parallel for reduction(+:...)`.
 *  A second correct implementation: swmm5.c summing the per-node array that
 *  swmm_getNodeStepDiagnostics returns, left to right in ordinal order, instead of
 *  dynwave.c summing its own array, gives the same bits and passes. Not
 *  established: that a compensated summation would pass; it would differ in the
 *  last bits, and the shape's "serial in node-index order" (section 2, T1a) is
 *  what this case holds the code to. */
void case_domain_sums_equal_serial_sum_of_per_node_values() {
  for (int threads : {1, 8}) {
    Run run = run_model(Layout::Interleaved, threads, threads == 1 ? "sums1" : "sums8");
    int bad = 0, nonzero = 0;
    for (const StepRead& r : run.steps) {
      double pos = 0.0, neg = 0.0;
      for (double v : r.p3) {
        if (v > 0.0)
          pos += v;
        else if (v < 0.0)
          neg += v;
      }
      if (std::memcmp(&pos, &r.p3Pos, 8) != 0 || std::memcmp(&neg, &r.p3Neg, 8) != 0) ++bad;
      if (r.p3Pos != 0.0 || r.p3Neg != 0.0) ++nonzero;
    }
    CHECK(bad == 0,
          "THREADS %d: %d of %zu steps carry a domain sum that is not the serial sum of its "
          "per-node values",
          threads, bad, run.steps.size());
    CHECK(nonzero > 10, "control: %d steps carry a non-zero domain sum", nonzero);
  }
}

/** Property: every captured quantity of a run is identical between THREADS 1 and
 *  THREADS 8 (the sums bitwise, the per-node values bitwise, the non-converged
 *  lists exactly, the trial counts exactly).
 *  Class: every routed step of one model at two thread counts.
 *  Why required: SWMM's own hydraulics are thread-count independent (the shape's
 *  neutrality cases N7 rest on it); a capture that is not would make the
 *  capture files a function of the job's thread setting.
 *  Killed by: T-R (see above), or a per-thread scratch array indexed by thread
 *  id inside setNodeDepth.
 *  A second correct implementation: any capture that records the same per-node
 *  values passes, whatever it sums with. */
void case_capture_is_thread_count_independent() {
  Run r1 = run_model(Layout::Interleaved, 1, "thr1");
  Run r8 = run_model(Layout::Interleaved, 8, "thr8");
  CHECK(r1.steps.size() == r8.steps.size() && !r1.steps.empty(), "same step count: %zu vs %zu",
        r1.steps.size(), r8.steps.size());
  int diffSums = 0, diffNode = 0, diffList = 0, diffTrials = 0;
  size_t n = std::min(r1.steps.size(), r8.steps.size());
  for (size_t i = 0; i < n; ++i) {
    const StepRead& a = r1.steps[i];
    const StepRead& b = r8.steps[i];
    if (std::memcmp(&a.p3Pos, &b.p3Pos, 8) != 0 || std::memcmp(&a.p3Neg, &b.p3Neg, 8) != 0)
      ++diffSums;
    if (a.p3.size() != b.p3.size() || std::memcmp(a.p3.data(), b.p3.data(), a.p3.size() * 8) != 0)
      ++diffNode;
    if (a.nonConverged != b.nonConverged) ++diffList;
    if (a.trials != b.trials) ++diffTrials;
  }
  CHECK(diffSums == 0, "%d steps differ in a domain sum between THREADS 1 and 8", diffSums);
  CHECK(diffNode == 0, "%d steps differ in per-node values", diffNode);
  CHECK(diffList == 0, "%d steps differ in the non-converged list", diffList);
  CHECK(diffTrials == 0, "%d steps differ in the trial count", diffTrials);
}

/** Property: the ordinal numbering of per-node values and of the non-converged
 *  list is the position among NON-OUTFALL nodes only, so moving outfalls within
 *  SWMM's node-index order does not move any value to another ordinal.
 *  Class: one physical model written under two node-index layouts (outfalls
 *  interleaved among the junctions versus all after the last non-outfall node);
 *  per-ordinal series compared between layouts, and the ordinal -> name list the
 *  public name API implies is checked against the count.
 *  Why required: the sidecar swmm_node_diag.nodes lists names in this numbering
 *  and the hhemt decoder indexes by it (erratum E-10); an outfall counted in the
 *  numbering shifts every later node's column in every T1 file.
 *  Killed by: returning raw SWMM node indices from swmm_getNonConvergedNodes, or
 *  leaving an outfall's slot in swmm_getNodeStepDiagnostics.
 *  A second correct implementation: a getter that fills a node-index-sized
 *  buffer and a compact ordinal list internally would pass; the case asserts on
 *  returned values only. Not established: that the two layouts route bit-for-bit
 *  alike, so per-node series are compared with a 1e-9 relative tolerance, which a
 *  shifted numbering (values differing by orders of magnitude between
 *  neighbouring nodes) cannot meet. */
void case_ordinals_follow_non_outfall_order() {
  Run a = run_model(Layout::Interleaved, 1, "layoutA");
  Run b = run_model(Layout::Trailing, 1, "layoutB");
  CHECK(a.nonOutfall == 360 && b.nonOutfall == 360, "non-outfall counts %d and %d", a.nonOutfall,
        b.nonOutfall);
  CHECK(a.outfallIndex.size() == 3 && b.outfallIndex.size() == 3, "three outfalls in both layouts");
  CHECK(a.outfallIndex[0] == 0 && a.outfallIndex[2] > a.outfallIndex[0] + 100 &&
            a.outfallIndex[2] + 1 < a.nodeCount,
        "interleaved layout really interleaves (outfalls at %d, %d, %d of %d)", a.outfallIndex[0],
        a.outfallIndex[1], a.outfallIndex[2], a.nodeCount);
  CHECK(b.outfallIndex[0] == 360 && b.outfallIndex[2] == 362, "trailing layout puts outfalls last");
  CHECK(a.nonOutfallNames == b.nonOutfallNames,
        "both layouts list the same non-outfall names in the same order");
  size_t n = std::min(a.steps.size(), b.steps.size());
  CHECK(n > 100 && a.steps.size() == b.steps.size(), "same step count: %zu vs %zu", a.steps.size(),
        b.steps.size());
  int diffVal = 0, diffList = 0, listsSeen = 0, rcWrong = 0;
  for (size_t i = 0; i < n; ++i) {
    const StepRead& x = a.steps[i];
    const StepRead& y = b.steps[i];
    if (x.nodeRc != a.nonOutfall || y.nodeRc != b.nonOutfall) ++rcWrong;
    if (x.p3.size() != y.p3.size()) {
      ++diffVal;
      continue;
    }
    for (size_t k = 0; k < x.p3.size(); ++k)
      if (std::fabs(x.p3[k] - y.p3[k]) > 1e-9 * std::fmax(1.0, std::fabs(x.p3[k]))) {
        ++diffVal;
        break;
      }
    if (x.nonConverged != y.nonConverged) ++diffList;
    if (!x.nonConverged.empty()) ++listsSeen;
    for (size_t k = 0; k < x.nonConverged.size(); ++k) {
      CHECK(x.nonConverged[k] >= 0 && x.nonConverged[k] < a.nonOutfall &&
                (k == 0 || x.nonConverged[k] > x.nonConverged[k - 1]),
            "step %zu: ordinal %d not in range and ascending", i, x.nonConverged[k]);
    }
    CHECK(int(x.nonConverged.size()) == x.nNonConverged, "step %zu: list length %zu vs count %d", i,
          x.nonConverged.size(), x.nNonConverged);
  }
  CHECK(rcWrong == 0, "%d steps where the per-node getter did not return the non-outfall count",
        rcWrong);
  CHECK(diffVal == 0, "%d steps carry per-ordinal values that moved when only the outfalls moved",
        diffVal);
  CHECK(diffList == 0,
        "%d steps carry a non-converged list that moved when only the outfalls moved", diffList);
  CHECK(listsSeen > 5, "control: %d steps had a non-empty non-converged list", listsSeen);
}

/** Property: the model this file drives reaches every branch the other cases
 *  quantify over, so their zeros are claims about the getters and not about an
 *  inert model.
 *  Class: the generated model at THREADS 1 (steps with a non-empty
 *  non-converged list, a positive path-3 sum, a negative path-3 sum, and a
 *  non-zero trial count above one).
 *  Why required: a model that never fails to converge would let a getter that
 *  always returns an empty list pass every other case.
 *  Killed by: changing the generator's inflows so that no node floods.
 *  A second correct implementation: any model with these four step classes. */
void case_model_reaches_every_branch() {
  Run run = run_model(Layout::Interleaved, 1, "coverage");
  int nonconv = 0, pos = 0, neg = 0, multi = 0;
  for (const StepRead& r : run.steps) {
    if (r.nNonConverged > 0) ++nonconv;
    if (r.p3Pos > 0.0) ++pos;
    if (r.p3Neg < 0.0) ++neg;
    if (r.trials > 1) ++multi;
  }
  std::printf("  coverage: %zu steps; non-converged %d, positive %d, negative %d, trials>1 %d\n",
              run.steps.size(), nonconv, pos, neg, multi);
  CHECK(nonconv > 5, "steps with a non-converged node: %d", nonconv);
  CHECK(pos > 5, "steps with a positive path-3 sum: %d", pos);
  CHECK(neg > 5, "steps with a negative path-3 sum: %d", neg);
  CHECK(multi > 5, "steps with more than one trial: %d", multi);
}

struct Case {
  const char* name;
  void (*fn)();
};

const Case kCases[] = {
    {"contracts_outside_a_solve", case_contracts_outside_a_solve},
    {"argument_and_capacity_refusals", case_argument_and_capacity_refusals},
    {"no_solve_step_reads_zero", case_no_solve_step_reads_zero},
    {"domain_sums_equal_serial_sum_of_per_node_values",
     case_domain_sums_equal_serial_sum_of_per_node_values},
    {"capture_is_thread_count_independent", case_capture_is_thread_count_independent},
    {"ordinals_follow_non_outfall_order", case_ordinals_follow_non_outfall_order},
    {"model_reaches_every_branch", case_model_reaches_every_branch},
};

}  // namespace

int main(int argc, char** argv) {
  const char* want = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (const Case& c : kCases) {
    if (want && std::strcmp(want, c.name) != 0) continue;
    int before = g_failures;
    std::printf("CASE %s\n", c.name);
    c.fn();
    std::printf("CASE %s: %s\n", c.name, g_failures == before ? "ok" : "FAILED");
    ++ran;
  }
  if (ran == 0) {
    std::printf("no case named %s\n", want ? want : "(none)");
    return 2;
  }
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
