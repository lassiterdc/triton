/** @file test_outfall_stage_api.cpp
 *  @brief Tier-1 tests of the engine surface the driven-stage open-water outfalls stand on:
 *         the existing setter swmm_setValue(swmm_NODE_HEAD, ...), driven through the public
 *         swmm5.h API on small generated dynamic-wave models.
 *
 *  Chunk 1j.1 edits no SWMM source file (TC-25): TRITON sets a listed outfall's stage before
 *  every SWMM step through the setter the engine already has. That is sound only if the setter
 *  does what the design says it does, so these cases pin it, with the engine as built:
 *
 *    setter_writes_the_fixed_stage_in_model_units  -- the stage a caller hands in, in the model's
 *        length unit, is the outfall's head the next step reports, in a metric and in a US model;
 *    setter_leaves_a_junction_untouched            -- the same call on a junction changes nothing,
 *        bit for bit, over the whole run;
 *    stage_above_the_crown_reverses_the_link_flow  -- a stage above the outfall conduit's crown,
 *        with a dry sewer behind it, drives water INTO the sewer: the API flow of the one
 *        connected link goes negative along the .inp orientation (backflow);
 *    stage_below_the_invert_is_the_free_outfall    -- a stage below the outfall's invert gives the
 *        flow a FREE outfall gives, bit for bit;
 *    reversed_conduit_keeps_the_inp_orientation    -- a conduit the .inp states as J1 -> OUT whose
 *        outfall end lies higher is reversed by SWMM (its end nodes swapped) while the API flow
 *        keeps the .inp orientation, which is why the driven outfall's flow sign comes from the
 *        .inp's from/to names and not from the engine's end nodes.
 *
 *  Venue. This program runs the solver, so it is built and run on Rivanna only. It uses only
 *  symbols exported by swmm5.h and links the swmm5 target alone (no Kokkos, no MPI).
 *
 *  Usage: test_outfall_stage_api [case name]   (no argument: every case)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

/** The one-junction, one-conduit, one-outfall model the cases vary. Lengths are in the model's
 *  own unit (metres for a CMS model, feet for a CFS one); `scale` is that unit's length in the
 *  metric model, so one set of numbers serves both. */
struct Model {
  bool us = false;                 ///< CFS (lengths in feet) instead of CMS (metres)
  const char* outfall = "FIXED";   ///< FIXED or FREE
  double outfall_invert = 9.5;     ///< metric metres
  double junction_invert = 10.0;   ///< metric metres
  double inflow = 0.0;             ///< metric m3/s into the junction (a DWF)
  double stage0 = 9.5;             ///< the .inp's initial fixed stage, metric metres
  int hours = 1;
};

const double kFtPerM = 3.28084;

void write_model(const std::string& path, const Model& m) {
  const double L = m.us ? kFtPerM : 1.0;            // model length per metric metre
  const double Q = m.us ? 35.3146667 : 1.0;         // model flow per metric m3/s
  std::FILE* f = std::fopen(path.c_str(), "w");
  std::fprintf(f,
               "[TITLE]\noutfall stage api\n\n[OPTIONS]\nFLOW_UNITS %s\n"
               "INFILTRATION HORTON\nFLOW_ROUTING DYNWAVE\nSTART_DATE 01/01/2020\n"
               "START_TIME 00:00:00\nREPORT_START_DATE 01/01/2020\n"
               "REPORT_START_TIME 00:00:00\nEND_DATE 01/01/2020\nEND_TIME %02d:00:00\n"
               "SWEEP_START 01/01\nSWEEP_END 12/31\nDRY_DAYS 0\nREPORT_STEP 00:01:00\n"
               "WET_STEP 00:01:00\nDRY_STEP 00:01:00\nROUTING_STEP 0:00:01\n"
               "ALLOW_PONDING NO\nINERTIAL_DAMPING PARTIAL\nNORMAL_FLOW_LIMITED BOTH\n"
               "FORCE_MAIN_EQUATION H-W\nSURCHARGE_METHOD EXTRAN\n"
               "VARIABLE_STEP 0\nLENGTHENING_STEP 0\nMIN_SURFAREA 12.557\n"
               "MAX_TRIALS 8\nHEAD_TOLERANCE 0.0005\nSYS_FLOW_TOL 5\nLAT_FLOW_TOL 5\n"
               "MINIMUM_STEP 0.5\nTHREADS 1\n\n",
               m.us ? "CFS" : "CMS", m.hours);
  std::fprintf(f, "[JUNCTIONS]\nJ1 %.6f %.6f 0 0 0\n\n", m.junction_invert * L, 6.0 * L);
  if (std::strcmp(m.outfall, "FIXED") == 0)
    std::fprintf(f, "[OUTFALLS]\nOUT %.6f FIXED %.6f NO\n\n", m.outfall_invert * L, m.stage0 * L);
  else
    std::fprintf(f, "[OUTFALLS]\nOUT %.6f FREE NO\n\n", m.outfall_invert * L);
  std::fprintf(f, "[CONDUITS]\nC1 J1 OUT %.3f 0.013 0 0 0 0\n\n", 100.0 * L);
  std::fprintf(f, "[XSECTIONS]\nC1 CIRCULAR %.6f 0 0 0 1\n\n", 1.0 * L);
  if (m.inflow > 0.0) std::fprintf(f, "[DWF]\nJ1 FLOW %.9f\n\n", m.inflow * Q);
  std::fprintf(f, "[COORDINATES]\n");
  std::fclose(f);
}

/** What was read after one step. */
struct Read {
  double out_head = 0, j_head = 0, flow = 0;
};

struct Run {
  bool ok = false;
  int out_idx = -1, j_idx = -1, link_node1 = -1, link_node2 = -1;
  std::vector<Read> steps;
};

/** Runs `n_steps` steps of 2 s. Before each step `stage_m(step)` (metric metres, NaN for "do not call")
 *  is handed to swmm_setValue(swmm_NODE_HEAD) on the outfall in the model's length unit, and
 *  `poke_junction`, when true, hands 999 (model units) to the same call on the junction. */
Run run(const Model& m, const char* tag, int n_steps, const std::function<double(int)>& stage_m, bool poke_junction) {
  Run r;
  const char* tmp = std::getenv("TMPDIR");
  const std::string base = std::string(tmp ? tmp : "/tmp") + "/outfall_stage_api_" + tag;
  const std::string inp = base + ".inp", rpt = base + ".rpt", out = base + ".out";
  write_model(inp, m);
  if (swmm_open(inp.c_str(), rpt.c_str(), out.c_str()) != 0) return r;
  if (swmm_start(0) != 0) { swmm_close(); return r; }
  r.out_idx = swmm_getIndex(swmm_NODE, "OUT");
  r.j_idx = swmm_getIndex(swmm_NODE, "J1");
  const int link = swmm_getIndex(swmm_LINK, "C1");
  r.link_node1 = (int)swmm_getValue(swmm_LINK_NODE1, link);
  r.link_node2 = (int)swmm_getValue(swmm_LINK_NODE2, link);
  const double L = m.us ? kFtPerM : 1.0;
  double t = 0.0;
  for (int s = 0; s < n_steps; ++s) {
    const double stage = stage_m ? stage_m(s) : std::nan("");
    if (!std::isnan(stage)) swmm_setValue(swmm_NODE_HEAD, r.out_idx, stage * L);
    if (poke_junction) swmm_setValue(swmm_NODE_HEAD, r.j_idx, 999.0);
    if (swmm_step(&t, nullptr, nullptr, 2.0) != 0) break;
    Read rd;
    rd.out_head = swmm_getValue(swmm_NODE_HEAD, r.out_idx);
    rd.j_head = swmm_getValue(swmm_NODE_HEAD, r.j_idx);
    rd.flow = swmm_getValue(swmm_LINK_FLOW, link);
    r.steps.push_back(rd);
    if (t == 0.0) break;
  }
  swmm_end();
  swmm_close();
  r.ok = (int)r.steps.size() == n_steps;
  return r;
}

// ---------------------------------------------------------------- the cases --

/** Property: a stage handed to the setter, in the model's length unit, is the outfall's head the
 *  step reports, for a FIXED outfall in a metric model and in a US model; the head before the
 *  first call is the .inp's own stage, so the change is the setter's.
 *  Why required: TRITON converts a stage to the model's unit once, before the call, and relies on
 *  the engine to convert it back to its internal feet exactly once.
 *  Killed by: a unit factor applied twice (or not at all) in the setter. */
void case_setter_writes_the_fixed_stage_in_model_units() {
  for (int us = 0; us < 2; ++us) {
    Model m;
    m.us = us != 0;
    const double stage = 12.5;   // metric metres: 3 m over the outfall's invert, above the 1 m conduit's crown
    const Run before = run(m, us ? "a_us_before" : "a_si_before", 3, nullptr, false);
    const Run after = run(m, us ? "a_us_after" : "a_si_after", 3, [&](int) { return stage; }, false);
    const double L = us ? kFtPerM : 1.0;
    CHECK(before.ok && after.ok, "both runs complete (%s)", us ? "US" : "metric");
    if (!before.ok || !after.ok) continue;
    CHECK(std::fabs(before.steps[2].out_head - m.stage0 * L) <= 1e-6 * L,
          "without the call the outfall head is the .inp's stage (%s): %.9g", us ? "US" : "metric", before.steps[2].out_head);
    for (int s = 0; s < 3; ++s)
      CHECK(std::fabs(after.steps[s].out_head - stage * L) <= 1e-9 * stage * L,
            "step %d: the outfall head is the stage handed in, in the model's unit (%s): got %.12g want %.12g", s,
            us ? "US" : "metric", after.steps[s].out_head, stage * L);
  }
}

/** Property: the same call on a junction changes nothing, bit for bit over the run: the junction's
 *  head and the link flow are those of a run that never made the call.
 *  Why required: the setter is the engine's general one; chunk 1j.1 calls it only for listed
 *  OUTFALL nodes, and this fixes that a call on any other node (a mis-indexed outfall) would be
 *  inert instead of corrupting a head.
 *  Killed by: a setter that writes a junction's state. */
void case_setter_leaves_a_junction_untouched() {
  Model m;
  m.inflow = 0.2;
  const Run base = run(m, "b_base", 200, nullptr, false);
  const Run poked = run(m, "b_poked", 200, nullptr, true);
  CHECK(base.ok && poked.ok, "both runs complete");
  if (!base.ok || !poked.ok) return;
  bool same = true;
  for (std::size_t s = 0; s < base.steps.size(); ++s)
    if (base.steps[s].j_head != poked.steps[s].j_head || base.steps[s].flow != poked.steps[s].flow ||
        base.steps[s].out_head != poked.steps[s].out_head)
      same = false;
  CHECK(same, "junction head, link flow and outfall head are bit-identical with and without the call on the junction");
  CHECK(base.steps.back().flow > 0.0, "control: the model carries a flow, so identical zeros would prove nothing (flow %.6g)", base.steps.back().flow);
}

/** Property: with a dry sewer behind it and a stage well above the outfall conduit's crown, the API
 *  flow of the one connected link is negative along the .inp orientation (J1 -> OUT) from the first
 *  step: water enters the sewer through the outfall (backflow); at a stage at the invert (the dry-cell
 *  value) the flow stays at zero.
 *  Killed by: a stage the setter ignores, or an outfall flow that cannot reverse. */
void case_stage_above_the_crown_reverses_the_link_flow() {
  Model m;
  const Run dry = run(m, "c_dry", 40, [&](int) { return m.outfall_invert; }, false);
  const Run high = run(m, "c_high", 40, [&](int) { return 12.5; }, false);
  CHECK(dry.ok && high.ok, "both runs complete");
  if (!dry.ok || !high.ok) return;
  CHECK(std::fabs(dry.steps.back().flow) < 1e-9, "a stage at the invert and no inflow: no flow (%.3g)", dry.steps.back().flow);
  // The junction is tiny and fills in seconds, so the level equalises and rings: the reversal is read
  // at the first steps, while the head difference is still there, and the junction's rise is its evidence.
  CHECK(high.steps[0].flow < -1e-2, "a stage above the crown: the link flow is reversed, into the sewer, from the first step (%.6g)", high.steps[0].flow);
  CHECK(high.steps[10].j_head > m.junction_invert + 0.1, "the junction fills from the outfall side (head %.6g at step 10)", high.steps[10].j_head);
  CHECK(high.link_node2 == high.out_idx, "control: the conduit is not reversed here, so its second end is the outfall");
}

/** Property: a stage below the outfall's invert, with the conduit at the outfall's invert (no offset),
 *  gives the flow of a FREE outfall in the same model, bit for bit over an hour of an inflow: the
 *  driven outfall at a dry cell behaves as the model's own free outfall.
 *  Killed by: a stage below the invert read as a fixed level above the conduit. */
void case_stage_below_the_invert_is_the_free_outfall() {
  Model fixed;
  fixed.inflow = 0.2;
  Model freeo = fixed;
  freeo.outfall = "FREE";
  const int n = 1800;   // an hour of 2 s steps
  const Run a = run(fixed, "e_fixed", n, [&](int) { return fixed.outfall_invert - 2.0; }, false);
  const Run b = run(freeo, "e_free", n, nullptr, false);
  CHECK(a.ok && b.ok, "both runs complete");
  if (!a.ok || !b.ok) return;
  bool same = true;
  for (int s = 0; s < n; ++s)
    if (a.steps[s].flow != b.steps[s].flow || a.steps[s].j_head != b.steps[s].j_head) same = false;
  CHECK(same, "flow and junction head are bit-identical to the FREE outfall's over the hour");
  CHECK(a.steps.back().flow > 0.05, "control: the model discharges (%.6g), so identical zeros would prove nothing", a.steps.back().flow);
}

/** Property: for a conduit the .inp states as J1 -> OUT whose outfall end lies HIGHER than the junction
 *  (slope negative), dynamic wave routing reverses the conduit: the engine's end nodes are swapped
 *  (its first end is the outfall) while the API flow keeps the .inp orientation, positive from J1
 *  toward OUT. A discharge reads positive and a backflow (a stage above the junction's head, no inflow)
 *  reads negative, in both cases against the engine's own end nodes.
 *  Why required: a driven outfall's flow sign is taken from the .inp's from/to names; this is the
 *  case that makes that, and not the engine's ends, the right source.
 *  Killed by: a sign taken from swmm_LINK_NODE1/NODE2, which is wrong exactly here. */
void case_reversed_conduit_keeps_the_inp_orientation() {
  Model m;
  m.outfall_invert = 11.0;   // above the junction's invert of 10: negative slope, reversed under DW
  m.stage0 = 11.0;
  m.inflow = 0.2;
  const int hour = 1800;
  const Run discharge = run(m, "d_discharge", hour, [&](int) { return m.outfall_invert - 2.0; }, false);
  CHECK(discharge.ok, "the discharge run completes");
  if (discharge.ok) {
    CHECK(discharge.link_node1 == discharge.out_idx && discharge.link_node2 == discharge.j_idx,
          "premise: the engine reversed the conduit, so its first end is the outfall (%d, %d; outfall %d)",
          discharge.link_node1, discharge.link_node2, discharge.out_idx);
    CHECK(discharge.steps.back().flow > 0.05,
          "a discharge reads positive along the .inp orientation although the engine's second end is the junction (%.6g)",
          discharge.steps.back().flow);
  }
  Model q = m;
  q.inflow = 0.0;
  const Run backflow = run(q, "d_backflow", 60, [&](int) { return 14.0; }, false);
  CHECK(backflow.ok, "the backflow run completes");
  if (backflow.ok)
    CHECK(backflow.steps[0].flow < -1e-2, "a stage above the junction with no inflow reads negative along the .inp orientation (%.6g)",
          backflow.steps[0].flow);
}

}  // namespace

struct CaseEntry {
  const char* name;
  void (*fn)();
};

int main(int argc, char** argv) {
  const CaseEntry cases[] = {
      {"setter_writes_the_fixed_stage_in_model_units", case_setter_writes_the_fixed_stage_in_model_units},
      {"setter_leaves_a_junction_untouched", case_setter_leaves_a_junction_untouched},
      {"stage_above_the_crown_reverses_the_link_flow", case_stage_above_the_crown_reverses_the_link_flow},
      {"stage_below_the_invert_is_the_free_outfall", case_stage_below_the_invert_is_the_free_outfall},
      {"reversed_conduit_keeps_the_inp_orientation", case_reversed_conduit_keeps_the_inp_orientation},
  };
  const char* want = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (const CaseEntry& c : cases) {
    if (want && std::strcmp(want, c.name) != 0) continue;
    std::printf("RUN  %s\n", c.name);
    const int before = g_failures;
    c.fn();
    std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", c.name);
    ++ran;
  }
  if (want && ran == 0) {
    std::printf("ERROR: no case named '%s'\n", want);
    return 2;
  }
  std::printf("%d case(s), %d check(s), %d failure(s)\n", ran, g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
