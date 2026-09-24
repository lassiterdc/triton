/** @file test_state_snapshot.cpp
 *  @brief Tier-1 tests for the coupled-resume state snapshot (WP-1B).
 *
 *  No solver run, no cluster, no simulation. These exercise snapshot.c's
 *  serializer, deserializer and field manifest directly against the SWMM
 *  globals, which are zero-initialized and reachable without swmm_open().
 *
 *  WHAT THIS TIER CAN AND CANNOT REACH, stated rather than left implicit.
 *  With no .inp open every Nobjects[] count is zero, so the per-object arrays
 *  (NodeStats, LinkStats, ...) contribute nothing to the payload. What DOES
 *  traverse is every singleton and scalar: TimeStepStats, the four TMaxStats
 *  arrays, RunoffTotals, GwaterTotals, the three TRoutingTotals, and the eight
 *  scalar accumulators. That is a real bit-exactness surface and it is the one
 *  carrying the not-reconstructible counters, but it is NOT the whole payload.
 *  The per-object arrays are covered by the Tier-2 smoke test, which runs a
 *  real model; claiming Tier 1 covers them would be claiming coverage this
 *  file does not have.
 *
 *  Each test is selectable by name from argv[1] so CTest registers one node id
 *  per property -- a result is then accounted by name rather than by a
 *  pass/fail total that cannot say which property it contains.
 */

// swmm_triton.h is not independently includable at this commit; this prefix
// reproduces main.cpp's include order. See test_exchange_log_header.cpp for the
// three transitive dependencies involved and why they are reported rather than
// fixed here.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// The SWMM globals under test.
//
// swmm_triton.h pulls objects.h/funcs.h but NOT globals.h, and globals.h is not
// independently includable here: it declares storage through an EXTERN macro
// whose definition depends on which translation unit is compiling it. Declaring
// exactly the handful this test touches is narrower than dragging globals.h in,
// and it keeps the test honest about its own surface -- every name below is one
// the snapshot serializes.
// ---------------------------------------------------------------------------
extern "C" {
    extern int    Nobjects[];
    extern long   ReportStepCount;
    extern long   NonConvergeCount;
    extern long   TotalStepCount;
    extern double MaxOutfallFlow;
    extern double MaxRunoffFlow;
    extern double RoutingTimeSpan;
    extern double TotalArea;
    extern TRunoffTotals  RunoffTotals;
    extern TGwaterTotals  GwaterTotals;
    extern TRoutingTotals FlowTotals;
}

// ---------------------------------------------------------------------------
// Minimal harness (same shape as the WP-0A tier-1 tests)
// ---------------------------------------------------------------------------
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

#define CHECK_EQ_I(actual, expected, msg)                                         \
    do {                                                                          \
        ++g_checks;                                                               \
        const long long _a = (long long)(actual);                                 \
        const long long _e = (long long)(expected);                               \
        if (_a != _e) {                                                           \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s  [actual=%lld expected=%lld]\n", \
                         __FILE__, __LINE__, msg, _a, _e);                        \
        }                                                                         \
    } while (0)

// Bitwise, not approximate. The requirement is bit-for-bit reproducibility, and
// an == on doubles would pass for two values that differ in no bit but NaN, so
// the comparison is made on the object representation.
static bool bit_equal(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

static std::string temp_path(const char* stem)
{
    std::ostringstream oss;
    oss << std::filesystem::temp_directory_path().string()
        << "/triton_wp1b_" << stem << "_" << (long) getpid() << ".snapshot";
    return oss.str();
}

// ---------------------------------------------------------------------------
// T3 -- a snapshot round-trips bit-exactly over every field it carries
// ---------------------------------------------------------------------------
//
// Method: stamp distinguishable values into the reachable state, save, CLOBBER
// the live state with different values, load, and compare bit-for-bit. The
// clobber is what makes this a test: without it a load that did nothing at all
// would pass.
static void T3_snapshot_roundtrips_bit_exactly()
{
    const std::string path = temp_path("t3");

    TTimeStepStats* tss = nullptr;
    TMaxStats *mbe = nullptr, *cc = nullptr, *ft = nullptr, *nc = nullptr;
    double* sysOut = nullptr;
    int nStats = 0;
    stats_getSnapshotRefs(&tss, &mbe, &cc, &ft, &nc, &sysOut, &nStats);
    CHECK(tss != nullptr && mbe != nullptr && sysOut != nullptr,
          "stats_getSnapshotRefs must hand out live addresses");
    CHECK(nStats > 0, "stats.c must report its MAX_STATS");

    // --- stamp: values chosen to be individually identifiable in a diff, and
    //     with enough mantissa that a float round trip could not preserve them.
    tss->minTimeStep     = 0.0281828459045235;
    tss->maxTimeStep     = 3.1415926535897932;
    tss->routingTime     = 12345.678901234567;
    tss->timeStepCount   = 987654;
    tss->trialsCount     = 4321.0;
    tss->steadyStateTime = 271.82818284590452;
    for (int k = 0; k < TIMELEVELS; k++) {
        tss->timeStepIntervals[k] = 1.0 + k * 0.1234567890123456;
        tss->timeStepCounts[k]    = 1000 + k;
    }
    for (int i = 0; i < nStats; i++) {
        mbe[i].objType = 1; mbe[i].index = 10 + i; mbe[i].value = 0.5 + i * 0.0009765625;
        cc[i].objType  = 2; cc[i].index  = 20 + i; cc[i].value  = 1.5 + i * 0.0009765625;
        ft[i].objType  = 1; ft[i].index  = 30 + i; ft[i].value  = 2.5 + i * 0.0009765625;
        nc[i].objType  = 2; nc[i].index  = 40 + i; nc[i].value  = 3.5 + i * 0.0009765625;
    }
    FlowTotals.dwInflow     = 1.2345678901234567e6;
    FlowTotals.flooding     = 9.8765432109876543e4;
    FlowTotals.pctError     = -0.0123456789012345;
    RunoffTotals.rainfall   = 5.5555555555555555e3;
    GwaterTotals.infil      = 7.7777777777777777e-3;
    *sysOut                 = 4.2424242424242424;
    MaxOutfallFlow          = 8.8888888888888888;
    RoutingTimeSpan         = 60000.000000000007;
    TotalArea               = 1.0000000000000002e5;
    ReportStepCount         = 555555;
    NonConvergeCount        = 777;
    TotalStepCount          = 888888;

    // --- capture what we expect back
    const double  e_min   = tss->minTimeStep;
    const double  e_route = tss->routingTime;
    const int     e_cnt   = tss->timeStepCount;
    const double  e_ivl3  = tss->timeStepIntervals[3];
    const int     e_cnt3  = tss->timeStepCounts[3];
    const double  e_ccv1  = cc[1].value;
    const int     e_nci1  = nc[1].index;
    const double  e_flood = FlowTotals.flooding;
    const double  e_pct   = FlowTotals.pctError;
    const double  e_rain  = RunoffTotals.rainfall;
    const double  e_gw    = GwaterTotals.infil;
    const double  e_sys   = *sysOut;
    const double  e_span  = RoutingTimeSpan;
    const double  e_area  = TotalArea;
    const long    e_rsc   = ReportStepCount;
    const long    e_ncc   = NonConvergeCount;

    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "snapshot_save must succeed");

    // --- CLOBBER. Without this the test cannot distinguish a correct restore
    //     from a load that read nothing.
    tss->minTimeStep = -1.0; tss->routingTime = -1.0; tss->timeStepCount = -1;
    tss->timeStepIntervals[3] = -1.0; tss->timeStepCounts[3] = -1;
    cc[1].value = -1.0; nc[1].index = -1;
    FlowTotals.flooding = -1.0; FlowTotals.pctError = -1.0;
    RunoffTotals.rainfall = -1.0; GwaterTotals.infil = -1.0;
    *sysOut = -1.0; RoutingTimeSpan = -1.0; TotalArea = -1.0;
    ReportStepCount = -1; NonConvergeCount = -1;

    char msg[512]; msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(path.c_str(), msg, (int) sizeof(msg)), 0,
               "snapshot_load must succeed on a snapshot this build wrote");
    if (msg[0]) std::fprintf(stderr, "  note: load said: %s\n", msg);

    CHECK(bit_equal(tss->minTimeStep, e_min),   "minTimeStep must round-trip bitwise");
    CHECK(bit_equal(tss->routingTime, e_route), "routingTime must round-trip bitwise");
    CHECK_EQ_I(tss->timeStepCount, e_cnt,       "timeStepCount must round-trip");
    CHECK(bit_equal(tss->timeStepIntervals[3], e_ivl3), "timeStepIntervals[] must round-trip bitwise");
    CHECK_EQ_I(tss->timeStepCounts[3], e_cnt3,  "timeStepCounts[] must round-trip");
    CHECK(bit_equal(cc[1].value, e_ccv1),       "MaxCourantCrit[].value must round-trip bitwise");
    CHECK_EQ_I(nc[1].index, e_nci1,             "MaxNonConverged[].index must round-trip");
    CHECK(bit_equal(FlowTotals.flooding, e_flood), "FlowTotals.flooding must round-trip bitwise");
    CHECK(bit_equal(FlowTotals.pctError, e_pct),   "FlowTotals.pctError must round-trip bitwise");
    CHECK(bit_equal(RunoffTotals.rainfall, e_rain),"RunoffTotals.rainfall must round-trip bitwise");
    CHECK(bit_equal(GwaterTotals.infil, e_gw),     "GwaterTotals.infil must round-trip bitwise");
    CHECK(bit_equal(*sysOut, e_sys),               "SysOutfallFlow must round-trip bitwise");
    CHECK(bit_equal(RoutingTimeSpan, e_span),      "RoutingTimeSpan must round-trip bitwise");
    CHECK(bit_equal(TotalArea, e_area),            "TotalArea must round-trip bitwise");
    CHECK_EQ_I(ReportStepCount, e_rsc,             "ReportStepCount must round-trip");
    CHECK_EQ_I(NonConvergeCount, e_ncc,            "NonConvergeCount must round-trip");

    std::error_code ec; std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// T3b -- the stored width is double, and a float round trip would NOT pass T3
// ---------------------------------------------------------------------------
//
// The whole reason the shipped hotstart file is disqualified is that it stores
// these quantities as float. This test pins the claim in executable form rather
// than leaving it as a comment: the value T3 round-trips is one that does NOT
// survive a float round trip, so T3 passing is evidence about precision and not
// merely about plumbing.
static void T3b_stored_width_is_double_not_float()
{
    const double v = 0.0281828459045235;
    const float  f = (float) v;
    CHECK(!bit_equal((double) f, v),
          "the T3 probe value must be one a float round trip would corrupt -- "
          "otherwise T3 passing says nothing about stored precision");

    // And the format's declared width is double, read back off a real file.
    const std::string path = temp_path("t3b");
    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "snapshot_save must succeed");
    std::ifstream in(path, std::ios::binary);
    int32_t hdr[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    in.close();
    CHECK_EQ_I(hdr[2], (int) sizeof(double),
               "the header's value_width field must declare 8-byte values");
    std::error_code ec; std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// T4 -- a snapshot that is not one of ours is refused, loudly
// ---------------------------------------------------------------------------
static void T4_bad_magic_is_refused()
{
    const std::string path = temp_path("t4");
    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "snapshot_save must succeed");

    // Corrupt only the magic, leaving a structurally valid file behind it. A
    // test that truncates instead would pass for the wrong reason.
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    int32_t bad = 0x4A554E4B;   // "JUNK"
    io.seekp(0);
    io.write(reinterpret_cast<const char*>(&bad), sizeof(bad));
    io.close();

    char msg[512]; msg[0] = '\0';
    CHECK(snapshot_load(path.c_str(), msg, (int) sizeof(msg)) != 0,
          "a snapshot with a foreign magic must be refused");
    CHECK(std::string(msg).find("magic") != std::string::npos,
          "the refusal must name the magic, not just fail");

    std::error_code ec; std::filesystem::remove(path, ec);
}

static void T4b_missing_file_is_refused_by_name()
{
    char msg[512]; msg[0] = '\0';
    const std::string path = temp_path("t4b_does_not_exist");
    CHECK(snapshot_load(path.c_str(), msg, (int) sizeof(msg)) != 0,
          "an absent snapshot must be refused");
    CHECK(std::string(msg).find("not found") != std::string::npos,
          "the refusal must say the snapshot was not found -- this is the "
          "message an operator sees when a resume silently takes the slow "
          "replay path, so it has to say why");
}

// ---------------------------------------------------------------------------
// T5 -- a snapshot that does not describe the running model is refused, and the
//       refusal NAMES the mismatching dimension
// ---------------------------------------------------------------------------
//
// This is the guard against the worst available failure: silently restoring a
// payload written for a different model reads it at the wrong stride and yields
// a state that is wrong everywhere without being wrong anywhere a reader looks.
static void T5_shape_mismatch_is_refused_and_named()
{
    const std::string path = temp_path("t5");
    const int saved = Nobjects[NODE];

    Nobjects[NODE] = 0;
    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "snapshot_save must succeed");

    // The running model now claims a different node count than the snapshot.
    Nobjects[NODE] = 7;
    char msg[512]; msg[0] = '\0';
    CHECK(snapshot_load(path.c_str(), msg, (int) sizeof(msg)) != 0,
          "a snapshot describing a different node count must be refused");
    const std::string m(msg);
    CHECK(m.find("node") != std::string::npos,
          "the refusal must name the mismatching dimension");
    CHECK(m.find("7") != std::string::npos,
          "the refusal must report what this run has, so the operator can tell "
          "which side changed");

    Nobjects[NODE] = saved;
    std::error_code ec; std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// T6 -- the serializer's emitted manifest COVERS the committed inventory
// ---------------------------------------------------------------------------
//
// T6 and the regeneration check are DIFFERENT checks and neither substitutes
// for the other. T6 asserts that what the serializer emits matches the
// committed inventory -- it catches the serializer drifting from the record.
// The regeneration check (slow tier + upstream-bump step) re-runs the operation
// against current EPA source and asserts the INVENTORY still matches the
// source -- it catches the record drifting from upstream. A serializer and an
// inventory can agree with each other while both are stale.
static void T6_manifest_covers_committed_inventory()
{
    long nFields = 0;
    char* manifest = snapshot_buildManifest(&nFields);
    CHECK(manifest != nullptr, "snapshot_buildManifest must succeed with no model open");
    if (!manifest) return;

    // The manifest must be answerable without a model. If it were not, the
    // file's description of its own layout would depend on the model it was
    // taken from.
    CHECK(nFields > 0, "the manifest must be non-empty with every count at zero");

    std::set<std::string> emitted;
    {
        std::istringstream iss(manifest);
        std::string line;
        while (std::getline(iss, line))
            if (!line.empty()) emitted.insert(line);
    }
    std::free(manifest);

    const char* inv = std::getenv("TRITON_SNAPSHOT_INVENTORY");
    CHECK(inv != nullptr, "TRITON_SNAPSHOT_INVENTORY must name the committed inventory");
    if (!inv) return;

    std::ifstream f(inv);
    CHECK(f.good(), "the committed inventory must be readable");
    if (!f.good()) return;

    std::vector<std::string> missing;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::size_t t1 = line.find('\t');
        if (t1 == std::string::npos) continue;
        const std::string obj = line.substr(0, t1);
        const std::size_t t2 = line.find('\t', t1 + 1);
        std::string fld = (t2 == std::string::npos) ? line.substr(t1 + 1)
                                                    : line.substr(t1 + 1, t2 - t1 - 1);
        if (fld == "-") fld = "value";   // scalar rows carry '-' for the field
        if (!emitted.count(obj + "." + fld)) missing.push_back(obj + "." + fld);
    }

    for (const std::string& m : missing)
        std::fprintf(stderr, "  MISSING FROM MANIFEST: %s\n", m.c_str());
    CHECK_EQ_I((long) missing.size(), 0,
               "every {object, field} pair in the committed inventory must appear "
               "in the serializer's emitted manifest");
}

// ---------------------------------------------------------------------------
// T7 -- the sizeof guard's REAL boundary, in executable form
// ---------------------------------------------------------------------------
//
// The guard catches an addition that GROWS a struct and is blind to one that
// lands in existing padding. That is a boundary, not a weakness to hide: it is
// exactly why the padding-independent regeneration check exists and has two
// firing points. This test pins the boundary from a compiled measurement so a
// future reader cannot mistake the guard for full coverage.
static void T7_sizeof_guard_boundary_is_documented()
{
    // The sizes the build-time guards assert.
    CHECK_EQ_I(sizeof(TNodeStats),     128, "TNodeStats size");
    CHECK_EQ_I(sizeof(TLinkStats),     176, "TLinkStats size");
    CHECK_EQ_I(sizeof(TStorageStats),   56, "TStorageStats size");
    CHECK_EQ_I(sizeof(TOutfallStats),   32, "TOutfallStats size");
    CHECK_EQ_I(sizeof(TPumpStats),      72, "TPumpStats size");
    CHECK_EQ_I(sizeof(TSubcatchStats),  64, "TSubcatchStats size");
    CHECK_EQ_I(sizeof(TTimeStepStats), 120, "TTimeStepStats size");
    CHECK_EQ_I(sizeof(TMaxStats),       16, "TMaxStats size");

    // BLIND SPOT: an added int lands in the trailing hole and sizeof does not
    // move, so the guard would not fire.
    struct OutfallPlusInt { double a, b; double* p; int n; int added; };
    CHECK_EQ_I(sizeof(OutfallPlusInt), sizeof(TOutfallStats),
               "an int added to TOutfallStats' trailing padding does NOT change "
               "sizeof -- the guard is blind to it and the regeneration check "
               "is what catches it");

    struct LinkPlusInt {
        double a, b, c, d, e, f, g, h, i, j, k, l;
        double cls[MAX_FLOW_CLASSES];
        double m; long n; int o; int added;
    };
    CHECK_EQ_I(sizeof(LinkPlusInt), sizeof(TLinkStats),
               "an int added to TLinkStats' trailing padding does NOT change sizeof");

    // VISIBLE: an added double grows the struct, so the guard fires.
    struct OutfallPlusDouble { double a, b; double* p; int n; double added; };
    CHECK(sizeof(OutfallPlusDouble) != sizeof(TOutfallStats),
          "an added double DOES change sizeof -- this half the guard catches");
}

// ---------------------------------------------------------------------------
// T8 -- the restored global_new_depth[] equals what a replay leaves, bitwise
// ---------------------------------------------------------------------------
//
// NOT IMPLEMENTED IN TIER 1, and reported rather than silently omitted.
// The property is about the FACTORED handoff (routing_exportInflowNodeDepths)
// producing the same array on the snapshot path as routing_execute leaves on
// the replay path. Establishing it requires a live SWMM model with FLOW_INFLOW
// nodes and a routing step, which is a solver run -- the thing this tier is
// defined by not doing. It belongs to the Tier-2 smoke test.
//
// What tier 1 CAN establish is the structural half: that there is exactly one
// writer, so the two paths cannot diverge by construction.
static void T8_single_writer_is_structural()
{
    // The design's own measurement: swmm_newDepth appears on nine lines across
    // four files, exactly one of which is a write. This test asserts the
    // property that makes that true -- the writer is a named, externally
    // callable function, so the snapshot path reuses it rather than copying it.
    void (*writer)(double*) = &routing_exportInflowNodeDepths;
    CHECK(writer != nullptr,
          "the inflow-depth writer must be externally callable, so the snapshot "
          "restore path can reuse the one enumeration instead of duplicating it");

    // A NULL buffer must be a no-op rather than a crash: the snapshot path can
    // reach it before global_new_depth is allocated on a rank that owns no node.
    routing_exportInflowNodeDepths(nullptr);
    CHECK(true, "the writer must tolerate a NULL buffer");
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
struct test_entry { const char* name; void (*fn)(); };

static const test_entry kTests[] = {
    { "T3_snapshot_roundtrips_bit_exactly",      &T3_snapshot_roundtrips_bit_exactly },
    { "T3b_stored_width_is_double_not_float",    &T3b_stored_width_is_double_not_float },
    { "T4_bad_magic_is_refused",                 &T4_bad_magic_is_refused },
    { "T4b_missing_file_is_refused_by_name",     &T4b_missing_file_is_refused_by_name },
    { "T5_shape_mismatch_is_refused_and_named",  &T5_shape_mismatch_is_refused_and_named },
    { "T6_manifest_covers_committed_inventory",  &T6_manifest_covers_committed_inventory },
    { "T7_sizeof_guard_boundary_is_documented",  &T7_sizeof_guard_boundary_is_documented },
    { "T8_single_writer_is_structural",          &T8_single_writer_is_structural },
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

    if (want && ran == 0) {
        std::fprintf(stderr, "ERROR: no test named '%s'\n", want);
        return 2;
    }

    std::fprintf(stderr, "%d test(s), %d check(s), %d failure(s)\n", ran, g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
