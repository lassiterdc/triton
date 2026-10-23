/** @file test_state_snapshot.cpp
 *  @brief Tier-1 tests for the coupled-resume state snapshot.
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

// ---------------------------------------------------------------------------
// C LINKAGE for output_end -- AND THE POSITION OF THIS BLOCK IS LOAD-BEARING.
//
// funcs.h opens its `extern "C"` block partway down the file (`extern "C" {`
// at :298, closing at :393), so every declaration outside that window -- which
// includes `output_end` at funcs.h:163 -- takes C++ LANGUAGE LINKAGE in a C++
// translation unit. output_end is defined at output.c:822, which CMake builds
// as C, so T14's call below would emit a MANGLED reference against an
// UNMANGLED definition and fail at LINK with an undefined reference. Measured
// on the unrepaired TU shape: `nm -u` reports `U _Z10output_endv`; with this
// block it reports `U output_end`.
//
// This declaration MUST precede `#include "swmm_triton.h"`, which is what
// pulls funcs.h. Placed AFTER it -- for instance in the globals block below --
// it is a second declaration giving a different language linkage to a name
// already declared, which [dcl.link]p6 makes ill-formed and which g++ 13.3
// rejects outright:
//     error: conflicting declaration of 'void output_end()' with 'C' linkage
//     note:  previous declaration with 'C++' linkage
// Placed BEFORE, the later unadorned funcs.h declaration does not disturb the
// linkage explicitly specified here -- the same rule read the other way.
//
// So the globals block below is NOT the model for this one, and the difference
// is exactly what makes output_end a defect: those names have no prior
// declaration to conflict with, because globals.h is never included here.
// output_end has one. Do not fold this block into that one.
// ---------------------------------------------------------------------------
extern "C" {
    void output_end(void);
}

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

    // Added with the Criterion-P routing fields. Same rule as
    // above: every name here is one the snapshot serializes.
    extern int        Nnodes[];
    extern int        Nlinks[];
    extern TNode*     Node;
    extern TLink*     Link;
    extern TConduit*  Conduit;
    extern TOutfall*  Outfall;
    extern TStorage*  Storage;
    extern TNodeStats*    NodeStats;
    extern TLinkStats*    LinkStats;
    extern TStorageStats* StorageStats;
    extern TOutfallStats* OutfallStats;
    extern double*    NodeInflow;
    extern double*    NodeOutflow;

    // Nperiods is the output-period count the snapshot now carries; it
    // is a globals.h `long`, NOT an int, which is the one type a reader would
    // guess wrong. Fout is the binary output file handle the continuity work
    // positions and truncates.
    extern long       Nperiods;
    extern TFile      Fout;

    // Added with the PAYLOAD-LENGTH nodes. Same rule as above: every name here
    // is one the snapshot serializes. These two are the per-pollutant totals,
    // and they are POINTERS indexed by pollutant rather than singletons -- the
    // traversal guards at snapshot.c:750 and :764 refuse when Nobjects[POLLUT]
    // is non-zero and either is NULL, so a pollutant-bearing fixture must
    // allocate them. Declared here and NOT in the block above because they are
    // massbal.c globals rather than stats.c ones; snapshot.c declares them the
    // same way at :97 and :100.
    extern TLoadingTotals* LoadingTotals;
    extern TRoutingTotals* QualTotals;
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
// T3c -- the Criterion-P ROUTING fields round-trip bit-exactly
// ---------------------------------------------------------------------------
//
// T3 proves the report accumulators survive a round trip. It cannot say
// anything about the routing fields chunk (10) added, because with no model
// open every routing count is zero and every routing loop body runs zero times
// -- so T3 would pass unchanged if chunk (10) had serialized nothing at all.
// That is the shape this test exists to close: a green that is green because it
// examined nothing.
//
// Method: stand up a MINIMAL synthetic model by setting the counts and
// allocating the arrays the traversal walks, stamp distinguishable values,
// save, CLOBBER, load, compare bitwise. The clobber is what makes it a test.
//
// WHAT THIS DOES NOT COVER, and it is not coverable from here: Xnode. dynwave.c
// allocates it in dynwave_init() and it is `static`, so a test that never runs
// the router has no Xnode and dynwave_getSnapshotCount() correctly returns 0 --
// the loop emits nothing and the two Xnode names reach the manifest only
// through MANIFEST mode. T6 still asserts both names are emitted; what is
// unproven here is their round trip, which needs a real dynamic-wave run. Said
// plainly rather than implied by a green.
//
// Each CTest node id runs this binary with ONE subtest name, so the globals
// this mutates are private to its own process.
static void T3c_routing_fields_roundtrip_bit_exactly()
{
    const std::string path = temp_path("t3c");

    // --- stand up the smallest model that exercises every routing loop.
    const int nNode = 3, nLink = 2, nCond = 2, nOut = 1, nStor = 1;
    Nobjects[NODE]  = nNode;
    Nobjects[LINK]  = nLink;
    Nnodes[OUTFALL] = nOut;
    Nnodes[STORAGE] = nStor;
    Nlinks[CONDUIT] = nCond;

    Node    = (TNode*)    std::calloc(nNode, sizeof(TNode));
    Link    = (TLink*)    std::calloc(nLink, sizeof(TLink));
    Conduit = (TConduit*) std::calloc(nCond, sizeof(TConduit));
    Outfall = (TOutfall*) std::calloc(nOut,  sizeof(TOutfall));
    Storage = (TStorage*) std::calloc(nStor, sizeof(TStorage));
    // The traversal refuses (c->error) rather than indexing a null array, so the
    // report-side arrays must exist too once their counts are non-zero. Their
    // CONTENTS are T3's subject, not this test's.
    NodeStats    = (TNodeStats*)    std::calloc(nNode, sizeof(TNodeStats));
    LinkStats    = (TLinkStats*)    std::calloc(nLink, sizeof(TLinkStats));
    StorageStats = (TStorageStats*) std::calloc(nStor, sizeof(TStorageStats));
    OutfallStats = (TOutfallStats*) std::calloc(nOut,  sizeof(TOutfallStats));
    NodeInflow   = (double*)        std::calloc(nNode, sizeof(double));
    NodeOutflow  = (double*)        std::calloc(nNode, sizeof(double));

    CHECK(Node && Link && Conduit && Outfall && Storage && NodeStats &&
          LinkStats && StorageStats && OutfallStats && NodeInflow && NodeOutflow,
          "the synthetic model must allocate");
    if (!Node || !Link || !Conduit || !Outfall || !Storage) return;

    // --- stamp. Mantissa-heavy so a float round trip could not preserve them.
    Node[1].inflow        = 1.2345678901234567e3;
    Node[1].outflow       = 2.3456789012345678e3;
    Node[1].losses        = 3.4567890123456789e-2;
    Node[1].newVolume     = 4.5678901234567890e4;
    Node[1].overflow      = 5.6789012345678901e1;
    Node[1].oldDepth      = 6.7890123456789012;
    Node[1].newDepth      = 7.8901234567890123;
    Node[1].newLatFlow    = 8.9012345678901234;
    Node[1].oldFlowInflow = 9.0123456789012345;
    Node[1].oldNetInflow  = 1.0234567890123456;
    Node[1].updated       = (char) 1;

    Link[1].newFlow       = 1.1345678901234567e2;
    Link[1].newDepth      = 2.2456789012345678;
    Link[1].newVolume     = 3.3567890123456789e3;
    Link[1].setting       = 0.4567890123456789;
    Link[1].targetSetting = 0.5678901234567890;
    Link[1].timeLastSet   = 4.5678901234567890e4;
    Link[1].froude        = 0.6789012345678901;
    Link[1].dqdh          = 7.8901234567890123e-3;
    Link[1].flowClass     = 3;
    Link[1].normalFlow    = (char) 1;
    Link[1].inletControl  = (char) 1;

    Conduit[1].a1              = 1.9345678901234567;
    Conduit[1].q1              = 2.8456789012345678e2;
    Conduit[1].q2              = 3.7567890123456789e2;
    Conduit[1].evapLossRate    = 4.6678901234567890e-4;
    Conduit[1].seepLossRate    = 5.5789012345678901e-4;
    Conduit[1].capacityLimited = (char) 1;
    Conduit[1].fullState       = (char) 2;

    Outfall[0].vRouted = 6.4890123456789012e5;
    Storage[0].hrt       = 7.3901234567890123e3;
    Storage[0].evapLoss  = 8.2012345678901234e-1;
    Storage[0].exfilLoss = 9.1123456789012345e-1;

    // --- capture
    const double e_ninf  = Node[1].inflow,      e_nout = Node[1].outflow;
    const double e_nlos  = Node[1].losses,      e_nvol = Node[1].newVolume;
    const double e_novf  = Node[1].overflow,    e_nod  = Node[1].oldDepth;
    const double e_nnd   = Node[1].newDepth,    e_nlat = Node[1].newLatFlow;
    const double e_nofi  = Node[1].oldFlowInflow, e_noni = Node[1].oldNetInflow;
    const char   e_nupd  = Node[1].updated;
    const double e_lflow = Link[1].newFlow,     e_ldep = Link[1].newDepth;
    const double e_lvol  = Link[1].newVolume,   e_lset = Link[1].setting;
    const double e_ltgt  = Link[1].targetSetting, e_ltls = Link[1].timeLastSet;
    const double e_lfr   = Link[1].froude,      e_ldq  = Link[1].dqdh;
    const int    e_lfc   = Link[1].flowClass;
    const char   e_lnf   = Link[1].normalFlow,  e_lic  = Link[1].inletControl;
    const double e_ca1   = Conduit[1].a1,       e_cq1  = Conduit[1].q1;
    const double e_cq2   = Conduit[1].q2,       e_cev  = Conduit[1].evapLossRate;
    const double e_cse   = Conduit[1].seepLossRate;
    const char   e_ccl   = Conduit[1].capacityLimited, e_cfs = Conduit[1].fullState;
    const double e_ovr   = Outfall[0].vRouted;
    const double e_shrt  = Storage[0].hrt,      e_sev  = Storage[0].evapLoss;
    const double e_sex   = Storage[0].exfilLoss;

    CHECK_EQ_I(snapshot_save(path.c_str()), 0,
               "snapshot_save must succeed over a synthetic routing model");

    // --- CLOBBER every captured slot.
    Node[1].inflow = Node[1].outflow = Node[1].losses = Node[1].newVolume =
        Node[1].overflow = Node[1].oldDepth = Node[1].newDepth =
        Node[1].newLatFlow = Node[1].oldFlowInflow = Node[1].oldNetInflow = -1.0;
    Node[1].updated = (char) 0;
    Link[1].newFlow = Link[1].newDepth = Link[1].newVolume = Link[1].setting =
        Link[1].targetSetting = Link[1].timeLastSet = Link[1].froude =
        Link[1].dqdh = -1.0;
    Link[1].flowClass = -1; Link[1].normalFlow = 0; Link[1].inletControl = 0;
    Conduit[1].a1 = Conduit[1].q1 = Conduit[1].q2 =
        Conduit[1].evapLossRate = Conduit[1].seepLossRate = -1.0;
    Conduit[1].capacityLimited = 0; Conduit[1].fullState = 0;
    Outfall[0].vRouted = -1.0;
    Storage[0].hrt = Storage[0].evapLoss = Storage[0].exfilLoss = -1.0;

    char msg[512]; msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(path.c_str(), msg, (int) sizeof(msg)), 0,
               "snapshot_load must succeed on a snapshot this build wrote");
    if (msg[0]) std::fprintf(stderr, "  note: load said: %s\n", msg);

    CHECK(bit_equal(Node[1].inflow, e_ninf),        "Node.inflow must round-trip bitwise");
    CHECK(bit_equal(Node[1].outflow, e_nout),       "Node.outflow must round-trip bitwise");
    CHECK(bit_equal(Node[1].losses, e_nlos),        "Node.losses must round-trip bitwise");
    CHECK(bit_equal(Node[1].newVolume, e_nvol),     "Node.newVolume must round-trip bitwise");
    CHECK(bit_equal(Node[1].overflow, e_novf),      "Node.overflow must round-trip bitwise");
    CHECK(bit_equal(Node[1].oldDepth, e_nod),       "Node.oldDepth must round-trip bitwise");
    CHECK(bit_equal(Node[1].newDepth, e_nnd),       "Node.newDepth must round-trip bitwise");
    CHECK(bit_equal(Node[1].newLatFlow, e_nlat),    "Node.newLatFlow must round-trip bitwise");
    CHECK(bit_equal(Node[1].oldFlowInflow, e_nofi), "Node.oldFlowInflow must round-trip bitwise");
    CHECK(bit_equal(Node[1].oldNetInflow, e_noni),  "Node.oldNetInflow must round-trip bitwise");
    CHECK_EQ_I(Node[1].updated, e_nupd,             "Node.updated must round-trip");

    CHECK(bit_equal(Link[1].newFlow, e_lflow),      "Link.newFlow must round-trip bitwise");
    CHECK(bit_equal(Link[1].newDepth, e_ldep),      "Link.newDepth must round-trip bitwise");
    CHECK(bit_equal(Link[1].newVolume, e_lvol),     "Link.newVolume must round-trip bitwise");
    CHECK(bit_equal(Link[1].setting, e_lset),       "Link.setting must round-trip bitwise");
    CHECK(bit_equal(Link[1].targetSetting, e_ltgt), "Link.targetSetting must round-trip bitwise");
    CHECK(bit_equal(Link[1].timeLastSet, e_ltls),   "Link.timeLastSet must round-trip bitwise");
    CHECK(bit_equal(Link[1].froude, e_lfr),         "Link.froude must round-trip bitwise");
    CHECK(bit_equal(Link[1].dqdh, e_ldq),           "Link.dqdh must round-trip bitwise");
    CHECK_EQ_I(Link[1].flowClass, e_lfc,            "Link.flowClass must round-trip");
    CHECK_EQ_I(Link[1].normalFlow, e_lnf,           "Link.normalFlow must round-trip");
    CHECK_EQ_I(Link[1].inletControl, e_lic,         "Link.inletControl must round-trip");

    CHECK(bit_equal(Conduit[1].a1, e_ca1),          "Conduit.a1 must round-trip bitwise");
    CHECK(bit_equal(Conduit[1].q1, e_cq1),          "Conduit.q1 must round-trip bitwise");
    CHECK(bit_equal(Conduit[1].q2, e_cq2),          "Conduit.q2 must round-trip bitwise");
    CHECK(bit_equal(Conduit[1].evapLossRate, e_cev),"Conduit.evapLossRate must round-trip bitwise");
    CHECK(bit_equal(Conduit[1].seepLossRate, e_cse),"Conduit.seepLossRate must round-trip bitwise");
    CHECK_EQ_I(Conduit[1].capacityLimited, e_ccl,   "Conduit.capacityLimited must round-trip");
    CHECK_EQ_I(Conduit[1].fullState, e_cfs,         "Conduit.fullState must round-trip");

    CHECK(bit_equal(Outfall[0].vRouted, e_ovr),     "Outfall.vRouted must round-trip bitwise");
    CHECK(bit_equal(Storage[0].hrt, e_shrt),        "Storage.hrt must round-trip bitwise");
    CHECK(bit_equal(Storage[0].evapLoss, e_sev),    "Storage.evapLoss must round-trip bitwise");
    CHECK(bit_equal(Storage[0].exfilLoss, e_sex),   "Storage.exfilLoss must round-trip bitwise");

    // Node.surDepth is .inp CONFIGURATION and is deliberately NOT serialized.
    // Asserting its ABSENCE here is what keeps a future "while I am in this
    // loop" re-admission from passing silently: a reader who restored it would
    // see this line fail and read the reason above it.
    Node[1].surDepth = 12345.0;
    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "second save must succeed");
    Node[1].surDepth = -1.0;
    msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(path.c_str(), msg, (int) sizeof(msg)), 0,
               "second load must succeed");
    CHECK(Node[1].surDepth == -1.0,
          "Node.surDepth must NOT be restored -- it is .inp configuration, and "
          "restoring it lets a stale snapshot override the running model");

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

    // THE INVENTORY HAS THREE SECTIONS AND THEY DEMAND DIFFERENT THINGS.
    //
    //   Criterion R  -- every row is serialized. Row shape is
    //                   object<TAB>field<TAB>declared<TAB>read_by_report_path.
    //   Criterion P  -- only the ADMITTED rows are serialized. Row shape is
    //                   object<TAB>field<TAB>liveness<TAB>disposition<TAB>reason,
    //                   and the EXCLUDED and KILLED rows MUST NOT appear, which
    //                   is the half a coverage-only check cannot state.
    //   TFILE-TRIAGE         -- stream HANDLES, not fields. No row is serialized and no
    //                   row names a {object, field} pair at all.
    //
    // A section-blind walk demanded all three. Measured on the inventory as
    // chunk (9) committed it: 381 rows demanded, of which 209 are P-section and
    // TFILE-TRIAGE rows the serializer does not and must not emit. This test was RED
    // from the moment the P section landed; the assertion below is what chunk
    // (13) owes it, not a tightening of a passing check.
    //
    // The EXCLUDED/KILLED direction is not symmetry for its own sake. Under
    // Criterion P over-capture is ordinarily a maintenance cost, EXCEPT for
    // .inp configuration, where a stale snapshot silently overrides the model
    // the operator is running. That exception is exactly what an over-capture
    // check catches, and the corrected Node.surDepth row is the instance it
    // would have caught.
    enum Section { SEC_R, SEC_P, SEC_DR6 };
    Section sec = SEC_R;

    // THE TWO CRITERIA OVERLAP ON NAMES, AND THE OVER-CAPTURE RULE MUST SAY SO.
    // TimeStepStats' eight fields are ADMITTED under Criterion R and EXCLUDED
    // under Criterion P, with the P-side reason reading "report accumulator;
    // admitted by Criterion R above and restored by the same snapshot". They
    // are serialized, correctly, and a naive "no refused row may be emitted"
    // rule reports all eight as over-capture. So `required` is accumulated
    // first, across BOTH sections, and a refused key is a violation only if
    // nothing else required it. Measured: without this scoping the check is red
    // on 8 names that are right.
    std::set<std::string> required;  // R rows + ADMITTED P rows
    std::set<std::string> refused;   // EXCLUDED / KILLED P rows
    long nR = 0, nAdmitted = 0, nRefused = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') {
            // MONOTONE. The string "CRITERION P" recurs in the trailing
            // "# CRITERION P ROWS: N" footer, which sits AFTER the TFILE-TRIAGE table;
            // a non-monotone machine would step back into SEC_P there. No data
            // row follows it today, so the bug would be silent until one did.
            if (sec == SEC_R && line.find("CRITERION P") != std::string::npos) sec = SEC_P;
            else if (sec == SEC_P && line.find("TFILE-TRIAGE --") != std::string::npos) sec = SEC_DR6;
            continue;
        }
        if (line.empty()) continue;
        if (sec == SEC_DR6) continue;      // handles, not fields

        std::vector<std::string> col;
        std::size_t start = 0, tab;
        while ((tab = line.find('\t', start)) != std::string::npos) {
            col.push_back(line.substr(start, tab - start));
            start = tab + 1;
        }
        col.push_back(line.substr(start));
        if (col.size() < 2) continue;

        std::string obj = col[0];
        std::string fld = col[1];

        // A SCALAR ROW CARRIES A '-' SENTINEL IN THE COLUMN ITS SECTION DOES
        // NOT USE, AND THE TWO SECTIONS SPELL IT IN OPPOSITE ORDERS.
        //
        //   Criterion R   name<TAB>-      -- the sentinel is the FIELD
        //   Criterion P   -<TAB>name      -- the sentinel is the OBJECT
        //
        // Both normalize onto "Name.value", which is what the serializer emits
        // for a singleton (snap_name(c, "NewRoutingTime", "value")).
        //
        // THE NORMALIZATION USED TO BE SECTION-BLIND -- a bare
        // `if (fld == "-") fld = "value";` -- and that made the OVER-CAPTURE
        // half of this test VACUOUS ON THE ENTIRE SCALAR AXIS. A P scalar row
        // left the blind form as the key `-.RouteModel`, while a serializer
        // that wrongly captured that config value would emit `RouteModel.value`.
        // The two can never be equal, so `overcaptured` stayed empty no matter
        // what the serializer did. The defect was latent for as long as every P
        // scalar row was UNTRIAGED, because an untriaged row lands in `refused`
        // and a refused key that matches nothing produces no finding -- so the
        // check reported green while measuring nothing, which is the exact
        // failure mode the denominator assertions at the end of this function
        // exist to make visible.
        //
        // Over-capture is the harmful direction here and that is why this is
        // worth a branch: under Criterion P over-capturing routing state is a
        // maintenance cost, but over-capturing .inp CONFIGURATION lets a stale
        // snapshot silently override the model the operator is running. Every
        // config scalar in the P section is spelled in the order this branch
        // repairs.
        if (sec == SEC_R) {
            if (fld == "-") fld = "value";
        } else if (sec == SEC_P) {
            if (obj == "-") { obj = fld; fld = "value"; }
        }
        const std::string key = obj + "." + fld;

        if (sec == SEC_R) {
            nR++;
            required.insert(key);
            continue;
        }

        // Criterion P: the disposition column decides which way the row points.
        const std::string disp = (col.size() >= 4) ? col[3] : std::string("-");
        if (disp == "ADMITTED") { nAdmitted++; required.insert(key); }
        else                    { nRefused++;  refused.insert(key); }
    }

    std::vector<std::string> missing;      // required, but absent from manifest
    std::vector<std::string> overcaptured; // refused by both criteria, yet present
    for (const std::string& k : required)
        if (!emitted.count(k)) missing.push_back(k);
    for (const std::string& k : refused)
        if (emitted.count(k) && !required.count(k)) overcaptured.push_back(k);

    for (const std::string& m : missing)
        std::fprintf(stderr, "  MISSING FROM MANIFEST: %s\n", m.c_str());
    for (const std::string& m : overcaptured)
        std::fprintf(stderr, "  OVER-CAPTURED (excluded/killed, yet serialized): %s\n",
                     m.c_str());

    CHECK_EQ_I((long) missing.size(), 0,
               "every Criterion-R row and every ADMITTED Criterion-P row must "
               "appear in the serializer's emitted manifest");
    CHECK_EQ_I((long) overcaptured.size(), 0,
               "no EXCLUDED or KILLED Criterion-P row may appear in the emitted "
               "manifest -- over-capture of .inp configuration lets a stale "
               "snapshot override the running model");

    // A denominator, so a vacuous pass is legible in the artifact itself. A
    // parser that silently matched nothing would satisfy both checks above.
    std::fprintf(stderr, "  T6 examined: %ld Criterion-R rows, %ld ADMITTED, "
                         "%ld EXCLUDED/KILLED\n", nR, nAdmitted, nRefused);
    CHECK(nR > 0,        "T6 must find Criterion-R rows -- zero means the parse failed");
    CHECK(nAdmitted > 0, "T6 must find ADMITTED Criterion-P rows -- zero means the parse failed");
    CHECK(nRefused > 0,  "T6 must find refused Criterion-P rows -- zero means the parse failed");
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
// Full-window SWMM output continuity.
//
// WHAT THIS TIER CAN AND CANNOT REACH, stated rather than left implicit.
// output.c's OutputStartPos and BytesPerPeriod are file-scope statics that
// only output_open sets, and output_open needs an open .inp -- so at this tier
// both are ZERO. That is not a defect in the tests below; it is their
// operating point, and each one says which property it therefore reaches:
//
//   REACHED HERE  the graduated return code (T13); Nperiods' bit-exact round
//                 trip (T3d); the refusal when the restored count exceeds the
//                 periods on disk (T12); and the FINAL-LENGTH property, whose
//                 arithmetic degenerates at a zero stride to "the trailer is
//                 the whole file" -- which still exercises the truncation
//                 primitive end to end against a real file (T14).
//   NOT REACHED   the non-truncating open and the prologue comparison, which
//                 have no callable entry point (output_openOutFile has
//                 internal linkage); they are covered structurally by the
//                 SOURCE-OUT-CONTINUITY node and behaviourally in Tier 2.
//                 T14's second arm -- the vendored outfile validator returning
//                 0 rather than 435 on a shortened run -- is Tier 2 for the
//                 same reason: it needs a real .out.
// ---------------------------------------------------------------------------

// Opens a scratch file of `bytes` length and installs it as Fout. The caller
// restores Fout and closes the handle itself.
static std::string install_scratch_fout(const char* stem, long bytes)
{
    const std::string path = temp_path(stem);
    FILE* f = std::fopen(path.c_str(), "w+b");
    for (long i = 0; i < bytes; ++i) std::fputc(0, f);
    std::fflush(f);
    Fout.file = f;
    return path;
}

static long file_size_of(const std::string& path)
{
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    return ec ? -1L : (long) n;
}

static void T3d_nperiods_roundtrips_bit_exactly()
{
    // The output-period count is what a resumed run's .out offsets, its
    // trailer and report.c's four detail-table loops are all derived from, so
    // a count that does not survive the round trip silently truncates three
    // artifact families at once.
    const std::string path = temp_path("t3d");
    const long saved = Nperiods;

    Nperiods = 1234567L;
    CHECK_EQ_I(snapshot_save(path.c_str()), 0, "snapshot_save must succeed");

    Nperiods = 0L;                       // what swmm_start leaves behind
    char msg[512]; msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(path.c_str(), msg, (int) sizeof(msg)), SNAPSHOT_OK,
               "snapshot_load must succeed");
    CHECK_EQ_I(Nperiods, 1234567L,
               "Nperiods must round-trip: without it the resumed run restarts "
               "the period count at zero and the .out, the .rpt detail tables "
               "and node{N}.out all cover only the post-resume tail");

    Nperiods = saved;
    std::error_code ec; std::filesystem::remove(path, ec);
}

static void T13_snapshot_load_return_is_graduated()
{
    char msg[512];

    // ABSENT -- the one legitimate refusal, and the ONLY one the caller may
    // fall back on.
    msg[0] = '\0';
    const std::string gone = temp_path("t13_does_not_exist");
    CHECK_EQ_I(snapshot_load(gone.c_str(), msg, (int) sizeof(msg)), SNAPSHOT_ABSENT,
               "a missing snapshot must return SNAPSHOT_ABSENT, because that is "
               "the only value the caller is allowed to fall back on");

    // UNREADABLE -- present and unusable. Falling back here would produce
    // correct numbers slowly while hiding a build or corruption defect.
    const std::string bad = temp_path("t13_bad_magic");
    CHECK_EQ_I(snapshot_save(bad.c_str()), 0, "snapshot_save must succeed");
    {
        std::fstream io(bad, std::ios::binary | std::ios::in | std::ios::out);
        int32_t junk = 0x4A554E4B;
        io.seekp(0);
        io.write(reinterpret_cast<const char*>(&junk), sizeof(junk));
    }
    msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(bad.c_str(), msg, (int) sizeof(msg)), SNAPSHOT_UNREADABLE,
               "a corrupt snapshot must return SNAPSHOT_UNREADABLE, not ABSENT");

    // A WRONG VERSION is the same class, and it is the one a routine pin bump
    // produces, so it is asserted separately rather than assumed to follow.
    const std::string oldv = temp_path("t13_old_version");
    CHECK_EQ_I(snapshot_save(oldv.c_str()), 0, "snapshot_save must succeed");
    {
        std::fstream io(oldv, std::ios::binary | std::ios::in | std::ios::out);
        int32_t v = 1;
        io.seekp(sizeof(int32_t));
        io.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    msg[0] = '\0';
    CHECK_EQ_I(snapshot_load(oldv.c_str(), msg, (int) sizeof(msg)), SNAPSHOT_UNREADABLE,
               "a pre-item-4 snapshot must return SNAPSHOT_UNREADABLE");

    // THE THREE VALUES MUST BE DISTINCT, or the graduation is decorative.
    CHECK(SNAPSHOT_OK != SNAPSHOT_ABSENT && SNAPSHOT_ABSENT != SNAPSHOT_UNREADABLE
          && SNAPSHOT_OK != SNAPSHOT_UNREADABLE,
          "the three outcomes must be distinguishable by VALUE -- a message a "
          "human reads is not a value a program branches on");

    std::error_code ec;
    std::filesystem::remove(bad, ec);
    std::filesystem::remove(oldv, ec);
}

static void T12_resume_refuses_more_periods_than_the_file_holds()
{
    // REQUIREMENT 5, and the fixture is the one a builder would not invent
    // from "flush before the snapshot" alone: a file SHORTER than the restored
    // count claims, which is what a SIGKILL between the last flush and the
    // checkpoint leaves behind. Without the refusal the resumed run seeks past
    // end-of-file and the append creates a zero-filled hole that parses as
    // valid data -- silent wrong numbers on exactly the kill path the snapshot
    // exists to serve.
    TFile saved_fout = Fout;
    const long saved_np = Nperiods;

    const std::string path = install_scratch_fout("t12_short", 64);
    char msg[512]; msg[0] = '\0';

    Nperiods = 5L;                       // a claim the file cannot back
    CHECK(output_positionForResume(msg, (int) sizeof(msg)) != 0,
          "a restored period count the .out cannot back must FAIL LOUD rather "
          "than seek past EOF");
    CHECK(std::string(msg).find("period") != std::string::npos,
          "the refusal must name the period counts, not just fail");
    CHECK_EQ_I(file_size_of(path), 64,
               "a REFUSED positioning must not have truncated anything");

    // The accepting branch, which also proves the truncation primitive works
    // against a real file rather than merely compiling.
    Nperiods = 0L;
    msg[0] = '\0';
    CHECK_EQ_I(output_positionForResume(msg, (int) sizeof(msg)), 0,
               "a count the file can back must be accepted");
    CHECK_EQ_I(file_size_of(path), 0,
               "the stale tail beyond the resumed position must be dropped");

    std::fclose(Fout.file);
    Fout = saved_fout;
    Nperiods = saved_np;
    std::error_code ec; std::filesystem::remove(path, ec);
}

static void T14_output_end_leaves_the_trailer_last()
{
    // REQUIREMENT 8. An external .out reader does not locate the trailer from
    // a stored offset -- it seeks -6*RECORDSIZE from SEEK_END and compares the
    // trailing magic against the leading one, so ANY byte surviving past the
    // trailer makes every reader refuse the file with error 435. The reachable
    // case is a SHORTENED run writing its trailer above a longer previous
    // exec's tail, which is exactly the fixture below.
    TFile saved_fout = Fout;
    const long saved_np = Nperiods;

    const std::string path = install_scratch_fout("t14_long_tail", 4096);
    Nperiods = 0L;                       // zero stride at this tier
    std::fseek(Fout.file, 0, SEEK_SET);

    output_end();

    const long want = 6L * (long) sizeof(int);
    CHECK_EQ_I(file_size_of(path), want,
               "output_end must leave the six-INT4 trailer as the LAST bytes "
               "of the file -- a surviving tail makes the trailing magic "
               "disagree with the leading one and every external reader "
               "refuses the file outright");

    std::fclose(Fout.file);
    Fout = saved_fout;
    Nperiods = saved_np;
    std::error_code ec; std::filesystem::remove(path, ec);
}

// ===========================================================================
// PAYLOAD LENGTH -- the one property in this tier that can detect a SYMMETRIC
// write/read defect.
//
// WHY THE ROUND-TRIP NODES ABOVE CANNOT. The repaired guard sat above a
// traversal SHARED across manifest, write and read modes, so a build that wrote
// the payload short read it short too. A round-trip oracle writes short, reads
// short, and reports bit-exactness. Measured rather than argued: the 13
// pre-existing nodes return `100% tests passed, 0 tests failed out of 13` on
// the FIXED tree (job 21271626) and IDENTICALLY on the PRE-FIX control tree
// (job 21271675), against a byte-identical copy of this file -- so T3 and T3c
// PASS on a tree where every campaign snapshot is short by 2,287 doubles.
//
// SO EVERY ASSERTION BELOW COMPARES AGAINST AN EXPECTATION DERIVED FROM THE
// TRAVERSAL'S OWN STRUCTURE, never against the artifact's own round trip.
// Two independent expectations are used, and neither carries a hand-maintained
// constant about the format:
//
//   T15  the payload length is a function of the TWELVE-DIMENSION SHAPE ALONE.
//        snapshot.c states this as the ground for the fixed-width wRouted
//        placeholder: "NONE of the twelve records routeTo, so a length that
//        varied with it would produce a file passing all twelve SHAPE_CHECKs
//        and the manifest strcmp and then MIS-STRIDING every later field".
//        Two saves at IDENTICAL shape differing ONLY in routeTo must therefore
//        produce IDENTICAL file lengths.
//
//   T16  the length's SLOPE along the pollutant axis is the one the traversal's
//        own MANIFEST predicts. Per unit of Nobjects[POLLUT] the traversal emits
//        one OutfallStats.totalLoad and one Outfall.wRouted per outfall, plus
//        one full LoadingTotals row and one full QualTotals row -- and the row
//        widths are COUNTED FROM THE MANIFEST at run time rather than written
//        here, so a field added to either struct cannot silently invalidate
//        this node.
//
// WHY A FILE-SIZE DIFFERENCE NEEDS NO HEADER ARITHMETIC, which is what keeps
// both nodes free of a hand constant. A snapshot is {4 ints} {12 ints}
// {manifest} {payload}. The manifest is built by SNAP_MANIFEST, under which
// SNAP_N returns 1 and every array guard is mode-gated off -- so the manifest
// bytes are IDENTICAL FOR ANY SHAPE, and the shape block is a fixed int count.
// The non-payload prefix is therefore the same length in any two files this
// build writes, and a DIFFERENCE of file sizes is exactly a difference of
// payload bytes. T16 does not assume that: it reads each file's own recorded
// manifest length and asserts the two agree before it uses the difference.
// ===========================================================================

// The manifest length this build records in header word 3, read from the file
// rather than recomputed. -1 when the file cannot be read.
static long recorded_manifest_len(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return -1L;
    int32_t hdr[4];
    in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!in) return -1L;
    return (long) hdr[3];
}

// Number of manifest lines whose object column is `obj`. The manifest is
// "{object}.{field}\n" per line (snap_name), and it is generated by the SAME
// traversal that writes the payload -- which is what makes a count taken from
// it a description of the format rather than a second claim about it.
static long manifest_rows_for(const char* obj)
{
    long  nFields = 0;
    char* m = snapshot_buildManifest(&nFields);
    if (!m) return -1L;

    const std::string prefix = std::string(obj) + ".";
    long n = 0;
    std::istringstream iss(m);
    std::string line;
    while (std::getline(iss, line))
        if (line.rfind(prefix, 0) == 0) ++n;

    std::free(m);
    return n;
}

// ---------------------------------------------------------------------------
// The minimal POLLUT-BEARING model, which is the only shape on which the
// defect is reachable at all: the retired guard was
// `Nobjects[POLLUT] > 0 && !s->wRouted`, so a model with no pollutants cannot
// enter it and a node built on one would be green on both trees.
//
// Deliberately the SMALLEST such model. Every count not needed to reach the
// outfall block is left at zero, so each allocation below exists because a
// named guard in snapshot.c demands it and for no other reason:
//   OutfallStats + its per-pollutant totalLoad   snapshot.c:598 and :609
//   Outfall                                      snapshot.c:940
//   LoadingTotals / QualTotals                   snapshot.c:750 and :764
// Nobjects[NODE], [LINK], [SUBCATCH], Nnodes[STORAGE] and Nlinks[PUMP],
// [CONDUIT] stay 0, so their guards and loops are inert and their arrays are
// never formed.
//
// `routeTo` < 0 is the ROUTE-LESS outfall -- a legitimate, routine state that
// node.c:163-168 leaves with wRouted == NULL, and the state on which the
// retired guard refused every snapshot the campaign wrote.
// ---------------------------------------------------------------------------
static const int kPayloadOutfalls = 1;

static void build_pollutant_model(int nPollut, int routeTo)
{
    Nobjects[NODE]     = 0;
    Nobjects[LINK]     = 0;
    Nobjects[SUBCATCH] = 0;
    Nobjects[POLLUT]   = nPollut;
    Nnodes[OUTFALL]    = kPayloadOutfalls;
    Nnodes[STORAGE]    = 0;
    Nlinks[CONDUIT]    = 0;
    Nlinks[PUMP]       = 0;

    OutfallStats = (TOutfallStats*) std::calloc(kPayloadOutfalls, sizeof(TOutfallStats));
    Outfall      = (TOutfall*)      std::calloc(kPayloadOutfalls, sizeof(TOutfall));
    LoadingTotals = (TLoadingTotals*) std::calloc((size_t) nPollut, sizeof(TLoadingTotals));
    QualTotals    = (TRoutingTotals*) std::calloc((size_t) nPollut, sizeof(TRoutingTotals));

    if (OutfallStats)
        for (int i = 0; i < kPayloadOutfalls; i++)
            OutfallStats[i].totalLoad = (double*) std::calloc((size_t) nPollut, sizeof(double));

    if (Outfall)
        for (int i = 0; i < kPayloadOutfalls; i++)
        {
            Outfall[i].routeTo = routeTo;
            // THE ALLOCATION IS KEYED ON routeTo, which is the whole subject:
            // node.c allocates wRouted only under `routeTo >= 0`.
            Outfall[i].wRouted = (routeTo >= 0)
                ? (double*) std::calloc((size_t) nPollut, sizeof(double))
                : nullptr;
        }
}

// Frees the model and returns every count to zero, so one subtest cannot leave
// a shape standing for the next. Each CTest node id runs this binary with ONE
// subtest name, but a node that calls the builder twice relies on this.
static void free_pollutant_model()
{
    if (OutfallStats)
    {
        for (int i = 0; i < kPayloadOutfalls; i++) std::free(OutfallStats[i].totalLoad);
        std::free(OutfallStats);
        OutfallStats = nullptr;
    }
    if (Outfall)
    {
        for (int i = 0; i < kPayloadOutfalls; i++) std::free(Outfall[i].wRouted);
        std::free(Outfall);
        Outfall = nullptr;
    }
    std::free(LoadingTotals); LoadingTotals = nullptr;
    std::free(QualTotals);    QualTotals    = nullptr;

    Nobjects[POLLUT] = 0;
    Nnodes[OUTFALL]  = 0;
}

// Saves the model currently standing and returns the file size, or -1 when the
// save refused. `rc_out` receives snapshot_save's own return so a caller can
// assert on the refusal separately from the length.
static long save_and_size(const char* stem, int* rc_out, std::string* path_out)
{
    const std::string path = temp_path(stem);
    const int rc = snapshot_save(path.c_str());
    if (rc_out)   *rc_out   = rc;
    if (path_out) *path_out = path;
    return file_size_of(path);
}

static void T15_payload_length_does_not_vary_with_routeto()
{
    const int nPollut = 2;

    // --- arm A: the ROUTE-LESS outfall. The campaign's own state.
    build_pollutant_model(nPollut, -1);
    int  rcA = -1; std::string pathA;
    const long sizeA = save_and_size("t15_routeless", &rcA, &pathA);
    const long mlenA = recorded_manifest_len(pathA);
    free_pollutant_model();

    // --- arm B: the ROUTED outfall, at the IDENTICAL twelve-dimension shape.
    build_pollutant_model(nPollut, 0);
    int  rcB = -1; std::string pathB;
    const long sizeB = save_and_size("t15_routed", &rcB, &pathB);
    const long mlenB = recorded_manifest_len(pathB);
    free_pollutant_model();

    // A refused save is the FIRST failing assertion on a pre-fix build: the
    // retired guard latched c->error on the route-less arm, so snapshot_save
    // returned non-zero there while the routed arm succeeded.
    CHECK_EQ_I(rcB, 0,
               "snapshot_save must succeed over a ROUTED outfall carrying "
               "pollutants -- if this fails the fixture is wrong, not the guard");
    CHECK_EQ_I(rcA, 0,
               "snapshot_save must succeed over a ROUTE-LESS outfall carrying "
               "pollutants: routeTo < 0 with wRouted == NULL is the routine "
               "state node.c leaves, and refusing it refused every snapshot "
               "the campaign wrote");

    CHECK(sizeA > 0 && sizeB > 0,
          "both arms must leave a readable file to measure");

    // The manifest is generated under SNAP_MANIFEST, where SNAP_N returns 1 and
    // every array guard is mode-gated off, so it cannot vary with the shape.
    // Asserted rather than assumed, because the size DIFFERENCE below is a
    // difference of PAYLOAD bytes only while it holds.
    CHECK_EQ_I(mlenA, mlenB,
               "the embedded manifest must be identical across two saves at the "
               "same shape -- otherwise a file-size difference is not a payload "
               "difference and this node's arithmetic does not hold");

    if (sizeA > 0 && sizeB > 0 && sizeA != sizeB)
    {
        const long shortBy = (sizeB - sizeA) / (long) sizeof(double);
        std::fprintf(stderr,
            "  note: the ROUTE-LESS payload is short by %ld double(s) "
            "(%ld bytes) against the ROUTED payload at the same shape\n",
            shortBy, sizeB - sizeA);
    }

    CHECK_EQ_I(sizeA, sizeB,
               "THE PAYLOAD LENGTH MUST BE A FUNCTION OF THE TWELVE-DIMENSION "
               "SHAPE ALONE. None of the twelve records routeTo, so a length "
               "that varies with it yields a file passing all twelve "
               "SHAPE_CHECKs and the manifest comparison and then MIS-STRIDING "
               "every later field -- a silent wrong-state restore");

    std::error_code ec;
    std::filesystem::remove(pathA, ec);
    std::filesystem::remove(pathB, ec);
}

static void T16_payload_length_tracks_the_pollutant_axis()
{
    // The SLOPE along the pollutant axis, against the slope the traversal's own
    // manifest predicts. This is the assertion a round trip cannot make: it
    // never reads either file back, so a writer and a reader that are short by
    // the same amount cannot satisfy it.
    const int P = 2;

    build_pollutant_model(P, -1);
    int  rcLo = -1; std::string pathLo;
    const long sizeLo = save_and_size("t16_p2", &rcLo, &pathLo);
    const long mlenLo = recorded_manifest_len(pathLo);
    free_pollutant_model();

    build_pollutant_model(P + 1, -1);
    int  rcHi = -1; std::string pathHi;
    const long sizeHi = save_and_size("t16_p3", &rcHi, &pathHi);
    const long mlenHi = recorded_manifest_len(pathHi);
    free_pollutant_model();

    CHECK_EQ_I(rcLo, 0, "the lower-pollutant save must succeed");
    CHECK_EQ_I(rcHi, 0, "the higher-pollutant save must succeed");
    CHECK(sizeLo > 0 && sizeHi > 0, "both arms must leave a readable file");
    CHECK_EQ_I(mlenLo, mlenHi,
               "the manifest must not vary with Nobjects[POLLUT] -- SNAP_N "
               "returns 1 under SNAP_MANIFEST, so a difference here means the "
               "size delta below is not a pure payload delta");

    // --- the EXPECTED slope, every term read from the traversal or set by this
    //     test. Per unit of Nobjects[POLLUT] the traversal emits:
    //       OutfallStats.totalLoad   one per outfall   (snapshot.c:605-610)
    //       Outfall.wRouted          one per outfall   (snapshot.c:962-971)
    //       LoadingTotals            one full row      (snapshot.c:751-767)
    //       QualTotals               one full row      (snapshot.c:765-781)
    //     The two ROW WIDTHS are COUNTED FROM THE MANIFEST rather than written
    //     here, so adding a field to either struct cannot invalidate this node
    //     while leaving it green.
    const long ltRow = manifest_rows_for("LoadingTotals");
    const long qtRow = manifest_rows_for("QualTotals");
    CHECK(ltRow > 0, "the manifest must name the LoadingTotals fields");
    CHECK(qtRow > 0, "the manifest must name the QualTotals fields");

    const long expectedDelta = 2L * (long) kPayloadOutfalls + ltRow + qtRow;
    const long observedDelta = (sizeHi - sizeLo) / (long) sizeof(double);

    CHECK_EQ_I(observedDelta, expectedDelta,
               "one additional pollutant must add exactly one "
               "OutfallStats.totalLoad and one Outfall.wRouted PER OUTFALL plus "
               "one LoadingTotals row and one QualTotals row. A short slope "
               "means the traversal stopped emitting before the Outfall[] "
               "block, which is a payload the reader of the SAME build reads "
               "short too -- so no round-trip node can see it");

    std::error_code ec;
    std::filesystem::remove(pathLo, ec);
    std::filesystem::remove(pathHi, ec);
}

static void T17_a_refused_save_publishes_nothing()
{
    // The WRITE-SIDE half of the killed-write mechanism, and the only half this
    // tier can reach: the resume-side consultation is a swmm_triton member
    // whose two inputs (rank_, snapshot_path_stem) are PRIVATE and whose only
    // writer is init_swmm, so a Tier-1 test can CALL it but cannot ARRANGE it.
    // That limitation is recorded in test/test_snapshot_stem_suppression_source.py
    // and is NOT worked around here.
    //
    // What IS reachable is snapshot_save's publish contract: write to a
    // temporary, verify the length, publish by rename. A REFUSED save must
    // leave NO file at the published name -- before that, a killed or refused
    // write left a SHORT file at the published name, the next resume passed it
    // through every pre-payload check, and a killed write and a build mismatch
    // produced the same refusal.
    const std::string path = temp_path("t17_refused");
    const std::string part = path + ".part";

    // The input the REPAIRED guard still refuses, and it is a real defect in
    // node.c's allocation rather than a routine state: routeTo >= 0 says the
    // array SHOULD exist, and it does not.
    build_pollutant_model(2, 0);
    std::free(Outfall[0].wRouted);
    Outfall[0].wRouted = nullptr;

    const int rc = snapshot_save(path.c_str());
    CHECK(rc != 0,
          "a routed outfall whose wRouted array is missing is a real allocation "
          "defect and the save must refuse it");
    CHECK(!std::filesystem::exists(path),
          "A REFUSED SAVE MUST PUBLISH NOTHING. A short file left at the "
          "published name is loaded by the next resume, clears the header, the "
          "shape block and the manifest, and is refused only at the first read "
          "past the written extent -- which is indistinguishable from a build "
          "mismatch and sends the operator to the wrong diagnosis");
    CHECK(!std::filesystem::exists(part),
          "a refused save must leave no .part, because a surviving .part reads "
          "at the next resume as a write cut off by a kill; the length-check "
          "arm keeps its bytes under .failed instead, which T32 asserts");

    free_pollutant_model();

    // The ACCEPTING arm, which is what proves the publish step runs at all
    // rather than that nothing was ever written.
    const std::string ok   = temp_path("t17_published");
    const std::string okPt = ok + ".part";
    build_pollutant_model(2, -1);
    CHECK_EQ_I(snapshot_save(ok.c_str()), 0,
               "a route-less outfall carrying pollutants must save");
    CHECK(std::filesystem::exists(ok),
          "an accepted save must publish under the name a resume will load");
    CHECK(!std::filesystem::exists(okPt),
          "the temporary must not survive an accepted save either -- the "
          "publish is a rename, not a copy");
    free_pollutant_model();

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(part, ec);
    std::filesystem::remove(path + ".failed", ec);
    std::filesystem::remove(ok, ec);
    std::filesystem::remove(okPt, ec);
}

// ===========================================================================
// THE RESUME-SIDE KILLED-WRITE CONSULTATION -- and the arrangement seam it
// needs, which is the non-obvious construct in this file.
// ===========================================================================
//
// WHAT IS BEING TESTED. try_restore_state_snapshot's three-arm consultation
// (swmm_triton.h), which decides what a resume does when a ".part" or
// ".killed" marker survives for a checkpoint and no published snapshot does.
// That state is what an interrupted write leaves, and the arm must REFUSE
// rather than fall back -- Absent is the arm that enters the exchange replay,
// and the replay is measured not to converge at a high checkpoint index.
//
// WHY A SEAM IS NEEDED AT ALL. The three entry points are PUBLIC
// (try_restore_state_snapshot, classify_missing_snapshot,
// sweep_snapshot_markers) but the two inputs they read are PRIVATE --
// `rank_` at swmm_triton.h:357 and `snapshot_path_stem` at :310, both after
// the `private:` specifier at :298 -- and the stem's ONLY assignment site is
// inside init_swmm, which calls swmm_open and therefore needs a real .inp. So
// a Tier-1 test can CALL the consultation but cannot ARRANGE it. Verified
// against the live header at this commit rather than carried from a note.
//
// THE MECHANISM, and it is the FRIEND-INJECTION form of the
// explicit-template-instantiation idiom rather than the naive form.
//
//   A tag struct declares a `friend` function returning a pointer-to-member
//   type. A template DEFINES that friend in its body and is EXPLICITLY
//   INSTANTIATED with the private member as a template argument. [temp.spec]
//   exempts names used to specify an explicit instantiation from access
//   checking, so the private member is named exactly once, inside the
//   instantiation. The call site then assigns THROUGH the returned
//   pointer-to-member and never names the private member at all.
//
//   The NAIVE form -- writing `&SwmmTriton::rank_` at the call site to fetch
//   the pointer -- DOES NOT COMPILE, at either -std=c++17 or -std=c++20:
//   "error: 'int swmm_triton::rank_' is private within this context". A call
//   site is not an exempt context. Only the explicit instantiation is.
//
// WHY NOT THE THREE OBVIOUS ALTERNATIVES. Each was considered and rejected,
// and the reasons are recorded here because a maintainer meeting the construct
// below without them will re-derive them or delete it.
//
//   1. A `friend` declaration for this test inside class swmm_triton.
//      REJECTED: it adds permanent production surface to a shared header for
//      the sake of a test. Note the distinction that makes the construct below
//      different rather than a workaround of the same thing -- the friend here
//      is declared in THIS FILE'S OWN tag structs, never in swmm_triton, so
//      the production header is byte-unchanged.
//
//   2. A test-only public setter on swmm_triton.
//      REJECTED: same objection, and worse -- it is reachable from production
//      code, so nothing keeps it test-only but convention.
//
//   3. `#define private public` before the include.
//      REJECTED: undefined behaviour, and specifically hazardous in this
//      translation unit, which parses Kokkos headers before swmm_triton.h --
//      the macro would reach standard-library and Kokkos internals and risk
//      ODR divergence against the Kokkos libraries this binary links.
//
//   A fourth route, calling the public initialize() at swmm_triton.h:177, was
//   rejected on cost rather than validity: 15 parameters including an
//   MpiUtils::partition_data_t descriptor and the full grid definition, with a
//   body reaching read_inp_file, process_swmm_node_locations and init_swmm
//   into swmm_open. It needs a real .inp fixture and an initialised MPI
//   context, which would reclassify this file from unit to integration for
//   EVERY assertion it holds -- against its own header, which states "No
//   solver run, no cluster, no simulation."
// ---------------------------------------------------------------------------

#include <type_traits>

using SwmmTriton  = SWMM_triton::swmm_triton;
using SnapRestore = SwmmTriton::SnapshotRestore;

struct RankTag { using type = int         SwmmTriton::*; friend type retrieve(RankTag); };
struct StemTag { using type = std::string SwmmTriton::*; friend type retrieve(StemTag); };
struct FailedReplayTag { using type = bool SwmmTriton::*; friend type retrieve(FailedReplayTag); };

template <typename Tag, typename Tag::type M>
struct Stow { friend typename Tag::type retrieve(Tag) { return M; } };

// The ONLY three sites in this file that name a private member of swmm_triton,
// and all three are explicit-instantiation arguments, which is the exempt context.
template struct Stow<RankTag, &SwmmTriton::rank_>;
template struct Stow<StemTag, &SwmmTriton::snapshot_path_stem>;
template struct Stow<FailedReplayTag, &SwmmTriton::snapshot_failed_replay>;

// ---------------------------------------------------------------------------
// (d)'s disposal site is ABSENT from the pre-fix tree, so the call to it must
// be instantiated only where it exists -- otherwise this translation unit does
// not COMPILE against that tree, the whole test binary fails to build, and the
// two-tree control loses every OTHER node in this file, including the
// payload-length reds. Measured at the base commit: `sweep_snapshot_markers`
// occurs 0 times in swmm_triton.h against 2 at this commit, with `class
// swmm_triton` occurring once in both as a must-be-non-zero control.
//
// A discarded `if constexpr` branch is still instantiated in a NON-template
// function, so the call has to sit in a function TEMPLATE for the discard to
// bite. The node below turns a false return into a FAILED ASSERTION, which is
// what keeps a missing disposal site a red assertion rather than a build
// failure -- the distinction DoD 10 draws.
// ---------------------------------------------------------------------------
template <typename T, typename = void>
struct has_disposal_site : std::false_type {};
template <typename T>
struct has_disposal_site<T, decltype(void(&T::sweep_snapshot_markers))> : std::true_type {};

template <typename T>
static bool sweep_markers_if_this_build_has_the_disposal_site(T& obj)
{
    if constexpr (has_disposal_site<T>::value) { obj.sweep_snapshot_markers(); return true; }
    else { (void) obj; return false; }
}

// Creates a private directory and returns the snapshot STEM inside it. A
// dedicated directory rather than the shared temp root, because the
// consultation and the classifier both ENUMERATE the stem's parent and a
// directory holding only this node's files makes the arrangement exact.
static std::string make_snapshot_stem(const char* tag)
{
    std::ostringstream oss;
    oss << std::filesystem::temp_directory_path().string()
        << "/triton_wp1_cons_" << tag << "_" << (long) getpid();
    const std::string dir = oss.str();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir + "/model_state";
}

// The path rule the production helper uses, reproduced here because that
// helper is private. Kept beside the stem builder so the two cannot drift
// apart unnoticed.
static std::string snapshot_path_at(const std::string& stem, int cid)
{
    return stem + "_cp" + std::to_string(cid) + ".snapshot";
}

// Arranges a default-constructed swmm_triton as rank 0 with a NON-EMPTY stem.
//
// NON-EMPTY IS LOAD-BEARING AND AN EMPTY STEM IS NOT A NEUTRAL DEFAULT.
// swmm_triton.h:1260-1272 records that an empty snapshot_path_stem is the
// INTENDED, configuration-visible route by which a caller FORCES the restore
// to decline -- init_swmm leaves it empty when swmm_snapshot_disable is set,
// and reaching that return "is not a sign of a defect and must not be read as
// one". An empty stem therefore makes every call below return Absent through
// the disable route and arranges none of the positive assertions.
static void arrange_rank0_with_stem(SwmmTriton& obj, const std::string& stem)
{
    obj.*retrieve(RankTag{}) = 0;
    obj.*retrieve(StemTag{}) = stem;
}

// Captures everything the consultation writes to std::cerr for the duration of
// one call. The refusal is emitted to the stream rather than into an errMsg
// buffer, so this is the only way to assert on its text.
class CerrCapture
{
public:
    CerrCapture() : prev_(std::cerr.rdbuf(buf_.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(prev_); }
    std::string str() const { return buf_.str(); }
private:
    std::ostringstream buf_;
    std::streambuf*    prev_;
};

// Writes a COMPLETE snapshot at `cid` using the live serializer, so the
// "prior snapshot complete" half of the fixture is a real artifact rather than
// a plausible-looking stub. No model is stood up: with every object count at
// zero the traversal emits its singleton and scalar surface, which is a
// complete payload for that shape.
static bool publish_complete_snapshot(const std::string& stem, int cid)
{
    return snapshot_save(snapshot_path_at(stem, cid).c_str()) == 0;
}

static void touch_marker(const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    f << "partial";
}

static void T18_killed_part_beside_a_complete_prior_snapshot_is_refused()
{
    // DoD 10(a). A ".part" at cp{N} beside a COMPLETE cp{N-1}, and the
    // consultation must return a NON-Restored value.
    //
    // `!= Restored` is asserted rather than `== KilledWrite`, and that is
    // deliberate rather than loose: Restored is one of the enumerators present
    // at the pre-fix commit and KilledWrite is NOT (measured 0 there against 3
    // here), so naming KilledWrite would make this node fail to COMPILE
    // against the control tree instead of failing an assertion there. The
    // DoD's own wording is "a non-Restored return", and the second CHECK below
    // is what makes that predicate discriminating rather than weak.
    const int N = 4;
    const std::string stem = make_snapshot_stem("t18");

    CHECK(publish_complete_snapshot(stem, N - 1),
          "the fixture must publish a COMPLETE prior snapshot at cp{N-1}");
    touch_marker(snapshot_path_at(stem, N) + ".part");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);

    SnapRestore sr;
    { CerrCapture cap; sr = obj.try_restore_state_snapshot(N, (value_t) 0.0); }

    CHECK(sr != SnapRestore::Restored,
          "a surviving .part at this checkpoint with no published snapshot at "
          "it must NOT report Restored -- the accumulators were never loaded");
    CHECK(sr != SnapRestore::Absent,
          "and it must NOT report Absent either: Absent is the one arm the "
          "caller is allowed to fall back on, and it routes into the exchange "
          "replay, which is measured not to converge at a high checkpoint "
          "index. A killed write misreported as Absent is the silent-wrong "
          "path this arm exists to close");

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(stem).parent_path(), ec);
}

static void T19_the_killed_write_refusal_names_cause_fallback_and_replay_cost()
{
    // DoD 10(b). Three properties of the refusal TEXT, asserted separately so
    // a result names which one is missing.
    //
    // Every span below is matched against the RUNTIME stream, not against
    // source lines, and each lies inside one emitted segment. That matters:
    // these messages are written as adjacent string literals wrapped across
    // source lines, so a span that reads as one sentence spans two lines on
    // disk and a line-anchored source grep over it returns 0 on text the file
    // plainly carries. Matching the stream sidesteps the wrapping; the spans
    // additionally avoid the embedded newlines.
    const int N = 7;
    const std::string stem = make_snapshot_stem("t19");

    CHECK(publish_complete_snapshot(stem, N - 1),
          "the fixture must publish a COMPLETE prior snapshot at cp{N-1}");
    touch_marker(snapshot_path_at(stem, N) + ".part");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);

    std::string out;
    { CerrCapture cap; (void) obj.try_restore_state_snapshot(N, (value_t) 0.0); out = cap.str(); }

    CHECK(out.find("was KILLED before it completed.") != std::string::npos,
          "the refusal must name the KILLED WRITE as the cause -- before this, "
          "a killed write and a build mismatch produced the same refusal and "
          "an operator could not tell them apart");

    CHECK(out.find("highest COMPLETE snapshot found below") != std::string::npos,
          "the refusal must name a COMPLETE prior snapshot as the fallback");
    CHECK(out.find(snapshot_path_at(stem, N - 1)) != std::string::npos,
          "and it must name that snapshot BY PATH rather than merely assert one "
          "exists -- the path is what an operator resumes from");

    CHECK(out.find("each successive resume replays a longer prefix") != std::string::npos,
          "the refusal must state the replay cost as growing with the "
          "checkpoint index, which is why the replay is not offered as a "
          "substitute here");

    if (out.find("was KILLED before it completed.") == std::string::npos)
        std::fprintf(stderr, "  note: captured stderr was:\n%s\n", out.c_str());

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(stem).parent_path(), ec);
}

static void T20_the_part_is_renamed_killed_and_the_classifier_returns_an_existing_token()
{
    // DoD 10(c). Two properties: the rename, and that the classifier's reason
    // vocabulary did not grow.
    //
    // THE RENAME IS WHAT MAKES THE REFUSAL IDEMPOTENT, which is the half a
    // reader is most likely to mistake for tidiness. Keyed on ".part" OR
    // ".killed", every resume at N refuses with the same message; keyed on
    // ".part" alone, the first resume refused and every resume after it saw
    // only a ".killed", fell through to Absent, and was placed into the replay
    // with no refusal emitted at all.
    const int N = 3;
    const std::string stem = make_snapshot_stem("t20");

    CHECK(publish_complete_snapshot(stem, N - 1),
          "the fixture must publish a COMPLETE prior snapshot at cp{N-1}");
    const std::string part   = snapshot_path_at(stem, N) + ".part";
    const std::string killed = snapshot_path_at(stem, N) + ".killed";
    touch_marker(part);

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    { CerrCapture cap; (void) obj.try_restore_state_snapshot(N, (value_t) 0.0); }

    CHECK(!std::filesystem::exists(part),
          "the .part must not survive the consultation");
    CHECK(std::filesystem::exists(killed),
          "the .part must be RENAMED to .killed -- that is what makes a second "
          "resume at the same checkpoint refuse again instead of seeing no "
          ".part, falling through to Absent, and entering the replay silently");

    // A SECOND consultation, which is the property the rename buys and which a
    // single call cannot observe.
    SwmmTriton obj2;
    arrange_rank0_with_stem(obj2, stem);
    SnapRestore sr2;
    { CerrCapture cap; sr2 = obj2.try_restore_state_snapshot(N, (value_t) 0.0); }
    CHECK(sr2 != SnapRestore::Absent,
          "a SECOND resume at the same checkpoint must refuse too, keyed on the "
          ".killed the first one left");

    const std::string reason = obj.classify_missing_snapshot(N);
    const bool is_existing_token = (reason == "absent")
                                || (reason == "pre-snapshot-checkpoint")
                                || (reason == "retention-collision");
    CHECK(is_existing_token,
          "the classifier must return one of its EXISTING reason tokens. A new "
          "token would cost an interface item and would reach the harvester as "
          "an unrecognized value, so the killed-write state is carried by the "
          ".killed file plus the refusal text rather than by a fourth reason");
    if (!is_existing_token)
        std::fprintf(stderr, "  note: classifier returned '%s'\n", reason.c_str());

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(stem).parent_path(), ec);
}

static void T21_a_clean_start_disposes_both_markers()
{
    // DoD 10(d). A clean start is the ONE point at which a marker is provably
    // no longer evidence -- it consults no snapshot, so no ".part" or
    // ".killed" can be telling it anything. On the ABORT path the marker is
    // deliberately RETAINED, because there it is the artifact that lets an
    // operator see a killed write rather than a missing snapshot. So this node
    // asserts disposal at the clean start and nowhere else.
    //
    // The third assertion was INVERTED when the clean start began disposing of
    // the earlier run's PUBLISHED snapshots too: left in place, one of them
    // would be loaded by a later resume of the new run whose own write at that
    // checkpoint failed, with no error. With the keep-previous policy unset
    // (the default this node runs under) the published snapshot is removed.
    const std::string stem = make_snapshot_stem("t21");

    const std::string part   = snapshot_path_at(stem, 1) + ".part";
    const std::string killed = snapshot_path_at(stem, 2) + ".killed";
    touch_marker(part);
    touch_marker(killed);
    CHECK(std::filesystem::exists(part) && std::filesystem::exists(killed),
          "the fixture must stand up both marker kinds");

    // A published snapshot from the EARLIER run, which the clean start must
    // now dispose of as well.
    CHECK(publish_complete_snapshot(stem, 5),
          "the fixture must publish an earlier-run snapshot");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);

    const bool ran = sweep_markers_if_this_build_has_the_disposal_site(obj);
    CHECK(ran,
          "this build must carry the clean-start marker-disposal site. A build "
          "without it has no point at which a stale marker is reclaimed, so a "
          "resume at that checkpoint refuses on a marker whose condition has "
          "already lapsed");

    if (ran)
    {
        CHECK(!std::filesystem::exists(part),   "a clean start must leave no .part");
        CHECK(!std::filesystem::exists(killed), "a clean start must leave no .killed");
        CHECK(!std::filesystem::exists(snapshot_path_at(stem, 5)),
              "and with keep-previous unset it must leave the earlier run's "
              "PUBLISHED snapshot ABSENT at its name -- a survivor would be "
              "loaded by a later resume of the new run at that checkpoint");
    }

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(stem).parent_path(), ec);
}

// ===========================================================================
// THE STEP-BACK LINES, THE FAILED-WRITE STOP, AND THE CLEAN-START DISPOSAL.
// ===========================================================================
//
// WHAT IS BEING TESTED. Three behaviours added together:
//   1. on a resume at a checkpoint whose snapshot write was cut off or failed,
//      the refusal prints ONE machine-readable line telling the orchestration
//      what to do next -- resume from an earlier published snapshot, stop, or
//      start clean -- and a resume that restored from a snapshot below a
//      still-marked checkpoint prints a confirmation naming both ids;
//   2. a snapshot write that fails the writer's own length check keeps its
//      bytes as "{name}.failed", and a resume there STOPS;
//   3. a clean start disposes of the earlier run's published snapshots and
//      markers -- removing them, or renaming them to "{name}.prev{k}" when the
//      keep-previous policy is set -- and keeps a ".failed" either way.
//
// The machine-readable lines are a contract another repository parses, so the
// tests below spell each one out in full rather than matching a fragment: a
// reworded line is a broken contract even when every fragment survives.
//
// TWO-TREE RULE, carried from the nodes above. Every new entry point is called
// through a detection template, so this file still COMPILES against a tree
// that lacks it and the missing behaviour shows up as a failed assertion
// rather than as a build failure that would also hide every other node.
// ---------------------------------------------------------------------------
#include <utility>
#include <csignal>
#include <sys/resource.h>

template <typename T, typename = void>
struct has_dispose : std::false_type {};
template <typename T>
struct has_dispose<T, std::void_t<decltype(std::declval<T&>().dispose_snapshot_artifacts(true))>>
    : std::true_type {};

template <typename T>
static bool dispose_if_present(T& obj, bool keep_previous, std::string* refused)
{
    refused->clear();
    if constexpr (has_dispose<T>::value) { *refused = obj.dispose_snapshot_artifacts(keep_previous); return true; }
    else { (void) obj; (void) keep_previous; return false; }
}

template <typename T, typename = void>
struct has_live_marker_scan : std::false_type {};
template <typename T>
struct has_live_marker_scan<T, std::void_t<decltype(std::declval<const T&>().highest_live_marker_above(0))>>
    : std::true_type {};

// Returns -2 when this build has no scan at all, so a caller can tell "no
// marker found" (-1) from "nothing to call".
template <typename T>
static int live_marker_scan_if_present(const T& obj, int checkpoint_id)
{
    if constexpr (has_live_marker_scan<T>::value) return obj.highest_live_marker_above(checkpoint_id);
    else { (void) obj; (void) checkpoint_id; return -2; }
}

static std::string read_bytes(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

static void write_bytes(const std::string& p, const std::string& s)
{
    std::ofstream f(p, std::ios::binary);
    f << s;
}

static std::string dir_of(const std::string& stem)
{
    return std::filesystem::path(stem).parent_path().string();
}

static std::set<std::string> listing(const std::string& dir)
{
    std::set<std::string> names;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
        names.insert(e.path().filename().string());
    return names;
}

static std::size_t occurrences(const std::string& hay, const std::string& needle)
{
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos;
         at = hay.find(needle, at + needle.size()))
        ++n;
    return n;
}

// The three KilledWrite lines and the confirmation line, spelled in full.
static std::string row_step_back(int n, int m)
{
    return "SWMM snapshot step-back required: checkpoint " + std::to_string(n)
         + " unusable [resume-event checkpoint=" + std::to_string(n)
         + " path=step-back reason=killed-write target=" + std::to_string(m) + "]";
}
static std::string row_stop(int n, int m)
{
    return "SWMM snapshot stop: checkpoint " + std::to_string(n)
         + " unusable [resume-event checkpoint=" + std::to_string(n)
         + " path=stop reason=write-failed target=" + std::to_string(m) + "]";
}
static std::string row_clean_start(int n, const char* reason)
{
    return "SWMM snapshot step-back impossible: checkpoint " + std::to_string(n)
         + " unusable, no published snapshot below it [resume-event checkpoint="
         + std::to_string(n) + " path=clean-start reason=" + reason + " target=0]";
}

static std::string resume_and_capture(const std::string& stem, int n)
{
    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    std::string out;
    { CerrCapture cap; (void) obj.try_restore_state_snapshot(n, (value_t) 0.0); out = cap.str(); }
    return out;
}

// True when the test is NOT running as root. Permission-based fixtures do not
// bind for root, so under root they would pass without testing anything; the
// nodes that rely on them assert this first and fail loudly instead.
static bool permissions_bind()
{
    return geteuid() != 0;
}

static void T22_a_killed_write_with_a_published_snapshot_below_emits_the_step_back_line()
{
    // DoD 10(e).
    // Property: a ".part" at cp{N} beside a published cp{N-1} prints the
    //   step-back line naming N-1 as the target, exactly once, and the
    //   existing human refusal text is unchanged.
    // Class: NEW CAPABILITY. A plausible wrong implementation prints the line
    //   with the stale "write-failed" reason or prints it twice (once per
    //   marker); both make the count-of-one check red.
    // Why required: the orchestration relaunches from the named target; a
    //   missing or duplicated line is a relaunch from the wrong checkpoint or
    //   none.
    // What kills it: deleting the step-back branch, or changing any byte of the
    //   line, reds the first check.
    // A second correct implementation that still passes: building the line
    //   with one snprintf instead of stream insertions, or emitting it before
    //   the human text rather than after.
    const int N = 5;
    const std::string stem = make_snapshot_stem("t22");
    CHECK(publish_complete_snapshot(stem, N - 1), "the fixture must publish cp{N-1}");
    touch_marker(snapshot_path_at(stem, N) + ".part");

    const std::string out = resume_and_capture(stem, N);
    CHECK_EQ_I(occurrences(out, row_step_back(N, N - 1)), 1,
               "the step-back line naming N-1 must appear exactly once");
    CHECK(out.find("was KILLED before it completed.") != std::string::npos,
          "the human refusal text must still name the killed write");
    CHECK(out.find("highest COMPLETE snapshot found below") != std::string::npos,
          "and still name the complete snapshot below");
    CHECK(out.find(snapshot_path_at(stem, N - 1)) != std::string::npos,
          "and still name it by path");
    CHECK(out.find("each successive resume replays a longer prefix") != std::string::npos,
          "and still state the replay cost");
    if (occurrences(out, row_step_back(N, N - 1)) != 1)
        std::fprintf(stderr, "  note: captured stderr was:\n%s\n", out.c_str());

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

static void T23_no_published_snapshot_below_emits_the_clean_start_line_with_its_reason()
{
    // DoD 10(f), both arms.
    // Property: with no published snapshot below N the line says only a clean
    //   start remains, and its reason names the marker found -- killed-write
    //   for a ".part", write-failed for a ".failed".
    // Class: NEW CAPABILITY. A plausible wrong implementation hard-codes
    //   reason=killed-write; the ".failed" arm reds.
    // Why required: the orchestration raises a different alarm for a failed
    //   write than for a killed one, keyed on this token.
    // What kills it: dropping the ".failed" case from the reason choice.
    // A second correct implementation that still passes: deciding the reason
    //   from a single precedence function shared with the human text.
    const int N = 3;
    const std::string stemA = make_snapshot_stem("t23a");
    touch_marker(snapshot_path_at(stemA, N) + ".part");
    const std::string outA = resume_and_capture(stemA, N);
    CHECK_EQ_I(occurrences(outA, row_clean_start(N, "killed-write")), 1,
               "a .part with nothing published below must print the clean-start "
               "line with reason=killed-write");

    const std::string stemB = make_snapshot_stem("t23b");
    touch_marker(snapshot_path_at(stemB, N) + ".failed");
    const std::string outB = resume_and_capture(stemB, N);
    CHECK_EQ_I(occurrences(outB, row_clean_start(N, "write-failed")), 1,
               "a .failed with nothing published below must print the clean-start "
               "line with reason=write-failed");
    CHECK(outB.find("This resume STOPS") != std::string::npos,
          "and the human text must state the stop");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stemA), ec);
    std::filesystem::remove_all(dir_of(stemB), ec);
}

static void T24_a_failed_write_with_a_published_snapshot_below_emits_the_stop_line()
{
    // DoD 10(g).
    // Property: a ".failed" at cp{N} beside a published cp{N-1} prints the
    //   STOP line exactly once, prints no step-back line, states the stop in
    //   the human text without offering the earlier snapshot as a route, keeps
    //   the ".failed", and prints the identical stop line on a second resume.
    // Class: NEW CAPABILITY. A plausible wrong implementation treats ".failed"
    //   like ".killed" and prints the step-back line; checks 1 and 2 red.
    // Why required: a failed write is ruled a stop, so a line telling the
    //   orchestration to step back would relaunch past a live fault.
    // What kills it: removing ".failed" from the precedence, or from Arm 2's
    //   key (then the resume reaches the snapshot load and returns Absent).
    // A second correct implementation that still passes: printing the human
    //   stop text after the machine line rather than before it.
    const int N = 6;
    const std::string stem = make_snapshot_stem("t24");
    CHECK(publish_complete_snapshot(stem, N - 1), "the fixture must publish cp{N-1}");
    const std::string failed = snapshot_path_at(stem, N) + ".failed";
    touch_marker(failed);

    const std::string out = resume_and_capture(stem, N);
    CHECK_EQ_I(occurrences(out, row_stop(N, N - 1)), 1,
               "the stop line naming N-1 for information must appear exactly once");
    CHECK(out.find("SWMM snapshot step-back required") == std::string::npos,
          "a failed write must not print a step-back line");
    CHECK(out.find("This resume STOPS") != std::string::npos,
          "the human text must state the stop");
    CHECK(out.find("the cheap route") == std::string::npos,
          "and must not offer the earlier snapshot as a route");
    CHECK(std::filesystem::exists(failed), "the .failed marker must survive the resume");

    const std::string out2 = resume_and_capture(stem, N);
    CHECK_EQ_I(occurrences(out2, row_stop(N, N - 1)), 1,
               "a second resume at N must print the identical stop line");
    if (occurrences(out, row_stop(N, N - 1)) != 1)
        std::fprintf(stderr, "  note: captured stderr was:\n%s\n", out.c_str());

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

static void T25_the_step_back_line_is_identical_on_a_second_resume()
{
    // DoD 10(h).
    // Property: the second resume at N, which now finds a ".killed" rather
    //   than a ".part", prints the byte-identical step-back line.
    // Class: NEW CAPABILITY. A plausible wrong implementation reads the
    //   reason from which marker was renamed and so prints a different line,
    //   or none, on the second resume.
    // Why required: every relaunch reads the line from the most recent log
    //   only; a second resume printing nothing would strand the run.
    // What kills it: keying the line on havePart alone.
    // A second correct implementation that still passes: not renaming the
    //   ".part" at all and keying on it alone across both resumes -- the line
    //   is the same either way.
    const int N = 4;
    const std::string stem = make_snapshot_stem("t25");
    CHECK(publish_complete_snapshot(stem, N - 1), "the fixture must publish cp{N-1}");
    touch_marker(snapshot_path_at(stem, N) + ".part");

    const std::string out1 = resume_and_capture(stem, N);
    CHECK(std::filesystem::exists(snapshot_path_at(stem, N) + ".killed"),
          "the first resume must leave a .killed, so the second sees a different marker");
    const std::string out2 = resume_and_capture(stem, N);
    CHECK_EQ_I(occurrences(out1, row_step_back(N, N - 1)), 1, "first resume prints the line once");
    CHECK_EQ_I(occurrences(out2, row_step_back(N, N - 1)), 1,
               "the second resume prints the identical line once");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

static void T26_the_step_back_confirmation_scan_reads_only_live_markers()
{
    // DoD 10(i).
    // Property: the scan returns the highest checkpoint above M carrying an
    //   EXACT marker name with no published name at that id; a marker made
    //   stale by a later publication, and a retained "*.prev{k}" name, are not
    //   live.
    // Class: NEW CAPABILITY. A plausible wrong implementation matches
    //   ".failed" or ".killed" as a substring; the retained ".prev1" names at
    //   7 and 8 then win over 5 and the first check reds.
    // Why required: the confirmation line's from= is read as the checkpoint
    //   the resume stepped over; a wrong id raises a false mismatch alarm.
    // What kills it: dropping the published-name check (then the second check
    //   returns 5), or substring matching (then the first returns 8).
    // A second correct implementation that still passes: probing each id from
    //   a known upper bound downward by exact-name existence instead of
    //   listing the directory.
    const int M = 2;
    const std::string stem = make_snapshot_stem("t26");
    CHECK(publish_complete_snapshot(stem, M), "the fixture must publish cp{M}");
    touch_marker(snapshot_path_at(stem, 5) + ".killed");
    touch_marker(snapshot_path_at(stem, 7) + ".failed.prev1");
    touch_marker(snapshot_path_at(stem, 8) + ".killed.prev1");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    const int first = live_marker_scan_if_present(obj, M);
    CHECK(first != -2, "this build must carry the live-marker scan");
    CHECK_EQ_I(first, 5, "a live .killed at 5 above M must be returned, and the "
                         "retained .prev1 names at 7 and 8 must not be");

    CHECK(publish_complete_snapshot(stem, 5), "publish cp5 so its marker lapses");
    CHECK_EQ_I(live_marker_scan_if_present(obj, M), -1,
               "with cp5 published its .killed has lapsed and nothing is live");

    touch_marker(snapshot_path_at(stem, 9) + ".failed");
    CHECK_EQ_I(live_marker_scan_if_present(obj, M), 9,
               "an exact .failed with nothing published at its id is live");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

// The fixture DoD 10(j) names: a ".failed" with two retained names on the SAME
// base, and the directory maximum on a DIFFERENT base, so one fixture catches
// a string maximum (picks 10 and overwrites the same-base .prev10) and a
// per-base maximum (picks 11).
static void j_fixture(const std::string& stem, int k, std::string* failed, std::string* x)
{
    *failed = snapshot_path_at(stem, k) + ".failed";
    *x = dir_of(stem) + "/x.prev12";
    write_bytes(*failed, "failed-bytes");
    write_bytes(*failed + ".prev9", "prev9-bytes");
    write_bytes(*failed + ".prev10", "prev10-bytes");
    write_bytes(*x, "x-bytes");
}

static void T27_a_clean_start_keeps_a_failed_marker_under_one_directory_wide_k()
{
    // DoD 10(j), under both keep-previous policies.
    // Property: the ".failed" is renamed to ".prev13" (one more than the
    //   directory-wide maximum 12), byte-identical, nothing else changes, and
    //   no file is lost.
    // Class: NEW CAPABILITY. Plausible wrong implementations: a string maximum
    //   picks 10 and replaces the existing .prev10 (checks 2, 3, 4 red); a
    //   per-base maximum picks 11 (check 2 red); removing the .failed when the
    //   policy is unset (checks 2 and 4 red on the first fixture).
    // Why required: the .failed is the evidence for a ruled stop, and the
    //   operator's response to that stop is this clean start.
    // What kills it: any of the three wrong implementations above.
    // A second correct implementation that still passes: computing k by
    //   probing ".prev1", ".prev2", ... until a free number is found ABOVE
    //   every existing one, instead of parsing the maximum.
    for (int pass = 0; pass < 2; ++pass)
    {
        const bool keep = (pass == 1);
        const int K = 4;
        const std::string stem = make_snapshot_stem(keep ? "t27k" : "t27n");
        std::string failed, x;
        j_fixture(stem, K, &failed, &x);
        const std::size_t before = listing(dir_of(stem)).size();

        SwmmTriton obj;
        arrange_rank0_with_stem(obj, stem);
        std::string refused, out;
        bool ran;
        { CerrCapture cap; ran = dispose_if_present(obj, keep, &refused); out = cap.str(); }
        CHECK(ran, "this build must carry the clean-start disposal");
        CHECK(refused.empty(), "nothing here should refuse the clean start");

        CHECK(!std::filesystem::exists(failed), "no file may remain under the .failed name");
        CHECK(read_bytes(failed + ".prev13") == "failed-bytes",
              "the former marker must be kept as .prev13, byte-identical "
              "(.prev10 is the string-maximum answer and .prev11 the per-base one)");
        CHECK(read_bytes(failed + ".prev9") == "prev9-bytes"
              && read_bytes(failed + ".prev10") == "prev10-bytes"
              && read_bytes(x) == "x-bytes",
              "the existing retained names must be byte-unchanged");
        CHECK_EQ_I(listing(dir_of(stem)).size(), before, "no file may be lost");

        std::error_code ec;
        std::filesystem::remove_all(dir_of(stem), ec);
    }
}

// The fixture DoD 10(k1)-(k3) share: an earlier-run published snapshot, one
// marker of each kind, and a retained name carrying the directory maximum.
struct KFixture { std::string stem, published, part, killed, failed, x; };

static KFixture k_fixture(const char* tag)
{
    KFixture f;
    f.stem      = make_snapshot_stem(tag);
    f.published = snapshot_path_at(f.stem, 3);
    f.part      = snapshot_path_at(f.stem, 5) + ".part";
    f.killed    = snapshot_path_at(f.stem, 6) + ".killed";
    f.failed    = snapshot_path_at(f.stem, 7) + ".failed";
    f.x         = dir_of(f.stem) + "/x.prev12";
    write_bytes(f.published, "published-bytes");
    write_bytes(f.part, "part-bytes");
    write_bytes(f.killed, "killed-bytes");
    write_bytes(f.failed, "failed-bytes");
    write_bytes(f.x, "x-bytes");
    return f;
}

static void T28_a_clean_start_without_keep_previous_removes_the_earlier_run()
{
    // DoD 10(k1).
    // Property: with the policy unset, the earlier run's published snapshot,
    //   ".part" and ".killed" are removed, the ".failed" is kept as ".prev13",
    //   the retained name is untouched, and exactly three files go.
    // Class: REGRESSION for the published snapshot (the pre-change sweep left
    //   it in place) and NEW CAPABILITY for the ".failed" rename.
    // Why required: a surviving earlier-run snapshot is loaded by a later
    //   resume of the new run whose own write at that checkpoint failed.
    // What kills it: skipping published names, or removing the ".failed".
    // A second correct implementation that still passes: processing all
    //   entries in one loop with the published ones sorted first.
    const KFixture f = k_fixture("t28");
    const std::size_t before = listing(dir_of(f.stem)).size();
    SwmmTriton obj;
    arrange_rank0_with_stem(obj, f.stem);
    std::string refused;
    bool ran;
    { CerrCapture cap; ran = dispose_if_present(obj, false, &refused); }
    CHECK(ran, "this build must carry the clean-start disposal");
    CHECK(refused.empty(), "nothing here should refuse");
    CHECK(!std::filesystem::exists(f.published), "the earlier-run published snapshot must be removed");
    CHECK(!std::filesystem::exists(f.part), "the .part must be removed");
    CHECK(!std::filesystem::exists(f.killed), "the .killed must be removed");
    CHECK(read_bytes(f.failed + ".prev13") == "failed-bytes" && !std::filesystem::exists(f.failed),
          "the .failed must be kept as .prev13, byte-identical");
    CHECK(read_bytes(f.x) == "x-bytes", "the retained name must be untouched");
    CHECK_EQ_I((long) before - (long) listing(dir_of(f.stem)).size(), 3,
               "exactly three files must be removed");
    std::error_code ec;
    std::filesystem::remove_all(dir_of(f.stem), ec);
}

static void T29_a_clean_start_with_keep_previous_keeps_every_artifact()
{
    // DoD 10(k2).
    // Property: with the policy set, all four earlier-run artifacts are
    //   renamed to "{name}.prev13" under ONE k, byte-identical, and nothing is
    //   lost.
    // Class: NEW CAPABILITY. A plausible wrong implementation recomputes k per
    //   file, so the second rename uses 14; the per-file checks red.
    // Why required: the policy exists so an operator debugging a run can keep
    //   every earlier artifact together under one suffix.
    // What kills it: removing instead of renaming any class, or a per-file k.
    // A second correct implementation that still passes: copying each file to
    //   its retained name and then removing the original.
    const KFixture f = k_fixture("t29");
    const std::size_t before = listing(dir_of(f.stem)).size();
    SwmmTriton obj;
    arrange_rank0_with_stem(obj, f.stem);
    std::string refused;
    bool ran;
    { CerrCapture cap; ran = dispose_if_present(obj, true, &refused); }
    CHECK(ran, "this build must carry the clean-start disposal");
    CHECK(refused.empty(), "nothing here should refuse");
    CHECK(read_bytes(f.published + ".prev13") == "published-bytes", "published kept as .prev13");
    CHECK(read_bytes(f.part + ".prev13") == "part-bytes", ".part kept as .prev13");
    CHECK(read_bytes(f.killed + ".prev13") == "killed-bytes", ".killed kept as .prev13");
    CHECK(read_bytes(f.failed + ".prev13") == "failed-bytes", ".failed kept as .prev13");
    CHECK(!std::filesystem::exists(f.published) && !std::filesystem::exists(f.part)
          && !std::filesystem::exists(f.killed) && !std::filesystem::exists(f.failed),
          "no artifact may remain under its original name");
    CHECK(read_bytes(f.x) == "x-bytes", "the retained name must be untouched");
    CHECK_EQ_I(listing(dir_of(f.stem)).size(), before, "no file may be lost");
    std::error_code ec;
    std::filesystem::remove_all(dir_of(f.stem), ec);
}

static void T30_an_undisposable_published_snapshot_refuses_the_clean_start()
{
    // DoD 10(k3), under both policies.
    // Property: when the earlier-run published snapshot cannot be disposed
    //   of, the call returns its path and touches no entry at all.
    // Class: NEW CAPABILITY. A plausible wrong implementation warns and
    //   continues, returning success; the refusal check reds.
    // Why required: the caller stops the run on this return, because a
    //   surviving earlier-run snapshot is the cross-run hazard.
    // What kills it: treating a published-name failure like a marker failure.
    // A second correct implementation that still passes: checking write
    //   permission on the directory up front and refusing before any attempt.
    // The caller's stop is in src/triton.h and is checked at source, because
    // an abort ends the process.
    CHECK(permissions_bind(),
          "this node must not run as root: a read-only directory does not bind "
          "for root, so the refusal would never be exercised");
    if (!permissions_bind()) return;

    for (int pass = 0; pass < 2; ++pass)
    {
        const bool keep = (pass == 1);
        const KFixture f = k_fixture(keep ? "t30k" : "t30n");
        const std::string dir = dir_of(f.stem);
        const std::set<std::string> before = listing(dir);
        std::error_code pec;
        std::filesystem::permissions(dir, std::filesystem::perms::owner_read
                                        | std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace, pec);
        SwmmTriton obj;
        arrange_rank0_with_stem(obj, f.stem);
        std::string refused;
        bool ran;
        { CerrCapture cap; ran = dispose_if_present(obj, keep, &refused); }
        std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace, pec);
        CHECK(ran, "this build must carry the clean-start disposal");
        CHECK(refused == f.published, "the refusal must name the published snapshot");
        CHECK(listing(dir) == before, "no entry may be renamed or removed");

        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
}

static void T31_an_undisposable_marker_warns_and_the_clean_start_continues()
{
    // DoD 10(k4).
    // Property: a ".part" name that cannot be removed (it is a non-empty
    //   directory) produces a warning naming it, the call still succeeds, and
    //   the published snapshot is removed.
    // Class: REGRESSION guard. The pre-change sweep already warned on a marker
    //   it could not remove; this keeps that property while the published
    //   snapshot is now also removed (that half is red on the pre-change tree).
    // Why required: no marker is ever loaded as state, so stopping a clean
    //   start over one would cost a run for nothing.
    // What kills it: returning the marker's path as a refusal.
    // A second correct implementation that still passes: checking whether the
    //   marker name is a directory first and warning without attempting the
    //   removal at all.
    const std::string stem = make_snapshot_stem("t31");
    const std::string published = snapshot_path_at(stem, 3);
    const std::string partDir   = snapshot_path_at(stem, 5) + ".part";
    write_bytes(published, "published-bytes");
    std::error_code mec;
    std::filesystem::create_directories(partDir, mec);
    write_bytes(partDir + "/occupant", "x");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    std::string refused, out;
    bool ran;
    { CerrCapture cap; ran = dispose_if_present(obj, false, &refused); out = cap.str(); }
    CHECK(ran, "this build must carry the clean-start disposal");
    CHECK(refused.empty(), "a marker failure must not refuse the clean start");
    CHECK(out.find(partDir) != std::string::npos, "the warning must name the marker");
    CHECK(!std::filesystem::exists(published), "the published snapshot must still be removed");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

static void T32_a_failed_snapshot_write_keeps_its_marker_and_says_the_resume_will_stop()
{
    // DoD 10(l1), plus the no-marker counterpart of the warning.
    // Property: a write whose payload fails to reach the disk lands in the
    //   writer's length-check arm, publishes nothing, keeps "{path}.failed",
    //   and the warning says the next resume will STOP; a write that fails
    //   before anything is written (the directory is read-only) keeps no
    //   marker and the warning says the resume starts from an earlier snapshot
    //   or replays.
    // Class: NEW CAPABILITY. A plausible wrong implementation keeps the old
    //   single warning text, which claims a replay for both; arm 1's text
    //   check reds, and the old writer removed the temporary, so its marker
    //   check reds too.
    // Why required: the operator reads this warning while the run is still
    //   going; a text promising a replay for a write that will stop the resume
    //   is a false statement about the next job.
    // What kills it: removing the temporary on the length-check arm, or
    //   printing one text regardless of the marker.
    // A second correct implementation that still passes: deciding the text
    //   from snapshot_save's return value if it distinguished the arms, rather
    //   than from the marker's existence.
    //
    // HOW ARM 1 REACHES THE LENGTH-CHECK ARM, and why it is not T17's input. A
    // model with a missing array is refused by the COUNT pass that runs before
    // the temporary is even opened -- every structural guard is gated on the
    // mode being other than the manifest pass, so the count pass refuses
    // exactly as the write pass would -- and snapshot_save returns before
    // writing any snapshot bytes (it leaves a one-line ".failed" instead, which
    // T33 asserts). The length-check arm is reached only when the WRITE
    // itself fails after the header. So arm 1 lowers the process's file-size
    // limit to one double short of a complete snapshot of the same model: the
    // header and manifest fit, the payload does not. SIGXFSZ is ignored for
    // the duration so the oversize write fails with an error instead of
    // ending the process.
    const int N = 3;
    {
        const std::string stem = make_snapshot_stem("t32a");
        const std::string path = snapshot_path_at(stem, N);

        // Measure a complete snapshot of the same (empty) model first.
        const std::string probe = dir_of(stem) + "/probe.snapshot";
        CHECK_EQ_I(snapshot_save(probe.c_str()), 0, "the probe save must succeed");
        const long full = file_size_of(probe);
        std::error_code rmec;
        std::filesystem::remove(probe, rmec);
        CHECK(full > 8, "the probe snapshot must carry a payload");

        struct rlimit old_lim;
        getrlimit(RLIMIT_FSIZE, &old_lim);
        struct rlimit low_lim = old_lim;
        low_lim.rlim_cur = (rlim_t) (full - 8);
        void (*old_handler)(int) = std::signal(SIGXFSZ, SIG_IGN);
        const bool limited = setrlimit(RLIMIT_FSIZE, &low_lim) == 0;
        CHECK(limited, "the test must be able to lower its own file-size limit");

        SwmmTriton obj;
        arrange_rank0_with_stem(obj, stem);
        std::string out;
        { CerrCapture cap; obj.write_state_snapshot(N); out = cap.str(); }

        setrlimit(RLIMIT_FSIZE, &old_lim);
        std::signal(SIGXFSZ, old_handler);

        CHECK(!std::filesystem::exists(path), "a refused write must publish nothing");
        CHECK(std::filesystem::exists(path + ".failed"), "the length-check arm must keep {path}.failed");
        {
            const std::string kept = read_bytes(path + ".failed");
            CHECK(kept.size() > 16 && kept.rfind("SNAPSHOT REFUSED BEFORE WRITE", 0) != 0,
                  "and that .failed must hold the partial snapshot bytes, not the "
                  "count-pass refusal line");
        }
        CHECK(!std::filesystem::exists(path + ".part"), "and leave no .part behind");
        CHECK(out.find("will STOP with an error naming it") != std::string::npos,
              "the warning must say the resume will stop");
        CHECK(out.find("step back") == std::string::npos,
              "and must not say step back");
        std::error_code ec;
        std::filesystem::remove_all(dir_of(stem), ec);
    }

    CHECK(permissions_bind(),
          "this node must not run as root: a read-only directory does not bind "
          "for root, so the open failure would never be exercised");
    if (!permissions_bind()) return;
    {
        const std::string stem = make_snapshot_stem("t32b");
        const std::string dir  = dir_of(stem);
        const std::string path = snapshot_path_at(stem, N);
        std::error_code pec;
        std::filesystem::permissions(dir, std::filesystem::perms::owner_read
                                        | std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace, pec);
        SwmmTriton obj;
        arrange_rank0_with_stem(obj, stem);
        std::string out;
        { CerrCapture cap; obj.write_state_snapshot(N); out = cap.str(); }
        std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace, pec);

        CHECK(!std::filesystem::exists(path) && !std::filesystem::exists(path + ".part")
              && !std::filesystem::exists(path + ".failed"),
              "a write that could not open its temporary must leave no file at all");
        CHECK(out.find("no marker was kept") != std::string::npos,
              "the warning must say no marker was kept");
        CHECK(out.find("itself replays the exchange history") != std::string::npos,
              "and name the replay for a launch at this checkpoint");
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
}

// The one line a count-pass refusal writes into its ".failed" marker. Spelled
// out here rather than taken from snapshot.c, so a reworded line reds the
// test instead of moving with it.
static const char* const kRefusedLine =
    "SNAPSHOT REFUSED BEFORE WRITE: a structural guard rejected the model state in the count pass";

static void T33_a_count_pass_refusal_leaves_a_one_line_failed_marker()
{
    // DoD 10(l2).
    // Property: a save the COUNT pass refuses (a routed outfall whose wRouted
    //   array is missing) publishes nothing, writes no .part, and leaves
    //   "{path}.failed" holding exactly the refusal line; the warning says
    //   the next resume will STOP.
    // Class: NEW CAPABILITY. The plausible wrong implementation is the prior
    //   one -- return before anything exists -- under which the refusal
    //   reaches the next resume as an unmarked missing snapshot and is
    //   resumed past silently; checks 3 and 4 red.
    // Why required: a structural refusal is a defect in the model or the
    //   writer, and the resume must stop on it, which it does only for a
    //   checkpoint carrying ".failed".
    // What kills it: dropping the marker creation, writing it as ".part", or
    //   writing snapshot bytes into it.
    // A second correct implementation that still passes: writing the marker
    //   to a temporary name and renaming it into place.
    const int N = 4;
    const std::string stem = make_snapshot_stem("t33");
    const std::string path = snapshot_path_at(stem, N);
    build_pollutant_model(2, 0);
    std::free(Outfall[0].wRouted);
    Outfall[0].wRouted = nullptr;

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    std::string out;
    { CerrCapture cap; obj.write_state_snapshot(N); out = cap.str(); }
    free_pollutant_model();

    CHECK(!std::filesystem::exists(path), "a refused save must publish nothing");
    CHECK(!std::filesystem::exists(path + ".part"), "and create no .part");
    CHECK(std::filesystem::exists(path + ".failed"),
          "a count-pass refusal must leave {path}.failed");
    CHECK(read_bytes(path + ".failed") == std::string(kRefusedLine) + "\n",
          "holding exactly the refusal line");
    CHECK(out.find("will STOP with an error naming it") != std::string::npos,
          "the warning must say the resume will stop");
    if (read_bytes(path + ".failed") != std::string(kRefusedLine) + "\n")
        std::fprintf(stderr, "  note: marker held '%s'\n", read_bytes(path + ".failed").c_str());

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

static void T34_a_count_pass_failed_marker_stops_the_resume_with_the_stop_line()
{
    // DoD 10(m).
    // Property: a count-pass ".failed" at cp{N} beside a published cp{N-1}
    //   makes the consultation return KilledWrite and print the STOP line
    //   (reason=write-failed) exactly once, under the same assertions as
    //   DoD 10 (g): no step-back line, the stop stated, no earlier snapshot
    //   offered, the marker kept, and the identical line on a second resume.
    // Class: NEW CAPABILITY. A plausible wrong implementation reads the
    //   marker's content and treats a text marker as stale; checks 1-2 red.
    // Why required: the refused checkpoint must stop the restart whichever
    //   writer arm left the marker.
    // What kills it: removing ".failed" from Arm 2's key, or discriminating on
    //   the marker's content.
    // A second correct implementation that still passes: a consultation that
    //   reads the marker's first line to choose the human text, provided it
    //   still returns KilledWrite and prints the same machine line.
    const int N = 6;
    const std::string stem = make_snapshot_stem("t34");
    CHECK(publish_complete_snapshot(stem, N - 1), "the fixture must publish cp{N-1}");
    const std::string failed = snapshot_path_at(stem, N) + ".failed";
    write_bytes(failed, std::string(kRefusedLine) + "\n");

    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    SnapRestore sr;
    std::string out;
    { CerrCapture cap; sr = obj.try_restore_state_snapshot(N, (value_t) 0.0); out = cap.str(); }
    CHECK(sr == SnapRestore::KilledWrite, "the consultation must return KilledWrite");
    CHECK_EQ_I(occurrences(out, row_stop(N, N - 1)), 1,
               "the stop line naming N-1 for information must appear exactly once");
    CHECK(out.find("SWMM snapshot step-back required") == std::string::npos,
          "a refused write must not print a step-back line");
    CHECK(out.find("This resume STOPS") != std::string::npos, "the human text must state the stop");
    CHECK(out.find("the cheap route") == std::string::npos,
          "and must not offer the earlier snapshot as a route");
    CHECK(read_bytes(failed) == std::string(kRefusedLine) + "\n",
          "the marker must survive the resume unchanged");
    const std::string out2 = resume_and_capture(stem, N);
    CHECK_EQ_I(occurrences(out2, row_stop(N, N - 1)), 1,
               "a second resume at N must print the identical stop line");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
}

// ---------------------------------------------------------------------------
// swmm_snapshot_failed_replay: the opt-in replay at a failed-write checkpoint.
//
// The key is latched into a private member by initialize(), which this tier
// cannot call (see the access note above), so the fixture sets the member
// through the third explicit instantiation and calls the consultation
// directly. The text constants are spelled out here rather than taken from
// swmm_triton.h, so a reworded notice or sentence reds these nodes instead of
// moving with them.
// ---------------------------------------------------------------------------
static const char* const kReplayNotice =
    "swmm_snapshot_failed_replay=1 is set, so this resume takes the FULL EXCHANGE "
    "REPLAY from t=0 instead of stopping.";
static const char* const kReplayHint =
    "To get correct results now without diagnosing the cause, relaunch with "
    "swmm_snapshot_failed_replay=1 in the cfg, which takes the full exchange "
    "replay from t=0 instead of stopping.";

// One resume at checkpoint n with the key set as given; returns the captured
// stderr and stores the consultation's return in *sr.
static std::string resume_with_failed_replay(const std::string& stem, int n, bool key,
                                             SnapRestore* sr)
{
    SwmmTriton obj;
    arrange_rank0_with_stem(obj, stem);
    obj.*retrieve(FailedReplayTag{}) = key;
    std::string out;
    { CerrCapture cap; *sr = obj.try_restore_state_snapshot(n, (value_t) 0.0); out = cap.str(); }
    return out;
}

static void T35_failed_replay_on_returns_the_replay_and_keeps_the_marker()
{
    // Property: with swmm_snapshot_failed_replay on, a ".failed" at cp{N} and no
    //   published cp{N} makes the consultation return FailedWriteReplay, print
    //   the replay notice once, print none of the three bracketed stop lines,
    //   and leave the marker on disk byte for byte -- and a second resume at N
    //   does the same. Checked with a published cp{N-1} below and with none,
    //   because the stop lines those two states select differ.
    // Class: NEW CAPABILITY. The plausible wrong implementation ignores the key
    //   and returns KilledWrite as before; the return check and the notice
    //   check red, and the stop-line checks red.
    // Why required: the key's whole effect is this return value -- triton.h
    //   sends only FailedWriteReplay to the replay -- and the marker is the
    //   only record that the write failed, so removing it on this route would
    //   lose that record and make a later resume read an unmarked missing
    //   snapshot.
    // What kills it: ignoring the key (KilledWrite returned regardless), or
    //   removing the ".failed" marker on the override route.
    // A second correct implementation that still passes: making the key test
    //   before the earlier-snapshot search instead of after it, which changes
    //   no output.
    const int N = 6;
    const std::string stems[2] = { make_snapshot_stem("t35a"), make_snapshot_stem("t35b") };
    CHECK(publish_complete_snapshot(stems[0], N - 1), "the fixture must publish cp{N-1}");

    for (const std::string& stem : stems) {
        const std::string failed = snapshot_path_at(stem, N) + ".failed";
        write_bytes(failed, std::string(kRefusedLine) + "\n");

        for (int pass = 1; pass <= 2; ++pass) {
            SnapRestore sr = SnapRestore::Restored;
            const std::string out = resume_with_failed_replay(stem, N, true, &sr);
            CHECK(sr == SnapRestore::FailedWriteReplay,
                  "the key on must make the consultation return FailedWriteReplay");
            CHECK_EQ_I(occurrences(out, kReplayNotice), 1,
                       "the replay notice must appear exactly once");
            CHECK(out.find("[resume-event") == std::string::npos,
                  "none of the bracketed stop lines may be printed on this route");
            CHECK(out.find("This resume STOPS") == std::string::npos,
                  "and the stop text must not be printed");
            CHECK(read_bytes(failed) == std::string(kRefusedLine) + "\n",
                  "the .failed marker must survive the resume unchanged");
            if (sr != SnapRestore::FailedWriteReplay)
                std::fprintf(stderr, "  note: pass %d captured stderr was:\n%s\n", pass, out.c_str());
        }
    }

    std::error_code ec;
    for (const std::string& stem : stems) std::filesystem::remove_all(dir_of(stem), ec);
}

static void T36_failed_replay_off_keeps_the_stop_and_names_the_key()
{
    // Property: with the key off, a ".failed" at cp{N} and no published cp{N}
    //   returns KilledWrite and prints the stop line (a published cp{N-1}
    //   below) or the clean-start line (none below) exactly as before, plus
    //   the one sentence naming the key, exactly once, and not the replay
    //   notice.
    // Class: NEW CAPABILITY for the sentence; REGRESSION for the return and the
    //   machine lines, which T24 and T23 already hold at the pre-change tree.
    //   The plausible wrong implementation reads the key inverted, so an
    //   absent key replays; the return check and the stop-line checks red.
    // Why required: an absent key must keep the stop every run had before the
    //   key existed, and the sentence is the only place the stop tells an
    //   operator the override exists.
    // What kills it: taking the override branch when the key is off, or
    //   dropping the added sentence.
    // A second correct implementation that still passes: printing the sentence
    //   as part of the line before it rather than on its own line, since the
    //   assertion is on the sentence and not on its line break.
    const int N = 6;
    const std::string stemA = make_snapshot_stem("t36a");
    const std::string stemB = make_snapshot_stem("t36b");
    CHECK(publish_complete_snapshot(stemA, N - 1), "the fixture must publish cp{N-1}");
    write_bytes(snapshot_path_at(stemA, N) + ".failed", std::string(kRefusedLine) + "\n");
    write_bytes(snapshot_path_at(stemB, N) + ".failed", std::string(kRefusedLine) + "\n");

    SnapRestore srA = SnapRestore::Restored, srB = SnapRestore::Restored;
    const std::string outA = resume_with_failed_replay(stemA, N, false, &srA);
    const std::string outB = resume_with_failed_replay(stemB, N, false, &srB);
    CHECK(srA == SnapRestore::KilledWrite && srB == SnapRestore::KilledWrite,
          "the key off must leave the consultation returning KilledWrite");
    CHECK_EQ_I(occurrences(outA, row_stop(N, N - 1)), 1,
               "with cp{N-1} published the stop line must appear exactly once");
    CHECK_EQ_I(occurrences(outB, row_clean_start(N, "write-failed")), 1,
               "with nothing published below the clean-start line must appear exactly once");
    CHECK_EQ_I(occurrences(outA, kReplayHint) + occurrences(outB, kReplayHint), 2,
               "the sentence naming the key must appear exactly once in each stop");
    CHECK(outA.find(kReplayNotice) == std::string::npos && outB.find(kReplayNotice) == std::string::npos,
          "and the replay notice must not");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stemA), ec);
    std::filesystem::remove_all(dir_of(stemB), ec);
}

static void T37_failed_replay_on_leaves_a_killed_write_on_the_step_back_route()
{
    // Property: the key on does not reach a cut-off write. A ".killed" alone
    //   at cp{N} beside a published cp{N-1} returns KilledWrite and prints the
    //   step-back line exactly as with the key off; a ".killed" beside a
    //   ".failed" takes the override, because ".failed" decides.
    // Class: NEW CAPABILITY. The plausible wrong implementation tests the key
    //   for every live marker rather than inside the ".failed" arm, so a
    //   killed write replays; the first two checks red.
    // Why required: the override is for a failed or refused save only; a
    //   cut-off write already has the step-back route to an earlier snapshot.
    // What kills it: moving the key test above the marker-precedence branch.
    // A second correct implementation that still passes: testing
    //   `haveFailed && snapshot_failed_replay` as one condition before the
    //   precedence branch, which selects the same arm for every marker set.
    const int N = 6;
    const std::string stem = make_snapshot_stem("t37");
    CHECK(publish_complete_snapshot(stem, N - 1), "the fixture must publish cp{N-1}");
    touch_marker(snapshot_path_at(stem, N) + ".killed");

    SnapRestore sr = SnapRestore::Restored;
    const std::string out = resume_with_failed_replay(stem, N, true, &sr);
    CHECK(sr == SnapRestore::KilledWrite, "a killed write must still return KilledWrite");
    CHECK_EQ_I(occurrences(out, row_step_back(N, N - 1)), 1,
               "and print the step-back line exactly once");
    CHECK(out.find(kReplayNotice) == std::string::npos, "and not the replay notice");

    write_bytes(snapshot_path_at(stem, N) + ".failed", std::string(kRefusedLine) + "\n");
    SnapRestore sr2 = SnapRestore::Restored;
    const std::string out2 = resume_with_failed_replay(stem, N, true, &sr2);
    CHECK(sr2 == SnapRestore::FailedWriteReplay,
          "with a .failed beside the .killed the override must be taken");
    CHECK_EQ_I(occurrences(out2, kReplayNotice), 1, "with the notice printed once");

    std::error_code ec;
    std::filesystem::remove_all(dir_of(stem), ec);
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
    { "T3c_routing_fields_roundtrip_bit_exactly",&T3c_routing_fields_roundtrip_bit_exactly },
    { "T6_manifest_covers_committed_inventory",  &T6_manifest_covers_committed_inventory },
    { "T7_sizeof_guard_boundary_is_documented",  &T7_sizeof_guard_boundary_is_documented },
    { "T8_single_writer_is_structural",          &T8_single_writer_is_structural },
    { "T3d_nperiods_roundtrips_bit_exactly",     &T3d_nperiods_roundtrips_bit_exactly },
    { "T12_resume_refuses_more_periods_than_the_file_holds",
                                                 &T12_resume_refuses_more_periods_than_the_file_holds },
    { "T13_snapshot_load_return_is_graduated",   &T13_snapshot_load_return_is_graduated },
    { "T14_output_end_leaves_the_trailer_last",  &T14_output_end_leaves_the_trailer_last },
    { "T15_payload_length_does_not_vary_with_routeto",
                                                 &T15_payload_length_does_not_vary_with_routeto },
    { "T16_payload_length_tracks_the_pollutant_axis",
                                                 &T16_payload_length_tracks_the_pollutant_axis },
    { "T17_a_refused_save_publishes_nothing",    &T17_a_refused_save_publishes_nothing },
    { "T18_killed_part_beside_a_complete_prior_snapshot_is_refused",
                                                 &T18_killed_part_beside_a_complete_prior_snapshot_is_refused },
    { "T19_the_killed_write_refusal_names_cause_fallback_and_replay_cost",
                                                 &T19_the_killed_write_refusal_names_cause_fallback_and_replay_cost },
    { "T20_the_part_is_renamed_killed_and_the_classifier_returns_an_existing_token",
                                                 &T20_the_part_is_renamed_killed_and_the_classifier_returns_an_existing_token },
    { "T21_a_clean_start_disposes_both_markers", &T21_a_clean_start_disposes_both_markers },
    { "T22_a_killed_write_with_a_published_snapshot_below_emits_the_step_back_line",
                                                 &T22_a_killed_write_with_a_published_snapshot_below_emits_the_step_back_line },
    { "T23_no_published_snapshot_below_emits_the_clean_start_line_with_its_reason",
                                                 &T23_no_published_snapshot_below_emits_the_clean_start_line_with_its_reason },
    { "T24_a_failed_write_with_a_published_snapshot_below_emits_the_stop_line",
                                                 &T24_a_failed_write_with_a_published_snapshot_below_emits_the_stop_line },
    { "T25_the_step_back_line_is_identical_on_a_second_resume",
                                                 &T25_the_step_back_line_is_identical_on_a_second_resume },
    { "T26_the_step_back_confirmation_scan_reads_only_live_markers",
                                                 &T26_the_step_back_confirmation_scan_reads_only_live_markers },
    { "T27_a_clean_start_keeps_a_failed_marker_under_one_directory_wide_k",
                                                 &T27_a_clean_start_keeps_a_failed_marker_under_one_directory_wide_k },
    { "T28_a_clean_start_without_keep_previous_removes_the_earlier_run",
                                                 &T28_a_clean_start_without_keep_previous_removes_the_earlier_run },
    { "T29_a_clean_start_with_keep_previous_keeps_every_artifact",
                                                 &T29_a_clean_start_with_keep_previous_keeps_every_artifact },
    { "T30_an_undisposable_published_snapshot_refuses_the_clean_start",
                                                 &T30_an_undisposable_published_snapshot_refuses_the_clean_start },
    { "T31_an_undisposable_marker_warns_and_the_clean_start_continues",
                                                 &T31_an_undisposable_marker_warns_and_the_clean_start_continues },
    { "T32_a_failed_snapshot_write_keeps_its_marker_and_says_the_resume_will_stop",
                                                 &T32_a_failed_snapshot_write_keeps_its_marker_and_says_the_resume_will_stop },
    { "T33_a_count_pass_refusal_leaves_a_one_line_failed_marker",
                                                 &T33_a_count_pass_refusal_leaves_a_one_line_failed_marker },
    { "T34_a_count_pass_failed_marker_stops_the_resume_with_the_stop_line",
                                                 &T34_a_count_pass_failed_marker_stops_the_resume_with_the_stop_line },
    { "T35_failed_replay_on_returns_the_replay_and_keeps_the_marker",
                                                 &T35_failed_replay_on_returns_the_replay_and_keeps_the_marker },
    { "T36_failed_replay_off_keeps_the_stop_and_names_the_key",
                                                 &T36_failed_replay_off_keeps_the_stop_and_names_the_key },
    { "T37_failed_replay_on_leaves_a_killed_write_on_the_step_back_route",
                                                 &T37_failed_replay_on_leaves_a_killed_write_on_the_step_back_route },
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
