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
          "the temporary must not survive a refused save");

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

template <typename Tag, typename Tag::type M>
struct Stow { friend typename Tag::type retrieve(Tag) { return M; } };

// The ONLY two sites in this file that name a private member of swmm_triton,
// and both are explicit-instantiation arguments, which is the exempt context.
template struct Stow<RankTag, &SwmmTriton::rank_>;
template struct Stow<StemTag, &SwmmTriton::snapshot_path_stem>;

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
    const std::string stem = make_snapshot_stem("t21");

    const std::string part   = snapshot_path_at(stem, 1) + ".part";
    const std::string killed = snapshot_path_at(stem, 2) + ".killed";
    touch_marker(part);
    touch_marker(killed);
    CHECK(std::filesystem::exists(part) && std::filesystem::exists(killed),
          "the fixture must stand up both marker kinds");

    // A published snapshot that must SURVIVE the sweep. Without it the node
    // would pass for a disposal step that deleted the whole directory.
    CHECK(publish_complete_snapshot(stem, 5),
          "the fixture must publish a snapshot the sweep must not touch");

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
        CHECK(std::filesystem::exists(snapshot_path_at(stem, 5)),
              "and it must leave the PUBLISHED snapshots alone -- the disposal "
              "is scoped to markers, not to the snapshot directory");
    }

    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::path(stem).parent_path(), ec);
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
