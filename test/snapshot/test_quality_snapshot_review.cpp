/** @file test_quality_snapshot_review.cpp
 *  @brief Reviewer test for the water-quality state in the SWMM snapshot (format V7).
 *
 *  Independent of test_state_snapshot.cpp's T38-T40 by construction: its own
 *  fixture, its own value stamp, a different model shape, and two properties
 *  those nodes do not state.
 *
 *  RV3 -- the quality state survives a save and a load on a model the coder's
 *         T38 does not build: three pollutants (an odd count), one STORAGE node
 *         carrying a non-zero hydraulic residence time (Storage.hrt, which the
 *         treatment path reads), one ROUTED outfall (routeTo >= 0, so wRouted is
 *         allocated) beside one route-less outfall, four nodes and three links.
 *         Every element of Node.oldQual/newQual, Link.oldQual/newQual/totalLoad,
 *         every field of every StepQualTotals row, Storage.hrt, the routed
 *         outfall's wRouted and every OutfallStats.totalLoad round-trip bitwise.
 *
 *  RV3b -- emission does not depend on IgnoreQuality: the same model saved with
 *         IgnoreQuality = 0 and = 1 yields files of identical length, and the
 *         IgnoreQuality = 1 save still round-trips the quality state bitwise.
 *         The shape requires unconditional emission so the payload length is a
 *         function of the shape block alone; a writer that skipped the quality
 *         loops under IGNORE_QUALITY YES would make a file whose length depends
 *         on a field no shape dimension records.
 *
 *  Same include prefix as test_state_snapshot.cpp; output_end is not called, so
 *  that file's C-linkage block for it is not needed here.
 */

#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unistd.h>

extern "C" {
    extern int            Nobjects[];
    extern int            Nnodes[];
    extern int            Nlinks[];
    extern int            IgnoreQuality;
    extern TNode*         Node;
    extern TLink*         Link;
    extern TOutfall*      Outfall;
    extern TStorage*      Storage;
    extern TNodeStats*    NodeStats;
    extern TLinkStats*    LinkStats;
    extern TStorageStats* StorageStats;
    extern TOutfallStats* OutfallStats;
    extern double*        NodeInflow;
    extern double*        NodeOutflow;
    extern TLoadingTotals* LoadingTotals;
    extern TRoutingTotals* QualTotals;
    extern TRoutingTotals* StepQualTotals;
}

static int g_failures = 0;

#define CHECK(expr, msg)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "  FAIL (%s:%d): %s\n", __FILE__, __LINE__, msg);\
        }                                                                         \
    } while (0)

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

static std::string rv_path(const char* stem)
{
    std::ostringstream oss;
    oss << std::filesystem::temp_directory_path().string()
        << "/triton_rv3_" << stem << "_" << (long) getpid() << ".snapshot";
    return oss.str();
}

static const int kP  = 3;   // pollutants (odd, unlike T38's 2)
static const int kNN = 4;   // nodes
static const int kNL = 3;   // links
static const int kNO = 2;   // outfalls: [0] routed, [1] route-less
static const int kNS = 1;   // storage units

static double* dvec(int n) { return (double*) std::calloc((size_t) n, sizeof(double)); }

static void build()
{
    Nobjects[NODE] = kNN; Nobjects[LINK] = kNL; Nobjects[POLLUT] = kP; Nobjects[SUBCATCH] = 0;
    Nnodes[OUTFALL] = kNO; Nnodes[STORAGE] = kNS;
    Nlinks[CONDUIT] = 0;  Nlinks[PUMP] = 0;

    Node        = (TNode*)         std::calloc(kNN, sizeof(TNode));
    Link        = (TLink*)         std::calloc(kNL, sizeof(TLink));
    NodeStats   = (TNodeStats*)    std::calloc(kNN, sizeof(TNodeStats));
    LinkStats   = (TLinkStats*)    std::calloc(kNL, sizeof(TLinkStats));
    NodeInflow  = dvec(kNN);
    NodeOutflow = dvec(kNN);
    Outfall     = (TOutfall*)      std::calloc(kNO, sizeof(TOutfall));
    OutfallStats= (TOutfallStats*) std::calloc(kNO, sizeof(TOutfallStats));
    Storage     = (TStorage*)      std::calloc(kNS, sizeof(TStorage));
    StorageStats= (TStorageStats*) std::calloc(kNS, sizeof(TStorageStats));
    LoadingTotals  = (TLoadingTotals*) std::calloc(kP, sizeof(TLoadingTotals));
    QualTotals     = (TRoutingTotals*) std::calloc(kP, sizeof(TRoutingTotals));
    StepQualTotals = (TRoutingTotals*) std::calloc(kP, sizeof(TRoutingTotals));

    for (int j = 0; j < kNN; j++) { Node[j].oldQual = dvec(kP); Node[j].newQual = dvec(kP); }
    for (int j = 0; j < kNL; j++) { Link[j].oldQual = dvec(kP); Link[j].newQual = dvec(kP); Link[j].totalLoad = dvec(kP); }
    for (int i = 0; i < kNO; i++)
    {
        OutfallStats[i].totalLoad = dvec(kP);
        Outfall[i].routeTo = (i == 0) ? 0 : -1;          // node.c:164 allocates wRouted on routeTo >= 0
        Outfall[i].wRouted = (i == 0) ? dvec(kP) : nullptr;
    }
}

static void teardown()
{
    for (int j = 0; j < kNN; j++) { std::free(Node[j].oldQual); std::free(Node[j].newQual); }
    for (int j = 0; j < kNL; j++) { std::free(Link[j].oldQual); std::free(Link[j].newQual); std::free(Link[j].totalLoad); }
    for (int i = 0; i < kNO; i++) { std::free(OutfallStats[i].totalLoad); std::free(Outfall[i].wRouted); }
    std::free(Node); std::free(Link); std::free(NodeStats); std::free(LinkStats);
    std::free(NodeInflow); std::free(NodeOutflow); std::free(Outfall); std::free(OutfallStats);
    std::free(Storage); std::free(StorageStats);
    std::free(LoadingTotals); std::free(QualTotals); std::free(StepQualTotals);
    Node = nullptr; Link = nullptr; NodeStats = nullptr; LinkStats = nullptr;
    NodeInflow = nullptr; NodeOutflow = nullptr; Outfall = nullptr; OutfallStats = nullptr;
    Storage = nullptr; StorageStats = nullptr;
    LoadingTotals = nullptr; QualTotals = nullptr; StepQualTotals = nullptr;
    Nobjects[NODE] = 0; Nobjects[LINK] = 0; Nobjects[POLLUT] = 0;
    Nnodes[OUTFALL] = 0; Nnodes[STORAGE] = 0;
    IgnoreQuality = 0;
}

// A stamp unlike T38's: powers of two plus a prime-scaled offset, so the two
// suites cannot share a coincidental pass on a mis-strided slot.
static double stamp(int a, int e, int p) { return std::ldexp(1.0, a + 3) + 7.0 * e + 0.001953125 * (p + 1) + 1.0e-9 * (a * 997 + e * 31 + p); }

static double TRoutingTotals::* const kRT[13] = {
    &TRoutingTotals::dwInflow, &TRoutingTotals::wwInflow, &TRoutingTotals::gwInflow,
    &TRoutingTotals::iiInflow, &TRoutingTotals::exInflow, &TRoutingTotals::flooding,
    &TRoutingTotals::outflow,  &TRoutingTotals::evapLoss, &TRoutingTotals::seepLoss,
    &TRoutingTotals::reacted,  &TRoutingTotals::initStorage, &TRoutingTotals::finalStorage,
    &TRoutingTotals::pctError };

static void fill(double clobber, bool use_clobber)
{
    for (int p = 0; p < kP; p++)
    {
        for (int j = 0; j < kNN; j++) {
            Node[j].oldQual[p] = use_clobber ? clobber : stamp(0, j, p);
            Node[j].newQual[p] = use_clobber ? clobber : stamp(1, j, p);
        }
        for (int j = 0; j < kNL; j++) {
            Link[j].oldQual[p]   = use_clobber ? clobber : stamp(2, j, p);
            Link[j].newQual[p]   = use_clobber ? clobber : stamp(3, j, p);
            Link[j].totalLoad[p] = use_clobber ? clobber : stamp(4, j, p);
        }
        for (int f = 0; f < 13; f++) StepQualTotals[p].*kRT[f] = use_clobber ? clobber : stamp(5, f, p);
        for (int i = 0; i < kNO; i++) OutfallStats[i].totalLoad[p] = use_clobber ? clobber : stamp(6, i, p);
        Outfall[0].wRouted[p] = use_clobber ? clobber : stamp(7, 0, p);
    }
    Storage[0].hrt = use_clobber ? clobber : stamp(8, 0, 0);
}

// Returns the number of slots that did not come back bit-exact; reports each class.
static long verify(const char* label)
{
    long bad = 0, nq = 0, nst = 0, nout = 0;
    for (int p = 0; p < kP; p++)
    {
        for (int j = 0; j < kNN; j++) {
            if (!same_bits(Node[j].oldQual[p], stamp(0, j, p))) { ++bad; ++nq; }
            if (!same_bits(Node[j].newQual[p], stamp(1, j, p))) { ++bad; ++nq; }
        }
        for (int j = 0; j < kNL; j++) {
            if (!same_bits(Link[j].oldQual[p],   stamp(2, j, p))) { ++bad; ++nq; }
            if (!same_bits(Link[j].newQual[p],   stamp(3, j, p))) { ++bad; ++nq; }
            if (!same_bits(Link[j].totalLoad[p], stamp(4, j, p))) { ++bad; ++nq; }
        }
        for (int f = 0; f < 13; f++)
            if (!same_bits(StepQualTotals[p].*kRT[f], stamp(5, f, p))) { ++bad; ++nst; }
        for (int i = 0; i < kNO; i++)
            if (!same_bits(OutfallStats[i].totalLoad[p], stamp(6, i, p))) { ++bad; ++nout; }
        if (!same_bits(Outfall[0].wRouted[p], stamp(7, 0, p))) { ++bad; ++nout; }
    }
    const bool hrt_bad = !same_bits(Storage[0].hrt, stamp(8, 0, 0));
    if (hrt_bad) ++bad;
    std::fprintf(stderr, "  %s: element-quality slots wrong=%ld/%d, StepQualTotals wrong=%ld/%d, "
                 "outfall slots wrong=%ld/%d, Storage.hrt %s\n", label,
                 nq, kP * (2 * kNN + 3 * kNL), nst, kP * 13, nout, kP * (kNO + 1),
                 hrt_bad ? "WRONG" : "ok");
    return bad;
}

static long roundtrip(const char* stem, int ignoreQuality, long* fileBytes)
{
    const std::string path = rv_path(stem);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    IgnoreQuality = ignoreQuality;
    fill(0.0, false);
    const int rcSave = snapshot_save(path.c_str());
    CHECK(rcSave == 0, "snapshot_save must succeed on the review model");
    *fileBytes = (long) std::filesystem::file_size(path, ec);
    if (ec) *fileBytes = -1;
    fill(-3.5, true);
    char msg[512]; msg[0] = '\0';
    const int rcLoad = snapshot_load(path.c_str(), msg, (int) sizeof(msg));
    if (rcLoad != SNAPSHOT_OK) std::fprintf(stderr, "  load said: %s\n", msg);
    CHECK(rcLoad == SNAPSHOT_OK, "snapshot_load must succeed on a snapshot this build wrote");
    const long bad = verify(stem);
    std::filesystem::remove(path, ec);
    return bad;
}

static void RV3()
{
    build();
    long bytes = 0;
    const long bad = roundtrip("rv3", 0, &bytes);
    CHECK(bad == 0, "every quality slot, every StepQualTotals field, Storage.hrt and the "
                    "outfall pollutant arrays must round-trip bitwise");
    teardown();
}

static void RV3b()
{
    build();
    long bytesOff = 0, bytesOn = 0;
    const long badOff = roundtrip("rv3b_off", 0, &bytesOff);
    const long badOn  = roundtrip("rv3b_on",  1, &bytesOn);
    std::fprintf(stderr, "  file bytes: IgnoreQuality=0 -> %ld, IgnoreQuality=1 -> %ld\n", bytesOff, bytesOn);
    CHECK(bytesOff > 0 && bytesOn > 0, "both saves must leave a file");
    CHECK(bytesOff == bytesOn, "the payload length must not depend on IgnoreQuality");
    CHECK(badOff == 0 && badOn == 0, "the quality state must round-trip under either IgnoreQuality value");
    teardown();
}

int main(int argc, char** argv)
{
    const char* want = (argc > 1) ? argv[1] : nullptr;
    int ran = 0;
    if (!want || !std::strcmp(want, "RV3"))  { std::fprintf(stderr, "RUN  RV3\n");  RV3();  ++ran; }
    if (!want || !std::strcmp(want, "RV3b")) { std::fprintf(stderr, "RUN  RV3b\n"); RV3b(); ++ran; }
    if (ran == 0) { std::fprintf(stderr, "ERROR: no test named '%s'\n", want); return 2; }
    std::fprintf(stderr, "%d test(s), %d failure(s)\n", ran, g_failures);
    return g_failures == 0 ? 0 : 1;
}
