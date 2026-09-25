//-----------------------------------------------------------------------------
//   snapshot.c
//
//   Project:  TRITON-SWMM coupling (vendored EPA SWMM tree)
//   Purpose:  Full-precision capture and restore of SWMM's routing state and
//             its statistics and mass-balance accumulators, so a coupled
//             hotstart resume can reconstruct SWMM at t_k directly instead of
//             replaying its entire exchange history from t=0.
//
//   THIS FILE IS NOT EPA SOURCE. It is TRITON-owned and lives in the vendored
//   tree only because the accumulators it captures are `static` in stats.c and
//   cannot be reached by `extern` from outside it.
//
//   WHY THIS EXISTS
//   ---------------
//   On resume the coupled run replayed every recorded exchange record from t=0,
//   one swmm_step per record, BEFORE any GPU array existed -- so the device sat
//   idle for the whole replay. Two production members exceeded a measured
//   11,107 s replay floor and were cancelled still replaying, on a cluster that
//   kills GPU jobs idle at 0% utilisation. Each resume replays a longer prefix,
//   so the sequence does not converge: it is a class of member that cannot
//   complete. This snapshot makes the restore cost independent of t_k.
//
//   WHY NOT THE SHIPPED HOTSTART FILE
//   ---------------------------------
//   hotstart.c declares `float x[3]` and casts node newDepth/newLatFlow,
//   storage hrt, link newFlow/newDepth/setting and every newQual through it.
//   Every cast source is genuinely `double`, so the round trip loses roughly 29
//   bits of mantissa at the resume boundary. Bit-for-bit reproducibility is a
//   hard requirement here, so a candidate that changes results fails rather
//   than being offered with a caveat. EVERY quantity this file stores is
//   `double`, including the integer counters, which is why the format carries
//   exactly one value width.
//
//   WHY THE ACCUMULATORS AND NOT JUST THE ROUTING STATE
//   ---------------------------------------------------
//   The duration counters -- timeFlooded, timeSurcharged, timeCourantCritical,
//   timeNormalFlow, timeInletControl, timeCapacityLimited, timeInFlowClass[] --
//   plus nonConvergedCount, flowTurns and the mass-balance loss terms are NOT
//   RECONSTRUCTIBLE from the reported .out variables. They are tallied at
//   routing-step resolution against per-step predicates the report never
//   carries; reconstruction would be quantized at REPORT_STEP = 120 s against a
//   dt of ~0.028 s. Dropping them yields a run whose report renders cleanly and
//   whose duration counters are silently low -- and resumed members whose SWMM
//   summaries cover a different window than unresumed members' is the
//   worst-blast-radius failure available here, because it is invisible in every
//   artifact a reader would check.
//
//   FORMAT
//   ------
//   All values are little-endian IEEE-754 binary64. Layout:
//
//     magic        int32   0x534E5331 ("SNS1")
//     version      int32
//     value_width  int32   sizeof(double)
//     shape[]      int32   the model shape this snapshot describes
//     manifest_len int32   bytes of manifest text that follow
//     manifest     char[]  newline-separated "Object.field" names, in the exact
//                          order the payload writes them
//     payload      double[]
//
//   The manifest is IN the file rather than beside it, so a snapshot cannot be
//   separated from the description of what it contains.
//
//   WRITES ARE FIELD-BY-FIELD, NEVER WHOLE-STRUCT
//   ---------------------------------------------
//   Not for portability of a single run -- one build writes and reads it -- but
//   so the file is readable across compilers, and so a field REMOVED or RENAMED
//   upstream fails the build here rather than silently shifting the payload.
//   A whole-struct fwrite would bake this platform's padding into the file and
//   would keep compiling after an upstream rename.
//-----------------------------------------------------------------------------

#define _CRT_SECURE_NO_DEPRECATE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "headers.h"

//-----------------------------------------------------------------------------
//  Imported variables
//-----------------------------------------------------------------------------
extern TSubcatchStats* SubcatchStats;          // defined in STATS.C
extern TNodeStats*     NodeStats;
extern TLinkStats*     LinkStats;
extern TStorageStats*  StorageStats;
extern TOutfallStats*  OutfallStats;
extern TPumpStats*     PumpStats;
extern double          MaxOutfallFlow;
extern double          MaxRunoffFlow;
extern double          RoutingTimeSpan;

extern TRunoffTotals   RunoffTotals;           // defined in MASSBAL.C
extern TLoadingTotals* LoadingTotals;
extern TGwaterTotals   GwaterTotals;
extern TRoutingTotals  FlowTotals;
extern TRoutingTotals* QualTotals;
extern TRoutingTotals  StepFlowTotals;
extern TRoutingTotals  OldStepFlowTotals;
extern double*         NodeInflow;
extern double*         NodeOutflow;
extern double          TotalArea;

//-----------------------------------------------------------------------------
//  Constants
//-----------------------------------------------------------------------------
#define SNAPSHOT_MAGIC    0x534E5331   // "SNS1"

// VERSION 2 adds the Criterion-P routing-state fields to the traversal, and two
// dimensions (nConduit, nXnode) to the shape block.
//
// A V1 FILE IS REFUSED, NOT MIS-STRIDED, AND THE REFUSAL IS THE EXACT-EQUALITY
// TEST IN snapshot_load, NOT AN INFERENCE FROM THE BUMP.  The check is
// `hdr[1] != SNAPSHOT_VERSION`, it runs BEFORE the shape block is read, and it
// names both versions in the refusal.  Two properties make that sound: the test
// is equality rather than `<`, so a V1 file cannot be accepted as an old-but-
// compatible one; and it precedes the shape read, so the widened shape block is
// never interpreted against a V1 file's narrower one.  The magic is deliberately
// NOT bumped -- the format family is the same, and a version mismatch produces a
// message naming the two versions, where a magic mismatch would only say the
// file is foreign.
#define SNAPSHOT_VERSION  2

// Must match stats.c's private MAX_STATS. stats.c hands us its value at
// runtime through stats_getSnapshotRefs(); this is only the compile-time
// bound on the local buffer, and a disagreement is caught there.
#define SNAPSHOT_MAX_STATS 5

//-----------------------------------------------------------------------------
//  THE sizeof GUARD
//-----------------------------------------------------------------------------
//  What it catches, stated as a boundary rather than as a claim of coverage:
//
//    REMOVAL  - caught, and caught earlier by the field-by-field writes, which
//               name the field and stop compiling without it.
//    RENAME   - caught by the field-by-field writes, for the same reason. The
//               sizeof is unchanged by a rename, so this line is not what
//               catches it.
//    REORDER  - a NON-ISSUE by construction. The payload order is this file's
//               write order, not the struct's declaration order, so reordering
//               a struct cannot shift the payload. That is the third thing
//               field-by-field writing buys.
//    ADDITION - caught ONLY when it grows the struct.
//
//  IT IS BLIND TO AN ADDITION THAT LANDS IN EXISTING PADDING, and that is not
//  hypothetical. Measured with a compiled probe on this platform (x86-64
//  System V LP64, gcc 11.4.0):
//
//    TNodeStats      128  nonConvergedCount is an int at offset 104, so bytes
//                         108-111 are a hole an added int occupies for free
//    TOutfallStats    32  totalPeriods is an int at 24; 28-31 trailing hole
//    TLinkStats      176  flowTurnSign is an int at 168; 172-175 trailing hole
//    TTimeStepStats  120  timeStepCount is an int at 24; 28-31 hole
//    TStorageStats    56  dense
//    TMaxStats        16  dense
//    TPumpStats       72  dense
//    TSubcatchStats   64  dense
//
//  So an added `int` is invisible to this guard on four of the eight structs
//  and visible on the other four. THE ADDITION CASE IS CLOSED BY THE INVENTORY
//  REGENERATION CHECK, not by these lines -- that check is padding-independent
//  and has two declared firing points (a slow-tier CTest node and a named step
//  in the upstream-bump procedure). These asserts are the cheap build-time half
//  of a two-part guard, and neither half is sufficient alone.
//-----------------------------------------------------------------------------
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  #define SNAPSHOT_SIZE_GUARD(T, n) \
      _Static_assert(sizeof(T) == (n), \
          #T " changed size: the vendored EPA struct no longer matches what " \
             "snapshot.c serializes. Re-run " \
             "test/inventory/regenerate_snapshot_inventory.py and update the " \
             "serializer and deserializer together.")
#else
  #define SNAPSHOT_SIZE_GUARD(T, n) \
      typedef char snapshot_size_guard_##T[(sizeof(T) == (n)) ? 1 : -1]
#endif

SNAPSHOT_SIZE_GUARD(TNodeStats,     128);
SNAPSHOT_SIZE_GUARD(TLinkStats,     176);
SNAPSHOT_SIZE_GUARD(TStorageStats,   56);
SNAPSHOT_SIZE_GUARD(TOutfallStats,   32);
SNAPSHOT_SIZE_GUARD(TPumpStats,      72);
SNAPSHOT_SIZE_GUARD(TSubcatchStats,  64);
SNAPSHOT_SIZE_GUARD(TTimeStepStats, 120);
SNAPSHOT_SIZE_GUARD(TMaxStats,       16);

//-----------------------------------------------------------------------------
//  Shape: the model dimensions a snapshot describes.
//
//  A snapshot that does not match the running model is REFUSED, loudly and with
//  the mismatching dimension named. Silently restoring a mismatched snapshot
//  would read the payload at the wrong stride and produce a state that is
//  wrong everywhere without being wrong anywhere a reader would look.
//-----------------------------------------------------------------------------
typedef struct
{
    int nNodes, nLinks, nSubcatch, nPollut;
    int nStorage, nOutfall, nPump;
    int maxStats, timeLevels, flowClasses;
    // --- added at SNAPSHOT_VERSION 2, with the Criterion-P routing fields.
    //
    //     nConduit is Nlinks[CONDUIT]; the Conduit[] array is indexed
    //     independently of Link[], so nLinks does not bound it.
    //
    //     nXnode is dynwave.c's extended nodal array length, which is
    //     Nobjects[NODE] under dynamic-wave routing and 0 otherwise. Recording
    //     it is what keeps a routing-model change from being silent: a snapshot
    //     taken under DYNWAVE carries nXnode > 0 and is REFUSED by the shape
    //     comparison if reloaded into a run with no Xnode, instead of having its
    //     Xnode payload read at the wrong offset. The alternative -- always
    //     emitting Nobjects[NODE] Xnode rows and zero-filling when the array is
    //     absent -- keeps the payload length constant and is strictly worse,
    //     because it restores zeros into a live Xnode without any refusal.
    int nConduit, nXnode;
} TSnapshotShape;

#define SNAPSHOT_SHAPE_FIELDS 12

static void snapshot_currentShape(TSnapshotShape* s, int maxStats)
{
    s->nNodes      = Nobjects[NODE];
    s->nLinks      = Nobjects[LINK];
    s->nSubcatch   = Nobjects[SUBCATCH];
    s->nPollut     = Nobjects[POLLUT];
    s->nStorage    = Nnodes[STORAGE];
    s->nOutfall    = Nnodes[OUTFALL];
    s->nPump       = Nlinks[PUMP];
    s->nConduit    = Nlinks[CONDUIT];
    s->nXnode      = dynwave_getSnapshotCount();
    s->maxStats    = maxStats;
    s->timeLevels  = TIMELEVELS;
    s->flowClasses = MAX_FLOW_CLASSES;
}

//-----------------------------------------------------------------------------
//  Emitter state.
//
//  One traversal drives three consumers -- the manifest, the write payload and
//  the read payload -- so the three can never fall out of order with each
//  other. Keeping them in separate traversals is how a serializer and its
//  deserializer come to disagree about field order while both compile.
//-----------------------------------------------------------------------------
typedef enum { SNAP_MANIFEST, SNAP_WRITE, SNAP_READ } TSnapMode;

typedef struct
{
    TSnapMode mode;
    FILE*     f;
    char*     manifest;      // SNAP_MANIFEST only
    size_t    manifestLen;
    size_t    manifestCap;
    long      count;         // values visited
    int       error;
} TSnapCtx;

static void snap_name(TSnapCtx* c, const char* obj, const char* field)
{
    size_t need;
    char   line[128];

    if ( c->mode != SNAP_MANIFEST ) return;
    snprintf(line, sizeof(line), "%s.%s\n", obj, field);
    need = strlen(line);
    if ( c->manifestLen + need + 1 > c->manifestCap )
    {
        size_t cap = (c->manifestCap ? c->manifestCap * 2 : 4096);
        char*  p;
        while ( cap < c->manifestLen + need + 1 ) cap *= 2;
        p = (char*) realloc(c->manifest, cap);
        if ( !p ) { c->error = 1; return; }
        c->manifest    = p;
        c->manifestCap = cap;
    }
    memcpy(c->manifest + c->manifestLen, line, need);
    c->manifestLen += need;
    c->manifest[c->manifestLen] = '\0';
}

// Visit one double-valued slot. On SNAP_WRITE it is written; on SNAP_READ it is
// overwritten from the file; on SNAP_MANIFEST nothing is touched.
static void snap_d(TSnapCtx* c, double* v)
{
    if ( c->error ) return;
    c->count++;
    if ( c->mode == SNAP_WRITE )
    {
        if ( fwrite(v, sizeof(double), 1, c->f) != 1 ) c->error = 1;
    }
    else if ( c->mode == SNAP_READ )
    {
        if ( fread(v, sizeof(double), 1, c->f) != 1 ) c->error = 1;
    }
}

// Integer-valued slots are STORED AS double so the file carries exactly one
// value width. Every integer this serializes is far inside binary64's exact
// integer range (2^53), so the round trip is lossless -- it is a change of
// container, not of precision.
static void snap_i(TSnapCtx* c, int* v)
{
    double d;
    if ( c->error ) return;
    c->count++;
    if ( c->mode == SNAP_WRITE )
    {
        d = (double) *v;
        if ( fwrite(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
    }
    else if ( c->mode == SNAP_READ )
    {
        if ( fread(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
        else *v = (int) d;
    }
}

static void snap_l(TSnapCtx* c, long* v)
{
    double d;
    if ( c->error ) return;
    c->count++;
    if ( c->mode == SNAP_WRITE )
    {
        d = (double) *v;
        if ( fwrite(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
    }
    else if ( c->mode == SNAP_READ )
    {
        if ( fread(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
        else *v = (long) d;
    }
}

// char-valued slots (SWMM's boolean flags) go through the same double container
// for the same reason the ints do: the file carries exactly one value width.
static void snap_c(TSnapCtx* c, char* v)
{
    double d;
    if ( c->error ) return;
    c->count++;
    if ( c->mode == SNAP_WRITE )
    {
        d = (double) *v;
        if ( fwrite(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
    }
    else if ( c->mode == SNAP_READ )
    {
        if ( fread(&d, sizeof(double), 1, c->f) != 1 ) c->error = 1;
        else *v = (char) d;
    }
}

#define SNAP_D(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_d(ctx, &((s).f)); } while (0)
#define SNAP_I(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_i(ctx, &((s).f)); } while (0)
#define SNAP_L(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_l(ctx, &((s).f)); } while (0)
#define SNAP_C(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_c(ctx, &((s).f)); } while (0)

//-----------------------------------------------------------------------------
//  The traversal.
//
//  Field order here IS the payload order and IS the manifest order. The
//  inventory at test/inventory/snapshot_inventory.txt is the committed record
//  of what this traversal must cover, and the T6 test asserts the emitted
//  manifest against it.
//-----------------------------------------------------------------------------
// MANIFEST mode must not touch model memory: the manifest describes what this
// BUILD serializes and has to be answerable before a model is open, and with
// every count at zero.  Without these dummies the manifest of an empty model is
// empty -- which would make the file's own description of its layout depend on
// the model it was taken from, and would make the manifest untestable without a
// live solver.  They are written to in MANIFEST mode only in the sense that
// nothing writes them at all: snap_* touch the slot only in WRITE and READ.
static TNodeStats     _snapDummyNode;
static TLinkStats     _snapDummyLink;
static TStorageStats  _snapDummyStorage;
static TOutfallStats  _snapDummyOutfall;
static TPumpStats     _snapDummyPump;
static TSubcatchStats _snapDummySubcatch;
static TTimeStepStats _snapDummyTimeStep;
static TMaxStats      _snapDummyMax;
static TRunoffTotals  _snapDummyRunoff;
static TGwaterTotals  _snapDummyGwater;
static TRoutingTotals _snapDummyRouting;
static TLoadingTotals _snapDummyLoading;
static double         _snapDummyScalar;
static long           _snapDummyLong;
// Criterion-P routing objects. Named ...Obj because _snapDummyNode,
// _snapDummyLink, _snapDummyStorage and _snapDummyOutfall above are the *Stats*
// structs, which are DIFFERENT TYPES under the same names. Dropping the suffix
// on any of the four is a redefinition, not a shadow.
static TNode          _snapDummyNodeObj;
static TLink          _snapDummyLinkObj;
static TConduit       _snapDummyConduit;
static TOutfall       _snapDummyOutfallObj;
static TStorage       _snapDummyStorageObj;

#define SNAP_N(c, real)  ((c)->mode == SNAP_MANIFEST ? 1 : (real))
#define SNAP_P(c, arr, i, dummy)  ((c)->mode == SNAP_MANIFEST ? &(dummy) : &((arr)[i]))

static void snapshot_traverse(TSnapCtx* c, int maxStats)
{
    int i, k;
    TTimeStepStats* tss    = NULL;
    TMaxStats*      mbe    = NULL;
    TMaxStats*      cc     = NULL;
    TMaxStats*      ft     = NULL;
    TMaxStats*      nc     = NULL;
    double*         sysOut = NULL;
    int             nStats = 0;

    if ( c->mode != SNAP_MANIFEST )
    {
        stats_getSnapshotRefs(&tss, &mbe, &cc, &ft, &nc, &sysOut, &nStats);
        // stats.c owns MAX_STATS as a private #define. If its value and ours
        // ever diverge, refuse rather than index an array whose length we
        // guessed -- the guess would be wrong by exactly the amount that makes
        // the overrun hard to see.
        if ( nStats != maxStats ) { c->error = 1; return; }
    }

    // --- NodeStats
    if ( c->mode != SNAP_MANIFEST && Nobjects[NODE] > 0 && !NodeStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[NODE]); i++)
    {
        TNodeStats* s = SNAP_P(c, NodeStats, i, _snapDummyNode);
        SNAP_D(c, "NodeStats", *s, avgDepth);
        SNAP_D(c, "NodeStats", *s, maxDepth);
        SNAP_D(c, "NodeStats", *s, maxDepthDate);
        SNAP_D(c, "NodeStats", *s, maxRptDepth);
        SNAP_D(c, "NodeStats", *s, volFlooded);
        SNAP_D(c, "NodeStats", *s, timeFlooded);
        SNAP_D(c, "NodeStats", *s, timeSurcharged);
        SNAP_D(c, "NodeStats", *s, timeCourantCritical);
        SNAP_D(c, "NodeStats", *s, totLatFlow);
        SNAP_D(c, "NodeStats", *s, maxLatFlow);
        SNAP_D(c, "NodeStats", *s, maxInflow);
        SNAP_D(c, "NodeStats", *s, maxOverflow);
        SNAP_D(c, "NodeStats", *s, maxPondedVol);
        SNAP_I(c, "NodeStats", *s, nonConvergedCount);
        SNAP_D(c, "NodeStats", *s, maxInflowDate);
        SNAP_D(c, "NodeStats", *s, maxOverflowDate);
    }

    // --- LinkStats
    if ( c->mode != SNAP_MANIFEST && Nobjects[LINK] > 0 && !LinkStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[LINK]); i++)
    {
        TLinkStats* s = SNAP_P(c, LinkStats, i, _snapDummyLink);
        SNAP_D(c, "LinkStats", *s, maxFlow);
        SNAP_D(c, "LinkStats", *s, maxFlowDate);
        SNAP_D(c, "LinkStats", *s, maxVeloc);
        SNAP_D(c, "LinkStats", *s, maxDepth);
        SNAP_D(c, "LinkStats", *s, maxStreetFilled);
        SNAP_D(c, "LinkStats", *s, timeNormalFlow);
        SNAP_D(c, "LinkStats", *s, timeInletControl);
        SNAP_D(c, "LinkStats", *s, timeSurcharged);
        SNAP_D(c, "LinkStats", *s, timeFullUpstream);
        SNAP_D(c, "LinkStats", *s, timeFullDnstream);
        SNAP_D(c, "LinkStats", *s, timeFullFlow);
        SNAP_D(c, "LinkStats", *s, timeCapacityLimited);
        for (k = 0; k < MAX_FLOW_CLASSES; k++)
        {
            snap_name(c, "LinkStats", "timeInFlowClass");
            snap_d(c, &s->timeInFlowClass[k]);
        }
        SNAP_D(c, "LinkStats", *s, timeCourantCritical);
        SNAP_L(c, "LinkStats", *s, flowTurns);
        SNAP_I(c, "LinkStats", *s, flowTurnSign);
    }

    // --- StorageStats
    if ( c->mode != SNAP_MANIFEST && Nnodes[STORAGE] > 0 && !StorageStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nnodes[STORAGE]); i++)
    {
        TStorageStats* s = SNAP_P(c, StorageStats, i, _snapDummyStorage);
        SNAP_D(c, "StorageStats", *s, initVol);
        SNAP_D(c, "StorageStats", *s, avgVol);
        SNAP_D(c, "StorageStats", *s, maxVol);
        SNAP_D(c, "StorageStats", *s, maxFlow);
        SNAP_D(c, "StorageStats", *s, evapLosses);
        SNAP_D(c, "StorageStats", *s, exfilLosses);
        SNAP_D(c, "StorageStats", *s, maxVolDate);
    }

    // --- OutfallStats. totalLoad is a per-pollutant array hanging off the
    //     struct; the POINTER is not serialized, its contents are.
    if ( c->mode != SNAP_MANIFEST && Nnodes[OUTFALL] > 0 && !OutfallStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nnodes[OUTFALL]); i++)
    {
        TOutfallStats* s = SNAP_P(c, OutfallStats, i, _snapDummyOutfall);
        SNAP_D(c, "OutfallStats", *s, avgFlow);
        SNAP_D(c, "OutfallStats", *s, maxFlow);
        SNAP_I(c, "OutfallStats", *s, totalPeriods);
        for (k = 0; k < SNAP_N(c, Nobjects[POLLUT]); k++)
        {
            snap_name(c, "OutfallStats", "totalLoad");
            if ( c->mode == SNAP_MANIFEST ) continue;
            if ( !s->totalLoad ) { c->error = 1; return; }
            snap_d(c, &s->totalLoad[k]);
        }
    }

    // --- PumpStats
    if ( c->mode != SNAP_MANIFEST && Nlinks[PUMP] > 0 && !PumpStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nlinks[PUMP]); i++)
    {
        TPumpStats* s = SNAP_P(c, PumpStats, i, _snapDummyPump);
        SNAP_D(c, "PumpStats", *s, utilized);
        SNAP_D(c, "PumpStats", *s, minFlow);
        SNAP_D(c, "PumpStats", *s, avgFlow);
        SNAP_D(c, "PumpStats", *s, maxFlow);
        SNAP_D(c, "PumpStats", *s, volume);
        SNAP_D(c, "PumpStats", *s, energy);
        SNAP_D(c, "PumpStats", *s, offCurveLow);
        SNAP_D(c, "PumpStats", *s, offCurveHigh);
        SNAP_I(c, "PumpStats", *s, startUps);
        SNAP_I(c, "PumpStats", *s, totalPeriods);
    }

    // --- SubcatchStats
    if ( c->mode != SNAP_MANIFEST && Nobjects[SUBCATCH] > 0 && !SubcatchStats ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[SUBCATCH]); i++)
    {
        TSubcatchStats* s = SNAP_P(c, SubcatchStats, i, _snapDummySubcatch);
        SNAP_D(c, "SubcatchStats", *s, precip);
        SNAP_D(c, "SubcatchStats", *s, runon);
        SNAP_D(c, "SubcatchStats", *s, evap);
        SNAP_D(c, "SubcatchStats", *s, infil);
        SNAP_D(c, "SubcatchStats", *s, runoff);
        SNAP_D(c, "SubcatchStats", *s, maxFlow);
        SNAP_D(c, "SubcatchStats", *s, impervRunoff);
        SNAP_D(c, "SubcatchStats", *s, pervRunoff);
    }

    // --- TimeStepStats: a single instance, private to stats.c
    {
        TTimeStepStats* s = (c->mode == SNAP_MANIFEST) ? &_snapDummyTimeStep : tss;
        if ( !s ) { c->error = 1; return; }
        SNAP_D(c, "TimeStepStats", *s, minTimeStep);
        SNAP_D(c, "TimeStepStats", *s, maxTimeStep);
        SNAP_D(c, "TimeStepStats", *s, routingTime);
        SNAP_I(c, "TimeStepStats", *s, timeStepCount);
        SNAP_D(c, "TimeStepStats", *s, trialsCount);
        SNAP_D(c, "TimeStepStats", *s, steadyStateTime);
        for (k = 0; k < TIMELEVELS; k++)
        {
            snap_name(c, "TimeStepStats", "timeStepIntervals");
            snap_d(c, &s->timeStepIntervals[k]);
        }
        for (k = 0; k < TIMELEVELS; k++)
        {
            snap_name(c, "TimeStepStats", "timeStepCounts");
            snap_i(c, &s->timeStepCounts[k]);
        }
    }

    // --- the four critical-statistics arrays.
    //
    //     stats_findMaxStats() RECOMPUTES all four at report time from
    //     NodeStats / LinkStats / NodeInflow / NodeOutflow / timeStepCount /
    //     ReportStepCount, so restoring them is redundant on the happy path.
    //     They are serialized anyway: the criterion admits them (the report
    //     writers read them), and a redundant restore that a later recompute
    //     overwrites costs 60 doubles and cannot be wrong -- whereas omitting
    //     them would make this file's contents depend on a derivation holding
    //     somewhere else.
    {
        TMaxStats*  arrays[4];
        const char* names[4] = { "MaxMassBalErrs", "MaxCourantCrit",
                                 "MaxFlowTurns",   "MaxNonConverged" };
        int a;
        arrays[0] = mbe; arrays[1] = cc; arrays[2] = ft; arrays[3] = nc;
        for (a = 0; a < 4; a++)
        {
            for (i = 0; i < SNAP_N(c, maxStats); i++)
            {
                TMaxStats* s = (c->mode == SNAP_MANIFEST)
                             ? &_snapDummyMax : &arrays[a][i];
                if ( c->mode != SNAP_MANIFEST && !arrays[a] ) { c->error = 1; return; }
                SNAP_I(c, names[a], *s, objType);
                SNAP_I(c, names[a], *s, index);
                SNAP_D(c, names[a], *s, value);
            }
        }
    }

    // --- mass-balance accumulators
    {
        TRunoffTotals* s = (c->mode == SNAP_MANIFEST) ? &_snapDummyRunoff : &RunoffTotals;
        SNAP_D(c, "RunoffTotals", *s, rainfall);
        SNAP_D(c, "RunoffTotals", *s, evap);
        SNAP_D(c, "RunoffTotals", *s, infil);
        SNAP_D(c, "RunoffTotals", *s, runoff);
        SNAP_D(c, "RunoffTotals", *s, drains);
        SNAP_D(c, "RunoffTotals", *s, runon);
        SNAP_D(c, "RunoffTotals", *s, initStorage);
        SNAP_D(c, "RunoffTotals", *s, finalStorage);
        SNAP_D(c, "RunoffTotals", *s, initSnowCover);
        SNAP_D(c, "RunoffTotals", *s, finalSnowCover);
        SNAP_D(c, "RunoffTotals", *s, snowRemoved);
        SNAP_D(c, "RunoffTotals", *s, pctError);
    }
    {
        TGwaterTotals* s = (c->mode == SNAP_MANIFEST) ? &_snapDummyGwater : &GwaterTotals;
        SNAP_D(c, "GwaterTotals", *s, infil);
        SNAP_D(c, "GwaterTotals", *s, upperEvap);
        SNAP_D(c, "GwaterTotals", *s, lowerEvap);
        SNAP_D(c, "GwaterTotals", *s, lowerPerc);
        SNAP_D(c, "GwaterTotals", *s, gwater);
        SNAP_D(c, "GwaterTotals", *s, initStorage);
        SNAP_D(c, "GwaterTotals", *s, finalStorage);
        SNAP_D(c, "GwaterTotals", *s, pctError);
    }
    {
        TRoutingTotals* rt[3];
        const char* rtn[3] = { "FlowTotals", "StepFlowTotals", "OldStepFlowTotals" };
        int a;
        rt[0] = &FlowTotals; rt[1] = &StepFlowTotals; rt[2] = &OldStepFlowTotals;
        for (a = 0; a < 3; a++)
        {
            TRoutingTotals* s = (c->mode == SNAP_MANIFEST) ? &_snapDummyRouting : rt[a];
            SNAP_D(c, rtn[a], *s, dwInflow);
            SNAP_D(c, rtn[a], *s, wwInflow);
            SNAP_D(c, rtn[a], *s, gwInflow);
            SNAP_D(c, rtn[a], *s, iiInflow);
            SNAP_D(c, rtn[a], *s, exInflow);
            SNAP_D(c, rtn[a], *s, flooding);
            SNAP_D(c, rtn[a], *s, outflow);
            SNAP_D(c, rtn[a], *s, evapLoss);
            SNAP_D(c, rtn[a], *s, seepLoss);
            SNAP_D(c, rtn[a], *s, reacted);
            SNAP_D(c, rtn[a], *s, initStorage);
            SNAP_D(c, rtn[a], *s, finalStorage);
            SNAP_D(c, rtn[a], *s, pctError);
        }
    }

    // --- per-pollutant totals
    if ( c->mode != SNAP_MANIFEST && Nobjects[POLLUT] > 0 && !LoadingTotals ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[POLLUT]); i++)
    {
        TLoadingTotals* s = SNAP_P(c, LoadingTotals, i, _snapDummyLoading);
        SNAP_D(c, "LoadingTotals", *s, initLoad);
        SNAP_D(c, "LoadingTotals", *s, buildup);
        SNAP_D(c, "LoadingTotals", *s, deposition);
        SNAP_D(c, "LoadingTotals", *s, sweeping);
        SNAP_D(c, "LoadingTotals", *s, bmpRemoval);
        SNAP_D(c, "LoadingTotals", *s, infil);
        SNAP_D(c, "LoadingTotals", *s, runoff);
        SNAP_D(c, "LoadingTotals", *s, finalLoad);
        SNAP_D(c, "LoadingTotals", *s, pctError);
    }
    if ( c->mode != SNAP_MANIFEST && Nobjects[POLLUT] > 0 && !QualTotals ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[POLLUT]); i++)
    {
        TRoutingTotals* s = SNAP_P(c, QualTotals, i, _snapDummyRouting);
        SNAP_D(c, "QualTotals", *s, dwInflow);
        SNAP_D(c, "QualTotals", *s, wwInflow);
        SNAP_D(c, "QualTotals", *s, gwInflow);
        SNAP_D(c, "QualTotals", *s, iiInflow);
        SNAP_D(c, "QualTotals", *s, exInflow);
        SNAP_D(c, "QualTotals", *s, flooding);
        SNAP_D(c, "QualTotals", *s, outflow);
        SNAP_D(c, "QualTotals", *s, evapLoss);
        SNAP_D(c, "QualTotals", *s, seepLoss);
        SNAP_D(c, "QualTotals", *s, reacted);
        SNAP_D(c, "QualTotals", *s, initStorage);
        SNAP_D(c, "QualTotals", *s, finalStorage);
        SNAP_D(c, "QualTotals", *s, pctError);
    }

    // --- per-node mass-balance volumes. stats_findMaxStats divides by these,
    //     so a resume that leaves them at the post-resume-only accumulation
    //     ranks mass-balance errors against the wrong denominator.
    if ( c->mode != SNAP_MANIFEST && Nobjects[NODE] > 0 && (!NodeInflow || !NodeOutflow) )
    { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[NODE]); i++)
    {
        snap_name(c, "NodeInflow", "value");
        snap_d(c, (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &NodeInflow[i]);
    }
    for (i = 0; i < SNAP_N(c, Nobjects[NODE]); i++)
    {
        snap_name(c, "NodeOutflow", "value");
        snap_d(c, (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &NodeOutflow[i]);
    }

    // --- scalar accumulators.
    //
    //     ReportStepCount is a globals.h global rather than a stats.c static,
    //     and it is the divisor stats_findMaxStats uses for the flow-turns
    //     percentage. NonConvergeCount is the same shape: the time-step
    //     frequency table divides by it at report.c:1076. Both are easy to miss
    //     for the same reason -- they are in no stats struct and among no
    //     file's statics, so an inventory seeded on either would not reach
    //     them. The inventory script's discovery pass is what found the second.
    {
        double* dd; long* ll;
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &MaxOutfallFlow;
        snap_name(c, "MaxOutfallFlow", "value");  snap_d(c, dd);
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &MaxRunoffFlow;
        snap_name(c, "MaxRunoffFlow", "value");   snap_d(c, dd);
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &RoutingTimeSpan;
        snap_name(c, "RoutingTimeSpan", "value"); snap_d(c, dd);
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &TotalArea;
        snap_name(c, "TotalArea", "value");       snap_d(c, dd);
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : sysOut;
        if ( !dd ) { c->error = 1; return; }
        snap_name(c, "SysOutfallFlow", "value");  snap_d(c, dd);
        ll = (c->mode == SNAP_MANIFEST) ? &_snapDummyLong : &ReportStepCount;
        snap_name(c, "ReportStepCount", "value");  snap_l(c, ll);
        ll = (c->mode == SNAP_MANIFEST) ? &_snapDummyLong : &NonConvergeCount;
        snap_name(c, "NonConvergeCount", "value"); snap_l(c, ll);
        ll = (c->mode == SNAP_MANIFEST) ? &_snapDummyLong : &TotalStepCount;
        snap_name(c, "TotalStepCount", "value");   snap_l(c, ll);
    }

    // =====================================================================
    // CRITERION P -- routing state LIVE ON ENTRY at swmm_step's boundary.
    //
    //  These enter through the SAME traversal as the report state above, so the
    //  manifest, the writer and the reader follow automatically -- that is what
    //  the one-traversal design buys and why adding them is not a three-site
    //  edit.
    //
    //  THE FIELD SET IS THE ADMITTED COLUMN OF THE COMMITTED INVENTORY'S
    //  CRITERION-P SECTION, not a list rewritten here from the same sources.
    //  T6 asserts the correspondence in both directions: every ADMITTED row
    //  appears below, and no EXCLUDED or KILLED row does.
    //
    //  ORDER IS THE PAYLOAD ORDER. Appending here, after every V1 field, is
    //  deliberate: it keeps the V1 prefix of a V2 file byte-identical to a V1
    //  file, which makes a hexdump diff of the two readable. It buys no
    //  compatibility -- a V1 file is refused outright on the version test --
    //  and it is not relied on for any.
    // =====================================================================

    // --- Node[]: the routing half of TNode.
    //
    //     Node.surDepth is NOT here, and its absence is load-bearing. It was
    //     ADMITTED by the inventory as "surcharge depth carried across steps";
    //     measured, its only writes are node.c's .inp readers and its only
    //     reads are in dynwave.c -- no routing write exists. It is .inp
    //     CONFIGURATION, which is the ONE class Criterion P excludes by triage
    //     rather than admitting freely, because a stale snapshot of it silently
    //     overrides the model the operator is running. The inventory row was
    //     corrected in the same commit that added this block.
    if ( c->mode != SNAP_MANIFEST && Nobjects[NODE] > 0 && !Node ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[NODE]); i++)
    {
        TNode* s = SNAP_P(c, Node, i, _snapDummyNodeObj);
        SNAP_D(c, "Node", *s, inflow);
        SNAP_D(c, "Node", *s, outflow);
        SNAP_D(c, "Node", *s, losses);
        SNAP_D(c, "Node", *s, newVolume);
        SNAP_D(c, "Node", *s, overflow);
        SNAP_D(c, "Node", *s, oldDepth);
        SNAP_D(c, "Node", *s, newDepth);
        SNAP_D(c, "Node", *s, newLatFlow);
        SNAP_D(c, "Node", *s, oldFlowInflow);
        SNAP_D(c, "Node", *s, oldNetInflow);
        SNAP_C(c, "Node", *s, updated);
    }

    // --- Link[]: the routing half of TLink.
    if ( c->mode != SNAP_MANIFEST && Nobjects[LINK] > 0 && !Link ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nobjects[LINK]); i++)
    {
        TLink* s = SNAP_P(c, Link, i, _snapDummyLinkObj);
        SNAP_D(c, "Link", *s, newFlow);
        SNAP_D(c, "Link", *s, newDepth);
        SNAP_D(c, "Link", *s, newVolume);
        SNAP_D(c, "Link", *s, setting);
        SNAP_D(c, "Link", *s, targetSetting);
        SNAP_D(c, "Link", *s, timeLastSet);
        SNAP_D(c, "Link", *s, froude);
        SNAP_D(c, "Link", *s, dqdh);
        SNAP_I(c, "Link", *s, flowClass);
        SNAP_C(c, "Link", *s, normalFlow);
        SNAP_C(c, "Link", *s, inletControl);
    }

    // --- Conduit[]: indexed by Nlinks[CONDUIT], NOT by Nobjects[LINK].
    if ( c->mode != SNAP_MANIFEST && Nlinks[CONDUIT] > 0 && !Conduit ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nlinks[CONDUIT]); i++)
    {
        TConduit* s = SNAP_P(c, Conduit, i, _snapDummyConduit);
        SNAP_D(c, "Conduit", *s, a1);
        SNAP_D(c, "Conduit", *s, q1);
        SNAP_D(c, "Conduit", *s, q2);
        SNAP_D(c, "Conduit", *s, evapLossRate);
        SNAP_D(c, "Conduit", *s, seepLossRate);
        SNAP_C(c, "Conduit", *s, capacityLimited);
        SNAP_C(c, "Conduit", *s, fullState);
    }

    // --- Outfall[]: vRouted is a scalar; wRouted is a per-pollutant ARRAY, so
    //     it contributes Nobjects[POLLUT] values per outfall and exactly one
    //     manifest line (the manifest is a set of names, not of slots).
    if ( c->mode != SNAP_MANIFEST && Nnodes[OUTFALL] > 0 && !Outfall ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nnodes[OUTFALL]); i++)
    {
        TOutfall* s = SNAP_P(c, Outfall, i, _snapDummyOutfallObj);
        SNAP_D(c, "Outfall", *s, vRouted);
        if ( c->mode != SNAP_MANIFEST && Nobjects[POLLUT] > 0 && !s->wRouted )
        { c->error = 1; return; }
        for (k = 0; k < SNAP_N(c, Nobjects[POLLUT]); k++)
        {
            snap_name(c, "Outfall", "wRouted");
            snap_d(c, (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar
                                                 : &s->wRouted[k]);
        }
    }

    // --- Storage[]: indexed by Nnodes[STORAGE].
    if ( c->mode != SNAP_MANIFEST && Nnodes[STORAGE] > 0 && !Storage ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nnodes[STORAGE]); i++)
    {
        TStorage* s = SNAP_P(c, Storage, i, _snapDummyStorageObj);
        SNAP_D(c, "Storage", *s, hrt);
        SNAP_D(c, "Storage", *s, evapLoss);
        SNAP_D(c, "Storage", *s, exfilLoss);
    }

    // --- Xnode[]: dynwave.c's extended nodal array.
    //
    //     Reached ONLY through dynwave_getSnapshotRefs, because TXnode is
    //     declared inside dynwave.c and appears in no header -- this file
    //     cannot name the type, let alone reach the static. The count comes
    //     from the same file and is 0 when routing is not dynamic wave, in
    //     which case this loop emits nothing in write/read mode while the
    //     manifest still names both fields.
    for (i = 0; i < SNAP_N(c, dynwave_getSnapshotCount()); i++)
    {
        double* osa = &_snapDummyScalar;
        double* dyd = &_snapDummyScalar;
        if ( c->mode != SNAP_MANIFEST &&
             !dynwave_getSnapshotRefs(i, &osa, &dyd) ) { c->error = 1; return; }
        snap_name(c, "Xnode", "oldSurfArea"); snap_d(c, osa);
        snap_name(c, "Xnode", "dYdT");        snap_d(c, dyd);
    }
}

//=============================================================================

char* snapshot_buildManifest(long* nFields)
//
//  Input:   nFields = receives the number of manifest entries (may be NULL)
//  Output:  returns a newly allocated NUL-terminated manifest string, or NULL
//  Purpose: emits the FIELD MANIFEST -- the {object, field} names this
//           serializer writes, in write order, one per line.
//
//  The manifest is generated by the SAME traversal that writes the payload, so
//  it cannot describe a different set of fields than the payload contains. A
//  manifest maintained separately would be a second claim about the format
//  rather than a description of it.
//
{
    TSnapCtx c;
    memset(&c, 0, sizeof(c));
    c.mode = SNAP_MANIFEST;
    snapshot_traverse(&c, SNAPSHOT_MAX_STATS);
    if ( c.error ) { free(c.manifest); return NULL; }
    if ( nFields ) *nFields = c.count;
    return c.manifest;
}

//=============================================================================

int snapshot_save(const char* path)
//
//  Input:   path = snapshot file to create
//  Output:  returns 0 on success, non-zero on failure
//  Purpose: writes SWMM's statistics and mass-balance accumulators at full
//           double precision.
//
{
    FILE*  f;
    TSnapCtx c;
    TSnapshotShape shape;
    int    hdr[4];
    int    shp[SNAPSHOT_SHAPE_FIELDS];
    char*  manifest;
    long   nFields = 0;
    int    mlen;

    manifest = snapshot_buildManifest(&nFields);
    if ( !manifest ) return 1;
    mlen = (int) strlen(manifest);

    f = fopen(path, "wb");
    if ( !f ) { free(manifest); return 1; }

    snapshot_currentShape(&shape, SNAPSHOT_MAX_STATS);
    hdr[0] = SNAPSHOT_MAGIC;
    hdr[1] = SNAPSHOT_VERSION;
    hdr[2] = (int) sizeof(double);
    hdr[3] = mlen;
    memcpy(shp, &shape, sizeof(shp));

    if ( fwrite(hdr, sizeof(int), 4, f) != 4 ||
         fwrite(shp, sizeof(int), SNAPSHOT_SHAPE_FIELDS, f) != SNAPSHOT_SHAPE_FIELDS ||
         (mlen > 0 && fwrite(manifest, 1, (size_t) mlen, f) != (size_t) mlen) )
    {
        fclose(f); free(manifest); return 1;
    }
    free(manifest);

    memset(&c, 0, sizeof(c));
    c.mode = SNAP_WRITE;
    c.f    = f;
    snapshot_traverse(&c, SNAPSHOT_MAX_STATS);

    if ( fclose(f) != 0 ) return 1;
    return c.error;
}

//=============================================================================

int snapshot_load(const char* path, char* errMsg, int errMsgLen)
//
//  Input:   path      = snapshot file to read
//           errMsg    = buffer receiving a human-readable refusal reason
//           errMsgLen = its size
//  Output:  returns 0 on success, non-zero on refusal
//  Purpose: restores SWMM's accumulators from a snapshot, REFUSING loudly and
//           specifically rather than restoring a file that does not describe
//           the running model.
//
//  Every refusal names the mismatching quantity. A generic "bad snapshot" would
//  send an operator to re-run from scratch without telling them which of the
//  .inp, the build precision or the toolkit version changed under them.
//
{
    FILE* f;
    TSnapCtx c;
    TSnapshotShape want, got;
    int   hdr[4];
    int   shp[SNAPSHOT_SHAPE_FIELDS];
    char* manifest = NULL;
    char* mine     = NULL;
    long  nFields  = 0;
    int   rc = 1;

    if ( errMsg && errMsgLen > 0 ) errMsg[0] = '\0';

    f = fopen(path, "rb");
    if ( !f )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot not found: %s", path);
        return 1;
    }

    if ( fread(hdr, sizeof(int), 4, f) != 4 )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot header is short (file truncated): %s", path);
        fclose(f); return 1;
    }
    if ( hdr[0] != SNAPSHOT_MAGIC )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot has a bad magic (0x%08X, expected 0x%08X): %s",
            (unsigned) hdr[0], (unsigned) SNAPSHOT_MAGIC, path);
        fclose(f); return 1;
    }
    if ( hdr[1] != SNAPSHOT_VERSION )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot is version %d but this build reads version %d",
            hdr[1], SNAPSHOT_VERSION);
        fclose(f); return 1;
    }
    if ( hdr[2] != (int) sizeof(double) )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot stores %d-byte values but this build's double is %d bytes",
            hdr[2], (int) sizeof(double));
        fclose(f); return 1;
    }
    if ( fread(shp, sizeof(int), SNAPSHOT_SHAPE_FIELDS, f) != SNAPSHOT_SHAPE_FIELDS )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot shape block is short (file truncated): %s", path);
        fclose(f); return 1;
    }
    memcpy(&got, shp, sizeof(got));
    snapshot_currentShape(&want, SNAPSHOT_MAX_STATS);

#define SHAPE_CHECK(field, label)                                              \
    if ( got.field != want.field )                                             \
    {                                                                          \
        if ( errMsg ) snprintf(errMsg, errMsgLen,                              \
            "snapshot describes %d %s but this run has %d -- the .inp or the "  \
            "coupling changed between runs", got.field, label, want.field);    \
        fclose(f); return 1;                                                   \
    }
    SHAPE_CHECK(nNodes,      "node(s)")
    SHAPE_CHECK(nLinks,      "link(s)")
    SHAPE_CHECK(nSubcatch,   "subcatchment(s)")
    SHAPE_CHECK(nPollut,     "pollutant(s)")
    SHAPE_CHECK(nStorage,    "storage node(s)")
    SHAPE_CHECK(nOutfall,    "outfall node(s)")
    SHAPE_CHECK(nPump,       "pump(s)")
    SHAPE_CHECK(maxStats,    "critical-statistics slot(s)")
    SHAPE_CHECK(timeLevels,  "time-step level(s)")
    SHAPE_CHECK(flowClasses, "flow class(es)")
    SHAPE_CHECK(nConduit,    "conduit link(s)")
    // nXnode is 0 unless routing is dynamic wave, so this row also refuses a
    // snapshot taken under a different routing model rather than reading its
    // Xnode payload at the wrong offset.
    SHAPE_CHECK(nXnode,      "dynamic-wave extended node(s)")
#undef SHAPE_CHECK

    // --- the manifest must describe the field set this build serializes.
    //     This is the check that catches an upstream struct change the shape
    //     block cannot see: the counts agree, the payload strides differently.
    if ( hdr[3] > 0 )
    {
        manifest = (char*) malloc((size_t) hdr[3] + 1);
        if ( !manifest )
        {
            if ( errMsg ) snprintf(errMsg, errMsgLen, "out of memory reading snapshot manifest");
            fclose(f); return 1;
        }
        if ( fread(manifest, 1, (size_t) hdr[3], f) != (size_t) hdr[3] )
        {
            if ( errMsg ) snprintf(errMsg, errMsgLen,
                "snapshot manifest is short (file truncated): %s", path);
            free(manifest); fclose(f); return 1;
        }
        manifest[hdr[3]] = '\0';
    }
    mine = snapshot_buildManifest(&nFields);
    if ( !mine )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen, "could not build this build's snapshot manifest");
        free(manifest); fclose(f); return 1;
    }
    if ( !manifest || strcmp(manifest, mine) != 0 )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot field manifest does not match this build's serializer -- "
            "the vendored EPA structs changed between the writing and reading "
            "builds; re-run from a clean start (checkpoint_id=0)");
        free(manifest); free(mine); fclose(f); return 1;
    }
    free(manifest); free(mine);

    memset(&c, 0, sizeof(c));
    c.mode = SNAP_READ;
    c.f    = f;
    snapshot_traverse(&c, SNAPSHOT_MAX_STATS);
    rc = c.error;
    if ( rc && errMsg ) snprintf(errMsg, errMsgLen,
        "snapshot payload is short or unreadable: %s", path);

    fclose(f);
    return rc;
}
