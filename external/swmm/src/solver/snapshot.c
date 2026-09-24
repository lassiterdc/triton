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
#define SNAPSHOT_VERSION  1

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
} TSnapshotShape;

#define SNAPSHOT_SHAPE_FIELDS 10

static void snapshot_currentShape(TSnapshotShape* s, int maxStats)
{
    s->nNodes      = Nobjects[NODE];
    s->nLinks      = Nobjects[LINK];
    s->nSubcatch   = Nobjects[SUBCATCH];
    s->nPollut     = Nobjects[POLLUT];
    s->nStorage    = Nnodes[STORAGE];
    s->nOutfall    = Nnodes[OUTFALL];
    s->nPump       = Nlinks[PUMP];
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

#define SNAP_D(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_d(ctx, &((s).f)); } while (0)
#define SNAP_I(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_i(ctx, &((s).f)); } while (0)
#define SNAP_L(ctx, obj, s, f)  do { snap_name(ctx, obj, #f); snap_l(ctx, &((s).f)); } while (0)

//-----------------------------------------------------------------------------
//  The traversal.
//
//  Field order here IS the payload order and IS the manifest order. The
//  inventory at test/inventory/snapshot_inventory.txt is the committed record
//  of what this traversal must cover, and the T6 test asserts the emitted
//  manifest against it.
//-----------------------------------------------------------------------------
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

    stats_getSnapshotRefs(&tss, &mbe, &cc, &ft, &nc, &sysOut, &nStats);
    if ( nStats != maxStats ) { c->error = 1; return; }

    // --- node statistics
    for (i = 0; i < Nobjects[NODE] && NodeStats; i++)
    {
        SNAP_D(c, "NodeStats", NodeStats[i], avgDepth);
        SNAP_D(c, "NodeStats", NodeStats[i], maxDepth);
        SNAP_D(c, "NodeStats", NodeStats[i], maxDepthDate);
        SNAP_D(c, "NodeStats", NodeStats[i], maxRptDepth);
        SNAP_D(c, "NodeStats", NodeStats[i], volFlooded);
        SNAP_D(c, "NodeStats", NodeStats[i], timeFlooded);
        SNAP_D(c, "NodeStats", NodeStats[i], timeSurcharged);
        SNAP_D(c, "NodeStats", NodeStats[i], timeCourantCritical);
        SNAP_D(c, "NodeStats", NodeStats[i], totLatFlow);
        SNAP_D(c, "NodeStats", NodeStats[i], maxLatFlow);
        SNAP_D(c, "NodeStats", NodeStats[i], maxInflow);
        SNAP_D(c, "NodeStats", NodeStats[i], maxOverflow);
        SNAP_D(c, "NodeStats", NodeStats[i], maxPondedVol);
        SNAP_I(c, "NodeStats", NodeStats[i], nonConvergedCount);
        SNAP_D(c, "NodeStats", NodeStats[i], maxInflowDate);
        SNAP_D(c, "NodeStats", NodeStats[i], maxOverflowDate);
        if ( c->mode == SNAP_MANIFEST ) break;   // one instance names the fields
    }

    // --- link statistics
    for (i = 0; i < Nobjects[LINK] && LinkStats; i++)
    {
        SNAP_D(c, "LinkStats", LinkStats[i], maxFlow);
        SNAP_D(c, "LinkStats", LinkStats[i], maxFlowDate);
        SNAP_D(c, "LinkStats", LinkStats[i], maxVeloc);
        SNAP_D(c, "LinkStats", LinkStats[i], maxDepth);
        SNAP_D(c, "LinkStats", LinkStats[i], maxStreetFilled);
        SNAP_D(c, "LinkStats", LinkStats[i], timeNormalFlow);
        SNAP_D(c, "LinkStats", LinkStats[i], timeInletControl);
        SNAP_D(c, "LinkStats", LinkStats[i], timeSurcharged);
        SNAP_D(c, "LinkStats", LinkStats[i], timeFullUpstream);
        SNAP_D(c, "LinkStats", LinkStats[i], timeFullDnstream);
        SNAP_D(c, "LinkStats", LinkStats[i], timeFullFlow);
        SNAP_D(c, "LinkStats", LinkStats[i], timeCapacityLimited);
        for (k = 0; k < MAX_FLOW_CLASSES; k++)
        {
            snap_name(c, "LinkStats", "timeInFlowClass");
            snap_d(c, &LinkStats[i].timeInFlowClass[k]);
        }
        SNAP_D(c, "LinkStats", LinkStats[i], timeCourantCritical);
        SNAP_L(c, "LinkStats", LinkStats[i], flowTurns);
        SNAP_I(c, "LinkStats", LinkStats[i], flowTurnSign);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- storage statistics
    for (i = 0; i < Nnodes[STORAGE] && StorageStats; i++)
    {
        SNAP_D(c, "StorageStats", StorageStats[i], initVol);
        SNAP_D(c, "StorageStats", StorageStats[i], avgVol);
        SNAP_D(c, "StorageStats", StorageStats[i], maxVol);
        SNAP_D(c, "StorageStats", StorageStats[i], maxFlow);
        SNAP_D(c, "StorageStats", StorageStats[i], evapLosses);
        SNAP_D(c, "StorageStats", StorageStats[i], exfilLosses);
        SNAP_D(c, "StorageStats", StorageStats[i], maxVolDate);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- outfall statistics. totalLoad is a per-pollutant array hanging off
    //     the struct; the pointer is NOT serialized, its contents are.
    for (i = 0; i < Nnodes[OUTFALL] && OutfallStats; i++)
    {
        SNAP_D(c, "OutfallStats", OutfallStats[i], avgFlow);
        SNAP_D(c, "OutfallStats", OutfallStats[i], maxFlow);
        SNAP_I(c, "OutfallStats", OutfallStats[i], totalPeriods);
        for (k = 0; k < Nobjects[POLLUT]; k++)
        {
            snap_name(c, "OutfallStats", "totalLoad");
            if ( OutfallStats[i].totalLoad ) snap_d(c, &OutfallStats[i].totalLoad[k]);
            else if ( c->mode != SNAP_MANIFEST ) { c->error = 1; return; }
        }
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- pump statistics
    for (i = 0; i < Nlinks[PUMP] && PumpStats; i++)
    {
        SNAP_D(c, "PumpStats", PumpStats[i], utilized);
        SNAP_D(c, "PumpStats", PumpStats[i], minFlow);
        SNAP_D(c, "PumpStats", PumpStats[i], avgFlow);
        SNAP_D(c, "PumpStats", PumpStats[i], maxFlow);
        SNAP_D(c, "PumpStats", PumpStats[i], volume);
        SNAP_D(c, "PumpStats", PumpStats[i], energy);
        SNAP_D(c, "PumpStats", PumpStats[i], offCurveLow);
        SNAP_D(c, "PumpStats", PumpStats[i], offCurveHigh);
        SNAP_I(c, "PumpStats", PumpStats[i], startUps);
        SNAP_I(c, "PumpStats", PumpStats[i], totalPeriods);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- subcatchment statistics
    for (i = 0; i < Nobjects[SUBCATCH] && SubcatchStats; i++)
    {
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], precip);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], runon);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], evap);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], infil);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], runoff);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], maxFlow);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], impervRunoff);
        SNAP_D(c, "SubcatchStats", SubcatchStats[i], pervRunoff);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- time-step statistics (a single instance, reached via stats.c)
    if ( tss )
    {
        SNAP_D(c, "TimeStepStats", *tss, minTimeStep);
        SNAP_D(c, "TimeStepStats", *tss, maxTimeStep);
        SNAP_D(c, "TimeStepStats", *tss, routingTime);
        SNAP_I(c, "TimeStepStats", *tss, timeStepCount);
        SNAP_D(c, "TimeStepStats", *tss, trialsCount);
        SNAP_D(c, "TimeStepStats", *tss, steadyStateTime);
        for (k = 0; k < TIMELEVELS; k++)
        {
            snap_name(c, "TimeStepStats", "timeStepIntervals");
            snap_d(c, &tss->timeStepIntervals[k]);
        }
        for (k = 0; k < TIMELEVELS; k++)
        {
            snap_name(c, "TimeStepStats", "timeStepCounts");
            snap_i(c, &tss->timeStepCounts[k]);
        }
    }

    // --- the four critical-statistics arrays.
    //
    //     stats_findMaxStats() RECOMPUTES all four at report time from
    //     NodeStats / LinkStats / NodeInflow / NodeOutflow / timeStepCount /
    //     ReportStepCount, so restoring them is redundant on the happy path.
    //     They are serialized anyway because the criterion admits them -- the
    //     report writers read them -- and because a redundant restore that a
    //     later recompute overwrites costs 60 doubles and cannot be wrong,
    //     while omitting them would make this file's contents depend on a
    //     derivation holding somewhere else.
    {
        TMaxStats* arrays[4];
        const char* names[4] = { "MaxMassBalErrs", "MaxCourantCrit",
                                 "MaxFlowTurns",   "MaxNonConverged" };
        int a;
        arrays[0] = mbe; arrays[1] = cc; arrays[2] = ft; arrays[3] = nc;
        for (a = 0; a < 4; a++)
        {
            if ( !arrays[a] ) continue;
            for (i = 0; i < maxStats; i++)
            {
                SNAP_I(c, names[a], arrays[a][i], objType);
                SNAP_I(c, names[a], arrays[a][i], index);
                SNAP_D(c, names[a], arrays[a][i], value);
                if ( c->mode == SNAP_MANIFEST ) break;
            }
        }
    }

    // --- mass-balance accumulators
    SNAP_D(c, "RunoffTotals", RunoffTotals, rainfall);
    SNAP_D(c, "RunoffTotals", RunoffTotals, evap);
    SNAP_D(c, "RunoffTotals", RunoffTotals, infil);
    SNAP_D(c, "RunoffTotals", RunoffTotals, runoff);
    SNAP_D(c, "RunoffTotals", RunoffTotals, drains);
    SNAP_D(c, "RunoffTotals", RunoffTotals, runon);
    SNAP_D(c, "RunoffTotals", RunoffTotals, initStorage);
    SNAP_D(c, "RunoffTotals", RunoffTotals, finalStorage);
    SNAP_D(c, "RunoffTotals", RunoffTotals, initSnowCover);
    SNAP_D(c, "RunoffTotals", RunoffTotals, finalSnowCover);
    SNAP_D(c, "RunoffTotals", RunoffTotals, snowRemoved);
    SNAP_D(c, "RunoffTotals", RunoffTotals, pctError);

    SNAP_D(c, "GwaterTotals", GwaterTotals, infil);
    SNAP_D(c, "GwaterTotals", GwaterTotals, upperEvap);
    SNAP_D(c, "GwaterTotals", GwaterTotals, lowerEvap);
    SNAP_D(c, "GwaterTotals", GwaterTotals, lowerPerc);
    SNAP_D(c, "GwaterTotals", GwaterTotals, gwater);
    SNAP_D(c, "GwaterTotals", GwaterTotals, initStorage);
    SNAP_D(c, "GwaterTotals", GwaterTotals, finalStorage);
    SNAP_D(c, "GwaterTotals", GwaterTotals, pctError);

    {
        TRoutingTotals* rt[3];
        const char* rtn[3] = { "FlowTotals", "StepFlowTotals", "OldStepFlowTotals" };
        int a;
        rt[0] = &FlowTotals; rt[1] = &StepFlowTotals; rt[2] = &OldStepFlowTotals;
        for (a = 0; a < 3; a++)
        {
            SNAP_D(c, rtn[a], *rt[a], dwInflow);
            SNAP_D(c, rtn[a], *rt[a], wwInflow);
            SNAP_D(c, rtn[a], *rt[a], gwInflow);
            SNAP_D(c, rtn[a], *rt[a], iiInflow);
            SNAP_D(c, rtn[a], *rt[a], exInflow);
            SNAP_D(c, rtn[a], *rt[a], flooding);
            SNAP_D(c, rtn[a], *rt[a], outflow);
            SNAP_D(c, rtn[a], *rt[a], evapLoss);
            SNAP_D(c, rtn[a], *rt[a], seepLoss);
            SNAP_D(c, rtn[a], *rt[a], reacted);
            SNAP_D(c, rtn[a], *rt[a], initStorage);
            SNAP_D(c, rtn[a], *rt[a], finalStorage);
            SNAP_D(c, rtn[a], *rt[a], pctError);
        }
    }

    // --- per-pollutant totals
    for (i = 0; i < Nobjects[POLLUT] && LoadingTotals; i++)
    {
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], initLoad);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], buildup);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], deposition);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], sweeping);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], bmpRemoval);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], infil);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], runoff);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], finalLoad);
        SNAP_D(c, "LoadingTotals", LoadingTotals[i], pctError);
        if ( c->mode == SNAP_MANIFEST ) break;
    }
    for (i = 0; i < Nobjects[POLLUT] && QualTotals; i++)
    {
        SNAP_D(c, "QualTotals", QualTotals[i], dwInflow);
        SNAP_D(c, "QualTotals", QualTotals[i], wwInflow);
        SNAP_D(c, "QualTotals", QualTotals[i], gwInflow);
        SNAP_D(c, "QualTotals", QualTotals[i], iiInflow);
        SNAP_D(c, "QualTotals", QualTotals[i], exInflow);
        SNAP_D(c, "QualTotals", QualTotals[i], flooding);
        SNAP_D(c, "QualTotals", QualTotals[i], outflow);
        SNAP_D(c, "QualTotals", QualTotals[i], evapLoss);
        SNAP_D(c, "QualTotals", QualTotals[i], seepLoss);
        SNAP_D(c, "QualTotals", QualTotals[i], reacted);
        SNAP_D(c, "QualTotals", QualTotals[i], initStorage);
        SNAP_D(c, "QualTotals", QualTotals[i], finalStorage);
        SNAP_D(c, "QualTotals", QualTotals[i], pctError);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- per-node mass-balance volumes. stats_findMaxStats divides by these,
    //     so a resume that leaves them at the post-resume-only accumulation
    //     reports mass-balance errors against the wrong denominator.
    for (i = 0; i < Nobjects[NODE] && NodeInflow; i++)
    {
        snap_name(c, "NodeInflow", "value");
        snap_d(c, &NodeInflow[i]);
        if ( c->mode == SNAP_MANIFEST ) break;
    }
    for (i = 0; i < Nobjects[NODE] && NodeOutflow; i++)
    {
        snap_name(c, "NodeOutflow", "value");
        snap_d(c, &NodeOutflow[i]);
        if ( c->mode == SNAP_MANIFEST ) break;
    }

    // --- scalar accumulators.
    //
    //     ReportStepCount is a globals.h global rather than a stats.c static,
    //     and it is the divisor stats_findMaxStats uses for the flow-turns
    //     percentage. It is easy to miss for exactly that reason: it is not in
    //     any stats struct and not among stats.c's statics, so an inventory
    //     seeded on either would not reach it.
    //     NonConvergeCount is the same shape and was found the same way: the
    //     time-step frequency table divides by it at report.c:1076, and it
    //     lives in neither a stats struct nor stats.c's statics.
    snap_name(c, "MaxOutfallFlow",   "value"); snap_d(c, &MaxOutfallFlow);
    snap_name(c, "MaxRunoffFlow",    "value"); snap_d(c, &MaxRunoffFlow);
    snap_name(c, "RoutingTimeSpan",  "value"); snap_d(c, &RoutingTimeSpan);
    snap_name(c, "TotalArea",        "value"); snap_d(c, &TotalArea);
    if ( sysOut ) { snap_name(c, "SysOutfallFlow", "value"); snap_d(c, sysOut); }
    snap_name(c, "ReportStepCount",  "value"); snap_l(c, &ReportStepCount);
    snap_name(c, "NonConvergeCount", "value"); snap_l(c, &NonConvergeCount);
    snap_name(c, "TotalStepCount",   "value"); snap_l(c, &TotalStepCount);
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
