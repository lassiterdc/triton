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
//   idle for the whole replay. A long run was measured spending 11,107 s
//   replaying before the resumed segment began any work, and was cancelled
//   still replaying; an idle GPU for that whole period is also a job a
//   scheduler may reclaim. Each successive resume replays a longer prefix, so
//   the sequence does not converge -- a run interrupted often enough cannot
//   complete at all. This snapshot makes the restore cost independent of t_k.
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
//   whose duration counters are silently low. A resumed run whose SWMM
//   summaries cover a different window than an uninterrupted run's is the
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
#include <errno.h>
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
// VERSION 3 adds the four Criterion-P GLOBAL ROUTING CLOCKS -- NewRoutingTime,
// ReportTime, NewRuleTime and NextEvent.  It adds NO shape dimension: each is a
// singleton, so the shape block is byte-identical between V2 and V3.
//
// VERSION 4 adds the two Criterion-P routing SENTINELS that closed on triage
// rather than on measurement -- BetweenEvents (routing.c) and VariableStep
// (dynwave.c).  Also singletons, so the shape block is again unchanged.
//
// THE VERSION NAMES THE FIELD SET, so it moves whenever the field set moves,
// even twice in one working session.  V3 and V4 both exist on this branch, a
// build from either could have written a file, and a V3 file met by a V4 reader
// must say "version 3 against version 4" rather than blaming the vendored tree.
// Collapsing the two bumps into one would save a constant and buy a wrong
// message on exactly the file class the intermediate commit creates.
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
//
// WHY V2 -> V3 IS A BUMP WHEN THE REFUSAL DOES NOT NEED ONE.  A V2 file is
// ALREADY refused without it, twice over: snapshot_load strcmp's the file's
// EMBEDDED manifest against this build's, and the four names change that text;
// and the payload is four doubles short, so the traversal's final fread fails.
// Neither path can mis-stride.  The bump is not for the refusal -- it is for
// WHAT THE REFUSAL SAYS.  The manifest path reports "the vendored EPA structs
// changed between the writing and reading builds", which is FALSE here: EPA's
// structs did not move, TRITON's admitted set grew.  That message sends an
// operator to diff the vendored tree for a change that is not in it.  The
// version test fires first and says "snapshot is version 2 but this build reads
// version 3", which is true and actionable.  A bump costs one constant; a
// refusal that blames the wrong party costs an investigation.
// BUMPED 4 -> 5 with the output-period count (Nperiods)
// admitted.  Same ground as every prior bump and it is NOT the refusal itself:
// the manifest comparison and the short final fread ALREADY refuse a V4 file
// against a V5 build.  The bump is for WHAT THE REFUSAL SAYS.  The manifest
// path reports that "the vendored EPA structs changed between the writing and
// reading builds", which is false here -- EPA's structs did not move, TRITON's
// admitted set grew -- and sends an operator to diff the vendored tree for a
// change that is not in it.  The version test fires first and says "snapshot
// is version 4 but this build reads version 5", which is true and actionable.
//
// VERIFIED AT SOURCE BEFORE BUMPING rather than taken from the instruction.
// An earlier draft of this step said "bump from 1"; executing that literally
// would have set the constant to 2, after which snapshot_load's version test
// REJECTS EVERY REAL SNAPSHOT IN THE TREE and silently routes every resume
// back to the replay path -- reproducing the very coverage regression this
// change exists to prevent. The constant tracks the format, not the step.
//
// BUMPED 5 -> 6 with the Outfall.wRouted FIXED-WIDTH PLACEHOLDER, and this one
// bump sits awkwardly against the doctrine sentence above, so the reasoning is
// recorded HERE rather than inferred.  THE FIELD SET DID NOT CHANGE: the
// manifest is byte-identical across V5 and V6, because the manifest is a set of
// NAMES and wRouted was already named unconditionally.  What changed is the
// number of VALUES the payload carries per outfall -- previously
// Nobjects[POLLUT] per outfall only when that outfall had a route, and now
// Nobjects[POLLUT] per outfall always.  So a reader must not infer a field-set
// change from this constant.
//
// WHY BUMP AT ALL WHEN SAFETY DOES NOT NEED IT.  Measured in both directions,
// and both already refuse loudly without a bump: a V6 reader meeting a V5 file
// fails at the first fread past the written extent and returns
// SNAPSHOT_UNREADABLE; a V5 reader meeting a V6 file reaches its own unrepaired
// guard on the read traversal, sets c->error there, and returns
// SNAPSHOT_UNREADABLE too.  The bump is for WHAT THE REFUSAL SAYS, which is the
// same ground as every bump above it.  Without it the V5-reader direction says
// the payload is short about a file that is COMPLETE -- the exact misattribution
// the four-site message repair exists to remove, one layer down.
#define SNAPSHOT_VERSION  6

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
// SNAP_COUNT visits every slot, TOUCHES NO FILE, and leaves c->count holding
// the number of double-width values the payload carries for the RUNNING model.
//
// It costs no edit to snap_d/snap_i/snap_l/snap_c.  Each opens an error-gated
// return, then increments c->count, then branches on SNAP_WRITE or SNAP_READ --
// so a fourth mode falls through both branches and does exactly the counting.
// The increment is therefore NOT unconditional: a latched c->error stops it,
// which is the same early return the Outfall guard below relies on when it sets
// c->error and continues.  Those two statements have to agree, and an earlier
// form of this comment said "unconditionally" while the guard's comment said
// "returns immediately on it", so the file asserted both halves of one
// contradiction.
//
// WHAT THIS MODE DOES READ FROM MODEL MEMORY, stated because an earlier form of
// this comment claimed it read none and that was false.  SNAP_P's non-manifest
// branch is &((arr)[i]), and every SNAP_D/I/L/C expands to an ADDRESS-OF on
// that element, which is address arithmetic rather than a load -- in this mode
// nothing reads through the address.  Exactly ONE predicate in the whole
// traversal loads a FIELD of the per-element pointer in a non-manifest mode:
// the Outfall guard's `s->routeTo` and `s->wRouted`.  So the count pass reads
// model memory at one site, not none.
//
// WHY THE ARRAY GUARDS STILL NEED NO EDIT, which is the conclusion that
// survives the correction above.  They need none because they FIRE in this mode
// -- every one is gated on the mode being other than SNAP_MANIFEST, so a count
// pass over a model whose arrays are absent refuses exactly as a write pass
// does.  It is NOT because the count pass dereferences nothing.  TWO guards do
// two different jobs at the Outfall block and both are load-bearing: the
// PRE-LOOP guard on `!Outfall` returns before the loop, so the element address
// is never formed on a null array; the PER-ELEMENT guard's two loads are then
// safe only BECAUSE that pre-loop guard already fired.
//
// Deliberately NOT narrowed to `SNAP_WRITE || SNAP_READ`, and deliberately not
// given a count-only fast path that skips the pre-loop guard.  Either edit
// reads as a no-op under the retired reason and is a null dereference under the
// real one: with the pre-loop guard narrowed, a count pass over a model with
// Nnodes[OUTFALL] > 0 and a null Outfall forms &Outfall[i], reaches the
// per-element guard, and faults on the load of s->routeTo.
typedef enum { SNAP_MANIFEST, SNAP_WRITE, SNAP_READ, SNAP_COUNT } TSnapMode;

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
static int            _snapDummyInt;
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
    // The Outfall.wRouted fixed-width placeholder.  Function-local rather than
    // the file-scope _snapDummyScalar so this file's bytes do not depend on a
    // property of every other use of that shared scratch.  WRITE mode never
    // assigns it, so it stays 0.0 for the whole pass; READ mode assigns it as a
    // discard, where the value is not read back.
    double          wRoutedPlaceholder = 0.0;
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
    //     measured, its only writes are node.c's .inp readers (:143/:152/:179/
    //     :192) and NO routing write exists anywhere in the tree, so it carries
    //     nothing across steps. It is .inp CONFIGURATION, which is the ONE class
    //     Criterion P excludes by triage rather than admitting freely, because a
    //     stale snapshot of it silently overrides the model the operator is
    //     running. The inventory row was corrected in the same commit that added
    //     this block.
    //
    //     An earlier form of this note ALSO claimed its reads are confined to
    //     dynwave.c. That was false: surDepth is read at dynwave.c:736/:799,
    //     stats.c:617, link.c:450/:458 and node.c:216. The claim is removed
    //     rather than softened, because the exclusion never rested on it -- the
    //     write side is the whole argument, and a reader who checked the read
    //     claim would have found it wrong and had no way to tell whether the
    //     verdict went with it. It does not.
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
    //
    //     THE PER-OUTFALL VALUE COUNT IS FIXED AT Nobjects[POLLUT] AND DOES NOT
    //     DEPEND ON routeTo.  node.c:163-168 sets wRouted = NULL and allocates
    //     the array only under `routeTo >= 0`, so a route-less outfall has no
    //     array to read -- and the payload emits Nobjects[POLLUT] placeholder
    //     zeros for it anyway.  That is what keeps the twelve-dimension shape
    //     block sufficient: NONE of the twelve records routeTo, so a length that
    //     varied with it would produce a file passing all twelve SHAPE_CHECKs
    //     and the manifest strcmp and then MIS-STRIDING every later field --
    //     a silent wrong-state restore, which is the one outcome this format
    //     exists to refuse.
    //
    //     Deliberately NOT emitting nothing for an absent array: that is the
    //     mis-stride above.  Deliberately NOT sourcing the placeholder from
    //     _snapDummyScalar: it is a shared file-scope scratch written on no
    //     reachable path today, so sourcing from it would make this file's bytes
    //     depend on a property of every other use of that scratch, and nothing
    //     here or in the tests checks that property.  A function-local double
    //     removes the dependency at no cost.
    if ( c->mode != SNAP_MANIFEST && Nnodes[OUTFALL] > 0 && !Outfall ) { c->error = 1; return; }
    for (i = 0; i < SNAP_N(c, Nnodes[OUTFALL]); i++)
    {
        TOutfall* s = SNAP_P(c, Outfall, i, _snapDummyOutfallObj);
        SNAP_D(c, "Outfall", *s, vRouted);

        // RE-PREDICATED ON routeTo, which is what the allocation is keyed on.
        // The old predicate was `Nobjects[POLLUT] > 0 && !s->wRouted`, which
        // fires on EVERY route-less outfall in a model carrying pollutants --
        // a legitimate, routine state -- and so refused every snapshot the
        // Norfolk campaign wrote.  It now fires only when the array SHOULD
        // exist and does not, which is a real defect in node.c's allocation.
        //
        // CONTINUE RATHER THAN RETURN.  The old `return` abandoned the whole
        // traversal from inside the outfall loop, in all three modes, so one
        // outfall's state determined whether the file described the model at
        // all.  c->error is already latched -- snap_d returns immediately on it
        // and snapshot_save propagates it -- so continuing costs nothing and
        // keeps the failure per-outfall instead of whole-file.
        if ( c->mode != SNAP_MANIFEST && Nobjects[POLLUT] > 0
             && s->routeTo >= 0 && !s->wRouted )
        { c->error = 1; continue; }

        for (k = 0; k < SNAP_N(c, Nobjects[POLLUT]); k++)
        {
            // Three-way slot selection, widening the pattern this file already
            // uses at SNAP_P and at the mode == SNAP_MANIFEST ternaries rather
            // than introducing a mechanism.  On WRITE with no array the
            // placeholder is the SOURCE (a zero); on READ with no array it is
            // the DISCARD.  Nothing writes wRoutedPlaceholder in WRITE mode, so
            // it holds its initializer for the whole pass.
            double* slot;
            snap_name(c, "Outfall", "wRouted");
            if      ( c->mode == SNAP_MANIFEST ) slot = &_snapDummyScalar;
            else if ( s->wRouted )               slot = &s->wRouted[k];
            else                                 slot = &wRoutedPlaceholder;
            snap_d(c, slot);
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

    // --- the GLOBAL ROUTING CLOCKS (V3).
    //
    //     These are singletons, not per-object arrays, so they carry the same
    //     "Name.value" shape the Criterion-R scalar accumulators above use --
    //     the object column is the quantity's own name and the field column is
    //     the literal "value". The committed inventory's Criterion-P section
    //     spells a scalar row in the OPPOSITE column order (`-<TAB>Name`), and
    //     T6 normalizes the two onto this one key; see the section-aware
    //     normalization in test/snapshot/test_state_snapshot.cpp.
    //
    //     EVERY ONE IS RESET BY THE START PATH AND READ BEFORE THE STEP WRITES
    //     IT. That pairing is what makes the liveness consequential: a clock
    //     that were live but never reset would survive a resume by accident.
    //     Deciding sites, from the committed inventory's ADMITTED rows:
    //
    //       NewRoutingTime  swmm_step's FIRST guarded statement reads it
    //                       (swmm5.c:439) before anything in the step writes
    //                       it; swmm_start sets it 0.0 at swmm5.c:354.
    //       ReportTime      read at swmm5.c:596 before saveResults advances it
    //                       at :618; its only PURE write is swmm_start's
    //                       initialization at :355.
    //       NewRuleTime     read at routing.c:191 before evaluateControlRules
    //                       advances it at :374; routing_open resets it at
    //                       :128.
    //       NextEvent       `Event[NextEvent].end` read at routing.c:411
    //                       before the `++` at :413; routing_open resets it at
    //                       :126.
    //
    //     NOT CALLER-DRIVEN, and the doc comment says otherwise. swmm_step's
    //     `elapsedTime` parameter is pure OUT: the function's first executable
    //     statement is `*elapsedTime = 0.0` (swmm5.c:425), which discards
    //     whatever the caller passed, and :458 writes it back from the internal
    //     global. The comment directly above the signature reads
    //     "Input: elapsedTime" and is wrong about the direction, which is why
    //     "the caller drives these clocks" is a plausible misreading. It does
    //     not hold, so the start-path zeroing governs unopposed.
    //
    //     NewRoutingTime and ReportTime are EXTERN doubles in globals.h and are
    //     named directly. NewRuleTime and NextEvent are `static` in routing.c
    //     and are reached through routing_getSnapshotRefs -- the same linkage
    //     pattern stats.c and dynwave.c already use, not a new mechanism.
    {
        double* nrt = &_snapDummyScalar;
        int*    nev = &_snapDummyInt;
        int*    bev = &_snapDummyInt;
        double* vst = &_snapDummyScalar;
        double* dd;

        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &NewRoutingTime;
        snap_name(c, "NewRoutingTime", "value"); snap_d(c, dd);
        dd = (c->mode == SNAP_MANIFEST) ? &_snapDummyScalar : &ReportTime;
        snap_name(c, "ReportTime", "value");     snap_d(c, dd);

        if ( c->mode != SNAP_MANIFEST ) routing_getSnapshotRefs(&nrt, &nev, &bev);
        snap_name(c, "NewRuleTime", "value");    snap_d(c, nrt);
        snap_name(c, "NextEvent", "value");      snap_i(c, nev);

        // --- the two routing SENTINELS (V4).
        //
        //     Both are read as a carried-over value before the step writes
        //     them, and both are reset by the start path -- the same pair of
        //     facts as the four clocks above. They are separated here only
        //     because they closed on a call-order argument rather than on a
        //     single function's statement order, which is what left them
        //     untriaged after the clocks were settled.
        //
        //       BetweenEvents  read routing.c:166 (routing_getRoutingStep),
        //                      written :235 (routing_execute); swmm5.c calls
        //                      the two at :540 and :570 in that order, so the
        //                      read precedes the write WITHIN a step.
        //                      routing_open resets it at :127.
        //       VariableStep   read dynwave.c:258, written :260/:264, all
        //                      inside dynwave_getRoutingStep -- but the read is
        //                      the FIRST reference on every path reaching it,
        //                      so it consumes the PREVIOUS call's value.
        //                      dynwave_init resets it at :176.
        //
        //     VariableStep's consequence is a SIDE EFFECT, not the routing step
        //     TRITON discards at swmm5.c:546: the non-sentinel branch calls
        //     getVariableStep -> stats_updateCriticalTimeCount, which
        //     increments the timeCourantCritical counters this same snapshot
        //     serializes under Criterion R.
        snap_name(c, "BetweenEvents", "value");  snap_i(c, bev);

        if ( c->mode != SNAP_MANIFEST ) dynwave_getVariableStepRef(&vst);
        snap_name(c, "VariableStep", "value");   snap_d(c, vst);
    }

    // =====================================================================
    // THE OUTPUT-PERIOD COUNT (format V5).
    //
    //  Nperiods is a globals.h `long` -- hence snap_l and not snap_i, which is
    //  the one operand in this chunk that a reader would guess wrong; the
    //  three scalars beside it in globals.h's EXTERN long block are the same
    //  type and already use snap_l.
    //
    //  IT WAS EXCLUDED, AND THE EXCLUSION'S GROUND WAS CIRCULAR. The inventory
    //  recorded it as "the period count of a file that is unrestorable by ANY
    //  offset, because output_openOutFile reopens Fout 'w+b' (truncating) and
    //  output_open resets Nperiods = 0, so there is nothing for a restored
    //  count to index." Every clause was true and the conclusion did not
    //  follow: the truncating open is a PROPERTY OF THE CODE, not a constraint
    //  on it. This package removes the truncation, and the count becomes both
    //  restorable and load-bearing.
    //
    //  WHAT IT IS LOAD-BEARING FOR, three consumers rather than one:
    //    - the resumed run's period offsets, via output_positionForResume;
    //    - output_end's trailer, whose fourth INT4 IS this count and which an
    //      external reader believes;
    //    - report.c's four detail-table loops, which run `period = 1 ..
    //      Nperiods` and read back from the .out, so the .rpt detail tables
    //      and the TRITON-added node{N}.out files truncate with it.
    //
    //  Appended AFTER every V4 field: field order here is the payload order
    //  and the manifest order, so appending is the only change that cannot
    //  re-stride an existing reader.
    // =====================================================================
    {
        long* np = (c->mode == SNAP_MANIFEST) ? &_snapDummyLong : &Nperiods;
        snap_name(c, "Nperiods", "value");       snap_l(c, np);
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

// The number of double-width values the payload carries for the RUNNING model,
// or -1 when the traversal refuses.  File-internal on purpose: the only two
// consumers are snapshot_save's own length verification and snapshot_load's
// expected-versus-actual refusal, both below, and keeping it static means this
// package adds no declaration to funcs.h.  The regression test computes its
// expected count FROM THE SHAPE rather than by calling this, so that the test
// compiles at both the pre-fix and post-fix commits.
static long snapshot_expectedValueCount(int maxStats)
{
    TSnapCtx c;
    memset(&c, 0, sizeof(c));
    c.mode = SNAP_COUNT;
    snapshot_traverse(&c, maxStats);
    if ( c.error ) return -1;
    return c.count;
}

//=============================================================================

// Suffix of the temporary a snapshot is written to before it is published.
//
// The extension is LOAD-BEARING and not cosmetic.  swmm_triton.h's
// classify_missing_snapshot enumerates the snapshot directory and skips any
// entry whose extension is not ".snapshot", so a file carrying this suffix is
// invisible to it -- which is what keeps every existing reason token unchanged
// and keeps a partial write from being counted as a published snapshot.  The
// retention prune in write_state_snapshot targets the exact ".snapshot" name and
// is blind to this suffix for the same reason.
#define SNAPSHOT_PART_SUFFIX  ".part"

// Suffix of the marker snapshot_save leaves on two failure arms: the count
// pass refusing the model state (the marker then holds one text line naming
// the refusal) and the length check failing after the write (the temporary is
// renamed to it, so the bytes that failed survive for inspection).  It must
// agree byte-for-byte with the ".failed" name swmm_triton.h's resume
// consultation and clean-start disposal test for, and it carries the same
// load-bearing property as ".part": its extension is not ".snapshot", so the
// classifier and the retention prune never count it as a published snapshot.
#define SNAPSHOT_FAILED_SUFFIX  ".failed"

// The one line a count-pass refusal writes into its ".failed" marker.
#define SNAPSHOT_REFUSED_TEXT \
    "SNAPSHOT REFUSED BEFORE WRITE: a structural guard rejected the model state in the count pass"

int snapshot_save(const char* path)
//
//  Input:   path = snapshot file to create
//  Output:  returns 0 on success, non-zero on failure
//  Purpose: writes SWMM's statistics and mass-balance accumulators at full
//           double precision.
//
//  WRITE TO A TEMPORARY, VERIFY ITS LENGTH, THEN PUBLISH BY RENAME.  An
//  unusable file is never published under the name a resume will load.  The
//  property this buys is not tidiness: before it, a process killed inside the
//  traversal left a SHORT file at the published name, which the next resume
//  opened, passed through every header and shape check, and then refused on a
//  truncated payload -- so a killed write and a build mismatch produced the
//  same refusal and the operator could not tell them apart.  After it, a killed
//  write leaves the temporary and NO published file, which is a state the
//  consultation on the resume path can name.
//
//  THE LENGTH VERIFICATION IS THE WRITER'S OWN, taken before the rename, and it
//  is what makes the publish decision a measurement rather than an assumption.
//  c.count is the number of values the traversal visited; the expected count for
//  the running model is computed independently by a count-only pass.  A
//  disagreement means the traversal did not emit what this model requires, and
//  the temporary is kept under the ".failed" suffix rather than published.
//
//  Deliberately NOT fsync: [Q465] fixed the durability primitive for this
//  package at the userspace flush and DECLINED fflush+fsync, on the ground that
//  every terminal failure the campaign observed was a scheduler walltime kill,
//  which is process death and is exactly the class a userspace flush covers.
//  A later node-crash or power-loss terminal failure revives that choice.
//
{
    FILE*  f;
    TSnapCtx c;
    TSnapshotShape shape;
    int    hdr[4];
    int    shp[SNAPSHOT_SHAPE_FIELDS];
    char*  manifest;
    char*  part;
    size_t partLen;
    long   nFields = 0;
    long   expected;
    int    mlen;

    // A REFUSAL IN THE COUNT PASS LEAVES THE ".failed" MARKER TOO.  The count
    // pass touches no file, so its error flag can be set only by a structural
    // guard rejecting the model state -- a defect in the model's allocation or
    // in this writer, not a storage problem.  Without a marker that refusal
    // would reach the next resume as a plain missing snapshot, which the
    // orchestration is allowed to resume past silently; with one, the resume
    // stops on it exactly as it stops on a failed write.  The file holds one
    // text line rather than snapshot bytes, so an operator can tell the two
    // causes apart by reading it.  It is created exclusively ("wx"): an
    // existing marker at this name is evidence too and is never overwritten.
    // If it cannot be created the refusal returns as before, unmarked.
    expected = snapshot_expectedValueCount(SNAPSHOT_MAX_STATS);
    if ( expected < 0 )
    {
        size_t refusedLen = strlen(path) + strlen(SNAPSHOT_FAILED_SUFFIX) + 1;
        char*  refused    = (char*) malloc(refusedLen);
        if ( refused )
        {
            FILE* rf;
            snprintf(refused, refusedLen, "%s%s", path, SNAPSHOT_FAILED_SUFFIX);
            rf = fopen(refused, "wx");
            if ( rf )
            {
                fputs(SNAPSHOT_REFUSED_TEXT "\n", rf);
                fclose(rf);
            }
            free(refused);
        }
        return 1;
    }

    manifest = snapshot_buildManifest(&nFields);
    if ( !manifest ) return 1;
    mlen = (int) strlen(manifest);

    // The temporary's name is the published name plus the suffix, so the two
    // sort together in a directory listing and an operator finding a leftover
    // knows exactly which checkpoint it belongs to.
    partLen = strlen(path) + strlen(SNAPSHOT_PART_SUFFIX) + 1;
    part    = (char*) malloc(partLen);
    if ( !part ) { free(manifest); return 1; }
    snprintf(part, partLen, "%s%s", path, SNAPSHOT_PART_SUFFIX);

    f = fopen(part, "wb");
    if ( !f ) { free(manifest); free(part); return 1; }

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
        fclose(f); remove(part); free(manifest); free(part); return 1;
    }
    free(manifest);

    memset(&c, 0, sizeof(c));
    c.mode = SNAP_WRITE;
    c.f    = f;
    snapshot_traverse(&c, SNAPSHOT_MAX_STATS);

    if ( fclose(f) != 0 || c.error || c.count != expected )
    {
        // THE LENGTH-CHECK ARM KEEPS ITS TEMPORARY, RENAMED TO ".failed".  This
        // arm is the writer's own verification saying the bytes it wrote are
        // wrong -- a failed fwrite (a full disk or an exceeded quota), an
        // fclose failure, or a value-count disagreement between the count pass
        // and this write pass.  It is NOT reached by a structural guard: every
        // structural guard already fired in the count pass above, which then
        // left its own one-line ".failed" and returned.  c.error cannot say
        // which of the three happened, so the bytes are evidence an operator
        // needs, and they are RETAINED rather than deleted.  The header-write,
        // open, allocation and rename arms leave no marker, because nothing
        // they wrote is a snapshot attempt worth inspecting.
        //
        // ".failed" and not ".killed", because ".killed" is written by the
        // RESUME side when it finds a surviving ".part" (a write cut off by a
        // kill); one name meaning two different causes would make the resume
        // refusal unable to say which one happened.  The resume side treats a
        // ".failed" as a stop, not as a reason to step back to an earlier
        // snapshot, and the published name is still never created here.
        //
        // If the rename itself fails the temporary is removed instead, and both
        // failures are reported: a ".part" left behind would be read by the
        // next resume as a write cut off by a kill, which is a wrong diagnosis.
        // remove's own failure is reported rather than swallowed, because a
        // directory that cannot be written is a different operator problem from
        // a traversal that cannot emit.
        size_t failedLen = strlen(path) + strlen(SNAPSHOT_FAILED_SUFFIX) + 1;
        char*  failed    = (char*) malloc(failedLen);
        int    kept      = 0;
        if ( failed )
        {
            snprintf(failed, failedLen, "%s%s", path, SNAPSHOT_FAILED_SUFFIX);
            if ( rename(part, failed) == 0 ) kept = 1;
            else
                fprintf(stderr,
                    "WARNING: could not keep the failed SWMM snapshot %s as %s "
                    "(%s); removing it instead\n", part, failed, strerror(errno));
            free(failed);
        }
        if ( !kept && remove(part) != 0 )
            fprintf(stderr,
                "WARNING: could not remove the partial SWMM snapshot %s (%s); "
                "it will not be loaded but it will not be reclaimed either\n",
                part, strerror(errno));
        free(part);
        return 1;
    }

    // PUBLISH.  rename within one directory is the step the whole guarantee
    // rests on; its atomicity on the campaign's filesystem is an OPEN ITEM the
    // shape carries and no measurement here establishes it.
    if ( rename(part, path) != 0 )
    {
        fprintf(stderr,
            "WARNING: could not publish the SWMM snapshot %s from its temporary "
            "%s (%s)\n", path, part, strerror(errno));
        if ( remove(part) != 0 )
            fprintf(stderr,
                "WARNING: could not remove the unpublished SWMM snapshot "
                "temporary %s (%s)\n", part, strerror(errno));
        free(part);
        return 1;
    }

    free(part);
    return 0;
}

//=============================================================================

int snapshot_load(const char* path, char* errMsg, int errMsgLen)
//
//  Input:   path      = snapshot file to read
//           errMsg    = buffer receiving a human-readable refusal reason
//           errMsgLen = its size
//  Output:  SNAPSHOT_OK, SNAPSHOT_ABSENT or SNAPSHOT_UNREADABLE
//  Purpose: restores SWMM's accumulators from a snapshot, REFUSING loudly and
//           specifically rather than restoring a file that does not describe
//           the running model.
//
//  Every refusal names the mismatching quantity. A generic "bad snapshot" would
//  send an operator to re-run from scratch without telling them which of the
//  .inp, the build precision or the solver version changed under them.
//
//  THE RETURN IS GRADUATED. It was a uniform 1 across eleven
//  refusal sites, so no caller could branch on absent-versus-unreadable and
//  every refusal degraded to the slow replay -- including the ones that mean a
//  real defect. Exactly ONE site returns SNAPSHOT_ABSENT: the fopen failure,
//  and only when errno says the file is not there. A fopen that fails for any
//  OTHER reason (a permission error, a directory in the way) is a defect
//  wearing absence's clothing and returns SNAPSHOT_UNREADABLE, because the
//  caller's whole reason for distinguishing the two is that absence is
//  LEGITIMATE and everything else is not.
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
    long  payloadStart = -1;
    long  expected     = -1;
    long  actual       = -1;
    int   rc = SNAPSHOT_UNREADABLE;

    if ( errMsg && errMsgLen > 0 ) errMsg[0] = '\0';

    f = fopen(path, "rb");
    if ( !f )
    {
        // ENOENT is the legitimate case the caller falls back on; anything
        // else is a defect and must not be laundered into a silent fallback.
        if ( errno == ENOENT )
        {
            if ( errMsg ) snprintf(errMsg, errMsgLen,
                "snapshot not found: %s", path);
            return SNAPSHOT_ABSENT;
        }
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot could not be opened (%s): %s", strerror(errno), path);
        return SNAPSHOT_UNREADABLE;
    }

    if ( fread(hdr, sizeof(int), 4, f) != 4 )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot header is short (file truncated): %s", path);
        fclose(f); return SNAPSHOT_UNREADABLE;
    }
    if ( hdr[0] != SNAPSHOT_MAGIC )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot has a bad magic (0x%08X, expected 0x%08X): %s",
            (unsigned) hdr[0], (unsigned) SNAPSHOT_MAGIC, path);
        fclose(f); return SNAPSHOT_UNREADABLE;
    }
    if ( hdr[1] != SNAPSHOT_VERSION )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot is version %d but this build reads version %d",
            hdr[1], SNAPSHOT_VERSION);
        fclose(f); return SNAPSHOT_UNREADABLE;
    }
    if ( hdr[2] != (int) sizeof(double) )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot stores %d-byte values but this build's double is %d bytes",
            hdr[2], (int) sizeof(double));
        fclose(f); return SNAPSHOT_UNREADABLE;
    }
    if ( fread(shp, sizeof(int), SNAPSHOT_SHAPE_FIELDS, f) != SNAPSHOT_SHAPE_FIELDS )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot shape block is short (file truncated): %s", path);
        fclose(f); return SNAPSHOT_UNREADABLE;
    }
    memcpy(&got, shp, sizeof(got));
    snapshot_currentShape(&want, SNAPSHOT_MAX_STATS);

#define SHAPE_CHECK(field, label)                                              \
    if ( got.field != want.field )                                             \
    {                                                                          \
        if ( errMsg ) snprintf(errMsg, errMsgLen,                              \
            "snapshot describes %d %s but this run has %d -- the .inp or the "  \
            "coupling changed between runs", got.field, label, want.field);    \
        fclose(f); return SNAPSHOT_UNREADABLE;                                                   \
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
            fclose(f); return SNAPSHOT_UNREADABLE;
        }
        if ( fread(manifest, 1, (size_t) hdr[3], f) != (size_t) hdr[3] )
        {
            if ( errMsg ) snprintf(errMsg, errMsgLen,
                "snapshot manifest is short (file truncated): %s", path);
            free(manifest); fclose(f); return SNAPSHOT_UNREADABLE;
        }
        manifest[hdr[3]] = '\0';
    }
    mine = snapshot_buildManifest(&nFields);
    if ( !mine )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen, "could not build this build's snapshot manifest");
        free(manifest); fclose(f); return SNAPSHOT_UNREADABLE;
    }
    if ( !manifest || strcmp(manifest, mine) != 0 )
    {
        if ( errMsg ) snprintf(errMsg, errMsgLen,
            "snapshot field manifest does not match this build's serializer -- "
            "the vendored EPA structs changed between the writing and reading "
            "builds; re-run from a clean start (checkpoint_id=0)");
        free(manifest); free(mine); fclose(f); return SNAPSHOT_UNREADABLE;
    }
    free(manifest); free(mine);

    // THE PAYLOAD EXTENT, measured before the read traversal starts, so the
    // refusal below can carry a NUMBER rather than a cause.
    //
    // This is the discriminator the old message had no way to supply.  The
    // retired wording -- "short or unreadable" -- is kept on ONE LINE here on
    // purpose: a quoted phrase wrapped across comment lines is invisible to a
    // line-anchored grep, which is how a post-condition certifying this phrase
    // absent came to read 0 on a file that still carried it.  It is true of a
    // file written by an older build, of a corrupt
    // file, and of a file whose writer was killed mid-traversal, and it tells
    // the three apart not at all -- which is measured rather than argued: a
    // size census over 856 campaign snapshots returned a single size class and
    // READ AS HEALTHY, because nothing in it compared any file against the
    // length the running model requires. A reader given expected-versus-actual
    // sees the class on sight.
    payloadStart = ftell(f);
    expected     = snapshot_expectedValueCount(SNAPSHOT_MAX_STATS);
    if ( payloadStart >= 0 && fseek(f, 0, SEEK_END) == 0 )
    {
        const long fileEnd = ftell(f);
        if ( fileEnd >= payloadStart ) actual = (fileEnd - payloadStart) / (long) sizeof(double);
        if ( fseek(f, payloadStart, SEEK_SET) != 0 )
        {
            if ( errMsg ) snprintf(errMsg, errMsgLen,
                "snapshot payload could not be positioned for reading: %s", path);
            fclose(f); return SNAPSHOT_UNREADABLE;
        }
    }

    memset(&c, 0, sizeof(c));
    c.mode = SNAP_READ;
    c.f    = f;
    snapshot_traverse(&c, SNAPSHOT_MAX_STATS);
    rc = c.error ? SNAPSHOT_UNREADABLE : SNAPSHOT_OK;

    // ASSERTS NO CAUSE.  It states what disagreed and by how much, and leaves
    // the cause to the reader who can see the directory.  Both figures are
    // reported even when one is unavailable (-1), because a missing figure is
    // itself information about which measurement failed.
    if ( rc && errMsg ) snprintf(errMsg, errMsgLen,
        "snapshot payload holds %ld value(s) and this model requires %ld "
        "(values read before the traversal stopped: %ld): %s",
        actual, expected, c.count, path);

    fclose(f);
    return rc;
}
