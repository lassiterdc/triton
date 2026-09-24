/** @file test_exchange_log_header.cpp
 *  @brief Tier-1 tests for the exchange-replay side-file header (WP-0A).
 *
 *  No solver run, no cluster, no SWMM call: these exercise the free functions in
 *  swmm_triton.h that are the single expression of the side-file's header layout.
 *
 *  The defect under test: `replay_exchange_history` derives its record stride from
 *  `sizeof(value_t)` in the RUNNING build, while the V1 header carried only `magic`
 *  and `num_nodes` -- both precision-independent. A side-file written by one
 *  precision build and replayed by the other therefore passed both guards and then
 *  strode every record by the wrong number of bytes, replaying garbage exchange
 *  values through `swmm_step` with no error raised anywhere.
 *
 *  Each test function is selectable by name from argv[1] so CTest can register one
 *  node id per property; with no argument every test runs.
 */

// swmm_triton.h is NOT independently includable at this commit: it compiles only
// as part of src/main.cpp's include chain. Three separate transitive dependencies,
// each measured from a compile failure rather than assumed:
//   <iostream>       -- src/matrix.h (via mpi_utils.h) uses std::cerr/std::endl
//                       without including it.
//   "mpi.h"          -- constants.h / mpi_utils.h / Ensify.h name MPI_DOUBLE,
//                       MPI_Request, MPI_COMM_WORLD without including it.
//   "kokkos_utils.h" -- swmm_triton.h #includes it INSIDE a function body
//                       (compute_swmm_triton_exchange), which is well-formed only
//                       because triton.h has already consumed its include guard.
// This prefix reproduces main.cpp's order so the gap stays out of this test's
// scope: none of matrix.h, constants.h or the swmm_triton.h include site is
// WP-0A's to edit, and all three are reported as findings instead.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace SWMM_triton;

// ---------------------------------------------------------------------------
// Minimal harness
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Reads one little-/native-endian int32 out of a byte buffer at `off`. */
static int32_t i32_at(const std::string& bytes, std::size_t off)
{
    int32_t v = 0;
    std::memcpy(&v, bytes.data() + off, sizeof(int32_t));
    return v;
}

// ---------------------------------------------------------------------------
// T9 -- design section 5 property 2: the magic is bumped and the OLD symbol RETAINED.
//
// The bump is what lets the reader distinguish the two header generations before it
// interprets a third field that a V1 file does not carry. If the two symbols were
// ever collapsed onto one value, every legacy file would be accepted as V2 and its
// first record's leading value_t would be read as the stored width.
// ---------------------------------------------------------------------------
static_assert(EXCHANGE_LOG_MAGIC_V2 != EXCHANGE_LOG_MAGIC,
              "V2 magic must differ from the retained V1 magic");

static void T9_magic_v2_differs_from_retained_v1()
{
    CHECK(EXCHANGE_LOG_MAGIC_V2 != EXCHANGE_LOG_MAGIC,
          "EXCHANGE_LOG_MAGIC_V2 must differ from the retained EXCHANGE_LOG_MAGIC");
    CHECK_EQ_I(EXCHANGE_LOG_MAGIC, 0x53574D4D,
               "the retained V1 magic must keep its historical value (ASCII \"SWMM\")");
    CHECK_EQ_I(EXCHANGE_LOG_MAGIC_V2, 0x53574D32,
               "the V2 magic must be ASCII \"SWM2\"");
}

// ---------------------------------------------------------------------------
// T10 -- design section 5 property 1, write side: the header records the stored width.
//
// Asserts the emitted byte layout directly rather than round-tripping it, so a
// silent field reorder is caught even if the reader is reordered to match.
// ---------------------------------------------------------------------------
static void T10_header_write_layout()
{
    const int32_t kNodes = 14;

    std::ostringstream os(std::ios::binary);
    write_exchange_log_header(os, kNodes);
    const std::string bytes = os.str();

    CHECK_EQ_I(bytes.size(), EXCHANGE_LOG_HEADER_BYTES,
               "the writer must emit exactly EXCHANGE_LOG_HEADER_BYTES");
    CHECK_EQ_I(EXCHANGE_LOG_HEADER_BYTES, 3 * sizeof(int32_t),
               "the V2 header is three int32 fields");

    CHECK_EQ_I(i32_at(bytes, 0), EXCHANGE_LOG_MAGIC_V2, "field 0 is the V2 magic");
    CHECK_EQ_I(i32_at(bytes, 4), kNodes,                "field 1 is the node count");
    CHECK_EQ_I(i32_at(bytes, 8), (int32_t)sizeof(value_t),
               "field 2 is the stored value_t width of the WRITING build");

    CHECK_EQ_I(EXCHANGE_LOG_VALUE_WIDTH, (int32_t)sizeof(value_t),
               "EXCHANGE_LOG_VALUE_WIDTH must track this build's value_t");
}

// ---------------------------------------------------------------------------
// Registry / driver
// ---------------------------------------------------------------------------
struct test_entry { const char* name; void (*fn)(); };

static const test_entry kTests[] = {
    { "T9_magic_v2_differs_from_retained_v1", &T9_magic_v2_differs_from_retained_v1 },
    { "T10_header_write_layout",              &T10_header_write_layout },
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
