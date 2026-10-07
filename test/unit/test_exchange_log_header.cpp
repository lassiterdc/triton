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
#include <filesystem>
#include <sstream>
#include <unistd.h>
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

/** Lays down an arbitrary three-field header as raw bytes. Used to synthesise
 *  headers this build would never write -- a foreign precision, a V1 magic --
 *  without hand-constructing an exchange_log_header_t, so every header under test
 *  has been through the SHIPPED reader and carries a reader-produced `complete`. */
static std::string raw_header_bytes(int32_t magic, int32_t num_nodes, int32_t value_width)
{
    std::ostringstream os(std::ios::binary);
    os.write(reinterpret_cast<const char*>(&magic),       sizeof(int32_t));
    os.write(reinterpret_cast<const char*>(&num_nodes),   sizeof(int32_t));
    os.write(reinterpret_cast<const char*>(&value_width), sizeof(int32_t));
    return os.str();
}

/** raw_header_bytes() round-tripped through the shipped reader. */
static exchange_log_header_t hdr_of(int32_t magic, int32_t num_nodes, int32_t value_width)
{
    std::istringstream is(raw_header_bytes(magic, num_nodes, value_width), std::ios::binary);
    return read_exchange_log_header(is);
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
// T11 -- write/read round trip through the shipped layout.
//
// The stride the reader uses and the bytes the writer emits are single-sourced
// through EXCHANGE_LOG_HEADER_BYTES; this asserts they agree by construction, and
// that a clean header leaves the get pointer exactly at the first record.
// ---------------------------------------------------------------------------
static void T11_header_roundtrip_ok()
{
    const int32_t kNodes = 14;

    std::ostringstream os(std::ios::binary);
    write_exchange_log_header(os, kNodes);
    // A first record, so a stride error would show up as a non-EOF stream position.
    const value_t dt = (value_t)1.25;
    os.write(reinterpret_cast<const char*>(&dt), sizeof(value_t));

    std::istringstream is(os.str(), std::ios::binary);
    const exchange_log_header_t h = read_exchange_log_header(is);

    CHECK_EQ_I(h.magic,       EXCHANGE_LOG_MAGIC_V2,       "round-tripped magic");
    CHECK_EQ_I(h.num_nodes,   kNodes,                      "round-tripped node count");
    CHECK_EQ_I(h.value_width, (int32_t)sizeof(value_t),    "round-tripped stored width");

    CHECK(validate_exchange_log_header(h, kNodes, EXCHANGE_LOG_VALUE_WIDTH)
              == exchange_header_status::ok,
          "a header written by this build must validate against this build");

    CHECK_EQ_I(is.tellg(), (long long)EXCHANGE_LOG_HEADER_BYTES,
               "after reading the header the get pointer must sit at the first record");
}

// ---------------------------------------------------------------------------
// T1 -- design section 5 property 1: a side-file whose stored width disagrees is REFUSED.
//
// This is the defect itself. Before the guard, the only two validated fields were
// precision-independent, so the cross-precision file passed and then strode every
// record by the wrong number of bytes.
// ---------------------------------------------------------------------------
static void T1_stored_width_disagreement_is_refused()
{
    const int32_t kNodes = 14;
    const int32_t kThis  = EXCHANGE_LOG_VALUE_WIDTH;
    const int32_t kOther = (kThis == 8) ? 4 : 8;   // the other precision build's width

    // A well-formed V2 header from a build of the OTHER precision.
    const exchange_log_header_t foreign = hdr_of(EXCHANGE_LOG_MAGIC_V2, kNodes, kOther);

    CHECK(validate_exchange_log_header(foreign, kNodes, kThis)
              == exchange_header_status::value_width_mismatch,
          "a V2 header storing the other precision's width must be refused");

    // ... and the same header IS accepted by a build of the width it records, so the
    // guard keys on disagreement rather than on a hardcoded width.
    CHECK(validate_exchange_log_header(foreign, kNodes, kOther)
              == exchange_header_status::ok,
          "the guard must key on disagreement, not on a hardcoded width");

    CHECK(kThis == 4 || kThis == 8,
          "value_t is expected to be float or double; a third width needs review here");
}

// ---------------------------------------------------------------------------
// T12 -- node-count disagreement keeps its own distinct diagnostic.
// ---------------------------------------------------------------------------
static void T12_node_count_mismatch_refused()
{
    const exchange_log_header_t h = hdr_of(EXCHANGE_LOG_MAGIC_V2, 14, EXCHANGE_LOG_VALUE_WIDTH);

    CHECK(validate_exchange_log_header(h, 13, EXCHANGE_LOG_VALUE_WIDTH)
              == exchange_header_status::node_count_mismatch,
          "a node-count disagreement must be refused as node_count_mismatch");
}

// ---------------------------------------------------------------------------
// T13 -- a short or empty header presents as bad_magic, not as a stream error.
//
// Any field that cannot be read in full is left at 0, and 0 is not the V2 magic.
// ---------------------------------------------------------------------------
static void T13_short_header_is_bad_magic()
{
    const int32_t kNodes = 14;

    struct { const char* what; std::size_t nbytes; } cases[] = {
        { "empty file",          0 },
        { "1 byte",              1 },
        { "magic only",          4 },
        { "magic + node count",  8 },
        { "header less 1 byte",  EXCHANGE_LOG_HEADER_BYTES - 1 },
    };

    std::ostringstream os(std::ios::binary);
    write_exchange_log_header(os, kNodes);
    const std::string full = os.str();

    for (const auto& c : cases) {
        std::istringstream is(full.substr(0, c.nbytes), std::ios::binary);
        const exchange_log_header_t h = read_exchange_log_header(is);
        const exchange_header_status st =
            validate_exchange_log_header(h, kNodes, EXCHANGE_LOG_VALUE_WIDTH);
        CHECK(!h.complete, c.what);
        CHECK(st == exchange_header_status::bad_magic, c.what);
    }
}

// ---------------------------------------------------------------------------
// T14 -- design section 5 property 3: THE VALIDATION ORDER IS THE SPECIFICATION.
//
// Each case is wrong in MORE THAN ONE way, so only the ordering decides which
// status comes back. Reorder validate_exchange_log_header() and this test fails;
// a single-defect case would pass under any order and prove nothing.
// ---------------------------------------------------------------------------
static void T14_validation_order_magic_then_nodes_then_width()
{
    const int32_t kNodes = 14;
    const int32_t kThis  = EXCHANGE_LOG_VALUE_WIDTH;
    const int32_t kOther = (kThis == 8) ? 4 : 8;

    // Wrong in all three ways -> the FIRST check must win.
    const exchange_log_header_t all_wrong = hdr_of(EXCHANGE_LOG_MAGIC, kNodes + 1, kOther);
    CHECK(validate_exchange_log_header(all_wrong, kNodes, kThis)
              == exchange_header_status::bad_magic,
          "magic is checked first: a legacy file must report bad_magic, never a width "
          "or node-count error");

    // Magic right, the other two wrong -> the node count must win over the width.
    const exchange_log_header_t nodes_and_width = hdr_of(EXCHANGE_LOG_MAGIC_V2, kNodes + 1, kOther);
    CHECK(validate_exchange_log_header(nodes_and_width, kNodes, kThis)
              == exchange_header_status::node_count_mismatch,
          "node count is checked before stored width");

    // Only the width wrong -> the width check is reached and fires.
    const exchange_log_header_t width_only = hdr_of(EXCHANGE_LOG_MAGIC_V2, kNodes, kOther);
    CHECK(validate_exchange_log_header(width_only, kNodes, kThis)
              == exchange_header_status::value_width_mismatch,
          "stored width is checked last and does fire when it is the only disagreement");
}

// ---------------------------------------------------------------------------
// Legacy (V1) fixtures.
//
// A V1 side-file is exactly what open_exchange_log_truncate() wrote before the
// bump: a two-field header {magic, num_nodes} followed by records. It has NO third
// header field, which is the whole reason the magic had to move.
// ---------------------------------------------------------------------------

/** Bytes of a V1 side-file: the two-field header plus `n_records` records. The
 *  first record's leading value_t is `first_dt`, so a caller can control exactly
 *  what a three-field reader would mistake for the stored width. */
static std::string v1_side_file_bytes(int32_t num_nodes, int n_records, value_t first_dt)
{
    std::ostringstream os(std::ios::binary);
    const int32_t magic = EXCHANGE_LOG_MAGIC;
    os.write(reinterpret_cast<const char*>(&magic),     sizeof(int32_t));
    os.write(reinterpret_cast<const char*>(&num_nodes), sizeof(int32_t));

    for (int r = 0; r < n_records; ++r) {
        const value_t dt = (r == 0) ? first_dt : (value_t)1.0;
        os.write(reinterpret_cast<const char*>(&dt), sizeof(value_t));
        const std::vector<value_t> q((std::size_t)num_nodes, (value_t)0.5);
        os.write(reinterpret_cast<const char*>(q.data()), sizeof(value_t) * (std::size_t)num_nodes);
    }
    return os.str();
}

/** A per-process temp path, so a concurrent build on the same filesystem cannot
 *  collide with this one. */
static std::filesystem::path tmp_path(const char* stem)
{
    return std::filesystem::temp_directory_path() /
           (std::string("triton_wp0a_") + stem + "_" +
            std::to_string((long long)::getpid()) + ".bin");
}

// ---------------------------------------------------------------------------
// T2 -- design section 5 properties 2 and 3, and WP-0A chunk 3's whole claim:
//       a legacy file lands on the BAD-MAGIC path.
//
// Exercised through a real file on disk and a real std::ifstream, the same way
// replay_exchange_history() reaches the header -- not only through a stringstream.
//
// This test FAILS if the legacy path ever silently accepted a V1 header: `ok`,
// `node_count_mismatch` and `value_width_mismatch` are each a distinct failure
// here, so the assertion cannot be satisfied by any status but bad_magic.
// ---------------------------------------------------------------------------
static void T2_v1_legacy_file_lands_on_bad_magic()
{
    const int32_t kNodes = 14;
    const std::filesystem::path f = tmp_path("v1");

    {
        std::ofstream os(f, std::ios::binary | std::ios::trunc);
        const std::string bytes = v1_side_file_bytes(kNodes, 3, (value_t)0.25);
        os.write(bytes.data(), (std::streamsize)bytes.size());
    }

    std::ifstream in(f, std::ios::binary);
    CHECK(in.is_open(), "the V1 fixture file must be readable");

    const exchange_log_header_t h = read_exchange_log_header(in);
    const exchange_header_status st =
        validate_exchange_log_header(h, kNodes, EXCHANGE_LOG_VALUE_WIDTH);

    CHECK(st == exchange_header_status::bad_magic,
          "a V1 legacy side-file must be refused as bad_magic");
    CHECK(st != exchange_header_status::ok,
          "a V1 legacy side-file must NOT be accepted");
    CHECK(st != exchange_header_status::node_count_mismatch,
          "a V1 legacy side-file must not be blamed on the .inp / coupling");
    CHECK(st != exchange_header_status::value_width_mismatch,
          "a V1 legacy side-file must not be reported as a wrong stored width");

    // The V1 magic survives into the header so replay_exchange_history() can emit
    // the legacy-specific hint rather than a generic parse failure. The node count
    // is genuinely correct here, which is exactly why a weaker ordering would have
    // fallen through to the width check.
    CHECK_EQ_I(h.magic, EXCHANGE_LOG_MAGIC, "the retained V1 magic must reach the caller");
    CHECK_EQ_I(h.num_nodes, kNodes, "a V1 file's second field really is the node count");
    CHECK(h.complete, "a V1 file with records is long enough to fill three fields");

    in.close();
    std::error_code ec;
    std::filesystem::remove(f, ec);
}

// ---------------------------------------------------------------------------
// T15 -- the adversarial case the magic bump exists for.
//
// A V1 file's bytes 8..11 are the first record's leading value_t, not a width. This
// fixture chooses a first dt whose low four bytes ARE exactly this build's
// sizeof(value_t), so the synthetic "width field" agrees and the node count agrees:
// a validator that checked width without first checking magic would return `ok` and
// replay the file with a wrong stride and no error anywhere.
//
// Checking magic first is the only thing standing between that file and swmm_step.
// ---------------------------------------------------------------------------
static void T15_v1_third_field_is_record_bytes_not_a_width()
{
    const int32_t kNodes = 14;
    const int32_t kThis  = EXCHANGE_LOG_VALUE_WIDTH;

    // Build a first dt whose leading four bytes read back as the int32 kThis.
    value_t dt = (value_t)0.0;
    std::memcpy(&dt, &kThis, sizeof(int32_t));

    const std::string bytes = v1_side_file_bytes(kNodes, 2, dt);
    std::istringstream is(bytes, std::ios::binary);
    const exchange_log_header_t h = read_exchange_log_header(is);

    // The trap is real: field 2 of this V1 file reads back as a plausible width.
    CHECK_EQ_I(h.value_width, kThis,
               "the fixture must actually reproduce the trap -- a V1 file whose "
               "third int32 coincides with this build's value_t width");
    CHECK_EQ_I(h.num_nodes, kNodes, "and whose node count agrees too");

    // Despite every later field agreeing, the magic decides.
    CHECK(validate_exchange_log_header(h, kNodes, kThis) == exchange_header_status::bad_magic,
          "a V1 file must be refused on magic even when its record bytes happen to "
          "read back as a matching width and node count");
}

// ---------------------------------------------------------------------------
// T16 -- the converse: a V2 file is never mistaken for legacy.
//
// Chunk 3's claim is two-sided, and T2 only covers one side. Without this, a
// reader that returned bad_magic unconditionally would pass every legacy test.
// ---------------------------------------------------------------------------
static void T16_v2_file_is_not_mistaken_for_legacy()
{
    const int32_t kNodes = 14;
    const std::filesystem::path f = tmp_path("v2");

    {
        std::ofstream os(f, std::ios::binary | std::ios::trunc);
        write_exchange_log_header(os, kNodes);
        const value_t dt = (value_t)0.25;
        os.write(reinterpret_cast<const char*>(&dt), sizeof(value_t));
        const std::vector<value_t> q((std::size_t)kNodes, (value_t)0.5);
        os.write(reinterpret_cast<const char*>(q.data()), sizeof(value_t) * (std::size_t)kNodes);
    }

    std::ifstream in(f, std::ios::binary);
    const exchange_log_header_t h = read_exchange_log_header(in);

    CHECK(h.magic != EXCHANGE_LOG_MAGIC,
          "a file this build wrote must not carry the legacy magic");
    CHECK(validate_exchange_log_header(h, kNodes, EXCHANGE_LOG_VALUE_WIDTH)
              == exchange_header_status::ok,
          "a file this build wrote must validate against this build");

    in.close();
    std::error_code ec;
    std::filesystem::remove(f, ec);
}

// ---------------------------------------------------------------------------
// Registry / driver
// ---------------------------------------------------------------------------
struct test_entry { const char* name; void (*fn)(); };

static const test_entry kTests[] = {
    { "T9_magic_v2_differs_from_retained_v1", &T9_magic_v2_differs_from_retained_v1 },
    { "T10_header_write_layout",              &T10_header_write_layout },
    { "T11_header_roundtrip_ok",              &T11_header_roundtrip_ok },
    { "T1_stored_width_disagreement_is_refused", &T1_stored_width_disagreement_is_refused },
    { "T12_node_count_mismatch_refused",      &T12_node_count_mismatch_refused },
    { "T13_short_header_is_bad_magic",        &T13_short_header_is_bad_magic },
    { "T14_validation_order_magic_then_nodes_then_width", &T14_validation_order_magic_then_nodes_then_width },
    { "T2_v1_legacy_file_lands_on_bad_magic", &T2_v1_legacy_file_lands_on_bad_magic },
    { "T15_v1_third_field_is_record_bytes_not_a_width", &T15_v1_third_field_is_record_bytes_not_a_width },
    { "T16_v2_file_is_not_mistaken_for_legacy", &T16_v2_file_is_not_mistaken_for_legacy },
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
