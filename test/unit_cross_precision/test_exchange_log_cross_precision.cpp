/** @file test_exchange_log_cross_precision.cpp
 *  @brief WP-0A REVIEW test: the stored-width guard across a REAL precision boundary.
 *
 *  WHAT THIS FALSIFIES, AND WHY THE SHIPPED SUITE CANNOT.
 *
 *  Every assertion in test/unit/test_exchange_log_header.cpp is evaluated inside ONE
 *  binary, at ONE value_t width. The cross-precision cases there are SYNTHESISED: a
 *  foreign header is fabricated as an exchange_log_header_t (or as hand-laid bytes)
 *  whose `value_width` field is set to the width this build does not have. That
 *  establishes that validate_exchange_log_header() returns the right enum for a
 *  struct, which is a property of the validator.
 *
 *  It does NOT establish the property the defect is about: that a side-file PRODUCED
 *  by the shipped writer in a build of one precision is REFUSED by the shipped reader
 *  in a build of the other. That composition spans two binaries -- one compiled with
 *  -DUSE_SINGLE_PRECISION and one without -- so no single-binary test can reach it.
 *  Everything between the writer's `out.write` and the reader's `in.read` is untested
 *  by the shipped suite: whether the writer actually STAMPS sizeof(value_t) rather
 *  than a constant, whether the header's three int32 fields are width-invariant so
 *  the two builds agree on where the third field begins, and whether the reader takes
 *  its expectation from its OWN value_t rather than from the file.
 *
 *  This driver supplies that composition. It is run by run_cross_precision_check.sh,
 *  which builds this file twice and crosses the artifacts.
 *
 *  It additionally re-evaluates the PRE-FIX predicate (magic + node count only) on the
 *  very same bytes, so the test does not merely assert that the guard fires -- it
 *  exhibits the file that the two-field validator would have ACCEPTED and then strode
 *  wrongly. A test that only asserts the new return value cannot distinguish a guard
 *  that closes the defect from a guard that fires on files that were never dangerous.
 *
 *  Modes:
 *    write        <path> <nodes> <records>   write a real side-file with the shipped writer
 *    read         <path> <nodes> <expect>    read+validate; expect = ok|width|magic|nodes
 *    forge-width  <path> <width>             rewrite ONLY the stored-width header field
 *
 *  Exit 0 on the expected outcome, 1 otherwise. Every line it prints names the build's
 *  own width, so a crossed pair is legible in the log without the caller's bookkeeping.
 */

// swmm_triton.h is not independently includable at this commit; this prefix
// reproduces main.cpp's include order. See the note in test/unit/.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace SWMM_triton;

static int fail(const char* msg)
{
    std::fprintf(stderr, "  FAIL [width=%d]: %s\n", (int)EXCHANGE_LOG_VALUE_WIDTH, msg);
    return 1;
}

/** Writes a side-file exactly as open_exchange_log_truncate() + log_exchange_step() do:
 *  the shipped three-field header, then `records` records of {value_t dt, value_t q[nodes]}. */
static int do_write(const char* path, int nodes, int records)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return fail("could not open side-file for writing");

    write_exchange_log_header(out, (int32_t)nodes);

    std::vector<value_t> q((std::size_t)nodes);
    for (int r = 0; r < records; ++r) {
        const value_t dt = (value_t)(0.5 + r);
        for (int i = 0; i < nodes; ++i) q[(std::size_t)i] = (value_t)(r * 100 + i);
        out.write(reinterpret_cast<const char*>(&dt), sizeof(value_t));
        out.write(reinterpret_cast<const char*>(q.data()), sizeof(value_t) * (std::size_t)nodes);
    }
    out.close();

    std::fprintf(stdout, "wrote %s : width=%d nodes=%d records=%d header_bytes=%zu record_bytes=%zu\n",
                 path, (int)EXCHANGE_LOG_VALUE_WIDTH, nodes, records,
                 (std::size_t)EXCHANGE_LOG_HEADER_BYTES,
                 (std::size_t)(sizeof(value_t) * (std::size_t)(1 + nodes)));
    return 0;
}

/** Rewrites ONLY the third header int32 (value_width) in an existing side-file.
 *
 *  This forges, at the BYTE level on disk, the file a coupled build of the other
 *  precision would have written -- without requiring that build to exist. It is the
 *  fallback leg: the true two-binary cross is the stronger evidence, but it is
 *  currently unbuildable (see the harness note on USE_SINGLE_PRECISION), and a forged
 *  file still crosses the shipped writer/reader boundary on real disk bytes, which no
 *  test in test/unit/ does. It rewrites the stored width and NOTHING else, so the
 *  magic and node count remain exactly what the shipped writer emitted -- which is
 *  what makes the pre-fix predicate accept it.
 */
static int do_forge_width(const char* path, int newwidth)
{
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!f.is_open()) return fail("could not open side-file to forge its width");
    const int32_t w = (int32_t)newwidth;
    f.seekp((std::streamoff)(2 * sizeof(int32_t)), std::ios::beg);
    f.write(reinterpret_cast<const char*>(&w), sizeof(w));
    f.close();
    std::fprintf(stdout, "forged %s : stored width field rewritten to %d\n", path, newwidth);
    return 0;
}

static const char* status_name(exchange_header_status s)
{
    switch (s) {
        case exchange_header_status::ok:                   return "ok";
        case exchange_header_status::bad_magic:            return "magic";
        case exchange_header_status::node_count_mismatch:  return "nodes";
        case exchange_header_status::value_width_mismatch: return "width";
    }
    return "?";
}

static int do_read(const char* path, int nodes, const std::string& expect)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return fail("side-file not found for reading");

    const exchange_log_header_t hdr = read_exchange_log_header(in);
    const exchange_header_status st =
        validate_exchange_log_header(hdr, (int32_t)nodes, EXCHANGE_LOG_VALUE_WIDTH);
    in.close();

    std::fprintf(stdout,
                 "read  %s : this_build_width=%d file_records_width=%d file_nodes=%d -> %s (expected %s)\n",
                 path, (int)EXCHANGE_LOG_VALUE_WIDTH, (int)hdr.value_width,
                 (int)hdr.num_nodes, status_name(st), expect.c_str());

    if (expect != status_name(st)) return fail("validator returned the wrong status");

    // --- the pre-fix predicate, on the SAME bytes -----------------------------------
    // Before the bump the header was {magic, num_nodes} and validation checked exactly
    // those two. Re-evaluate that predicate here. On a crossed file it must ACCEPT --
    // which is the defect -- and the record stride this build would then have used must
    // disagree with the stride the file was written at. If the pre-fix predicate
    // REJECTED the crossed file, the guard would be firing on a file that was never
    // dangerous, and this test would be certifying the wrong property.
    if (expect == "width") {
        const bool prefix_two_field_accepts =
            (hdr.magic == EXCHANGE_LOG_MAGIC_V2 || hdr.magic == EXCHANGE_LOG_MAGIC) &&
            (hdr.num_nodes == (int32_t)nodes);
        if (!prefix_two_field_accepts)
            return fail("pre-fix two-field predicate did not accept the crossed file; "
                        "this file does not exhibit the defect and the test proves nothing");

        const std::size_t this_stride  = sizeof(value_t) * (std::size_t)(1 + nodes);
        const std::size_t file_stride  = (std::size_t)hdr.value_width * (std::size_t)(1 + nodes);
        if (this_stride == file_stride)
            return fail("strides agree; the crossed file is not actually cross-precision");

        std::fprintf(stdout,
                     "      pre-fix predicate ACCEPTS this file; stride would be %zu bytes "
                     "against the %zu bytes it was written at -- the defect, now refused.\n",
                     this_stride, file_stride);
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s write <path> <nodes> <records>\n"
                     "       %s read  <path> <nodes> <ok|width|magic|nodes>\n", argv[0], argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "write") {
        if (argc < 5) return 2;
        return do_write(argv[2], std::atoi(argv[3]), std::atoi(argv[4]));
    }
    if (mode == "read")  return do_read(argv[2], std::atoi(argv[3]), argv[4]);
    if (mode == "forge-width") return do_forge_width(argv[2], std::atoi(argv[3]));
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
