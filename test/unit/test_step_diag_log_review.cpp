/** @file test_step_diag_log_review.cpp
 *  @brief Reviewer test for chunk 2a.2: the capture-file positioner must not let the
 *         content of a record's count field decide how much memory a resume allocates.
 *
 *  The property. diag_log::counted_block_record_reader (the T1a positioner,
 *  parameters 48, 20, 4) reads the int32 at byte 20 of a record's fixed part and
 *  allocates count * item_bytes before it knows whether that many bytes exist in the
 *  file. A record whose count field is larger than the rest of the file therefore
 *  asks the allocator for up to 8 GB (count 0x7fffffff, 4-byte items) and zero-fills
 *  it, where the correct result is "incomplete record, stop here" (the same answer
 *  the reader already gives for a negative count and for a short block). A resume
 *  that meets such a file must keep the complete records before it and truncate the
 *  rest, never run the node out of memory.
 *
 *  Class. Every T1a-shaped file whose final record carries a count field greater than
 *  the bytes that follow it. Reachability: a killed run's prefix cannot produce one (an
 *  append-only stream's prefix keeps the count field either whole and true, or cut,
 *  and a cut field makes the fixed read fail first); a count field that is not what
 *  the writer wrote reaches it, which is file corruption (a node crash that leaves
 *  stale blocks in the extended file, a damaged copy, a file of another layout whose
 *  header happened to validate). This is therefore a robustness defect of the resume
 *  path, not a defect a clean kill-and-resume run exposes.
 *
 *  Both-states anchor. The assertions turn on the reader's return value and on the
 *  process surviving under an address-space cap, which any implementation of the
 *  positioner has before and after a change of wording.
 *
 *  What kills it. The tip's reader, which allocates the block before the read: under the
 *  cap, std::bad_alloc for counts 0x10000000 and 0x7fffffff. What keeps it green: any
 *  reader that never allocates more than the bytes that exist, for example one that
 *  compares the block size with the stream's remaining length before it allocates.
 *  A second correct implementation, differing in the thing the test asserts on: a
 *  reader that reads the block in fixed-size chunks and stops at the first short read.
 *  It allocates 64 KB at most, returns false for every count below, and keeps the
 *  control below green.
 *
 *  Controls that must stay green on every implementation: a whole record with count 3
 *  is read (a reader that refuses every block fails), and count 1 << 20 (4 MB, below
 *  the cap) on a file that ends sooner returns false.
 *
 *  Not established: this file does not show that a real filesystem produces such a
 *  count after a crash, and it does not run the solver. It is a chunk test of chunk
 *  2a.2; the module whose behaviour it guards is diag_log.h (the positioners), and it
 *  is to be promoted into a module test at the landing or deleted with its passing
 *  evidence kept. It carries no stored blob and no absolute count of solver output.
 *
 *  Registration: not in test/unit/CMakeLists.txt (the reviewer commits nothing); the
 *  coordinator's registration is one add_executable linking kokkos and swmm5 like
 *  test_step_diag_log, and one add_test per case name in kTests.
 */

// The include prefix of test_step_diag_log.cpp: the headers compile only after the
// include chain main.cpp gives them.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

namespace fs = std::filesystem;

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

// --- helpers -----------------------------------------------------------------

/// The T1a parameters: fixed 48 bytes, count at byte 20, 4-byte items.
static diag_log::record_reader t1a_reader()
{
    return diag_log::counted_block_record_reader(48, 20, 4);
}

/// One T1a-shaped fixed part: stamp, dt, trials, count, then three doubles, 48 bytes.
static std::vector<char> t1a_fixed(double stamp, int32_t count)
{
    std::vector<char> b(48, 0);
    const double dt = 1.0;
    const int32_t trials = 1;
    std::memcpy(b.data() + 0, &stamp, 8);
    std::memcpy(b.data() + 8, &dt, 8);
    std::memcpy(b.data() + 16, &trials, 4);
    std::memcpy(b.data() + 20, &count, 4);
    return b;
}

/// A fresh per-process directory under the system temporary directory, created on first use.
static std::string scratch_dir()
{
    static std::string dir;
    if (dir.empty()) {
        dir = (fs::temp_directory_path() / ("test_step_diag_log_review_" + std::to_string((long)::getpid()))).string();
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    return dir;
}

/// A file of a valid T1a header, one whole record stamped 100 s with 3 ordinals, and
/// then `tail` raw bytes. Returns the path.
static std::string write_file(const std::string& name, const std::vector<char>& tail)
{
    const std::string path = scratch_dir() + "/" + name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    diag_log::write_header(f, diag_log::MAGIC_SWMM_STEP_DIAG, 0);
    const std::vector<char> rec = t1a_fixed(100.0, 3);
    f.write(rec.data(), (std::streamsize)rec.size());
    const int32_t ords[3] = {0, 1, 2};
    f.write(reinterpret_cast<const char*>(ords), sizeof(ords));
    f.write(tail.data(), (std::streamsize)tail.size());
    return path;
}

/// The current virtual size of the process in bytes (field 1 of /proc/self/statm).
static unsigned long long vm_size_bytes()
{
    std::ifstream f("/proc/self/statm");
    unsigned long long pages = 0;
    f >> pages;
    return pages * (unsigned long long)::sysconf(_SC_PAGESIZE);
}

/// Caps the address space to 256 MB above what the process holds now, so a request for
/// a block of a gigabyte or more fails at once instead of zero-filling node memory.
/// Returns false when the cap could not be set (the test then cannot discriminate).
static bool cap_address_space(rlimit& saved)
{
    if (::getrlimit(RLIMIT_AS, &saved) != 0) return false;
    rlimit lim = saved;
    lim.rlim_cur = (rlim_t)(vm_size_bytes() + 256ULL * 1024ULL * 1024ULL);
    return ::setrlimit(RLIMIT_AS, &lim) == 0;
}

/// What reading the second record did: threw std::bad_alloc, returned, and what it returned.
struct read_outcome
{
    bool threw = false;
    bool returned = false;
    bool ok = false;
};

/// Runs the T1a reader on the record that follows the header and the whole first record
/// of `path`, and reports whether it threw std::bad_alloc or returned.
static read_outcome read_second_record(const std::string& path)
{
    read_outcome out;
    std::ifstream in(path, std::ios::binary);
    diag_log::read_header(in);
    double stamp = 0.0;
    t1a_reader()(in, stamp);   // the whole first record
    try {
        out.ok = t1a_reader()(in, stamp);
        out.returned = true;
    } catch (const std::bad_alloc&) {
        out.threw = true;
    }
    return out;
}

// --- tests -------------------------------------------------------------------

/// R1. The reader answers "incomplete record" for a count field larger than the rest of
/// the file, without allocating the block. Class: counts 1 << 20, 1 << 28, 0x7fffffff.
static void R1_garbage_count_is_an_incomplete_record_not_an_allocation()
{
    const int32_t counts[] = {1 << 20, 1 << 28, 0x7fffffff};
    for (int32_t c : counts) {
        const std::string path = write_file("r1_" + std::to_string(c) + ".bin", t1a_fixed(200.0, c));
        rlimit saved;
        const bool capped = cap_address_space(saved);
        CHECK(capped, "the address-space cap was set, so a large allocation fails instead of filling memory");
        const read_outcome o = read_second_record(path);
        if (capped) ::setrlimit(RLIMIT_AS, &saved);
        CHECK(!o.threw, "the reader did not ask the allocator for the block a garbage count names");
        CHECK(o.returned && !o.ok, "a count larger than the remaining bytes is an incomplete record (returns false)");
    }
}

/// R2. The positioner keeps the complete records before a garbage-count record and cuts
/// the file there, so a resume can continue. Class: the same counts, through position_binary.
static void R2_position_keeps_the_records_before_a_garbage_count()
{
    const int32_t counts[] = {1 << 20, 1 << 28, 0x7fffffff};
    const std::uintmax_t first_record_end = diag_log::HEADER_BYTES + 48 + 3 * 4;
    for (int32_t c : counts) {
        const std::string path = write_file("r2_" + std::to_string(c) + ".bin", t1a_fixed(200.0, c));
        rlimit saved;
        const bool capped = cap_address_space(saved);
        CHECK(capped, "the address-space cap was set");
        bool threw = false;
        diag_log::position_result r;
        try {
            r = diag_log::position_binary(path, diag_log::MAGIC_SWMM_STEP_DIAG, 0, t1a_reader(), 1000.0);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        if (capped) ::setrlimit(RLIMIT_AS, &saved);
        CHECK(!threw, "position_binary did not ask the allocator for the block a garbage count names");
        if (!threw) {
            CHECK(r.status == diag_log::position_status::ok, "the file is positioned");
            CHECK(r.records == 1, "the one complete record before the garbage count is kept");
            CHECK(r.kept_bytes == first_record_end, "the file is cut at the end of that record");
        }
    }
}

/// R3. Controls, green on every implementation: a whole record with count 3 and a
/// zero-count record are read, so a reader that refuses every block fails here.
static void R3_whole_blocks_are_still_read()
{
    const std::string path = write_file("r3.bin", std::vector<char>());
    std::ifstream in(path, std::ios::binary);
    diag_log::read_header(in);
    double stamp = 0.0;
    CHECK(t1a_reader()(in, stamp), "a whole record with a 3-ordinal block is read");
    CHECK(stamp == 100.0, "its stamp is the first 8 bytes");

    std::vector<char> zero = t1a_fixed(150.0, 0);
    const std::string p2 = write_file("r3b.bin", zero);
    std::ifstream in2(p2, std::ios::binary);
    diag_log::read_header(in2);
    t1a_reader()(in2, stamp);
    CHECK(t1a_reader()(in2, stamp), "a record with an empty block is read");
    CHECK(stamp == 150.0, "its stamp is read");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"R1_garbage_count_is_an_incomplete_record_not_an_allocation", R1_garbage_count_is_an_incomplete_record_not_an_allocation},
    {"R2_position_keeps_the_records_before_a_garbage_count", R2_position_keeps_the_records_before_a_garbage_count},
    {"R3_whole_blocks_are_still_read", R3_whole_blocks_are_still_read},
};

/** Runs the test named in argv[1], or every test; exits 1 on any failure. */
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
    fs::remove_all(scratch_dir());
    if (want && ran == 0) {
        std::fprintf(stderr, "ERROR: no test named '%s'\n", want);
        return 2;
    }
    std::fprintf(stderr, "%d test(s), %d check(s), %d failure(s)\n", ran, g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
