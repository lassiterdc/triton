/** @file test_step_diag_log.cpp
 *  @brief Tier-1 tests for the T1 SWMM-step capture writer (swmm_triton.h) and for the
 *         resume positioning of the capture files (diag_log.h).
 *
 *  No solver run, no cluster, no SWMM call, no MPI. Two things are checked.
 *
 *  1. The bytes of every T1 record. The encoders in swmm_triton.h turn values read from
 *     the SWMM getters into the records of swmm_step_diag.bin (T1a), swmm_node_path3.bin
 *     (T1b), swmm_node_path3_step.bin (T1c) and swmm_link_flow_diag.bin (T1d), and the
 *     link selection and the sidecar text. The layouts are the shape's section 2 table
 *     and erratum E-2; the node block of a T1a record carries non-outfall ORDINALS
 *     (erratum E-10).
 *
 *  2. The positioner. A killed run can leave a capture file ending in a partial record;
 *     a resume must keep the last complete record stamped at or before the checkpoint
 *     and nothing after it. A kill during a flush is not reproducible on demand, so the
 *     file is cut at every byte offset of a final record instead (the deterministic
 *     stand-in, triton-custom reviewer TC-17): a fixed-size binary file, the
 *     variable-block binary file of T1a (parameters 48, 20, 4) and an interval csv.
 *
 *  Each test function is selectable by name from argv[1] so CTest registers one node id
 *  per property; with no argument every test runs. These are chunk tests of chunk 2a.2;
 *  the modules whose behaviour they guard are swmm_triton.h (the T1 encoders and link
 *  selection) and diag_log.h (the positioners).
 *
 *  Every test below is class NEW CAPABILITY (the chunk adds the writer): the plausible
 *  wrong implementation each one is red under is named in its comment, and each positioner
 *  test carries a control input (a whole final record, or a whole file) that must be kept,
 *  so a positioner that truncates everything fails it.
 *  Not established: this file does not run the live call sites in triton.h or the SWMM
 *  getters; the neutrality suite (chunk 3a.1) reads the files a solver run writes.
 */

// The include prefix of test_coupling_interval.cpp: swmm_triton.h compiles only after
// the include chain main.cpp gives it.
#include <iostream>
#include "mpi.h"
#include "constants.h"
#include "kokkos_utils.h"

#include "swmm_triton.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

using namespace SWMM_triton;
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

/// A fresh per-process directory under the system temporary directory.
static std::string scratch_dir()
{
    static std::string dir;
    if (dir.empty()) {
        dir = (fs::temp_directory_path() / ("test_step_diag_log_" + std::to_string((long)::getpid()))).string();
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    return dir;
}

static std::string path_of(const std::string& name) { return scratch_dir() + "/" + name; }

static void write_bytes(const std::string& path, const std::vector<char>& b, std::size_t upto)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(b.data(), (std::streamsize)upto);
}

static std::vector<char> read_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<char> header_bytes(int32_t magic, int32_t n_items)
{
    std::ostringstream o;
    diag_log::write_header(o, magic, n_items);
    const std::string s = o.str();
    return std::vector<char>(s.begin(), s.end());
}

static void append_to(std::vector<char>& dst, const std::vector<char>& src) { dst.insert(dst.end(), src.begin(), src.end()); }

template <class T> static T at(const std::vector<char>& b, std::size_t off)
{
    T v;
    std::memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

/// One T1a record at stamp `t` with `n_nc` non-converged ordinals 7, 8, ... and p3_pos
/// `p3_pos`.
static std::vector<char> t1a_record(double t, int n_nc, double p3_pos)
{
    std::vector<int32_t> ord;
    for (int k = 0; k < n_nc; ++k) ord.push_back(7 + k);
    return encode_step_diag_record(t, 0.25, 3, n_nc, p3_pos, -2.0, 11.0, ord.data());
}

// --- T1 record layouts ----------------------------------------------------------

// Property: a T1a record is 48 fixed bytes {f64 simtime, f64 dt, i32 trials, i32 n_nonconv,
// f64 p3_pos, f64 p3_neg, f64 link_volume} plus 4 bytes per listed ordinal, the count at
// byte 20 and the stamp at byte 0.
// Class: NEW CAPABILITY. Wrong implementation: the 24-byte fixed part of chunk 1a.1's
// hand-off note, or a count written at another offset; the positioner reads (48, 20, 4)
// and would walk a file of such records to the wrong boundaries.
// Why required: the resume positioner finds each record's end from these two numbers.
// Killed by: moving any field, or writing the ordinals before the fixed part.
// Second correct implementation: building the bytes with a packed struct instead of
// memcpy calls, which yields the same offsets.
static void S1_step_record_layout()
{
    const std::vector<char> r = t1a_record(12.5, 3, 9.0);
    CHECK(r.size() == STEP_DIAG_FIXED_BYTES + 3 * STEP_DIAG_ITEM_BYTES, "48 fixed bytes plus 4 per ordinal");
    CHECK(STEP_DIAG_FIXED_BYTES == 48 && STEP_DIAG_COUNT_OFFSET == 20 && STEP_DIAG_ITEM_BYTES == 4,
          "the positioner parameters are (48, 20, 4)");
    CHECK(at<double>(r, 0) == 12.5, "stamp at byte 0");
    CHECK(at<double>(r, 8) == 0.25, "dt at byte 8");
    CHECK(at<int32_t>(r, 16) == 3, "trials at byte 16");
    CHECK(at<int32_t>(r, 20) == 3, "n_nonconv at byte 20");
    CHECK(at<double>(r, 24) == 9.0, "p3_pos at byte 24");
    CHECK(at<double>(r, 32) == -2.0, "p3_neg at byte 32");
    CHECK(at<double>(r, 40) == 11.0, "link_volume at byte 40");
    CHECK(at<int32_t>(r, 48) == 7 && at<int32_t>(r, 52) == 8 && at<int32_t>(r, 56) == 9, "ordinals follow the fixed part");
    CHECK(t1a_record(1.0, 0, 0.0).size() == 48, "a step with no non-converged node is the 48 fixed bytes");
}

// Property: the T1b interval record is {f64 t_end, f64[n] pos, f64[n] neg, i32[n] n_nonconv},
// pos and neg are the per-node sums of the positive and of the negative path-3 terms of
// the interval's steps, and take() resets every sum so the next interval starts from zero.
// Class: NEW CAPABILITY. Wrong implementation: summing signed values into one array
// (pos+neg collapsed), or not resetting at the print instant (the sums would carry into the
// next interval and into a checkpoint, which a resumed run cannot reproduce).
// Why required: invariant A needs every value carried across intervals to be reset at the
// print instant or re-seeded (shape rule 6, C6).
// Killed by: dropping the reset, or counting a converged node as non-converged.
// Second correct implementation: keeping signed running totals and splitting them at take()
// is NOT equivalent (cancellation), so the second implementation is one array per sign
// updated per step, which this test also accepts.
static void S2_interval_sums_and_reset()
{
    node_path3_interval_t acc;
    acc.reset(3);
    const double a[3] = {1.5, -0.5, 0.0};
    const double b[3] = {-2.0, -0.25, 4.0};
    const int32_t nc_a[1] = {2};
    const int32_t nc_b[2] = {0, 2};
    CHECK(acc.add_step(a, nc_a, 1), "step 1 accepted");
    CHECK(acc.add_step(b, nc_b, 2), "step 2 accepted");
    CHECK(acc.steps == 2, "two steps counted");
    const std::vector<char> r = acc.take(240.0);
    CHECK(r.size() == node_path3_record_bytes(3) && r.size() == 8 + 20 * 3, "8 + 20 n bytes");
    CHECK(at<double>(r, 0) == 240.0, "t_end first");
    // pos: node0 1.5, node1 0, node2 4.0; neg: node0 -2.0, node1 -0.75, node2 0.
    CHECK(at<double>(r, 8) == 1.5 && at<double>(r, 16) == 0.0 && at<double>(r, 24) == 4.0, "positive sums per node");
    CHECK(at<double>(r, 32) == -2.0 && at<double>(r, 40) == -0.75 && at<double>(r, 48) == 0.0, "negative sums per node");
    CHECK(at<int32_t>(r, 56) == 1 && at<int32_t>(r, 60) == 0 && at<int32_t>(r, 64) == 2, "non-converged step counts");
    CHECK(acc.steps == 0, "the step count is reset");
    const std::vector<char> z = acc.take(360.0);
    bool all_zero = z.size() == r.size();
    for (std::size_t i = 8; i < z.size(); ++i) all_zero = all_zero && z[i] == 0;
    CHECK(all_zero, "every sum is zero after take(): nothing carries into the next interval");
    // Must-fire control: an ordinal outside [0, n) is refused and adds nothing.
    node_path3_interval_t c;
    c.reset(3);
    const int32_t bad[1] = {3};
    CHECK(!c.add_step(a, bad, 1) && c.steps == 0, "an ordinal outside the node range is refused with nothing added");
}

// Property: T1c is {f64 simtime, f64 dt, f64[n]} and T1d is {f64 simtime, f64[L]}; both
// start with the stamp, and the sidecar is one name per line with no header.
// Class: NEW CAPABILITY. Wrong implementation: a T1d record that carries dt (the reader
// would walk 8 bytes off per record), or a sidecar with a header row.
// Killed by: any change of the record sizes (the positioner strides use these sizes).
// Second correct implementation: assembling the record in a single memcpy pass.
static void S3_other_record_layouts()
{
    const double p3[2] = {0.5, -1.5};
    const std::vector<char> c = encode_node_path3_step_record(3.0, 0.5, p3, 2);
    CHECK(c.size() == node_path3_step_record_bytes(2) && c.size() == 32, "T1c record is 16 + 8 n");
    CHECK(at<double>(c, 0) == 3.0 && at<double>(c, 8) == 0.5 && at<double>(c, 16) == 0.5 && at<double>(c, 24) == -1.5,
          "T1c fields in order");
    const double fl[3] = {1.0, 2.0, 3.0};
    const std::vector<char> d = encode_link_flow_record(3.0, fl, 3);
    CHECK(d.size() == link_flow_record_bytes(3) && d.size() == 32, "T1d record is 8 + 8 L");
    CHECK(at<double>(d, 0) == 3.0 && at<double>(d, 8) == 1.0 && at<double>(d, 24) == 3.0, "T1d stamp then flows");
    CHECK(format_name_list({"J1", "J2"}) == "J1\nJ2\n", "sidecar: one name per line, no header");
    CHECK(format_name_list({}).empty(), "an empty list writes an empty sidecar");
}

// Property: header bytes of the capture files are {magic, 1, 8, n_items} in the order
// of erratum E-2, and the T1 magics are the ASCII tags DSP1, DPN3, DPS3, DLF1.
// Class: NEW CAPABILITY. Wrong implementation: a swapped magic between two files (a reader
// refuses a wrong magic per file, so a swap makes the file unreadable).
// Killed by: editing a magic constant. Second correct implementation: building the
// constants from four characters at compile time.
static void H1_headers_and_magics()
{
    CHECK(diag_log::MAGIC_SWMM_STEP_DIAG == 0x44535031, "DSP1");
    CHECK(diag_log::MAGIC_SWMM_NODE_PATH3 == 0x44504E33, "DPN3");
    CHECK(diag_log::MAGIC_SWMM_NODE_PATH3_STEP == 0x44505333, "DPS3");
    CHECK(diag_log::MAGIC_SWMM_LINK_FLOW == 0x444C4631, "DLF1");
    const std::vector<char> h = header_bytes(diag_log::MAGIC_SWMM_NODE_PATH3, 1133);
    CHECK(h.size() == 16 && at<int32_t>(h, 0) == 0x44504E33 && at<int32_t>(h, 4) == 1 && at<int32_t>(h, 8) == 8
              && at<int32_t>(h, 12) == 1133,
          "16 bytes: magic, version 1, value_width 8, n_items");
}

// Property: swmm_diag_links ALL lists every link in index order; a list picks the named
// links in file order, matched case-insensitively and returned in the engine's spelling;
// an unknown name, a repeated name and an empty list are refused with a message.
// Class: NEW CAPABILITY. Wrong implementation: matching case-sensitively (SWMM's lookup is
// not), or sorting the list into index order (the file order is the column order).
// Killed by: either change. Second correct implementation: an unordered_map lookup.
static void L1_link_selection()
{
    const std::vector<std::string> all = {"C1", "C2", "Pump3", "C4"};
    std::vector<int> idx;
    std::string err;
    std::vector<std::string> r = select_diag_links("ALL", {}, all, idx, err);
    CHECK(r == all && idx == std::vector<int>({0, 1, 2, 3}) && err.empty(), "ALL: every link in index order");
    r = select_diag_links("/some/list", {"c4", "PUMP3"}, all, idx, err);
    CHECK(r == std::vector<std::string>({"C4", "Pump3"}) && idx == std::vector<int>({3, 2}) && err.empty(),
          "a list: file order, case-insensitive match, engine spelling");
    r = select_diag_links("/some/list", {"C1", "nope"}, all, idx, err);
    CHECK(r.empty() && !err.empty() && err.find("nope") != std::string::npos, "an unknown name is refused and named");
    r = select_diag_links("/some/list", {"C1", "c1"}, all, idx, err);
    CHECK(r.empty() && !err.empty(), "a name repeated after the case fold is refused");
    r = select_diag_links("/some/list", {}, all, idx, err);
    CHECK(r.empty() && !err.empty(), "an empty list is refused");
}

// --- positioner: fixed-size binary file --------------------------------------------

// Property: a fixed-size binary capture file cut at ANY byte offset inside its final record
// is truncated to its last complete record; nothing after it survives; and a record
// stamped after the checkpoint time is a stale tail and is dropped as well.
// Class: NEW CAPABILITY. Wrong implementation: truncating to the file length rounded down
// to a whole number of records without reading stamps (fails the stale-tail arm), or
// keeping a partial record (fails every cut offset).
// Why required: the deterministic stand-in for a kill during a flush (reviewer TC-17).
// Killed by: dropping the partial-record check or the stamp comparison.
// Second correct implementation: walking with a seek-based reader of the stamp field only,
// which keeps the same set of records.
// Control: the cut at a whole record (offset = record size) keeps that record.
static void P1_fixed_record_file_cut_at_every_offset()
{
    const int L = 3;
    const double flow[3] = {1.0, 2.0, 3.0};
    const std::size_t rec = link_flow_record_bytes(L);
    std::vector<char> full = header_bytes(diag_log::MAGIC_SWMM_LINK_FLOW, L);
    for (int i = 1; i <= 6; ++i) append_to(full, encode_link_flow_record((double)i, flow, L));
    const std::size_t head = diag_log::HEADER_BYTES;
    const std::size_t sixth = head + 5 * rec;
    const std::string p = path_of("p1.bin");
    const diag_log::record_reader rd = diag_log::fixed_record_reader(rec);

    int partial_cuts_kept_five = 0;
    for (std::size_t k = 0; k < rec; ++k) {
        write_bytes(p, full, sixth + k);
        const diag_log::position_result r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_LINK_FLOW, L, rd, 100.0);
        if (r.status == diag_log::position_status::ok && r.records == 5 && r.kept_bytes == sixth
            && fs::file_size(p) == sixth)
            ++partial_cuts_kept_five;
    }
    CHECK(partial_cuts_kept_five == (int)rec, "every cut inside the final record keeps exactly the five complete records");

    write_bytes(p, full, sixth + rec);
    diag_log::position_result r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_LINK_FLOW, L, rd, 100.0);
    CHECK(r.records == 6 && fs::file_size(p) == sixth + rec, "control: a file cut at a whole record keeps all six");

    write_bytes(p, full, sixth + rec);
    r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_LINK_FLOW, L, rd, 3.0);
    CHECK(r.records == 3 && fs::file_size(p) == head + 3 * rec, "records stamped after t_k are a stale tail and are dropped");

    write_bytes(p, full, head - 5);
    r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_LINK_FLOW, L, rd, 100.0);
    CHECK(r.status == diag_log::position_status::bad_header, "a header cut short is refused, never read as a stride");
    r = diag_log::position_binary(path_of("absent.bin"), diag_log::MAGIC_SWMM_LINK_FLOW, L, rd, 1.0);
    CHECK(r.status == diag_log::position_status::missing_file, "a missing file is reported");
}

// --- positioner: variable-block binary file (the T1a shape) ---------------------------

// Property: a T1a file (48 fixed bytes, then n_nonconv i32 ordinals) cut at any byte offset
// inside its final record, in the fixed part or in the block, keeps its last complete
// record and nothing after it, with the positioner parameters (48, 20, 4).
// Class: NEW CAPABILITY. Wrong implementation: the (24, 20, 4) parameters of chunk 1a.1's
// hand-off note, which read a block from byte 24 of a 48-byte record; S1 pins the 48 and the
// 20, and this test walks real encoded records with them, so a changed parameter walks the
// file to other boundaries and the cut arms below fail. (A control walking this file with
// (24, 20, 4) is NOT run: its count field would read bytes of a double, a multi-gigabyte
// block size, and the reader would allocate it.)
// Why required: reviewer finding F-1 of chunk 2a.1; the parameters are the contract between
// this writer and the positioner.
// Killed by: changing any of the three parameters. Second correct implementation: a
// positioner that reads the block length from a trailer instead of a count field would pass
// this file but is not the shape's format, so the test pins the parameters, not the idea.
static void P2_variable_block_file_cut_at_every_offset()
{
    const int blocks[6] = {0, 3, 0, 2, 1, 4};
    std::vector<char> full = header_bytes(diag_log::MAGIC_SWMM_STEP_DIAG, 0);
    std::vector<std::size_t> end_of;   // file length after each record
    for (int i = 0; i < 6; ++i) {
        append_to(full, t1a_record((double)(i + 1), blocks[i], 1.0));
        end_of.push_back(full.size());
    }
    const std::string p = path_of("p2.bin");
    const diag_log::record_reader rd =
        diag_log::counted_block_record_reader(STEP_DIAG_FIXED_BYTES, STEP_DIAG_COUNT_OFFSET, STEP_DIAG_ITEM_BYTES);

    const std::size_t five = end_of[4];
    int ok_cuts = 0;
    const std::size_t sixth_len = end_of[5] - five;
    for (std::size_t k = 0; k < sixth_len; ++k) {
        write_bytes(p, full, five + k);
        const diag_log::position_result r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd, 100.0);
        if (r.status == diag_log::position_status::ok && r.records == 5 && r.kept_bytes == five && fs::file_size(p) == five)
            ++ok_cuts;
    }
    CHECK(ok_cuts == (int)sixth_len, "every cut inside the final record (fixed part or block) keeps the five complete records");

    write_bytes(p, full, end_of[5]);
    diag_log::position_result r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd, 100.0);
    CHECK(r.records == 6 && fs::file_size(p) == end_of[5], "control: the whole file keeps all six records");

    write_bytes(p, full, end_of[5]);
    r = diag_log::position_binary(p, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd, 4.0);
    CHECK(r.records == 4 && fs::file_size(p) == end_of[3], "records stamped after t_k are dropped, a block record included");
}

// --- positioner: interval csv ----------------------------------------------------------

// Property: an interval csv cut at any byte offset inside its final row keeps its last
// complete row (a row is complete when its newline is on disk) and nothing after it; rows
// stamped after t_k are dropped; a header other than the expected one is refused.
// Class: NEW CAPABILITY. Wrong implementation: keeping a row whose newline is missing
// (a partial number would parse as a smaller time and a wrong value).
// Killed by: dropping the newline test. Second correct implementation: reading the file in
// one pass into memory and cutting at the last '\n' at or before t_k.
// Control: the file whose final row is complete keeps that row.
static void P3_interval_csv_cut_at_every_offset()
{
    const std::string header = "%t_end, d_storage_m3";
    const std::string rows[4] = {"120.000000, 1.5\n", "240.000000, 2.5\n", "360.000000, 3.5\n", "480.000000, 4.5\n"};
    std::string full = header + "\n";
    for (const std::string& r : rows) full += r;
    const std::size_t three = header.size() + 1 + rows[0].size() + rows[1].size() + rows[2].size();
    const std::string p = path_of("p3.csv");
    auto write_text = [&](std::size_t upto) {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        f.write(full.data(), (std::streamsize)upto);
    };

    int ok_cuts = 0;
    for (std::size_t k = 0; k < rows[3].size(); ++k) {
        write_text(three + k);
        const diag_log::position_result r = diag_log::position_text(p, header, 0, 1000.0);
        if (r.status == diag_log::position_status::ok && r.records == 3 && r.kept_bytes == three && fs::file_size(p) == three)
            ++ok_cuts;
    }
    CHECK(ok_cuts == (int)rows[3].size(), "every cut inside the final row keeps the three complete rows");

    write_text(full.size());
    diag_log::position_result r = diag_log::position_text(p, header, 0, 1000.0);
    CHECK(r.records == 4 && fs::file_size(p) == full.size(), "control: a complete final row is kept");

    write_text(full.size());
    r = diag_log::position_text(p, header, 0, 240.0);
    CHECK(r.records == 2, "rows stamped after t_k are dropped");

    write_text(full.size());
    r = diag_log::position_text(p, "%other, header", 0, 1000.0);
    CHECK(r.status == diag_log::position_status::bad_header, "a header other than the expected one is refused");
    r = diag_log::position_text(path_of("absent.csv"), header, 0, 1.0);
    CHECK(r.status == diag_log::position_status::missing_file, "a missing file is reported");
}

// --- the writer class: a resumed file equals the clean file ----------------------------

// Property: a T1a file whose writer was killed after a flush, leaving a partial record,
// and was then resumed through capture_file::open_resume at the checkpoint time, ends
// byte-identical to the file of an uninterrupted writer (invariant A, the clean-versus-
// resumed equality, at the writer level).
// Class: NEW CAPABILITY. Wrong implementation: open_resume that reopens for append without
// truncating (the partial bytes stay in the middle of the file).
// Killed by: dropping the truncation in open_resume or the positioner's partial-record rule.
// Second correct implementation: a resume that rewrites the kept prefix into a new file
// and renames it, which yields the same bytes.
static void W1_resumed_file_equals_clean_file()
{
    std::vector<std::vector<char>> recs;
    const int blocks[5] = {0, 2, 0, 3, 1};
    for (int i = 0; i < 5; ++i) recs.push_back(t1a_record(10.0 * (i + 1), blocks[i], 1.0));
    const diag_log::record_reader rd =
        diag_log::counted_block_record_reader(STEP_DIAG_FIXED_BYTES, STEP_DIAG_COUNT_OFFSET, STEP_DIAG_ITEM_BYTES);

    const std::string pa = path_of("w1_clean.bin");
    {
        diag_log::capture_file a;
        a.declare_binary(pa, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd);
        a.open_clean();
        for (const auto& r : recs) a.append(r.data(), r.size());
        a.flush();
        a.close();
    }

    const std::string pb = path_of("w1_resumed.bin");
    {
        diag_log::capture_file b;
        b.declare_binary(pb, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd);
        b.open_clean();
        for (int i = 0; i < 3; ++i) b.append(recs[i].data(), recs[i].size());   // up to the checkpoint at t = 30
        b.flush();
        b.close();
        std::ofstream raw(pb, std::ios::binary | std::ios::app);                 // the killed run's partial next record
        raw.write(recs[3].data(), (std::streamsize)(STEP_DIAG_FIXED_BYTES + 3));
    }
    {
        diag_log::capture_file c;
        c.declare_binary(pb, diag_log::MAGIC_SWMM_STEP_DIAG, 0, rd);
        const long kept = c.open_resume(30.0);
        CHECK(kept == 3, "the resume keeps the three records at or before t_k");
        for (int i = 3; i < 5; ++i) c.append(recs[i].data(), recs[i].size());
        c.flush();
        c.close();
    }
    CHECK(read_bytes(pa) == read_bytes(pb), "the resumed file is byte-identical to the clean file");
}

// Property: a T1d flow is converted from the model's flow unit to cfs by the table of
// erratum E-11 (cfs 1, gpm 448.831, mgd 0.64632, cms 0.0283168, lps 28.3168, mld 2.44657
// units per cfs), the sign is kept, and a unit outside the enum is refused.
// Class: NEW CAPABILITY. Wrong implementation: writing the getter's value unconverted
// (0.01 on a CMS model where cfs is 0.35315), or converting with the factor inverted, or
// dropping the sign. Killed by: any of the three. Second correct implementation: a lookup
// from the unit name instead of the enum value.
static void F1_flow_units_convert_to_cfs()
{
    bool ok = false;
    double v = swmm_flow_to_cfs(0.01, swmm_CMS, ok);
    CHECK(ok && std::fabs(v - 0.01 / 0.0283168) < 1e-12, "0.01 cms is 0.35315 cfs");
    v = swmm_flow_to_cfs(448.831, swmm_GPM, ok);
    CHECK(ok && std::fabs(v - 1.0) < 1e-12, "448.831 gpm is 1 cfs");
    v = swmm_flow_to_cfs(0.64632, swmm_MGD, ok);
    CHECK(ok && std::fabs(v - 1.0) < 1e-12, "0.64632 mgd is 1 cfs");
    v = swmm_flow_to_cfs(28.3168, swmm_LPS, ok);
    CHECK(ok && std::fabs(v - 1.0) < 1e-12, "28.3168 lps is 1 cfs");
    v = swmm_flow_to_cfs(2.44657, swmm_MLD, ok);
    CHECK(ok && std::fabs(v - 1.0) < 1e-12, "2.44657 mld is 1 cfs");
    v = swmm_flow_to_cfs(-2.5, swmm_CFS, ok);
    CHECK(ok && v == -2.5, "cfs is unchanged and the sign is kept");
    v = swmm_flow_to_cfs(-0.01, swmm_CMS, ok);
    CHECK(ok && v < 0.0 && std::fabs(v + 0.01 / 0.0283168) < 1e-12, "a negative cms flow stays negative");
    v = swmm_flow_to_cfs(1.0, 99, ok);
    CHECK(!ok && v == 0.0, "a unit outside the enum is refused");
}

/// True when every call of `call` in `text` (comments stripped) sits between an open and a
/// close of the timer `timer`: the last start before it is after the last stop before it, and
/// a stop follows before the next start.
static bool calls_are_bracketed(const std::string& text, const std::string& call, const std::string& timer, int& n_calls)
{
    std::string code;
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) { const size_t c = line.find("//"); code += (c == std::string::npos ? line : line.substr(0, c)) + "\n"; }
    const std::string start = "st.start(" + timer + ")", stop = "st.stop(" + timer + ")";
    n_calls = 0;
    bool all = true;
    for (size_t p = code.find(call); p != std::string::npos; p = code.find(call, p + 1)) {
        ++n_calls;
        const size_t a = code.rfind(start, p), b = code.rfind(stop, p);
        const size_t nstop = code.find(stop, p), nstart = code.find(start, p);
        const bool open_before = a != std::string::npos && (b == std::string::npos || a > b);
        const bool closes_after = nstop != std::string::npos && (nstart == std::string::npos || nstop < nstart);
        if (!(open_before && closes_after)) all = false;
    }
    return all;
}

// Property: every swmm_step call site of src/triton.h sits inside a SWMM_SOLVE bracket and
// every log_exchange_step call inside a SWMM_LOG bracket, so the T3 branch reports the same
// SWMM_SOLVE and SWMM_LOG as the default block (experiment E1 reads SWMM_SOLVE).
// Class: REGRESSION (red on the round-1 tree, whose T3 branch called swmm_step bare).
// Killed by: removing a start or a stop at either site. Second correct implementation: a
// helper function that brackets the call, which this reader would read as one call site.
// Control: the same predicate on a copy with the first SWMM_SOLVE start removed must fail.
static void B1_swmm_step_and_log_call_sites_are_bracketed()
{
    std::ifstream in(TRITON_H_PATH);
    CHECK(in.is_open(), "src/triton.h opens");
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string t = ss.str();
    int n = 0;
    CHECK(calls_are_bracketed(t, "swmm_step(", "SWMM_SOLVE", n) && n >= 2, "every swmm_step call is inside SWMM_SOLVE, and there are at least two");
    CHECK(calls_are_bracketed(t, "log_exchange_step(", "SWMM_LOG", n) && n >= 2, "every log_exchange_step call is inside SWMM_LOG, and there are at least two");
    std::string mutant = t;
    const size_t k = mutant.find("st.start(SWMM_SOLVE)");
    CHECK(k != std::string::npos, "control: a SWMM_SOLVE start exists to remove");
    if (k != std::string::npos) mutant.erase(k, std::string("st.start(SWMM_SOLVE)").size());
    CHECK(!calls_are_bracketed(mutant, "swmm_step(", "SWMM_SOLVE", n), "control: with one start removed the predicate fails");
}

struct test_entry { const char* name; void (*fn)(); };
static const test_entry kTests[] = {
    {"S1_step_record_layout", S1_step_record_layout},
    {"S2_interval_sums_and_reset", S2_interval_sums_and_reset},
    {"S3_other_record_layouts", S3_other_record_layouts},
    {"H1_headers_and_magics", H1_headers_and_magics},
    {"L1_link_selection", L1_link_selection},
    {"P1_fixed_record_file_cut_at_every_offset", P1_fixed_record_file_cut_at_every_offset},
    {"P2_variable_block_file_cut_at_every_offset", P2_variable_block_file_cut_at_every_offset},
    {"P3_interval_csv_cut_at_every_offset", P3_interval_csv_cut_at_every_offset},
    {"W1_resumed_file_equals_clean_file", W1_resumed_file_equals_clean_file},
    {"F1_flow_units_convert_to_cfs", F1_flow_units_convert_to_cfs},
    {"B1_swmm_step_and_log_call_sites_are_bracketed", B1_swmm_step_and_log_call_sites_are_bracketed},
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
