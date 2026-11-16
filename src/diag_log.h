/** @file diag_log.h
 *  @brief The one place the diagnostics-capture files get their header, their
 *         resume positioning, their checkpoint flush and their failure policy.
 *
 *  A capture file (per-step SWMM records, edge fluxes, the step-limiting cell, the
 *  interval csv files) has to satisfy one property: the bytes of a run that was
 *  killed and resumed from a checkpoint equal the bytes of a run that was never
 *  interrupted. That needs the same four decisions at every capture, and a capture
 *  that makes them alone makes them differently. This header makes them once:
 *
 *   1. Header. A binary file starts {int32 magic, int32 version, int32 value_width,
 *      int32 n_items} (16 bytes, host byte order, which the #error below limits to
 *      little-endian), the layout of the exchange side-file's V2 header in
 *      swmm_triton.h with the version field added. A text file starts a header row
 *      whose first character is '%', the performance.txt style. A reader refuses a
 *      file whose header disagrees with what it expects, so a file written under a
 *      different layout is never walked with the wrong record stride.
 *   2. Resume positioning. At a resume from checkpoint time t_k each file is
 *      truncated to its last complete record whose stamp is at or before t_k and
 *      reopened for append, the rule position_exchange_log in swmm_triton.h applies
 *      to the exchange side-file. The file then holds exactly the records a clean
 *      run held at t_k, and the live segment continues it.
 *   3. Durability. flush_all() runs at each checkpoint write, before the
 *      checkpoint's cfg is written (triton.h). Nothing here flushes per step.
 *   4. Failure. A failed header write, append, flush or reposition aborts the whole
 *      job through MPI_Abort. A stream's failure state latches, so continuing would
 *      write every later checkpoint over a file that stopped growing, and each such
 *      checkpoint would be a resume point that cannot reach t_k.
 *
 *  Included from triton.h in every build configuration: a capture file exists with
 *  and without TRITON_SWMM, so nothing in this header depends on it. The values in
 *  the records are 8-byte floats whatever value_t is (value_width is always 8), so
 *  this header uses neither constants.h nor value_t.
 *
 *  The positioning functions below touch no MPI state and return a status instead
 *  of aborting, so a unit test can run them over a truncated file in a plain
 *  process. capture_file::open_resume is the abort-on-failure form the solver uses.
 */

#ifndef DIAG_LOG_H
#define DIAG_LOG_H

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Ensify.h"

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "diag_log.h writes native-byte-order headers and records, which the capture readers decode as little-endian"
#endif

namespace diag_log
{
	// Named FORMAT_VERSION because the vendored SWMM headers (consts.h) define the
	// macro VERSION, which would turn this declaration into a syntax error in any
	// translation unit that includes them first.
	static constexpr int32_t FORMAT_VERSION = 1;
	static constexpr int32_t VALUE_WIDTH = 8;
	static constexpr std::size_t HEADER_BYTES = 4 * sizeof(int32_t);

	// A record's stamp is the simulated time its content ends at; the positioner
	// keeps a record when stamp <= t_k + STAMP_TOLERANCE_S. The tolerance is the one
	// position_exchange_log uses (its EPS), so the two positioners cannot disagree
	// about whether a record ending at t_k belongs to the resumed segment's past.
	// A tolerance this small is safe only while the time step stays above it; a step
	// below 1e-6 s would let a post-checkpoint record at t_k + dt survive truncation.
	static constexpr double STAMP_TOLERANCE_S = 1e-6;

	// Magic numbers of the capture files (the file names are fixed by the shape).
	// Each int32 is the ASCII tag read most-significant byte first, so the bytes on
	// disk read in reverse: DSP1 is 0x44535031 and sits in the file as "1PSD".
	static constexpr int32_t MAGIC_SWMM_STEP_DIAG = 0x44535031;       // "DSP1" swmm_step_diag.bin
	static constexpr int32_t MAGIC_SWMM_NODE_PATH3 = 0x44504E33;      // "DPN3" swmm_node_path3.bin
	static constexpr int32_t MAGIC_SWMM_NODE_PATH3_STEP = 0x44505333; // "DPS3" swmm_node_path3_step.bin
	static constexpr int32_t MAGIC_SWMM_LINK_FLOW = 0x444C4631;       // "DLF1" swmm_link_flow_diag.bin
	static constexpr int32_t MAGIC_EDGE_FLUX = 0x44454631;            // "DEF1" edge_flux.bin
	static constexpr int32_t MAGIC_DT_LIMITER = 0x44444C31;           // "DDL1" dt_limiter.bin

	// --- header ---------------------------------------------------------------

	/** @brief Writes the 16-byte binary header. */
	static inline void write_header(std::ostream& out, int32_t magic, int32_t n_items)
	{
		const int32_t fields[4] = { magic, FORMAT_VERSION, VALUE_WIDTH, n_items };
		out.write(reinterpret_cast<const char*>(fields), sizeof(fields));
	}

	/** @brief The four header fields as read off disk. */
	struct header_t
	{
		int32_t magic = 0;
		int32_t version = 0;
		int32_t value_width = 0;
		int32_t n_items = 0;
		bool    complete = false;   ///< false when the file ended mid-header
	};

	/** @brief Reads the header; a short file reports complete=false and the fields
	 *  read in full are kept, so a truncated header is refused as a bad magic and
	 *  never as a stream error. On a complete header the get pointer sits at the
	 *  first record. */
	static inline header_t read_header(std::istream& in)
	{
		header_t h;
		int32_t fields[4] = { 0, 0, 0, 0 };
		in.read(reinterpret_cast<char*>(fields), sizeof(fields));
		if (in.gcount() != (std::streamsize)sizeof(fields)) return h;
		h.magic = fields[0]; h.version = fields[1]; h.value_width = fields[2]; h.n_items = fields[3];
		h.complete = true;
		return h;
	}

	/** @brief Why a capture file could not be positioned. One value per diagnostic. */
	enum class position_status
	{
		ok = 0,
		missing_file,           ///< the file is absent: a resume cannot complete a file it never had
		bad_header,             ///< short header, wrong magic, or (text) a header row other than the expected one
		bad_version,
		value_width_mismatch,
		n_items_mismatch,       ///< the layout (nodes, links, edges) changed between the legs of one run
		bad_record,             ///< (text) a complete data row whose time column does not parse
		resize_failed
	};

	/** @brief The outcome of positioning one file. */
	struct position_result
	{
		position_status status = position_status::ok;
		long            records = 0;        ///< records kept
		std::uintmax_t  kept_bytes = 0;     ///< the file's length after truncation
	};

	/** @brief Checks a read header against what this run expects. The order is
	 *  load-bearing: magic (with completeness) first so a foreign file is reported as
	 *  one and never as a wrong width, then version, width, item count. */
	static inline position_status validate_header(const header_t& h, int32_t magic, int32_t n_items)
	{
		if (!h.complete || h.magic != magic) return position_status::bad_header;
		if (h.version != FORMAT_VERSION)            return position_status::bad_version;
		if (h.value_width != VALUE_WIDTH)    return position_status::value_width_mismatch;
		if (h.n_items != n_items)            return position_status::n_items_mismatch;
		return position_status::ok;
	}

	// --- record walkers ---------------------------------------------------------

	/** @brief Reads ONE record from a stream sitting at a record start, advances past
	 *  it, and sets `stamp` to its end time. Returns false when the file ends before
	 *  the record does (a partial record from a kill during a write). */
	using record_reader = std::function<bool(std::istream& in, double& stamp)>;

	/** @brief A record of `record_bytes` whose first 8 bytes are the stamp as a double
	 *  (every fixed-size binary capture file: edge_flux, dt_limiter, T1b, T1c, T1d). */
	static inline record_reader fixed_record_reader(std::size_t record_bytes)
	{
		return [record_bytes](std::istream& in, double& stamp) -> bool {
			std::vector<char> buf(record_bytes);
			in.read(buf.data(), (std::streamsize)record_bytes);
			if (in.gcount() != (std::streamsize)record_bytes) return false;
			std::memcpy(&stamp, buf.data(), sizeof(double));
			return true;
		};
	}

	/** @brief A record of `fixed_bytes` (first 8 bytes the stamp) followed by a block
	 *  of N items of `item_bytes`, N being the int32 at byte `count_offset` of the
	 *  fixed part (the T1a shape: the per-step record, then the non-converged node
	 *  indices). A negative count is read as an incomplete record: it can only come
	 *  from bytes that are not a record. */
	static inline record_reader counted_block_record_reader(std::size_t fixed_bytes, std::size_t count_offset,
	                                                        std::size_t item_bytes)
	{
		return [fixed_bytes, count_offset, item_bytes](std::istream& in, double& stamp) -> bool {
			std::vector<char> buf(fixed_bytes);
			in.read(buf.data(), (std::streamsize)fixed_bytes);
			if (in.gcount() != (std::streamsize)fixed_bytes) return false;
			int32_t n = 0;
			std::memcpy(&n, buf.data() + count_offset, sizeof(int32_t));
			if (n < 0) return false;
			std::memcpy(&stamp, buf.data(), sizeof(double));
			const std::size_t block = (std::size_t)n * item_bytes;
			if (block > 0)
			{
				// A block longer than what is left of the stream is an incomplete record;
				// asking the allocator for it first would let a count field of a corrupted
				// or mismatched file request gigabytes. A killed run's prefix never reaches
				// this: its count field is valid, so the block read below fails instead.
				const std::streamoff here = in.tellg();
				in.seekg(0, std::ios::end);
				const std::streamoff end = in.tellg();
				in.seekg(here);
				if (here < 0 || end < here || (std::uintmax_t)block > (std::uintmax_t)(end - here)) return false;
				std::vector<char> blk(block);
				in.read(blk.data(), (std::streamsize)block);
				if (in.gcount() != (std::streamsize)block) return false;
			}
			return true;
		};
	}

	// --- positioning ------------------------------------------------------------

	/** @brief Truncates a binary capture file to its last complete record with
	 *  stamp <= up_to_time + STAMP_TOLERANCE_S.
	 *
	 *  The walk stops at the first record that is partial or stamped after t_k. Stamps
	 *  of one file do not decrease, so everything after that point is a tail the
	 *  killed run wrote past the checkpoint. A file whose last record ends before t_k
	 *  is NOT refused: a capture with no record near t_k (a fixed time increment, a
	 *  capture that writes nothing in an interval) is indistinguishable from a file
	 *  that lost its tail, and durability is guaranteed by flush_all() running before
	 *  the checkpoint cfg rather than by a length check here. */
	static inline position_result position_binary(const std::string& path, int32_t magic, int32_t n_items,
	                                              const record_reader& read_record, double up_to_time)
	{
		position_result r;
		std::error_code ec;
		if (!std::filesystem::exists(path, ec)) { r.status = position_status::missing_file; return r; }

		std::ifstream in(path, std::ios::binary);
		if (!in.is_open()) { r.status = position_status::missing_file; return r; }

		const header_t h = read_header(in);
		r.status = validate_header(h, magic, n_items);
		if (r.status != position_status::ok) return r;

		std::uintmax_t kept = HEADER_BYTES;
		while (true)
		{
			double stamp = 0.0;
			if (!read_record(in, stamp)) break;                                  // partial record: drop it
			if (stamp > up_to_time + STAMP_TOLERANCE_S) break;                   // written after t_k: stale tail
			kept = (std::uintmax_t)in.tellg();
			r.records++;
		}
		in.close();

		std::filesystem::resize_file(path, kept, ec);
		if (ec) { r.status = position_status::resize_failed; return r; }
		r.kept_bytes = kept;
		return r;
	}

	/** @brief The text-file form of position_binary: the file is a header row, then
	 *  one row per record with the stamp in comma-separated column `time_column`.
	 *  A final row with no terminating newline is a partial row and is dropped. */
	static inline position_result position_text(const std::string& path, const std::string& header_row,
	                                            int time_column, double up_to_time)
	{
		position_result r;
		std::error_code ec;
		if (!std::filesystem::exists(path, ec)) { r.status = position_status::missing_file; return r; }

		std::ifstream in(path, std::ios::binary);
		if (!in.is_open()) { r.status = position_status::missing_file; return r; }

		std::string line;
		if (!std::getline(in, line) || in.eof() || line != header_row)
		{
			r.status = position_status::bad_header;
			return r;
		}
		std::uintmax_t kept = line.size() + 1;

		while (std::getline(in, line))
		{
			if (in.eof()) break;                                                 // no newline: partial row
			std::stringstream fields(line);
			std::string field;
			bool found = false;
			for (int c = 0; std::getline(fields, field, ','); c++)
			{
				if (c == time_column) { found = true; break; }
			}
			char* end = nullptr;
			const double stamp = found ? std::strtod(field.c_str(), &end) : 0.0;
			if (!found || end == field.c_str())
			{
				r.status = position_status::bad_record;
				return r;
			}
			if (stamp > up_to_time + STAMP_TOLERANCE_S) break;
			kept += line.size() + 1;
			r.records++;
		}
		in.close();

		std::filesystem::resize_file(path, kept, ec);
		if (ec) { r.status = position_status::resize_failed; return r; }
		r.kept_bytes = kept;
		return r;
	}

	// --- the writer ------------------------------------------------------------------

	class capture_file;

	/** @brief Every open capture file of this process, so one call flushes or closes
	 *  them all. A function-local static in an external-linkage inline function, so every
	 *  translation unit shares one list (a `static` function would give each its own). */
	inline std::vector<capture_file*>& open_files()
	{
		static std::vector<capture_file*> files;
		return files;
	}

	/** @brief Prints the reason and aborts the whole job. MPI_Abort rather than
	 *  exit(): a rank that exits alone leaves every other rank waiting at the next
	 *  collective. */
	[[noreturn]] static inline void fatal(const std::string& what)
	{
		std::cerr << "[ERROR] diagnostics capture: " << what << std::endl;
		MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
		std::abort();
	}

	/** @brief One capture file: declared with its layout, then opened for a clean
	 *  start or for a resume, appended to, flushed at checkpoints and closed at the
	 *  end of the run. Not copyable: it registers its address in open_files(). */
	class capture_file
	{
	public:
		capture_file() = default;
		capture_file(const capture_file&) = delete;
		capture_file& operator=(const capture_file&) = delete;
		~capture_file() { close_quietly(); }

		/** @brief Declares a binary file: its magic, item count and record walker. The
		 *  file is not touched until open_clean or open_resume. */
		void declare_binary(const std::string& path, int32_t magic, int32_t n_items, record_reader reader)
		{
			path_ = path; text_ = false; magic_ = magic; n_items_ = n_items; reader_ = std::move(reader);
		}

		/** @brief Declares a text file: its header row (leading '%') and the
		 *  comma-separated column holding each row's time stamp. */
		void declare_text(const std::string& path, const std::string& header_row, int time_column)
		{
			path_ = path; text_ = true; header_row_ = header_row; time_column_ = time_column;
		}

		/** @brief Clean start: truncate, write the header, flush it. */
		void open_clean()
		{
			close_quietly();
			out_.open(path_, std::ios::binary | std::ios::trunc);
			if (!out_.is_open()) fatal("could not open " + path_ + " for writing");
			if (text_) out_ << header_row_ << '\n';
			else write_header(out_, magic_, n_items_);
			out_.flush();
			if (!out_.good()) fatal("could not write the header of " + path_);
			register_open();
		}

		/** @brief Resume from checkpoint time `up_to_time`: truncate to the last
		 *  complete record stamped at or before it and reopen for append. Returns the
		 *  records kept. A missing file or a header that disagrees aborts: appending
		 *  to it would give a file that is not the clean run's, and nothing later
		 *  would say so. */
		long open_resume(double up_to_time)
		{
			close_quietly();
			const position_result r = text_ ? position_text(path_, header_row_, time_column_, up_to_time)
			                                : position_binary(path_, magic_, n_items_, reader_, up_to_time);
			if (r.status != position_status::ok)
				fatal("cannot resume " + path_ + ": " + describe(r.status)
				      + ". Re-run from a clean start (checkpoint_id=0).");
			out_.open(path_, std::ios::binary | std::ios::app);
			if (!out_.is_open()) fatal("could not reopen " + path_ + " for append");
			register_open();
			return r.records;
		}

		/** @brief True between a successful open_clean/open_resume and close(). */
		bool is_open() const { return out_.is_open(); }

		/** @brief Appends bytes; aborts when the stream fails. */
		void append(const void* data, std::size_t bytes)
		{
			require_open("append to");
			out_.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
			if (!out_.good()) fatal("could not append to " + path_ + ". Aborting at this step: the stream's "
			                        "failure state latches, so every later checkpoint would sit over a file "
			                        "that stopped growing. The latest complete checkpoint remains resumable.");
		}

		/** @brief Appends one text row (the newline is added here). */
		void append_row(const std::string& row)
		{
			const std::string line = row + "\n";
			append(line.data(), line.size());
		}

		/** @brief Flushes; aborts when the stream fails. Called from flush_all() at
		 *  each checkpoint write, before the checkpoint's cfg exists. */
		void flush()
		{
			require_open("flush");
			out_.flush();
			if (!out_.good()) fatal("could not flush " + path_ + " at a checkpoint write. Aborting before this "
			                        "checkpoint's cfg is written, so the resume picker returns the previous "
			                        "checkpoint, whose records are all on disk.");
		}

		/** @brief Flushes and closes. */
		void close()
		{
			if (!out_.is_open()) return;
			flush();
			out_.close();
			unregister();
		}

		/** @brief The declared file path. */
		const std::string& path() const { return path_; }

		/** @brief The phrase fatal() prints after a failed positioning, one per status. */
		static const char* describe(position_status s)
		{
			switch (s)
			{
				case position_status::missing_file:         return "the file does not exist";
				case position_status::bad_header:           return "its header is not the one this run writes";
				case position_status::bad_version:          return "it was written with another capture-file version";
				case position_status::value_width_mismatch: return "it stores another value width";
				case position_status::n_items_mismatch:     return "its item count differs from this run's (the model or coupling changed between runs)";
				case position_status::bad_record:           return "a complete row has an unreadable time column";
				case position_status::resize_failed:        return "it could not be truncated";
				default:                                    return "ok";
			}
		}

	private:
		/// Aborts with the file named when a write is attempted before the file was opened.
		void require_open(const char* verb) const
		{
			if (!out_.is_open()) fatal(std::string("tried to ") + verb + " " + path_ + " before it was opened");
		}
		/// Adds this file to the process-wide list flush_all()/close_all() walk.
		void register_open() { open_files().push_back(this); }
		/// Removes this file from that list.
		void unregister()
		{
			std::vector<capture_file*>& v = open_files();
			v.erase(std::remove(v.begin(), v.end(), this), v.end());
		}
		// The destructor path: never aborts, so a process already ending cannot loop.
		void close_quietly()
		{
			if (!out_.is_open()) return;
			out_.close();
			unregister();
		}

		std::string path_;
		bool text_ = false;
		int32_t magic_ = 0;
		int32_t n_items_ = 0;
		record_reader reader_;
		std::string header_row_;
		int time_column_ = 0;
		std::ofstream out_;
	};

	/** @brief True when this rank owns an open capture file. triton.h brackets the
	 *  checkpoint flush with DIAG_TIME only then, so a run with every capture off
	 *  reports DIAG exactly 0. */
	static inline bool any_open() { return !open_files().empty(); }

	/** @brief Flushes every open capture file (the checkpoint flush). */
	static inline void flush_all()
	{
		const std::vector<capture_file*> files = open_files();
		for (capture_file* f : files) f->flush();
	}

	/** @brief Flushes and closes every open capture file (the end-of-run close). */
	static inline void close_all()
	{
		const std::vector<capture_file*> files = open_files();
		for (capture_file* f : files) f->close();
	}

	/** @brief Opens every declared file for a clean start. */
	static inline void open_clean_all(const std::vector<capture_file*>& files)
	{
		for (capture_file* f : files) f->open_clean();
	}

	/** @brief Opens every declared file for a resume from checkpoint time up_to_time. */
	static inline void open_resume_all(const std::vector<capture_file*>& files, double up_to_time)
	{
		for (capture_file* f : files) f->open_resume(up_to_time);
	}
}

#endif
