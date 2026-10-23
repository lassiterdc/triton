/** @file swmm_triton.h
 *  @brief Couples the TRITON surface solver to the SWMM sewer network, and makes
 *         a hotstart resume reproduce the full-window SWMM trajectory.
 *
 *  Maps each SWMM surface node onto the TRITON cell containing it, partitions the
 *  node set across the owning MPI ranks, and exchanges a weir/orifice flux each
 *  step. SWMM holds routing state and accumulators no TRITON checkpoint records,
 *  so a resume rebuilds them from a per-checkpoint state snapshot, with the
 *  per-step exchange side-file replay (exact, cost growing with t_k) as fallback.
 *
 *  @author Mario Morales Hernandez
 *  @author J. Fernández-Pato
 *  @author Daniel Lassiter
 *  @author Sudershan Gangrade
 *  @author Shih-Chieh Kao
 *  @bug No known bugs.
 */



#ifndef SWMM_TRITON_H
#define SWMM_TRITON_H

#ifdef TRITON_SWMM

// SWMM internal headers (must be included before swmm5.h for TRUE/FALSE macros)
#include "consts.h"
#include "datetime.h"
#include "enums.h"
#include "error.h"
#include "objects.h"
#include "funcs.h"
#include "macros.h"
#include "text.h"

// SWMM API header
#include "swmm5.h"

// Defined in the vendored routing.c beside routing_exportInflowNodeDepths and
// declared here, with its only caller: whether node j's external inflows include
// a FLOW inflow, the predicate addExternalInflows uses to give a node a slot.
extern "C" int routing_nodeHasFlowInflow(int j);

#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <cmath>
#include <vector>
#include <cstdint>
#include <filesystem>

#include "constants.h"
#include "string_utils.h"
#include "mpi_utils.h"
#include <sys/stat.h>
#include <dirent.h>

namespace SWMM_triton
{

	// Exchange-replay side-file header (V2): {int32 magic, int32 num_nodes, int32 value_width}.
	// V1 stored only magic+num_nodes -- both precision-independent -- so a side-file written
	// by a build of the other value_t replayed every record with a wrong stride, silently.
	// V2 records the stored width and refuses a disagreement; a V1 file fails the magic
	// check (never read as a width) and must be re-created from a clean start.
	static constexpr int32_t EXCHANGE_LOG_MAGIC = 0x53574D4D;        // ASCII "SWMM": V1, kept for the legacy diagnostic
	static constexpr int32_t EXCHANGE_LOG_MAGIC_V2 = 0x53574D32;     // ASCII "SWM2"
	static constexpr int32_t EXCHANGE_LOG_VALUE_WIDTH = static_cast<int32_t>(sizeof(value_t));
	static constexpr std::size_t EXCHANGE_LOG_HEADER_BYTES = 3 * sizeof(int32_t);

	// Single expression of the V2 layout: the writer and the reader below share it, so
	// the two sides cannot drift and a test can exercise the layout without a solver run.

	/** @brief Writes the V2 side-file header: magic, node count, stored value_t width. */
	static inline void write_exchange_log_header(std::ostream& out, int32_t num_nodes)
	{
		const int32_t magic = EXCHANGE_LOG_MAGIC_V2;
		const int32_t n     = num_nodes;
		const int32_t width = EXCHANGE_LOG_VALUE_WIDTH;
		out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
		out.write(reinterpret_cast<const char*>(&n),     sizeof(n));
		out.write(reinterpret_cast<const char*>(&width), sizeof(width));
	}

	/** @brief The three V2 header fields, as read off disk. */
	struct exchange_log_header_t
	{
		int32_t magic       = 0;
		int32_t num_nodes   = 0;
		int32_t value_width = 0;
		bool    complete    = false;   ///< false when the file ended mid-header; fields read in full are kept
	};

	/** @brief Why a side-file header was refused. One value per diagnostic. */
	enum class exchange_header_status
	{
		ok = 0,
		bad_magic,             ///< not a readable V2 header (V1 legacy file, or short/absent header)
		node_count_mismatch,   ///< the .inp / coupling changed between runs
		value_width_mismatch   ///< written by a build whose value_t differs from this one
	};

	/** @brief Reads the V2 header, leaving fields that cannot be read in full at 0 and
	 *  complete=false, so a truncated header is refused as bad_magic rather than as a
	 *  stream error. On a complete header the get pointer sits at the first record. */
	static inline exchange_log_header_t read_exchange_log_header(std::istream& in)
	{
		exchange_log_header_t h;
		const std::streamsize w = (std::streamsize)sizeof(int32_t);

		in.read(reinterpret_cast<char*>(&h.magic), sizeof(int32_t));
		if (in.gcount() != w) { h.magic = 0; return h; }
		in.read(reinterpret_cast<char*>(&h.num_nodes), sizeof(int32_t));
		if (in.gcount() != w) { h.num_nodes = 0; return h; }
		in.read(reinterpret_cast<char*>(&h.value_width), sizeof(int32_t));
		if (in.gcount() != w) { h.value_width = 0; return h; }
		h.complete = true;
		return h;
	}

	/** @brief Validates a V2 header against the running build. Check order is
	 *  load-bearing: magic (with completeness) before the fields, so a V1 legacy
	 *  file is reported as a bad header and never as a wrong width; node count
	 *  before width, matching the on-disk field order. */
	static inline exchange_header_status
	validate_exchange_log_header(const exchange_log_header_t& h,
	                             int32_t expected_nodes,
	                             int32_t running_value_width)
	{
		if (!h.complete)                          return exchange_header_status::bad_magic;
		if (h.magic != EXCHANGE_LOG_MAGIC_V2)     return exchange_header_status::bad_magic;
		if (h.num_nodes != expected_nodes)        return exchange_header_status::node_count_mismatch;
		if (h.value_width != running_value_width) return exchange_header_status::value_width_mismatch;
		return exchange_header_status::ok;
	}

	// --- The coupled-node list, read as SWMM reads the same input ------------------
	// TRITON and SWMM pair coupled nodes by POSITION, not by name: SWMM consumes the
	// exchange array and fills the depth array over its FLOW-inflow nodes in node
	// order, so a list that differs from SWMM's (a node missing or extra, or another
	// order) hands each surface cell another node's sewer flow with no error. These
	// helpers reproduce SWMM's section rule, tokenizer and FLOW test (input.c,
	// inflow.c), and init_swmm checks the result against the running engine.

	/** @brief Splits one .inp line into tokens exactly as SWMM's getTokens does.
	 *
	 *  Text from the first ';' onward is dropped, even inside quotes; tokens are
	 *  separated by space, tab, CR or LF; a token that opens with '"' runs to the next
	 *  '"' (or the end of the line) and excludes both quotes.
	 */
	static inline std::vector<std::string> swmm_inp_tokens(const std::string& line)
	{
		const std::string s = line.substr(0, line.find(';'));
		const char* sep = " \t\n\r";
		std::vector<std::string> toks;
		std::size_t i = 0;
		while (i < s.size()) {
			if (std::strchr(sep, s[i]) != nullptr) { ++i; continue; }
			const bool quoted = (s[i] == '"');
			const std::size_t start = quoted ? i + 1 : i;
			std::size_t end = quoted ? s.find_first_of("\"\n", start) : s.find_first_of(sep, start);
			if (end == std::string::npos) end = s.size();
			toks.push_back(s.substr(start, end - start));
			i = end + 1;
		}
		return toks;
	}

	/** @brief ASCII upper case, the folding SWMM applies when it compares names and
	 *  keywords (UCHAR in hash.c), so two spellings SWMM treats as one name compare
	 *  equal here too. */
	static inline std::string swmm_upper(std::string s)
	{
		for (char& c : s) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
		return s;
	}

	/** @brief What parse_inp_coupled_list reads from an .inp file. */
	struct inp_coupled_list_t
	{
		std::vector<std::string> names;              ///< coupled nodes, in SWMM node order
		std::vector<std::string> junction_names;     ///< [JUNCTIONS] data lines, file order
		std::vector<double>      junction_max_depth; ///< each junction's third token (0 if absent)
		std::vector<std::string> coord_names;        ///< [COORDINATES] data lines, file order
		std::vector<double>      coord_x;
		std::vector<double>      coord_y;
	};

	/** @brief Reads the coupled-node list, the junction depths and the node coordinates.
	 *
	 *  A section starts at any line whose first token begins with '[' and runs to the
	 *  next such line, wherever the line is indented and whatever comment follows it,
	 *  and every occurrence of a section is read -- as SWMM reads its input. A node is
	 *  coupled when an [INFLOWS] line's second token is not a [POLLUTANTS] name and
	 *  begins, case-insensitively, with FLOW (SWMM's own test), at most once per node.
	 *  The list is ordered by each node's position in [JUNCTIONS], because that is the
	 *  order SWMM numbers nodes and walks its FLOW inflows; a node absent from
	 *  [JUNCTIONS] keeps its relative place after them, and init_swmm refuses it.
	 *  Node names are compared case-insensitively, as SWMM compares them.
	 */
	static inline inp_coupled_list_t parse_inp_coupled_list(std::istream& in)
	{
		inp_coupled_list_t r;
		std::vector<std::pair<std::string, std::string>> inflow_lines;
		std::vector<std::string> pollutants;
		enum { OTHER, JUNCTIONS, COORDINATES, INFLOWS, POLLUTANTS } sect = OTHER;
		std::string line;
		while (std::getline(in, line)) {
			const std::vector<std::string> tok = swmm_inp_tokens(line);
			if (tok.empty()) continue;
			if (tok[0][0] == '[') {
				const std::string h = swmm_upper(tok[0]);
				// SWMM's section keywords (text.h), which SWMM matches as a case-insensitive
				// prefix of the header token (findmatch in input.c), so "[INFLOW]" opens [INFLOWS].
				if      (h.rfind("[JUNC", 0) == 0)       sect = JUNCTIONS;
				else if (h.rfind("[COORDINATE", 0) == 0) sect = COORDINATES;
				else if (h.rfind("[INFLOW", 0) == 0)     sect = INFLOWS;
				else if (h.rfind("[POLLUT", 0) == 0)     sect = POLLUTANTS;
				else                                       sect = OTHER;
				continue;
			}
			if (sect == JUNCTIONS && tok.size() >= 2) {
				r.junction_names.push_back(tok[0]);
				r.junction_max_depth.push_back(tok.size() >= 3 ? std::strtod(tok[2].c_str(), nullptr) : 0.0);
			}
			else if (sect == COORDINATES && tok.size() >= 3) {
				r.coord_names.push_back(tok[0]);
				r.coord_x.push_back(std::strtod(tok[1].c_str(), nullptr));
				r.coord_y.push_back(std::strtod(tok[2].c_str(), nullptr));
			}
			else if (sect == INFLOWS && tok.size() >= 2) {
				inflow_lines.emplace_back(tok[0], tok[1]);
			}
			else if (sect == POLLUTANTS) {
				pollutants.push_back(swmm_upper(tok[0]));
			}
		}

		// Pollutants are collected before any inflow line is classified, because the
		// [POLLUTANTS] section may follow [INFLOWS] in the file.
		std::vector<std::string> coupled, coupled_upper;
		for (const auto& fl : inflow_lines) {
			const std::string param = swmm_upper(fl.second);
			if (std::find(pollutants.begin(), pollutants.end(), param) != pollutants.end()) continue;
			if (param.rfind("FLOW", 0) != 0) continue;
			const std::string node = swmm_upper(fl.first);
			if (std::find(coupled_upper.begin(), coupled_upper.end(), node) != coupled_upper.end()) continue;
			coupled.push_back(fl.first);
			coupled_upper.push_back(node);
		}

		std::vector<std::string> junction_upper;
		for (const auto& j : r.junction_names) junction_upper.push_back(swmm_upper(j));
		std::vector<std::pair<std::size_t, std::size_t>> key;  // (sort key, list position)
		for (std::size_t i = 0; i < coupled.size(); ++i) {
			const auto it = std::find(junction_upper.begin(), junction_upper.end(), coupled_upper[i]);
			const std::size_t k = (it != junction_upper.end())
			                          ? static_cast<std::size_t>(it - junction_upper.begin())
			                          : junction_upper.size() + i;
			key.emplace_back(k, i);
		}
		std::sort(key.begin(), key.end());
		for (const auto& k : key) r.names.push_back(coupled[k.second]);
		return r;
	}

	/** @brief The outcome of check_coupled_list_against_swmm. */
	struct coupled_list_check_t
	{
		bool        ok = true;
		int         entry = -1;   ///< first failing list position, or -1 for the count check
		std::string reason;
	};

	/** @brief Checks TRITON's coupled list against SWMM's FLOW-inflow nodes.
	 *
	 *  Passes iff every entry's SWMM node index exceeds the previous entry's, every
	 *  entry is a node SWMM gives a FLOW inflow, and the list is as long as SWMM's
	 *  FLOW-inflow set. Together those make the list exactly SWMM's set in SWMM's
	 *  order, which is what the positional pairing requires. Returns the first
	 *  failure, entry checks before the count.
	 *
	 *  @param swmm_index              each entry's swmm_getIndex(swmm_NODE, name)
	 *  @param has_flow_inflow         each entry's routing_nodeHasFlowInflow
	 *  @param swmm_flow_inflow_count  routing_nodeHasFlowInflow summed over SWMM's nodes
	 */
	static inline coupled_list_check_t
	check_coupled_list_against_swmm(const std::vector<int>& swmm_index,
	                                const std::vector<int>& has_flow_inflow,
	                                int swmm_flow_inflow_count)
	{
		coupled_list_check_t r;
		const int n = static_cast<int>(swmm_index.size());
		for (int i = 0; i < n; ++i) {
			if (i > 0 && swmm_index[i] <= swmm_index[i - 1]) {
				r.ok = false; r.entry = i;
				r.reason = "out of order: its SWMM node index does not exceed the previous entry's";
				return r;
			}
			if (!has_flow_inflow[i]) {
				r.ok = false; r.entry = i;
				r.reason = "SWMM gives this node no FLOW inflow";
				return r;
			}
		}
		if (n != swmm_flow_inflow_count) {
			r.ok = false; r.entry = -1;
			r.reason = "TRITON couples " + std::to_string(n) + " nodes but SWMM gives "
			         + std::to_string(swmm_flow_inflow_count) + " nodes a FLOW inflow";
		}
		return r;
	}

	/** @brief A coupled node's rim from its parsed [JUNCTIONS] depth and SWMM's full depth.
	 *
	 *  SWMM's full depth wins, because it is the depth SWMM itself floods and surcharges
	 *  at. The parsed token is kept only where it equals SWMM's value to within a
	 *  relative 1e-12: SWMM returns an unraised depth through two unit conversions that
	 *  can move its last bit, so keeping the token there leaves a model with no
	 *  crown-raised coupled node byte-identical to the pre-item-11 solver.
	 */
	static inline double coupled_node_rim(double token, double swmm_full_depth)
	{
		return std::fabs(swmm_full_depth - token) <= 1e-12 * std::fabs(swmm_full_depth)
		           ? token : swmm_full_depth;
	}

	/** @brief True for a SWMM report error line: optional whitespace, "ERROR ", a digit
	 *  (^\s*ERROR [0-9]+). A bare substring test would also echo a title, a WARNING
	 *  naming an object, or any other report line that merely contains ERROR. */
	static inline bool is_swmm_error_line(const std::string& line)
	{
		std::size_t i = 0;
		while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
		return line.compare(i, 6, "ERROR ") == 0 && i + 6 < line.size()
		       && std::isdigit(static_cast<unsigned char>(line[i + 6]));
	}

	/** @brief True when a SWMM error line is a per-line INPUT error, which SWMM ends with
	 *  "at line N of input file:" or "at line N of [SECTION] section:" and follows with
	 *  the rejected input line (report_writeInputErrorMsg). Only then is the next report
	 *  line part of the message. */
	static inline bool swmm_error_line_names_input_line(std::string line)
	{
		while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
		auto ends = [&](const char* s) {
			const std::size_t n = std::strlen(s);
			return line.size() >= n && line.compare(line.size() - n, n, s) == 0;
		};
		return ends("input file:") || ends("section:");
	}

	/** @brief Echoes a closed SWMM report's error lines to `out`: a header naming the
	 *  report, every line is_swmm_error_line accepts, and after a per-line input error the
	 *  rejected input line (skipped if blank). Prints one line instead when the report
	 *  cannot be read, so a missing echo is never silent. Returns the lines echoed. */
	static inline int echo_swmm_report_errors(const char* report_path, std::ostream& out)
	{
		std::ifstream rpt(report_path);
		if (!rpt.is_open()) {
			out << "         The SWMM report " << report_path << " could not be read, so its error lines are not shown." << std::endl;
			return 0;
		}
		out << "         SWMM's report (" << report_path << ") says:\n";
		int echoed = 0;
		bool want_input_line = false;
		std::string line;
		while (std::getline(rpt, line)) {
			if (!line.empty() && line.back() == '\r') line.pop_back();
			if (want_input_line) {
				want_input_line = false;
				if (line.find_first_not_of(" \t") != std::string::npos && !is_swmm_error_line(line)) {
					out << "         " << line << "\n";
					++echoed;
					continue;
				}
			}
			if (is_swmm_error_line(line)) {
				out << "         " << line << "\n";
				++echoed;
				want_input_line = swmm_error_line_names_input_line(line);
			}
		}
		out << std::flush;
		return echoed;
	}


	class swmm_triton	/**< Main class for SWMM coupling. */
	{

	public:

/** @brief It initializes the simulation.
*
*  @param rank_ Subdomain id
*  @param size_ Number of subdomain
*/
		/// swmm_snapshot_keep_previous is consumed only by the clean-start
		/// disposal; it defaults to false, the removal policy an unset key means.
		/// swmm_snapshot_failed_replay is consumed only by
		/// try_restore_state_snapshot at a ".failed" marker; it defaults to
		/// false, which keeps the stop there. swmm_node_rim_from_dem only labels
		/// the rims init_swmm prints; triton.h applies the DEM rim.
		void initialize(int rank, int size, std::string inp_filename, std::string project_dir, std::string output_folder, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss, const bool swmm_snapshot_disable, const bool swmm_snapshot_keep_all, const bool swmm_snapshot_keep_previous = false, const bool swmm_node_rim_from_dem = false, const bool swmm_snapshot_failed_replay = false);


		void end_swmm(std::string output_dir);
		void stop_on_swmm_error(int code, const char* call, const char* when, const char* report_path = nullptr);   ///< rank 0: MPI_Abort with SWMM's own message when a SWMM API call returns non-zero; with report_path (the startup calls only), close SWMM first and echo its report's ERROR lines

		// --- TRITON->SWMM exchange-series persist/replay (hotstart-resume support) ---
		// The SWMM trajectory is a deterministic function of its per-step inputs
		// (global exchange flux + dt). We durably log them every step and, on resume,
		// fast-replay 0..t_k so SWMM's .out/stats rebuild the full window.
		void open_exchange_log_truncate();                       ///< clean start: fresh side-file + header
		/// Resume: replay 0..t_k, truncate, reopen for append. `reason` (the
		/// classify_missing_snapshot verdict) is carried into the resume-event
		/// log record, not used for control flow.
		void replay_exchange_history(value_t up_to_time, int checkpoint_id = 0,
		                             const char* reason = "absent");        ///< resume: replay 0..t_k, truncate, reopen for append
		void log_exchange_step(value_t dt, const value_t* gq);   ///< append one step's (dt, exchange_q) record
		void flush_exchange_log();                               ///< flush (called at each checkpoint write)
		void close_exchange_log();                               ///< close (called at end_swmm)

		// --- full-precision state snapshot (bounds the resume cost) ---------
		// The replay above is exact but its cost grows with t_k, and the GPU sits
		// idle for its whole duration (no device vector exists until it returns).
		// The snapshot makes the restore cost independent of t_k; the replay is
		// retained as the fallback for checkpoints that have no snapshot.

		/// Write the full-precision SWMM state snapshot for this checkpoint.
		/// Rank 0 only, charged to IO_TIME by the caller. A failure warns and does
		/// not abort the live run, which is worth more than one checkpoint's
		/// snapshot; the cost lands at the next resume from this checkpoint: a
		/// stop when the writer kept a ".failed" marker, otherwise a resume from
		/// an earlier snapshot or a replay.
		void write_state_snapshot(int checkpoint_id);

		/// Resume: restore SWMM's state from this checkpoint's snapshot.
		/// The return is graduated: Restored -- continue on the snapshot;
		/// Absent -- legitimate (a checkpoint predating snapshots, or one the
		/// keep-2 retention pruned), run the replay fallback; Unreadable --
		/// present but disagreeing with this build's reader on the quantity msg
		/// names, ABORT (a wrongly falling-back resume hides a real defect
		/// behind correct numbers delivered slowly); OutputMismatch -- the
		/// snapshot loaded but the .out holds fewer periods than it claims,
		/// ABORT and do not fall back: the accumulators are already restored and
		/// a replay on top of them would double-apply the 0..t_k prefix;
		/// KilledWrite -- a ".part", ".killed" or ".failed" marker survives for
		/// this checkpoint and no published snapshot does, so the write was cut
		/// off (".part"/".killed") or failed the writer's own check (".failed"),
		/// ABORT and NOT Absent: Absent enters the replay, which does not
		/// converge at a high checkpoint index. One machine-readable line names
		/// the next step: resume from the published snapshot below (cut-off
		/// write), stop (failed write), or start clean (nothing published below);
		/// FailedWriteReplay -- a ".failed" survives, nothing is published, and
		/// the run set swmm_snapshot_failed_replay=1: run the full exchange
		/// replay with reason "write-failed". The marker stays, so every later
		/// resume at this checkpoint takes the same route.
		enum class SnapshotRestore { Restored, Absent, Unreadable, OutputMismatch,
		                             KilledWrite, FailedWriteReplay };

		SnapshotRestore try_restore_state_snapshot(int checkpoint_id, value_t up_to_time);

		/// Clean start only: dispose of the EARLIER run's snapshot artifacts
		/// (published snapshots, ".part", ".killed" and ".failed" markers) under the
		/// keep-previous policy latched at initialize(). Returns the path of a
		/// published snapshot that could not be disposed of, or "" on success;
		/// the caller stops the run on a non-empty return. Published snapshots
		/// go too because a clean start removes the earlier run's checkpoint
		/// cfgs, and a survivor could be loaded by a later resume of the new
		/// run whose own write at that checkpoint failed. Nothing is disposed of
		/// on the abort path, where a marker is the operator's evidence.
		std::string sweep_snapshot_markers();

		/// The disposal with the policy passed in, so both policies are testable
		/// without a cfg. keep_previous renames every artifact to
		/// "{name}.prev{k}"; otherwise they are removed, except a ".failed",
		/// which is renamed and kept either way: it is the evidence for a stop,
		/// and this clean start is the operator's natural response to that stop.
		/// k is ONE number for the
		/// call, one above the largest j of any "*.prev{j}" in the directory, so
		/// no rename lands on (and replaces) an existing name. Published names
		/// are handled first and the first failure stops the call; a marker
		/// failure only warns, since no marker is ever loaded as state.
		std::string dispose_snapshot_artifacts(bool keep_previous);

		/// Highest checkpoint id above checkpoint_id carrying a live marker (an
		/// exact ".part"/".killed"/".failed" name with no published snapshot at that id),
		/// or -1. Used after a restore to name the checkpoint stepped over.
		int highest_live_marker_above(int checkpoint_id) const;

		/// Classifies WHY no snapshot was available for this checkpoint
		/// (pre-snapshot-checkpoint / retention-collision / absent), for the
		/// resume-event record. Rank 0.
		const char* classify_missing_snapshot(int checkpoint_id) const;

		void local_to_global(value_t*  local, value_t*  global, int* dict);
		void global_to_local(value_t*  global, value_t*  local, int* dict);

		std::string get_output_folder() const { return output_folder; }

		int num_of_swmm_links = 0; /**< Number of SWMM nodes connected to the surface per subdomain */
		int num_of_swmm_nodes = 0; /**< Number of SWMM total nodes */
		int global_num_of_swmm_links = 0; /**< Number of (full domain) SWMM nodes connected to the surface */

		/// Zero-initialised so a coupled build running a cfg with no inp_filename
		/// (a non-coupled case) leaves every node-count guard on its uncoupled
		/// branch instead of reading indeterminate values.
		int units; /**< IS units (0) or imperial units (1) */
		std::vector<value_t> x;
		std::vector<value_t> y;
		std::vector<std::string> nodeID;
		/// Every coupled node's name, in global-index (SWMM node) order, on every rank.
		/// Kept by process_swmm_node_locations, which narrows nodeID to this rank's
		/// nodes; init_swmm needs the whole list to look each node up in the engine.
		std::vector<std::string> global_nodeID;
		/// This rank's coupled nodes' inverts, from the engine, in the model's length
		/// unit; triton.h forms the DEM-derived rim from them.
		std::vector<value_t> node_invert;
		std::vector<value_t> loss;
		std::vector<value_t> diameter;
		std::vector<value_t> max_depth;
		std::vector<value_t> new_depth;
		std::vector<value_t> aux_new_depth;
		std::vector<value_t> exchange_q;
		std::vector<value_t> aux_exchange_q;


		std::vector<int> relative_swmm_node_index;
		std::vector<int> swmm_pos_arr;
		Constants::sources_list_t swmm_cells;

		int *counts = NULL;	/**< Array to hold every subdomains cell count */
		int *displs = NULL;	/**< Position array to hold each sub domains starting point in main domain */
		value_t* global_new_depth = NULL;
		value_t* aux_global_new_depth = NULL;
		value_t* global_exchange_q = NULL;
		value_t* aux_global_exchange_q = NULL;
		int* node_to_rank_dict = NULL;

	private:
		std::string output_folder; /**< Output folder path from config */

		// --- exchange-series side-file (rank 0 only) for hotstart-resume replay ---
		std::string exchange_log_path;   /**< path to the durable exchange-replay side-file */
		std::ofstream exchange_log;      /**< append stream for live exchange records (rank 0) */
		int exchange_num_nodes = 0;      /**< number of global SWMM surface nodes per record */

		/// Stem for the per-checkpoint state snapshots (checkpoint id appended).
		/// Keyed per checkpoint rather than rolling, so a resume from an OLDER
		/// checkpoint cannot silently load a LATER state -- it finds no file for
		/// its id and takes the replay.
		std::string snapshot_path_stem;

		/// Latched from the cfg key of the same name in initialize(); consumed
		/// by write_state_snapshot, which runs long after initialize returns.
		bool snapshot_keep_all = false;

		/// Latched from swmm_snapshot_keep_previous, for the same reason: its
		/// consumer, the clean-start disposal, runs after initialize returns.
		bool snapshot_keep_previous = false;

		/// Latched from swmm_snapshot_failed_replay, for the same reason: its
		/// consumer, try_restore_state_snapshot, runs after initialize returns.
		bool snapshot_failed_replay = false;

		/// Latched from swmm_node_rim_from_dem. Read only by init_swmm, to label the
		/// SWMM rims it prints as replaced when triton.h is about to set every rim
		/// from the DEM instead. False keeps SWMM's rims.
		bool node_rim_from_dem = false;

		std::string snapshot_path_for(int checkpoint_id) const;

		/// snapshot_save's temporary, and the name a resume renames a surviving
		/// temporary to. Neither ends in ".snapshot", so neither is counted or
		/// pruned as a published snapshot.
		std::string snapshot_part_path_for(int checkpoint_id) const;
		std::string snapshot_killed_path_for(int checkpoint_id) const;
		/// The marker snapshot_save leaves when its count pass refuses the model
		/// state (one text line) or its length check fails (the written bytes).
		std::string snapshot_failed_path_for(int checkpoint_id) const;

		/// Position the side-file for the live segment: validate the header, walk
		/// records up to up_to_time, truncate any stale tail or partial
		/// kill-record, reopen for append. With replay_steps true each record is
		/// pushed through swmm_step (the replay); with it false the records are
		/// only counted (the snapshot path has already restored their state).
		/// Both routes share this function so they agree exactly on where the
		/// live segment starts.
		long position_exchange_log(value_t up_to_time, bool replay_steps);

		int calc_swmm_node_col(value_t  node_x, value_t  xllc, value_t  cell_size_);

		int calc_swmm_node_row(value_t  node_y, value_t  yllc, value_t  cell_size_, int nrows);

		void read_inp_file(std::string inp_filename, const value_t  dx, const value_t manhole_diameter, const value_t manhole_loss);


		void process_swmm_node_locations(const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd);

		void init_swmm(std::string project_dir, std::string inp_filename, const bool swmm_snapshot_disable);


		int rank_;
		int size_;
		std::vector<value_t> x_all;
		std::vector<value_t> y_all;
		std::vector<std::string> nodeID_all;
		std::vector<value_t> max_depth_all;
		std::vector<std::string> junctionID_all;
		std::vector<std::string> diameter_all;
		std::vector<std::string> conduits_all;
		std::vector<std::string> conduits_node1_all;
		std::vector<std::string> conduits_node2_all;


	};





	void swmm_triton::initialize(int rank, int size, std::string inp_filename, std::string project_dir, std::string output_folder, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss, const bool swmm_snapshot_disable, const bool swmm_snapshot_keep_all, const bool swmm_snapshot_keep_previous, const bool swmm_node_rim_from_dem, const bool swmm_snapshot_failed_replay)
	{
		rank_=rank;
		size_=size;
		units=0;
		this->output_folder = output_folder;
		this->snapshot_keep_all = swmm_snapshot_keep_all;
		this->snapshot_keep_previous = swmm_snapshot_keep_previous;
		this->snapshot_failed_replay = swmm_snapshot_failed_replay;
		this->node_rim_from_dem = swmm_node_rim_from_dem;

		read_inp_file(inp_filename,dx, manhole_diameter, manhole_loss);
		process_swmm_node_locations(xll, yll, dx, global_rows, global_cols, pd);
		init_swmm(project_dir, inp_filename, swmm_snapshot_disable);
	}



	void swmm_triton::read_inp_file(std::string inp_filename, const value_t  dx, const value_t manhole_diameter, const value_t manhole_loss)
	{

		std::ifstream infile(inp_filename.c_str());

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << inp_filename.c_str() << std::endl;
			exit(EXIT_FAILURE);
		}

		std::string line;
		while (std::getline(infile, line)) {
			if (line.find("FLOW UNITS") != std::string::npos){
				std::string unitstext, trashtext;
            std::istringstream iss(line);
            iss >> trashtext >> unitstext;
				if(unitstext=="CFS"){
					units=1;
				}
				break;
			}
		}

		infile.close();

		// [COORDINATES], [JUNCTIONS] and [INFLOWS] are read with SWMM's own section rule,
		// tokenizer and FLOW test, so the coupled list is SWMM's FLOW-inflow set in SWMM's
		// node order (see parse_inp_coupled_list). init_swmm checks it against the engine.
		infile.open(inp_filename.c_str());
		const inp_coupled_list_t parsed = parse_inp_coupled_list(infile);
		infile.close();

		for (std::size_t j = 0; j < parsed.coord_names.size(); ++j) {
			double xij = parsed.coord_x[j];
			double yij = parsed.coord_y[j];
			nodeID_all.push_back(parsed.coord_names[j]);
			if(units){
				xij*=FT_TO_M_FACTOR;
				yij*=FT_TO_M_FACTOR;
			}
			x_all.push_back(xij);
			y_all.push_back(yij);
		}
		for (std::size_t j = 0; j < parsed.junction_names.size(); ++j) {
			junctionID_all.push_back(parsed.junction_names[j]);
			max_depth_all.push_back(parsed.junction_max_depth[j]);
		}
		nodeID = parsed.names;

		num_of_swmm_nodes=nodeID_all.size();
		global_num_of_swmm_links=nodeID.size();
		int nJunctions=junctionID_all.size();

		x.resize(global_num_of_swmm_links);
		y.resize(global_num_of_swmm_links);
		max_depth.resize(global_num_of_swmm_links);
		loss.resize(global_num_of_swmm_links);
		diameter.resize(global_num_of_swmm_links);

		// Find the coordinates (x, y) of the nodes connected to the surface. Names
		// compare case-insensitively, as SWMM's own lookup does; the first entry wins.
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			 for (int j = 0; j < num_of_swmm_nodes; j++) {
				  if (swmm_upper(nodeID[i]) == swmm_upper(nodeID_all[j])) {
						x[i] = x_all[j];
						y[i] = y_all[j];
						break;
				  }
			 }
		}

		// The parsed [JUNCTIONS] depth of each coupled node. It is provisional: init_swmm
		// replaces it with SWMM's full depth wherever the two differ.
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			 for (int j = 0; j < nJunctions; j++) {
				  if (swmm_upper(nodeID[i]) == swmm_upper(junctionID_all[j])) {
						max_depth[i] = max_depth_all[j];
						break;
				  }
			 }
		}
	//to be changed in the future, mainly the diameter
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			loss[i]=manhole_loss;
			//diameter[i]=min(0.5*dx,1.2); //1.2=4 ft. Can range from 1.2m to 1.8m (4ft to 6ft)0.5dx is just to avoid numerical instabilities.
			diameter[i]=manhole_diameter;
		}

		if (manhole_diameter > dx){
			if (rank_ == 0){
				std::cerr << WARN "Manhole diameter is greater than grid resolution. This may cause numerical instabilities" << std::endl;
			}
		}

		if (rank_ == 0){
			std::cerr << IN "SWMM inp file read" << std::endl;
		}

	}


	void swmm_triton::process_swmm_node_locations(const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd)
	{

		Constants::sources_list_t global_swmm_cells;

		//copy first all the global vectors to a local copy and assign them zero size
		std::vector<value_t> global_node_swmm_x = x;
		std::vector<value_t> global_node_swmm_y = y;
		global_nodeID = nodeID;
		std::vector<value_t> global_max_depth = max_depth;
		std::vector<value_t> global_loss = loss;
		std::vector<value_t> global_diameter = diameter;


		x.resize(0);
		y.resize(0);
		nodeID.resize(0);
		max_depth.resize(0);
		loss.resize(0);
		diameter.resize(0);

		num_of_swmm_links = 0;

		std::vector<int> node_swmm_rows, node_swmm_cols;

		node_swmm_rows.assign(global_num_of_swmm_links, -1);
		node_swmm_cols.assign(global_num_of_swmm_links, -1);


		int exit_failure=0;
		for (int i = 0; i < global_num_of_swmm_links; ++i)
		{

			node_swmm_cols[i] = calc_swmm_node_col(global_node_swmm_x[i], xll, dx);
			node_swmm_rows[i] = calc_swmm_node_row(global_node_swmm_y[i], yll, dx, global_rows);

			if(node_swmm_cols[i] >= global_cols || node_swmm_rows[i] >= global_rows || node_swmm_cols[i]<0 || node_swmm_rows[i]<0){
				std::cerr << ERROR "Node_swmm " << global_nodeID[i]  << " is out of bounds" << std::endl;
				exit_failure=1;
			}
			std::pair<int, int> scell(node_swmm_rows[i]+ GHOST_CELL_PADDING, node_swmm_cols[i]+ GHOST_CELL_PADDING);
			global_swmm_cells.push_back(scell);

			//check if the has been already assigned from another node
			for (int j = 0; j < i; ++j){

				if((scell.first == node_swmm_rows[j]+ GHOST_CELL_PADDING)&& (scell.second == node_swmm_cols[j]+ GHOST_CELL_PADDING)){
					std::cerr << ERROR "Node_swmm " << global_nodeID[i]  << " has the same TRITON cell than node_swmm " << global_nodeID[j]  <<std::endl;
					exit_failure=1;
				}
			}
		}


		for (int i = 0; i < global_num_of_swmm_links; ++i)
		{
			int srank = 0;
			int prev_rows_sum = 0;

			if(size_ > 1){
				int node_swmm_row = node_swmm_rows[i];
				int rows_sum = pd.part_dims[0].first - 2 * GHOST_CELL_PADDING;

				if(node_swmm_row >= rows_sum){
					for(int j=1; j<size_; j++){
						prev_rows_sum = rows_sum;
						rows_sum += pd.part_dims[j].first - 2 * GHOST_CELL_PADDING;
						if(node_swmm_row < rows_sum){
							srank = j;
							break;
						}
					}
				}
			}
			node_swmm_rows[i] = node_swmm_rows[i] - prev_rows_sum + GHOST_CELL_PADDING;
			node_swmm_cols[i] = node_swmm_cols[i] + GHOST_CELL_PADDING;

			if (rank_ == srank)
			{
				relative_swmm_node_index.push_back(i);
				x.push_back(global_node_swmm_x[i]);
				y.push_back(global_node_swmm_y[i]);
				nodeID.push_back(global_nodeID[i]);
				max_depth.push_back(global_max_depth[i]);
				loss.push_back(global_loss[i]);
				diameter.push_back(global_diameter[i]);
				std::pair<int, int> scell(node_swmm_rows[i], node_swmm_cols[i]);
				swmm_cells.push_back(scell);
				swmm_pos_arr.push_back(scell.first*(global_cols+2*GHOST_CELL_PADDING) + scell.second);
				num_of_swmm_links++;
			}
		}

		new_depth.resize(num_of_swmm_links);
		exchange_q.resize(num_of_swmm_links);

		if(exit_failure) exit(EXIT_FAILURE);

		if (rank_ == 0){
			std::cerr << IN "SWMM node locations processed" << std::endl;
		}

	}


	void swmm_triton::init_swmm(std::string project_dir, std::string inp_filename, const bool swmm_snapshot_disable)
	{

		//initialize to zero all the exchange_q values and new_depth values
		std::fill(exchange_q.begin(), exchange_q.end(), 0.0); //local variables
		std::fill(new_depth.begin(), new_depth.end(), 0.0); //local variables

		// Every coupled node's rim and invert, by global index: filled from the engine on
		// rank 0 below and broadcast to every rank, because only rank 0 runs SWMM.
		std::vector<double> rim_global(global_num_of_swmm_links, 0.0);
		std::vector<double> invert_global(global_num_of_swmm_links, 0.0);

		if (counts != NULL)
			delete[] counts;
		if (displs != NULL)
			delete[] displs;

		if (rank_ == 0)
			counts = new int[size_];
		MPI_Gather(&num_of_swmm_links, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (rank_ == 0)
		{
			displs = new int[size_];
			displs[0] = 0;

			for (int i = 1; i < size_; i++)
			{
				displs[i] = displs[i - 1] + (long long)counts[i - 1];
			}

		}
		if(rank_==0){
			node_to_rank_dict = new int[global_num_of_swmm_links];
		}

		MPI_Gatherv(relative_swmm_node_index.data(), num_of_swmm_links, MPI_INT, node_to_rank_dict, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);


		if(rank_==0){
			global_new_depth = new value_t[global_num_of_swmm_links];
			aux_global_new_depth = new value_t[global_num_of_swmm_links];
			global_exchange_q = new value_t[global_num_of_swmm_links];
			aux_global_exchange_q = new value_t[global_num_of_swmm_links];

			for(int i=0;i<global_num_of_swmm_links;i++){
				global_new_depth[i]=0.0;
				aux_global_new_depth[i]=0.0;
				global_exchange_q[i]=0.0;
				aux_global_exchange_q[i]=0.0;

			}

    		std::string filenameWithoutPath = inp_filename.substr(inp_filename.find_last_of("/\\") + 1);
			std::string filenameWithoutExtension = filenameWithoutPath.substr(0, filenameWithoutPath.rfind("."));

			std::string root_dir(project_dir + "/" + output_folder + "/");
			std::string output_dir_swmm = project_dir + "/" + output_folder + "/swmm/";


			DIR* dir;
			if(output_dir_swmm.empty())
			{
				dir = opendir(".");
			}
			else
			{
				dir = opendir(root_dir.c_str());
			}
			if (!dir)
			{
				mkdir(root_dir.c_str(), S_IRWXU);
				mkdir(output_dir_swmm.c_str(), S_IRWXU);
			}
			else
			{
				closedir(dir);
				DIR *dir2;
				dir2 = opendir(output_dir_swmm.c_str());
				if (!dir2)
				{
					mkdir(output_dir_swmm.c_str(), S_IRWXU);
				}
				else
				{
					closedir(dir2);
				}
			}

			std::string report_filename = output_dir_swmm + filenameWithoutExtension + ".rpt";
			std::string binary_filename = output_dir_swmm + filenameWithoutExtension + ".out";

			// Durable side-file logging the per-step TRITON->SWMM exchange series, so that a
			// hotstart-resumed allocation can fast-replay 0..t_k and rebuild a full-window
			// hydraulics.rpt/.out instead of a truncated post-checkpoint segment.
			exchange_log_path = output_dir_swmm + filenameWithoutExtension + "_exchange_replay.bin";
			// Per-checkpoint state snapshots live beside the side-file in the same
			// swmm/ subtree, so anything that preserves one preserves the other.
			// An empty stem (swmm_snapshot_disable, or no surface nodes) is the
			// single signal both the writer and the restore path test.
			if (!swmm_snapshot_disable)
			{
				snapshot_path_stem = output_dir_swmm + filenameWithoutExtension + "_state";
			}
			else
			{
				std::cerr << WARN "SWMM state snapshots DISABLED by cfg "
				             "(swmm_snapshot_disable=1).\n"
				          << "         Any hotstart resume will take the exchange-replay "
				             "fallback and record reason `absent`."
				          << std::endl;
			}
			exchange_num_nodes = global_num_of_swmm_links;

			// SWMM reports a rejected input (a missing external time-series file, a bad
			// [OPTIONS] keyword, anything its own validation finds after read_inp_file
			// passed) only through these two return codes. Ignored, SWMM stays in its
			// error state and every later step returns at once without routing, so the
			// coupled run would complete against a sewer that never moves.
			const int swmm_open_rc =
			swmm_open(inp_filename.c_str(), report_filename.c_str(), binary_filename.c_str());
			stop_on_swmm_error(swmm_open_rc, "swmm_open", "while reading and validating the SWMM input",
			                   report_filename.c_str());
			stop_on_swmm_error(swmm_start(TRUE), "swmm_start", "while starting the SWMM analysis",
			                   report_filename.c_str());

			// Each coupled node's rim, from the engine that routes the model. SWMM raises a
			// node's full depth to the crown of its highest connecting link, so a junction
			// recorded at 0 under a 1.2 m pipe is 1.2 m deep to SWMM; the recorded token
			// would give TRITON a rim of 0, and the exchange law would then push water from
			// the node onto a dry street whenever the pipe holds any. Ineligible nodes stop
			// the run here, before any checkpoint is read or written: a coupled node that is
			// not a JUNCTION, or a list that is not SWMM's FLOW-inflow set in SWMM's order.
			static const char* node_type_name[] = {"JUNCTION", "OUTFALL", "STORAGE", "DIVIDER"};
			std::vector<int> swmm_index(global_num_of_swmm_links, -1);
			std::vector<int> has_flow(global_num_of_swmm_links, 0);
			int refused = 0;
			for (int i = 0; i < global_num_of_swmm_links; i++) {
				const int idx = swmm_getIndex(swmm_NODE, global_nodeID[i].c_str());
				swmm_index[i] = idx;
				if (idx < 0) {
					std::cerr << ERROR "Coupled node " << global_nodeID[i]
					          << " is not a node of the SWMM model." << std::endl;
					refused = 1;
					continue;
				}
				const int type = static_cast<int>(swmm_getValue(swmm_NODE_TYPE, idx));
				if (type != swmm_JUNCTION) {
					std::cerr << ERROR "Coupled node " << global_nodeID[i] << " is a SWMM "
					          << ((type >= 0 && type <= 3) ? node_type_name[type] : "non-junction")
					          << " node; this build couples JUNCTION nodes only." << std::endl;
					refused = 1;
					continue;
				}
				has_flow[i] = routing_nodeHasFlowInflow(idx);
				invert_global[i] = swmm_getValue(swmm_NODE_ELEV, idx);
				double token = 0.0;
				for (std::size_t j = 0; j < junctionID_all.size(); j++) {
					if (swmm_upper(junctionID_all[j]) == swmm_upper(global_nodeID[i])) {
						token = max_depth_all[j];
						break;
					}
				}
				rim_global[i] = coupled_node_rim(token, swmm_getValue(swmm_NODE_MAXDEPTH, idx));
			}
			if (refused) MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);

			int swmm_flow_count = 0;
			for (int j = 0; j < swmm_getCount(swmm_NODE); j++) swmm_flow_count += routing_nodeHasFlowInflow(j);
			const coupled_list_check_t chk =
				check_coupled_list_against_swmm(swmm_index, has_flow, swmm_flow_count);
			if (!chk.ok) {
				std::cerr << ERROR "TRITON's coupled-node list does not match SWMM's FLOW-inflow nodes: ";
				if (chk.entry >= 0)
					std::cerr << "entry " << chk.entry << " (" << global_nodeID[chk.entry] << "): ";
				std::cerr << chk.reason << "." << std::endl;
				MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
			}

			char rim_line[512];
			for (int i = 0; i < global_num_of_swmm_links; i++) {
				std::snprintf(rim_line, sizeof(rim_line), "SWMM coupled node %s: rim %.17g, invert %.17g%s",
				              global_nodeID[i].c_str(), rim_global[i], invert_global[i],
				              node_rim_from_dem ? " (replaced by the DEM rim)" : "");
				std::cerr << IN << rim_line << std::endl;
			}

			std::cerr << IN "SWMM initialized" << std::endl;
		}

		// Every rank needs its own nodes' rims, but only rank 0 can ask SWMM. The
		// broadcast is on ENSIFY_COMM_WORLD so an ensemble member exchanges only with
		// its own ranks.
		if (global_num_of_swmm_links > 0) {
			MPI_Bcast(rim_global.data(), global_num_of_swmm_links, MPI_DOUBLE, 0, ENSIFY_COMM_WORLD);
			MPI_Bcast(invert_global.data(), global_num_of_swmm_links, MPI_DOUBLE, 0, ENSIFY_COMM_WORLD);
		}
		node_invert.resize(num_of_swmm_links);
		for (int k = 0; k < num_of_swmm_links; k++) {
			max_depth[k] = static_cast<value_t>(rim_global[relative_swmm_node_index[k]]);
			node_invert[k] = static_cast<value_t>(invert_global[relative_swmm_node_index[k]]);
		}

	}

	// Stops the whole run when a SWMM API call returns non-zero, printing SWMM's own
	// code and message. Rank 0 only (init_swmm, the per-step solve, the replay walk),
	// so MPI_Abort and not exit(): a rank-0-only exit() would hang every other rank at
	// the next collective. A zero code returns at once, so a healthy run is unchanged.
	//
	// report_path is passed by the two startup calls only. For a per-line input error
	// (ERROR 2xx) swmm_getError returns just "ERROR 200", and the line naming the
	// problem is only in the report's buffered stream, which an abort discards; so
	// SWMM is closed first (flushing the report) and the report's error lines are
	// echoed. Deliberately NOT after a swmm_step failure: closing mid-run without
	// swmm_end skips the output and routing close, which SWMM does not document, and
	// a step's message already reaches swmm_getError.
	void swmm_triton::stop_on_swmm_error(int code, const char* call, const char* when, const char* report_path)
	{
		if (code == 0) return;
		char swmm_msg[MAXMSG + 1];
		swmm_msg[0] = '\0';
		swmm_getError(swmm_msg, MAXMSG);
		std::cerr << ERROR "SWMM " << call << " returned error code " << code << " " << when << ":\n"
		          << "         " << swmm_msg << "\n"
		          << "         Stopping the run. With SWMM in an error state every later swmm_step\n"
		          << "         returns at once without routing, so continuing would complete the\n"
		          << "         coupled simulation against a sewer that never moves."
		          << std::endl;
		if (report_path != nullptr) {
			swmm_close();
			echo_swmm_report_errors(report_path, std::cerr);
		}
		MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
	}

	void swmm_triton::end_swmm(std::string output_dir)
	{
		if(rank_==0){
			close_exchange_log();
			swmm_end();
			swmm_report(output_dir.c_str());
			swmm_close();
		}
	}


	// ---------------------------------------------------------------------------
	// Exchange-series side-file (hotstart-resume replay support). Rank 0 only.
	//
	// File layout:
	//   header : int32 magic (= EXCHANGE_LOG_MAGIC_V2) ; int32 num_nodes ;
	//            int32 value_width (= sizeof(value_t) in the writing build)
	//   record : value_t dt ; value_t exchange_q[0..num_nodes-1]   (one per TRITON step)
	//
	// Clean start: truncate the file and write a fresh header.
	// Resume: replay records 0..t_k through swmm_step (rebuilding SWMM's .out/stats
	// for the full window), truncate any stale/partial tail, then reopen for append
	// so the live segment t_k..end continues the series.
	// ---------------------------------------------------------------------------

	void swmm_triton::open_exchange_log_truncate()
	{
		if (rank_ != 0) return;
		exchange_log.close();
		exchange_log.open(exchange_log_path, std::ios::binary | std::ios::trunc);
		if (!exchange_log.is_open())
		{
			std::cerr << ERROR "Could not open SWMM exchange-replay side-file for writing: "
			          << exchange_log_path << std::endl;
			exit(EXIT_FAILURE);
		}
		write_exchange_log_header(exchange_log, static_cast<int32_t>(exchange_num_nodes));
		exchange_log.flush();
	}


	// is_open() is only the not-initialised sentinel (no SWMM surface nodes); the
	// durability outcome is read through good(), because a stream can be open and
	// failed at once. A failure aborts rather than continuing: ofstream failure
	// latches, so every later checkpoint would sit over a dead series. The flush
	// runs before the checkpoint's cfg is written (triton.h), so an abort there
	// leaves the previous checkpoint resumable. MPI_Abort, not exit(): both
	// callers are rank-0-only.
	void swmm_triton::log_exchange_step(value_t dt, const value_t* gq)
	{
		if (rank_ != 0) return;
		if (!exchange_log.is_open()) return;   // no SWMM surface nodes / not initialized
		exchange_log.write(reinterpret_cast<const char*>(&dt), sizeof(value_t));
		exchange_log.write(reinterpret_cast<const char*>(gq), sizeof(value_t) * exchange_num_nodes);
		if (!exchange_log.good())
		{
			std::cerr << ERROR "Could not append to the SWMM exchange-replay side-file "
			          << exchange_log_path << ".\n"
			          << "         Aborting at this step rather than continuing. The "
			             "stream's failure state latches, so continuing would write "
			             "every later checkpoint over a dead series and make each one "
			             "a resume point whose replay cannot reach it. The latest "
			             "complete checkpoint remains resumable."
			          << std::endl;
			MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
		}
	}


	void swmm_triton::flush_exchange_log()
	{
		if (rank_ != 0) return;
		if (!exchange_log.is_open()) return;   // no SWMM surface nodes / not initialized
		exchange_log.flush();
		if (!exchange_log.good())
		{
			std::cerr << ERROR "Could not flush the SWMM exchange-replay side-file "
			          << exchange_log_path << " at a checkpoint write.\n"
			          << "         Aborting before this checkpoint's cfg is written, so "
			             "the resume picker returns the previous checkpoint, whose "
			             "series is covered and whose snapshot is complete."
			          << std::endl;
			MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
		}
	}


	void swmm_triton::close_exchange_log()
	{
		if (rank_ != 0) return;
		if (exchange_log.is_open()) { exchange_log.flush(); exchange_log.close(); }
	}


	std::string swmm_triton::sweep_snapshot_markers()
	{
		return dispose_snapshot_artifacts(snapshot_keep_previous);
	}


	// Splits a snapshot-directory entry name into its checkpoint id and the
	// suffix after ".snapshot" when the name is exactly
	// "{prefix}{digits}.snapshot{suffix}" with a suffix this mechanism writes;
	// false for every other name, including a retained "*.prev{k}", which is how
	// a retained file stays invisible to both readers below.
	static bool parse_snapshot_entry_name(const std::string& name,
	                                      const std::string& prefix,
	                                      long* id_out, std::string* suffix_out)
	{
		if (name.rfind(prefix, 0) != 0) return false;
		const std::size_t digits_at = prefix.size();
		std::size_t i = digits_at;
		while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
		if (i == digits_at) return false;
		static const std::string kExt = ".snapshot";
		if (name.compare(i, kExt.size(), kExt) != 0) return false;
		const std::string suffix = name.substr(i + kExt.size());
		if (suffix != "" && suffix != ".part" && suffix != ".killed" && suffix != ".failed")
			return false;
		*id_out = std::stol(name.substr(digits_at, i - digits_at));
		*suffix_out = suffix;
		return true;
	}


	std::string swmm_triton::dispose_snapshot_artifacts(bool keep_previous)
	{
		if (rank_ != 0) return std::string();
		if (snapshot_path_stem.empty()) return std::string();   // snapshots disabled by cfg

		const std::filesystem::path stem(snapshot_path_stem);
		const std::filesystem::path dir = stem.parent_path();
		const std::string prefix = stem.filename().string() + "_cp";

		// Collect first, act second: renaming or removing entries under an open
		// directory_iterator leaves unspecified whether it sees the new names.
		std::vector<std::filesystem::path> published, markers;
		long max_prev = 0;
		{
			std::error_code ec;
			for (const auto& e : std::filesystem::directory_iterator(dir, ec))
			{
				if (ec) break;
				const std::string name = e.path().filename().string();

				// k scans every "*.prev{j}" in the directory, not only this
				// stem's, so the rename target cannot exist anywhere.
				const std::size_t p = name.rfind(".prev");
				if (p != std::string::npos && p + 5 < name.size() &&
				    name.find_first_not_of("0123456789", p + 5) == std::string::npos)
				{
					const long j = std::stol(name.substr(p + 5));
					if (j > max_prev) max_prev = j;
					continue;
				}

				long id = 0;
				std::string suffix;
				if (!parse_snapshot_entry_name(name, prefix, &id, &suffix)) continue;
				(suffix.empty() ? published : markers).push_back(e.path());
			}
		}
		const std::string prev_suffix = ".prev" + std::to_string(max_prev + 1);

		auto keep = [&](const std::filesystem::path& from, std::error_code& rec) {
			std::filesystem::rename(from, from.string() + prev_suffix, rec);
		};

		// Published names first: a survivor is the cross-run hazard, so it
		// stops the clean start.
		for (const auto& f : published)
		{
			std::error_code rec;
			if (keep_previous) keep(f, rec);
			else               std::filesystem::remove(f, rec);
			if (rec)
			{
				std::cerr << ERROR "Clean start refused: the earlier run's SWMM state "
				             "snapshot " << f.string() << " could not be "
				          << (keep_previous ? "renamed" : "removed")
				          << " (" << rec.message() << ").\n"
				          << "         Left in place, a later resume of this run at "
				             "that checkpoint could load the earlier run's SWMM state. "
				             "Remove or move it by hand, then start again."
				          << std::endl;
				return f.string();
			}
		}

		for (const auto& f : markers)
		{
			const bool is_failed = f.extension() == ".failed";
			std::error_code rec;
			if (keep_previous || is_failed) keep(f, rec);
			else                            std::filesystem::remove(f, rec);
			if (rec)
				std::cerr << WARN "Could not dispose of the earlier run's snapshot "
				             "marker " << f.string() << " (" << rec.message()
				          << "); it is not a snapshot and is never loaded, so the "
				             "clean start continues." << std::endl;
		}
		return std::string();
	}


	int swmm_triton::highest_live_marker_above(int checkpoint_id) const
	{
		if (snapshot_path_stem.empty()) return -1;

		const std::filesystem::path stem(snapshot_path_stem);
		const std::filesystem::path dir = stem.parent_path();
		const std::string prefix = stem.filename().string() + "_cp";

		long highest = -1;
		std::error_code ec;
		for (const auto& e : std::filesystem::directory_iterator(dir, ec))
		{
			if (ec) break;
			long id = 0;
			std::string suffix;
			if (!parse_snapshot_entry_name(e.path().filename().string(), prefix, &id, &suffix))
				continue;
			if (suffix.empty() || id <= (long) checkpoint_id || id <= highest) continue;
			// A marker beside a published name at the same id has lapsed: the
			// later write at that id succeeded.
			std::error_code sec;
			if (std::filesystem::exists(snapshot_path_for((int) id), sec)) continue;
			highest = id;
		}
		return (int) highest;
	}


	void swmm_triton::replay_exchange_history(value_t up_to_time, int checkpoint_id,
	                                          const char* reason)
	{
		if (rank_ != 0) return;
		const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/true);
		// The text through "t=" and the number after it stay byte-identical to
		// the pre-sentence form; the [resume-event ...] fields are appended so
		// existing parsers and matchers keep working.
		std::cerr << IN "SWMM exchange history replayed to t=" << up_to_time
		          << " s (" << rec_count << " steps); resuming live segment"
		          << " [resume-event checkpoint=" << checkpoint_id
		          << " path=replay reason=" << (reason ? reason : "absent") << "]"
		          << std::endl;
	}


	long swmm_triton::position_exchange_log(value_t up_to_time, bool replay_steps)
	{
		if (rank_ != 0) return 0;

		const std::size_t header_bytes = EXCHANGE_LOG_HEADER_BYTES;
		const std::size_t record_bytes = sizeof(value_t) * (1 + exchange_num_nodes);
		const value_t EPS = (value_t)1e-6;

		std::ifstream in(exchange_log_path, std::ios::binary);
		if (!in.is_open())
		{
			std::cerr << ERROR "Coupled TRITON-SWMM resume requires the exchange-replay side-file,\n"
			          << "         but it was not found: " << exchange_log_path << "\n"
			          << "         This checkpoint predates the resume fix (no side-file was written).\n"
			          << "         Re-run this coupled simulation from a clean start (checkpoint_id=0)."
			          << std::endl;
			exit(EXIT_FAILURE);
		}

		// --- validate header, in the order the diagnostics depend on ---
		const exchange_log_header_t hdr = read_exchange_log_header(in);
		const exchange_header_status st =
			validate_exchange_log_header(hdr,
			                             static_cast<int32_t>(exchange_num_nodes),
			                             EXCHANGE_LOG_VALUE_WIDTH);

		if (st == exchange_header_status::bad_magic)
		{
			std::cerr << ERROR "Exchange-replay side-file has a bad/missing header: "
			          << exchange_log_path << "\n";
			if (hdr.magic == EXCHANGE_LOG_MAGIC)
				std::cerr << "         It carries the pre-versioning (V1) magic, which recorded no\n"
			          << "         stored float width and so could not be replayed safely.\n";
			std::cerr << "         Re-run this coupled simulation from a clean start (checkpoint_id=0)."
			          << std::endl;
			exit(EXIT_FAILURE);
		}
		if (st == exchange_header_status::node_count_mismatch)
		{
			std::cerr << ERROR "Exchange-replay side-file node count (" << hdr.num_nodes
			          << ") does not match this run (" << exchange_num_nodes
			          << "). The .inp / coupling changed between runs." << std::endl;
			exit(EXIT_FAILURE);
		}
		if (st == exchange_header_status::value_width_mismatch)
		{
			std::cerr << ERROR "Exchange-replay side-file stores " << hdr.value_width
			          << "-byte values but this build's value_t is " << EXCHANGE_LOG_VALUE_WIDTH
			          << " bytes.\n"
			          << "         The side-file was written by a build of the other precision\n"
			          << "         (USE_SINGLE_PRECISION differs). Replaying it would stride every\n"
			          << "         record wrongly. Re-run with the writing build's precision, or\n"
			          << "         from a clean start (checkpoint_id=0)." << std::endl;
			exit(EXIT_FAILURE);
		}

		// --- walk records 0..up_to_time (SWMM-only, rank 0) ---
		// With replay_steps each record is pushed through swmm_step, rebuilding
		// SWMM's state and its .out/stats for the full window. Without it the
		// records are only counted: the snapshot has already restored the state
		// those steps would have produced, and the walk exists solely to find
		// where the live segment starts.
		std::vector<value_t> q(exchange_num_nodes);
		value_t dt;
		double elapsed = 0.0;          // swmm_step writes elapsed days here; not used by TRITON
		value_t cum = (value_t)0.0;
		long rec_count = 0;

		while (true)
		{
			in.read(reinterpret_cast<char*>(&dt), sizeof(value_t));
			if (in.gcount() != (std::streamsize)sizeof(value_t)) break;                       // EOF/partial
			in.read(reinterpret_cast<char*>(q.data()), sizeof(value_t) * exchange_num_nodes);
			if (in.gcount() != (std::streamsize)(sizeof(value_t) * exchange_num_nodes)) break; // partial -> stop
			if (cum >= up_to_time - EPS) break;                                               // reached t_k
			if (replay_steps)
				stop_on_swmm_error(swmm_step(&elapsed, q.data(), global_new_depth, dt), "swmm_step",
				                   "while replaying the exchange history for this resume");
			cum += dt;
			rec_count++;
		}
		in.close();

		if (cum < up_to_time - EPS)
		{
			std::cerr << ERROR "Exchange-replay side-file is incomplete for this checkpoint: reached\n"
			          << "         t=" << cum << " s but checkpoint_id requires " << up_to_time
			          << " s (side-file/checkpoint mismatch, e.g. truncated by a hard kill).\n"
			          << "         Re-run this coupled simulation from a clean start (checkpoint_id=0)."
			          << std::endl;
			exit(EXIT_FAILURE);
		}

		// --- drop any stale tail (a previous longer allocation) and any partial kill-record,
		//     so appending the live segment yields a clean 0..end series ---
		std::error_code ec;
		std::filesystem::resize_file(exchange_log_path, header_bytes + (std::size_t)rec_count * record_bytes, ec);
		if (ec)
		{
			std::cerr << ERROR "Could not resize exchange-replay side-file: " << ec.message() << std::endl;
			exit(EXIT_FAILURE);
		}

		// --- reopen for append: the live segment continues the series after the
		//     records walked above. The side-file stays a complete 0..end series
		//     whichever path positioned it, so a later resume can still take the
		//     replay fallback if its snapshot is missing. ---
		exchange_log.close();
		exchange_log.open(exchange_log_path, std::ios::binary | std::ios::app);
		if (!exchange_log.is_open())
		{
			std::cerr << ERROR "Could not reopen exchange-replay side-file for append: "
			          << exchange_log_path << std::endl;
			exit(EXIT_FAILURE);
		}

		return rec_count;
	}


	// ---------------------------------------------------------------------------
	// Full-precision state snapshot: one file per checkpoint id, keyed rather
	// than rolling (see snapshot_path_stem) so a resume from an older checkpoint
	// takes the replay instead of loading a state from the future of its t_k.
	// ---------------------------------------------------------------------------

	std::string swmm_triton::snapshot_path_for(int checkpoint_id) const
	{
		return snapshot_path_stem + "_cp" + std::to_string(checkpoint_id) + ".snapshot";
	}


	// ".part" must match SNAPSHOT_PART_SUFFIX in snapshot.c: the resume has to
	// find the file the writer named. ".killed" is what the resume renames a
	// surviving ".part" to, so the cut-off write stays visible to an operator.
	std::string swmm_triton::snapshot_part_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".part";
	}


	std::string swmm_triton::snapshot_killed_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".killed";
	}


	// ".failed" is the WRITER's marker and must match SNAPSHOT_FAILED_SUFFIX in
	// snapshot.c.
	std::string swmm_triton::snapshot_failed_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".failed";
	}


	void swmm_triton::write_state_snapshot(int checkpoint_id)
	{
		if (rank_ != 0) return;
		if (snapshot_path_stem.empty()) return;   // snapshots disabled by cfg, or init_swmm not run

		// Flush the .out first so the period count the snapshot is about to
		// record is a claim about bytes on disk: fwrite is stdio-buffered, and a
		// hard kill after the snapshot write would otherwise leave the resumed
		// run seeking past EOF into a zero-filled hole that parses as data.
		output_flush();

		const std::string path = snapshot_path_for(checkpoint_id);
		if (snapshot_save(path.c_str()) != 0)
		{
			// Not fatal: aborting a live run over one checkpoint's snapshot trades
			// a certain large loss for an uncertain small one. Nothing is
			// published (snapshot_save renames into place only after its length
			// check). The two texts are keyed on one read of the ".failed"
			// marker, because the next resume differs on exactly that fact: it
			// stops on a ".failed"; with no marker the orchestration resumes from
			// the newest published snapshot below, and a launch at this
			// checkpoint itself replays.
			const std::string failed = snapshot_failed_path_for(checkpoint_id);
			std::error_code fec;
			const bool haveFailed = std::filesystem::exists(failed, fec);
			std::cerr << WARN "Could not write SWMM state snapshot: " << path << "\n";
			if (haveFailed)
				std::cerr << "         Nothing was published under that name. The "
				             "writer refused the model state or failed its own length "
				             "check, and kept " << failed << ", so a resume from "
				             "checkpoint " << checkpoint_id << " will STOP with an "
				             "error naming it." << std::endl;
			else
				std::cerr << "         Nothing was published under that name and no "
				             "marker was kept. A resume should start from the newest "
				             "published snapshot below checkpoint " << checkpoint_id
				          << "; a resume launched at checkpoint " << checkpoint_id
				          << " itself replays the exchange history." << std::endl;
			return;
		}

		// Keep at most the current and the immediately previous snapshot; the
		// exchange side-file remains the unbounded-history fallback.
		// swmm_snapshot_keep_all skips the prune (which route a resume takes
		// then no longer depends on how far the run advanced).
		if (!snapshot_keep_all && checkpoint_id >= 2)
		{
			std::error_code ec;
			std::filesystem::remove(snapshot_path_for(checkpoint_id - 2), ec);
		}
	}


	swmm_triton::SnapshotRestore
	swmm_triton::try_restore_state_snapshot(int checkpoint_id, value_t up_to_time)
	{
		if (rank_ != 0) return SnapshotRestore::Absent;
		if (snapshot_path_stem.empty()) return SnapshotRestore::Absent;

		const std::string path = snapshot_path_for(checkpoint_id);

		// THE KILLED-WRITE CONSULTATION, published artifact first. Under the
		// temp-then-rename publish a snapshot at N exists only after a complete
		// write, so a published snapshot proves any marker at N stale (Arm 1).
		// Arm 2 keys on ".part" OR ".killed" OR ".failed": keyed on ".part"
		// alone, the rename below would make every resume after the first see
		// only a ".killed", fall through to Absent and enter the replay with no
		// refusal. Arm 3 is the snapshot load below, unchanged.
		{
			const std::string part   = snapshot_part_path_for(checkpoint_id);
			const std::string killed = snapshot_killed_path_for(checkpoint_id);
			const std::string failed = snapshot_failed_path_for(checkpoint_id);
			std::error_code   ec;

			// Each name is read once: two reads of one name can disagree under a
			// concurrent writer, and the arm and its message would then describe
			// two different directory states.
			const bool havePublished = std::filesystem::exists(path,   ec);
			const bool havePart      = std::filesystem::exists(part,   ec);
			const bool haveKilled    = std::filesystem::exists(killed, ec);
			const bool haveFailed    = std::filesystem::exists(failed, ec);

			if (havePublished)
			{
				// ARM 1 -- the marker has lapsed; remove it where it is superseded.
				// A ".failed" is evidence of a writer failure and only a clean
				// start disposes of it.
				if (haveKilled) std::filesystem::remove(killed, ec);
			}
			else if (havePart || haveKilled || haveFailed)
			{
				// ARM 2 -- the marker is LIVE.
				if (havePart)
				{
					std::error_code rec;
					std::filesystem::rename(part, killed, rec);
					if (rec)
						std::cerr << WARN "Could not rename the killed-write marker "
						          << part << " to " << killed << " (" << rec.message()
						          << "); the refusal below still holds." << std::endl;
				}

				// The highest complete snapshot this run can actually find, by
				// existence check: cp{N-1} may not exist (the first checkpoint
				// written is 1, and keep-2 may have pruned it).
				int highest = -1;
				for (int cid = checkpoint_id - 1; cid >= 1; --cid)
				{
					std::error_code sec;
					if (std::filesystem::exists(snapshot_path_for(cid), sec))
					{ highest = cid; break; }
				}

				// Precedence ".failed" over ".killed" over ".part": the writer's own
				// verified failure is the most specific diagnosis, and it STOPS
				// where a cut-off write may resume from an earlier snapshot. The
				// losing markers stay; the precedence only selects the route.
				if (haveFailed)
				{
					// The opt-in override, swmm_snapshot_failed_replay=1, for a run
					// that needs correct results now: the same full replay an Absent
					// snapshot takes, so its results are that route's. The marker is
					// kept (the only evidence of the failure, and it sends every
					// later resume here again); no stop line is printed, and the
					// replay marker carries the one resume-event row. Only ".failed"
					// reaches here, so a cut-off write keeps its step-back route.
					if (snapshot_failed_replay)
					{
						std::cerr << WARN "Coupled resume at checkpoint " << checkpoint_id
						          << ": the SWMM state snapshot write for this checkpoint "
						             "was REFUSED by the writer or FAILED its own length "
						             "check, and swmm_snapshot_failed_replay=1 is set, so "
						             "this resume takes the FULL EXCHANGE REPLAY from t=0 "
						             "instead of stopping.\n"
						          << "         The marker " << failed << " is kept. The "
						             "cause of the failed write is NOT diagnosed or "
						             "cleared: a full disk or a structural defect in the "
						             "model state that produced it may still be present, "
						             "and the next checkpoint's snapshot write may fail "
						             "the same way.\n"
						          << "         The replay reproduces the run's state "
						             "exactly; it costs time that grows with t="
						          << up_to_time << " s." << std::endl;
						return SnapshotRestore::FailedWriteReplay;
					}

					// A failed or refused write STOPS, with or without an earlier
					// snapshot, and offers none as a route: its cause (a full disk,
					// or a structural defect in the model state) would still be
					// present on a resume from an earlier one.
					std::cerr << ERROR "Coupled resume refused at checkpoint "
					          << checkpoint_id << ": the SWMM state snapshot write "
					             "for this checkpoint was REFUSED by the writer or "
					             "FAILED its own length check.\n"
					          << "         The marker " << failed << " survives and "
					             "no snapshot was ever published at " << path << ". "
					             "A marker holding one text line records a refusal of "
					             "the model state; one holding snapshot bytes records "
					             "a write that did not match what the running model "
					             "requires.\n"
					          << "         This resume STOPS, and it does not continue "
					             "from an earlier snapshot, because the cause may still "
					             "be present. Inspect the marker and clear the cause "
					             "before relaunching.\n"
					          << "         To get correct results now without "
					             "diagnosing the cause, relaunch with "
					             "swmm_snapshot_failed_replay=1 in the cfg, which takes "
					             "the full exchange replay from t=0 instead of stopping."
					          << std::endl;
				}
				else
				{
					std::cerr << ERROR "Coupled resume refused at checkpoint "
					          << checkpoint_id << ": the SWMM state snapshot write "
					             "for this checkpoint was KILLED before it completed.\n"
					          << "         The marker " << killed << " survives and no "
					             "snapshot was ever published at " << path << ". That "
					             "is a measured cause, not an inference: nothing but an "
					             "interrupted write leaves this pair of states.\n";

					if (highest >= 1)
						std::cerr << "         The highest COMPLETE snapshot found below "
						             "this checkpoint is " << snapshot_path_for(highest)
						          << " (checkpoint " << highest << "); resuming this "
						             "member from there is the cheap route.\n";
					else
						std::cerr << "         NO complete snapshot exists below this "
						             "checkpoint, so a clean start (checkpoint_id=0) is "
						             "the only route.\n";

					std::cerr << "         The exchange replay is NOT offered as a "
					             "substitute. It was measured at 11,107 s on a run that "
					             "was cancelled still replaying, and each successive "
					             "resume replays a longer prefix, so the sequence does "
					             "not converge."
					          << std::endl;
				}

				// One machine-readable line per resume, for the orchestration that
				// relaunches the run. Parsed by the toolkit: no colour prefix, and
				// not to be reworded. A cut-off write with a published snapshot
				// below steps back to it; a failed write stops (the snapshot below
				// is named for information only); with nothing published below only
				// a clean start remains, and the reason names the marker found.
				if (highest >= 1 && !haveFailed)
					std::cerr << "SWMM snapshot step-back required: checkpoint "
					          << checkpoint_id << " unusable [resume-event checkpoint="
					          << checkpoint_id << " path=step-back reason=killed-write"
					             " target=" << highest << "]" << std::endl;
				else if (highest >= 1)
					std::cerr << "SWMM snapshot stop: checkpoint " << checkpoint_id
					          << " unusable [resume-event checkpoint=" << checkpoint_id
					          << " path=stop reason=write-failed target=" << highest
					          << "]" << std::endl;
				else
					std::cerr << "SWMM snapshot step-back impossible: checkpoint "
					          << checkpoint_id << " unusable, no published snapshot "
					             "below it [resume-event checkpoint=" << checkpoint_id
					          << " path=clean-start reason="
					          << (haveFailed ? "write-failed" : "killed-write")
					          << " target=0]" << std::endl;

				return SnapshotRestore::KilledWrite;
			}
		}

		char msg[512];
		msg[0] = '\0';

		const int rc = snapshot_load(path.c_str(), msg, (int)sizeof(msg));
		if (rc != SNAPSHOT_OK)
		{
			if (rc == SNAPSHOT_ABSENT)
			{
				std::cerr << INFO "SWMM state snapshot not used (" << msg << ").\n"
				          << "         Falling back to replaying the exchange history."
				          << std::endl;
				return SnapshotRestore::Absent;
			}
			// No cause is named: msg carries the quantity that disagreed. A
			// writer aborting mid-payload produced exactly this refusal on files
			// written by the build that read them, so "different build or
			// corrupt" was not a safe attribution.
			std::cerr << ERROR "SWMM state snapshot is present but UNUSABLE ("
			          << msg << ").\n"
			          << "         Refusing to continue. The file exists and "
			             "disagrees with this build's reader; the parenthesis "
			             "above carries the specific quantity that disagreed, "
			             "and no cause is asserted here.\n"
			          << "         Re-run from a clean start (checkpoint_id=0). "
			             "The exchange replay is an exit only where a replay "
			             "side-file covering this checkpoint survives, and the "
			             "campaign that produced this code path measured "
			             "replayed=0 across all 45 of its per-model logs -- so "
			             "do not count on it without checking." << std::endl;
			return SnapshotRestore::Unreadable;
		}

		// The snapshot restored SWMM's accumulators and routing state, but the
		// array TRITON reads is filled at exactly one site inside routing_execute,
		// which no restore passes through. Call the factored-out writer rather
		// than duplicating the loop that defines the inflow-node index space.
		// The replay fallback must NOT call it: its steps have already left
		// global_new_depth[] at exactly these values.
		routing_exportInflowNodeDepths(global_new_depth);

		// Position the side-file so the live segment appends after t_k without
		// replaying: the state those steps would have produced is already restored.
		const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/false);

		// Position the .out for the resumed segment from the restored Nperiods.
		// A failure here cannot fall back: the accumulators are already restored
		// and a replay on top of them would double-apply the 0..t_k prefix.
		{
			char pmsg[512];
			pmsg[0] = '\0';
			if (output_positionForResume(pmsg, (int)sizeof(pmsg)) != 0)
			{
				std::cerr << ERROR "SWMM state snapshot loaded but the binary output "
				             "file cannot carry the resumed segment (" << pmsg << ").\n"
				          << "         Refusing to continue: falling back to the exchange replay\n"
				          << "         would re-apply the 0..t_k prefix on top of restored state.\n"
				          << "         Re-run from a clean start (checkpoint_id=0)." << std::endl;
				return SnapshotRestore::OutputMismatch;
			}
		}

		// Same "...to t=" shape as the replay marker so one matcher discriminates both.
		std::cerr << IN "SWMM state restored from snapshot to t=" << up_to_time
		          << " s (" << rec_count << " steps skipped); resuming live segment"
		          << " [resume-event checkpoint=" << checkpoint_id
		          << " path=snapshot]"
		          << std::endl;

		// THE STEP-BACK CONFIRMATION. A live marker above this checkpoint means
		// the orchestration stepped back over an unusable one, and this line
		// records both ids in the log of the launch that actually resumes (the
		// launch that found the unusable checkpoint may have had its log
		// overwritten). Same parsed contract as the KilledWrite line.
		const int stepped_from = highest_live_marker_above(checkpoint_id);
		if (stepped_from >= 1)
			std::cerr << "SWMM snapshot step-back taken: resumed from checkpoint "
			          << checkpoint_id << " [resume-event checkpoint=" << checkpoint_id
			          << " path=step-back-taken from=" << stepped_from << "]"
			          << std::endl;
		return SnapshotRestore::Restored;
	}


	const char* swmm_triton::classify_missing_snapshot(int checkpoint_id) const
	{
		if (rank_ != 0 || snapshot_path_stem.empty()) return "absent";

		// All three verdicts ground out in the same errno; the discriminator is
		// the shape of the snapshot series on disk: no snapshot at all means the
		// run predates the mechanism, a snapshot for a HIGHER id means keep-2
		// pruned the one this resume wants, anything else is a plain miss.
		const std::filesystem::path stem(snapshot_path_stem);
		const std::filesystem::path dir = stem.parent_path();
		const std::string prefix = stem.filename().string() + "_cp";

		bool any = false, higher = false;
		std::error_code ec;
		for (const auto& e : std::filesystem::directory_iterator(dir, ec))
		{
			if (ec) break;
			const std::string name = e.path().filename().string();
			if (name.rfind(prefix, 0) != 0) continue;
			if (e.path().extension() != ".snapshot") continue;
			const std::string idtxt =
				name.substr(prefix.size(), name.size() - prefix.size() - 9);
			if (idtxt.empty() ||
			    idtxt.find_first_not_of("0123456789") != std::string::npos) continue;
			any = true;
			if (std::stol(idtxt) > (long)checkpoint_id) higher = true;
		}
		if (!any)   return "pre-snapshot-checkpoint";
		if (higher) return "retention-collision";
		return "absent";
	}


	int swmm_triton::calc_swmm_node_col(value_t  node_x, value_t  xllc, value_t  cell_size_)
	{
		return ceil(((node_x - xllc) / cell_size_)+1e-16) - 1;
	}


	int swmm_triton::calc_swmm_node_row(value_t  node_y, value_t  yllc, value_t  cell_size_, int nrows)
	{
		return ceil((nrows - ((node_y - yllc) / cell_size_))+1e-16) - 1;
	}

	void swmm_triton::local_to_global(value_t*  local, value_t*  global, int* dict)
	{
		for(int i=0;i<global_num_of_swmm_links;i++){
			global[dict[i]]=local[i];
		}
	}


	void swmm_triton::global_to_local(value_t*  global, value_t*  local, int* dict)
	{
		for(int i=0;i<global_num_of_swmm_links;i++){
			local[i]=global[dict[i]];
		}

	}




/** @brief It computes the flow exchange (in ft3/s) between TRITON and SWMM for each swmm node
*
*  @param size Array size
*  @param dx Cell size
*  @param dt Time step size
*  @param h_arr Water depth array
*  @param qx_arr Discharge in x direction array
*  @param qy_arr Discharge in y direction array
*  @param hextra Minimum depth (tolerance below water is at rest)
*  @param pos_arr Flow location position array
*	@param swmm_loss Manhole's loss coefficient
*	@param swmm_d Manhole's diameter
*	@param swmm_max_depth maximumDepth: distance between the bed of the flume and the invert level of the sewer
*	@param swmm_new_depth new_depth: pressure head in the pipe
*	@param exchange_q flow exchange between TRITON and SWMM
*/

	template<typename T>
	void compute_swmm_triton_exchange(int size, T dx, T dt, T *h_arr, T *qx_arr, T *qy_arr, T hextra, int *pos_arr,
	T *swmm_loss, T *swmm_d, T *swmm_max_depth, T *swmm_new_depth, T *exchange_q)
	{
		#include "kokkos_utils.h"
		triton::parallel_for( AUTO_LABEL() , size , KOKKOS_LAMBDA (int id) {
			T flow=0.0;
			int sid = pos_arr[id];
			T hij = h_arr[sid];
			T hold = hij;
			T new_depth=swmm_new_depth[id];
			T max_depth=swmm_max_depth[id];
			T loss=swmm_loss[id];
			T diam=swmm_d[id];
			T areaM = PI_*0.25*diam*diam;                       // Manhole's area


		   // Case 1 (Surface to sewer - weir-type equation)
			if(hij>0.0 && new_depth <= max_depth){
				flow=-(2.0/3.0)*loss*PI_*diam*sqrt(2.0*_G_*hij)*hij;
			}
			// Case 2 (Surface to sewer)
			else if(hij>0.0 && new_depth <= max_depth + hij){
				flow=-loss*areaM*sqrt(2.0*_G_*(hij + max_depth - new_depth));
			}
			// Case 3 (Sewer to surface)
			else if(new_depth > max_depth + hij){
				flow=loss*areaM*sqrt(2.0*_G_*(new_depth - max_depth - hij ) );
			}

			T h_src = (flow * dt) / (dx * dx);
			hij += h_src;

			//if water is below hextra, velocities are removed
			if (hij < hextra)
			{
				//negative water removed
				if (hij < EPS12)
				{
					hij = 0.0;
					flow=-hold*dx*dx/dt;
				}

				qx_arr[sid] = 0.0;
				qy_arr[sid] = 0.0;
			}

			h_arr[sid]=hij;
			exchange_q[id]=-flow/FT3_TO_M3_FACTOR; // m3/s to ft3/s

		});
	}

}

#endif // TRITON_SWMM

#endif
