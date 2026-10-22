/** @file swmm_triton.h
 *  @brief Couples the TRITON surface solver to the SWMM sewer network, and makes
 *         a hotstart resume reproduce the full-window SWMM trajectory.
 *
 *  TRITON solves the 2D surface flow and SWMM solves the sewer network; this file
 *  is the seam between them. It maps each SWMM surface node onto the TRITON cell
 *  containing it, partitions that node set across the MPI ranks owning those
 *  cells, and exchanges a weir/orifice flux between surface and sewer each step.
 *
 *  THE SEAM'S HARD PROBLEM IS RESUMING, and most of this file exists for it. SWMM
 *  holds routing state and mass-balance accumulators that no TRITON checkpoint
 *  records, and init_swmm calls swmm_start() unconditionally, which reopens the
 *  .out from the beginning -- so a naive resume restarts the sewer model at t=0
 *  and yields a report covering only the post-checkpoint segment while its header
 *  still claims the full window: a wrong answer that reads healthy. Two mechanisms
 *  below prevent that. A durable per-step exchange side-file can be replayed
 *  0..t_k, which is exact but whose cost grows with t_k; a per-checkpoint
 *  full-precision state snapshot makes the restore cost independent of t_k, with
 *  the replay retained as its fallback.
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

// Defined in the vendored routing.c, and declared here rather than in funcs.h so
// each declaration sits with its only caller (init_swmm).
// routing_nodeHasFlowInflow reports whether node j's external inflows include a
// FLOW inflow; it is read only when no coupling node is declared, where the
// coupled list is the FLOW-inflow set, and by the startup disclosure line.
// routing_setCouplingNodes builds the slot map that gives each coupling node its
// exchange slot, from the nodes' SWMM indices in slot order; it returns a SWMM
// error code.
extern "C" int routing_nodeHasFlowInflow(int j);
extern "C" int routing_setCouplingNodes(const int* nodeIndex, int n);

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

	// Magic header for the exchange-replay side-file.
	//
	// V1 (RETAINED, and never re-used as the current magic): a two-field header
	// {int32 magic, int32 num_nodes}. BOTH fields are precision-independent, so a
	// side-file written by one value_t build and replayed by the other passed every
	// guard and then strode each record by the wrong number of bytes -- silent
	// corruption inside a correctness mechanism. The symbol is kept so the V2 reader
	// can NAME what it rejected instead of reporting a generic parse failure.
	static constexpr int32_t EXCHANGE_LOG_MAGIC = 0x53574D4D;        // ASCII "SWMM"

	// V2 (current): a three-field header {int32 magic, int32 num_nodes, int32 value_width}.
	// The magic bump is the load-bearing half of the guard. A V1 file read by a
	// three-field reader consumes the first record's leading value_t as the third
	// header field, so the reader must be able to tell the two generations apart
	// BEFORE it interprets that field at all -- which is why the magic is checked
	// first and a legacy file is reported as a bad header, never as a wrong width.
	static constexpr int32_t EXCHANGE_LOG_MAGIC_V2 = 0x53574D32;     // ASCII "SWM2"

	// The value_t width this build writes into, and demands of, a V2 side-file.
	static constexpr int32_t EXCHANGE_LOG_VALUE_WIDTH = static_cast<int32_t>(sizeof(value_t));

	// Bytes occupied by the V2 header. position_exchange_log() strides records from
	// this, on BOTH the replay path and the snapshot path, so it is declared once
	// beside the layout rather than recomputed at either use site.
	static constexpr std::size_t EXCHANGE_LOG_HEADER_BYTES = 3 * sizeof(int32_t);

	// --- V2 exchange-log header: the single expression of the layout ---------------
	// open_exchange_log_truncate() writes through these and position_exchange_log()
	// reads through them, rather than either laying out the bytes itself, so the write
	// and read sides cannot drift, and a test can exercise the shipped layout with no
	// solver run and no SWMM call.

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
		/// false when the file ended mid-header. Kept separate from the field values so
		/// a V1 file that was killed before its first record still exposes its magic and
		/// earns the legacy-specific diagnostic.
		bool    complete    = false;
	};

	/** @brief Why a side-file header was refused. One value per diagnostic. */
	enum class exchange_header_status
	{
		ok = 0,
		bad_magic,             ///< not a readable V2 header: a V1 legacy file, or a short/absent header
		node_count_mismatch,   ///< the .inp / coupling changed between runs
		value_width_mismatch   ///< written by a build whose value_t differs from this one
	};

	/** @brief Reads the V2 header, tolerating a short or empty file.
	 *
	 *  A field that cannot be read in full is left at 0 and `complete` is left false,
	 *  so a truncated or absent header is refused as a bad header rather than
	 *  surfacing as a stream error or -- the case a test caught here -- as a spurious
	 *  "node count 0 does not match" that blames the .inp. Fields that WERE read are
	 *  preserved, so an 8-byte V1 file still earns the legacy-specific diagnostic.
	 *  Stream state is left as the read left it: on a complete header the get pointer
	 *  sits at the first record.
	 */
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

	/** @brief Validates a side-file header against the running build.
	 *
	 *  THE ORDER IS THE SPECIFICATION, not a style choice. Completeness and magic are
	 *  checked FIRST -- together they are the "is this a readable V2 header at all"
	 *  question -- so that a V1 legacy file, whose third "field" is really the first
	 *  record's leading value_t bytes, is reported as a bad header and never as a
	 *  wrong width. Node count is checked before width because a node-count disagreement
	 *  means the .inp / coupling changed, which is a different and more likely
	 *  operator error than a precision mismatch, and because it matches the field
	 *  order on disk.
	 *
	 *  @param h                    header as read from the side-file
	 *  @param expected_nodes       this run's global SWMM surface-node count
	 *  @param running_value_width  sizeof(value_t) in the build about to replay
	 */
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
	// (file) order. A coupled list that differs from SWMM's -- a node missing, an
	// extra node, or the same nodes in another order -- therefore hands each surface
	// cell some other node's sewer flow, with no error. These helpers reproduce SWMM's
	// own section rule, tokenizer and FLOW test (input.c getTokens and match,
	// inflow.c), and init_swmm then checks the result against the running engine
	// rather than trusting the reproduction.

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

	/** @brief What the common read of an .inp file yields, and the coupled list. */
	struct inp_coupled_list_t
	{
		std::vector<std::string> names;              ///< coupled nodes, in SWMM node order
		std::vector<std::string> junction_names;     ///< [JUNCTIONS] data lines, file order
		std::vector<double>      junction_max_depth; ///< each junction's third token (0 if absent)
		std::vector<std::string> coord_names;        ///< [COORDINATES] data lines, file order
		std::vector<double>      coord_x;
		std::vector<double>      coord_y;
		/// [INFLOWS] data lines as (node, constituent) tokens, file order. Read in every
		/// branch, because reading is not selecting; only the legacy branch derives
		/// coupling nodes from them.
		std::vector<std::pair<std::string, std::string>> inflow_lines;
		std::vector<std::string> pollutants;         ///< [POLLUTANTS] names, upper case
	};

	/** @brief Reads [JUNCTIONS], [COORDINATES], [INFLOWS] and [POLLUTANTS] from an .inp.
	 *
	 *  A section starts at any line whose first token begins with '[' and runs to the
	 *  next such line, wherever the line is indented and whatever comment follows it,
	 *  and every occurrence of a section is read -- as SWMM reads its input. It selects
	 *  no coupling node: names is left empty for resolve_coupling_nodes to fill.
	 */
	static inline inp_coupled_list_t read_inp_sections(std::istream& in)
	{
		inp_coupled_list_t r;
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
				r.inflow_lines.emplace_back(tok[0], tok[1]);
			}
			else if (sect == POLLUTANTS) {
				r.pollutants.push_back(swmm_upper(tok[0]));
			}
		}
		return r;
	}

	/** @brief Orders node names by their position in [JUNCTIONS].
	 *
	 *  That is the order SWMM numbers junctions, so it is SWMM's node order for a
	 *  junction-only list. A name absent from [JUNCTIONS] keeps its relative place
	 *  after them, and init_swmm refuses it. Names compare case-insensitively, as
	 *  SWMM's own lookup does.
	 */
	static inline std::vector<std::string> order_by_junction_position(const std::vector<std::string>& names,
	                                                                  const std::vector<std::string>& junction_names)
	{
		std::vector<std::string> junction_upper;
		for (const auto& j : junction_names) junction_upper.push_back(swmm_upper(j));
		std::vector<std::pair<std::size_t, std::size_t>> key;  // (sort key, list position)
		for (std::size_t i = 0; i < names.size(); ++i) {
			const auto it = std::find(junction_upper.begin(), junction_upper.end(), swmm_upper(names[i]));
			const std::size_t k = (it != junction_upper.end())
			                          ? static_cast<std::size_t>(it - junction_upper.begin())
			                          : junction_upper.size() + i;
			key.emplace_back(k, i);
		}
		std::sort(key.begin(), key.end());
		std::vector<std::string> out;
		for (const auto& k : key) out.push_back(names[k.second]);
		return out;
	}

	/** @brief The legacy coupled list: the nodes SWMM gives a FLOW inflow.
	 *
	 *  A node qualifies when an [INFLOWS] line's second token is not a [POLLUTANTS]
	 *  name and begins, case-insensitively, with FLOW (SWMM's own test), at most once
	 *  per node, ordered by order_by_junction_position. Pollutants are collected
	 *  before any inflow line is classified, because [POLLUTANTS] may follow
	 *  [INFLOWS] in the file. Used ONLY when no coupling node is declared: under a
	 *  declaration no [INFLOWS] line may add, remove or reorder a coupling node.
	 */
	static inline std::vector<std::string> derive_flow_inflow_coupled_nodes(const inp_coupled_list_t& m)
	{
		std::vector<std::string> coupled, coupled_upper;
		for (const auto& fl : m.inflow_lines) {
			const std::string param = swmm_upper(fl.second);
			if (std::find(m.pollutants.begin(), m.pollutants.end(), param) != m.pollutants.end()) continue;
			if (param.rfind("FLOW", 0) != 0) continue;
			const std::string node = swmm_upper(fl.first);
			if (std::find(coupled_upper.begin(), coupled_upper.end(), node) != coupled_upper.end()) continue;
			coupled.push_back(fl.first);
			coupled_upper.push_back(node);
		}
		return order_by_junction_position(coupled, m.junction_names);
	}

	/** @brief How the coupling nodes are declared: the cfg key swmm_coupling_nodes. */
	enum coupling_kind_t
	{
		COUPLING_LEGACY,        ///< key absent or empty: derived from [INFLOWS] FLOW lines
		COUPLING_LIST,          ///< key is a path: a one-column file of node names
		COUPLING_ALL_JUNCTIONS  ///< key is exactly ALL_JUNCTIONS
	};

	/** @brief A coupling declaration as read from the cfg (and, for a list, its file). */
	struct coupling_declaration_t
	{
		coupling_kind_t          kind = COUPLING_LEGACY;
		std::string              path;   ///< the list file (COUPLING_LIST)
		std::vector<std::string> names;  ///< the list's names, file order (COUPLING_LIST)
		std::vector<int>         lines;  ///< each name's 1-based line in the list file
	};

	/** @brief The coupling nodes a declaration selects from a parsed model.
	 *
	 *  The three branches are exclusive: a list gives exactly its names, in
	 *  [JUNCTIONS] order; ALL_JUNCTIONS gives exactly the junctions, in file order;
	 *  only with no declaration are the nodes derived from the [INFLOWS] FLOW lines,
	 *  and that derivation is called from this else and from nowhere else.
	 */
	static inline std::vector<std::string> resolve_coupling_nodes(const coupling_declaration_t& d,
	                                                              const inp_coupled_list_t& m)
	{
		if (d.kind == COUPLING_LIST) {
			return order_by_junction_position(d.names, m.junction_names);
		}
		else if (d.kind == COUPLING_ALL_JUNCTIONS) {
			return m.junction_names;
		}
		else {
			return derive_flow_inflow_coupled_nodes(m);
		}
	}

	/** @brief Reads a coupling-node list file: one name per line.
	 *
	 *  Surrounding whitespace is stripped; blank lines and lines beginning with ';'
	 *  are skipped; there is no header. Each name's 1-based line number is kept so a
	 *  refusal can name the line.
	 */
	static inline void read_coupling_list(std::istream& in, std::vector<std::string>& names, std::vector<int>& lines)
	{
		std::string line;
		int n = 0;
		while (std::getline(in, line)) {
			++n;
			const std::size_t b = line.find_first_not_of(" \t\r");
			if (b == std::string::npos) continue;
			const std::size_t e = line.find_last_not_of(" \t\r");
			const std::string name = line.substr(b, e - b + 1);
			if (name[0] == ';') continue;
			names.push_back(name);
			lines.push_back(n);
		}
	}

	/** @brief The first pair of names equal after SWMM's case fold, or {-1, -1}. */
	static inline std::pair<int, int> find_repeated_name(const std::vector<std::string>& names)
	{
		for (std::size_t i = 0; i < names.size(); ++i)
			for (std::size_t j = 0; j < i; ++j)
				if (swmm_upper(names[i]) == swmm_upper(names[j]))
					return {static_cast<int>(j), static_cast<int>(i)};
		return {-1, -1};
	}

	/** @brief The coupled list with no coupling declaration (the legacy branch).
	 *
	 *  read_inp_sections followed by resolve_coupling_nodes on an empty declaration:
	 *  the coupled list a cfg without swmm_coupling_nodes gets.
	 */
	static inline inp_coupled_list_t parse_inp_coupled_list(std::istream& in)
	{
		inp_coupled_list_t r = read_inp_sections(in);
		r.names = resolve_coupling_nodes(coupling_declaration_t(), r);
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

	/** @brief True when SWMM's flow unit is a US one (CFS, GPM or MGD).
	 *
	 *  SWMM ties its length unit to its flow unit: in a US-flow model every length the
	 *  engine returns (swmm_NODE_ELEV, swmm_NODE_MAXDEPTH) is in feet, in an SI-flow
	 *  model in metres. TRITON works in metres, so a US model's lengths are converted.
	 */
	static inline bool swmm_flow_units_are_us(int flow_units)
	{
		return flow_units >= swmm_CFS && flow_units <= swmm_MGD;
	}

	/** @brief A length the SWMM engine returned, in metres.
	 *
	 *  Multiplies a US model's feet by 0.3048, SWMM's own feet-to-metres length factor
	 *  (MperFT in consts.h; UCF(LENGTH) in swmm5.c), which the engine's getters already
	 *  apply to every length of an SI model. So a US model's rims and inverts are bit
	 *  for bit those of its SI twin; any other factor makes the two models differ.
	 */
	static inline double swmm_engine_length_m(double length, bool us)
	{
		return us ? length * 0.3048 : length;
	}

	/** @brief A coupled node's rim in metres, from the engine's full depth.
	 *
	 *  In an SI model this is coupled_node_rim. In a US model the parsed [JUNCTIONS]
	 *  token is in feet, so it is never kept: the rim is SWMM's full depth in metres.
	 */
	static inline double coupled_node_rim_m(double token, double swmm_full_depth, bool us)
	{
		return us ? swmm_engine_length_m(swmm_full_depth, true) : coupled_node_rim(token, swmm_full_depth);
	}

	/** @brief True for a SWMM report error line: optional whitespace, "ERROR ", a digit
	 *  (the form ^\s*ERROR [0-9]+). A bare substring test would also echo a title, a
	 *  WARNING naming an object, or any other report line that merely contains ERROR. */
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
	 *  line part of the message; after any other error it is unrelated text or blank. */
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
		/// `swmm_snapshot_keep_previous` is the cfg key of the same name, consumed
		/// only by the clean-start disposal. It DEFAULTS to false so a caller
		/// that does not yet pass it gets the removal behaviour, which is what
		/// an unset key means.
		void initialize(int rank, int size, std::string inp_filename, std::string project_dir, std::string output_folder, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss, const bool swmm_snapshot_disable, const bool swmm_snapshot_keep_all, const bool swmm_snapshot_keep_previous = false, const bool swmm_node_rim_from_dem = false, const std::string& swmm_coupling_nodes = std::string());


		void end_swmm(std::string output_dir);
		void stop_on_swmm_error(int code, const char* call, const char* when, const char* report_path = nullptr);   ///< rank 0: MPI_Abort with SWMM's own message when a SWMM API call returns non-zero; with report_path (the startup calls only), close SWMM first and echo its report's ERROR lines

		// --- TRITON->SWMM exchange-series persist/replay (hotstart-resume support) ---
		// The SWMM trajectory is a deterministic function of its per-step inputs
		// (global exchange flux + dt). We durably log them every step and, on resume,
		// fast-replay 0..t_k so SWMM's .out/stats rebuild the full window.
		void open_exchange_log_truncate();                       ///< clean start: fresh side-file + header
		/// Resume: replay 0..t_k, truncate, reopen for append. `reason` is the
		/// classified ground for taking this path rather than the snapshot; it
		/// is carried into the resume-event record, not used for control flow.
		void replay_exchange_history(value_t up_to_time,
		                             int checkpoint_id = 0,
		                             const char* reason = "absent");
		void log_exchange_step(value_t dt, const value_t* gq);   ///< append one step's (dt, exchange_q) record
		void flush_exchange_log();                               ///< flush (called at each checkpoint write)
		void close_exchange_log();                               ///< close (called at end_swmm)

		// --- full-precision state snapshot (bounds the resume cost) ---------
		//
		// The replay above is exact but its cost grows with t_k: a long run was
		// measured spending 11,107 s replaying before the resumed segment began
		// any work, and was cancelled still replaying, with the GPU idle
		// throughout because the device
		// vectors are not created until after the replay returns. The snapshot
		// makes the restore cost independent of t_k; the replay is RETAINED as
		// the fallback for checkpoints written before a snapshot existed.

		/// Write the full-precision SWMM state snapshot for this checkpoint.
		/// Charged to IO_TIME by the caller. Rank 0 only. A failure warns and
		/// does not abort the live run, which is worth more than one
		/// checkpoint's snapshot. The cost lands at the next resume from this
		/// checkpoint: a stop when the writer kept a ".failed" marker, otherwise
		/// a resume from an earlier snapshot or a replay.
		void write_state_snapshot(int checkpoint_id);

		/// Resume: restore SWMM's state from this checkpoint's snapshot.
		///
		/// FAIL-FAST IS THE DEFAULT AND THE RETURN IS GRADUATED. A bool could
		/// carry only "engaged / did not", so every refusal degraded to the
		/// replay -- correct numbers, slowly, and a real defect hidden. The
		/// caller's dispositions differ by cause and it could not see them:
		///
		///   Restored       -- continue on the snapshot.
		///   Absent         -- legitimate (a checkpoint predating snapshots,
		///                     or one the keep-2 retention pruned). Run the
		///                     replay fallback.
		///   Unreadable     -- present, but it DISAGREES WITH THIS BUILD'S
		///                     READER on a named quantity. `msg` carries that
		///                     quantity; no cause is asserted, because the one
		///                     this used to name was measured FALSE -- see the
		///                     refusal arm in try_restore_state_snapshot.
		///                     ABORT.
		///   OutputMismatch -- the snapshot loaded, but its output-period count
		///                     exceeds the periods the .out actually holds.
		///                     ABORT -- and it must not fall back, because the
		///                     accumulators are ALREADY restored and a replay
		///                     on top of them would double-apply the prefix.
		///   KilledWrite    -- a ".part", ".killed" or ".failed" marker survives
		///                     for this checkpoint and no published snapshot
		///                     does, so the write was cut off (".part"/".killed")
		///                     or failed the writer's own check (".failed").
		///                     ABORT -- and specifically NOT Absent, because
		///                     Absent is the arm that enters the replay and the
		///                     replay is measured not to converge at a high
		///                     checkpoint index. One machine-readable line names
		///                     what the caller's orchestration should do next:
		///                     resume from an earlier published snapshot (a cut
		///                     off write), stop (a failed write), or start clean
		///                     (no published snapshot below).
		///
		/// The cost asymmetry is what sets the default: a wrongly aborting
		/// resume costs one job and a loud message; a wrongly falling-back
		/// resume costs a completed run whose numbers are wrong and whose
		/// every artifact reads healthy.
		enum class SnapshotRestore { Restored, Absent, Unreadable, OutputMismatch,
		                             KilledWrite };

		SnapshotRestore try_restore_state_snapshot(int checkpoint_id,
		                                           value_t up_to_time);

		/// Clean start only: dispose of the EARLIER run's snapshot artifacts --
		/// every published snapshot and every ".part", ".killed" and ".failed"
		/// marker in the snapshot directory -- with the keep-previous policy
		/// latched from the cfg at initialize(). Returns the path of a published
		/// snapshot that could not be disposed of, or an empty string on success;
		/// the caller must stop the run on a non-empty return, before its first
		/// write.
		///
		/// Why published snapshots go too: a clean start removes the earlier
		/// run's checkpoint configurations but would otherwise leave its
		/// snapshots, and if the new run's write at some checkpoint then failed,
		/// a resume there would load the WRONG run's SWMM state with no error.
		/// A clean start consults no snapshot, so nothing it disposes of can be
		/// evidence it needs. On a RESUME nothing is disposed of except a
		/// ".killed" made stale by a published snapshot at the same id. On the
		/// ABORT path every marker is deliberately RETAINED -- there it is the
		/// artifact an operator needs in order to see that a write was killed
		/// or failed rather than that a snapshot was missing.
		std::string sweep_snapshot_markers();

		/// The disposal itself, with the policy passed in rather than read from
		/// the latched member, so it can be exercised for both policies without
		/// a cfg.
		///
		///   keep_previous true  -- every published name and marker is RENAMED
		///                          to "{name}.prev{k}" and kept.
		///   keep_previous false -- published names, ".part" and ".killed" are
		///                          REMOVED; a ".failed" is still renamed and
		///                          kept, because it is the evidence for a
		///                          stop and the operator's natural response
		///                          to that stop is exactly this clean start.
		///
		/// k is ONE integer for the whole call: one more than the largest j of
		/// any "*.prev{j}" name already in the directory, whatever its base, or
		/// 1 if there is none. A per-base or string maximum would let a rename
		/// land on an existing name, and rename(2) onto an existing file
		/// replaces it.
		///
		/// Published names are handled FIRST, and the first one that cannot be
		/// disposed of stops the call with its path returned and no marker
		/// touched: a surviving earlier-run snapshot is the hazard above. A
		/// marker that cannot be disposed of is a warning and the call goes on,
		/// because no marker is ever loaded as state.
		std::string dispose_snapshot_artifacts(bool keep_previous);

		/// The highest checkpoint id ABOVE `checkpoint_id` carrying a LIVE
		/// marker -- an exact "{published name}.part", ".killed" or ".failed"
		/// with no published name at that id -- or -1 if there is none.
		/// Directory names only, so it says nothing it did not read. Used after
		/// a successful restore to name the unusable checkpoint the resume
		/// stepped over.
		int highest_live_marker_above(int checkpoint_id) const;

		/// Classifies WHY no snapshot was available for this checkpoint, into
		/// the three reason values the resume-event record carries. Rank 0.
		const char* classify_missing_snapshot(int checkpoint_id) const;

		void local_to_global(value_t*  local, value_t*  global, int* dict);
		void global_to_local(value_t*  global, value_t*  local, int* dict);

		std::string get_output_folder() const { return output_folder; }

		int num_of_swmm_links = 0; /**< Number of SWMM nodes connected to the surface per subdomain */
		int num_of_swmm_nodes; /**< Number of SWMM total nodes */
		int global_num_of_swmm_links = 0; /**< Number of total (full domain) SWMM nodes connected to the surface */
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

		/// Stem for the per-checkpoint state snapshots; the checkpoint id is
		/// appended. Keyed per checkpoint rather than written to one rolling
		/// path so a resume from an OLDER checkpoint cannot silently load a
		/// LATER state -- it finds no file for its id and takes the replay.
		std::string snapshot_path_stem;

		/// Latched from the cfg key of the same name in initialize(). A MEMBER
		/// rather than a threaded parameter because its consumer is
		/// write_state_snapshot, which is called once per checkpoint long after
		/// initialize returns -- unlike swmm_snapshot_disable, whose consumer is
		/// init_swmm inside the same call and which therefore needs no member.
		/// Default-initialised false so a non-coupled or partially-constructed
		/// object retains the keep-2 default rather than an indeterminate one.
		bool snapshot_keep_all = false;

		/// Latched from the cfg key `swmm_snapshot_keep_previous` in
		/// initialize(), for the same reason as snapshot_keep_all: its consumer,
		/// the clean-start disposal, runs after initialize returns. False keeps
		/// the removal default.
		bool snapshot_keep_previous = false;

		/// Latched from the cfg key `swmm_node_rim_from_dem` in initialize(). Read only
		/// by init_swmm, to label the SWMM rims it prints as replaced when triton.h is
		/// about to set every rim from the DEM instead. False keeps SWMM's rims.
		bool node_rim_from_dem = false;

		std::string snapshot_path_for(int checkpoint_id) const;

		/// The temporary snapshot_save writes before it publishes by rename, and
		/// the marker a killed write leaves behind. Both carry a suffix on the
		/// published name, which is what makes them invisible to
		/// classify_missing_snapshot's ".snapshot"-extension filter and to the
		/// retention prune -- so a partial or killed write is never counted as a
		/// published snapshot and never pruned as one.
		std::string snapshot_part_path_for(int checkpoint_id) const;
		std::string snapshot_killed_path_for(int checkpoint_id) const;
		/// The marker snapshot_save leaves when its count pass refuses the model
		/// state (one text line) or its length check fails after the write (the
		/// written bytes). Carries the same suffix property as the two above.
		std::string snapshot_failed_path_for(int checkpoint_id) const;

		/// Position the side-file for the live segment: validate the header,
		/// walk records up to `up_to_time`, truncate any stale tail or partial
		/// kill-record, and reopen for append. With `replay_steps` true each
		/// record is pushed through swmm_step (the replay); with it false the
		/// records are only counted (the snapshot path, which has already
		/// restored the state those steps would have produced).
		///
		/// The two paths share this function because they must agree EXACTLY on
		/// where the live segment starts. Two implementations of the same
		/// striding arithmetic would be free to disagree by one record, and a
		/// one-record disagreement is invisible in every artifact.
		long position_exchange_log(value_t up_to_time, bool replay_steps);

		int calc_swmm_node_col(value_t  node_x, value_t  xllc, value_t  cell_size_);

		int calc_swmm_node_row(value_t  node_y, value_t  yllc, value_t  cell_size_, int nrows);

		void read_inp_file(std::string inp_filename, const value_t  dx, const value_t manhole_diameter, const value_t manhole_loss);


		void process_swmm_node_locations(const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd);

		void init_swmm(std::string project_dir, std::string inp_filename, const bool swmm_snapshot_disable);


		int rank_;
		int size_;
		/// The cfg's swmm_coupling_nodes value, and the declaration read_inp_file makes
		/// of it on every rank; init_swmm keys the legacy-only checks on its kind.
		std::string coupling_nodes_key;
		coupling_declaration_t coupling;
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





	void swmm_triton::initialize(int rank, int size, std::string inp_filename, std::string project_dir, std::string output_folder, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss, const bool swmm_snapshot_disable, const bool swmm_snapshot_keep_all, const bool swmm_snapshot_keep_previous, const bool swmm_node_rim_from_dem, const std::string& swmm_coupling_nodes)
	{
		rank_=rank;
		size_=size;
		this->output_folder = output_folder;
		this->snapshot_keep_all = swmm_snapshot_keep_all;
		this->snapshot_keep_previous = swmm_snapshot_keep_previous;
		this->node_rim_from_dem = swmm_node_rim_from_dem;
		this->coupling_nodes_key = swmm_coupling_nodes;

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

		infile.close();

		// The coupling declaration, decided on every rank from swmm_coupling_nodes:
		// a list file, ALL_JUNCTIONS, or nothing (the legacy derivation from the
		// [INFLOWS] FLOW lines). The branches are exclusive; resolve_coupling_nodes
		// applies the one chosen here.
		coupling = coupling_declaration_t();
		if (coupling_nodes_key.empty()) {
			coupling.kind = COUPLING_LEGACY;
		}
		else if (coupling_nodes_key == "ALL_JUNCTIONS") {
			coupling.kind = COUPLING_ALL_JUNCTIONS;
		}
		else {
			coupling.kind = COUPLING_LIST;
			coupling.path = coupling_nodes_key;
			std::ifstream list_file(coupling.path.c_str());
			if (!list_file.is_open()) {
				std::cerr << ERROR "Cannot read the coupling-node list " << coupling.path
				          << " (swmm_coupling_nodes)." << std::endl;
				exit(EXIT_FAILURE);
			}
			read_coupling_list(list_file, coupling.names, coupling.lines);
			const std::pair<int, int> rep_pair = find_repeated_name(coupling.names);
			if (rep_pair.first >= 0) {
				std::cerr << ERROR "Coupling node " << coupling.names[rep_pair.second] << " on line "
				          << coupling.lines[rep_pair.second] << " of " << coupling.path
				          << " repeats " << coupling.names[rep_pair.first] << " on line "
				          << coupling.lines[rep_pair.first]
				          << " (SWMM compares node names case-insensitively)." << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		// [COORDINATES], [JUNCTIONS], [INFLOWS] and [POLLUTANTS] are read with SWMM's
		// own section rule and tokenizer in every branch; the coupled list is then the
		// declaration's (see resolve_coupling_nodes). init_swmm checks it against the engine.
		infile.open(inp_filename.c_str());
		inp_coupled_list_t parsed = read_inp_sections(infile);
		infile.close();
		parsed.names = resolve_coupling_nodes(coupling, parsed);

		// Coordinate contract: SWMM's [COORDINATES] are map coordinates, tied to no flow
		// unit, and TRITON places each node in its DEM's grid from them. So they must be
		// in the DEM's coordinate system in every model, SI or US, and are used as read.
		for (std::size_t j = 0; j < parsed.coord_names.size(); ++j) {
			nodeID_all.push_back(parsed.coord_names[j]);
			x_all.push_back(parsed.coord_x[j]);
			y_all.push_back(parsed.coord_y[j]);
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
		// A DECLARED node with no [COORDINATES] entry stops the run on every rank,
		// since it would otherwise be placed at (0, 0); the legacy branch keeps its
		// behaviour.
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			 bool found = false;
			 for (int j = 0; j < num_of_swmm_nodes; j++) {
				  if (swmm_upper(nodeID[i]) == swmm_upper(nodeID_all[j])) {
						x[i] = x_all[j];
						y[i] = y_all[j];
						found = true;
						break;
				  }
			 }
			 if (!found && coupling.kind != COUPLING_LEGACY) {
				  std::cerr << ERROR "Coupling node " << nodeID[i]
				            << " has no [COORDINATES] entry in " << inp_filename << "." << std::endl;
				  exit(EXIT_FAILURE);
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
			exchange_num_nodes = global_num_of_swmm_links;

			// Per-checkpoint full-precision state snapshots live beside the
			// side-file, in the same swmm/ subtree, so any orchestration that
			// preserves one preserves the other.
			//
			// STEM SUPPRESSION. Leaving the stem EMPTY is the cfg-gated mechanism by
			// which a caller FORCES the replay route rather than
			// racing the retention collision for it. The empty stem takes
			// classify_missing_snapshot's FOURTH EXIT, which fires BEFORE any directory
			// scan runs, so the recorded reason is `absent` whatever the snapshot
			// directory holds.
			//
			// ONE ASSIGNMENT SITE, THREE AGREEING READERS, AND THAT IS WHY NOTHING ELSE
			// IS EDITED. The same .empty() predicate already governs
			// write_state_snapshot (which then writes nothing) and
			// try_restore_state_snapshot (which then returns Absent). Guarding those
			// too would put the decision in three places that can disagree; guarding
			// the single assignment keeps it in one that cannot.
			//
			// DEFAULT-INERT, which is what lets this key be added without perturbing any
			// run that predates it. The ground is WHICH ARM an unset flag takes, NOT the
			// absence of an else: argsd's "0" default leaves the flag false, so `!flag`
			// is true and the guard's FIRST arm executes the pre-fix assignment verbatim
			// -- a cfg omitting the key is byte-equivalent to one written before the key
			// existed. There IS an else, four lines below, and it is reached ONLY when
			// suppression is enabled.
			//
			// THE ELSE IS LOAD-BEARING, so do not delete it to simplify this comment.
			// Its rank-0 WARN is the only artifact asserting that the suppression branch
			// EXECUTED. The resume-event record on this arm reads `reason=absent`, and
			// `absent` is ALSO the ladder tail's return, so that field cannot
			// discriminate a configured decline from a genuinely missing snapshot. The
			// per-checkpoint cfg dump copies the key verbatim and so records that
			// suppression was CONFIGURED, which is the weaker claim. S9 in
			// test/test_snapshot_stem_suppression_source.py goes red if this comment
			// and the branch structure below it disagree again.
			//
			// NOT "withholding the snapshot", which is a DIFFERENT mechanism with a
			// DIFFERENT recorded reason. Withholding while leaving the stem intact
			// leaves the scan finding nothing and yields `pre-snapshot-checkpoint`,
			// which is a DIFFERENT value from the `absent` this mechanism must yield;
			// deleting at the resume id while a HIGHER one survives yields
			// `retention-collision`, which records a snapshot the keep-2 prune removed
			// -- a claim about retention rather than about configuration, and so false
			// of a run that declined by cfg. Those two are refused by name, not merely
			// unused.
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

			// SWMM reports a rejected input -- a missing external time-series file, a bad
			// [OPTIONS] keyword, any error its own validation finds after TRITON's
			// read_inp_file has passed -- only through these two return codes. Ignored,
			// SWMM stays in its error state and every later step returns at once without
			// routing, so the coupled run would complete against a sewer that never moves.
			// The open call starts its own line so the stem-suppression probe X5
			// (test_snapshot_stem_suppression_source.py), which inserts text before
			// that line by its leading three tabs, still finds it.
			const int swmm_open_rc =
			swmm_open(inp_filename.c_str(), report_filename.c_str(), binary_filename.c_str());
			stop_on_swmm_error(swmm_open_rc, "swmm_open", "while reading and validating the SWMM input",
			                   report_filename.c_str());
			stop_on_swmm_error(swmm_start(TRUE), "swmm_start", "while starting the SWMM analysis",
			                   report_filename.c_str());

			// Each coupled node's rim, from the engine that routes the model. SWMM raises a
			// node's full depth to the crown of its highest connecting link, so a junction
			// recorded at 0 under a 1.2 m pipe is 1.2 m deep to SWMM; taking the recorded
			// token instead gives TRITON a rim of 0, and the exchange law then pushes water
			// from the node onto a dry street whenever the pipe holds any. Ineligible nodes
			// stop the run here, before any checkpoint is read or written: a coupled node
			// that is not a JUNCTION (storage units and dividers are not coupled in this
			// build; outfalls never are), or a list that is not SWMM's FLOW-inflow set in
			// SWMM's order, which would pair surface cells with the wrong sewer nodes.
			static const char* node_type_name[] = {"JUNCTION", "OUTFALL", "STORAGE", "DIVIDER"};
			std::vector<int> swmm_index(global_num_of_swmm_links, -1);
			std::vector<int> has_flow(global_num_of_swmm_links, 0);
			int refused = 0;
			// A US-unit model returns its lengths in feet; both are put in metres here,
			// on rank 0, before the broadcast, so no other rank needs the unit system and
			// the DEM rim (z_cell - invert) subtracts metres from metres.
			const bool us_units = swmm_flow_units_are_us(static_cast<int>(swmm_getValue(swmm_FLOWUNITS, 0)));
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
				if (coupling.kind == COUPLING_LEGACY) has_flow[i] = routing_nodeHasFlowInflow(idx);
				invert_global[i] = swmm_engine_length_m(swmm_getValue(swmm_NODE_ELEV, idx), us_units);
				double token = 0.0;
				for (std::size_t j = 0; j < junctionID_all.size(); j++) {
					if (swmm_upper(junctionID_all[j]) == swmm_upper(global_nodeID[i])) {
						token = max_depth_all[j];
						break;
					}
				}
				rim_global[i] = coupled_node_rim_m(token, swmm_getValue(swmm_NODE_MAXDEPTH, idx), us_units);
			}
			if (refused) MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);

			// The FLOW-inflow half of the check (each entry has a FLOW inflow; the list is
			// as long as SWMM's FLOW-inflow set) runs only in the legacy branch, where the
			// list IS that set. Under a declaration no FLOW-inflow fact may refuse a
			// coupling node, so every entry is passed as qualifying and the count as the
			// list's own, which leaves only the order test live.
			int swmm_flow_count = global_num_of_swmm_links;
			if (coupling.kind == COUPLING_LEGACY) {
				swmm_flow_count = 0;
				for (int j = 0; j < swmm_getCount(swmm_NODE); j++) swmm_flow_count += routing_nodeHasFlowInflow(j);
			}
			else {
				std::fill(has_flow.begin(), has_flow.end(), 1);
			}
			const coupled_list_check_t chk =
				check_coupled_list_against_swmm(swmm_index, has_flow, swmm_flow_count);
			if (!chk.ok) {
				std::cerr << ERROR << (coupling.kind == COUPLING_LEGACY
				                           ? "TRITON's coupled-node list does not match SWMM's FLOW-inflow nodes: "
				                           : "TRITON's coupling nodes are not in SWMM's node order: ");
				if (chk.entry >= 0)
					std::cerr << "entry " << chk.entry << " (" << global_nodeID[chk.entry] << "): ";
				std::cerr << chk.reason << "." << std::endl;
				MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
			}

			// Activate the coupling: each node's exchange slot, in SWMM node-index order,
			// before any step, replay or snapshot restore reads it.
			const int slot_rc = routing_setCouplingNodes(swmm_index.data(), global_num_of_swmm_links);
			if (slot_rc != 0) {
				std::cerr << ERROR "SWMM could not set the coupling-node slots (error code " << slot_rc
				          << ")." << std::endl;
				MPI_Abort(ENSIFY_COMM_WORLD, EXIT_FAILURE);
			}

			// One line naming the declaration used and the coupled count, so the legacy
			// fallback is visible on every run.
			if (coupling.kind == COUPLING_LIST)
				std::cerr << IN "SWMM coupling nodes: " << global_num_of_swmm_links << ", listed in "
				          << coupling.path << std::endl;
			else if (coupling.kind == COUPLING_ALL_JUNCTIONS)
				std::cerr << IN "SWMM coupling nodes: " << global_num_of_swmm_links
				          << ", all junctions (swmm_coupling_nodes = ALL_JUNCTIONS)" << std::endl;
			else
				std::cerr << IN "SWMM coupling nodes: " << global_num_of_swmm_links
				          << ", derived from the [INFLOWS] FLOW lines (legacy: no swmm_coupling_nodes given)"
				          << std::endl;
			if (coupling.kind != COUPLING_LEGACY) {
				// Disclosure only: a declared node that also carries a FLOW inflow keeps it,
				// summed with the exchange. This reads the inflow and changes no set.
				std::string with_flow;
				for (int i = 0; i < global_num_of_swmm_links; i++)
					if (routing_nodeHasFlowInflow(swmm_index[i])) with_flow += " " + global_nodeID[i];
				if (!with_flow.empty())
					std::cerr << INFO "Coupling nodes that also carry a FLOW inflow, applied with the exchange:"
					          << with_flow << std::endl;
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

	// Stop the whole run when a SWMM API call returns non-zero, printing SWMM's own
	// code and message. Called on rank 0 only (init_swmm's rank-0 block, the
	// per-step solve under triton.h's rank-0 guard, the rank-0 replay walk), so the
	// stop is MPI_Abort AND NOT exit(): a rank-0-only exit() would leave every other
	// rank blocked at the next collective, turning a loud failure into a hang. A
	// zero code returns at once, so a healthy run takes no new branch and prints
	// nothing new.
	//
	// report_path is passed by the two startup calls only. For a per-line input error
	// (ERROR 2xx) swmm_getError returns just the summary "ERROR 200", and SWMM writes
	// the line that names the problem -- "ERROR 209: undefined object NOSUCH at line
	// 119 of [INFLOW] section:" and the rejected line -- only into its report's
	// buffered stream, which an abort discards, leaving a 0-byte report. So on those
	// calls SWMM is closed first, which flushes the report, and the report's error
	// lines are echoed to stderr. Deliberately NOT done after a swmm_step failure:
	// closing mid-run without swmm_end skips the output and routing close, a sequence
	// SWMM does not document, and a step's message already reaches swmm_getError.
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


	// AN ORDERING GUARANTEE IS NOT A DURABILITY GUARANTEE, and the two functions
	// below are where that distinction is enforced rather than assumed.
	//
	// is_open() IS RETAINED, and only as the NOT-INITIALISED sentinel it has
	// always been -- a run with no SWMM surface nodes never opens this stream and
	// must return silently. What is NOT retained is is_open() as the predicate
	// governing the DURABILITY CLAIM: a stream can be open and failed at the same
	// time, so `if (is_open()) flush();` asserts an invariant whose truth it never
	// checks. Both functions now read the OUTCOME through good().
	//
	// A FAILURE ABORTS, and that is a bound disposition rather than a default.
	// The three arms considered were abort, continue-anyway and disable-the-
	// snapshot-route, and they are not three comparable options. ofstream failure
	// state LATCHES, so under continue-anyway every later append fails too and
	// every subsequent checkpoint becomes a resume point whose replay cannot
	// reach it -- one message per coupled timestep for the rest of the run, and a
	// poisoned series behind each one. Under abort, the flush now sits at the TOP
	// of the checkpoint block (triton.h), BEFORE out.write_output, so a failure
	// at checkpoint N terminates the exec BEFORE config_N.cfg is written; the
	// resume picker then returns config_{N-1}.cfg, whose series the invariant
	// covers and whose snapshot is complete. Abort therefore costs one attempt
	// and leaves the member automatically recoverable at N-1.
	//
	// MPI_Abort AND NOT exit(). Both call sites are inside a rank-0 guard, so a
	// rank-0-only exit() would leave every other rank blocked at the next
	// collective -- a loud failure turned into a hang.
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


	std::string swmm_triton::sweep_snapshot_markers()
	{
		return dispose_snapshot_artifacts(snapshot_keep_previous);
	}


	// Splits a snapshot-directory entry name into its checkpoint id and the
	// suffix after ".snapshot", when the name is exactly
	// "{prefix}{digits}.snapshot{suffix}". Returns false for every other name,
	// including a retained "*.prev{k}" (its suffix is not one of the four this
	// mechanism writes) -- which is how a retained file stays invisible to both
	// readers below.
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

		// Collect first, act second: renaming or removing entries while a
		// directory_iterator is open over the same directory leaves it
		// unspecified whether the iterator sees the renamed name again.
		std::vector<std::filesystem::path> published, markers;
		long max_prev = 0;
		{
			std::error_code ec;
			for (const auto& e : std::filesystem::directory_iterator(dir, ec))
			{
				if (ec) break;
				const std::string name = e.path().filename().string();

				// k scans EVERY "*.prev{j}" in the directory, not only this
				// stem's, because the rename target must not exist anywhere a
				// later k could reach it.
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

		// Published names first. A survivor here is the cross-run hazard this
		// function exists to remove, so it stops the clean start.
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


	void swmm_triton::close_exchange_log()
	{
		if (rank_ != 0) return;
		if (exchange_log.is_open()) { exchange_log.flush(); exchange_log.close(); }
	}


	void swmm_triton::replay_exchange_history(value_t up_to_time, int checkpoint_id,
	                                          const char* reason)
	{
		if (rank_ != 0) return;
		const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/true);

		// THE FOUR-FIELD RESUME-EVENT RECORD, replay path.
		//
		// The four fields are the checkpoint id, the resume time, the path
		// taken, and -- on this arm only -- the classified reason. The PATH is
		// carried by WHICH marker fires: these two literals already discriminate
		// snapshot from replay with zero ambiguity, and they are what a log
		// reader greps for, so the record is APPENDED to the
		// existing sentence rather than emitted as a separate line.
		//
		// The leading text through "to t=" and the numeric that follows it are
		// BYTE-UNCHANGED, deliberately: the consumer's parse takes the first
		// whitespace token after the marker, so appending after it cannot move
		// the number, and any consumer matching the whole existing sentence
		// still matches. One row per resume event: on a resume that PROCEEDS exactly
		// one of the two markers fires -- this one on the replay route, and the
		// restore marker in try_restore_state_snapshot on the snapshot route.
		// NEITHER fires on a clean start, on a run with no SWMM surface nodes (the
		// caller gates this whole block on rank 0 holding at least one), or on a
		// resume that aborts, because Unreadable, OutputMismatch and KilledWrite
		// all return before the restore marker is reached. A consumer therefore
		// gets AT MOST one row per exec and must not require exactly one.
		std::cerr << IN "SWMM exchange history replayed to t=" << up_to_time
		          << " s (" << rec_count << " steps); resuming live segment"
		          << " [resume-event checkpoint=" << checkpoint_id
		          << " path=replay reason=" << (reason ? reason : "absent") << "]"
		          << std::endl;
	}


	// Shared by the replay path and the snapshot path. See the declaration for
	// why they must not each carry their own copy of this arithmetic.
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

		// --- walk records 0..up_to_time (SWMM-only, rank 0) -------------------
		// With replay_steps: each record is pushed through swmm_step, rebuilding
		// SWMM's state and its .out/stats for the full window -- exact, but its
		// cost grows with t_k.
		// Without: the records are only counted. The snapshot has already
		// restored the state these steps would have produced, so the walk exists
		// solely to find where the live segment starts, and it does no SWMM work.
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
		//     whichever path positioned it, so a LATER resume can still take the
		//     replay fallback if its snapshot is missing.
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
	// Full-precision state snapshot.
	//
	// One file per checkpoint id. Keyed rather than rolling because a resume from
	// an OLDER checkpoint must not silently load a LATER state: with a rolling
	// path the file would exist, the shape checks would all pass, and the restore
	// would install a state from the future of the requested t_k. Keyed, that
	// resume finds no file for its id and takes the replay, which is correct.
	// ---------------------------------------------------------------------------

	std::string swmm_triton::snapshot_path_for(int checkpoint_id) const
	{
		return snapshot_path_stem + "_cp" + std::to_string(checkpoint_id) + ".snapshot";
	}


	// The two marker paths. ".part" is snapshot_save's own temporary and must
	// agree byte-for-byte with SNAPSHOT_PART_SUFFIX in snapshot.c -- that is the
	// one cross-file literal this pair carries, and it is here because the
	// consultation on the resume path has to find a file the writer named.
	// ".killed" is what the consultation renames a surviving ".part" to, so an
	// operator can still see that a write was killed. The rename is NOT what
	// makes the refusal idempotent -- Arm 2 keying on ".part" OR ".killed" is.
	// Reading the rename as the suppressor is the retired account under which
	// the first resume refused and every resume after it saw only a ".killed",
	// fell through to Absent, and entered the replay with no refusal emitted.
	std::string swmm_triton::snapshot_part_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".part";
	}


	std::string swmm_triton::snapshot_killed_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".killed";
	}


	// ".failed" is the WRITER's marker -- snapshot_save creates it holding one
	// text line when its count pass refuses the model state, and renames its
	// temporary to it when its length check fails after the write -- and must
	// agree byte-for-byte with SNAPSHOT_FAILED_SUFFIX in snapshot.c.
	std::string swmm_triton::snapshot_failed_path_for(int checkpoint_id) const
	{
		return snapshot_path_for(checkpoint_id) + ".failed";
	}


	void swmm_triton::write_state_snapshot(int checkpoint_id)
	{
		if (rank_ != 0) return;
		if (snapshot_path_stem.empty()) return;   // snapshots disabled by cfg, or init_swmm not run

		// WHY THE FLUSH IS HERE. The period count this snapshot is about to record is a
		// claim about the .out, and fwrite is stdio-buffered -- the only flush
		// anywhere in the vendored solver is the implicit one inside fclose at
		// shutdown. Without this the bytes Nperiods counts may not be on disk
		// when the process is SIGKILLed, and the resumed run would then seek
		// past end-of-file and append across a zero-filled hole that parses as
		// valid data. A silent-wrong-numbers failure on exactly the kill path
		// the snapshot exists to serve, and invisible to any check that reads
		// the file's period count rather than its length.
		//
		// Distinct from the close-time flush in output_end, which is there so a
		// TRUNCATION acts on flushed bytes: different site, different dependent,
		// and neither implies the other -- this one fires per checkpoint on the
		// kill path, that one once per exec on the normal path.
		output_flush();

		const std::string path = snapshot_path_for(checkpoint_id);
		if (snapshot_save(path.c_str()) != 0)
		{
			// NOT fatal, and the asymmetry is deliberate. A failed write costs
			// something only at the NEXT resume from this checkpoint, and only if
			// there is one: a stop when the writer kept a ".failed" marker, and
			// otherwise a resume from an earlier published snapshot or a replay.
			// Aborting a live simulation over one checkpoint's snapshot would
			// trade a certain large loss for an uncertain small one.
			//
			// NOTHING IS PUBLISHED AT THIS POINT. snapshot_save writes a
			// ".part" temporary, verifies its value count against the running
			// model, and only then renames it into place -- so a non-zero
			// return means no file exists at `path`. The temporary is removed on
			// every failure arm except the length-check arm, which keeps it as
			// "{path}.failed" so the bytes that failed can be inspected; a
			// leftover ".part" means remove ALSO failed, which snapshot_save
			// reports separately. A refusal in the count pass, before any
			// temporary exists, writes "{path}.failed" holding one text line
			// that names the refusal, so a structural rejection stops the next
			// resume just as a failed write does.
			//
			// THE TWO TEXTS ARE KEYED ON ONE READ of whether that marker exists,
			// because the next resume's behaviour differs on exactly that fact:
			// it stops on a ".failed", and on a missing snapshot with no marker
			// the orchestration resumes from the newest published snapshot below
			// this checkpoint (a launch at this checkpoint itself replays).
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

		// RETENTION. BY DEFAULT keep at most the current and the immediately
		// previous snapshot. The series is per-checkpoint and would otherwise grow
		// without bound over a long run; two is enough for the resume that is about
		// to happen, and the exchange side-file remains the unbounded-history
		// fallback.
		//
		// `swmm_snapshot_keep_all=1` SKIPS the prune entirely, so every
		// per-checkpoint snapshot survives for the life of the run. What that buys
		// is not a correctness fix -- keep-2 is correct and its fallback is correct
		// -- it is ROUTE UNIFORMITY. Under keep-2 whether the snapshot a resume
		// wants still exists depends on how far the run advanced before the
		// interruption fell, which is a wall-clock property: measured at this pin
		// over 30 runs that all resumed at the SAME checkpoint id, the split was
		// 22 snapshot / 8 replay, and four configurations split between their own
		// two repeats on identical inputs -- so the route is not even a function
		// of the configuration. Setting the key removes that variable when one run
		// is compared against another. The cost is bounded and measured: one
		// snapshot is ~13 KiB, so a 144-checkpoint run holds ~2 MiB and 30 such
		// runs ~59 MiB.
		//
		// The comment above is therefore no longer the whole retention story, and
		// that is why it was rewritten rather than left: `two is enough for the
		// resume that is about to happen` is true of a single resume and false
		// whenever a NOMINATED checkpoint must survive on every run regardless of
		// when each run was interrupted.
		if (!snapshot_keep_all && checkpoint_id >= 2)
		{
			std::error_code ec;
			std::filesystem::remove(snapshot_path_for(checkpoint_id - 2), ec);
		}
	}


	// Classifies an ABSENT snapshot into the three reason values the
	// resume-event record carries. All three ground out in the same errno, so
	// the discriminator is the SHAPE OF THE SNAPSHOT SERIES on disk rather than
	// the failure itself:
	//
	//   pre-snapshot-checkpoint  no snapshot exists for ANY checkpoint, so this
	//                            run predates the snapshot mechanism entirely.
	//   retention-collision      a snapshot exists for a HIGHER checkpoint id,
	//                            so the series reached this point and keep-2
	//                            pruned the one this resume wanted. That is the
	//                            measured collision: write_state_snapshot keeps
	//                            2 against the highest index WRITTEN while the
	//                            resume is subsequently launched from a fixed
	//                            LOWER checkpoint id.
	//   absent                   snapshots exist but none above this id -- the
	//                            write for this checkpoint did not land.
	const char* swmm_triton::classify_missing_snapshot(int checkpoint_id) const
	{
		if (rank_ != 0 || snapshot_path_stem.empty()) return "absent";

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


	swmm_triton::SnapshotRestore
	swmm_triton::try_restore_state_snapshot(int checkpoint_id, value_t up_to_time)
	{
		// TWO REFUSALS WITH TWO DIFFERENT STATUSES, and the distinction is what
		// this opening used to get wrong. It read `Defensive only:` and sat above
		// BOTH refusals while its reasoning covered only the first, so a reader
		// inherited the word `defensive` for a branch that is now a PRODUCTION
		// ROUTE.
		//
		// The rank_ refusal below IS defensive only: the sole call site is inside
		// the caller's
		// `rank == 0 && global_num_of_swmm_links > 0` guard, so rank_ is 0 at
		// every reachable invocation. The guard's PLACEMENT is load-bearing and
		// must be preserved -- hoisting this call above it would make a
		// not-the-owning-rank outcome reachable, and neither mapping is safe:
		// Unreadable would abort every multi-rank coupled resume, and Absent
		// would emit one fallback record per non-zero rank per resume, so a reader
		// counting resume-event rows with path=replay would see one per rank where
		// the record carries one per resume.
		//
		// The snapshot_path_stem refusal below is NOT defensive. An empty stem is
		// the INTENDED and cfg-visible route by which a caller forces this
		// function to decline: init_swmm leaves the stem empty when
		// swmm_snapshot_disable is set, so this return is reached on
		// purpose, by configuration, on every resume of such a run. Reaching it
		// is not a sign of a defect and must not be read as one.
		if (rank_ != 0) return SnapshotRestore::Absent;
		if (snapshot_path_stem.empty()) return SnapshotRestore::Absent;

		const std::string path = snapshot_path_for(checkpoint_id);

		// ---------------------------------------------------------------------
		// THE KILLED-WRITE CONSULTATION. THREE ARMS, AND THE ORDER IS THE
		// MECHANISM: the PUBLISHED ARTIFACT IS CHECKED FIRST.
		//
		// Arm 1 exists because a marker's condition can LAPSE. Under the
		// write-to-temp-then-rename publish a snapshot at N exists only after a
		// complete write, so a published snapshot at N PROVES any marker at N is
		// stale -- a later successful write at the same id superseded it. The
		// staleness is drawn by the existence of the published artifact, which is
		// precisely the thing whose ABSENCE made the marker meaningful. No
		// timestamp, no counter, no second artifact.
		//
		// Arm 2 keys on ".part" OR ".killed", and the OR is what makes the
		// refusal IDEMPOTENT. The rename was originally introduced to stop the
		// consultation firing twice, which was correct when the remedy
		// RECOVERED -- there a second firing would have undone the recovery. The
		// remedy became a refusal and the suppression was carried without being
		// re-derived, so a mechanism built to suppress a second RECOVERY was
		// suppressing the second REFUSAL: the first resume refused, and every
		// resume after it saw only a ".killed", fell through to Absent, and was
		// placed into the replay with no refusal emitted at all. Keyed on both,
		// every resume at N refuses with the same message and no attempt is ever
		// silently entered into the replay. The writer's ".failed" is a third
		// alternative in the same OR, so a failed write refuses on every resume
		// at N for the same reason.
		//
		// Arm 3 is the unchanged path.
		{
			const std::string part   = snapshot_part_path_for(checkpoint_id);
			const std::string killed = snapshot_killed_path_for(checkpoint_id);
			const std::string failed = snapshot_failed_path_for(checkpoint_id);
			std::error_code   ec;

			// Each of the four names is read EXACTLY ONCE, into a local. Two
			// reads of one name inside one decision can disagree -- a concurrent
			// writer can publish between them -- and then the arm taken and the
			// message emitted would describe two different directory states.
			const bool havePublished = std::filesystem::exists(path,   ec);
			const bool havePart      = std::filesystem::exists(part,   ec);
			const bool haveKilled    = std::filesystem::exists(killed, ec);
			const bool haveFailed    = std::filesystem::exists(failed, ec);

			if (havePublished)
			{
				// ARM 1 -- the marker has LAPSED. Remove it where it is
				// superseded rather than tolerating it or leaving it to a sweep
				// on a route the operator may never take. A ".failed" is NOT
				// removed here: it is the evidence of a writer failure, and only
				// a clean start disposes of it.
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

				// THE FALLBACK THIS MESSAGE OFFERS is the highest complete
				// snapshot this run can actually FIND, by existence check
				// rather than by assertion.
				// Asserting cp{N-1} complete would be a false claim on two
				// reachable inputs: at N=1 there is no predecessor at all
				// (print_id starts at 0 and is incremented INSIDE the checkpoint
				// block ahead of the write, so the first id ever written is 1 and
				// "..._cp0.snapshot" is a name no run produces), and under the
				// keep-2 retention the predecessor may have been pruned. The
				// branch condition here guarantees no published snapshot exists
				// at N, so this search cannot name the file it is declining to
				// load and cannot pass over a better checkpoint.
				int highest = -1;
				for (int cid = checkpoint_id - 1; cid >= 1; --cid)
				{
					std::error_code sec;
					if (std::filesystem::exists(snapshot_path_for(cid), sec))
					{ highest = cid; break; }
				}

				// WHICH CAUSE DECIDES. When several markers exist for this id the
				// precedence is ".failed" over ".killed" over ".part": ".failed"
				// is the writer's own verified-negative result and so the most
				// specific diagnosis, and a failed write STOPS the run where a
				// cut-off write may resume from an earlier snapshot. The losing
				// markers stay on disk; the precedence only selects the text.
				if (haveFailed)
				{
					// A FAILED OR REFUSED WRITE STOPS, with or without an earlier
					// snapshot. The marker comes from one of two writer arms: the
					// count pass refusing the model state (a structural defect;
					// the marker then holds one text line naming it), or the
					// written bytes failing the writer's own check (a full disk,
					// a short close; the marker then holds those bytes). Either
					// cause would still be present on a resume from an earlier
					// snapshot: a full disk fails the next write too, and a
					// structural defect would carry the run past a live fault. So
					// this text deliberately offers no earlier snapshot as a
					// route.
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
					             "before relaunching." << std::endl;
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

				// THE MACHINE-READABLE LINE, one per resume at this checkpoint,
				// for the orchestration that relaunches the run. Its text is a
				// cross-repository contract (the toolkit parses the bracketed
				// fields), so it is printed with no colour prefix and must not be
				// reworded. Exactly one of three:
				//   a cut-off write with a published snapshot below: step back to it;
				//   a failed write with a published snapshot below: stop, and the
				//     snapshot below is named for information only;
				//   no published snapshot below: only a clean start remains, and
				//     the reason still says which kind of marker was found.
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
			// GRADUATED, and this is where fail-fast becomes the default. An
			// ABSENT snapshot is legitimate and degrades to the replay. Anything
			// else means the file DISAGREES WITH THIS BUILD'S READER on a named
			// quantity, and falling back there would produce correct numbers
			// slowly while HIDING a real defect, which is exactly the failure
			// mode the graduated return above exists to remove.
			//
			// NO CAUSE IS NAMED HERE, and the omission is the repair. This
			// comment and the message below BOTH USED TO read the retired cause
			// "written by a different build or is corrupt"
			// -- kept on ONE LINE on purpose, because a quoted phrase wrapped
			// across comment lines is invisible to a line-anchored grep, which
			// is how a post-condition certifying this phrase absent came to
			// read 0 on a file that still carried it. That cause was FALSE on
			// the failure that first produced this graduated return:
			// 45 per-model logs showed restored=0 and replayed=0,
			// the files were written by the build that read them, and they were
			// not corrupt -- they were SHORT, by exactly 18,296 bytes, because
			// the writer's own guard aborted the traversal. The quantity that
			// disagreed is in `msg`, which every refusal arm in snapshot_load
			// fills with its own number. A writer aborting mid-payload is one of
			// the ways a file comes to disagree, and this run is the proof.
			if (rc == SNAPSHOT_ABSENT)
			{
				std::cerr << INFO "SWMM state snapshot not used (" << msg << ").\n"
				          << "         Falling back to replaying the exchange history."
				          << std::endl;
				return SnapshotRestore::Absent;
			}
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
		// which no snapshot restore passes through. Call the factored-out writer
		// rather than duplicating its loop: it walks TRITON's slot map, which
		// routing_setCouplingNodes builds from the coupling nodes in SWMM node-index
		// order and which DEFINES the order, converting Node[].newDepth from feet
		// to metres x MperFT; a second enumeration would agree with it only until
		// one of them changed.
		//
		// This is also why the FALLBACK path must NOT call it: the replay reaches
		// routing_execute on every step and has already left global_new_depth[]
		// at exactly these values.
		routing_exportInflowNodeDepths(global_new_depth);

		// Position the side-file so the live segment appends after t_k, WITHOUT
		// replaying: the state those steps would have produced is already
		// restored. Same function the replay uses, so both paths agree on where
		// the live segment begins.
		const long rec_count = position_exchange_log(up_to_time, /*replay_steps=*/false);

		// POSITION THE BINARY OUTPUT FILE. RESTORE PATH ONLY, and
		// the branch scope is a TERM of this call rather than an omission.
		//
		// The snapshot has just restored Nperiods, so output.c can seek the
		// .out to where the resumed segment's first period belongs and drop any
		// partial or stale record beyond it. The REPLAY-FALLBACK branch gets no
		// such call and needs none: it restores no Nperiods, so this arithmetic
		// has no operand there, and output_open already leaves the stream at
		// OutputStartPos where output_saveResults writes period p to the
		// clean-run offset sequentially.
		//
		// A failure here CANNOT fall back. The accumulators are already
		// restored, so a replay on top of them would re-apply the whole 0..t_k
		// prefix to state that already contains it. The only safe disposition
		// is to stop.
		{
			char pmsg[512];
			pmsg[0] = '\0';
			if (output_positionForResume(pmsg, (int)sizeof(pmsg)) != 0)
			{
				std::cerr << ERROR "SWMM state snapshot loaded but the binary "
				             "output file cannot carry the resumed segment ("
				          << pmsg << ").\n"
				          << "         Refusing to continue. The snapshot's "
				             "accumulators are already restored, so falling back "
				             "to the exchange replay would re-apply the 0..t_k "
				             "prefix on top of state that already contains it.\n"
				          << "         Re-run from a clean start "
				             "(checkpoint_id=0)." << std::endl;
				return SnapshotRestore::OutputMismatch;
			}
		}

		// The restore marker, carrying the resume-event record. Same
		// "...to t=" shape as the replay marker so one matcher discriminates
		// both and the existing numeric parse works unchanged; the four fields
		// are appended AFTER that parse's token, so the number cannot move.
		// path=snapshot carries no reason -- the contract attaches a reason to
		// the replay arm only.
		std::cerr << IN "SWMM state restored from snapshot to t=" << up_to_time
		          << " s (" << rec_count << " steps skipped); resuming live segment"
		          << " [resume-event checkpoint=" << checkpoint_id
		          << " path=snapshot]"
		          << std::endl;

		// THE STEP-BACK CONFIRMATION. When a later checkpoint still carries a
		// live marker, this resume was launched below it -- the orchestration
		// stepped back over an unusable checkpoint -- and this line puts both
		// facts, which checkpoint was unusable and which one was used, into the
		// log of the launch that actually resumes. The launch that found the
		// unusable checkpoint may have had its log overwritten by the relaunch.
		// Same cross-repository contract as the KilledWrite line: no colour
		// prefix, and not to be reworded.
		const int stepped_from = highest_live_marker_above(checkpoint_id);
		if (stepped_from >= 1)
			std::cerr << "SWMM snapshot step-back taken: resumed from checkpoint "
			          << checkpoint_id << " [resume-event checkpoint=" << checkpoint_id
			          << " path=step-back-taken from=" << stepped_from << "]"
			          << std::endl;
		return SnapshotRestore::Restored;
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
