/** @file diagnostics.h
 *  @brief The surface diagnostics captures: the step-limiting cell (T6), the open-edge
 *         fluxes and the domain mass balance (T2), and the per-rank wet-cell counts (T5).
 *
 *  Problem this file solves. A coupled run that loses or gains water, or whose step
 *  size collapses, cannot be explained from its rasters alone: the rasters show where
 *  the water is, not which face moved it, which cell set the step size, or whether the
 *  domain balance closed. These captures record that, read-only, beside the run, and
 *  every one is off unless its cfg key is set (diag_dt_limiter, diag_domain_balance,
 *  diag_wet_cells).
 *
 *  What the captures may not do. Each one reads solver state and writes only memory it
 *  owns and its own files. None assigns to global_dt, h, qx, qy or any right-hand-side
 *  array, none reorders a kernel, and none adds a branch on a value that feeds state.
 *  Every call into this header sits at one statement position of triton.h, inside a
 *  DIAG_TIME bracket that no other timer category overlaps (the positions are fixed
 *  by the approved shape's rule-6 table and are named at each call site).
 *
 *  How the files stay resumable. The four files are opened through diag_log::capture_file,
 *  which truncates a resumed file to its last complete record at or before the
 *  checkpoint time. Records are therefore stamped with the END of the simulated time
 *  they cover (a per-step record the step's end time, an interval record the
 *  interval's end), and every value carried from one interval into the next (the
 *  interval buffers, the wet-cell sums, the held stored volume) is reset at the
 *  print instant or re-seeded before the step loop at a run start and at a resume.
 *
 *  Sign convention of the balance. Every term of domain_balance.csv and every column of
 *  edge_flux.bin is signed positive INTO the domain, and the residual is
 *  d_storage minus the signed sum of the other terms.
 */

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "constants.h"
#include "diag_log.h"
#include "output.h"
#include "Ensify.h"

namespace diagnostics
{
	// ---------------------------------------------------------------------------------
	// Device reductions. Read-only: each takes const pointers to solver arrays and
	// returns a host double. The cell layout is the solver's: row-major with
	// GHOST_CELL_PADDING ghost rows and columns around the interior, so the interior is
	// rows 1..rows-2 and columns 1..cols-2.
	// ---------------------------------------------------------------------------------

	/** @brief Volume flowing INTO the domain through its west edge this step, m3/s.
	 *
	 *  The interior cell (ix,1) receives its west face fluctuation d2 as rhsh1 = dt*d2/dx
	 *  (flux_x assigns it, so it holds the x contribution until flux_y runs). The face
	 *  flux is F = qx_cell - d2, positive eastward, which is into the domain here. The
	 *  call must therefore sit between the flux_x and flux_y launches. */
	template<typename T>
	double edge_west_into(int rows, int cols, double dx, double dt, T const* qx, T const* rhsh1)
	{
		double total = 0.0;
		const int n = rows - 2 * GHOST_CELL_PADDING;
		Kokkos::parallel_reduce(std::string("diag_edge_west"), n, KOKKOS_LAMBDA(const int i, double& acc) {
			const int id = (i + GHOST_CELL_PADDING) * cols + GHOST_CELL_PADDING;
			acc += dx * ((double)qx[id] - (double)rhsh1[id] * dx / dt);
		}, total);
		return total;
	}

	/** @brief Volume flowing INTO the domain through its east edge this step, m3/s (the
	 *  negative of the eastward face flux F = qx_cell + d1, d1 being rhsh0*dx/dt of the
	 *  last interior column). Same position requirement as edge_west_into. */
	template<typename T>
	double edge_east_into(int rows, int cols, double dx, double dt, T const* qx, T const* rhsh0)
	{
		double total = 0.0;
		const int n = rows - 2 * GHOST_CELL_PADDING;
		Kokkos::parallel_reduce(std::string("diag_edge_east"), n, KOKKOS_LAMBDA(const int i, double& acc) {
			const int id = (i + GHOST_CELL_PADDING) * cols + (cols - 1 - GHOST_CELL_PADDING);
			acc -= dx * ((double)qx[id] + (double)rhsh0[id] * dx / dt);
		}, total);
		return total;
	}

	/** @brief Sum of one array over the interior columns of one row. flux_y ACCUMULATES
	 *  into rhsh0 and rhsh1 on top of flux_x's values, so the north and south edge reads
	 *  are the difference of this sum after flux_y and before it. */
	template<typename T>
	double sum_interior_row(int cols, int row, T const* arr)
	{
		double total = 0.0;
		const int n = cols - 2 * GHOST_CELL_PADDING;
		Kokkos::parallel_reduce(std::string("diag_row_sum"), n, KOKKOS_LAMBDA(const int i, double& acc) {
			acc += (double)arr[row * cols + (i + GHOST_CELL_PADDING)];
		}, total);
		return total;
	}

	/** @brief Volume the negative-depth clamp of update_cells would add, m3: the sum over
	 *  interior cells of max(0, -(h - rhsh0 - rhsh1)) times the cell area, evaluated with
	 *  the same expression update_cells uses. Called before the update_cells launch. */
	template<typename T>
	double clamp_volume(int rows, int cols, double dx, T const* h, T const* rhsh0, T const* rhsh1)
	{
		double total = 0.0;
		const int size = rows * cols;
		const double area = dx * dx;
		Kokkos::parallel_reduce(std::string("diag_clamp"), size, KOKKOS_LAMBDA(const int id, double& acc) {
			const int ix = id / cols;
			const int iy = id % cols;
			if (ix <= GHOST_CELL_PADDING - 1 || ix >= rows - GHOST_CELL_PADDING ||
			    iy <= GHOST_CELL_PADDING - 1 || iy >= cols - GHOST_CELL_PADDING) return;
			const T hij = h[id] - rhsh0[id] - rhsh1[id];
			if (hij < 0.0) acc += (double)(-hij) * area;
		}, total);
		return total;
	}

	/** @brief Volume update_runoff adds this step, m3: intensity (m/s) times dt times the
	 *  cell area, summed over interior cells, indexed exactly as update_runoff indexes. */
	template<typename T>
	double runoff_volume(int rows, int cols, double dx, double dt, int const* runoff_id,
	                     int index_row_runoff, int n_rows_runoff, T const* runoff_intensity)
	{
		double total = 0.0;
		const int size = rows * cols;
		const double area = dx * dx;
		Kokkos::parallel_reduce(std::string("diag_runoff"), size, KOKKOS_LAMBDA(const int id, double& acc) {
			const int ix = id / cols;
			const int iy = id % cols;
			if (ix <= GHOST_CELL_PADDING - 1 || ix >= rows - GHOST_CELL_PADDING ||
			    iy <= GHOST_CELL_PADDING - 1 || iy >= cols - GHOST_CELL_PADDING) return;
			const int id_r = runoff_id[id];
			acc += (double)runoff_intensity[id_r * n_rows_runoff + index_row_runoff] * dt * area;
		}, total);
		return total;
	}

	/** @brief Volume the inflow hydrographs add this step, m3: the flow of each source
	 *  cell interpolated as compute_flow interpolates it, times dt. */
	template<typename T>
	double inflow_volume(int num_src, T const* hyd_time, T const* hyd_val, double dt, double simtime,
	                     int idx_low, int idx_high)
	{
		double total = 0.0;
		Kokkos::parallel_reduce(std::string("diag_inflow"), num_src, KOKKOS_LAMBDA(const int id, double& acc) {
			const double lo = (double)hyd_val[idx_low * num_src + id];
			const double hi = (double)hyd_val[idx_high * num_src + id];
			double flow;
			if (idx_low == idx_high)
			{
				flow = lo;
			}
			else
			{
				const double t_lo = (double)hyd_time[idx_low];
				const double t_hi = (double)hyd_time[idx_high];
				flow = lo + ((hi - lo) * (simtime - t_lo)) / (t_hi - t_lo);
			}
			acc += flow * dt;
		}, total);
		return total;
	}

	/** @brief Sum of h over the cells of the external-boundary list, each distinct cell
	 *  once (mask is 1 on a cell's first occurrence in the list, 0 on a repeat). */
	template<typename T>
	double extbc_depth_sum(int n, int const* relative_index, int const* mask, T const* h)
	{
		double total = 0.0;
		Kokkos::parallel_reduce(std::string("diag_extbc_h"), n, KOKKOS_LAMBDA(const int id, double& acc) {
			if (mask[id]) acc += (double)h[relative_index[id]];
		}, total);
		return total;
	}

	/** @brief Stored volume, m3: the sum of h times the cell area over interior cells. */
	template<typename T>
	double stored_volume(int rows, int cols, double dx, T const* h)
	{
		double total = 0.0;
		const int size = rows * cols;
		const double area = dx * dx;
		Kokkos::parallel_reduce(std::string("diag_stored"), size, KOKKOS_LAMBDA(const int id, double& acc) {
			const int ix = id / cols;
			const int iy = id % cols;
			if (ix <= GHOST_CELL_PADDING - 1 || ix >= rows - GHOST_CELL_PADDING ||
			    iy <= GHOST_CELL_PADDING - 1 || iy >= cols - GHOST_CELL_PADDING) return;
			acc += (double)h[id] * area;
		}, total);
		return total;
	}

	/** @brief Number of interior cells with h > hextra, the wet-cell test of
	 *  compute_dt_and_sqrt. */
	template<typename T>
	long wet_count(int rows, int cols, T const* h, double hextra)
	{
		long total = 0;
		const int size = rows * cols;
		Kokkos::parallel_reduce(std::string("diag_wet"), size, KOKKOS_LAMBDA(const int id, long& acc) {
			const int ix = id / cols;
			const int iy = id % cols;
			if (ix <= GHOST_CELL_PADDING - 1 || ix >= rows - GHOST_CELL_PADDING ||
			    iy <= GHOST_CELL_PADDING - 1 || iy >= cols - GHOST_CELL_PADDING) return;
			if ((double)h[id] > hextra) acc += 1;
		}, total);
		return total;
	}

	/** @brief The minimum of the per-cell step size array and the index it sits at: a
	 *  separate reduction beside find_min_dt, which stays unchanged. */
	template<typename T>
	void min_dt_loc(int size, T const* dt_arr, double& value, int& loc)
	{
		using minloc_t = Kokkos::MinLoc<double, int>;
		using vl_t = typename minloc_t::value_type;
		vl_t result;
		Kokkos::parallel_reduce(std::string("diag_min_dt_loc"), size, KOKKOS_LAMBDA(const int id, vl_t& upd) {
			const double v = (double)dt_arr[id];
			if (v < upd.val || (v == upd.val && id < upd.loc))
			{
				upd.val = v;
				upd.loc = id;
			}
		}, minloc_t(result));
		value = result.val;
		loc = result.loc;
	}

	/** @brief Spins the calling thread for `us` microseconds (the test-only switches'
	 *  delay: a sleep would not occupy the core a real capture occupies). */
	static inline void spin_us(double us)
	{
		const auto t0 = std::chrono::steady_clock::now();
		while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {}
	}

	// ---------------------------------------------------------------------------------
	// The capture state.
	// ---------------------------------------------------------------------------------

	/** @brief State of the three surface captures for one process: the switches, the
	 *  capture files (open on rank 0 only), the interval buffers and the held stored
	 *  volume. Everything the buffers carry across a print instant is reset there. */
	struct surface_t
	{
		bool dt_limiter = false;       ///< cfg diag_dt_limiter (T6)
		bool domain_balance = false;   ///< cfg diag_domain_balance (T2)
		bool wet_cells = false;        ///< cfg diag_wet_cells (T5)
		int rank = 0;
		int nprocs = 1;
		double dx = 0.0;

		diag_log::capture_file edge_flux_file, domain_balance_file;
		diag_log::capture_file dt_limiter_file;
		diag_log::capture_file wet_cells_file;

		// T2 interval state: per-step fluxes (m3/s, into the domain: west, east, south, north),
		// the step's end time and dt, and the interval sums of the balance's other terms.
		std::vector<double> step_t_end, step_dt, step_flux;
		double runoff_m3 = 0.0, inflow_m3 = 0.0, extbc_m3 = 0.0, clamp_m3 = 0.0, exchange_m3 = 0.0;
		double we_west = 0.0, we_east = 0.0;        ///< the W/E read of the step in progress
		double ns_north_before = 0.0, ns_south_before = 0.0; ///< rhsh sums before flux_y
		double extbc_before = 0.0;                   ///< external-boundary depth sum before its kernel
		double held_storage = 0.0;                   ///< S at the last print instant or the seed (rank 0)

		// External-boundary list de-duplication (a cell listed twice counts once).
		std::vector<int> bc_cache;
		int* bc_mask_dev = nullptr;
		int bc_mask_n = 0;

		// T5 interval state.
		double wet_sum = 0.0;
		long wet_steps = 0;

		// T6 step state: the local minimum, its cell's details, and the collective's outcome.
		double local_val = 0.0;
		int local_loc = -1;
		double local_h = 0.0, local_celerity = 0.0;
		int local_grow = -1, local_gcol = -1;
		double raw_dt = 0.0;           ///< the global minimum before the solver's own fallbacks and clip
		int winner = 0;
		std::vector<double> gathered;  ///< rank 0: nprocs x 5 payload {val, h, celerity, grow, gcol}
		long mismatch_logged = 0;

		surface_t() = default;
		surface_t(const surface_t&) = delete;
		surface_t& operator=(const surface_t&) = delete;
		~surface_t() = default;

		/** @brief Frees the device mask. Called at the end of simulate(), while Kokkos is
		 *  still initialised: a destructor of the triton object may run after it is not. */
		void release()
		{
			if (bc_mask_dev) { gpuFree(bc_mask_dev); bc_mask_dev = nullptr; }
		}

		/** @brief True when any surface capture is on. */
		bool any() const
		{
			return false
				|| dt_limiter
				|| domain_balance
				|| wet_cells
				;
		}

		/** @brief Records this process's place in the decomposition and the cell size. */
		void place(int rank_, int nprocs_, double cell_size)
		{
			rank = rank_; nprocs = nprocs_; dx = cell_size;
		}

		/** @brief Switches T6 on or off (cfg diag_dt_limiter). */
		void enable_dt_limiter(bool on) { dt_limiter = on; }

		/** @brief Switches T2 on or off (cfg diag_domain_balance). */
		void enable_domain_balance(bool on) { domain_balance = on; }

		/** @brief Switches T5 on or off (cfg diag_wet_cells). */
		void enable_wet_cells(bool on) { wet_cells = on; }

		/** @brief Opens the enabled capture files on rank 0: clean when the run starts at
		 *  checkpoint 0, positioned at the checkpoint time otherwise. Every build opens
		 *  them here (not inside the SWMM resume hook, which exists only with
		 *  TRITON_SWMM and only on a coupled model). */
		void init_files(const std::string& dir, int checkpoint_id, double sim_start_time)
		{
			if (rank != 0) return;
			std::vector<diag_log::capture_file*> files;
			if (domain_balance)
			{
				edge_flux_file.declare_binary(dir + DiagWriters::EDGE_FLUX_NAME, diag_log::MAGIC_EDGE_FLUX, 4,
				                              diag_log::fixed_record_reader(DiagWriters::EDGE_FLUX_RECORD_BYTES));
				domain_balance_file.declare_text(dir + DiagWriters::DOMAIN_BALANCE_NAME, DiagWriters::DOMAIN_BALANCE_HEADER, 0);
				files.push_back(&edge_flux_file);
				files.push_back(&domain_balance_file);
			}
			if (dt_limiter)
			{
				dt_limiter_file.declare_binary(dir + DiagWriters::DT_LIMITER_NAME, diag_log::MAGIC_DT_LIMITER, 0,
				                               diag_log::fixed_record_reader(DiagWriters::DT_LIMITER_RECORD_BYTES));
				files.push_back(&dt_limiter_file);
			}
			if (wet_cells)
			{
				wet_cells_file.declare_text(dir + DiagWriters::WET_CELLS_NAME, DiagWriters::WET_CELLS_HEADER, 1);
				files.push_back(&wet_cells_file);
			}
			if (checkpoint_id == 0) diag_log::open_clean_all(files);
			else diag_log::open_resume_all(files, sim_start_time);
		}

		// ---- T2 ----

		/** @brief Stored-volume seed S(t_start): the full-grid sum of h times the area,
		 *  reduced to rank 0 and held there. Entered by every rank once per process
		 *  start, after a hotstart restores h. */
		template<typename T>
		void seed_storage(int rows, int cols, T const* h)
		{
			if (!domain_balance) return;
			const double local = stored_volume(rows, cols, dx, h);
			double total = 0.0;
			MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, ENSIFY_COMM_WORLD);
			held_storage = total;
		}

		/** @brief The W/E edge read (between the flux_x and flux_y launches) and the sums
		 *  the N/S read differences against. */
		template<typename T>
		void read_west_east(int rows, int cols, double dt, T const* qx, T const* rhsh0, T const* rhsh1)
		{
			we_west = edge_west_into(rows, cols, dx, dt, qx, rhsh1);
			we_east = edge_east_into(rows, cols, dx, dt, qx, rhsh0);
			ns_north_before = (rank == 0) ? sum_interior_row(cols, GHOST_CELL_PADDING, rhsh0) : 0.0;
			ns_south_before = (rank == nprocs - 1) ? sum_interior_row(cols, rows - 1 - GHOST_CELL_PADDING, rhsh1) : 0.0;
		}

		/** @brief The N/S edge read and the clamp reduction (between the flux_y and
		 *  update_cells launches). Closes the step's edge record. North lies on rank 0 and
		 *  south on the last rank; a rank without that edge contributes zero. */
		template<typename T>
		void read_north_south_and_clamp(int rows, int cols, double dt, double t_step_end, T const* h,
		                                T const* qy, T const* rhsh0, T const* rhsh1)
		{
			double north = 0.0, south = 0.0;
			if (rank == 0)
			{
				const double q = sum_interior_row(cols, GHOST_CELL_PADDING, qy);
				const double after = sum_interior_row(cols, GHOST_CELL_PADDING, rhsh0);
				north = -dx * (q + (after - ns_north_before) * dx / dt);
			}
			if (rank == nprocs - 1)
			{
				const double q = sum_interior_row(cols, rows - 1 - GHOST_CELL_PADDING, qy);
				const double after = sum_interior_row(cols, rows - 1 - GHOST_CELL_PADDING, rhsh1);
				south = dx * (q - (after - ns_south_before) * dx / dt);
			}
			step_t_end.push_back(t_step_end);
			step_dt.push_back(dt);
			step_flux.push_back(we_west);
			step_flux.push_back(we_east);
			step_flux.push_back(south);
			step_flux.push_back(north);
			clamp_m3 += clamp_volume(rows, cols, dx, h, rhsh0, rhsh1);
		}

		/** @brief Rebuilds the first-occurrence mask when the external-boundary list
		 *  changed (a dynamic repartition rebuilds it). */
		void ensure_extbc_mask(const int* host_list, int n)
		{
			if (n == bc_mask_n && n == (int)bc_cache.size() && std::equal(bc_cache.begin(), bc_cache.end(), host_list))
				return;
			bc_cache.assign(host_list, host_list + n);
			std::vector<int> mask(n > 0 ? n : 1, 0);
			std::unordered_set<int> seen;
			for (int i = 0; i < n; i++) mask[i] = seen.insert(bc_cache[i]).second ? 1 : 0;
			if (bc_mask_dev) { gpuFree(bc_mask_dev); bc_mask_dev = nullptr; }
			gpuMalloc((void**)&bc_mask_dev, sizeof(int) * mask.size());
			gpuMemcpyAsync(bc_mask_dev, mask.data(), sizeof(int) * mask.size(), gpuMemcpyHostToDevice, 0);
			gpuStreamSynchronize(0);
			bc_mask_n = n;
		}

		/** @brief Depth sum over the list's distinct cells, read directly before the
		 *  external-boundary kernel. */
		template<typename T>
		void extbc_before_read(int n, int const* rel_idx_dev, T const* h)
		{
			extbc_before = extbc_depth_sum(n, rel_idx_dev, bc_mask_dev, h);
		}

		/** @brief Read directly after the kernel: the imposed volume of the step is the
		 *  change of that sum times the cell area. */
		template<typename T>
		void extbc_after_read(int n, int const* rel_idx_dev, T const* h)
		{
			extbc_m3 += (extbc_depth_sum(n, rel_idx_dev, bc_mask_dev, h) - extbc_before) * dx * dx;
		}

		/** @brief Net exchange with SWMM this step, m3, signed into the domain. exchange_q
		 *  is in ft3/s and positive INTO SWMM (the exchange kernel stores minus the flow
		 *  into the cell, converted from m3/s). */
		void add_exchange(const value_t* exchange_q, int n_local, double dt)
		{
			double q = 0.0;
			for (int i = 0; i < n_local; i++) q += (double)exchange_q[i];
			exchange_m3 += -q * dt * FT3_TO_M3_FACTOR;
		}

		// ---- T5 and T6 (the compute_local_dt row) ----

		/** @brief T5 sample: the interior wet-cell count of the state this step starts from,
		 *  added to the interval's running sum. */
		template<typename T>
		void sample_wet(int rows, int cols, T const* h_dev, double hextra)
		{
			wet_sum += (double)wet_count(rows, cols, h_dev, hextra);
			wet_steps += 1;
		}

		/** @brief Local minimum of the per-cell step size, with the limiting cell's global
		 *  interior indices, depth and celerity (the same expression compute_dt_and_sqrt
		 *  evaluates).
		 *  @param row_offset interior rows held by ranks before this one */
		template<typename T>
		void local_row(int rows, int cols, T const* dt_arr, T* h_dev, T* qx_dev, T* qy_dev, int row_offset)
		{
			min_dt_loc(rows * cols, dt_arr, local_val, local_loc);
			local_h = 0.0; local_celerity = 0.0; local_grow = -1; local_gcol = -1;
			if (local_val < MAX_VALUE - 1.0 && local_loc >= 0)
			{
				T h = 0, qx = 0, qy = 0;
				gpuMemcpyAsync(&h, h_dev + local_loc, sizeof(T), gpuMemcpyDeviceToHost, 0);
				gpuMemcpyAsync(&qx, qx_dev + local_loc, sizeof(T), gpuMemcpyDeviceToHost, 0);
				gpuMemcpyAsync(&qy, qy_dev + local_loc, sizeof(T), gpuMemcpyDeviceToHost, 0);
				gpuStreamSynchronize(0);
				local_h = (double)h;
				const double hd = (double)h;
				const double maxu = std::fmax(std::fabs((double)qx), std::fabs((double)qy)) / hd;
				local_celerity = maxu + SQRTG * std::sqrt(hd);
				local_grow = local_loc / cols - GHOST_CELL_PADDING + row_offset;
				local_gcol = local_loc % cols - GHOST_CELL_PADDING;
			}
			if (nprocs == 1)
			{
				raw_dt = local_val;
				winner = 0;
				gathered.assign(5, 0.0);
				store_payload(0);
			}
		}

		/** @brief The T6 collective: MPI_Allreduce with MPI_MINLOC on (dt, rank), then a
		 *  gather of each rank's limiting-cell details to rank 0 (the winner's slot is the
		 *  record). Entered by every rank under the cfg key, which is identical on all
		 *  ranks. Issued after the solver's MPI_MIN allreduce has synchronised the ranks,
		 *  so it carries no load-imbalance wait (the TEST-ONLY T-E row moves it before). */
		void minloc_collective()
		{
			if (!dt_limiter || nprocs == 1) return;
			struct { double val; int rank; } in, out;
			in.val = local_val;
			in.rank = rank;
			MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, ENSIFY_COMM_WORLD);
			raw_dt = out.val;
			winner = out.rank;
			double payload[5] = { local_val, local_h, local_celerity, (double)local_grow, (double)local_gcol };
			if (rank == 0) gathered.assign((std::size_t)nprocs * 5, 0.0);
			MPI_Gather(payload, 5, MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr, 5, MPI_DOUBLE, 0, ENSIFY_COMM_WORLD);
		}

		/** @brief Writes the step's dt_limiter.bin record (rank 0). Called once
		 *  compute_global_dt has returned, so global_dt is final.
		 *  @param t_step_end simtime + global_dt of the step about to run
		 *  @param global_dt the step size
		 *  @param init_dt the solver's fallback step size */
		void write_t6_record(double t_step_end, double global_dt, double init_dt)
		{
			if (!dt_limiter || rank != 0) return;
			const double pre_clip = (raw_dt >= MAX_VALUE - 1.0) ? init_dt : raw_dt;
			int reason = 0;
			if (global_dt < pre_clip) reason = 1;               // clipped to the print instant
			else if (raw_dt >= MAX_VALUE - 1.0) reason = 2;     // init_dt fallback
			if (reason == 0 && std::fabs(raw_dt - global_dt) > 0.0 && mismatch_logged < 5)
			{
				mismatch_logged++;
				std::cerr << "[WARNING] diagnostics capture: the step-limiting cell search found dt " << raw_dt
				          << " but the step used " << global_dt << " at t=" << t_step_end << std::endl;
			}
			const double* p = &gathered[(std::size_t)winner * 5];
			DiagWriters::write_dt_limiter_record(dt_limiter_file, t_step_end, global_dt, winner, (int32_t)p[3], (int32_t)p[4],
			                        reason, p[1], p[2]);
		}

		// ---- the print instant and the end of the run ----

		/** @brief True when an interval has unwritten rows (steps since the last print
		 *  instant). Identical on every rank. */
		bool pending() const { return false
			|| !step_t_end.empty()
			|| wet_steps > 0
			; }

		/** @brief The print-instant row: reduces the interval's terms and edge values to
		 *  rank 0, writes the per-step edge records, the balance line and the wet-cell
		 *  rows, then resets every interval accumulator. Collective: every rank calls it
		 *  because the print test reads the global simtime.
		 *  @param t_end interval end (the print instant, or the run's end time at the close) */
		template<typename T>
		void print_instant(int rows, int cols, T const* h_dev, double hextra, int print_id, double t_end)
		{
			if (domain_balance && !step_t_end.empty())
			{
				const double s_local = stored_volume(rows, cols, dx, h_dev);
				const double terms_local[6] = { runoff_m3, inflow_m3, extbc_m3, clamp_m3, exchange_m3, s_local };
				double terms[6] = { 0, 0, 0, 0, 0, 0 };
				MPI_Reduce(terms_local, terms, 6, MPI_DOUBLE, MPI_SUM, 0, ENSIFY_COMM_WORLD);
				std::vector<double> flux(step_flux.size(), 0.0);
				MPI_Reduce(step_flux.data(), flux.data(), (int)step_flux.size(), MPI_DOUBLE, MPI_SUM, 0, ENSIFY_COMM_WORLD);
				if (rank == 0)
				{
					double edge_m3[4] = { 0, 0, 0, 0 };
					for (std::size_t i = 0; i < step_t_end.size(); i++)
					{
						DiagWriters::write_edge_flux_record(edge_flux_file, step_t_end[i], &flux[4 * i]);
						for (int k = 0; k < 4; k++) edge_m3[k] += flux[4 * i + k] * step_dt[i];
					}
					const double s_end = terms[5];
					const double d_storage = s_end - held_storage;
					const double residual = d_storage - (terms[0] + terms[1] + terms[2] + terms[3] + terms[4]
					                                     + edge_m3[0] + edge_m3[1] + edge_m3[2] + edge_m3[3]);
					DiagWriters::write_domain_balance_row(domain_balance_file, t_end, d_storage, terms[0], terms[1], terms[2],
					                         terms[3], terms[4], edge_m3, residual);
					held_storage = s_end;
				}
			}
			if (wet_cells && wet_steps >= 0)
			{
				const long at_print = wet_count(rows, cols, h_dev, hextra);
				const double local[2] = { wet_steps > 0 ? wet_sum / (double)wet_steps : 0.0, (double)at_print };
				std::vector<double> all;
				if (rank == 0) all.assign((std::size_t)nprocs * 2, 0.0);
				MPI_Gather(local, 2, MPI_DOUBLE, rank == 0 ? all.data() : nullptr, 2, MPI_DOUBLE, 0, ENSIFY_COMM_WORLD);
				if (rank == 0)
				{
					for (int r = 0; r < nprocs; r++)
						DiagWriters::write_wet_cells_row(wet_cells_file, print_id, t_end, r, all[2 * r], (long)all[2 * r + 1]);
				}
			}
			reset_interval();
		}

		/** @brief Clears every value that must not cross a print instant. */
		void reset_interval()
		{
			step_t_end.clear(); step_dt.clear(); step_flux.clear();
			runoff_m3 = inflow_m3 = extbc_m3 = clamp_m3 = exchange_m3 = 0.0;
			wet_sum = 0.0; wet_steps = 0;
		}

	private:
		/** @brief Copies this process's limiting-cell details into the gathered table. */
		void store_payload(int slot)
		{
			double* p = &gathered[(std::size_t)slot * 5];
			p[0] = local_val; p[1] = local_h; p[2] = local_celerity; p[3] = (double)local_grow; p[4] = (double)local_gcol;
		}
	};
}

#endif
