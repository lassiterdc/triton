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

	/** @brief The minimum of the per-cell step size array and the index it sits at: a
	 *  separate reduction beside find_min_dt, which stays unchanged.
	 *
	 *  Tie rule: when several cells share the minimum, an interior cell wins over a ghost
	 *  or halo cell, and among cells of one class the smaller index wins. The external-
	 *  boundary kernel writes a boundary ghost cell and its interior twin with identical
	 *  h, qx and qy, so their step sizes are bit-equal, and on the north and west edges
	 *  the ghost has the smaller index; ordering by index alone would record a row or
	 *  column of -1. Two passes (the value, then the smallest tie key among the cells at
	 *  that value) because a one-pass MinLoc join does not honour a class order across
	 *  threads. A minimum held ONLY by ghost cells is returned as that ghost index.
	 *  @param rows,cols dimensions of the array including the ghost ring */
	template<typename T>
	void min_dt_loc(int rows, int cols, T const* dt_arr, double& value, int& loc)
	{
		const int size = rows * cols;
		double vmin = 0.0;
		Kokkos::parallel_reduce(std::string("diag_min_dt"), size, KOKKOS_LAMBDA(const int id, double& lmin) {
			const double v = (double)dt_arr[id];
			if (v < lmin) lmin = v;
		}, Kokkos::Min<double>(vmin));
		long long key = 0;
		Kokkos::parallel_reduce(std::string("diag_min_dt_tie"), size, KOKKOS_LAMBDA(const int id, long long& lkey) {
			if ((double)dt_arr[id] != vmin) return;
			const int ix = id / cols;
			const int iy = id % cols;
			const bool ghost = ix <= GHOST_CELL_PADDING - 1 || ix >= rows - GHOST_CELL_PADDING ||
			                   iy <= GHOST_CELL_PADDING - 1 || iy >= cols - GHOST_CELL_PADDING;
			const long long k = ghost ? (long long)id + (long long)size : (long long)id;
			if (k < lkey) lkey = k;
		}, Kokkos::Min<long long>(key));
		value = vmin;
		loc = (int)(key >= (long long)size ? key - (long long)size : key);
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
		int rank = 0;
		int nprocs = 1;
		double dx = 0.0;

		diag_log::capture_file dt_limiter_file;

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

		/** @brief True when any surface capture is on. */
		bool any() const
		{
			return false
				|| dt_limiter
				;
		}

		/** @brief Records this process's place in the decomposition and the cell size. */
		void place(int rank_, int nprocs_, double cell_size)
		{
			rank = rank_; nprocs = nprocs_; dx = cell_size;
		}

		/** @brief Switches T6 on or off (cfg diag_dt_limiter). */
		void enable_dt_limiter(bool on) { dt_limiter = on; }

		/** @brief Opens the enabled capture files on rank 0: clean when the run starts at
		 *  checkpoint 0, positioned at the checkpoint time otherwise. Every build opens
		 *  them here (not inside the SWMM resume hook, which exists only with
		 *  TRITON_SWMM and only on a coupled model). */
		void init_files(const std::string& dir, int checkpoint_id, double sim_start_time)
		{
			if (rank != 0) return;
			std::vector<diag_log::capture_file*> files;
			if (dt_limiter)
			{
				dt_limiter_file.declare_binary(dir + DiagWriters::DT_LIMITER_NAME, diag_log::MAGIC_DT_LIMITER, 0,
				                               diag_log::fixed_record_reader(DiagWriters::DT_LIMITER_RECORD_BYTES));
				files.push_back(&dt_limiter_file);
			}
			if (checkpoint_id == 0) diag_log::open_clean_all(files);
			else diag_log::open_resume_all(files, sim_start_time);
		}

		// ---- T5 and T6 (the compute_local_dt row) ----

		/** @brief Local minimum of the per-cell step size, with the limiting cell's global
		 *  interior indices, depth and celerity (the same expression compute_dt_and_sqrt
		 *  evaluates).
		 *  @param row_offset interior rows held by ranks before this one */
		template<typename T>
		void local_row(int rows, int cols, T const* dt_arr, T* h_dev, T* qx_dev, T* qy_dev, int row_offset)
		{
			min_dt_loc(rows, cols, dt_arr, local_val, local_loc);
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
