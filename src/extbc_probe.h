/** @file extbc_probe.h
 *  @brief Diagnostic probe for the external-boundary-condition / hotstart-resume
 *         reproducibility investigation. Compiled ONLY when TRITON_EXTBC_PROBE is
 *         defined; with the macro undefined this header contributes nothing and the
 *         binary is textually identical to the un-instrumented build.
 *
 *  WHAT IT CAPTURES
 *
 *  Two snapshots per solver timestep, taken inside compute_new_state():
 *    phase 0 -- at the TOP of the step, before flux_x/flux_y run. This is the state the
 *               first post-resume flux evaluation actually reads, INCLUDING the ghost
 *               ring, which the checkpoint rasters do not contain.
 *    phase 1 -- immediately AFTER compute_extbc_values(), so the boundary kernel's own
 *               writes (h pinned to the interpolated value, qx pinned to zero, qy carried
 *               through negated) are separable from what preceded them.
 *
 *  Each snapshot writes the FULL PADDED h/qx/qy arrays -- interior plus the ghost ring --
 *  as raw little-endian float64, together with simtime, dt, and the per-BC-cell
 *  interpolated boundary value `auxvalue` and its evaluation instant `lvar`. Everything
 *  is full-precision double; nothing is cast to float32 anywhere on this path.
 *
 *  WHY THE WINDOW IS TIME-TRIGGERED, NOT STEP-TRIGGERED
 *
 *  The comparison is between an uninterrupted reference run and a run resumed from a
 *  checkpoint at t_k. The resumed run's first step IS t_k; the reference run reaches t_k
 *  only after ~1e5 steps. A "first N steps of the process" trigger would therefore dump
 *  two different windows and the two runs would not be comparable. The probe instead
 *  arms at a simulation TIME (TRITON_EXTBC_PROBE_T0) and then records the next N steps,
 *  so both runs dump the same physical window and records join on ordinal + simtime.
 *
 *  ENVIRONMENT (all optional; the defaults dump the first 10 steps of the run)
 *    TRITON_EXTBC_PROBE_T0    simulation time in seconds at which to arm   (default 0.0)
 *    TRITON_EXTBC_PROBE_STEPS number of solver steps to record             (default 10)
 *    TRITON_EXTBC_PROBE_DIR   destination directory                        (default: the
 *                             run's output_folder)
 *
 *  OUTPUT
 *    <dir>/extbc_probe.bin   binary records, format below
 *    <dir>/extbc_probe.log   one human-readable line per record (scalars only)
 *
 *  BINARY FORMAT (little-endian, as written by the host)
 *    header:  int32 magic = 0x45584250 ("EXBP")
 *             int32 version = 1
 *             int32 rows_padded          (includes the ghost ring)
 *             int32 cols_padded          (includes the ghost ring)
 *             int32 num_extbc_cells
 *             int32 sizeof_value_t       (8 for the default double build)
 *    record:  int32   phase              (0 = top of step, 1 = after compute_extbc_values)
 *             int32   it_count           (diagnostic only -- it differs between the
 *                                         reference and resumed runs at the same simtime)
 *             value_t simtime
 *             value_t dt
 *             value_t h  [rows_padded*cols_padded]
 *             value_t qx [rows_padded*cols_padded]
 *             value_t qy [rows_padded*cols_padded]
 *             value_t auxvalue [num_extbc_cells]
 *             value_t lvar     [num_extbc_cells]
 *
 *  VOLUME
 *    One record is 2*4 + 2*sizeof(value_t) + (3*rows_padded*cols_padded + 2*num_extbc_cells)
 *    * sizeof(value_t) bytes. For the 64x120 case (padded 66x122 = 8052 cells, 61 BC cells)
 *    that is ~194 KB per record. Up to three records per step (phase 2 is emitted only when
 *    open_boundaries is on), so ~5.8 MB for the default 10-step window. Rank 0 writes only; the probe is a no-op on every other rank.
 *
 *  @author added for the hotstart-resume external-BC investigation, 2026-08-05
 */

#ifndef EXTBC_PROBE_H
#define EXTBC_PROBE_H

#ifdef TRITON_EXTBC_PROBE

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "constants.h"

namespace ExtbcProbe
{
  int32_t constexpr PROBE_MAGIC   = 0x45584250;  // "EXBP"
  int32_t constexpr PROBE_VERSION = 1;

  inline double env_double(char const *name, double fallback)
  {
    char const *v = std::getenv(name);
    if (v == NULL || *v == '\0') return fallback;
    return std::atof(v);
  }

  inline int env_int(char const *name, int fallback)
  {
    char const *v = std::getenv(name);
    if (v == NULL || *v == '\0') return fallback;
    return std::atoi(v);
  }

  /** @brief Holds the probe's open files and its arming state. One instance per solver. */
  template<typename T>
  class probe
  {
  public:
    bool   enabled     = false;  /**< rank 0 and the files opened successfully */
    bool   armed       = false;  /**< simtime has reached t0 */
    double t0          = 0.0;    /**< arm at this simulation time */
    int    steps_left  = 0;      /**< records remaining in the window (counted per step) */
    int    npad        = 0;      /**< rows_padded * cols_padded */
    int    nbc         = 0;      /**< number of external-BC cells on this rank */

    std::ofstream bin;
    std::ofstream txt;
    std::vector<T> hbuf, qxbuf, qybuf, auxbuf, lvarbuf;

    /** @brief Open the probe files and size the host staging buffers.
     *
     *  @param out_dir   destination directory; overridden by TRITON_EXTBC_PROBE_DIR
     *  @param rank      MPI rank; the probe is a no-op on any rank other than 0
     *  @param rows_pad  padded row count (interior + ghost ring)
     *  @param cols_pad  padded column count (interior + ghost ring)
     *  @param n_bc      number of external-BC cells owned by this rank
     */
    void open(std::string const &out_dir, int rank, int rows_pad, int cols_pad, int n_bc)
    {
      if (rank != 0) return;

      t0        = env_double("TRITON_EXTBC_PROBE_T0", 0.0);
      steps_left = env_int("TRITON_EXTBC_PROBE_STEPS", 10);
      npad      = rows_pad * cols_pad;
      nbc       = n_bc;

      std::string dir = out_dir;
      char const *dir_override = std::getenv("TRITON_EXTBC_PROBE_DIR");
      if (dir_override != NULL && *dir_override != '\0') dir = std::string(dir_override);
      if (!dir.empty() && dir[dir.size() - 1] != '/') dir += "/";

      std::string bin_path = dir + "extbc_probe.bin";
      std::string txt_path = dir + "extbc_probe.log";

      bin.open(bin_path.c_str(), std::ios::binary | std::ios::trunc);
      txt.open(txt_path.c_str(), std::ios::trunc);
      if (!bin.is_open() || !txt.is_open())
      {
        std::cerr << ERROR "TRITON_EXTBC_PROBE: could not open " << bin_path
                  << " (or its .log sibling) for writing. Probe disabled." << std::endl;
        return;
      }

      int32_t hdr[6];
      hdr[0] = PROBE_MAGIC;
      hdr[1] = PROBE_VERSION;
      hdr[2] = rows_pad;
      hdr[3] = cols_pad;
      hdr[4] = n_bc;
      hdr[5] = static_cast<int32_t>(sizeof(T));
      bin.write(reinterpret_cast<char const *>(hdr), sizeof(hdr));
      bin.flush();

      txt << "# TRITON external-BC / hotstart-resume probe\n";
      txt << "# rows_padded=" << rows_pad << " cols_padded=" << cols_pad
          << " num_extbc_cells=" << n_bc << " sizeof_value_t=" << sizeof(T) << "\n";
      txt << "# arm_at_simtime=" << std::setprecision(17) << t0
          << " steps=" << steps_left << "\n";
      txt << "# phase 0 = top of compute_new_state (pre-flux); phase 2 = after the open-boundary\n";
      txt << "# mirror and before compute_extbc_values; phase 1 = after compute_extbc_values.\n";
      txt << "# Chronological order within a step is 0, 2, 1.\n";
      txt << "record,phase,it_count,simtime,dt,aux0,lvar0\n";
      txt.flush();

      hbuf.resize(npad);
      qxbuf.resize(npad);
      qybuf.resize(npad);
      auxbuf.resize(nbc > 0 ? nbc : 1);
      lvarbuf.resize(nbc > 0 ? nbc : 1);

      enabled = true;

      std::cerr << IN "TRITON_EXTBC_PROBE active: arming at simtime=" << std::setprecision(17)
                << t0 << " s for " << steps_left << " steps -> " << bin_path << std::endl;
    }

    /** @brief True when a record should be taken for the step beginning at @p simtime.
     *
     *  Arms on the first step whose simtime has reached t0, then stays armed until the
     *  step budget is spent. Both snapshots of a step share one budget decrement, which
     *  is applied by the caller via step_consumed().
     */
    bool should_record(T simtime) const
    {
      if (!enabled) return false;
      if (steps_left <= 0) return false;
      if (!armed && static_cast<double>(simtime) < t0) return false;
      return true;
    }

    void arm_if_due(T simtime)
    {
      if (!enabled || armed) return;
      if (static_cast<double>(simtime) >= t0) armed = true;
    }

    void step_consumed()
    {
      if (enabled && armed && steps_left > 0) steps_left--;
      if (enabled && steps_left == 0 && bin.is_open())
      {
        bin.flush();
        txt.flush();
      }
    }

    /** @brief Write one record. Device pointers are staged to the host first, so this
     *         works identically on the SERIAL/OpenMP and CUDA/HIP backends.
     */
    void record(int phase, int it_count, T simtime, T dt,
                T *d_h, T *d_qx, T *d_qy, T *d_aux, T *d_lvar,
                gpuStream_t streams)
    {
      if (!enabled) return;

      std::size_t const nbytes = sizeof(T) * static_cast<std::size_t>(npad);
      gpuMemcpyAsync(hbuf.data(),  d_h,  nbytes, gpuMemcpyDeviceToHost, streams);
      gpuMemcpyAsync(qxbuf.data(), d_qx, nbytes, gpuMemcpyDeviceToHost, streams);
      gpuMemcpyAsync(qybuf.data(), d_qy, nbytes, gpuMemcpyDeviceToHost, streams);
      if (nbc > 0)
      {
        std::size_t const nbc_bytes = sizeof(T) * static_cast<std::size_t>(nbc);
        gpuMemcpyAsync(auxbuf.data(),  d_aux,  nbc_bytes, gpuMemcpyDeviceToHost, streams);
        gpuMemcpyAsync(lvarbuf.data(), d_lvar, nbc_bytes, gpuMemcpyDeviceToHost, streams);
      }
      gpuStreamSynchronize(streams);

      int32_t const ph = static_cast<int32_t>(phase);
      int32_t const it = static_cast<int32_t>(it_count);
      bin.write(reinterpret_cast<char const *>(&ph), sizeof(ph));
      bin.write(reinterpret_cast<char const *>(&it), sizeof(it));
      bin.write(reinterpret_cast<char const *>(&simtime), sizeof(T));
      bin.write(reinterpret_cast<char const *>(&dt), sizeof(T));
      bin.write(reinterpret_cast<char const *>(hbuf.data()),  nbytes);
      bin.write(reinterpret_cast<char const *>(qxbuf.data()), nbytes);
      bin.write(reinterpret_cast<char const *>(qybuf.data()), nbytes);
      if (nbc > 0)
      {
        std::size_t const nbc_bytes = sizeof(T) * static_cast<std::size_t>(nbc);
        bin.write(reinterpret_cast<char const *>(auxbuf.data()),  nbc_bytes);
        bin.write(reinterpret_cast<char const *>(lvarbuf.data()), nbc_bytes);
      }

      txt << record_count << "," << phase << "," << it_count << ","
          << std::setprecision(17) << simtime << "," << dt << ","
          << (nbc > 0 ? auxbuf[0] : T(0)) << "," << (nbc > 0 ? lvarbuf[0] : T(0)) << "\n";
      record_count++;
    }

    void close()
    {
      if (!enabled) return;
      if (bin.is_open()) { bin.flush(); bin.close(); }
      if (txt.is_open()) { txt.flush(); txt.close(); }
      enabled = false;
    }

  private:
    long record_count = 0;
  };
}

#endif  // TRITON_EXTBC_PROBE
#endif  // EXTBC_PROBE_H
