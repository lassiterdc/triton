/** @file test_dt_tiebreak.cpp
 *  @brief Chunk 2b.1 test of diagnostics::min_dt_loc: when the minimum step size is
 *         shared by a ghost cell and its interior twin, the recorded cell is the
 *         interior one.
 *
 *  Why. The external-boundary kernel writes a boundary ghost cell and its interior
 *  twin with identical state, so their DT is bit-equal; the north and west ghost has
 *  the smaller index, and a smaller-index tie-break recorded grow or gcol of -1.
 *
 *  Class: REGRESSION. T1 and T2 are the discriminators: they were measured red on the
 *  index-only tie-break (tree 5896190) and are green on the fix. T3 and T4 pin the
 *  behaviour the fix keeps (smaller index among interior ties; a strict ghost minimum is
 *  reported as the ghost index, erratum E-13 Form A) and are not claimed to discriminate
 *  the old rule. Second correct implementation: a single pass with a custom reducer
 *  ordering (class, index) passes the same cases.
 *  One-time chunk test of 2b.1: retire at the run's final landing.
 */
// The headers below assume the translation unit already pulled in these standard and
// system headers (main.cpp does); they are listed here because the test is its own unit.
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "constants.h"
#include "kokkos_utils.h"
#include "supertimer.h"
#include "mpi_utils.h"
#include "diagnostics.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL %s\n", m); failures++; } } while (0)

static const int ROWS = 6, COLS = 7;  // 4 x 5 interior cells, ghost ring around

static int run(const std::vector<double>& v)
{
	const int n = (int)v.size();
	double* dev = nullptr;
	gpuMalloc((void**)&dev, sizeof(double) * n);
	gpuMemcpyAsync(dev, const_cast<double*>(v.data()), sizeof(double) * n, gpuMemcpyHostToDevice, 0);
	gpuStreamSynchronize(0);
	double val; int loc;
	diagnostics::min_dt_loc(ROWS, COLS, (double const*)dev, val, loc);
	gpuFree(dev);
	return loc;
}

int main(int argc, char** argv)
{
	Kokkos::initialize(argc, argv);
	{
		const std::string which = argc > 1 ? argv[1] : "all";
		auto want = [&](const char* t) { return which == "all" || which == t; };
		std::vector<double> base(ROWS * COLS, 1e10);
		const int ghost_nw = 0 * COLS + 1;            // north ghost row, column 1
		const int twin = 1 * COLS + 1;                // its interior twin, first interior cell
		const int other = 3 * COLS + 3;               // a later interior cell

		if (want("T1_interior_wins_over_ghost_twin")) {
			auto v = base; v[ghost_nw] = 0.5; v[twin] = 0.5; v[other] = 0.9;
			CHECK(run(v) == twin, "T1 the tie went to the ghost cell");
		}
		if (want("T2_west_ghost_twin")) {
			auto v = base; const int g = 2 * COLS + 0, t = 2 * COLS + 1; v[g] = 0.25; v[t] = 0.25;
			CHECK(run(v) == t, "T2 the west ghost tie went to the ghost cell");
		}
		if (want("T3_smaller_index_among_interior")) {
			auto v = base; v[2 * COLS + 4] = 0.7; v[3 * COLS + 2] = 0.7;
			CHECK(run(v) == 2 * COLS + 4, "T3 an interior tie did not take the smaller index");
		}
		if (want("T4_strict_ghost_minimum_reports_the_ghost")) {
			auto v = base; v[ghost_nw] = 0.1; v[twin] = 0.5;
			CHECK(run(v) == ghost_nw, "T4 a strict ghost minimum was not reported as the ghost index");
		}
	}
	Kokkos::finalize();
	if (failures) return 1;
	std::printf("PASS %s\n", argc > 1 ? argv[1] : "all");
	return 0;
}
