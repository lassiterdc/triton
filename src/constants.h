/** @file Constants.h
 *  @brief Header containing the Constants class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for Constants class
 *
 *  @author Mario Morales Hernandez
 *  @author Md Bulbul Sharif
 *  @author Tigstu T. Dullo
 *  @author Sudershan Gangrade
 *  @author Alfred Kalyanapu
 *  @author Sheikh Ghafoor
 *  @author Shih-Chieh Kao
 *  @author Katherine J. Evans
 *  @bug No known bugs.
 */



#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace Constants
{
	typedef std::pair<int, int> dims_t;
	typedef std::vector<std::string> string_vector;
	typedef std::string::value_type char_t;
	typedef std::vector<std::pair<int, int>> sources_list_t;
	typedef unsigned long long ull;
}

typedef double value_t;

#define MPI_DATA_TYPE MPI_DOUBLE
#define MAX_VALUE DBL_MAX

#define INPUT_DIR "input"
#define OUTPUT_DIR "output"
#define CFG_DIR "cfg"
#define BIN_DIR "bin"
#define ASCII_DIR "asc"
#define TIME_SERIES_DIR "series"
#define DEFAULT_CFG "case4.cfg"

#define GHOST_CELL_PADDING 1
#define USE_MATRIX 0
#define USE_HALO 1
#define SRC_LOCATION 0
#define OBSERVATION_LOCATION 1

#define DEM_NCOLS_LINE 1
#define DEM_NROWS_LINE 2
#define DEM_XLL_CORNER_LINE 3
#define DEM_YLL_CORNER_LINE 4
#define DEM_CELL_SIZE_LINE 5
#define DEM_NODATA_VALUE_LINE 6
#define DEM_HEADER_SIZE 6

#define BIN_ROW_ID 0
#define BIN_COL_ID 1
#define BIN_DEFAULT_HEADER_SIZE 2

#define H 0
#define QX 1
#define QY 2
#define N 3
#define DEM 4
#define RHSH0 5
#define RHSH1 6
#define RHSQX0 7
#define RHSQX1 8
#define RHSQY0 9
#define RHSQY1 10
#define SQRTH 11
#define HALOH 12
#define HALOQXQY 13
#define DT 14
#define HYGT 15
#define HYGV 16
#define RUNIN 17
#define EXTBCV1 18
#define EXTBCV2 19

#define SRCP 0
#define RUNID 1
#define BCRELATIVEINDEX 2
#define BCTYPE 3
#define BCINDEXSTART 4
#define BCNROWSVARS 5

#define TIMER_NSECS 0
#define TIMER_SECS 1

#define G 9.81
#define SQRTG 3.132091953
#define EPS12 1e-12
#define FT3_TO_M3_FACTOR 0.028316847
#define FT_TO_M_FACTOR 0.3048
#define SEC_TO_HOUR_FACTOR 0.000277778
#define HOUR_TO_SEC_FACTOR 3600.0
#define MM_TO_M_FACTOR 0.001

#define THREAD_BLOCK 256

#define TOTAL_TIME "total_time"
#define SIMULATION_TIME "simulation_time"
#define COMPUTE_TIME "compute_time"
#define MPI_TIME "mpi_time"
#define IO_TIME "io_time"

//colors
#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define GRAY   "\033[90m"      /* Gray */

//mesagges
#define OK GREEN << "[OK] " << RESET
#define WARN YELLOW << "[!!] " << RESET
#define ERROR  RED << "[ERROR] " << RESET
#define IN GRAY << "[..] " << RESET
#define DASH BLUE << "[--] " << RESET

















#endif
