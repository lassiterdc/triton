/** @file output.h
 *  @brief Header containing the Output class
 *
 *  This contains the subroutines and eventually any
 *  macros, constants, etc. needed for Output class
 *  
 *  @author Mario Morales Hernandez
 *  @author Sudershan Gangrade
 *  @author Shih-Chieh Kao
 *  @author Michael Kelleher
 *  @author Matthew R. Norman
 *  @author Youngsung Kim
 *  @author Juan Manuel Perez Garcia de Carellan
 *  @author Ganesh Ghimire
 *  @bug No known bugs.
 */


#ifndef OUTPUT_H
#define OUTPUT_H

#include "constants.h"
#include "matrix.h"

#ifdef TRITON_GDAL
#include <gdal.h>  
#include <gdal_priv.h>  
#include <ogr_spatialref.h>  
#include <ogr_geometry.h>  
#include <cpl_conv.h> // For geotiff output
#endif

#include "Ensify.h"

// Additional includes for run log functions
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Output
{
	template<class T>
	class output	/**< Ths class handles all data outputs in file. */
	{
	public:
		
/** @brief Constructor.
*
*/	
		output<T>() {};
		
		
/** @brief Destructor. Releases any allocated memory.
*
*/		
		~output<T>();
		
		
/** @brief It initializes anything related to outputs in file.
*
*  @param rows Rows in subdomain
*  @param cols Columns in subdomain
*  @param xll X coordinate of the origin
*  @param yll Y coordinate of the origin
*  @param cellsize Cell size
*  @param rank Current sub domain id
*  @param size Number of sub domains
*  @param project_dir Main project directory
*  @param outfile_pattern Output file name pattern
*  @param time_series_flag Flag to output time series or not
*  @param cfg_content Contents of input cfg file
*  @param output_option Determines how to write data
*/		
		void init(int rows, int cols, T xll, T yll, T cellsize, int rank, int size, std::string project_dir, std::string output_folder, std::string outfile_pattern, int time_series_flag, std::string cfg_content, std::string output_option);
		
		
/** @brief It initializes time series outputs in a file.
*
*  @param num_of_obs_points number of local observation points
*  @param num_of_obs_points_global number of global observation points
*  @param relative_obs_index Relative array of indexes of local observation cells for the global array
*  @param observation_cells Local cell index
*  @param observation_cells_global Global cell index
*  @param print_option Which data to write
*  @param checkpoint_id Current checkpoint id
*/			
		void init_time_series(int num_of_obs_points, int num_of_obs_points_global, std::vector<int> relative_obs_index, Constants::sources_list_t observation_cells, Constants::sources_list_t observation_cells_global, std::string print_option, int checkpoint_id);
		
		
/** @brief It calculates which data to output in file. Also prints checkpoint id.
*
*  @param h_arr Water depth data
*  @param qx_arr Discharge in x direction data
*  @param qy_arr Discharge in y direction data
*  @param output_format Format of output files
*  @param projection Projection system
*  @param print_option Which data to write
*  @param print_id Current checkpoint id
*  @param it_count Number of iterations so far
*  @param max_value_h Max value of water depth data
*  @param max_value_print_option Which max value data to write
*/		
		void write_output(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, std::string output_format, std::string projection, std::string print_option, int print_id, int it_count, T simtime, T average_dt, Matrix::matrix<T>& max_value_h, std::string max_value_print_option);
		
		
/** @brief It outputs a specific data array's full domain in a single ascii file. 
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*/		
		void write_output_ascii_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id);
		
		
/** @brief It outputs a specific data array's sub domain in a ascii file. All subdomain outputs seperately in different file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*/			
		void write_output_ascii_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id);
		
		
/** @brief It outputs a specific data array's full domain in a single binary file. 
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*/		
		void write_output_binary_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id);
		
		
/** @brief It outputs a specific data array's sub domain in a binary file. All subdomain outputs seperately in different file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param projection Spatial reference system projection string  
*/			
		void write_output_binary_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id);

		void write_output_ghost_ring(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, int print_id);
		void read_output_ghost_ring(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, int print_id);


#ifdef TRITON_GDAL
/** @brief It outputs a specific data array's full domain in a single GeoTIFF file. 
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param projection Spatial reference system projection string  
*/		
		void write_output_geotiff_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, std::string projection);
		
		
/** @brief It outputs a specific data array's sub domain in a GeoTIFF file. All subdomain outputs seperately in different file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*/			
		void write_output_geotiff_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, std::string projection);

/** @brief Writes a Virtual Raster Table (VRT) file for combining multiple raster outputs
*
*  @param what_mat Data type identifier
*  @param print_id Current checkpoint id
*  @param all_rows Vector containing row information for each subdomain
*  @param raster_cols Number of columns in the raster
*  @param xll Lower left X coordinate
*  @param yll Lower left Y coordinate
*  @param cellsize Cell size/resolution
*  @param projection Spatial reference system projection string
*  @param file_dir Output directory path
*/
		void write_output_vrt(const std::string &what_mat, int print_id,
				const std::vector<int> &all_rows, int raster_cols,
				double xll, double yll, double cellsize,
				const std::string &projection,
				const std::string &file_dir);

#endif
		
		
/** @brief It calculates output file name.
*
*  @param what Data type
*  @param subdir Output format directory
*  @param print_id Current checkpoint id
*  @param extension File extension
*  @return File name
*/	
		std::string get_mat_path(std::string what, std::string root_dir, std::string subdir, int print_id, std::string extension);
		

/** @brief It outputs time series data in a file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param simtime Current time of simulation
*/		
		void output_time_series(T *value_obs, std::string what_mat, T simtime);

/** @brief It outputs time series data in a file.
*
*  @param h_arr_obs Water depth observation data
*  @param qx_arr_obs Discharge in x direction observation data
*  @param qy_arr_obs Discharge in y direction observation data
*  @param simtime Current time of simulation
*  @param print_option Which data to write
*/	

	
	void write_observation_data( T *h_arr_obs, T *qx_arr_obs, T *qy_arr_obs, T simtime, std::string print_option);

/** @brief It calculates content of updated configuration and outputs it in a file.
*
*  @param simtime Current time of simulation
*  @param print_id Current checkpoint id
*  @param average_dt Average time step size from the last output
*  @param it_count Total number of iterations so far
*/
		void output_cfg(T simtime, int print_id, T average_dt, int it_count);
		
		
/** @brief It calculates all custom timer value and output them.
*
*  @param st Timer object
*/		
		void write_times(SuperTimer::super_timer st, int print_id);
		
		
/** @brief It calculates average time of each timer for all MPI processes.
*
*  @param a Time array
*  @param n Size
*  @return Average value
*/		
		T average(T a[], int n);

/** @brief It writes the evolution of subdomain dimensions if dynamic load balancing is enabled.
*
*  @param pd Partition data
*  @param print_id Print output id
*/		
	void write_domain_decomposition(MpiUtils::partition_data_t pd, int print_id);
	




	private:
		int rows_;	/**< Number of rows in a subdomain */
		int cols_;	/**< Number of columns in a sub domain */
		T xll_;	/**< X coordinate of the origin */
		T yll_;	/**< Y coordinate of the origin */
		T cellsize_;	/**< Size of a cell */
		int rank_;	/**< Current subdomain id */
		int size_;	/**< Total number of sub domain */
		int time_series_flag_;	/**< Flag to determine output time series or not. If true, output the time series. */
		std::string project_dir_;	/**< Main project directory */
		std::string output_folder_;	/**< Output directory */
		std::string outfile_pattern_;	/**< Output file directory and name pattern */
		std::string cfg_content_;	/**< Contents of the input cfg (Configuration) file */
		std::string output_option_;	/**< Strategy to use for outputting into files. PAR for parallel outputs or SEQ for sequential outputs. PAR saves each MPI partitions subdomain in separate files and SEQ saves the whole domain into one file. */

		int num_of_obs_points_;	/**< Number of observation points per subdomain */
		int num_of_obs_points_global_;	/**< Number of observation points in the global domain */
		std::vector<int> relative_obs_index_;	/**< relative index position of observation cells per subdomain wrt to the global domain*/
		std::vector<int> time_series_index_relative_; /** relative index position of each observation cell in the global array after gathering*/
		int* relative_obs_index_global_;	/**< relative index position of observation cells in the global domain*/
		int* obs_points_per_subdomain;	/**< array of size MPI ranks containing the number of observation points per subdomain*/
		Constants::sources_list_t observation_cells_;	/**< Index position of observation cells in local domain */
		Constants::sources_list_t observation_cells_global_;	/**< Index position of observation cells in global domain */


	
	public:
		int cur_proc_data_size = 0;	/**< Number of cells in current subdomain */
		int *recvcounts = NULL;	/**< Array to hold every subdomains cell count */
		long long total_data_size = 0;	/**< Number of cells in main domain */
		int *displs = NULL;	/**< Position array to hold each sub domains starting point in main domain */
		T *total_data_arr = NULL;	/**< Main domains data or collection data of every subdomain */
		int *total_data_arr_int = NULL;	/**< Main domains data or collection data of every subdomain */
		int *displs_time_series = NULL ; /**< Position array to hold each sub domains starting point in main domain for time series */
		T *total_data_time_series = NULL ; /**< Main domain data for time series */
		
		

	};


	template<class T>
	output<T>::~output()
	{
		if (recvcounts != NULL)
		delete[] recvcounts;
		if (displs != NULL)
		delete[] displs;
		if (total_data_arr != NULL)
		delete[] total_data_arr;		
		if (displs_time_series != NULL)
		delete[] displs_time_series;
		if (total_data_time_series != NULL)
		delete[] total_data_time_series;		

	}


	template<typename T>
	void output<T>::init(int rows, int cols, T xll, T yll, T cellsize, int rank, int size, std::string project_dir, std::string output_folder, std::string outfile_pattern, int time_series_flag, std::string cfg_content, std::string output_option)
	{

		cur_proc_data_size = 0;
		if (recvcounts != NULL)
		delete[] recvcounts;
		total_data_size = 0;
		if (displs != NULL)
		delete[] displs;
		if (total_data_arr != NULL)
		delete[] total_data_arr;
		if (total_data_arr_int != NULL)
		delete[] total_data_arr_int;

		rows_ = rows;
		cols_ = cols;
		xll_ = xll;
		yll_ = yll;
		cellsize_ = cellsize;
		rank_ = rank;
		size_ = size;
		project_dir_ = project_dir;
		output_folder_ = output_folder;
		outfile_pattern_ = outfile_pattern;
		time_series_flag_ = time_series_flag;
		cfg_content_ = cfg_content;
		output_option_ = output_option;

		if (size == 1)
		{
			cur_proc_data_size = cols_ * rows_;
		}
		else if (rank_ == 0 || rank_ == size_ - 1)
		{
			cur_proc_data_size = cols_ * (rows_ - GHOST_CELL_PADDING);
		}
		else
		{
			cur_proc_data_size = cols_ * (rows_ - 2*GHOST_CELL_PADDING);
		}

		if (rank_ == 0)
		recvcounts = new int[size];
		MPI_Gather(&cur_proc_data_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, ENSIFY_COMM_WORLD);

		if (rank_ == 0)
		{
			displs = new int[size];
			displs[0] = 0;
			total_data_size += (long long) recvcounts[0];

			for (int i = 1; i < size_; i++)
			{
				total_data_size += (long long)recvcounts[i];
				displs[i] = displs[i - 1] + (long long)recvcounts[i - 1];
			}
			
			total_data_arr = new T[total_data_size];
			total_data_arr_int = new int[total_data_size];
		}

		MPI_Bcast(&total_data_size, 1, MPI_INT, 0, ENSIFY_COMM_WORLD);
	}


	template<typename T>
	void output<T>::init_time_series(int num_of_obs_points, int num_of_obs_points_global, std::vector<int> relative_obs_index, Constants::sources_list_t observation_cells, Constants::sources_list_t observation_cells_global, std::string print_option, int checkpoint_id)
	{

		if (displs_time_series != NULL)
		delete[] displs_time_series;
		if (total_data_time_series != NULL)
		delete[] total_data_time_series;	

		num_of_obs_points_ = num_of_obs_points;
		relative_obs_index_ = relative_obs_index;
		num_of_obs_points_global_=num_of_obs_points_global;
		observation_cells_ = observation_cells;
		observation_cells_global_ = observation_cells_global;

		if(rank_==0){
			obs_points_per_subdomain= new int[size_];
		}
		MPI_Gather(&num_of_obs_points_, 1, MPI_INT, obs_points_per_subdomain, 1, MPI_INT, 0, ENSIFY_COMM_WORLD);
		
		if (rank_ == 0)
		{
			displs_time_series = new int[size_];
			displs_time_series[0] = 0;

			for (int i = 1; i < size_; i++)
			{
				displs_time_series[i] = displs_time_series[i - 1] + obs_points_per_subdomain[i - 1];
			}
			
			total_data_time_series = new T[num_of_obs_points_global];
		}

		//auxiliary array to convert it from vector and to pass it to MPI_Gatherv
		int *relative_local_array = &relative_obs_index[0];

		relative_obs_index_global_ = NULL;
		if(rank_ == 0){
			relative_obs_index_global_ = (int*)malloc(num_of_obs_points_global_ * sizeof(int));
		}

    	MPI_Gatherv(relative_local_array, num_of_obs_points_, MPI_INT, relative_obs_index_global_, obs_points_per_subdomain, displs_time_series, MPI_INT, 0, ENSIFY_COMM_WORLD);

		if(rank_ == 0){
			for (int i = 0; i < num_of_obs_points_global_; i++){
				for (int j = 0; j < num_of_obs_points_global_; j++){
					if(i==relative_obs_index_global_[j]){
						time_series_index_relative_.push_back(j);
						break;
					}
				}
			}
		}

		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");
		DIR* dir;
		if(root_dir.empty())
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
		}
		else
		closedir(dir);
		root_dir.pop_back();

		std::string outdir(project_dir_ + "/" + output_folder_ + "/" + TIME_SERIES_DIR + "/");
		DIR* dir2;
		if(outdir.empty())
		{
			dir2 = opendir(".");
		}
		else
		{
			dir2 = opendir(outdir.c_str());
		}
		if (!dir2)
		{
			mkdir(outdir.c_str(), S_IRWXU);
		}
		else
		closedir(dir2);

		if (rank_ == 0 && checkpoint_id == 0)
		{
			if (print_option.find("h") != std::string::npos){
				std::string filedir = outdir + "H" + "_at_Xsec.txt";		

				std::string str = "Time(s)";
				for (int i = 0; i < num_of_obs_points_global_; i++)
				{
					str = str + "," + "H" + "_at_Point_" + std::to_string(i + 1);
				}
				str = str + "\n";
				std::ofstream output(filedir);
				output << str;
				output.close();
			}
			if (print_option.find("u") != std::string::npos){
				std::string filedir = outdir + "QX" + "_at_Xsec.txt";		

				std::string str = "Time(s)";
				for (int i = 0; i < num_of_obs_points_global_; i++)
				{
					str = str + "," + "QX" + "_at_Point_" + std::to_string(i + 1);
				}
				str = str + "\n";
				std::ofstream output(filedir);
				output << str;
				output.close();
			}
			if (print_option.find("v") != std::string::npos){
				std::string filedir = outdir + "QY" + "_at_Xsec.txt";		

				std::string str = "Time(s)";
				for (int i = 0; i < num_of_obs_points_global_; i++)
				{
					str = str + "," + "QY" + "_at_Point_" + std::to_string(i + 1);
				}
				str = str + "\n";
				std::ofstream output(filedir);
				output << str;
				output.close();
			}
		}

		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}

	}


	template<typename T>
	void output<T>::write_output(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, std::string output_format, std::string projection, std::string print_option, int print_id, int it_count, T simtime, T average_dt, Matrix::matrix<T>& max_value_h, std::string max_value_print_option)
	{
		if (strcmp(output_format.c_str(), "ASC") == 0)
		{
			if (print_option.find("h") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(h_arr, "H", print_id);
					write_output_ascii_sequential(h_arr, "H", print_id);
				}
				else
				{
					write_output_binary_parallel(h_arr, "H", print_id);
					write_output_ascii_parallel(h_arr, "H", print_id);
				}
			}

			if (print_option.find("u") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qx_arr, "QX", print_id);
					write_output_ascii_sequential(qx_arr, "QX", print_id);
				}
				else
				{
					write_output_binary_parallel(qx_arr, "QX", print_id);
					write_output_ascii_parallel(qx_arr, "QX", print_id);
				}
			}

			if (print_option.find("v") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qy_arr, "QY", print_id);
					write_output_ascii_sequential(qy_arr, "QY", print_id);
				}
				else
				{
					write_output_binary_parallel(qy_arr, "QY", print_id);
					write_output_ascii_parallel(qy_arr, "QY", print_id);
				}
			}
			
			
			if (max_value_print_option.size() > 0)
			{
				if (max_value_print_option.find("h") != std::string::npos)
				{
					if(strcmp(output_option_.c_str(), "SEQ") == 0)
					{
						write_output_binary_sequential(max_value_h, "MH", print_id);
						write_output_ascii_sequential(max_value_h, "MH", print_id);
					}
					else
					{
						write_output_binary_parallel(max_value_h, "MH", print_id);
						write_output_ascii_parallel(max_value_h, "MH", print_id);
					}
				}
			}
		}
		if (strcmp(output_format.c_str(), "BIN") == 0)
		{
			if (print_option.find("h") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(h_arr, "H", print_id);
				}
				else
				{
					write_output_binary_parallel(h_arr, "H", print_id);
				}
			}

			if (print_option.find("u") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qx_arr, "QX", print_id);
				}
				else
				{
					write_output_binary_parallel(qx_arr, "QX", print_id);
				}
			}

			if (print_option.find("v") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qy_arr, "QY", print_id);
				}
				else
				{
					write_output_binary_parallel(qy_arr, "QY", print_id);
				}
			}
			
			if (max_value_print_option.size() > 0)
			{
				if (max_value_print_option.find("h") != std::string::npos)
				{
					if(strcmp(output_option_.c_str(), "SEQ") == 0)
					{
						write_output_binary_sequential(max_value_h, "MH", print_id);
					}
					else
					{
						write_output_binary_parallel(max_value_h, "MH", print_id);
					}
				}
			}
		}
#ifdef TRITON_GDAL
		if (strcmp(output_format.c_str(), "GTIFF") == 0)
		{
			if (print_option.find("h") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_geotiff_sequential(h_arr, "H", print_id, projection);
					write_output_binary_sequential(h_arr, "H", print_id);      // write binary as well for hotstart 
				}
				else
				{
					write_output_geotiff_parallel(h_arr, "H", print_id, projection);
					write_output_binary_parallel(h_arr, "H", print_id);  // write binary as well for hotstart 
			}
		}

			if (print_option.find("u") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_geotiff_sequential(qx_arr, "QX", print_id, projection);
					write_output_binary_sequential(qx_arr, "QX", print_id);      // write binary as well for hotstart 
				}
				else
				{
					write_output_geotiff_parallel(qx_arr, "QX", print_id, projection);
					write_output_binary_parallel(qx_arr, "QX", print_id);  // write binary as well for hotstart 
				}
			}

			if (print_option.find("v") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_geotiff_sequential(qy_arr, "QY", print_id, projection);
					write_output_binary_sequential(qy_arr, "QY", print_id);      // write binary as well for hotstart 
				}
				else
				{
					write_output_geotiff_parallel(qy_arr, "QY", print_id, projection);
					write_output_binary_parallel(qy_arr, "QY", print_id);  // write binary as well for hotstart 
				}
			}
			
			if (max_value_print_option.size() > 0)
			{
				if (max_value_print_option.find("h") != std::string::npos)
				{
					if(strcmp(output_option_.c_str(), "SEQ") == 0)
					{
						write_output_geotiff_sequential(max_value_h, "MH", print_id, projection);
						write_output_binary_sequential(max_value_h, "MH", print_id);      // write binary as well for hotstart 
					}
					else
					{
						write_output_geotiff_parallel(max_value_h, "MH", print_id, projection);
						write_output_binary_parallel(max_value_h, "MH", print_id);  // write binary as well for hotstart 
					}
				}
			}
		}
#endif
		write_output_ghost_ring(h_arr, qx_arr, qy_arr, print_id);

		if (rank_ == 0)
		{

			std::cerr << BLUE << "[" << (print_id) << "]" << RESET " File written at time " << simtime << " seconds. " << std::endl;
			output_cfg(simtime, print_id, average_dt, it_count);
			std::ofstream cidfile("cid");
			if (cidfile.is_open())
			{
				cidfile << print_id;
			}
			cidfile.close();
		}

	}
	

	template<typename T>
	void output<T>::write_output_ascii_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id)
	{
		if(rank_ == 0)
		{
			std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

			DIR* dir;
			if(root_dir.empty())
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
			}
			else
			closedir(dir);
			root_dir.pop_back();

			std::string filepath = get_mat_path(what_mat, root_dir, ASCII_DIR, print_id, ".out");
			std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + ASCII_DIR + "/");

			DIR* dir2;
			if(file_dir.empty())
			{
				dir2 = opendir(".");
			}
			else
			{
				dir2 = opendir(file_dir.c_str());
			}
			if (!dir2)
			{
				mkdir(file_dir.c_str(), S_IRWXU);
			}
			else
			closedir(dir2);

			std::ofstream mat;
			mat.precision(6);
			mat.open((filepath).c_str());
			mat << std::fixed;
			
			int off = GHOST_CELL_PADDING;
			int total_cols = cols_;
			int total_rows = total_data_size/ total_cols;
			for(int i=off; i<total_rows-off; i++)
			{
				for(int j=off; j<total_cols-off;j++)
				{
					mat << total_data_arr[i*(long long)total_cols+j];

					if (j < total_cols - off - 1)
					{
						mat << " ";
					}
				}
				mat << std::endl;
			}
			mat.close();
		}
		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}
	}


	template<typename T>
	void output<T>::write_output_ghost_ring(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, int print_id)
	{
		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");
		root_dir.pop_back();

		std::string filepath = get_mat_path("GR", root_dir, BIN_DIR, print_id, ".out");

		std::ofstream ring((filepath).c_str(), std::ios::binary);
		if (!ring.is_open())
		{
			std::cerr << WARN "Could not write ghost-ring side-file " << filepath
			          << "; a resume from this checkpoint will fail." << std::endl;
			return;
		}

		int off = GHOST_CELL_PADDING;
		T put_rows_value = (T)rows_;
		T put_cols_value = (T)cols_;

		ring.write((char*)&put_rows_value, sizeof(T));
		ring.write((char*)&put_cols_value, sizeof(T));

		for (int i = 0; i < rows_; i++)
		{
			for (int j = 0; j < cols_; j++)
			{
				if (i < off || i >= rows_ - off || j < off || j >= cols_ - off)
				{
					ring.write((char*)h_arr.get_address_at(i, j),  sizeof(T));
					ring.write((char*)qx_arr.get_address_at(i, j), sizeof(T));
					ring.write((char*)qy_arr.get_address_at(i, j), sizeof(T));
				}
			}
		}
		ring.close();
	}


	template<typename T>
	void output<T>::read_output_ghost_ring(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, int print_id)
	{
		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");
		root_dir.pop_back();

		std::string filepath = get_mat_path("GR", root_dir, BIN_DIR, print_id, ".out");

		std::ifstream ring((filepath).c_str(), std::ios::binary);
		if (!ring.is_open())
		{
			std::cerr << ERROR "Ghost-ring side-file not found: " << filepath << std::endl;
			std::cerr << "      Checkpoints written before this fix do not carry it; re-run the clean leg." << std::endl;
			exit(EXIT_FAILURE);
		}

		int off = GHOST_CELL_PADDING;
		T get_rows_value = (T)0;
		T get_cols_value = (T)0;

		ring.read((char*)&get_rows_value, sizeof(T));
		ring.read((char*)&get_cols_value, sizeof(T));

		if ((int)get_rows_value != rows_ || (int)get_cols_value != cols_)
		{
			std::cerr << ERROR "Ghost-ring side-file dimensions (" << (int)get_rows_value << ", " << (int)get_cols_value
			          << ") do not match the current subdomain (" << rows_ << ", " << cols_ << ")." << std::endl;
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < rows_; i++)
		{
			for (int j = 0; j < cols_; j++)
			{
				if (i < off || i >= rows_ - off || j < off || j >= cols_ - off)
				{
					ring.read((char*)h_arr.get_address_at(i, j),  sizeof(T));
					ring.read((char*)qx_arr.get_address_at(i, j), sizeof(T));
					ring.read((char*)qy_arr.get_address_at(i, j), sizeof(T));
				}
			}
		}

		if (ring.gcount() != (std::streamsize)sizeof(T))
		{
			std::cerr << ERROR "Ghost-ring side-file is truncated: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		ring.close();
	}


	template<typename T>
	std::ostream& operator<<(std::ostream& out, Matrix::matrix<T>& M)
	{
		int m = M.get_num_rows();
		int n = M.get_num_cols();
		int off = GHOST_CELL_PADDING;
		for (int i = off; i < m - off; i++)
		{
			for (int j = off; j < n - off; j++)
			{
				out << (T)M(i, j);

				if (j < (n - off) - off)
				{
					out << " ";
				}
			}
			out << std::endl;
		}

		return out;
	}


	template<typename T>
	void output<T>::write_output_ascii_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id)
	{
		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

		DIR* dir;
		if(root_dir.empty())
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
		}
		else
		closedir(dir);
		
		root_dir.pop_back();

		std::string filepath = get_mat_path(what_mat, root_dir, ASCII_DIR, print_id, ".out");
		std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + ASCII_DIR + "/");

		DIR* dir2;
		if(file_dir.empty())
		{
			dir2 = opendir(".");
		}
		else
		{
			dir2 = opendir(file_dir.c_str());
		}
		if (!dir2)
		{
			mkdir(file_dir.c_str(), S_IRWXU);
		}
		else
		closedir(dir2);

		std::ofstream mat;
		mat.precision(6);
		mat.open((filepath).c_str());
		mat << std::fixed << arr;
		mat.close();


	}


	template<typename T>
	void output<T>::write_output_binary_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id)
	{
		if (rank_ == 0)
		{
			MPI_Gatherv(arr.get_address_at(0, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		}
		else
		{
			MPI_Gatherv(arr.get_address_at(GHOST_CELL_PADDING, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		}

		if (rank_ == 0)
		{
			std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

			DIR* dir;
			if(root_dir.empty())
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
			}
			else
			closedir(dir);

			root_dir.pop_back();


			std::string filepath = get_mat_path(what_mat, root_dir, BIN_DIR, print_id, ".out");
			std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + BIN_DIR + "/");

			DIR* dir2;
			if(file_dir.empty())
			{
				dir2 = opendir(".");
			}
			else
			{
				dir2 = opendir(file_dir.c_str());
			}
			if (!dir2)
			{
				mkdir(file_dir.c_str(), S_IRWXU);
			}
			else
			closedir(dir2);

			std::ofstream mat((filepath).c_str(), std::ios::binary);
			
			int total_cols = cols_;
			int total_rows = total_data_size/ total_cols;
			int off = GHOST_CELL_PADDING;
			
			T put_rows_value = (T)(total_rows - 2 * off);
			T put_cols_value = (T)(total_cols - 2 * off);
			
			mat.write((char*) &put_rows_value, sizeof(T));
			mat.write((char*) &put_cols_value, sizeof(T));
			
			for(int i=off; i<total_rows-off; i++)
			{
				mat.write((char*) &total_data_arr[i*(long long)total_cols+off], (total_cols-2*off) * sizeof(T));
			}
			mat.close();
		}
		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}
	}


	template<typename T>
	void output<T>::write_output_binary_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id)
	{
		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

		DIR* dir;
		if(root_dir.empty())
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
		}
		else
		closedir(dir);
		root_dir.pop_back();


		std::string filepath = get_mat_path(what_mat, root_dir, BIN_DIR, print_id, ".out");
		std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + BIN_DIR + "/");

		DIR* dir2;
		if(file_dir.empty())
		{
			dir2 = opendir(".");
		}
		else
		{
			dir2 = opendir(file_dir.c_str());
		}
		if (!dir2)
		{
			mkdir(file_dir.c_str(), S_IRWXU);
		}
		else
		closedir(dir2);

		std::ofstream mat((filepath).c_str(), std::ios::binary);
		int off = GHOST_CELL_PADDING;
		
		T put_rows_value = (T)(rows_ - 2 * off);
		T put_cols_value = (T)(cols_ - 2 * off);
		
		mat.write((char*) &put_rows_value, sizeof(T));
		mat.write((char*) &put_cols_value, sizeof(T));
		
		for(int i=off; i<rows_-off; i++)
		{
			mat.write((char*)arr.get_address_at(i,off), (cols_-2*off) * sizeof(T));
		}
		
		mat.close();
	}


#ifdef TRITON_GDAL
	template<typename T>
	void output<T>::write_output_geotiff_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, std::string projection)
	{
		if (rank_ == 0)
		{
			MPI_Gatherv(arr.get_address_at(0, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		}
		else
		{
			MPI_Gatherv(arr.get_address_at(GHOST_CELL_PADDING, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		}

		if (rank_ == 0)
		{
			std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

			DIR* dir;
			if(root_dir.empty())
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
			}
			else
			closedir(dir);

			root_dir.pop_back();


			std::string filepath = get_mat_path(what_mat, root_dir, std::string(GEO_DIR), print_id, ".tif");
			std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + GEO_DIR + "/");

			DIR* dir2;
			if(file_dir.empty())
			{
				dir2 = opendir(".");
			}
			else
			{
				dir2 = opendir(file_dir.c_str());
			}
			if (!dir2)
			{
				mkdir(file_dir.c_str(), S_IRWXU);
			}
			else
			closedir(dir2);

			// Initialize GDAL
			GDALAllRegister();

			int total_cols = cols_;
			int total_rows = total_data_size / total_cols;
			int off = GHOST_CELL_PADDING;

			int raster_rows = total_rows - 2 * off;
			int raster_cols = total_cols - 2 * off;

			GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
			if (poDriver == nullptr)
			{
				std::cerr << "Unable to obtain GeoTIFF driver" << std::endl;
				return;
			}

			GDALDataset* poDataset = poDriver->Create(filepath.c_str(), raster_cols, raster_rows, 1, GDT_Float32, nullptr);
			if (poDataset == nullptr)
			{
				std::cerr << "Error creating the GeoTIFF file in " << filepath << std::endl;
				return;
			}

			// Geo-referenced transformation. GeoTiff uses top left corner as origin
			// T adfGeoTransform[6] = { xOrigin, pixelWidth, xRotation, yOrigin, yRotation, pixelHeight };
			T yll = yll_;
			T xll = xll_;
			T pixel_size = cellsize_;
			T y_ul = yll + raster_rows * pixel_size; // Upper left corner y coordinate
			// Convert to double for GDAL  
			double adfGeoTransform[6] = {   
				static_cast<double>(xll),   
				static_cast<double>(pixel_size),   
				0.0,   
				static_cast<double>(y_ul),   
				0.0,   
				static_cast<double>(-pixel_size)   
			};  
		poDataset->SetGeoTransform(adfGeoTransform);

			// Assign coordinate system
			OGRSpatialReference oSRS;
			oSRS.SetFromUserInput(projection.c_str()); // Pass projection string from user #edited SG 
			char* pszSRSWKT = nullptr;
			oSRS.exportToWkt(&pszSRSWKT);
			poDataset->SetProjection(pszSRSWKT);
			CPLFree(pszSRSWKT);

			GDALRasterBand* poBand = poDataset->GetRasterBand(1);

			for (int i = 0; i < raster_rows; i++)
			{
				float *pafWriteline = (float *)CPLMalloc(sizeof(float) * raster_cols);
				for (int j = 0; j < raster_cols; j++)
				{
					pafWriteline[j] = total_data_arr[(i + off) * (long long)total_cols + j + off];
				}
				CPLErr err = poBand->RasterIO(GF_Write, 0, i, raster_cols, 1, pafWriteline, raster_cols, 1, GDT_Float32, 0, 0);
				if (err != CE_None) 
				{
					std::cerr << "Error writing to raster in row " << i << std::endl;
				}
				CPLFree(pafWriteline);
			}
		
			GDALClose(poDataset);
		}

		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}
	}


	template<typename T>
	void output<T>::write_output_geotiff_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, std::string projection)
	{
		std::string root_dir(project_dir_ + "/" + output_folder_ + "/");

		DIR* dir;
		if(root_dir.empty())
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
		}
		else
		closedir(dir);
		root_dir.pop_back();


		std::string filepath = get_mat_path(what_mat, root_dir, GEO_DIR, print_id, ".tif");
		std::string file_dir(project_dir_ + "/" + output_folder_ + "/" + GEO_DIR + "/");

		DIR* dir2;
		if(file_dir.empty())
		{
			dir2 = opendir(".");
		}
		else
		{
			dir2 = opendir(file_dir.c_str());
		}
		if (!dir2)
		{
			mkdir(file_dir.c_str(), S_IRWXU);
		}
		else
		closedir(dir2);

		// Initialize GDAL
		GDALAllRegister();

		int off = GHOST_CELL_PADDING;
		int raster_rows = rows_ - 2 * off; // number of local rows without ghost cells for each subdomain
		int raster_cols = cols_ - 2 * off;

		// Gather number of rows from all processes to calculate y_ul correctly
		std::vector<int> all_rows(size_);
		MPI_Allgather(&raster_rows, 1, MPI_INT, all_rows.data(), 1, MPI_INT, ENSIFY_COMM_WORLD);
		int rows_below = 0;
		for (int r = rank_; r < size_; ++r) {
			rows_below += all_rows[r];
		}
		
		GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
		if (poDriver == nullptr)
		{
			std::cerr << "Unable to obtain GeoTIFF driver" << std::endl;
			return;
		}

		GDALDataset* poDataset = poDriver->Create(filepath.c_str(), raster_cols, raster_rows, 1, GDT_Float32, nullptr);
		if (poDataset == nullptr)
		{
			std::cerr << "Error creating GeoTIFF file in" << filepath << std::endl;
			return;
		}


		// Geo-referenced transformation. GeoTiff uses top left corner as origin
		// T adfGeoTransform[6] = { xOrigin, pixelWidth, xRotation, yOrigin, yRotation, pixelHeight };
		T yll = yll_; 
		T xll = xll_;
		T pixel_size = cellsize_;
		T y_ul = yll + rows_below * pixel_size; // Upper left corner y coordinate
		//T adfGeoTransform[6] = { xll, pixel_size, 0.0, y_ul, 0.0, -pixel_size };

		// Convert to double for GDAL  
		double adfGeoTransform[6] = {   
			static_cast<double>(xll),   
			static_cast<double>(pixel_size),   
			0.0,   
			static_cast<double>(y_ul),   
			0.0,   
			static_cast<double>(-pixel_size)   
		};  

		poDataset->SetGeoTransform(adfGeoTransform);

		// Assign coordinate system
		OGRSpatialReference oSRS;
		oSRS.SetFromUserInput(projection.c_str()); // Pass projection string from user
		char* pszSRSWKT = nullptr;
		oSRS.exportToWkt(&pszSRSWKT);
		poDataset->SetProjection(pszSRSWKT);
		CPLFree(pszSRSWKT);

		GDALRasterBand* poBand = poDataset->GetRasterBand(1);

		for (int i = 0; i < raster_rows; i++)
		{
			float *pafWriteline = (float *)CPLMalloc(sizeof(float) * raster_cols);
			for (int j = 0; j < raster_cols; j++)
			{
				pafWriteline[j] = *arr.get_address_at(i,j); // This way we access the value of the pointer that points to the memory address (i,j) of the matrix, which is what we are interested in.
			}

			CPLErr err = poBand->RasterIO(GF_Write, 0, i, raster_cols, 1, pafWriteline, raster_cols, 1, GDT_Float32, 0, 0);
			if (err != CE_None) 
			{
				std::cerr << "Error writing to raster in row " << i << std::endl;
			}
			CPLFree(pafWriteline);
		}
		
		GDALClose(poDataset);
		
		// Ensure all processes have finished writing before creating VRT
		MPI_Barrier(ENSIFY_COMM_WORLD);
		if (rank_ == 0) {
			write_output_vrt(what_mat, print_id, all_rows, raster_cols, xll_, yll_, cellsize_, projection, file_dir);
		}
	}

	template <typename T>
	void output<T>::write_output_vrt(const std::string &what_mat, int print_id,
				const std::vector<int> &all_rows, int raster_cols,
				double xll, double yll, double cellsize,
				const std::string &projection,
				const std::string &file_dir)
	{
		// Total number of rows in the full domain
		int total_rows = 0;
		for (int r : all_rows) total_rows += r;

		// VRT file name
		std::ostringstream vrt_name;
		vrt_name << file_dir << what_mat << "_" << std::setw(2) << std::setfill('0') << print_id << ".vrt";

		std::ofstream vrt(vrt_name.str());
		vrt << "<VRTDataset rasterXSize=\"" << raster_cols
			<< "\" rasterYSize=\"" << total_rows << "\">\n";

		// GeoTransform
		double y_ul = yll + total_rows * cellsize;
		vrt << "  <GeoTransform> "
			<< xll << ", " << cellsize << ", 0.0, "
			<< y_ul << ", 0.0, " << -cellsize
			<< " </GeoTransform>\n";

		// Spatial Reference
		vrt << "  <SRS>" << projection << "</SRS>\n";

		vrt << "  <VRTRasterBand dataType=\"Float32\" band=\"1\">\n";
		vrt << "    <ColorInterp>Gray</ColorInterp>\n";

		int offset = 0;
		for (size_t r = 0; r < all_rows.size(); ++r) {
			std::ostringstream tif_name;
			tif_name << what_mat << "_" << std::setw(2) << std::setfill('0') << print_id
					<< "_" << std::setw(2) << r << ".tif";

			vrt << "    <SimpleSource>\n";
			vrt << "      <SourceFilename relativeToVRT=\"1\">" << tif_name.str() << "</SourceFilename>\n";
			vrt << "      <SourceBand>1</SourceBand>\n";
			vrt << "      <SourceProperties RasterXSize=\"" << raster_cols
				<< "\" RasterYSize=\"" << all_rows[r]
				<< "\" DataType=\"Float32\" BlockXSize=\"" << raster_cols
				<< "\" BlockYSize=\"1\" />\n";
			vrt << "      <SrcRect xOff=\"0\" yOff=\"0\" xSize=\"" << raster_cols
				<< "\" ySize=\"" << all_rows[r] << "\" />\n";
			vrt << "      <DstRect xOff=\"0\" yOff=\"" << offset
				<< "\" xSize=\"" << raster_cols
				<< "\" ySize=\"" << all_rows[r] << "\" />\n";
			vrt << "    </SimpleSource>\n";

			offset += all_rows[r];
		}

		vrt << "  </VRTRasterBand>\n";
		vrt << "</VRTDataset>\n";
		vrt.close();
	}


#endif

	template<typename T>
	std::string output<T>::get_mat_path(std::string what, std::string root_dir, std::string subdir, int print_id, std::string extension)
	{
		std::string format = outfile_pattern_ + extension;
		std::vector<char> buf(256);

		std::snprintf(
		&buf[0], buf.size(), format.c_str(),
		root_dir.c_str(),
		subdir.c_str(),
		what.c_str(),
		print_id,
		rank_
		);

		return std::string(&buf[0]);
	}

	template<typename T>
	void output<T>::output_time_series(T *value_obs, std::string what_mat, T simtime)
	{
		std::string outdir = project_dir_ + "/" + output_folder_ + "/" + TIME_SERIES_DIR + "/";
		std::string filedir = outdir + what_mat + "_at_Xsec.txt";		

		T* value_obs_global = NULL;

		if(rank_ == 0){
			value_obs_global = (T*)malloc(num_of_obs_points_global_ * sizeof(T));
		}

    	MPI_Gatherv(value_obs, num_of_obs_points_, MPI_DATA_TYPE, value_obs_global, obs_points_per_subdomain, displs_time_series, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);

		if(rank_ == 0){
			std::string str = std::to_string(simtime);
			std::ofstream out(filedir, std::ios::app);
			for (int i = 0; i < num_of_obs_points_global_; i++)
			{
				str = str + "," + std::to_string(value_obs_global[time_series_index_relative_[i]]);
			}	
			str = str + "\n";
			out << str;
			out.close();
		}

		if(rank_ == 0){
			free(value_obs_global);
		}

		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}

	}

	template<typename T>
	void output<T>::write_observation_data( T *h_arr_obs, T *qx_arr_obs, T *qy_arr_obs, T simtime, std::string print_option)
	{
		if (print_option.find("h") != std::string::npos)
		{
			output_time_series(h_arr_obs, "H", simtime);
		}
		if (print_option.find("u") != std::string::npos)
		{
			output_time_series(qx_arr_obs, "QX", simtime);
		}
		if (print_option.find("v") != std::string::npos)
		{
			output_time_series(qy_arr_obs, "QY", simtime);
		}
		if (rank_ == 0)
		{
			std::cerr << BLUE << "[OBS]" << RESET " Observation data written at time " << simtime << " seconds. " << std::endl;
		}

		
	}



	template<typename T>
	void output<T>::output_cfg(T simtime, int print_id, T average_dt, int it_count)
	{
		std::string str2 = cfg_content_;
		std::istringstream ss(str2);
		std::string line;
		while (getline(ss, line))
		{
			if (line.find("sim_start_time=") != std::string::npos)
			{
				size_t startPos = str2.find("sim_start_time=");
				std::ostringstream streamObj;
				streamObj << std::fixed;
				streamObj << std::setprecision(std::numeric_limits<T>::max_digits10);
				streamObj << simtime;
				std::string strObj = streamObj.str();

				str2.replace(startPos, line.length(), "sim_start_time=" + strObj);
			}
			else if (line.find("checkpoint_id=") != std::string::npos)
			{
				size_t startPos = str2.find("checkpoint_id=");
				str2.replace(startPos, line.length(), "checkpoint_id=" + std::to_string(print_id));
			}
			else if (line.find("time_step=") != std::string::npos)
			{
				size_t startPos = str2.find("time_step=");
				std::ostringstream streamObj;
				streamObj << std::fixed;
				streamObj << std::setprecision(std::numeric_limits<T>::max_digits10);
				streamObj << average_dt;
				std::string strObj = streamObj.str();

				str2.replace(startPos, line.length(), "time_step=" + strObj);
			}
			else if (line.find("it_count=") != std::string::npos)
			{
				size_t startPos = str2.find("it_count=");
				str2.replace(startPos, line.length(), "it_count=" + std::to_string(it_count));
			}
		}
		std::string outdir = project_dir_ + "/" + output_folder_ + "/" + CFG_DIR + "/";
		DIR* dir;
		if(outdir.empty())
		{
			dir = opendir(".");
		}
		else
		{
			dir = opendir(outdir.c_str());
		}
		if (!dir)
		{
			mkdir(outdir.c_str(), S_IRWXU);
		}
		else
		closedir(dir);

		std::string filedir = outdir + "config_" + std::to_string(print_id) + ".cfg";

		std::ofstream output(filedir);
		output << str2;
		output.close();
	}


	template<typename T>
	void output<T>::write_times(SuperTimer::super_timer st, int print_id)
	{

		T compute_time = st.get_custom_time(COMPUTE_TIME);
		T mpi_time = st.get_custom_time(MPI_TIME);
		T io_time = st.get_custom_time(IO_TIME);
		T resize_time = st.get_custom_time(RESIZE_TIME);
		T swmm_time = st.get_custom_time(SWMM_TIME);
		// The three MEASURED children of SWMM_TIME.  Their brackets sit inside the
		// parent's in triton.h, so each is already a PART of swmm_time rather than
		// an addition to it.
		T swmm_xfer_time = st.get_custom_time(SWMM_XFER);
		T swmm_mpi_time = st.get_custom_time(SWMM_MPI);
		T swmm_step_time = st.get_custom_time(SWMM_STEP);
		T simulation_time = st.get_custom_time(SIMULATION_TIME);
		T total_time = st.get_custom_time(TOTAL_TIME);
		// UNCHANGED, and deliberately so.  other_time is the SIMULATION-level
		// residual and subtracts the PARENT swmm_time; the three children are
		// already contained in it.  Subtracting them here as well would double
		// count and would stop the Simulation level closing.
		T other_time = simulation_time - compute_time - mpi_time - io_time - resize_time - swmm_time;
		// The SWMM-level residual, derived exactly as other_time and init_time are
		// and, like them, carrying no timer macro of its own.  This subtraction is
		// what makes the new level close: by construction
		// swmm_xfer + swmm_mpi + swmm_step + swmm_other == swmm_time.  It absorbs
		// the two sizeof-scaled assignments at the top of the coupling block, the
		// `if (rank == 0)` branch test that EVERY rank evaluates, and the category
		// lookups the instrumentation itself performs.
		T swmm_other_time = swmm_time - swmm_xfer_time - swmm_mpi_time - swmm_step_time;
		T init_time = total_time - simulation_time;

		T *compute_time_all = new T[size_];
		T *mpi_time_all = new T[size_];
		T *io_time_all = new T[size_];
		T *simulation_time_all = new T[size_];
		T *total_time_all = new T[size_];
		T *other_time_all = new T[size_];
		T *init_time_all = new T[size_];
		T *resize_time_all = new T[size_];
		T *swmm_time_all = new T[size_];
		T *swmm_xfer_time_all = new T[size_];
		T *swmm_mpi_time_all = new T[size_];
		T *swmm_step_time_all = new T[size_];
		T *swmm_other_time_all = new T[size_];


		MPI_Gather(&compute_time, 1, MPI_DATA_TYPE, &compute_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&mpi_time, 1, MPI_DATA_TYPE, &mpi_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&io_time, 1, MPI_DATA_TYPE, &io_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&simulation_time, 1, MPI_DATA_TYPE, &simulation_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&total_time, 1, MPI_DATA_TYPE, &total_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&other_time, 1, MPI_DATA_TYPE, &other_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&init_time, 1, MPI_DATA_TYPE, &init_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&resize_time, 1, MPI_DATA_TYPE, &resize_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&swmm_time, 1, MPI_DATA_TYPE, &swmm_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&swmm_xfer_time, 1, MPI_DATA_TYPE, &swmm_xfer_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&swmm_mpi_time, 1, MPI_DATA_TYPE, &swmm_mpi_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&swmm_step_time, 1, MPI_DATA_TYPE, &swmm_step_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);
		MPI_Gather(&swmm_other_time, 1, MPI_DATA_TYPE, &swmm_other_time_all[rank_], 1, MPI_DATA_TYPE, 0, ENSIFY_COMM_WORLD);

		if (size_ > 1)
		{
			MPI_Barrier(ENSIFY_COMM_WORLD);
		}

		if(rank_ == 0)
		{
			std::string outdir;
			std::string filedir;

			//not the final state
			if(print_id!=-1){
				std::string outdir = project_dir_ + "/" + output_folder_ + "/performance/";
				DIR* dir;
				if(outdir.empty())
				{
					dir = opendir(".");
				}
				else
				{
					dir = opendir(outdir.c_str());
				}
				if (!dir)
				{
					mkdir(outdir.c_str(), S_IRWXU);
				}
				else
				closedir(dir);

				filedir = outdir + "performance" +std::to_string(print_id) + ".txt";

			}else{
				//the final state
				std::string outdir = project_dir_ + "/" + output_folder_ + "/";
				filedir = outdir + "performance.txt";	
			}
			std::ofstream output(filedir);
			// The four new columns sit immediately after their parent, so a reader
			// meets SWMM and then its decomposition.  SWMM_OTHER is the residual and
			// carries no timer of its own; SWMM_MPI is distinct from the pre-existing
			// MPI column, which times TRITON's own halo exchange rather than the
			// coupling's gather/scatter.
			output << "%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER, Other, Simulation, Init, Total" << std::endl;

			for(int j=0;j<size_;j++){
				output << std::setprecision(4) << j << ", " << compute_time_all[j] << ", " <<  mpi_time_all[j] << ", " <<	io_time_all[j] << ", " <<	resize_time_all[j] << ", " << swmm_time_all[j] << ", "
				<< swmm_xfer_time_all[j] << ", " << swmm_mpi_time_all[j] << ", " << swmm_step_time_all[j] << ", " << swmm_other_time_all[j] << ", " << other_time_all[j] << ", "
				<< simulation_time_all[j] << ", " << init_time_all[j] <<  ", " << total_time_all[j] << std::endl;
			}
			// The Average row is an arithmetic mean over ranks, emitted for every
			// column so the row's arity stays fixed -- the downstream parser's
			// Average-presence detector depends on that.  It is OUTSIDE the closure
			// claim above: average(SWMM_STEP) is rank0/N, which halves as rank count
			// doubles while the serial solve is constant.  The serial-solve cost is
			// the MAX over Rank, not this mean.
			output << std::setprecision(4) << "Average" << ", " << average(compute_time_all,size_) << ", " <<  average(mpi_time_all,size_) << ", " <<	 average(io_time_all,size_) << ", " << average(resize_time_all,size_) << ", " << average(swmm_time_all,size_) << ", "
			<< average(swmm_xfer_time_all,size_) << ", " << average(swmm_mpi_time_all,size_) << ", " << average(swmm_step_time_all,size_) << ", " << average(swmm_other_time_all,size_) << ", " <<  average(other_time_all,size_) << ", " <<  average(simulation_time_all,size_) << ", " <<  average(init_time_all,size_) <<  ", " << average(total_time_all,size_) << std::endl;

			output.close();
		}
		
	}
	
	
	template<typename T>
	T output<T>::average(T a[], int n) 
	{ 
		T sum = 0; 
		for (int i=0; i<n; i++) 
		sum+= a[i]; 

		return sum/n; 
	}	 
	template<typename T>
	void output<T>::write_domain_decomposition(MpiUtils::partition_data_t pd, int print_id)
	{
		if(rank_ == 0)
		{

			std::string outdir = project_dir_ + "/" + output_folder_ + "/domain_decomposition/";
			DIR* dir;
			if(outdir.empty())
			{
				dir = opendir(".");
			}
			else
			{
				dir = opendir(outdir.c_str());
			}
			if (!dir)
			{
				mkdir(outdir.c_str(), S_IRWXU);
			}
			else
			closedir(dir);

			std::string filedir = outdir + "domain_decomposition" + std::to_string(print_id) + ".txt";
			std::ofstream output(filedir);
			output << "%Rank,Rows,Cols" << std::endl;

			for(int j=0;j<size_;j++){
				output << j << ","  << pd.part_dims[j].first-2*GHOST_CELL_PADDING << ","  << pd.part_dims[j].second  - 2*GHOST_CELL_PADDING << std::endl;
			}
			output.close();

		}
		
		 if (size_ > 1)
		 {
					MPI_Barrier(ENSIFY_COMM_WORLD);
		 }



	}


/** @brief Normalize a path by removing trailing slashes and handling empty paths.
*
*  @param path The input path to normalize
*  @return Normalized path without trailing slash (or empty string if input is empty)
*/
inline std::string normalize_path(std::string path)
{
	if (path.empty()) return path;

	// Remove trailing slashes
	size_t end = path.find_last_not_of("/");
	if (end != std::string::npos)
	{
		path = path.substr(0, end + 1);
	}

	return path;
}


/** @brief Build output directory path by joining project_dir and output_folder.
*
*  This function handles absolute paths correctly - if output_folder is absolute,
*  it is used as-is. If relative, it is joined with project_dir.
*
*  @param project_dir The project directory path
*  @param output_folder The output folder from config (may be absolute or relative)
*  @return Normalized output directory path
*/
inline std::string build_output_path(std::string project_dir, std::string output_folder)
{
	project_dir = normalize_path(project_dir);
	output_folder = normalize_path(output_folder);

	if (output_folder.empty()) return project_dir;

	// Check if output_folder is an absolute path
	if (output_folder[0] == '/')
	{
		return output_folder;
	}

	// Relative path - join with project_dir
	return project_dir + "/" + output_folder;
}


/** @brief Recursively create directories for a given path.
*
*  This function creates all parent directories if they don't exist.
*
*  @param path The full directory path to create
*  @return true if successful or directory already exists, false on failure
*/
inline bool create_directories_recursive(const std::string& path)
{
	if (path.empty()) return false;

	// Check if directory already exists
	DIR* dir = opendir(path.c_str());
	if (dir)
	{
		closedir(dir);
		return true;
	}

	// Find the last separator and recursively create parent directories
	size_t pos = path.find_last_of("/");
	if (pos == 0 || pos == std::string::npos)
	{
		// Root directory or no more parents - try direct mkdir
		return (mkdir(path.c_str(), S_IRWXU) == 0 || errno == EEXIST);
	}

	// Recursively create parent directory
	std::string parent = path.substr(0, pos);
	if (!create_directories_recursive(parent))
	{
		return false;
	}

	// Create the final directory
	return (mkdir(path.c_str(), S_IRWXU) == 0 || errno == EEXIST);
}


/** @brief Writes the TRITON run header to output/log.out
*
*  This function writes run configuration information including machine,
*  CPU, MPI tasks, OpenMP threads, GPU configuration, and git version.
*  Only MPI rank 0 writes to the file.
*
*  @param project_dir The project directory path
*  @param output_folder The output folder from config (may be empty for default)
*  @param rank Current MPI rank
*  @param size Total number of MPI ranks
*/
inline void triton_log_run_header(std::string project_dir, std::string output_folder, int rank, int size)
{
	// Only rank 0 writes the log header
	if (rank != 0) return;

	// Build output path using helper function
	std::string outdir = build_output_path(project_dir, output_folder);

	// Create output directory recursively if it doesn't exist
	create_directories_recursive(outdir);

	// Open log file
	std::string logfile = outdir + "/log.out";
	std::ofstream log(logfile);

	// Get machine name (hostname)
	char hostname[256];
	if (gethostname(hostname, sizeof(hostname)) == 0)
	{
		log << "---- TRITON RUN INFO ----" << std::endl;
		log << "Machine : " << hostname << std::endl;
	}
	else
	{
		log << "---- TRITON RUN INFO ----" << std::endl;
		log << "Machine : unknown" << std::endl;
	}

	// Get CPU model (Linux specific)
	#ifdef __linux__
	std::ifstream cpuinfo("/proc/cpuinfo");
	if (cpuinfo.is_open())
	{
		std::string line;
		while (std::getline(cpuinfo, line))
		{
			if (line.find("model name") != std::string::npos)
			{
				size_t pos = line.find(":");
				if (pos != std::string::npos)
				{
					log << "CPU : " << line.substr(pos + 2) << std::endl;
					break;
				}
			}
		}
		cpuinfo.close();
	}
	else
	{
		log << "CPU : unknown" << std::endl;
	}
	#else
	log << "CPU : unknown" << std::endl;
	#endif

	// GPU model (device name of the Kokkos-selected device on this rank).
	// Kokkos caches the device properties at Kokkos::initialize(); this header is
	// reached from main.cpp INSIDE the initialize()/finalize() scope, so the
	// accessors below are live. No new include: constants.h already pulls
	// Kokkos_Core.hpp, which declares Kokkos::Cuda / Kokkos::HIP when the backend
	// is enabled. Note the accessor asymmetry: cuda_device_prop() is a const
	// member, hip_device_prop() is static.
	#if defined(KOKKOS_ENABLE_CUDA)
	log << "GPU : " << Kokkos::Cuda().cuda_device_prop().name << std::endl;
	#elif defined(KOKKOS_ENABLE_HIP)
	log << "GPU : " << Kokkos::HIP::hip_device_prop().name << std::endl;
	#else
	log << "GPU : none" << std::endl;
	#endif

	// MPI tasks
	log << "nTasks : " << size << std::endl;

	// OpenMP threads
	#ifdef _OPENMP
	int omp_threads = omp_get_max_threads();
	log << "OMP threads per task : " << omp_threads << std::endl;
	#else
	log << "OMP threads per task : 1 (OpenMP disabled)" << std::endl;
	#endif

	// GPU configuration
	#if defined(TRITON_HIP_LAUNCHER)
		log << "GPUs per task : 1 (HIP backend)" << std::endl;
		log << "GPU backend : HIP" << std::endl;
	#elif defined(TRITON_CUDA_LAUNCHER)
		log << "GPUs per task : 1 (CUDA backend)" << std::endl;
		log << "GPU backend : CUDA" << std::endl;
	#elif defined(KOKKOS_ENABLE_CUDA)
		log << "GPUs per task : 1 (Kokkos CUDA)" << std::endl;
		log << "GPU backend : CUDA (Kokkos)" << std::endl;
	#elif defined(KOKKOS_ENABLE_HIP)
		log << "GPUs per task : 1 (Kokkos HIP)" << std::endl;
		log << "GPU backend : HIP (Kokkos)" << std::endl;
	#elif defined(KOKKOS_ENABLE_SYCL)
		log << "GPUs per task : 1 (Kokkos SYCL)" << std::endl;
		log << "GPU backend : SYCL (Kokkos)" << std::endl;
	#else
		log << "GPUs per task : 0 (CPU-only)" << std::endl;
		log << "GPU backend : none" << std::endl;
	#endif

	// Total GPUs (assuming 1 GPU per task for GPU builds)
	#if defined(TRITON_HIP_LAUNCHER) || defined(TRITON_CUDA_LAUNCHER) || defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
		log << "Total GPUs : " << size << std::endl;
	#else
		log << "Total GPUs : 0" << std::endl;
	#endif

	// Git version
	#ifdef TRITON_GIT_VERSION
		log << "TRITON_GIT_VERSION : " << TRITON_GIT_VERSION << std::endl;
	#else
		log << "TRITON_GIT_VERSION : unknown" << std::endl;
	#endif

	// Build type
	std::string build_type = "CPU";
	#if defined(TRITON_HIP_LAUNCHER)
		build_type = "GPU+HIP";
	#elif defined(TRITON_CUDA_LAUNCHER)
		build_type = "GPU+CUDA";
	#elif defined(KOKKOS_ENABLE_CUDA)
		build_type = "GPU+CUDA(Kokkos)";
	#elif defined(KOKKOS_ENABLE_HIP)
		build_type = "GPU+HIP(Kokkos)";
	#elif defined(KOKKOS_ENABLE_SYCL)
		build_type = "GPU+SYCL(Kokkos)";
	#elif defined(KOKKOS_ENABLE_OPENMP)
		build_type = "CPU+OMP";
	#endif
	log << "Build type : " << build_type << std::endl;

	log << "----------------------------" << std::endl;
	log.close();
}


/** @brief Appends the total wall time to output/log.out
*
*  This function appends the total simulation wall time to the log file.
*  Only MPI rank 0 writes to the file.
*
*  @param project_dir The project directory path
*  @param output_folder The output folder from config (may be empty for default)
*  @param total_time_sec Total wall time in seconds
*  @param rank Current MPI rank
*/
inline void triton_log_total_time(std::string project_dir, std::string output_folder, double total_time_sec, int rank)
{
	// Only rank 0 writes the log
	if (rank != 0) return;

	// Build output path using helper function
	std::string outdir = build_output_path(project_dir, output_folder);

	// Append to log file
	std::string logfile = outdir + "/log.out";
	std::ofstream log(logfile, std::ios::app);

	if (log.is_open())
	{
		log << std::fixed << std::setprecision(3);
		log << "TRITON total wall time [s] : " << total_time_sec << std::endl;
		log.close();
	}
}


}


#endif
