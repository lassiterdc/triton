/** @file output.h
 *  @brief Header containing the Output class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for Output class
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

#ifndef OUTPUT_H
#define OUTPUT_H

#include "constants.h"
#include "matrix.h"

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
*  @param rank Current sub domain id
*  @param size Number of sub domains
*  @param project_dir Main project directory
*  @param outfile_pattern Output file name pattern
*  @param time_series_flag Flag to output time series or not
*  @param cfg_content Contents of input cfg file
*  @param output_option Determines how to write data
*/		
		void init(int rows, int cols, int rank, int size, std::string project_dir, std::string outfile_pattern, int time_series_flag, std::string cfg_content, std::string output_option);
		
		
/** @brief It initializes time series outputs in a file.
*
*  @param observation_loc_size Number of cells
*  @param observation_cells All cell index
*/			
		void init_time_series(int observation_loc_size, Constants::sources_list_t observation_cells);
		
		
/** @brief It calculates which data to output in file. Also prints checkpoint id.
*
*  @param h_arr Water depth data
*  @param qx_arr Discharge in x direction data
*  @param qy_arr Discharge in y direction data
*  @param output_format Format of output files
*  @param print_option Which data to write
*  @param print_id Current checkpoint id
*  @param it_count Number of iterations so far
*  @param simtime Current time of simulation
*  @param average_dt Average time step size from the last output
*  @param max_value_h Max value of water depth data
*  @param max_value_print_option Which max value data to write
*/		
		void write_output(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, std::string output_format, std::string print_option, int print_id, int it_count, T simtime, T average_dt, Matrix::matrix<T>& max_value_h, std::string max_value_print_option);
		
		
/** @brief It outputs a specific data array's full domain in a single ascii file. 
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param simtime Current time of simulation
*/		
		void write_output_ascii_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime);
		
		
/** @brief It outputs a specific data array's sub domain in a ascii file. All subdomain outputs seperately in different file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param simtime Current time of simulation
*/			
		void write_output_ascii_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime);
		
		
/** @brief It outputs a specific data array's full domain in a single binary file. 
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param simtime Current time of simulation
*/		
		void write_output_binary_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime);
		
		
/** @brief It outputs a specific data array's sub domain in a binary file. All subdomain outputs seperately in different file.
*
*  @param arr Subdomain data
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param simtime Current time of simulation
*/			
		void write_output_binary_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime);
		
		
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
*  @param what_mat Data type
*  @param print_id Current checkpoint id
*  @param simtime Current time of simulation
*/		
		void output_time_series(std::string what_mat, int print_id, T simtime);


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
		double average(double a[], int n);

/** @brief It writes the evolution of subdomain dimensions if dynamic load balancing is enabled.
*
*  @param pd Partition data
*  @param print_id Print output id
*/		
	void write_domain_decomposition(MpiUtils::partition_data_t pd, int print_id);
	




	private:
		int rows_;	/**< Number of rows in a subdomain */
		int cols_;	/**< Number of columns in a sub domain */
		int rank_;	/**< Current subdomain id */
		int size_;	/**< Total number of sub domain */
		int time_series_flag_;	/**< Flag to determine output time series or not. If true, output the time series. */
		std::string project_dir_;	/**< Main project directory */
		std::string outfile_pattern_;	/**< Output file directory and name pattern */
		std::string cfg_content_;	/**< Contents of the input cfg (Configuration) file */
		std::string output_option_;	/**< Strategy to use for outputting into files. PAR for parallel outputs or SEQ for sequential outputs. PAR saves each MPI partitions subdomain in separate files and SEQ saves the whole domain into one file. */

		int observation_loc_size_;	/**< Number of observation cells */
		Constants::sources_list_t observation_cells_;	/**< Index position of observation cells in main domain */

	
	public:
		int cur_proc_data_size = 0;	/**< Number of cells in current subdomain */
		int *recvcounts = NULL;	/**< Array to hold every subdomains cell count */
		int total_data_size = 0;	/**< Number of cells in main domain */
		int *displs = NULL;	/**< Position array to hold each sub domains starting point in main domain */
		T *total_data_arr = NULL;	/**< Main domains data or collection data of every subdomain */
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
	}


	template<typename T>
	void output<T>::init(int rows, int cols, int rank, int size, std::string project_dir, std::string outfile_pattern, int time_series_flag, std::string cfg_content, std::string output_option)
	{

		cur_proc_data_size = 0;
		if (recvcounts != NULL)
		delete[] recvcounts;
		total_data_size = 0;
		if (displs != NULL)
		delete[] displs;
		if (total_data_arr != NULL)
		delete[] total_data_arr;

		rows_ = rows;
		cols_ = cols;
		rank_ = rank;
		size_ = size;
		project_dir_ = project_dir;
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
		MPI_Gather(&cur_proc_data_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (rank_ == 0)
		{
			displs = new int[size];
			displs[0] = 0;
			total_data_size += recvcounts[0];

			for (int i = 1; i < size_; i++)
			{
				total_data_size += recvcounts[i];
				displs[i] = displs[i - 1] + recvcounts[i - 1];
			}
			
			total_data_arr = new T[total_data_size];
		}
	}


	template<typename T>
	void output<T>::init_time_series(int observation_loc_size, Constants::sources_list_t observation_cells)
	{
		observation_loc_size_ = observation_loc_size;
		observation_cells_ = observation_cells;
	}


	template<typename T>
	void output<T>::write_output(Matrix::matrix<T>& h_arr, Matrix::matrix<T>& qx_arr, Matrix::matrix<T>& qy_arr, std::string output_format, std::string print_option, int print_id, int it_count, T simtime, T average_dt, Matrix::matrix<T>& max_value_h, std::string max_value_print_option)
	{
		if (strcmp(output_format.c_str(), "ASC") == 0)
		{
			if (print_option.find("h") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(h_arr, "H", print_id, simtime);
					write_output_ascii_sequential(h_arr, "H", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(h_arr, "H", print_id, simtime);
					write_output_ascii_parallel(h_arr, "H", print_id, simtime);
				}
			}

			if (print_option.find("u") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qx_arr, "QX", print_id, simtime);
					write_output_ascii_sequential(qx_arr, "QX", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(qx_arr, "QX", print_id, simtime);
					write_output_ascii_parallel(qx_arr, "QX", print_id, simtime);
				}
			}

			if (print_option.find("v") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qy_arr, "QY", print_id, simtime);
					write_output_ascii_sequential(qy_arr, "QY", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(qy_arr, "QY", print_id, simtime);
					write_output_ascii_parallel(qy_arr, "QY", print_id, simtime);
				}
			}
			
			
			if (max_value_print_option.size() > 0)
			{
				if (max_value_print_option.find("h") != std::string::npos)
				{
					if(strcmp(output_option_.c_str(), "SEQ") == 0)
					{
						write_output_binary_sequential(max_value_h, "MH", print_id, simtime);
						write_output_ascii_sequential(max_value_h, "MH", print_id, simtime);
					}
					else
					{
						write_output_binary_parallel(max_value_h, "MH", print_id, simtime);
						write_output_ascii_parallel(max_value_h, "MH", print_id, simtime);
					}
				}
			}
		}
		else
		{
			if (print_option.find("h") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(h_arr, "H", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(h_arr, "H", print_id, simtime);
				}
			}

			if (print_option.find("u") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qx_arr, "QX", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(qx_arr, "QX", print_id, simtime);
				}
			}

			if (print_option.find("v") != std::string::npos)
			{
				if(strcmp(output_option_.c_str(), "SEQ") == 0)
				{
					write_output_binary_sequential(qy_arr, "QY", print_id, simtime);
				}
				else
				{
					write_output_binary_parallel(qy_arr, "QY", print_id, simtime);
				}
			}
			
			if (max_value_print_option.size() > 0)
			{
				if (max_value_print_option.find("h") != std::string::npos)
				{
					if(strcmp(output_option_.c_str(), "SEQ") == 0)
					{
						write_output_binary_sequential(max_value_h, "MH", print_id, simtime);
					}
					else
					{
						write_output_binary_parallel(max_value_h, "MH", print_id, simtime);
					}
				}
			}
		}


		if (rank_ == 0)
		{

			std::cerr << BLUE << "[" << (print_id) << "]" << RESET " Time: " << simtime << "\tdt: " << average_dt << "\tit: " << it_count << std::endl;
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
	void output<T>::write_output_ascii_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime)
	{
		if(rank_ == 0)
		{
			std::string root_dir(project_dir_ + "/" + OUTPUT_DIR + "/");

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
			std::string file_dir(project_dir_ + "/" + OUTPUT_DIR + "/" + ASCII_DIR + "/");

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
					mat << total_data_arr[i*total_cols+j];

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
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}


	template<typename T>
	void output<T>::write_output_ascii_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime)
	{
		std::string root_dir(project_dir_ + "/" + OUTPUT_DIR + "/");

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
		std::string file_dir(project_dir_ + "/" + OUTPUT_DIR + "/" + ASCII_DIR + "/");

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
	void output<T>::write_output_binary_sequential(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime)
	{
		if (rank_ == 0)
		{
			MPI_Gatherv(arr.get_address_at(0, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Gatherv(arr.get_address_at(GHOST_CELL_PADDING, 0), cur_proc_data_size, MPI_DATA_TYPE, total_data_arr, recvcounts, displs, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		}

		if (rank_ == 0)
		{
			std::string root_dir(project_dir_ + "/" + OUTPUT_DIR + "/");

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
			std::string file_dir(project_dir_ + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/");

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
				mat.write((char*) &total_data_arr[i*total_cols+off], (total_cols-2*off) * sizeof(T));
			}
			mat.close();

			if (time_series_flag_)
			{
				output_time_series(what_mat, print_id, simtime);
			}
		}
		if (size_ > 1)
		{
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}


	template<typename T>
	void output<T>::write_output_binary_parallel(Matrix::matrix<T>& arr, std::string what_mat, int print_id, T simtime)
	{
		std::string root_dir(project_dir_ + "/" + OUTPUT_DIR + "/");

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
		std::string file_dir(project_dir_ + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/");

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
	void output<T>::output_time_series(std::string what_mat, int print_id, T simtime)
	{
		std::string outdir = project_dir_ + "/" + OUTPUT_DIR + "/" + TIME_SERIES_DIR + "/";
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

		std::string filedir = outdir + what_mat + "_at_Xsec.txt";
		if (print_id <= 1)
		{
			std::string str = "Time(s)";
			for (int i = 0; i < observation_loc_size_; i++)
			{
				str = str + "," + what_mat + "_at_Point_" + std::to_string(i + 1);
			}
			str = str + "\n";
			std::ofstream output(filedir);
			output << str;
			output.close();
		}
		std::string str = std::to_string(simtime);
		std::ofstream out(filedir, std::ios::app);
		for (int i = 0; i < observation_loc_size_; i++)
		{
			std::pair<int, int> pair = observation_cells_[i];
			T value = total_data_arr[pair.first * cols_ + pair.second];
			str = str + "," + std::to_string(value);
		}
		str = str + "\n";
		out << str;
		out.close();
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
		std::string outdir = project_dir_ + "/" + OUTPUT_DIR + "/" + CFG_DIR + "/";
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
	void output<T>::write_times(SuperTimer::super_timer st, int print_id)
	{

		double compute_time = st.get_custom_time(COMPUTE_TIME);
		double mpi_time = st.get_custom_time(MPI_TIME);
		double io_time = st.get_custom_time(IO_TIME);
		double resize_time = st.get_custom_time(RESIZE_TIME);
		double simulation_time = st.get_custom_time(SIMULATION_TIME);
		double total_time = st.get_custom_time(TOTAL_TIME);
		double other_time = simulation_time - compute_time - mpi_time - io_time - resize_time;
		double init_time = total_time - simulation_time;

		double *compute_time_all = new double[size_];
		double *mpi_time_all = new double[size_];
		double *io_time_all = new double[size_];
		double *simulation_time_all = new double[size_];
		double *total_time_all = new double[size_];
		double *other_time_all = new double[size_];
		double *init_time_all = new double[size_];
		double *resize_time_all = new double[size_];


		MPI_Gather(&compute_time, 1, MPI_DATA_TYPE, &compute_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&mpi_time, 1, MPI_DATA_TYPE, &mpi_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&io_time, 1, MPI_DATA_TYPE, &io_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&simulation_time, 1, MPI_DATA_TYPE, &simulation_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&total_time, 1, MPI_DATA_TYPE, &total_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&other_time, 1, MPI_DATA_TYPE, &other_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&init_time, 1, MPI_DATA_TYPE, &init_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		MPI_Gather(&resize_time, 1, MPI_DATA_TYPE, &resize_time_all[rank_], 1, MPI_DATA_TYPE, 0, MPI_COMM_WORLD);
		
		if (size_ > 1)
		{
			MPI_Barrier(MPI_COMM_WORLD);
		}

		if(rank_ == 0)
		{
			std::string outdir;
			std::string filedir;

			//not the final state
			if(print_id!=-1){
				std::string outdir = project_dir_ + "/" + OUTPUT_DIR + "/performance/";
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
				std::string outdir = project_dir_ + "/" + OUTPUT_DIR + "/";
				filedir = outdir + "performance.txt";	
			}
			std::ofstream output(filedir);
			output << "%Rank, Compute, MPI, IO, Resize, Other, Simulation, Init, Total" << std::endl;
			
			for(int j=0;j<size_;j++){
				output << std::setprecision(4) << j << ", " << compute_time_all[j] << ", " <<  mpi_time_all[j] << ", " <<	io_time_all[j] << ", " <<	resize_time_all[j] << ", " << other_time_all[j] << ", " 
				<< simulation_time_all[j] << ", " << init_time_all[j] <<  ", " << total_time_all[j] << std::endl;
			}
			output << std::setprecision(4) << "Average" << ", " << average(compute_time_all,size_) << ", " <<  average(mpi_time_all,size_) << ", " <<	 average(io_time_all,size_) << ", " << average(resize_time_all,size_) << ", " <<  average(other_time_all,size_) << ", " <<  average(simulation_time_all,size_) << ", " <<  average(init_time_all,size_) <<  ", " << average(total_time_all,size_) << std::endl;

			output.close();
		}
		
	}
	
	
	template<typename T>
	double output<T>::average(double a[], int n) 
	{ 
		double sum = 0; 
		for (int i=0; i<n; i++) 
		sum+= a[i]; 

		return sum/n; 
	}	 
	template<typename T>
	void output<T>::write_domain_decomposition(MpiUtils::partition_data_t pd, int print_id)
	{
		if(rank_ == 0)
		{

			std::string outdir = project_dir_ + "/" + OUTPUT_DIR + "/domain_decomposition/";
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
					MPI_Barrier(MPI_COMM_WORLD);
		 }



	}
}

#endif
