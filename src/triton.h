/** @file Triton.h
 *  @brief Header containing the Triton class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for Triton class
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



#ifndef TRITON_H
#define TRITON_H
#include "kernels.h"
#include "output.h"
#include "mpi_utils.h"

namespace Triton
{
	template<class T>
	class triton
	{
	public:
		triton<T>(int argc, char* argv[]);
		~triton<T>();

		void initialize(int rank_, int size_);
		void simulate();

	private:
		int rank;
		int size;
		int rows;
		int cols;
		int org_rows;
		int org_cols;
		int num_of_src;
		int num_of_extbc;
		int num_extbc_cells;
		int index_row_runoff;
		int idx_low = 0;
		int checkpoint_id;
		
		int host_src_pos_arr_size;
		int host_hyg_time_arr_size;
		int host_hyg_val_arr_size;
		int host_bc_cells_size;
		int host_bc_vars_arr_size;
		int host_runoff_intensity_arr_size;
		int host_reduce_dt_arr_sz;
		int host_halo_h_arr_size;
		int host_halo_qxqy_arr_size;
		int nbytes;
		int nbytes_halo_h;
		int nbytes_halo_qxqy;
		
		T simtime;
		T cell_size;
		T local_dt;
		T global_dt;
		
		std::string project_dir;
		std::string cfg_content;
		std::string cfg_dir;

		SuperTimer::super_timer st;
		ConfigUtils::arguments<T> arglist;
		Hydrograph::hydrograph<T> hyg, roff;
		Constants::sources_list_t observation_cells;
		MpiUtils::partition_data_t pd;
		DemFile::dem_file<T> dem, sub_dem;
		Matrix::matrix<T> hin, uin, vin, nin, hot_hin, hot_qxin, hot_qyin;
		Matrix::matrix<T> sub_hin, sub_qxin, sub_qyin, sub_nin, sub_hot_hin, sub_hot_qxin, sub_hot_qyin;
		Matrix::matrix<int> rin, sub_rin;
		
		int* host_src_pos_arr;
		int* host_relative_bc_index;
		int* host_bc_type;
		int* host_bc_start_index;
		int* host_bc_nrows_vars;
		int* host_runoff_id_arr;
		
		T* host_hyg_time_arr;
		T* host_hyg_val_arr;
		T* host_extbc_var1_arr;
		T* host_extbc_var2_arr;
		T* host_runoff_intensity_arr;
		T* host_halo_h_arr;
		T* host_halo_qxqy_arr;
		T* host_sqrth_arr;
		T* host_dt_values_arr;
		T* host_rhsh0;
		T* host_rhsh1;
		T* host_rhsqx0;
		T* host_rhsqx1;
		T* host_rhsqy0;
		T* host_rhsqy1;

		std::vector<T*> host_vec;
		std::vector<int*> host_vec_int;

#ifdef ACTIVE_GPU
		cudaStream_t streams;
		std::vector<T*> device_vec;
		std::vector<int*> device_vec_int;
#endif

		void compute_local_dt();
		void compute_global_dt(int print_id);
		void compute_new_state();
		int calc_src_col(T src_x, T xllc, T cell_size_);
		int calc_src_row(T src_y, T yllc, T cell_size_, int nrows);
		void read_configuration(std::string cfg_dir, int checkpoint_id);
		void read_inflows();
		void read_matrix_files();
		void process_source_locations();
		void process_observation_cells();
		void process_boundary_condition();
		void partition_matrix_files();
		void process_runoff();
		void create_host_aux_vectors();
		void create_host_vectors();
		void create_device_vectors();
	};

	template<class T>
	triton<T>::triton(int argc, char* argv[])
	{
		st = SuperTimer::super_timer();
		st.start(TOTAL_TIME);
		project_dir = ConfigUtils::get_root_dir(argv[0]);
		if(argc > 1)
		{
			cfg_dir = std::string(argv[1]);
		}
	
		#ifdef ACTIVE_OMP
		int threads = 1;
		if(argc > 2)
		{
			threads = atoi(argv[2]);
		}
		omp_set_num_threads(threads);
		#endif
	
		checkpoint_id = 0;
		if(argc > 3)
		{
			checkpoint_id = atoi(argv[3]);
		}

	}

	template<typename T>
	void triton<T>::initialize(int rank_, int size_)
	{
		rank = rank_;
		size = size_;

		read_configuration(cfg_dir, checkpoint_id);
		simtime = arglist.sim_start_time;
		
		read_inflows();
		
		read_matrix_files();
		
		pd = MpiUtils::partition_data_t(size, org_rows, org_cols);
		process_source_locations();
		process_observation_cells();
		
		process_boundary_condition();
		partition_matrix_files();
		
		process_runoff();
		create_host_aux_vectors();
		
		create_host_vectors();
		create_device_vectors();
	}
	
	template<typename T>
	void triton<T>::read_configuration(std::string cfg_dir, int checkpoint_id)
	{
		if (rank == 0){
			std::cerr << IN "Reading configuration file" << std::endl;
		}

		std::string cfg_path = project_dir + "/" + INPUT_DIR + "/" + CFG_DIR + "/" + DEFAULT_CFG;
		
		if(!(cfg_dir.empty()) && !(StringUtils::is_numeric(cfg_dir)))
		{
			cfg_path = cfg_dir;
		}

		if (checkpoint_id > 0)
		{
			cfg_path = project_dir + "/" + OUTPUT_DIR + "/" + CFG_DIR + "/config_" + to_string(checkpoint_id) + ".cfg";
		}

		cfg_content = ConfigUtils::file_content_to_string(cfg_path);
		arglist = ConfigUtils::get_args<T>(cfg_content);

		if (rank == 0){
			std::cerr << OK "Configuration file read" << std::endl;
			if (arglist.num_sources == 0)
			{
				std::cerr << DASH "No sources defined" << std::endl;
			}else{
				std::cerr << DASH << arglist.num_sources << " sources defined" << std::endl;
			}

			if (arglist.num_runoffs == 0)
			{
				std::cerr << DASH "No runoff defined" << std::endl;
			}else{
				std::cerr << DASH << arglist.num_runoffs << " runoffs defined" << std::endl;
			}

			if (arglist.num_extbc == 0)
			{
				std::cerr << DASH "No external boundary conditions defined" << std::endl;
			}else{
				std::cerr << DASH << arglist.num_extbc << " external boundary conditions defined" << std::endl;
			}

			if (arglist.time_series_flag == 0)
			{
				std::cerr << DASH "No observation points defined" << std::endl;
			}else{
				std::cerr << DASH << arglist.observation_x_loc.size() << " observation points defined" << std::endl;
			}
		}



	}
	
	template<typename T>
	void triton<T>::read_inflows()
	{
		if(arglist.num_sources > 0)
		{		
			if (rank == 0){
				std::cerr << IN "Reading and processing source hydrographs" << std::endl;
			}
			hyg = Hydrograph::hydrograph<T>(arglist.hydrograph_filename);
			hyg.convert_time_hr_to_secs();
			if (rank == 0){
				std::cerr << OK "Sources set" << std::endl;
			}
		}
		
		if (arglist.num_runoffs > 0)
		{
			if (rank == 0){
				std::cerr << IN "Reading and processing runoff rates" << std::endl;
			}
			roff = Hydrograph::hydrograph<T>(arglist.runoff_filename);
			roff.convert_time_hr_to_secs();
			roff.convert_rate_hr_to_secs();
			roff.convert_rate_mm_to_m();
			if (rank == 0){
				std::cerr << OK "Runoff rates set" << std::endl;
			}
		}
	}
	
	template<typename T>
	void triton<T>::read_matrix_files()
	{
		if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
		{
			dem.load_header_from_dem_file_ascii(arglist.dem_filename);
		}
		else
		{
			dem.load_header_from_dem_file_binary(arglist.dem_filename);
		}
		
		org_rows = dem.get_nrows();
		org_cols = dem.get_ncols();
		cell_size = dem.get_cell_size();
		
		if (rank == 0)
		{
			if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
			{
				dem.load_from_ascii_file(org_rows, org_cols, arglist.dem_filename, DEM_HEADER_SIZE);
			}
			else
			{
				dem.load_from_binary_file(org_rows, org_cols, arglist.dem_filename, DEM_HEADER_SIZE);
			}
			dem.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
			dem.set_nrows(dem.get_num_rows());
			dem.set_ncols(dem.get_num_cols());

			dem.set_infinite_walls();

			if(!arglist.n_infile.empty())
			{
				if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
				{
					nin.load_from_ascii_file(org_rows, org_cols, arglist.n_infile);
				}
				else
				{
					nin.load_from_binary_file(org_rows, org_cols, arglist.n_infile);
				}
			}
			else
			{
				nin.resize(org_rows, org_cols);
				nin.zero_fill();
				nin += arglist.const_mann;
			}
			nin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
			nin.copy_value_into_ghost_cells();
			nin.square();
			
			if (arglist.h_infile.size() > 0)
			{
				if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
				{
					hin.load_from_ascii_file(org_rows, org_cols, arglist.h_infile);
				}
				else
				{
					hin.load_from_binary_file(org_rows, org_cols, arglist.h_infile);
				}
				hin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
			}
			if (arglist.qx_infile.size() > 0)
			{
				if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
				{
					uin.load_from_ascii_file(org_rows, org_cols, arglist.qx_infile);
				}
				else
				{
					uin.load_from_binary_file(org_rows, org_cols, arglist.qx_infile);
				}
				uin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
			}
			if (arglist.qy_infile.size() > 0)
			{
				if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
				{
					vin.load_from_ascii_file(org_rows, org_cols, arglist.qy_infile);
				}
				else
				{
					vin.load_from_binary_file(org_rows, org_cols, arglist.qy_infile);
				}
				vin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
			}

			if (arglist.runoff_map.size() > 0)
			{
				if (strcmp(arglist.input_format.c_str(), "ASC") == 0)
				{
					rin.load_from_ascii_file(org_rows, org_cols, arglist.runoff_map);
				}
				else
				{
					rin.load_from_binary_file(org_rows, org_cols, arglist.runoff_map);
				}
				rin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0);
			}
			
			if (arglist.checkpoint_id > 0 && strcmp(arglist.output_option.c_str(), "SEQ") == 0)
			{
				if (rank == 0){
					std::cerr << IN "Reading checkpoint files" << std::endl;
				}
				string temp_num(to_string(arglist.checkpoint_id));
				if (arglist.checkpoint_id < 10)
				{
					temp_num = "0" + temp_num;
				}

				string filedirH(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/H_" + temp_num + "_00.out");
				hot_hin.load_from_binary_file(org_rows, org_cols, filedirH);
				hot_hin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);

				string filedirQX(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/QX_" + temp_num + "_00.out");
				hot_qxin.load_from_binary_file(org_rows, org_cols, filedirQX);
				hot_qxin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);

				string filedirQY(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/QY_" + temp_num + "_00.out");
				hot_qyin.load_from_binary_file(org_rows, org_cols, filedirQY);
				hot_qyin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
				if (rank == 0){
					std::cerr << OK "Checkpoint files read" << std::endl;
				}

			}
		}
	}
	
	template<typename T>
	void triton<T>::process_source_locations()
	{
		num_of_src = 0;
		std::vector<int> src_rows, src_cols;
		Constants::sources_list_t source_cells;
		std::vector<int> relative_src_index;
		
		if(arglist.num_sources > 0)
		{
			std::vector<T> src_x = arglist.src_x_loc;
			std::vector<T> src_y = arglist.src_y_loc;

			int num_sources = src_x.size();
			src_rows.assign(num_sources, 0);
			src_cols.assign(num_sources, 0);

			for (int i = 0; i < num_sources; ++i)
			{
				src_cols[i] = calc_src_col(src_x[i], dem.get_xll_corner(), dem.get_cell_size());
				src_rows[i] = calc_src_row(src_y[i], dem.get_yll_corner(), dem.get_cell_size(), org_rows);
			}

			for (int i = 0; i < arglist.num_sources; ++i)
			{
				int srank = 0;
				int prev_rows_sum = 0;

				if(size > 1){
					int source_row = src_rows[i];
					int rows_sum = pd.part_dims[0].first - 2 * GHOST_CELL_PADDING;
				
					if(source_row >= rows_sum){
						for(int j=1; j<size; j++){
							prev_rows_sum = rows_sum;
							rows_sum += pd.part_dims[j].first - 2 * GHOST_CELL_PADDING;
							if(source_row < rows_sum){
								srank = j;
								break;
							}
						}
					}
				}
				src_rows[i] = src_rows[i] - prev_rows_sum + GHOST_CELL_PADDING;
				src_cols[i] = src_cols[i] + GHOST_CELL_PADDING;
				
				if (rank == srank)
				{
					relative_src_index.push_back(i);
					
					std::pair<int, int> scell(src_rows[i], src_cols[i]);
					source_cells.push_back(scell);

					num_of_src++;
				}
			}
		}
		
		host_src_pos_arr_size = num_of_src;
		if (host_src_pos_arr_size == 0)
		{
			host_src_pos_arr_size = 1;
		}
		host_src_pos_arr = new int[host_src_pos_arr_size];
		for (int i = 0; i < host_src_pos_arr_size; i++)
		{
			host_src_pos_arr[i] = 0;
		}

		for (int i = 0; i<num_of_src; i++)
		{
			std::pair<int, int> pair = source_cells[i];
			host_src_pos_arr[i] = pair.first*(org_cols+2*GHOST_CELL_PADDING) + pair.second;
		}

		host_hyg_time_arr_size = 1;
		if(num_of_src > 0)
		{
			host_hyg_time_arr_size = hyg.get_num_inflow_rows();
		}
		
		host_hyg_time_arr = new T[host_hyg_time_arr_size];
		
		host_hyg_val_arr_size = host_hyg_time_arr_size*host_src_pos_arr_size;
		host_hyg_val_arr = new T[host_hyg_val_arr_size];
		if(num_of_src > 0)
		{
			for (int i = 0; i < host_hyg_time_arr_size; i++)
			{
				host_hyg_time_arr[i] = hyg.get_time_at(i);
				
				for (int j = 0; j < num_of_src; j++)
				{
					int src_number = relative_src_index[j] + 1;
					host_hyg_val_arr[i*num_of_src + j] = hyg.get_flow_at(i, src_number);
				}
			}
		}
		else
		{
			host_hyg_time_arr[0] = 0.0;
			host_hyg_val_arr[0] = 0.0;
		}
	}
	
	template<typename T>
	void triton<T>::process_observation_cells()
	{
		if (arglist.time_series_flag)
		{
			std::vector<T> observation_x = arglist.observation_x_loc;
			std::vector<T> observation_y = arglist.observation_y_loc;

			int num_observation_loc = observation_x.size();
			std::vector<int> observation_rows, observation_cols;

			observation_rows.assign(num_observation_loc, 0);
			observation_cols.assign(num_observation_loc, 0);

			for (int i = 0; i < num_observation_loc; ++i)
			{
				observation_cols[i] = calc_src_col(observation_x[i], dem.get_xll_corner(), dem.get_cell_size()) + GHOST_CELL_PADDING;
				observation_rows[i] = calc_src_row(observation_y[i], dem.get_yll_corner(), dem.get_cell_size(), org_rows) + GHOST_CELL_PADDING;

				std::pair<int, int> scell(observation_rows[i], observation_cols[i]);
				observation_cells.push_back(scell);
			}
		}
	}
	
	template<typename T>
	void triton<T>::process_boundary_condition()
	{
		int num_extbc = 1;
		if(arglist.num_extbc > 0) num_extbc = arglist.num_extbc;
		ExtBC::extBC<T> extbc[num_extbc];
		vector<int> *relative_bc_index = new vector<int> [size];
		
		if (arglist.num_extbc > 0)
		{
			for(int i=0; i<arglist.num_extbc; i++){
				extbc[i] = ExtBC::extBC<T>(arglist.extbc_fname[i], arglist.extbc_bctype[i]);
				if(arglist.extbc_bctype[i]==1){
					extbc[i].convert_to_secs();
				}
			}
			
			for(int i=0; i<arglist.num_extbc; i++){
				extbc[i].extreme_cols.assign(2, 0); //there are two points per extbc
				extbc[i].extreme_rows.assign(2, 0); //there are two points per extbc
			
				T extreme_x1 = arglist.extbc_x1_loc[i];
				T extreme_y1 = arglist.extbc_y1_loc[i];
				T extreme_x2 = arglist.extbc_x2_loc[i];
				T extreme_y2 = arglist.extbc_y2_loc[i];

				extbc[i].extreme_cols[0] = calc_src_col(extreme_x1, dem.get_xll_corner(), dem.get_cell_size());
				extbc[i].extreme_rows[0] = calc_src_row(extreme_y1, dem.get_yll_corner(), dem.get_cell_size(), org_rows);
				extbc[i].extreme_cols[1] = calc_src_col(extreme_x2, dem.get_xll_corner(), dem.get_cell_size());
				extbc[i].extreme_rows[1] = calc_src_row(extreme_y2, dem.get_yll_corner(), dem.get_cell_size(), org_rows);	

				extbc[i].ncells=extbc[i].check_extreme_extbc(extbc[i].extreme_cols,extbc[i].extreme_rows,org_cols,org_rows);
				extbc[i].ncells_local=0;
			}
			
			for(int i=0; i<arglist.num_extbc; i++){
				extbc[i].create_involved_cells(extbc[i].extreme_cols,extbc[i].extreme_rows,org_cols,org_rows,arglist.extbc_bctype[i]);
				if(rank==0){
					dem.copy_elevation_into_ghost_cells(extbc[i].i_rows,extbc[i].i_cols,extbc[i].ncells, extbc[i].location);
				}
			}
			
			for(int i=0; i<arglist.num_extbc; i++){
				for(int j=0; j<extbc[i].ncells; j++){
					int srank = 0;
					int prev_rows_sum = 0;
					int cell_row = extbc[i].i_rows[j];
					if(size > 1){
						int rows_sum = pd.part_dims[0].first - 2 * GHOST_CELL_PADDING;
						if(cell_row >= rows_sum){
							for(int k=1; k<size; k++){
								prev_rows_sum = rows_sum;
								rows_sum += pd.part_dims[k].first - 2 * GHOST_CELL_PADDING;
								if(cell_row < rows_sum){
									srank = k;
									break;
								}
							}
						}
					}
					cell_row = cell_row - prev_rows_sum + GHOST_CELL_PADDING;
					int new_index = cell_row * (org_cols+2*GHOST_CELL_PADDING) + extbc[i].i_cols[j]+GHOST_CELL_PADDING;
					relative_bc_index[srank].push_back(new_index);
					if (rank == srank){
						extbc[i].ncells_local++;
					}

				}
			}
		}
		
		num_of_extbc = arglist.num_extbc;
		num_extbc_cells = relative_bc_index[rank].size();
		
		host_bc_cells_size = 0;
		if (num_of_extbc > 0)
		{
			for(int i=0; i<num_of_extbc; i++){
				host_bc_cells_size += extbc[i].ncells_local;
			}
		}else{
			host_bc_cells_size = 1;
		}

		host_relative_bc_index = new int[max(host_bc_cells_size,1)];
		host_bc_type = new int[max(host_bc_cells_size,1)];
		host_bc_start_index = new int[max(host_bc_cells_size,1)];
		host_bc_nrows_vars = new int[max(host_bc_cells_size,1)];

		if (num_of_extbc > 0)
		{
			int moving_index = 0;
			int start_index = 0;
			for(int i=0; i<num_of_extbc; i++){
				int extbc_num_rows=extbc[i].get_num_rows();
				for(int j=0; j<extbc[i].ncells_local; j++){
					host_relative_bc_index[moving_index+j] = relative_bc_index[rank][moving_index+j];
					host_bc_type[moving_index+j] = arglist.extbc_bctype[i];
					host_bc_start_index[moving_index+j] = start_index;
					host_bc_nrows_vars[moving_index+j] = extbc_num_rows;
				}
				start_index +=extbc_num_rows;
				moving_index += extbc[i].ncells_local;
			}
		}
				
		host_bc_vars_arr_size = 0;
		if (num_of_extbc > 0)
		{
			for(int i=0; i<num_of_extbc; i++){
				host_bc_vars_arr_size += extbc[i].get_num_rows();
			}
		}else{
			host_bc_vars_arr_size = 1;
		}


		host_extbc_var1_arr = new T[host_bc_vars_arr_size];
		host_extbc_var2_arr = new T[host_bc_vars_arr_size];

		if (num_of_extbc > 0)
		{
			int moving_index2 = 0;
			for(int i=0; i<num_of_extbc; i++){
				int extbc_num_rows=extbc[i].get_num_rows();
				for (int j = 0; j < extbc_num_rows; j++)
				{
					host_extbc_var1_arr[moving_index2+j] = extbc[i].get_var1_at(j);
					host_extbc_var2_arr[moving_index2+j] = extbc[i].get_var2_at(j);
				}
				moving_index2 += extbc_num_rows;
			}

		}else{
			host_extbc_var1_arr[0]=0.0;
			host_extbc_var2_arr[0]=0.0;
		}
		
		delete[] relative_bc_index;
	}

	template<typename T>
	void triton<T>::partition_matrix_files()
	{
		if(size > 1 && rank == 0){
			std::cerr << IN "Creating partition data" << std::endl;
		}

		if(size > 1)
		{
			sub_dem = MpiUtils::scatter_exchange(dem.get_data(), pd, rank);
			sub_nin = MpiUtils::scatter_exchange(nin.get_data(), pd, rank);
		}
		else
		{
			sub_dem = dem;
			sub_nin = nin;
		}
		
		rows = sub_dem.get_num_rows();
		cols = sub_dem.get_num_cols();
		
		sub_dem.set_nrows(sub_dem.get_num_rows());
		sub_dem.set_ncols(sub_dem.get_num_cols());
		sub_dem.set_cell_size(dem.get_cell_size());
		sub_dem.set_xll_corner(dem.get_xll_corner());
		sub_dem.set_yll_corner(dem.get_yll_corner());
		sub_dem.set_no_data_value(dem.get_no_data_value());
		
		if(arglist.runoff_map.size() > 0)
		{
			if(size > 1)
			{
				sub_rin = MpiUtils::scatter_exchange_int(rin.get_data(), pd, rank);
			}
			else
			{
				sub_rin = rin;
			}
		}
		
		if(arglist.h_infile.size() > 0)
		{
			if(size > 1)
			{
				sub_hin = MpiUtils::scatter_exchange(hin.get_data(), pd, rank);
			}
			else
			{
				sub_hin = hin;
			}
		}
		
		if(arglist.qx_infile.size() > 0)
		{
			if(size > 1)
			{
				sub_qxin = MpiUtils::scatter_exchange(uin.get_data(), pd, rank);
			}
			else
			{
				sub_qxin = uin;
			}
		}
		
		if(arglist.qy_infile.size() > 0)
		{
			if(size > 1)
			{
				sub_qyin = MpiUtils::scatter_exchange(vin.get_data(), pd, rank);
			}
			else
			{
				sub_qyin = vin;
			}
		}
		
		if (arglist.checkpoint_id > 0)
		{
			if(strcmp(arglist.output_option.c_str(), "SEQ") == 0)
			{
				if(size > 1)
				{
					sub_hot_hin = MpiUtils::scatter_exchange(hot_hin.get_data(), pd, rank);
					sub_hot_qxin = MpiUtils::scatter_exchange(hot_qxin.get_data(), pd, rank);
					sub_hot_qyin = MpiUtils::scatter_exchange(hot_qyin.get_data(), pd, rank);
				}
				else
				{
					sub_hot_hin = hot_hin;
					sub_hot_qxin = hot_qxin;
					sub_hot_qyin = hot_qyin;
				}
			}
			else
			{
				string temp_num(to_string(arglist.checkpoint_id));
				if (arglist.checkpoint_id < 10)
				{
					temp_num = "0" + temp_num;
				}
				
				string temp_num_2(to_string(rank));
				if (rank < 10)
				{
					temp_num_2 = "0" + temp_num_2;
				}

				int host_dem_original_row = rows - 2 * GHOST_CELL_PADDING;
				int host_dem_original_col = cols - 2 * GHOST_CELL_PADDING;

				string filedirH(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/H_" + temp_num + "_" + temp_num_2 + ".out");
				sub_hot_hin.load_from_binary_file(host_dem_original_row, host_dem_original_col, filedirH);
				sub_hot_hin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);

				string filedirU(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/QX_" + temp_num + "_" + temp_num_2 + ".out");
				sub_hot_qxin.load_from_binary_file(host_dem_original_row, host_dem_original_col, filedirU);
				sub_hot_qxin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);

				string filedirV(project_dir + "/" + OUTPUT_DIR + "/" + BIN_DIR + "/QY_" + temp_num + "_" + temp_num_2 + ".out");
				sub_hot_qyin.load_from_binary_file(host_dem_original_row, host_dem_original_col, filedirV);
				sub_hot_qyin.add_ghost_cells(GHOST_CELL_PADDING, GHOST_CELL_PADDING, 0.0);
				
				MpiUtils::exchange(sub_hot_hin.begin(), rows, cols, rank, size, USE_MATRIX);
				MPI_Barrier(MPI_COMM_WORLD);
				MpiUtils::exchange(sub_hot_qxin.begin(), rows, cols, rank, size, USE_MATRIX);
				MPI_Barrier(MPI_COMM_WORLD);
				MpiUtils::exchange(sub_hot_qyin.begin(), rows, cols, rank, size, USE_MATRIX);
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}
		
		if (arglist.checkpoint_id > 0)
		{
			sub_hin = sub_hot_hin;
			sub_qxin = sub_hot_qxin;
			sub_qyin = sub_hot_qyin;
		}
		else
		{
			Matrix::matrix<T> values(rows, cols);
			values.zero_fill();

			if (arglist.h_infile.size() <= 0)
			{
				sub_hin = values;
			}
			if (arglist.qx_infile.size() <= 0)
			{
				sub_qxin = values;
			}
			if (arglist.qy_infile.size() <= 0)
			{
				sub_qyin = values;
			}
		}
		if(size > 1 && rank == 0){
			std::cerr << OK "Data has been partitioned" << std::endl;
		}

	}

	template<typename T>
	void triton<T>::process_runoff()
	{
		if (arglist.num_runoffs > 0)
		{
			index_row_runoff = 0;
			for (int j = 0; j < roff.get_num_inflow_rows() - 1; j++)
			{
				if (simtime >= roff.get_time_at(j) && simtime < roff.get_time_at(j + 1))
				{
					index_row_runoff = j;
				}
			}
		}

		int num_runoffs_temp = arglist.num_runoffs;
		if (num_runoffs_temp == 0)
		{
			num_runoffs_temp = 1;
		}
		
		int runoff_row_size_temp = 1;
		if (arglist.num_runoffs > 0)
		{
			runoff_row_size_temp = roff.get_num_inflow_rows();
		}
		host_runoff_id_arr = new int[rows * cols];
		host_runoff_intensity_arr_size = num_runoffs_temp * runoff_row_size_temp;
		host_runoff_intensity_arr = new T[host_runoff_intensity_arr_size];

		for (int j = 0; j < rows * cols; j++)
		{
			if (arglist.num_runoffs > 0)
			{
				host_runoff_id_arr[j] = sub_rin.get_value(j);
			}
			else
			{
				host_runoff_id_arr[j] = -1;
			}
		}

		for (int j = 0; j < num_runoffs_temp * runoff_row_size_temp; j++)
		{
			host_runoff_intensity_arr[j] = 0.0;
		}

		if (arglist.num_runoffs > 0)
		{
			for (int j = 0; j < arglist.num_runoffs; j++)
			{
				for (int k = 0; k < roff.get_num_inflow_rows(); k++)
				{
					host_runoff_intensity_arr[j*roff.get_num_inflow_rows() + k] = roff.get_flow_at(k, j + 1);
				}
			}
		}
	}

	template<typename T>
	void triton<T>::create_host_aux_vectors()
	{
		host_halo_h_arr_size = 4 * cols;
		host_halo_qxqy_arr_size = 8 * cols;
		host_halo_h_arr = new T[host_halo_h_arr_size];
		host_halo_qxqy_arr = new T[host_halo_qxqy_arr_size];
		for (int i = 0; i < host_halo_h_arr_size; i++)
		{
			host_halo_h_arr[i] = 0.0;
		}
		for (int i = 0; i < host_halo_qxqy_arr_size; i++)
		{
			host_halo_qxqy_arr[i] = 0.0;
		}

		host_sqrth_arr = new T[rows * cols];
		for (int j = 0; j < rows * cols; j++)
		{
			host_sqrth_arr[j] = 0.0;
		}
		
		#ifdef ACTIVE_GPU
		if ((rows*cols) % THREAD_BLOCK == 0)
		{
			host_reduce_dt_arr_sz = (rows*cols) / THREAD_BLOCK;
		}
		else
		{
			host_reduce_dt_arr_sz = (rows*cols) / THREAD_BLOCK + 1;
		}
		#else
		host_reduce_dt_arr_sz = rows * cols;
		#endif
		
		host_dt_values_arr = new T[host_reduce_dt_arr_sz];
		for (int i = 0; i < host_reduce_dt_arr_sz; i++)
		{
			host_dt_values_arr[i] = 0.0;
		}
		
		
		host_rhsh0 = new T[rows*cols];
		host_rhsh1 = new T[rows*cols];
		host_rhsqx0 = new T[rows*cols];
		host_rhsqx1 = new T[rows*cols];
		host_rhsqy0 = new T[rows*cols];
		host_rhsqy1 = new T[rows*cols];

		for (int i = 0; i < rows*cols; i++)
		{
			host_rhsh0[i] = 0.0;
			host_rhsh1[i] = 0.0;
			host_rhsqx0[i] = 0.0;
			host_rhsqx1[i] = 0.0;
			host_rhsqy0[i] = 0.0;
			host_rhsqy1[i] = 0.0;
		}
		
	}

	template<typename T>
	void triton<T>::create_host_vectors()
	{
		host_vec = std::vector<T*>();
		host_vec_int = std::vector<int*>();

		host_vec.push_back(sub_hin.get_data());
		host_vec.push_back(sub_qxin.get_data());
		host_vec.push_back(sub_qyin.get_data());
		host_vec.push_back(sub_nin.get_data());
		host_vec.push_back(sub_dem.get_data());

		host_vec.push_back(host_rhsh0);
		host_vec.push_back(host_rhsh1);
		host_vec.push_back(host_rhsqx0);
		host_vec.push_back(host_rhsqx1);
		host_vec.push_back(host_rhsqy0);
		host_vec.push_back(host_rhsqy1);

		host_vec.push_back(host_sqrth_arr);
		host_vec.push_back(host_halo_h_arr);
		host_vec.push_back(host_halo_qxqy_arr);
		host_vec.push_back(host_dt_values_arr);
		host_vec.push_back(host_hyg_time_arr);
		host_vec.push_back(host_hyg_val_arr);
		host_vec.push_back(host_runoff_intensity_arr);
		host_vec.push_back(host_extbc_var1_arr);
		host_vec.push_back(host_extbc_var2_arr);

		host_vec_int.push_back(host_src_pos_arr);
		host_vec_int.push_back(host_runoff_id_arr);
		host_vec_int.push_back(host_relative_bc_index);
		host_vec_int.push_back(host_bc_type);
		host_vec_int.push_back(host_bc_start_index);
		host_vec_int.push_back(host_bc_nrows_vars);
	}


	template<typename T>
	void triton<T>::create_device_vectors()
	{
		nbytes = (sizeof(T) * rows * cols);
		nbytes_halo_h = (sizeof(T) * host_halo_h_arr_size);
		nbytes_halo_qxqy = (sizeof(T) * host_halo_qxqy_arr_size);

		#ifdef ACTIVE_GPU
		cudaStreamCreate(&streams);

		device_vec = std::vector<T*>();
		device_vec_int = std::vector<int*>();

		int nbytes_dt = (sizeof(T) * host_reduce_dt_arr_sz);
		int nbytes_hyg_time = (sizeof(T) * host_hyg_time_arr_size);
		int nbytes_hyg_val = (sizeof(T) * host_hyg_time_arr_size * host_src_pos_arr_size);
		int nbytes_runoff_intensity = (sizeof(T) * host_runoff_intensity_arr_size);
		int nbytes_src_pos = (sizeof(int) * host_src_pos_arr_size);
		int nbytes_runoff_id = (sizeof(int) * rows * cols);
		int nbytes_bc_cell_size = (sizeof(int) * max(host_bc_cells_size,1));
		int nbytes_bc_vars = (sizeof(T) * host_bc_vars_arr_size);


		T *device_h, *device_qx, *device_qy,
			*device_n, *device_dem, *device_sqrth_arr, *device_halo_h_arr, *device_halo_qxqy_arr, *device_dt_values_arr,
			*device_rhsh0, *device_rhsh1, *device_rhsqx0, *device_rhsqx1, *device_rhsqy0, *device_rhsqy1,
			*device_hyg_time_arr, *device_hyg_val_arr,
			*device_runoff_intensity_arr,
			*device_bc_var1_arr, *device_bc_var2_arr;

		int *device_src_pos_arr, *device_runoff_id_arr, *device_relative_bc_index, *device_bc_type,
			*device_bc_start_index, *device_bc_nrows_vars;
		
		cudaMalloc((void**)&device_h, nbytes);
		cudaMalloc((void**)&device_qx, nbytes);
		cudaMalloc((void**)&device_qy, nbytes);
		cudaMalloc((void**)&device_n, nbytes);
		cudaMalloc((void**)&device_dem, nbytes);

		cudaMalloc((void**)&device_rhsh0, nbytes);
		cudaMalloc((void**)&device_rhsh1, nbytes);
		cudaMalloc((void**)&device_rhsqx0, nbytes);
		cudaMalloc((void**)&device_rhsqx1, nbytes);
		cudaMalloc((void**)&device_rhsqy0, nbytes);
		cudaMalloc((void**)&device_rhsqy1, nbytes);

		cudaMalloc((void**)&device_sqrth_arr, nbytes);
		cudaMalloc((void**)&device_halo_h_arr, nbytes_halo_h);
		cudaMalloc((void**)&device_halo_qxqy_arr, nbytes_halo_qxqy);
		cudaMalloc((void**)&device_dt_values_arr, nbytes_dt);
		cudaMalloc((void**)&device_hyg_time_arr, nbytes_hyg_time);
		cudaMalloc((void**)&device_hyg_val_arr, nbytes_hyg_val);
		cudaMalloc((void**)&device_runoff_intensity_arr, nbytes_runoff_intensity);
		cudaMalloc((void**)&device_bc_var1_arr, nbytes_bc_vars);
		cudaMalloc((void**)&device_bc_var2_arr, nbytes_bc_vars);


		cudaMalloc((void**)&device_src_pos_arr, nbytes_src_pos);
		cudaMalloc((void**)&device_runoff_id_arr, nbytes_runoff_id);
		cudaMalloc((void**)&device_relative_bc_index, nbytes_bc_cell_size);
		cudaMalloc((void**)&device_bc_type, nbytes_bc_cell_size);
		cudaMalloc((void**)&device_bc_start_index, nbytes_bc_cell_size);
		cudaMalloc((void**)&device_bc_nrows_vars, nbytes_bc_cell_size);

		device_vec.push_back(device_h);
		device_vec.push_back(device_qx);
		device_vec.push_back(device_qy);
		device_vec.push_back(device_n);
		device_vec.push_back(device_dem);

		device_vec.push_back(device_rhsh0);
		device_vec.push_back(device_rhsh1);
		device_vec.push_back(device_rhsqx0);
		device_vec.push_back(device_rhsqx1);
		device_vec.push_back(device_rhsqy0);
		device_vec.push_back(device_rhsqy1);

		device_vec.push_back(device_sqrth_arr);
		device_vec.push_back(device_halo_h_arr);
		device_vec.push_back(device_halo_qxqy_arr);
		device_vec.push_back(device_dt_values_arr);
		device_vec.push_back(device_hyg_time_arr);
		device_vec.push_back(device_hyg_val_arr);
		device_vec.push_back(device_runoff_intensity_arr);
		device_vec.push_back(device_bc_var1_arr);
		device_vec.push_back(device_bc_var2_arr);

		device_vec_int.push_back(device_src_pos_arr);
		device_vec_int.push_back(device_runoff_id_arr);
		device_vec_int.push_back(device_relative_bc_index);
		device_vec_int.push_back(device_bc_type);
		device_vec_int.push_back(device_bc_start_index);
		device_vec_int.push_back(device_bc_nrows_vars);


		cudaMemcpyAsync(device_vec[H], host_vec[H], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[QX], host_vec[QX], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[QY], host_vec[QY], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[N], host_vec[N], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[DEM], host_vec[DEM], nbytes, cudaMemcpyHostToDevice, streams);

		cudaMemcpyAsync(device_vec[RHSH0], host_vec[RHSH0], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RHSH1], host_vec[RHSH1], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RHSQX0], host_vec[RHSQX0], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RHSQX1], host_vec[RHSQX1], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RHSQY0], host_vec[RHSQY0], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RHSQY1], host_vec[RHSQY1], nbytes, cudaMemcpyHostToDevice, streams);

		cudaMemcpyAsync(device_vec[SQRTH], host_vec[SQRTH], nbytes, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[HALOH], host_vec[HALOH], nbytes_halo_h, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[HALOQXQY], host_vec[HALOQXQY], nbytes_halo_qxqy, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[DT], host_vec[DT], nbytes_dt, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[HYGT], host_vec[HYGT], nbytes_hyg_time, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[HYGV], host_vec[HYGV], nbytes_hyg_val, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[RUNIN], host_vec[RUNIN], nbytes_runoff_intensity, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[EXTBCV1], host_vec[EXTBCV1], nbytes_bc_vars, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec[EXTBCV2], host_vec[EXTBCV2], nbytes_bc_vars, cudaMemcpyHostToDevice, streams);


		cudaMemcpyAsync(device_vec_int[SRCP], host_vec_int[SRCP], nbytes_src_pos, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec_int[RUNID], host_vec_int[RUNID], nbytes_runoff_id, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec_int[BCRELATIVEINDEX], host_vec_int[BCRELATIVEINDEX], nbytes_bc_cell_size, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec_int[BCTYPE], host_vec_int[BCTYPE], nbytes_bc_cell_size, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec_int[BCINDEXSTART], host_vec_int[BCINDEXSTART], nbytes_bc_cell_size, cudaMemcpyHostToDevice, streams);
		cudaMemcpyAsync(device_vec_int[BCNROWSVARS], host_vec_int[BCNROWSVARS], nbytes_bc_cell_size, cudaMemcpyHostToDevice, streams);
		cudaStreamSynchronize(streams);
		#endif
	}
	/* --------------------------------------------------------------------------- */

	template<typename T>
	int triton<T>::calc_src_col(T src_x, T xllc, T cell_size_)
	{
		return ceil(((src_x - xllc) / cell_size_)) - 1;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int triton<T>::calc_src_row(T src_y, T yllc, T cell_size_, int nrows)
	{
		return ceil((nrows - ((src_y - yllc) / cell_size_))) - 1;
	}

	template<class T>
	triton<T>::~triton()
	{		
		//dont delete matrix object
		delete[] host_vec[EXTBCV2];
		delete[] host_vec[EXTBCV1];
		delete[] host_vec[RUNIN];
		delete[] host_vec[HYGV];
		delete[] host_vec[HYGT];
		delete[] host_vec[DT];
		delete[] host_vec[HALOQXQY];
		delete[] host_vec[HALOH];
		delete[] host_vec[SQRTH];
		delete[] host_vec[RHSQY1];
		delete[] host_vec[RHSQY0];
		delete[] host_vec[RHSQX1];
		delete[] host_vec[RHSQX0];
		delete[] host_vec[RHSH1];
		delete[] host_vec[RHSH0];

		delete[] host_vec_int[BCNROWSVARS];
		delete[] host_vec_int[BCINDEXSTART];
		delete[] host_vec_int[BCTYPE];
		delete[] host_vec_int[BCRELATIVEINDEX];
		delete[] host_vec_int[RUNID];
		delete[] host_vec_int[SRCP];

#ifdef ACTIVE_GPU
		while (!device_vec.empty())
		{
			cudaFree(device_vec.back());
			device_vec.pop_back();
		}

		while (!device_vec_int.empty())
		{
			cudaFree(device_vec_int.back());
			device_vec_int.pop_back();
		}
#endif
	}

	template<typename T>
	void triton<T>::simulate()
	{
		if(rank==0){
			std::cerr << OK "Simulation starts" << std::endl;
		}
		st.start(SIMULATION_TIME);
		
		Output::output<T> out;
		out.init(rows, cols, rank, size, project_dir, arglist.outfile_pattern, arglist.time_series_flag, cfg_content, arglist.output_option);

		if (arglist.time_series_flag)
		{
			out.init_time_series(arglist.observation_x_loc.size(), observation_cells);
		}

		global_dt = arglist.time_step;
		local_dt = arglist.time_step;
		int it_count = arglist.it_count;
		int print_id = arglist.checkpoint_id;

		while (simtime < arglist.sim_duration)
		{
			it_count++;

			if (!arglist.time_increment_fixed)
			{
				compute_local_dt();
				compute_global_dt(print_id);
			}
			
			compute_new_state();

			simtime += global_dt;

			if (simtime >= arglist.print_interval * (print_id + 1))
			{
				print_id++;

#ifdef ACTIVE_GPU
				st.start(COMPUTE_TIME);
				cudaMemcpyAsync(host_vec[H], device_vec[H], nbytes, cudaMemcpyDeviceToHost, streams);
				cudaMemcpyAsync(host_vec[QX], device_vec[QX], nbytes, cudaMemcpyDeviceToHost, streams);
				cudaMemcpyAsync(host_vec[QY], device_vec[QY], nbytes, cudaMemcpyDeviceToHost, streams);
				cudaStreamSynchronize(streams);
				st.stop(COMPUTE_TIME);
#endif

				st.start(IO_TIME);
				out.write_output(sub_hin, sub_qxin, sub_qyin, arglist.output_format, arglist.print_option, print_id, it_count, simtime, global_dt);
				st.stop(IO_TIME);
			}

		}
		st.stop(SIMULATION_TIME);
		st.stop(TOTAL_TIME);
		
		out.write_times(st);
		if(rank==0){
			std::cerr << OK "Simulation ends" << std::endl;
		}


	}

	template<typename T>
	void triton<T>::compute_local_dt()
	{
		st.start(COMPUTE_TIME);
#ifdef ACTIVE_GPU
		int cur_dt_arr_sz = host_reduce_dt_arr_sz;

		Kernels::compute_dt << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, THREAD_BLOCK * sizeof(T), streams >> > (rows*cols, cell_size,
			device_vec[QX], device_vec[QY], device_vec[H], device_vec[DT], arglist.courant, arglist.hextra);

		while (cur_dt_arr_sz > 1)
		{
			int temp_dt_arr_sz = (cur_dt_arr_sz / THREAD_BLOCK) + 1;
			if(cur_dt_arr_sz % THREAD_BLOCK == 0)
			{
				temp_dt_arr_sz = (cur_dt_arr_sz / THREAD_BLOCK);
			}
			Kernels::find_min_dt << <temp_dt_arr_sz, THREAD_BLOCK, THREAD_BLOCK * sizeof(T), streams >> > (cur_dt_arr_sz, device_vec[DT]);

			cur_dt_arr_sz = temp_dt_arr_sz;
		}

		cudaMemcpyAsync(&local_dt, device_vec[DT], sizeof(T), cudaMemcpyDeviceToHost, streams);
		cudaStreamSynchronize(streams);
#else
		Kernels::compute_dt(rows*cols, cell_size, host_vec[QX], host_vec[QY], host_vec[H], host_vec[DT], arglist.courant, arglist.hextra);
		Kernels::find_min_dt(rows*cols, host_vec[DT]);
		local_dt = host_vec[DT][0];
#endif
		st.stop(COMPUTE_TIME);
	}


	template<typename T>
	void triton<T>::compute_global_dt(int print_id)
	{
		if (size > 1)
		{
			st.start(MPI_TIME);
			MPI_Allreduce(&local_dt, &global_dt, 1, MPI_DATA_TYPE, MPI_MIN, MPI_COMM_WORLD);
			st.stop(MPI_TIME);
		}
		else
		{
			global_dt = local_dt;
		}

		if (global_dt >= MAX_VALUE - 1.0)
		{
			global_dt = arglist.time_step;
		}

		if (simtime + global_dt > arglist.print_interval * (print_id + 1))
		{
			global_dt = arglist.print_interval * (print_id + 1) - simtime;
		}
	}


	template<typename T>
	void triton<T>::compute_new_state()
	{
		st.start(COMPUTE_TIME);

#ifdef ACTIVE_GPU
		Kernels::initialize_sqrt << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, device_vec[H], device_vec[SQRTH],
			device_vec[RHSH0], device_vec[RHSH1], device_vec[RHSQX0], device_vec[RHSQX1], device_vec[RHSQY0], device_vec[RHSQY1]);

		Kernels::flux_x << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, rows, cols, cell_size, global_dt,
			device_vec[H], device_vec[QX], device_vec[QY], device_vec[DEM], device_vec[SQRTH],
			device_vec[RHSH0], device_vec[RHSH1], device_vec[RHSQX0], device_vec[RHSQX1], device_vec[RHSQY0], device_vec[RHSQY1], arglist.hextra);

		Kernels::flux_y << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, rows, cols, cell_size, global_dt,
			device_vec[H], device_vec[QX], device_vec[QY], device_vec[DEM], device_vec[SQRTH],
			device_vec[RHSH0], device_vec[RHSH1], device_vec[RHSQX0], device_vec[RHSQX1], device_vec[RHSQY0], device_vec[RHSQY1], arglist.hextra);

		Kernels::update_cells << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, rows, cols, global_dt,
			device_vec[H], device_vec[QX], device_vec[QY], device_vec[DEM], device_vec[N],
			device_vec[RHSH0], device_vec[RHSH1], device_vec[RHSQX0], device_vec[RHSQX1], device_vec[RHSQY0], device_vec[RHSQY1], arglist.hextra);
#else
		Kernels::initialize_sqrt(rows*cols, host_vec[H], host_vec[SQRTH],
			host_vec[RHSH0], host_vec[RHSH1], host_vec[RHSQX0], host_vec[RHSQX1], host_vec[RHSQY0], host_vec[RHSQY1]);

		Kernels::flux_x(rows*cols, rows, cols, cell_size, global_dt,
			host_vec[H], host_vec[QX], host_vec[QY], host_vec[DEM], host_vec[SQRTH],
			host_vec[RHSH0], host_vec[RHSH1], host_vec[RHSQX0], host_vec[RHSQX1], host_vec[RHSQY0], host_vec[RHSQY1], arglist.hextra);

		Kernels::flux_y(rows*cols, rows, cols, cell_size, global_dt,
			host_vec[H], host_vec[QX], host_vec[QY], host_vec[DEM], host_vec[SQRTH],
			host_vec[RHSH0], host_vec[RHSH1], host_vec[RHSQX0], host_vec[RHSQX1], host_vec[RHSQY0], host_vec[RHSQY1], arglist.hextra);

		Kernels::update_cells(rows*cols, rows, cols, global_dt,
			host_vec[H], host_vec[QX], host_vec[QY], host_vec[DEM], host_vec[N],
			host_vec[RHSH0], host_vec[RHSH1], host_vec[RHSQX0], host_vec[RHSQX1], host_vec[RHSQY0], host_vec[RHSQY1], arglist.hextra);
#endif

		if (arglist.num_runoffs > 0)
		{
			if (simtime > roff.get_time_at(index_row_runoff + 1))
			{
				index_row_runoff++;
			}

#ifdef ACTIVE_GPU
			Kernels::update_runoff << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, rows, cols, global_dt,
				device_vec_int[RUNID], index_row_runoff, roff.get_num_inflow_rows(), device_vec[RUNIN], device_vec[H], device_vec[QX], device_vec[QY], arglist.hextra);
#else
			Kernels::update_runoff(rows*cols, rows, cols, global_dt,
				host_vec_int[RUNID], index_row_runoff, roff.get_num_inflow_rows(), host_vec[RUNIN], host_vec[H], host_vec[QX], host_vec[QY], arglist.hextra);
#endif
		}

		if (num_of_src > 0)
		{
			bool check_flag = false;
			int idx_high = hyg.get_num_inflow_rows() - 1;
			for (int i = idx_low; i < idx_high; i++)
			{
				if (hyg.get_time_at(i + 1) > simtime)
				{
					idx_low = i;
					idx_high = i + 1;
					check_flag = true;
					break;
				}
			}
			if (!check_flag)
			{
				idx_low = idx_high;
			}

#ifdef ACTIVE_GPU
			Kernels::compute_flow << <(num_of_src + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (num_of_src, device_vec[HYGT], device_vec[HYGV],
				cell_size, global_dt, simtime, idx_low, idx_high, device_vec[H], device_vec_int[SRCP]);
#else
			Kernels::compute_flow(num_of_src, host_vec[HYGT], host_vec[HYGV],
				cell_size, global_dt, simtime, idx_low, idx_high, host_vec[H], host_vec_int[SRCP]);
#endif
		}


		if (num_of_extbc > 0 && num_extbc_cells > 0)
		{
			#ifdef ACTIVE_GPU
			Kernels::compute_extbc_values<<< (num_extbc_cells + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >>> (num_extbc_cells, rows, cols, global_dt, device_vec[H], device_vec[QX], device_vec[QY], device_vec[DEM], device_vec[N], device_vec_int[BCRELATIVEINDEX], device_vec_int[BCTYPE], device_vec_int[BCINDEXSTART], device_vec_int[BCNROWSVARS], device_vec[EXTBCV1], device_vec[EXTBCV2], simtime, rank, size);
		   #else
			Kernels::compute_extbc_values(num_extbc_cells, rows, cols, global_dt, host_vec[H], host_vec[QX], host_vec[QY], host_vec[DEM], host_vec[N], host_vec_int[BCRELATIVEINDEX], host_vec_int[BCTYPE], host_vec_int[BCINDEXSTART], host_vec_int[BCNROWSVARS], host_vec[EXTBCV1], host_vec[EXTBCV2], simtime, rank, size);

			#endif

		}

		if (size > 1)
		{
#ifdef ACTIVE_GPU
			Kernels::halo_copy_from_gpu_h << <(2 * cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (2 * cols, rows, cols, device_vec[H], device_vec[HALOH]);

			if (arglist.gpu_direct_flag)
			{
				cudaStreamSynchronize(streams);
				st.stop(COMPUTE_TIME);

				st.start(MPI_TIME);
				MpiUtils::exchange(device_vec[HALOH], 4, cols, rank, size, USE_HALO);
				st.stop(MPI_TIME);

				st.start(COMPUTE_TIME);
			}
			else
			{
				cudaMemcpyAsync(host_vec[HALOH], device_vec[HALOH], nbytes_halo_h, cudaMemcpyDeviceToHost, streams);
				cudaStreamSynchronize(streams);
				st.stop(COMPUTE_TIME);

				st.start(MPI_TIME);
				MpiUtils::exchange(host_vec[HALOH], 4, cols, rank, size, USE_HALO);
				st.stop(MPI_TIME);

				st.start(COMPUTE_TIME);
				cudaMemcpyAsync(device_vec[HALOH], host_vec[HALOH], nbytes_halo_h, cudaMemcpyHostToDevice, streams);
			}

			Kernels::halo_copy_to_gpu_h << <(2 * cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (2 * cols, rows, cols, device_vec[H], device_vec[HALOH]);
#else
			Kernels::halo_copy_from_gpu_h(2 * cols, rows, cols, host_vec[H], host_vec[HALOH]);

			st.stop(COMPUTE_TIME);

			st.start(MPI_TIME);
			MpiUtils::exchange(host_vec[HALOH], 4, cols, rank, size, USE_HALO);
			st.stop(MPI_TIME);

			st.start(COMPUTE_TIME);

			Kernels::halo_copy_to_gpu_h(2 * cols, rows, cols, host_vec[H], host_vec[HALOH]);
#endif
		}

#ifdef ACTIVE_GPU
		Kernels::wet_dry << <(rows*cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (rows*cols, rows, cols, global_dt, device_vec[H], device_vec[QX], device_vec[QY], device_vec[DEM], arglist.hextra);
#else
		Kernels::wet_dry(rows*cols, rows, cols, global_dt, host_vec[H], host_vec[QX], host_vec[QY], host_vec[DEM], arglist.hextra);
#endif

		if (size > 1)
		{
#ifdef ACTIVE_GPU
			Kernels::halo_copy_from_gpu_qxqy << <(2 * cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (2 * cols, rows, cols, device_vec[QX], device_vec[QY], device_vec[HALOQXQY]);

			if (arglist.gpu_direct_flag)
			{
				cudaStreamSynchronize(streams);
				st.stop(COMPUTE_TIME);

				st.start(MPI_TIME);
				MpiUtils::exchange(device_vec[HALOQXQY], 8, cols, rank, size, USE_HALO);
				st.stop(MPI_TIME);

				st.start(COMPUTE_TIME);
			}
			else
			{
				cudaMemcpyAsync(host_vec[HALOQXQY], device_vec[HALOQXQY], nbytes_halo_qxqy, cudaMemcpyDeviceToHost, streams);
				cudaStreamSynchronize(streams);
				st.stop(COMPUTE_TIME);

				st.start(MPI_TIME);
				MpiUtils::exchange(host_vec[HALOQXQY], 8, cols, rank, size, USE_HALO);
				st.stop(MPI_TIME);

				st.start(COMPUTE_TIME);
				cudaMemcpyAsync(device_vec[HALOQXQY], host_vec[HALOQXQY], nbytes_halo_qxqy, cudaMemcpyHostToDevice, streams);
			}

			Kernels::halo_copy_to_gpu_qxqy << <(2 * cols + THREAD_BLOCK - 1) / THREAD_BLOCK, THREAD_BLOCK, 0, streams >> > (2 * cols, rows, cols, device_vec[QX], device_vec[QY], device_vec[HALOQXQY]);
#else
			Kernels::halo_copy_from_gpu_qxqy(2 * cols, rows, cols, host_vec[QX], host_vec[QY], host_vec[HALOQXQY]);

			st.stop(COMPUTE_TIME);

			st.start(MPI_TIME);
			MpiUtils::exchange(host_vec[HALOQXQY], 8, cols, rank, size, USE_HALO);
			st.stop(MPI_TIME);

			st.start(COMPUTE_TIME);

			Kernels::halo_copy_to_gpu_qxqy(2 * cols, rows, cols, host_vec[QX], host_vec[QY], host_vec[HALOQXQY]);
#endif
		}
		
		st.stop(COMPUTE_TIME);
	}
}

#endif
