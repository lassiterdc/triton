/** @file swmm_triton.h
 *  @brief Header containing the swmm_triton class
 *
 *  This contains the subroutines and eventually any
 *  macros, constants, etc. needed for swmm_triton class
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

#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>

#include "constants.h"
#include "string_utils.h"
#include "mpi_utils.h"
#include <sys/stat.h>
#include <dirent.h>

namespace SWMM_triton
{

	class swmm_triton	/**< Main class for SWMM coupling. */
	{

	public:

/** @brief It initializes the simulation.
*
*  @param rank_ Subdomain id
*  @param size_ Number of subdomain
*/
		void initialize(int rank, int size, std::string inp_filename, std::string project_dir, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss);


		void end_swmm(std::string output_dir);

		void local_to_global(value_t*  local, value_t*  global, int* dict);
		void global_to_local(value_t*  global, value_t*  local, int* dict);

		int num_of_swmm_links; /**< Number of SWMM nodes connected to the surface per subdomain */
		int num_of_swmm_nodes; /**< Number of SWMM total nodes */
		int global_num_of_swmm_links; /**< Number of total (full domain) SWMM nodes connected to the surface */
		int units; /**< IS units (0) or imperial units (1) */
		std::vector<value_t> x;
		std::vector<value_t> y;
		std::vector<std::string> nodeID;
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

		int calc_swmm_node_col(value_t  node_x, value_t  xllc, value_t  cell_size_);

		int calc_swmm_node_row(value_t  node_y, value_t  yllc, value_t  cell_size_, int nrows);

		void read_inp_file(std::string inp_filename, const value_t  dx, const value_t manhole_diameter, const value_t manhole_loss);


		void process_swmm_node_locations(const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd);

		void init_swmm(std::string project_dir, std::string inp_filename);


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





	void swmm_triton::initialize(int rank, int size, std::string inp_filename, std::string project_dir, const value_t  xll, const value_t  yll, const value_t  dx, const int global_rows, const int global_cols, const MpiUtils::partition_data_t pd, const value_t manhole_diameter, const value_t manhole_loss)
	{
		rank_=rank;
		size_=size;
		units=0;

		read_inp_file(inp_filename,dx, manhole_diameter, manhole_loss);
		process_swmm_node_locations(xll, yll, dx, global_rows, global_cols, pd);
		init_swmm(project_dir, inp_filename);
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

		infile.open(inp_filename.c_str());
		while (std::getline(infile, line)) {
			if (line.find("COORDINATES") != std::string::npos){
				std::getline(infile, line); // Skip header line
				std::getline(infile, line); // Skip separator line
				while (std::getline(infile, line)) {
					if (std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); })) {
						// Line consists only of whitespace characters
						break; // Stop reading
				  	}

					std::string nodeij;
					double xij, yij;
					std::istringstream iss(line);
					if (iss >> nodeij >> xij >> yij) {
						nodeID_all.push_back(nodeij);
						if(units){
							xij*=FT_TO_M_FACTOR;
							yij*=FT_TO_M_FACTOR;
						}
						x_all.push_back(xij);
						y_all.push_back(yij);
					}
				}
			}
		}
		infile.close();

		infile.open(inp_filename.c_str());
		while (std::getline(infile, line)) {
			if (line.find("JUNCTIONS") != std::string::npos){
				std::getline(infile, line); // Skip header line
				std::getline(infile, line); // Skip separator line
            while (std::getline(infile, line)) {
				   if (std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); })) {
						// Line consists only of whitespace characters
						break; // Stop reading
				  	}
            	std::string junctionij;
               double elevationij,max_depthij;
               std::istringstream iss(line);
               if (iss >> junctionij >> elevationij >> max_depthij) {
               	junctionID_all.push_back(junctionij);
						//Note that maxdepth is always in meters (double check)
                  max_depth_all.push_back(max_depthij);
              	}
           	}
			}
		}
		infile.close();

		infile.open(inp_filename.c_str());
		while (std::getline(infile, line)) {
			if (line.find("INFLOWS") != std::string::npos){
				std::getline(infile, line); // Skip header line
				std::getline(infile, line); // Skip separator line
            while (std::getline(infile, line)) {
            	if (std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); })) {
						// Line consists only of whitespace characters
						break; // Stop reading
				  	}
            	std::string nodeij;
               std::istringstream iss(line);
               if (iss >> nodeij) {
               	nodeID.push_back(nodeij);
              	}
           	}
			}
		}
		infile.close();

		num_of_swmm_nodes=nodeID_all.size();
		global_num_of_swmm_links=nodeID.size();
		int nJunctions=junctionID_all.size();

		x.resize(global_num_of_swmm_links);
		y.resize(global_num_of_swmm_links);
		max_depth.resize(global_num_of_swmm_links);
		loss.resize(global_num_of_swmm_links);
		diameter.resize(global_num_of_swmm_links);

		// Find the coordinates (x, y) of the nodes connected to the surface
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			 for (int j = 0; j < num_of_swmm_nodes; j++) {
				  if (strcmp(nodeID[i].c_str(), nodeID_all[j].c_str()) == 0) {
						x[i] = x_all[j];
						y[i] = y_all[j];
				  }
			 }
		}

		// Find the max_depth values of the nodes connected to the surface
		for (int i = 0; i < global_num_of_swmm_links; i++) {
			 for (int j = 0; j < nJunctions; j++) {
				  if (strcmp(nodeID[i].c_str(), junctionID_all[j].c_str()) == 0) {
						max_depth[i] = max_depth_all[j];
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
		std::vector<std::string> global_nodeID = nodeID;
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


	void swmm_triton::init_swmm(std::string project_dir, std::string inp_filename)
	{

		//initialize to zero all the exchange_q values and new_depth values
		std::fill(exchange_q.begin(), exchange_q.end(), 0.0); //local variables
		std::fill(new_depth.begin(), new_depth.end(), 0.0); //local variables

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

			std::string root_dir(project_dir + "/" + OUTPUT_DIR + "/");
			std::string output_dir_swmm = project_dir + "/" + OUTPUT_DIR + "/swmm/";


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

			swmm_open(inp_filename.c_str(), report_filename.c_str(), binary_filename.c_str());
			swmm_start(TRUE);

			std::cerr << IN "SWMM initialized" << std::endl;
		}

	}

	void swmm_triton::end_swmm(std::string output_dir)
	{
		if(rank_==0){
			swmm_end();
			swmm_report(output_dir.c_str());
			swmm_close();
		}
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
