/** @file MpiUtils.h
 *  @brief Header containing the MpiUtils class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for MpiUtils class
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



#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "matrix.h"
#include "constants.h"

namespace MpiUtils
{
	Constants::dims_t get_local_dims(int globalrows, int globalcols, int rank, int size);

	struct partition_data_t
	{
		int  size, rows, cols, sub_rows, sub_cols;
		int  cart_dims[2];
		std::vector<Constants::dims_t > part_dims;

		partition_data_t() : size(0), rows(0), cols(0) {}

		partition_data_t(int s, int r, int c) : size(s), rows(r), cols(c)
		{
			part_dims.assign(size, Constants::dims_t(0, 0));

			for (int p = 0; p < size; ++p)
			{
				part_dims[p] = get_local_dims(rows, cols, p, size);
			}
		}
	};

	/* --------------------------------------------------------------------------- */

	Constants::dims_t get_local_dims(int globalrows, int globalcols, int rank, int size)
	{
		int halo = GHOST_CELL_PADDING;
		Constants::dims_t localgrid(halo * 2, halo * 2);

		localgrid.first += (globalrows / size);
		localgrid.second += globalcols;

		int rem = (globalrows % size);

		if (rank < rem && rem > 0)
		{
			localgrid.first++;
		}

		return localgrid;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void exchange(T* local, int lrows, int lcols, int rank, int size, int type)
	{
		int factor = 1;
		if (type == USE_HALO)
		{
			factor = lrows / 4;
		}

		int
			pr = (lcols * (lrows - 2 * factor)),
			lr = (lcols * (lrows - 1 * factor)),
			sr = (lcols * 1 * factor);

		int data_size = lcols * factor;

		MPI_Request send_request, recv_request;
		MPI_Status status;

		if (rank < size - 1)
		{
			MPI_Isend(&(local[pr]), data_size, MPI_DATA_TYPE, (rank + 1), 0, MPI_COMM_WORLD, &send_request);
		}

		if (rank > 0)
		{
			MPI_Irecv(&(local[0]), data_size, MPI_DATA_TYPE, (rank - 1), 0, MPI_COMM_WORLD, &recv_request);
		}

		if (rank < size - 1)
		{
			MPI_Wait(&send_request, &status);
		}

		if (rank > 0)
		{
			MPI_Wait(&recv_request, &status);
		}

		if (rank > 0)
		{
			MPI_Isend(&(local[sr]), data_size, MPI_DATA_TYPE, (rank - 1), 0, MPI_COMM_WORLD, &send_request);
		}

		if (rank < size - 1)
		{
			MPI_Irecv(&(local[lr]), data_size, MPI_DATA_TYPE, (rank + 1), 0, MPI_COMM_WORLD, &recv_request);
		}

		if (rank > 0)
		{
			MPI_Wait(&send_request, &status);
		}

		if (rank < size - 1)
		{
			MPI_Wait(&recv_request, &status);
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	Matrix::matrix<T> scatter_exchange(T* global, partition_data_t pd, int rank)
	{
		Constants::dims_t lgrid = get_local_dims(pd.rows, pd.cols, rank, pd.size);
		int
			lrows = lgrid.first,
			lcols = lgrid.second,
			subsize = lrows * lcols;

		Matrix::matrix<T> sub_grid(lrows, lcols);
		sub_grid.zero_fill();

		if (rank == 0)
		{
			memcpy(sub_grid.get_address_at(0, 0), &global[0], subsize * sizeof(T));

			int row_pos = lrows - 2;
			
			for (int p = 1; p < pd.size; p++)
			{
				Constants::dims_t lgrid2 = get_local_dims(pd.rows, pd.cols, p, pd.size);
				int
					lrows2 = lgrid2.first,
					lcols2 = lgrid2.second,
					subsize2 = lrows2 * lcols2;

				MPI_Send(&global[row_pos*lcols2], subsize2, MPI_DATA_TYPE, p, 0, MPI_COMM_WORLD);
				row_pos += lrows2 - 2;
			}
		}
		else
		{
			MPI_Recv(sub_grid.get_address_at(0, 0), subsize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		return sub_grid;
	}

	/* --------------------------------------------------------------------------- */

	Matrix::matrix<int> scatter_exchange_int(int* global, partition_data_t pd, int rank)
	{
		Constants::dims_t lgrid = get_local_dims(pd.rows, pd.cols, rank, pd.size);
		int
			lrows = lgrid.first,
			lcols = lgrid.second,
			subsize = lrows * lcols;

		Matrix::matrix<int> sub_grid(lrows, lcols);
		sub_grid.zero_fill_int();

		if (rank == 0)
		{
			memcpy(sub_grid.get_address_at(0, 0), &global[0], subsize * sizeof(int));
			
			int row_pos = lrows - 2;
			
			for (int p = 1; p < pd.size; p++)
			{
				Constants::dims_t lgrid2 = get_local_dims(pd.rows, pd.cols, p, pd.size);
				int
					lrows2 = lgrid2.first,
					lcols2 = lgrid2.second,
					subsize2 = lrows2 * lcols2;

				MPI_Send(&global[row_pos*lcols2], subsize2, MPI_INTEGER, p, 0, MPI_COMM_WORLD);
				row_pos += lrows2 - 2;
			}
		}
		else
		{
			MPI_Recv(sub_grid.get_address_at(0, 0), subsize, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		return sub_grid;
	}

}

#endif
