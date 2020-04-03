/** @file DemFile.h
 *  @brief Header containing the DemFile class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for DemFile class
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



#ifndef DEM_UTILS_H
#define DEM_UTILS_H

#include "matrix.h"

namespace DemFile
{
	template<class T>
	class dem_file : public Matrix::matrix<T>
	{
	public:
		dem_file<T>() : Matrix::matrix<T>() {}
		dem_file<T>(int rows, int cols) : Matrix::matrix<T>(rows, cols) {}
		dem_file<T>(Matrix::matrix<T> const& m) : Matrix::matrix<T>(m) {}

		int get_nrows() const;
		int get_ncols() const;
		T get_xll_corner() const;
		T get_yll_corner() const;
		T get_cell_size() const;
		int get_no_data_value() const;

		void set_nrows(int row);
		void set_ncols(int col);
		void set_xll_corner(T xll);
		void set_yll_corner(T xll);
		void set_cell_size(T cell_size);
		void set_no_data_value(int no_data_value);

		void load_header_from_dem_file_ascii(std::string filename);
		void load_header_from_dem_file_binary(std::string filename);

	private:
		int nrows_, ncols_, no_data_value_;
		T xllcorner_, yllcorner_, cellsize_;
	};

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::load_header_from_dem_file_ascii(std::string filename)
	{
		std::ifstream ifs(filename.c_str());
		if (!ifs.good())
		{
			std::cerr << ERROR "Error reading file: " << filename << std::endl;
			exit(EXIT_FAILURE);

		}

		int line_num = 0;

		for (;;)
		{
			std::string line;
			std::getline(ifs, line);
			if (!ifs)
				break;

			line_num++;
			if (line_num > DEM_HEADER_SIZE)
				break;

			line = StringUtils::trim(line);
			std::vector<std::string> tokens = StringUtils::split(line, ' ');
			const char* value = (*(tokens.end() - 1)).c_str();

			if (!line.empty() && line[0] != '#')
			{
				switch (line_num)
				{
				case DEM_NCOLS_LINE:
				{
					this->ncols_ = atoi(value);
					break;
				}
				case DEM_NROWS_LINE:
				{
					this->nrows_ = atoi(value);
					break;
				}
				case DEM_XLL_CORNER_LINE:
				{
					this->xllcorner_ = atof(value);
					break;
				}
				case DEM_YLL_CORNER_LINE:
				{
					this->yllcorner_ = atof(value);
					break;
				}
				case DEM_CELL_SIZE_LINE:
				{
					this->cellsize_ = atof(value);
					break;
				}
				case DEM_NODATA_VALUE_LINE:
				{
					this->no_data_value_ = atoi(value);
					break;
				}
				default:
				{

				}
				}
			}
		}
		ifs.close();
	}

	/* --------------------------------------------------------------------------- */
	
	template<typename T>
	void dem_file<T>::load_header_from_dem_file_binary(std::string filename)
	{
		std::ifstream ifs(filename.c_str(), ios::binary);
		if (!ifs.good())
		{
			std::cerr << ERROR "Error reading file: " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		T *arr = new T [DEM_HEADER_SIZE];

		ifs.read( (char*) arr, sizeof(T) * DEM_HEADER_SIZE );
		
		ifs.close();
		
		for(int i=0; i<DEM_HEADER_SIZE; i++)
		{
			int line_num = i+1;;
			T value = arr[i];
			
			switch (line_num)
			{
				case DEM_NCOLS_LINE:
				{
					this->ncols_ = (int)(value);
					break;
				}
				case DEM_NROWS_LINE:
				{
					this->nrows_ = (int)(value);
					break;
				}
				case DEM_XLL_CORNER_LINE:
				{
					this->xllcorner_ = value;
					break;
				}
				case DEM_YLL_CORNER_LINE:
				{
					this->yllcorner_ = value;
					break;
				}
				case DEM_CELL_SIZE_LINE:
				{
					this->cellsize_ = value;
					break;
				}
				case DEM_NODATA_VALUE_LINE:
				{
					this->no_data_value_ = (int)(value);
					break;
				}
				default:
				{

				}
			}	
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int dem_file<T>::get_nrows() const
	{
		return this->nrows_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int dem_file<T>::get_ncols() const
	{
		return this->ncols_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T dem_file<T>::get_cell_size() const
	{
		return this->cellsize_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int dem_file<T>::get_no_data_value() const
	{
		return this->no_data_value_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T dem_file<T>::get_xll_corner() const
	{
		return this->xllcorner_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T dem_file<T>::get_yll_corner() const
	{
		return this->yllcorner_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_nrows(int row)
	{
		this->nrows_ = row;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_ncols(int col)
	{
		this->ncols_ = col;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_xll_corner(T xll)
	{
		this->xllcorner_ = xll;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_yll_corner(T yll)
	{
		this->yllcorner_ = yll;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_cell_size(T cell_size)
	{
		this->cellsize_ = cell_size;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void dem_file<T>::set_no_data_value(int no_data_value)
	{
		this->no_data_value_ = no_data_value;
	}
}

#endif
