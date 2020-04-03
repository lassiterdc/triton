/** @file Matrix.h
 *  @brief Header containing the Matrix class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for Matrix class
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



#ifndef MATRIX_H
#define MATRIX_H

#include "string_utils.h"

namespace Matrix
{
	template<class T>
	class matrix
	{

	public:
		matrix<T>();
		matrix<T>(int rows, int cols);
		matrix<T>(int rows, int cols, T** arr);
		matrix<T>(matrix<T> const& m);

		~matrix<T>();

		T& operator()(int row, int col);
		T operator()(int row, int col) const;
		matrix& operator=(matrix<T> m);
		matrix& operator*=(T value);
		matrix& operator*(T value);
		matrix& operator+(T value);
		matrix& operator+(matrix const& m);
		matrix& operator+=(matrix const& m);
		matrix& operator+=(T value);
		matrix<T>& operator*(matrix const& m);

		T* get_data() const;
		T* begin();
		T* get_address_at(int row, int col);

		int get_num_rows() const;
		int get_num_cols() const;
		int get_ghost_nrows() const;
		int get_ghost_ncols() const;

		void set_size(int rows, int cols);
		void resize(int rows, int cols);

		void set_value(int row, int col, T value);
		void set_value(std::pair<int, int> cell, T value);
		void set_value(int index, T value);

		T get_value(int row, int col);
		T get_value(std::pair<int, int>);
		T get_value(int index);

		void add_ghost_cells(int grows, int gcols, T value);
		void remove_ghost_cells();
		void copy_value_into_ghost_cells();
		void copy_elevation_into_ghost_cells(std::vector<int> irows, std::vector<int> icols, int ncells, int location);
		void set_infinite_walls();

		bool is_inbounds(int row, int col);
		void zero_fill();
		void zero_fill_int();
		void pow(T e);
		void square();

		void load_from_ascii_file(std::string& filepath);
		void load_from_ascii_file(int rows, int cols, std::string& filepath);
		void load_from_ascii_file(int rows, int cols, std::string& filepath, int header_size);
		void load_from_binary_file(int rows, int cols, std::string& filepath);
		void load_from_binary_file(int rows, int cols, std::string& filepath, int header_size);
		std::pair<int, int> get_dims_2d(std::string& filepath);

	private:
		int ghost_nrows_ = 0, ghost_ncols_ = 0;
		int rows_, cols_;
		T* data_;
	};


	/* --------------------------------------------------------------------------- */

	/*class MatrixInvalidValueException : public std::exception
	{
		std::string msg;
	public:
		MatrixInvalidValueException(int r, int c)
			: msg(std::string("Spotted an invalid value (possible overflow) at: ") + std::to_string(r) + std::string(", ") + std::to_string(c) + std::string(")")) {}

		MatrixInvalidValueException(int i)
			: msg(std::string("Spotted an invalid value (possible overflow) at: ") + std::to_string(i) + std::string(")")) {}

		virtual const char* what() const throw ()
		{
			return msg.c_str();
		}
	};*/

	/* --------------------------------------------------------------------------- */

	template<class T>
	matrix<T>::matrix()
	{
		this->rows_ = 0;
		this->cols_ = 0;
		this->data_ = NULL;
	}

	/* --------------------------------------------------------------------------- */

	template<class T>
	matrix<T>::matrix(int rows, int cols) : rows_(rows), cols_(cols)
	{
		if (rows == 0 || cols == 0)
		{
			std::cerr << ERROR "Matrix constructor has 0 size" << std::endl;
			exit(EXIT_FAILURE);
		}

		this->data_ = new T[rows * cols]();
	}

	/* --------------------------------------------------------------------------- */

	template<class T>
	matrix<T>::matrix(int rows, int cols, T** arr)
	{
		if (rows == 0 || cols == 0)
		{
			std::cerr << ERROR "Matrix constructor has 0 size" << std::endl;
			exit(EXIT_FAILURE);
		}

		this->rows_ = rows;
		this->cols_ = cols;
		this->data_ = new T[rows * cols]();

		for (int i = 0; i < this->rows_; ++i)
		{
			for (int j = 0; j < this->cols_; ++j)
			{
				this->data_[this->cols_ * i + j] = arr[i][j];
			}
		}
	}

	/* --------------------------------------------------------------------------- */

	template<class T>
	matrix<T>::matrix(matrix<T> const& m)
	{
		T* data = m.get_data();

		this->rows_ = m.get_num_rows();
		this->cols_ = m.get_num_cols();
		this->data_ = new T[rows_ * cols_]();

		for (int i = 0; i < this->rows_ * this->cols_; ++i)
		{
			this->data_[i] = data[i];
		}
	}

	/* --------------------------------------------------------------------------- */

	template<class T>
	matrix<T>::~matrix()
	{
		if (this->data_ != NULL)
			delete[] this->data_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T& matrix<T>::operator()(int row, int col)
	{
		if (row >= this->rows_ || col >= this->cols_)
		{
			std::cerr << ERROR "Matrix index out of bounds" << std::endl;
			exit(EXIT_FAILURE);

		}

		return this->data_[this->cols_ * row + col];
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T matrix<T>::operator()(int row, int col) const
	{
		if (row >= this->rows_ || col >= this->cols_)
		{
			std::cerr << ERROR "Matrix index out of bounds" << std::endl;
			exit(EXIT_FAILURE);
		}

		return this->data_[this->cols_ * row + col];
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator*=(T value)
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[this->cols_ * i + j] *= value;
			}
		}

		return *this;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator*(T value)
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[this->cols_ * i + j] *= value;
			}
		}

		return *this;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator*(matrix const& m)
	{
		if (m.get_num_rows() != this->cols_)
		{
			std::cerr << ERROR "Invalid Matrix dimensions" << std::endl;
			exit(EXIT_FAILURE);

		}

		matrix<T> R;
		R.set_size(this->rows_, m.get_num_cols());

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				T sum = 0.0;

				for (int n = 0; n < m.get_num_cols(); n++)
				{
					sum += this->data_[this->cols_ * i + n] * m(n, j);
				}

				R.set_val(i, j, sum);
			}
		}

		return R;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator+(matrix const& m)
	{
		if (m.get_num_rows() != this->rows_ || m.get_num_cols() != this->cols_)
		{
			std::cerr << ERROR "Invalid Matrix dimensions" << std::endl;
			exit(EXIT_FAILURE);
		}

		matrix<T> R;
		R.set_size(this->rows_, this->cols_);

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				R.set_val(i, j, (this->data_[this->cols_ * i + j] + m(i, j)));
			}
		}

		return R;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator+=(matrix const& m)
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[this->cols_ * i + j] += m(i, j);
			}
		}

		return *this;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator+=(T value)
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[this->cols_ * i + j] = this->data_[this->cols_ * i + j] + value;
			}
		}

		return *this;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	matrix<T>& matrix<T>::operator=(matrix<T> m)
	{
		if (this->rows_ != m.get_num_rows() || this->cols_ != m.get_num_cols())
		{
			this->rows_ = m.get_num_rows();
			this->cols_ = m.get_num_cols();

			this->resize(this->rows_, this->cols_);
		}

		delete[] this->data_;
		this->data_ = new T[m.get_num_rows() * m.get_num_cols()]();
		for (int i = 0; i < m.get_num_rows() * m.get_num_cols(); ++i)
		{
			this->data_[i] = m.data_[i];
		}

		return *this;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T* matrix<T>::begin()
	{
		return &(this->data_[0]);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T* matrix<T>::get_address_at(int row, int col)
	{
		return &(this->data_[this->cols_ * row + col]);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T* matrix<T>::get_data() const
	{
		return this->data_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int matrix<T>::get_num_rows() const
	{
		return this->rows_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int matrix<T>::get_num_cols() const
	{
		return this->cols_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int matrix<T>::get_ghost_nrows() const
	{
		return this->ghost_nrows_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int matrix<T>::get_ghost_ncols() const
	{
		return this->ghost_ncols_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::set_size(int rows, int cols)
	{
		this->resize(rows, cols);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::resize(int rows, int cols)
	{
		if (this->data_ != NULL)
			delete[] this->data_;
		this->rows_ = rows;
		this->cols_ = cols;

		this->data_ = new T[rows_ * cols_]();
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::set_value(int index, T value)
	{
		if (index >= 0 && index < this->rows_ * this->cols_)
		{
			this->data_[index] = value;
		}
		else
		{
			std::cerr << ERROR "Matrix index out of bounds" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::set_value(int row, int col, T value)
	{
		this->set_value((this->cols_ * row + col), value);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::set_value(std::pair<int, int> cell, T value)
	{
		this->set_value((this->cols_ * cell.first + cell.second), value);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T matrix<T>::get_value(int index)
	{
		if (this->data_[index] != this->data_[index])
		{
			this->data_[index] = 0.0;
		}

		return this->data_[index];
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T matrix<T>::get_value(int row, int col)
	{
		if (!is_inbounds(row, col))
		{
			std::cerr << ERROR "Matrix index out of bounds" << std::endl;
			exit(EXIT_FAILURE);
		}

		return this->get_value(this->cols_ * row + col);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T matrix<T>::get_value(std::pair<int, int> cell)
	{
		return this->get_value(cell.first, cell.second);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::add_ghost_cells(int grows, int gcols, T value)
	{
		T* bak = new T[this->rows_ * this->cols_]();

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				bak[(this->cols_ * i) + j] = this->data_[(this->cols_ * i) + j];
			}
		}

		int ocols = cols_;

		this->resize(this->rows_ + grows * 2, this->cols_ + gcols * 2);

		for (int i = 0, k = i - grows; i < this->rows_; i++, k++)
		{
			for (int j = 0, l = j - gcols; j < this->cols_; j++, l++)
			{
				if((i >= grows) && (j >= gcols) && (i < (this->rows_ - grows)) && (j < (this->cols_ - gcols)))
				{
					this->data_[(this->cols_ * i) + j] = bak[(ocols * k) + l];
				}
				else
				{
					this->data_[(this->cols_ * i) + j] = value;
				}
			}
		}

		this->ghost_nrows_ += grows;
		this->ghost_ncols_ += gcols;

		delete[] bak;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::remove_ghost_cells()
	{
		T* bak = new T[this->rows_ * this->cols_]();

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				bak[(this->cols_ * i) + j] = this->data_[(this->cols_ * i) + j];
			}
		}

		resize(this->rows_ - this->ghost_nrows_ * 2, this->cols_ - this->ghost_ncols_ * 2);

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[(this->cols_ * i) + j] = bak[(this->cols_ + (this->ghost_ncols_ * 2)) * (i + this->ghost_nrows_) + (j + this->ghost_ncols_)];
			}
		}

		this->ghost_nrows_ = 0;
		this->ghost_ncols_ = 0;

		delete[] bak;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::copy_value_into_ghost_cells()
	{

		for (int i = 0; i < this->rows_; i++)
		{
			this->data_[i*this->cols_ + 0] = this->data_[i*this->cols_ + 1];
			this->data_[i*this->cols_ + (this->cols_ - 1)] = this->data_[i*this->cols_ + (this->cols_ - 2)];
		}
		for (int i = 0; i < this->cols_; i++)
		{
			this->data_[0 * this->cols_ + i] = this->data_[1 * this->cols_ + i];
			this->data_[(this->rows_ - 1)*this->cols_ + i] = this->data_[(this->rows_ - 2)*this->cols_ + i];
		}
	}


	template<typename T>
	void matrix<T>::copy_elevation_into_ghost_cells(std::vector<int> irows, std::vector<int> icols, int ncells, int location)
	{

		//int nrows=this->rows_;
		int ncols=this->cols_;

		for (int i = 0; i < ncells; i++)
		{
			int ix=irows[i]+GHOST_CELL_PADDING;
			int iy=icols[i]+GHOST_CELL_PADDING;

			if(location==0){
				this->data_[ix*ncols + iy-1] = this->data_[ix*ncols + iy];
			}
			if(location==2){
				this->data_[ix*ncols + iy+1] = this->data_[ix*ncols + iy];
			}
			if(location==1){
				this->data_[(ix-1)*ncols + iy] = this->data_[ix*ncols + iy];
			}
			if(location==3){
				this->data_[(ix+1)*ncols + iy] = this->data_[ix*ncols + iy];
			}
		}
	}



	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::set_infinite_walls()
	{
		for (int i = 0; i < this->rows_; i++)
		{
			this->data_[i*this->cols_ + 0] = 1e6;
			this->data_[i*this->cols_ + (this->cols_ - 1)] = 1e6;
		}
		for (int i = 0; i < this->cols_; i++)
		{
			this->data_[0 * this->cols_ + i] = 1e6;
			this->data_[(this->rows_ - 1)*this->cols_ + i] = 1e6;
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	bool matrix<T>::is_inbounds(int row, int col)
	{
		return (row < this->rows_ && row >= 0 && col < this->cols_ && col >= 0);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::zero_fill()
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[(this->cols_ * i) + j] = 0.0;
			}
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::zero_fill_int()
	{
		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[(this->cols_ * i) + j] = 0;
			}
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::pow(T e)
	{

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				this->data_[this->cols_ * i + j] = std::pow(this->data_[this->cols_ * i + j], e);
			}
		}
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::square()
	{

		for (int i = 0; i < this->rows_; i++)
		{
			for (int j = 0; j < this->cols_; j++)
			{
				T value = this->data_[this->cols_ * i + j];
				this->data_[this->cols_ * i + j] = value*value;
			}
		}
	}

	/* --------------------------------------------------------------------------- */


	template<typename T>
	void matrix<T>::load_from_ascii_file(std::string& filepath)
	{
		int j = 0;
		int rownum = 0;
		std::tuple<int, int> dims = get_dims_2d(filepath);

		std::ifstream infile(filepath.c_str());

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cerr << IN "Reading file " << filepath << std::endl;
		}

		std::string line;

		std::getline(infile, line);

		while (!infile.eof())
		{
			std::vector<std::string> row = StringUtils::split(line, ' ');

			if (rownum == 0)
			{
				this->rows_ = std::get<1>(dims);
				this->cols_ = std::get<0>(dims);

				this->set_size(this->rows_, this->cols_);
				this->zero_fill();
			}
			else
			{
				std::string val;
				std::vector<std::string>::iterator strit = row.begin();
				j = 0;

				for (; strit != row.end(); strit++, j++)
				{
					val = *strit;

					if (val.find(".") != std::string::npos)
					{
						this->data_[this->cols_ * (rownum - 1) + j] = atof(val.c_str());
					}
					else
					{
						this->data_[this->cols_ * (rownum - 1) + j] = atoi(val.c_str());
					}
				}
			}
			rownum++;
			std::getline(infile, line);
		}
		infile.close();
		std::cerr << OK "File " << filepath << " read" << std::endl;		
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::load_from_ascii_file(int rows, int cols, std::string& filepath)
	{
		std::tuple<int, int> dims = get_dims_2d(filepath);
		if (rows != std::get<1>(dims) || cols != std::get<0>(dims))
		{
			std::cerr << "Invalid dimension of " << filepath << ". (row,col): (" << std::get<1>(dims) << "," << std::get<0>(dims) << ")" << std::endl;
			exit(EXIT_FAILURE);
		}

		this->set_size(rows, cols);

		int i = 0;
		std::ifstream infile(filepath);

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cerr << IN "Reading file " << filepath << std::endl;
		}

		std::string line;

		while (infile.good())
		{
			int j = 0;
			std::getline(infile, line);
			std::vector<std::string> row = StringUtils::split(line, ' ');
			std::string val;
			std::vector<std::string>::iterator strit = row.begin();

			for (; strit != row.end(); strit++, j++)
			{
				val = *strit;

				if(val.find(".") != std::string::npos)
				{
					this->data_[(this->cols_ * i) + j] = (T)atof(val.c_str());
				}
				else
				{
					this->data_[(this->cols_ * i) + j] = (T)atoi(val.c_str());
				}
			}
			i++;
		}
		infile.close();
		std::cerr << OK "File " << filepath << " read" << std::endl;
		
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::load_from_ascii_file(int rows, int cols, std::string& filepath, int header_size)
	{

		this->set_size(rows, cols);

		int i = 0;
		std::ifstream infile(filepath);

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cerr << IN "Reading file " << filepath << std::endl;
		}

		std::string line;

		int line_number = 0;

		while (infile.good())
		{
			std::getline(infile, line);

			line_number++;
			if (line_number <= header_size)
				continue;

			int j = 0;
			std::vector<std::string> row = StringUtils::split(line, ' ');
			std::string val;
			std::vector<std::string>::iterator strit = row.begin();

			for (; strit != row.end(); strit++, j++)
			{
				val = *strit;

				if(val.find(".") != std::string::npos)
				{
					this->data_[(this->cols_ * i) + j] = (T)atof(val.c_str());
				}
				else
				{
					this->data_[(this->cols_ * i) + j] = (T)atoi(val.c_str());
				}
			}
			i++;
		}
		infile.close();
		std::cerr << OK "File " << filepath << " read" << std::endl;
		
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::load_from_binary_file(int rows, int cols, std::string& filepath)
	{
		this->set_size(rows, cols);
		std::ifstream infile(filepath, std::ios::binary);

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cerr << IN "Reading file " << filepath << std::endl;
		}
		
		T *arr = new T [BIN_DEFAULT_HEADER_SIZE];
		infile.read( (char*) arr, sizeof(T) * BIN_DEFAULT_HEADER_SIZE );
		int file_row = (int)arr[BIN_ROW_ID];
		int file_col = (int)arr[BIN_COL_ID];
		
		if(rows!= file_row || cols!=file_col)
		{
			infile.close();
			std::cerr << ERROR "Invalid Matrix dimensions" << std::endl;
			exit(EXIT_FAILURE);

		}

		infile.seekg(sizeof(T) * BIN_DEFAULT_HEADER_SIZE, std::ios::beg);
		infile.read((char*)this->data_, sizeof(T) * rows * cols);

		infile.close();
		std::cerr << OK "File " << filepath << " read" << std::endl;
		
	}
	
	/* --------------------------------------------------------------------------- */

	template<typename T>
	void matrix<T>::load_from_binary_file(int rows, int cols, std::string& filepath, int header_size)
	{

		this->set_size(rows, cols);
		std::ifstream infile(filepath, std::ios::binary);

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}
		else
		{
			std::cerr << IN "Reading file " << filepath << std::endl;
		}

		infile.seekg(sizeof(T) * header_size, std::ios::beg);
		infile.read((char*)this->data_, sizeof(T) * rows * cols);

		infile.close();
		std::cerr << OK "File " << filepath << " read" << std::endl;

	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	std::pair<int, int> matrix<T>::get_dims_2d(std::string& filepath)
	{
		int num_cols = 0, num_rows = 0;
		bool first = false;
		std::ifstream infile(filepath);

		if (!infile.is_open())
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}		
		std::string line;

		while (getline(infile, line))
		{
			std::vector<std::string> row = StringUtils::split(line, ' ');

			if (!first)
			{
				num_cols = row.size();
				first = true;
			}
			num_rows++;
		}

		infile.close();

		return std::pair<int, int>(num_cols, num_rows);
	}
}

#endif
