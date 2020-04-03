/** @file ExtBC.h
 *  @brief Header containing the ExtBC class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for ExtBC class
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



#ifndef EXTBC_H
#define EXTBC_H

namespace ExtBC
{
	template<class T>
	class extBC
	{
	public:
		extBC();
		extBC(std::string filename, int bctype);

		void load_from_file(std::string filename, int bctype);
		int check_extreme_extbc(std::vector<int> e_cols, std::vector<int> e_rows, int ncols, int nrows);
		void create_involved_cells(std::vector<int> e_cols, std::vector<int> e_rows, int ncols, int nrows, int bctype);

		std::vector<std::vector<T>> get_rows();
		
		T get_var1_at(int index);
		T get_var2_at(int index);

		int get_num_rows();
		void set_num_rows(int rows);

		void convert_to_secs();
		int ncells;
		int location; //0--> westBoundary  1-->northBoundary 2-->eastBoundary 3--> southBoundary
		int ncells_local;
		std::vector<int> extreme_rows, extreme_cols;
		std::vector<int> i_cols;
		std::vector<int> i_rows;

	private:
		int num_rows_;
		std::vector<std::vector<T> > data_;
	};

	/* --------------------------------------------------------------------------- */

	template<class T>
	extBC<T>::extBC()
	{
		set_num_rows(0);

	}



	/* --------------------------------------------------------------------------- */

	template<class T>
	extBC<T>::extBC(std::string filename, int bctype)
	{
		if(bctype==1){
			load_from_file(filename, bctype);
		}else{	
			std::istringstream iss(filename);
			T temp;
			iss >> temp;
			std::vector<T> row;
			row.push_back(temp);
			row.push_back(temp); //we replicate this value to have a row at least with two columns, like the other bc
			data_.push_back(row);
		}
		set_num_rows(data_.size());



	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int extBC<T>::get_num_rows()
	{
		return num_rows_;
	}


	/* --------------------------------------------------------------------------- */

	template<typename T>
	void extBC<T>::set_num_rows(int rows)
	{
		num_rows_ = rows;
	}

	/* --------------------------------------------------------------------------- */



	template<typename T>
	void extBC<T>::load_from_file(std::string filename, int bctype)
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
			line = StringUtils::trim(line);

			std::vector<std::string> str_data = StringUtils::split(line, '%');

			if ((StringUtils::trim(str_data[0])).size() > 0)
			{
				std::vector<std::string> tokens = StringUtils::split(StringUtils::trim(str_data[0]), ',');
				std::vector<T> row = StringUtils::vecstr_to_vecflt<T>(tokens);
				data_.push_back(row);
			}
		}


	}

	template<typename T>
	int extBC<T>::check_extreme_extbc(std::vector<int> e_cols, std::vector<int> e_rows, int ncols, int nrows)
	{
		if(e_cols[0]!=0 && e_cols[0]!=ncols-1 && e_rows[0]!=0 && e_rows[0]!=nrows-1){ //first point
			std::cerr << ERROR "Error selecting the external BC. It must be at the boundary of the domain" << std::endl;
			exit(EXIT_FAILURE);
		}
		if(e_cols[1]!=0 && e_cols[1]!=ncols-1 && e_rows[1]!=0 && e_rows[1]!=nrows-1){ //second point
			std::cerr << ERROR "Error selecting the external BC. It must be at the boundary of the domain" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		if(e_cols[0]!=e_cols[1] && e_rows[0]!=e_rows[1]){
			std::cerr << ERROR "Error selecting the external BC. The points must be aligned with a boundary of the domain" << std::endl;
			exit(EXIT_FAILURE);
		}else{
			if(e_cols[0]==e_cols[1]){
				return fabs(e_rows[0]-e_rows[1])+1;
			}else{
				return fabs(e_cols[0]-e_cols[1])+1;
			}
		}
	}


	template<typename T>
	void extBC<T>::create_involved_cells(std::vector<int> e_cols, std::vector<int> e_rows, int ncols, int nrows, int bctype)
	{
		i_cols.assign(ncells,-1);	
		i_rows.assign(ncells,-1);

		
		int cmin=std::min(e_cols[0],e_cols[1]);
		int rmin=std::min(e_rows[0],e_rows[1]);
		int cmax=std::max(e_cols[0],e_cols[1]);
		int rmax=std::max(e_rows[0],e_rows[1]);
		
		location = -1; 

		if(cmin==cmax){ //parallel to x-axis
			if(cmin==0){ //west boundary
				location=0;
			}else{ //east boundary
				location=2;
			}
		}

		if(rmin==rmax){ //parallel to y-axis
			if(rmin==0){ //north boundary
				location=1;
			}else{ //south boundary
				location=3;
			}
		}

		for(int i=0;i<ncells;i++){
			if(location==0){
				i_cols[i]=0;
				i_rows[i]=rmin+i;
			}
			if(location==2){
				i_cols[i]=ncols-1;
				i_rows[i]=rmin+i;
			}
			if(location==1){
				i_cols[i]=cmin+i;
				i_rows[i]=0;
			}
			if(location==3){
				i_cols[i]=cmin+i;
				i_rows[i]=nrows-1;
			}

		}
		
	}



	/* --------------------------------------------------------------------------- */

	template<typename T>
	std::vector<std::vector<T>> extBC<T>::get_rows()
	{
		return data_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T extBC<T>::get_var1_at(int index)
	{
		if (index >= static_cast<int>(data_.size()))
		{
			std::cerr << ERROR "Extbc index out of bounds" << std::endl;
			exit(EXIT_FAILURE);
		}

		return (data_.at(index)).at(0);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T extBC<T>::get_var2_at(int index)
	{
		if (index >= static_cast<int>(data_.size()))
		{
			std::cerr << ERROR "Extbc index out of bounds" << std::endl;
			exit(EXIT_FAILURE);

		}

		return (data_.at(index)).at(1);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void extBC<T>::convert_to_secs()
	{

		typename std::vector<std::vector<T>>::iterator it = data_.begin();

		for (; it != data_.end(); it++)
		{
			(*it)[0] = it->at(0) * HOUR_TO_SEC_FACTOR;
		}

	}

}

#endif
