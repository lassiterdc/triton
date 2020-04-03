/** @file Hydrograph.h
 *  @brief Header containing the Hydrograph class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for Hydrograph class
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



#ifndef INFLOW_H
#define INFLOW_H

#include "string_utils.h"

namespace Hydrograph
{
	template<class T>
	class hydrograph
	{
	public:
		hydrograph();
		hydrograph(std::string filename);

		void load_from_file(std::string filename);

		std::vector<std::vector<T>> get_rows();
		T get_flow_at(int index, int source_num);
		T get_time_at(int index);

		int get_num_inflow_rows();
		int get_num_inflows();
		void convert_time_hr_to_secs();
		void convert_rate_hr_to_secs();
		void convert_rate_mm_to_m();
		void set_num_flow_rows(int rows);
		void set_num_sources(int sources);

	private:
		bool time_is_hours_;
		int flow_rows_, num_sources_;
		std::vector<std::vector<T> > data_;
	};


	/* --------------------------------------------------------------------------- */

	template<class T>
	hydrograph<T>::hydrograph()
	{
		time_is_hours_ = true;
		set_num_sources(1);
	}

	/* --------------------------------------------------------------------------- */

	template<class T>
	hydrograph<T>::hydrograph(std::string filename)
	{
		time_is_hours_ = true;
		set_num_sources(1);
		load_from_file(filename);
		set_num_flow_rows(data_.size());
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int hydrograph<T>::get_num_inflow_rows()
	{
		return flow_rows_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	int hydrograph<T>::get_num_inflows()
	{
		return num_sources_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::set_num_flow_rows(int rows)
	{
		flow_rows_ = rows;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::set_num_sources(int sources)
	{
		num_sources_ = sources;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::load_from_file(std::string filename)
	{
		std::ifstream ifs(filename.c_str());
		if (!ifs.good())
		{
			std::cerr << ERROR "Error reading file: " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		std::vector<std::vector<T> > hygdata;

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

		if(data_.size() > 0)
		{
			set_num_sources(data_[data_.size() - 1].size() - 1);
		}
		else
		{
			set_num_sources(0);
		}

	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	std::vector<std::vector<T>> hydrograph<T>::get_rows()
	{
		return data_;
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	T hydrograph<T>::get_time_at(int index)
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
	T hydrograph<T>::get_flow_at(int index, int source_num)
	{
		if (index >= static_cast<int>(data_.size()))
		{
			std::cerr << ERROR "Extbc index out of bounds" << std::endl;
			exit(EXIT_FAILURE);

		}

		return (data_.at(index)).at(source_num);
	}

	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::convert_time_hr_to_secs()
	{
		if (time_is_hours_)
		{
			typename std::vector<std::vector<T>>::iterator it = data_.begin();

			for (; it != data_.end(); it++)
			{
				(*it)[0] = it->at(0) * HOUR_TO_SEC_FACTOR;
			}

			time_is_hours_ = false;
		}
	}


	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::convert_rate_hr_to_secs()
	{
		for (int i = 0; i < get_num_inflows(); ++i)
		{
			for (int j = 0; j < get_num_inflow_rows(); j++)
			{
				data_[j][i + 1] /= HOUR_TO_SEC_FACTOR;
			}
		}

	}



	/* --------------------------------------------------------------------------- */

	template<typename T>
	void hydrograph<T>::convert_rate_mm_to_m()
	{
		for (int i = 0; i < get_num_inflows(); ++i)
		{
			for (int j = 0; j < get_num_inflow_rows(); j++)
			{
				data_[j][i + 1] *= MM_TO_M_FACTOR;
			}
		}

	}

}

#endif
