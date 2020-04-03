/** @file ConfigUtils.h
 *  @brief Header containing the ConfigUtils class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for ConfigUtils class
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

#ifndef CONFIG_UTILS_H
#define CONFIG_UTILS_H

#include "string_utils.h"

namespace ConfigUtils
{
	template<typename T>
	struct arguments
	{
		bool
			time_increment_fixed,
			time_series_flag,
			gpu_direct_flag;

		int
			checkpoint_id,
			num_sources,
			num_runoffs,
			num_extbc,
			it_count;

		T
			time_step,
			sim_start_time,
			sim_duration,
			print_interval,
			courant,
			const_mann,
			hextra;

		std::string
			outfile_pattern,
			hydrograph_filename,
			runoff_filename,
			print_option,
			input_format,
			output_format,
			output_option,
			dem_filename,
			src_loc_file,
			runoff_map,
			observation_loc_file,
			extbc_file,
			extbc_dir,
			h_infile,
			qx_infile,
			qy_infile,
			n_infile;

		std::vector<T>
			src_x_loc,
			src_y_loc,
			observation_x_loc,
			observation_y_loc,
			extbc_x1_loc,
			extbc_y1_loc,
			extbc_x2_loc,
			extbc_y2_loc;
		
		std::vector<int>
			extbc_bctype;
			std::vector<std::string>
			extbc_fname;

	};

	/* --------------------------------------------------------------------------- */

	std::string argsd(std::string x, std::map<std::string, std::string> y, std::string d);
	std::string args(std::string x, std::map<std::string, std::string> y);
	std::map<std::string, std::string> parse_cfg(std::string cfg_content);
	std::map<std::string, std::string> parse_src_location(std::string filename, int type);
	std::map<std::string, std::string> parse_extbc_file(std::string filename, std::string dir);
	template<typename T>
	arguments<T> get_args(std::string cfg);
	std::string file_content_to_string(std::string filepath);
	std::string get_root_dir(const char* path);

	/* --------------------------------------------------------------------------- */

	std::string argsd(std::string x, std::map<std::string, std::string> y, std::string d)
	{
		if(y.find(x) != y.end())
		{
			return (y.find(x))->second;
		}
		else
		{
			return d;
		}
	}

	/* --------------------------------------------------------------------------- */

	std::string args(std::string x, std::map<std::string, std::string> y)
	{
		return argsd(x, y, "");
	}

	/* --------------------------------------------------------------------------- */

	std::map<std::string, std::string> parse_cfg(std::string cfg_content)
	{
		std::map<std::string, std::string> arglist;
		std::istringstream ifs(cfg_content.c_str());
		std::string line;

		if (!ifs.good())
		{
			std::cerr << "Error parsing configuration content." << std::endl;
			exit(EXIT_FAILURE);
		}

		while (std::getline(ifs, line))
		{
			line = StringUtils::trim(line);

			if (line.size() > 0 && line[0] != '#')
			{
				size_t incomm = line.find('#');
				if ((incomm != std::string::npos) && (line[incomm - 1] != '\'') && (line[incomm - 1] != '\"'))
				{
					line.erase(line.begin() + incomm);
				}
				std::vector<std::string> kv = StringUtils::split(line, '=');
				std::string key = kv[0];
				std::string value = kv[1];

				if (value[0] == '"' && value[value.size() - 1] == '"')
				{
					value.erase(value.begin());
					value.erase(value.end() - 1);
				}

				if (kv.size() > 2)
				{
					std::vector<std::string>::iterator it = kv.begin() + 2;
					for (; it != kv.end(); it++)
					{
						value += (*it);
					}
				}
				arglist.insert(std::pair<std::string, std::string>(key, value));
			}
		}	

		return arglist;
	}

	/* --------------------------------------------------------------------------- */

	std::map<std::string, std::string> parse_src_location(std::string filename, int type)
	{
		std::map<std::string, std::string> arglist;
		std::ifstream ifs(filename.c_str());
		std::string line;

		if (!ifs.good())
		{
			std::cerr << ERROR "Error reading file: " << filename << std::endl;
			exit(EXIT_FAILURE);
		}
		std::string src_x_loc = "", src_y_loc = "";

		while (std::getline(ifs, line))
		{
			line = StringUtils::trim(line);

			if (line.size() > 0 && line[0] != '%')
			{
				size_t incomm = line.find('#');
				if ((incomm != std::string::npos) && (line[incomm - 1] != '\'') && (line[incomm - 1] != '\"'))
				{
					line.erase(line.begin() + incomm);
				}
				std::vector<std::string> kv = StringUtils::split(line, ',');
				std::string x_value = kv[0];
				std::string y_value = kv[1];

				if (x_value[0] == '"' && x_value[x_value.size() - 1] == '"')
				{
					x_value.erase(x_value.begin());
					x_value.erase(x_value.end() - 1);
				}
				if (y_value[0] == '"' && y_value[y_value.size() - 1] == '"')
				{
					y_value.erase(y_value.begin());
					y_value.erase(y_value.end() - 1);
				}

				src_x_loc = src_x_loc + "," + x_value;
				src_y_loc = src_y_loc + "," + y_value;
			}
		}

		src_x_loc = src_x_loc.substr(1);
		src_y_loc = src_y_loc.substr(1);

		if (type == SRC_LOCATION)
		{
			arglist.insert(std::pair<std::string, std::string>("src_x_loc", src_x_loc));
			arglist.insert(std::pair<std::string, std::string>("src_y_loc", src_y_loc));
		}
		else if (type == OBSERVATION_LOCATION)
		{
			arglist.insert(std::pair<std::string, std::string>("observation_x_loc", src_x_loc));
			arglist.insert(std::pair<std::string, std::string>("observation_y_loc", src_y_loc));
		}

		ifs.close();

		return arglist;
	}


/****************************************************************************************/

	std::map<std::string, std::string> parse_extbc_file(std::string filename, std::string dir)
	{
		std::map<std::string, std::string> arglist;
		std::ifstream ifs(filename.c_str());
		std::string line;

		if (!ifs.good())
		{
			std::cerr << ERROR "Error reading file: " << filename << std::endl;
			exit(EXIT_FAILURE);
		}

		std::string bctype = ""; 
		std::string src_x1_loc = "", src_y1_loc = "";
		std::string src_x2_loc = "", src_y2_loc = "";
		std::string bcfname = ""; 


		while (std::getline(ifs, line))
		{
			line = StringUtils::trim(line);

			if (line.size() > 0 && line[0] != '%')
			{
				size_t incomm = line.find('#');
				if ((incomm != std::string::npos) && (line[incomm - 1] != '\'') && (line[incomm - 1] != '\"'))
				{
					line.erase(line.begin() + incomm);
				}
				std::vector<std::string> kv = StringUtils::split(line, ',');
				std::string bctype_value = kv[0];
				std::string x1_value = kv[1];
				std::string y1_value = kv[2];
				std::string x2_value = kv[3];
				std::string y2_value = kv[4];
				
				int auxtype=std::stoi(kv[0]);
				std::string bcfname_value ="";
				if(auxtype!=0){
					bcfname_value = kv[5];
				}else{
					bcfname_value = "0.0";
				}

				if (bctype_value[0] == '"' && bctype_value[bctype_value.size() - 1] == '"')
				{
					bctype_value.erase(bctype_value.begin());
					bctype_value.erase(bctype_value.end() - 1);
				}

				if (x1_value[0] == '"' && x1_value[x1_value.size() - 1] == '"')
				{
					x1_value.erase(x1_value.begin());
					x1_value.erase(x1_value.end() - 1);
				}
				if (y1_value[0] == '"' && y1_value[y1_value.size() - 1] == '"')
				{
					y1_value.erase(y1_value.begin());
					y1_value.erase(y1_value.end() - 1);
				}

				if (x2_value[0] == '"' && x2_value[x2_value.size() - 1] == '"')
				{
					x2_value.erase(x2_value.begin());
					x2_value.erase(x2_value.end() - 1);
				}
				if (y2_value[0] == '"' && y2_value[y2_value.size() - 1] == '"')
				{
					y2_value.erase(y2_value.begin());
					y2_value.erase(y2_value.end() - 1);
				}
				
				if (bcfname_value[0] == '"' && bcfname_value[bcfname_value.size() - 1] == '"')
				{
					bcfname_value.erase(bcfname_value.begin());
					bcfname_value.erase(bcfname_value.end() - 1);
				}

				bctype = bctype + "," + bctype_value;
				src_x1_loc = src_x1_loc + "," + x1_value;
				src_y1_loc = src_y1_loc + "," + y1_value;
				src_x2_loc = src_x2_loc + "," + x2_value;
				src_y2_loc = src_y2_loc + "," + y2_value;
				
				if(auxtype==1){
					bcfname = bcfname + "," + dir + "/" + bcfname_value;
				}else{ //case 0, 2, 3
					bcfname = bcfname + "," + bcfname_value;
				}
			}
		}

		
		bctype = bctype.substr(1);
		src_x1_loc = src_x1_loc.substr(1);
		src_y1_loc = src_y1_loc.substr(1);
		src_x2_loc = src_x2_loc.substr(1);
		src_y2_loc = src_y2_loc.substr(1);
		bcfname = bcfname.substr(1);

		arglist.insert(std::pair<std::string, std::string>("extbc_bctype", bctype));
		arglist.insert(std::pair<std::string, std::string>("extbc_x1_loc", src_x1_loc));
		arglist.insert(std::pair<std::string, std::string>("extbc_y1_loc", src_y1_loc));
		arglist.insert(std::pair<std::string, std::string>("extbc_x2_loc", src_x2_loc));
		arglist.insert(std::pair<std::string, std::string>("extbc_y2_loc", src_y2_loc));
		arglist.insert(std::pair<std::string, std::string>("extbc_fname", bcfname));

		ifs.close();


		return arglist;
	}



	/* --------------------------------------------------------------------------- */

	template<typename T>
	arguments<T> get_args(std::string cfg)
	{
		std::map<std::string, std::string> argmap = parse_cfg(cfg);

		arguments<T> arglist;

		arglist.outfile_pattern = argsd("outfile_pattern", argmap, "");
		arglist.hydrograph_filename = args("hydrograph_filename", argmap);
		arglist.runoff_filename = args("runoff_filename", argmap);
		arglist.dem_filename = args("dem_filename", argmap);
		arglist.src_loc_file = argsd("src_loc_file", argmap, "");
		arglist.runoff_map = argsd("runoff_map", argmap, "");
		arglist.observation_loc_file = argsd("observation_loc_file", argmap, "");
		arglist.extbc_file = argsd("extbc_file", argmap, "");
		arglist.extbc_dir = argsd("extbc_dir", argmap, "");

		arglist.const_mann = atof((args("const_mann", argmap)).c_str());
		arglist.time_step = atof((args("time_step", argmap)).c_str());
		arglist.time_increment_fixed = atoi((args("time_increment_fixed", argmap)).c_str());
		arglist.time_series_flag = atoi((args("time_series_flag", argmap)).c_str());
		arglist.courant = atof((args("courant", argmap)).c_str());
		arglist.hextra = atof((args("hextra", argmap)).c_str());
		arglist.gpu_direct_flag = atoi((args("gpu_direct_flag", argmap)).c_str());

		arglist.num_sources = atoi((args("num_sources", argmap)).c_str());
		arglist.num_runoffs = atoi((args("num_runoffs", argmap)).c_str());
		arglist.num_extbc = atoi((args("num_extbc", argmap)).c_str());
		arglist.checkpoint_id = atoi((argsd("checkpoint_id", argmap, "0")).c_str());
		arglist.it_count = atoi((argsd("it_count", argmap, "0")).c_str());
		arglist.print_option = StringUtils::tolower(args("print_option", argmap));
		arglist.input_format = args("input_format", argmap);
		arglist.output_format = args("output_format", argmap);
		arglist.output_option = args("output_option", argmap);

		arglist.h_infile = argsd("h_infile", argmap, "");
		arglist.qx_infile = argsd("qx_infile", argmap, "");
		arglist.qy_infile = argsd("qy_infile", argmap, "");
		arglist.n_infile = argsd("n_infile", argmap, "");

		arglist.sim_start_time = atof((args("sim_start_time", argmap)).c_str());
		arglist.sim_duration = atof((args("sim_duration", argmap)).c_str());
		arglist.print_interval = atof((args("print_interval", argmap)).c_str());

		if(arglist.num_sources > 0)
		{
			std::map<std::string, std::string> src_map = parse_src_location(arglist.src_loc_file, SRC_LOCATION);
			arglist.src_x_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("src_x_loc", src_map)), ','));
			arglist.src_y_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("src_y_loc", src_map)), ','));
		}

		if (arglist.time_series_flag)
		{
			std::map<std::string, std::string> observation_loc_map = parse_src_location(arglist.observation_loc_file, OBSERVATION_LOCATION);
			arglist.observation_x_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("observation_x_loc", observation_loc_map)), ','));
			arglist.observation_y_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("observation_y_loc", observation_loc_map)), ','));
		}

		//external BC
		if (arglist.num_extbc > 0)
		{
			std::map<std::string, std::string> extbc_map = parse_extbc_file(arglist.extbc_file,arglist.extbc_dir);
			arglist.extbc_bctype = StringUtils::vecstr_to_vecint(StringUtils::split((args("extbc_bctype", extbc_map)), ','));
			arglist.extbc_x1_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("extbc_x1_loc", extbc_map)), ','));
			arglist.extbc_y1_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("extbc_y1_loc", extbc_map)), ','));
			arglist.extbc_x2_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("extbc_x2_loc", extbc_map)), ','));
			arglist.extbc_y2_loc = StringUtils::vecstr_to_vecflt<T>(StringUtils::split((args("extbc_y2_loc", extbc_map)), ','));
			arglist.extbc_fname = StringUtils::split((args("extbc_fname", extbc_map)), ',');
		}

		return arglist;
	}

	/* --------------------------------------------------------------------------- */

	std::string file_content_to_string(std::string filepath)
	{
		std::string file_content;
		std::ifstream input(filepath);
		if (input.is_open())
		{
			input.seekg(0, std::ios::end);
			file_content.reserve(input.tellg());
			input.seekg(0, std::ios::beg);
			file_content.assign((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
			input.close();
		}
		else
		{
			std::cerr << ERROR "Error reading file: " << filepath << std::endl;
			exit(EXIT_FAILURE);
		}



		return file_content;
	}
	
	std::string get_root_dir(const char* path)
	{
		std::string spath(path);
		int found = spath.find_last_of("/\\");
		spath = spath.substr(0, found);

		return spath.substr(0, (spath.find_last_of("/\\")));
	}
}

#endif
