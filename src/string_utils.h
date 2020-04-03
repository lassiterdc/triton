/** @file StringUtils.h
 *  @brief Header containing the StringUtils class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for StringUtils class
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



#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include "constants.h"

namespace StringUtils
{
	Constants::char_t up_char(Constants::char_t ch);
	std::string toupper(const std::string &src);
	Constants::char_t down_char(Constants::char_t ch);
	std::string tolower(const std::string &src);
	bool is_numeric(const std::string& str);

	Constants::string_vector &split(const std::string &s, char delim, Constants::string_vector &elems);
	Constants::string_vector split(const std::string &s, char delim);
	std::vector<int> vecstr_to_vecint(std::vector<std::string> vs);
	template <typename T>
	std::vector<T> vecstr_to_vecflt(Constants::string_vector vs);

	std::string itoa(int i);
	std::string itos(int num);

	/* --------------------------------------------------------------------------- */

	static inline std::string& ltrim(std::string &s)
	{
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
		return s;
	}

	/* --------------------------------------------------------------------------- */

	static inline std::string& rtrim(std::string &s)
	{
		s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
		return s;
	}

	/* --------------------------------------------------------------------------- */

	static inline std::string& trim(std::string &s)
	{
		return ltrim(rtrim(s));
	}

	/* --------------------------------------------------------------------------- */

	Constants::char_t up_char(Constants::char_t ch)
	{
		return std::use_facet<std::ctype<Constants::char_t>>(std::locale()).toupper(ch);
	}

	/* --------------------------------------------------------------------------- */

	std::string toupper(const std::string &src)
	{
		std::string result;
		std::transform(src.begin(), src.end(), std::back_inserter(result), up_char);
		return result;
	}

	/* --------------------------------------------------------------------------- */

	Constants::char_t down_char(Constants::char_t ch)
	{
		return std::use_facet<std::ctype<Constants::char_t>>(std::locale()).tolower(ch);
	}

	/* --------------------------------------------------------------------------- */

	std::string tolower(const std::string &src)
	{
		std::string result;
		std::transform(src.begin(), src.end(), std::back_inserter(result), down_char);
		return result;
	}

	/* --------------------------------------------------------------------------- */

	bool is_numeric(const std::string& str)
	{
		if (str.size() == 0)
			return false;

		std::stringstream conv;
		double tmp;
		conv << str;
		conv >> tmp;
		return conv.eof();
	}

	/* --------------------------------------------------------------------------- */

	Constants::string_vector& split(const std::string &s, char delim, Constants::string_vector &elems)
	{
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim))
		{
			elems.push_back(item);
		}
		return elems;
	}

	/* --------------------------------------------------------------------------- */

	Constants::string_vector split(const std::string &s, char delim)
	{
		std::vector<std::string> elems;
		return split(s, delim, elems);
	}

	/* --------------------------------------------------------------------------- */

	std::vector<int> vecstr_to_vecint(Constants::string_vector vs)
	{
		std::vector<int> ret;
		for (Constants::string_vector::iterator it = vs.begin(); it != vs.end(); ++it)
		{
			std::istringstream iss(*it);
			int temp;
			iss >> temp;
			ret.push_back(temp);
		}
		return ret;
	}

	/* --------------------------------------------------------------------------- */

	template <typename T>
	std::vector<T> vecstr_to_vecflt(Constants::string_vector vs)
	{
		std::vector<T> ret;
		for (Constants::string_vector::iterator it = vs.begin(); it != vs.end(); ++it)
		{
			std::istringstream iss(*it);
			T temp;
			iss >> temp;
			ret.push_back(temp);
		}
		return ret;
	}

	/* --------------------------------------------------------------------------- */

	std::string itoa(int i)
	{
		return (static_cast<std::ostringstream*> (&(std::ostringstream() << i))->str());
	}

	/* --------------------------------------------------------------------------- */

	std::string itos(int num)
	{
		std::ostringstream ss;
		ss << num;
		return ss.str();
	}
}

#endif
