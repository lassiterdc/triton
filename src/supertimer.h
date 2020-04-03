/** @file SuperTimer.h
 *  @brief Header containing the SuperTimer class
 *
 *  This contains the subroutines and eventually any 
 *  macros, constants, etc. needed for SuperTimer class
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



#ifndef SUPERTIMER_H
#define SUPERTIMER_H

#include "constants.h"

namespace SuperTimer
{
	struct ci_less
	{
		struct nocase_compare
		{
			bool operator() (const unsigned char& c1, const unsigned char& c2) const
			{
				return tolower(c1) < tolower(c2);
			}
		};
		
		bool operator() (const std::string & s1, const std::string & s2) const
		{
			return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(), nocase_compare());
		}
	};

	/* --------------------------------------------------------------------------- */

	class super_timer
	{
	public:
		super_timer();

		void start(std::string category);
		void stop(std::string category);
		void reset();

		double get_total_time();
		double get_custom_time(std::string category);

		std::string get_current_date();
		std::string get_hostname();

		int add_new_timer(std::string category);

	private:
		int units_;
		double unit_factor_;
		std::vector<timeval> all_timers_;
		std::map<std::string, int, ci_less> timecats_;
		std::vector<Constants::ull> times_;

		void upd_time_(std::string category, Constants::ull time);
		int get_cat_index_(std::string category);
		void init_();
		void upd_time_(int timer, Constants::ull time);
		double convert_(Constants::ull time);
	};

	/* --------------------------------------------------------------------------- */

	super_timer::super_timer() : units_(TIMER_SECS)
	{
		init_();
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::init_()
	{
		unit_factor_ = 1;
		if(units_ == TIMER_SECS)
		{
			unit_factor_ = 0.000001;
		}
		else if (units_ == TIMER_NSECS)
		{
			unit_factor_ = 1000;
		}
	}

	/* --------------------------------------------------------------------------- */

	int super_timer::add_new_timer(std::string category)
	{
		int ncats = timecats_.size();
		std::map<std::string, int, ci_less>::iterator it = timecats_.find(category);
		if (it == timecats_.end())
		{
			timeval ntval;
			Constants::ull newcat_time = 0;

			timecats_.insert(std::pair<std::string, int>(category, ncats));
			times_.push_back(newcat_time);
			all_timers_.push_back(ntval);
		}

		return timecats_.size() - 1;
	}

	/* --------------------------------------------------------------------------- */

	int super_timer::get_cat_index_(std::string category)
	{
		int catidx;
		std::map<std::string, int, ci_less>::iterator it = timecats_.find(category);

		if (it != timecats_.end())
		{
			catidx = it->second;
		}
		else
		{
			add_new_timer(category);
			catidx = add_new_timer(category);
		}

		return catidx;
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::upd_time_(int catid, Constants::ull time)
	{
		if (catid < (int)times_.size())
		{
			times_[catid] += time;
		}
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::upd_time_(std::string category, Constants::ull time)
	{
		times_[get_cat_index_(category)] += time;
	}

	/* --------------------------------------------------------------------------- */

	double super_timer::convert_(Constants::ull time)
	{
		return (double)time * unit_factor_;
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::reset()
	{
		all_timers_.clear();
		times_.clear();

		init_();
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::start(std::string category)
	{
		timeval* t = &all_timers_[get_cat_index_(category)];

		gettimeofday(t, NULL);
	}

	/* --------------------------------------------------------------------------- */

	void super_timer::stop(std::string category)
	{
		timeval t1, t2;
		int catidx = get_cat_index_(category);
		gettimeofday(&t2, NULL);
		t1 = all_timers_[catidx];

		Constants::ull time_usecs = ((((Constants::ull) t2.tv_sec * 1000000) + t2.tv_usec) - (((Constants::ull) t1.tv_sec * 1000000) + t1.tv_usec));

		upd_time_(category, time_usecs);
	}

	/* --------------------------------------------------------------------------- */

	double super_timer::get_custom_time(std::string category)
	{
		return convert_(times_[get_cat_index_(category)]);
	}

	/* --------------------------------------------------------------------------- */

	double super_timer::get_total_time()
	{
		double total = 0.0;
		std::map<std::string, int>::iterator it = timecats_.begin();

		for (it = timecats_.begin(); it != timecats_.end(); it++)
		{
			total += get_custom_time(it->first);
		}

		return total;
	}

	/* --------------------------------------------------------------------------- */

	std::string super_timer::get_current_date()
	{
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[24];

		time(&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer, 24, "%D", timeinfo);

		return std::string(buffer);
	}

	/* --------------------------------------------------------------------------- */

	std::string super_timer::get_hostname()
	{
		char hostname[1024];
		hostname[1023] = '\0';
		gethostname(hostname, 1023);

		return std::string(hostname);
	}

}

#endif
