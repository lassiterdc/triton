/** @file test_supertimer_order.cpp
 *  @brief Chunk 2b.1 repair test: super_timer::start() reads the clock before it looks the
 *         category up, and super_timer::stop() looks the category up once, before its
 *         clock read.
 *
 *  Why. A category lookup between one timer call's clock read and the next call's clock
 *  read is time that belongs to no category and surfaces in Other. The old order (start:
 *  lookup then clock; stop: lookup, clock, then a second lookup inside upd_time_) left
 *  about 2.5 to 4.5 microseconds per category change in Other (measured on Rivanna).
 *
 *  Class: REGRESSION. T1 and T2 are red on the old bodies (control T3 and T4 run the same
 *  checks on the pre-repair text, embedded below, and require them to FAIL). It is a
 *  source-structure test: it reads src/supertimer.h as text with comments removed and does
 *  not time anything. Not established by it: the size of the effect (measured in the
 *  neutrality case, not here). Second correct implementation: start() storing the clock
 *  into a local and assigning it after the lookup passes; so does a stop() that caches the
 *  index in a variable and calls the int overload of upd_time_ after the clock read ONLY if
 *  that call is the single lookup, which this test does not allow (it requires no upd_time_
 *  call in stop()), so that variant is deliberately outside it.
 *  One-time chunk test: retire at the run's final landing.
 *
 *  Usage: test_supertimer_order <path to src/supertimer.h> [test name]
 */
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL %s\n", m); failures++; } } while (0)

/** @brief The text with // and slash-star comments removed. */
static std::string strip_comments(const std::string& s)
{
	std::string o;
	for (size_t i = 0; i < s.size(); i++) {
		if (s.compare(i, 2, "//") == 0) { while (i < s.size() && s[i] != '\n') i++; o += '\n'; }
		else if (s.compare(i, 2, "/*") == 0) { size_t e = s.find("*/", i + 2); i = (e == std::string::npos) ? s.size() : e + 1; }
		else o += s[i];
	}
	return o;
}

/** @brief The body of `void super_timer::{name}(` up to the next `void super_timer::`. */
static std::string body_of(const std::string& text, const std::string& name)
{
	const std::string key = "void super_timer::" + name + "(";
	size_t a = text.find(key);
	if (a == std::string::npos) return "";
	size_t b = text.find("void super_timer::", a + key.size());
	return text.substr(a, b == std::string::npos ? std::string::npos : b - a);
}

static int count_of(const std::string& s, const std::string& t)
{
	int n = 0;
	for (size_t p = s.find(t); p != std::string::npos; p = s.find(t, p + t.size())) n++;
	return n;
}

/** @brief True when start() reads the clock before it looks the category up. */
static bool start_reads_clock_first(const std::string& text)
{
	const std::string b = body_of(text, "start");
	const size_t clock = b.find("gettimeofday"), look = b.find("get_cat_index_");
	return clock != std::string::npos && look != std::string::npos && clock < look;
}

/** @brief True when stop() looks the category up exactly once, before its clock read, and
 *  makes no second lookup through upd_time_. */
static bool stop_looks_up_once_before_clock(const std::string& text)
{
	const std::string b = body_of(text, "stop");
	const size_t clock = b.find("gettimeofday"), look = b.find("get_cat_index_");
	return clock != std::string::npos && look != std::string::npos && look < clock
	       && count_of(b, "get_cat_index_") == 1 && b.find("upd_time_") == std::string::npos;
}

static const char* OLD_TEXT =
	"	void super_timer::start(std::string category)\n"
	"	{\n"
	"		timeval* t = &all_timers_[get_cat_index_(category)];\n"
	"		gettimeofday(t, NULL);\n"
	"	}\n"
	"	void super_timer::stop(std::string category)\n"
	"	{\n"
	"		timeval t1, t2;\n"
	"		int catidx = get_cat_index_(category);\n"
	"		gettimeofday(&t2, NULL);\n"
	"		t1 = all_timers_[catidx];\n"
	"		upd_time_(category, time_usecs);\n"
	"	}\n"
	"	void super_timer::restart(std::string category) {}\n";

int main(int argc, char** argv)
{
	if (argc < 2) { std::printf("usage: %s <src/supertimer.h> [test]\n", argv[0]); return 2; }
	std::ifstream in(argv[1]);
	std::stringstream ss; ss << in.rdbuf();
	const std::string live = strip_comments(ss.str());
	const std::string which = argc > 2 ? argv[2] : "all";
	auto want = [&](const char* t) { return which == "all" || which == t; };

	CHECK(!live.empty() && !body_of(live, "start").empty() && !body_of(live, "stop").empty(), "src/supertimer.h holds no start() or stop() body: the checks would be vacuous");
	if (want("T1_start_reads_the_clock_before_the_lookup"))
		CHECK(start_reads_clock_first(live), "T1 start() looks the category up before it reads the clock");
	if (want("T2_stop_looks_up_once_before_the_clock"))
		CHECK(stop_looks_up_once_before_clock(live), "T2 stop() does not look up exactly once before its clock read");
	if (want("T3_control_old_start_is_flagged"))
		CHECK(!start_reads_clock_first(OLD_TEXT), "T3 control: the pre-repair start() was not flagged");
	if (want("T4_control_old_stop_is_flagged"))
		CHECK(!stop_looks_up_once_before_clock(OLD_TEXT), "T4 control: the pre-repair stop() was not flagged");

	if (failures) return 1;
	std::printf("PASS %s\n", which.c_str());
	return 0;
}
