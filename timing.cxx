/* Copyright 2014-2015 Willi Mann
 *
 * This file is part of set_sim_join.
 *
 * set_sim_join is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Foobar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with set_sim_join.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "timing.h"

Timing::Interval * Timing::add(const std::string & descriptor) {
	Interval * in = new Interval(descriptor);
	intervals.push_back(in);

	// getrusage data
	
	struct rusage ru;
	int ret = getrusage(RUSAGE_SELF, &ru);
	if(ret != 0) {
		perror("Timing call to getrusage");
		abort();
	}

	in->usertime = ru.ru_utime;
	in->systemtime = ru.ru_stime;
	
	return in;
}

void Timing::finish(Interval * interval) {

	// getrusage data
	
	struct rusage ru;
	int ret = getrusage(RUSAGE_SELF, &ru);
	if(ret != 0) {
		perror("Timing call to getrusage");
		abort();
	}

	struct timeval systemres, userres;

	timersub(&ru.ru_utime, &interval->usertime, &userres);
	timersub(&ru.ru_stime, &interval->systemtime, &systemres);

	interval->usertime = userres;
	interval->systemtime = systemres;
}

Timing::~Timing() {
	std::vector<Interval*>::iterator tit = intervals.begin();
	for(; tit != intervals.end(); ++tit) {
		delete *tit;
	}
}

std::ostream & operator<<(std::ostream & os, const Timing & timing) {

	std::vector<Timing::Interval*>::const_iterator tit = timing.intervals.begin();
	for(; tit != timing.intervals.end(); ++tit) {
	
		unsigned int millisecs = (*tit)->usertime.tv_usec / 1000 + (*tit)->usertime.tv_sec * 1000 +
			(*tit)->systemtime.tv_usec / 1000 + (*tit)->systemtime.tv_sec * 1000;
	
		os << std::setw(16) << (*tit)->descriptor << std::setw(11) << millisecs << " ms" << std::endl;
	}
	return os;
}
