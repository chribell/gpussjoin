#ifndef SSJ_TIMING_H
#define SSJ_TIMING_H

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

#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <string>

class Timing {
	public:
		struct Interval {
			std::string descriptor;
			struct timeval usertime;
			struct timeval systemtime;
			Interval(const std::string & descriptor) : descriptor(descriptor) {}
		};
	protected:
		std::vector<Interval*> intervals;

	public:
		Interval * add(const std::string & descriptor);
		void finish(Interval * interval);
		~Timing();
		friend std::ostream & operator<<(std::ostream & os, const Timing & timing);

};

#endif
