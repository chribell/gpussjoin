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

#include <string>
#include <sparsehash/dense_hash_map>
#include "classes.h"

struct eqstr
{
  bool operator()(const std::string & s1, const std::string & s2) const
  {
    return (&s1 == &s2) || (s1 == s2);
  }
};

void intify(std::vector<Record<std::string> > inrecords, std::vector<Record<int> > outrecords)
{
	using google::dense_hash_map;

	dense_hash_map<std::string, int, eqstr> tokensint;
	
	int counter = 1;

	for (auto i : inrecords) {
		Record<int> outrecord;

		for (auto t : i.tokens) {
			 tint = tokensint[t];
			 if(tint == 0) {
				 tint = counter++;
				 tokensint[t] = tint;
			 }
			 outrecord.tokens.push_back(tint);
		}
		outrecords.push_back(outrecord);
	}
}
