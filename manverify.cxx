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

#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>

#include "verify.h"
#include "statistics.h"

ExtStatistics<RealIncreaserExt> extStatistics;

int main(int argc, char ** argv) {
	if(argc < 3) {
		std::cerr << "not enough arguments" << std::endl;
		exit(1);
	}
	typedef std::vector<unsigned int> recvec;
	std::vector<recvec> records;

	// Read records
	std::ifstream infile(argv[1]);
	std::string line;

	while(getline(infile, line)) {
		recvec record;
		std::istringstream oss(line);
		while(oss.good()) {
			unsigned int val;
			oss >> val;
			record.push_back(val);

		}
		records.push_back(record);
	}
	
	std::ifstream infile2(argv[2]);

	unsigned long resultcount = 0;
	unsigned long setsizepossum = 0;
	unsigned long setsizenegsum = 0;
	unsigned long setsizeposcnt = 0;
	unsigned long setsizenegcnt = 0;


	while(getline(infile2, line)) {
		recvec record;
		std::istringstream oss(line);
		while(oss.good()) {
			unsigned int recordid;
			oss >> recordid;
			unsigned int indexrecordid;
			oss >> indexrecordid;
			unsigned int minoverlap;
			oss >> minoverlap;
			unsigned int recpos;
			oss >> recpos;
			unsigned int indrecpos;
			oss >> indrecpos;
			unsigned int count;
			oss >> count;
			if(verifypair(records[recordid  ], records[indexrecordid ], minoverlap, recpos, indrecpos, count)) {
				setsizepossum += records[recordid ].size() + records[indexrecordid ].size();
				setsizeposcnt += 2;
				resultcount += 1;
			} else {
				setsizenegsum += records[recordid ].size() + records[indexrecordid ].size();
				setsizenegcnt += 2;
			}
		}
	}
	std::cout << "Result count: " << resultcount << std::endl;
	std::cout << "Extended statistics: " << extStatistics;
	std::cout << "Average set length positives: " << double(setsizepossum) / setsizeposcnt;
	std::cout << "Average set length negatives: " << double(setsizenegsum) / setsizenegcnt;
}
