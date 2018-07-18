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
#include <fstream>
#include <string>
#include <vector>
#include "data.h"
#include "classes.h"
#include "input.h"

	
void read_tokenfrequency(const std::string & filename, tokencountcounterhash & tk)
{
	std::ifstream inputfile;
	std::string line;

	inputfile.open(filename.c_str());
	if( inputfile.is_open()) {
		while(true) {
			unsigned int tokenfrequency, count;
			inputfile >> tokenfrequency;
			inputfile >> count;
			if(!inputfile.good()) {
				break;
			}
			inputfile.get();
			std::getline(inputfile, line);
			tokencount tc(line, count);
			tk[tc] = tokenfrequency;
			line = "";
		}
	} else {
		perror("could not open file");
		abort();
	}
}

void write_tokenfrequency(const std::string & filename, const tokencountcounterhash & tk)
{
	std::ofstream outputfile;
	std::string line;

	outputfile.open(filename.c_str());
	if( outputfile.is_open()) {
		tokencountcounterhash::const_iterator tit = tk.begin();
		for(; tit != tk.end(); ++tit) {
			outputfile << tit->second << " " << tit->first.count << " " << tit->first.token << std::endl;
		}
	} else {
		perror("could not open file for writing");
		abort();
	}
}

template<class record_add_class>
void add_raw_input<record_add_class>::add_input(Algorithm * algo, const std::string & filename, record_add_class rac) {
	std::ifstream infile;
	std::string line;
	infile.open(filename.c_str());
	if(infile.is_open()) {
		tokenize_whitespace tw;
		while(getline(infile, line)) {
			IntRecord rec;
			tw.setline(line.c_str());
			while(!tw.end()) {
				const char * token = tw.next();
				unsigned int nmb = atoi(token);
				rec.tokens.push_back(nmb);
			}
			//TODO: Error handling
			rac(algo, rec);
		}
	} else {
		perror("could not open file");
		exit(7);
	}
	infile.close();
}

template class add_raw_input<algoaddrecord>;
template class add_raw_input<algoaddforeignrecord>;
