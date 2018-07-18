#ifndef SSJ_INPUT_H
#define SSJ_INPUT_H

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

template <typename Tokenizer>
void get_int_records(const std::string & filename, Algorithm * algo, tokencountcounterhash & tokencounthash, const bool write_frequencies, Tokenizer & tokenizer) {

	char line[LINEBUFSIZE];
	std::ifstream inputfile;

	inputfile.open(filename.c_str());
	if( inputfile.is_open()) {
		if(tokencounthash.size() == 0) {
			while (inputfile.getline(line, LINEBUFSIZE)) {
				tokenizer.setline(line);
				update_counting_line ucl;
				while(!tokenizer.end()) {
					const char * token = tokenizer.next();
					ucl.update(token, tokencounthash);
				}
			}
		}
		if(!write_frequencies) {
			token2int(tokencounthash);
			inputfile.clear();
			inputfile.seekg(0);
			while (inputfile.getline(line, LINEBUFSIZE)) {
				tokenizer.setline(line);
				collect_sets_line csl(algo, tokencounthash);
				while(!tokenizer.end()) {
					const char * token = tokenizer.next();
					csl.update(token);
				}
			}
		}
	} else {
		perror("could not open file");
		abort();
	}
}

struct get_int_records_fast {
	// hash to store integer
	tokencountcounterhash thash;
	Algorithm * algo;
	add_int_records air;

	get_int_records_fast(Algorithm * algo) : algo(algo), air(algo) {
		thash.set_empty_key(std::string(""));
	}
	
	template <typename Tokenizer>
	void start(const std::string & filename, Tokenizer & tokenizer) {

		std::string line;
		std::ifstream inputfile;

		inputfile.open(filename.c_str());

		if( inputfile.is_open()) {
			while (getline(inputfile, line)) {

				tokenizer.setline(line.c_str());

				while(!tokenizer.end()) {
					const char * token = tokenizer.next();
					air.update(token, thash);
				}
				air.nextrecord();
			}
		} else {
			perror("could not open file");
			abort();
		}
	}

	template <typename Tokenizer>
	void foreignsets(const std::string & filename, Tokenizer & tokenizer) {

		std::string line;
		std::ifstream inputfile;

		inputfile.open(filename.c_str());

		if( inputfile.is_open()) {
			while (getline(inputfile, line)) {
				tokenizer.setline(line.c_str());
				while(!tokenizer.end()) {
					const char * token = tokenizer.next();
					air.assign(token, thash);
				}
				air.nextforeignrecord();
			}
		} else {
			perror("could not open file");
			abort();
		}
	}
};

struct algoaddrecord {
	void operator()(Algorithm * algo, IntRecord & record) {
		algo->addrawrecord(record);
	}
};

struct algoaddforeignrecord {
	void operator()(Algorithm * algo, IntRecord & record) {
		algo->addrawforeignrecord(record);
	}
};

template<class record_add_class>
struct add_raw_input {
	void add_input(Algorithm * algo, const std::string & filename, record_add_class rac);
};

#endif
