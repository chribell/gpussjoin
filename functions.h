#include <string>
#include <vector>
#include "classes.h"
#include "data.h"

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

void get_int_records(const std::string & filename, IntRecords & records, tokencountcounterhash & tk, const bool write_frequencies);

void read_tokenfrequency(const std::string & filename, tokencountcounterhash & ext_tokencount);

void write_tokenfrequency(const std::string & filename, const tokencountcounterhash & ext_tokencount);
