#ifndef SSJ_CMDLINE_H
#define SSJ_CMDLINE_H

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

#include <boost/program_options.hpp>
#include "classes.h"


Algorithm * mpjoin_cmd_line(boost::program_options::variables_map & vm);
Algorithm * groupjoin_cmd_line(boost::program_options::variables_map & vm);
Algorithm * adaptjoin_cmd_line(boost::program_options::variables_map & vm);
Algorithm * allpairs_cmd_line(boost::program_options::variables_map & vm);

#endif
