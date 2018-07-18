#ifndef SSJ_CMDLINE_MPJOIN_H
#define SSJ_CMDLINE_MPJOIN_H

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

#include <boost/mpl/vector/vector10.hpp>

#include "cmdline.h"

#include "mpjoin.h"

#include "similarity.h"
#include "lengthfilter.h"
#include "ppjoinpolicies.h"

#include "template_unroll.h"


template <typename Similarity>
class MpJoin_Unroll {

	private:

		typedef boost::mpl::vector2<PPJoinIndexPolicy,
#ifndef MPJ_LINKEDLIST
				MpJoinArrayLinkedIndexPolicy
#else
				MpJoinLinkedListIndexPolicy
#endif
				> MpJoinIndexStructures;
		typedef boost::mpl::vector3<IndexOnTheFlyPolicy, IndexFirstPolicy<false>, IndexFirstPolicy<true> > MpJoinIndexingStategies;
		typedef boost::mpl::vector2<DefaultLengthFilterPolicy, PlJoinLengthFilterPolicy> MpJoinLengthFilterPolicies;
		typedef boost::mpl::vector2<DisabledPPFilterPolicy, PPJoinPlusPolicy> MpJoinPPFilterPolicies;

		struct Execute {
			struct algoParams {
				double threshold;
			};
			template <class Class1, class Class2, class Class3, class Class4>
				static Algorithm * get_algo( 
						std::vector<bool> & d1Pattern,
						std::vector<bool> & d2Pattern, 
						std::vector<bool> & d3Pattern, 
						std::vector<bool> & d4Pattern,
						unsigned int c1distance,
						unsigned int c2distance,
						unsigned int c3distance,
						unsigned int c4distance,
						algoParams algo_params) {


					if(d1Pattern[c1distance] && d2Pattern[c2distance] && d3Pattern[c3distance] && d4Pattern[c4distance] ) {
						return new MpJoin<Similarity, Class1, Class2, Class3, Class4 >(algo_params.threshold);
					} else {
						return NULL;
					}
				}
		};
	public:
		static Algorithm * mpjoin_cmd_line(boost::program_options::variables_map & vm);
};

Algorithm * mpjoin_cmd_line(boost::program_options::variables_map & vm);

#endif
