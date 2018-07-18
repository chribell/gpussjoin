#include "cmdline_mpjoin.h"

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

template<typename Similarity>
Algorithm * MpJoin_Unroll<Similarity>::mpjoin_cmd_line(boost::program_options::variables_map & vm) {

	std::vector<bool> mpJoinIndexStructurePattern;
	std::vector<bool> mpJoinIndexingStrategyPattern;
	std::vector<bool> lengthFilterPolicyPattern;
	std::vector<bool> ppfilterPolicyPattern;

	mpJoinIndexStructurePattern.resize(4);
	mpJoinIndexingStrategyPattern.resize(3);
	lengthFilterPolicyPattern.resize(2);
	ppfilterPolicyPattern.resize(2);

	std::string cmd_algo = vm["algorithm"].as<std::string>();

	if(vm.count("mpjoin")) {
		/*if(vm.count("indexm2")) {
		  mpJoinIndexStructurePattern[1] = true;
		  } else if(vm.count("indexm3"))  {
		  mpJoinIndexStructurePattern[2] = true;
		  } else {
		  mpJoinIndexStructurePattern[3] = true;
		  }*/
		mpJoinIndexStructurePattern[1] = true;

	} else {
		//PPjoin structures
		mpJoinIndexStructurePattern[0] = true;
	}

	if(vm.count("foreign-linewise")) {
		mpJoinIndexingStrategyPattern[2] = 1;
	} else if(vm.count("indexfirst")) {
		mpJoinIndexingStrategyPattern[1] = 1;
	} else {
		mpJoinIndexingStrategyPattern[0] = 1;
	}


	if(vm.count("pljoin")) {
		lengthFilterPolicyPattern[1] = 1;
	} else {
		lengthFilterPolicyPattern[0] = 1;
	}

	if (vm.count("suffixfilter")) {
		ppfilterPolicyPattern[1] = 1;
	} else {
		ppfilterPolicyPattern[0] = 1;
	}

	double threshold = vm["threshold"].as<double>();

	typename Execute::algoParams algo_params;
	algo_params.threshold = threshold;

	return algo_template_unroll_d4::combine<
		MpJoinIndexStructures,
		MpJoinIndexingStategies,
		MpJoinLengthFilterPolicies,
		MpJoinPPFilterPolicies,
		Execute>::template Generate<>::get_algo(
				mpJoinIndexStructurePattern,
				mpJoinIndexingStrategyPattern,
				lengthFilterPolicyPattern,
				ppfilterPolicyPattern,
				algo_params);

}
