#ifndef SSH_PPJOINPOLICIES_H
#define SSH_PPJOINPOLICIES_H

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

#include "classes.h"
#include "cpucycles.h"

class DisabledPPFilterPolicy {
	public:
		struct IndexedRecordData {
		};

		template<typename Algorithm>
		struct PostPrefixFilter {
			PostPrefixFilter(Algorithm * algo) {}
			inline void largest_index_set_size(unsigned int largestsetsize) {}
			inline void probe_record_compute(typename Algorithm::Index::ProbeRecord & proberecord, unsigned int maxprefixlen) {}
			inline void create_for_indexed_records(typename Algorithm::IndexedRecords & indexedrecords) {}
			inline bool check_probe_against_index(
					typename Algorithm::Index::ProbeRecord & proberecord,
					typename Algorithm::IndexedRecord & indexrecord,
					unsigned recind, unsigned int overlapthres,
					unsigned int reclen, unsigned int recpos,
					unsigned int indreclen, unsigned int indrecpos, unsigned int foundoverlap=1) { return true;}
			inline void probe_to_index(typename Algorithm::IndexedRecord & indexrecord) {}
			inline void probe_to_index(typename Algorithm::ForeignRecord & foreignrecord) {}
			inline void cleanup_postprefixfilterdata(typename Algorithm::IndexedRecords & indexedrecords) {}

		};
};



class PPJoinPlusPolicy {
	public:
		struct IndexedRecordData {
			unsigned int lastprobeid;
			IndexedRecordData() : lastprobeid(INT_MAX) {}
		};

		template<typename Algorithm>
		struct PostPrefixFilter {

			static const unsigned int MAXDEPTH=2;

#ifdef CYCLE_COUNT_SF
#define CC_BUCKETS_SF 10000
#define CC_BUCKETSIZE_SF 1
			unsigned long truesfcycles[CC_BUCKETS_SF];
			unsigned long falsesfcycles[CC_BUCKETS_SF];
#endif
			typedef typename Algorithm::_Statistics Statistics;
			Statistics & statistics;

			PostPrefixFilter(Algorithm * algo) : statistics(algo->statistics) {
#ifdef CYCLE_COUNT_SF
				for(unsigned int i = 0; i < CC_BUCKETS_SF; ++i) {
					truesfcycles[i] = falsesfcycles[i] = 0;
				}
#endif
			}
			inline void largest_index_set_size(unsigned int largestsetsize) {}

			inline void probe_record_compute(typename Algorithm::Index::ProbeRecord & proberecord, unsigned int maxprefixlen) {
			}

			inline void create_for_indexed_records(typename Algorithm::IndexedRecords & indexedrecords) {}
			inline void probe_to_index(typename Algorithm::IndexedRecord & indexrecord) {}
			inline void probe_to_index(typename Algorithm::ForeignRecord & foreignrecord) {}
			inline void cleanup_postprefixfilterdata(typename Algorithm::IndexedRecords & indexedrecords) {
#ifdef CYCLE_COUNT_SF
				//SUFFIX FILTER
				for(unsigned int i = 0; i < CC_BUCKETS_SF; ++i) {
					if(truesfcycles[i] != 0) {
						std::cout << "tccsf" << i << ":\t" << truesfcycles[i] << std::endl;
					}
				}
				for(unsigned int i = 0; i < CC_BUCKETS_SF; ++i) {
					if(falsesfcycles[i] != 0) {
						std::cout << "fccsf" << i << ":\t" << falsesfcycles[i] << std::endl;
					}
				}
#endif
			}
			unsigned int recSuffixFilter(
					const typename Algorithm::Index::ProbeRecord & proberecord,
					const typename Algorithm::IndexedRecord & indexrecord,
					const int recposleft, const int recposright,
					const int indrecposleft, const int indrecposright,
					const unsigned int hammingMax, const unsigned int recdepth) {

				int reclen = recposright - recposleft;
				int indreclen = indrecposright - indrecposleft;
				int lendiff = abs(reclen - indreclen);
				if(reclen == 0 || indreclen == 0) {
					// return abs(|x|-|y|)
					return lendiff;
				}

				int indrecmid = indrecposleft + indreclen / 2;
				unsigned int indmidelem = indexrecord.tokens[indrecmid];

				// The next following 4 assignments represent the first call to Partition from the Pseudo-Code
				unsigned int indrecp1left = indrecposleft;
				unsigned int indrecp1right = indrecmid;

				unsigned int indrecp2left = indrecmid + 1;
				unsigned int indrecp2right = indrecposright;

				//search region
				int recsearchleft = recposleft + indreclen / 2 - (hammingMax - lendiff) / 2;
				int recsearchright = recposleft + indreclen / 2 + (hammingMax - lendiff) / 2 + 1;

				// account for different length
				if( reclen < indreclen ) {
					recsearchleft -= lendiff;
				} else {
					recsearchright += lendiff;
				}

				if(recsearchleft >= recposleft) {
					// Only if computed search region is within actual partition (left side of record partition)
					if(proberecord.tokens[recsearchleft] > indmidelem) {
						return hammingMax + 1;
					}
				} else {
					// otherwise set left bound for binary search to left bound of partition
					recsearchleft = recposleft;
				}
				if(recsearchright <= recposright) {
					// Only if computed search region is within actual partition (right side of record partition)
					if(proberecord.tokens[recsearchright - 1] < indmidelem) {
						return hammingMax + 1;
					}
				} else {
					// otherwise set right bound for binary search to right bound of partition
					recsearchright = recposright;
				}

				unsigned int recsearchmiddle;
				//Binary search
				while(recsearchright != recsearchleft) {
					recsearchmiddle = (recsearchright + recsearchleft) / 2;
					if(proberecord.tokens[recsearchmiddle] < indmidelem) {
						recsearchleft = recsearchmiddle + 1;
					} else {
						recsearchright = recsearchmiddle;
					}
				}

				unsigned int diff = 1;
				unsigned int recp1left = recposleft;
				unsigned int recp1right = recsearchleft;

				unsigned int recp2left = recsearchleft;
				if(recsearchleft < recposright && proberecord.tokens[recsearchleft] == indmidelem) {
					recp2left = recsearchleft + 1;
					diff = 0;
				}
				unsigned int recp2right = recposright;

				int recp1len = recp1right - recp1left;
				int recp2len = recp2right - recp2left;

				int indrecp1len = indrecp1right - indrecp1left;
				int indrecp2len = indrecp2right - indrecp2left;

				int lendiffp1 = abs(recp1len - indrecp1len);
				int lendiffp2 = abs(recp2len - indrecp2len);

				unsigned int hamming = lendiffp1 + lendiffp2 + diff;

				if(recdepth == 1 || hamming > hammingMax) {
					return hamming;
				}

				unsigned int hammingp1 = recSuffixFilter(
						proberecord,
						indexrecord,
						recp1left, recp1right,
						indrecp1left, indrecp1right,
						hammingMax - lendiffp2 - diff, recdepth - 1);
				hamming = hammingp1 + lendiffp2 + diff;
				if(hamming > hammingMax) {
					return hamming;
				}

				unsigned int hammingp2 = recSuffixFilter(
						proberecord,
						indexrecord,
						recp2left, recp2right,
						indrecp2left, indrecp2right,
						hammingMax - hammingp1 - diff, recdepth - 1);

				return hammingp1 + hammingp2 + diff;
			}




			inline bool check_probe_against_index(typename Algorithm::Index::ProbeRecord & proberecord, typename Algorithm::IndexedRecord & indexrecord, unsigned int recind, unsigned int minoverlap, unsigned int reclen, unsigned int recpos, unsigned int indreclen, unsigned int indrecpos, unsigned int foundoverlap) {
#ifdef CYCLE_COUNT_SF
				unsigned long beforesf = cpu_cycles_start();
#endif
				if(indexrecord.postprefixfilterdata.lastprobeid == recind) {
					return true;
				}
				indexrecord.postprefixfilterdata.lastprobeid = recind;

				//maximum allowed hamming distance right of match
				// more explicit: reclen - (recpos + 1) + indreclen - (indrecpos + 1) - 2 * (minoverlap - 1)
				unsigned int hammingMaxR = reclen - recpos + indreclen - indrecpos - 2 * (minoverlap - foundoverlap + 1);
				unsigned int hamming = recSuffixFilter(
						proberecord,
						indexrecord,
						recpos + 1, reclen,
						indrecpos + 1, indreclen,
						hammingMaxR, MAXDEPTH);
#ifdef CYCLE_COUNT_SF
				unsigned long cycles_needed = cpu_cycles_stop() - beforesf;
				unsigned int bucket = cycles_needed / CC_BUCKETSIZE_SF;
				if(bucket >= CC_BUCKETS_SF) {
					bucket = CC_BUCKETS_SF - 1;
				}
				if(hamming <= hammingMaxR) {
					truesfcycles[bucket] += 1;
					statistics.sfPassingCycles.add(cycles_needed);
				} else {
					falsesfcycles[bucket] += 1;
					statistics.sfFilteredCycles.add(cycles_needed);
				}
#endif
				return hamming <= hammingMaxR;
			}
		};
};
#endif
