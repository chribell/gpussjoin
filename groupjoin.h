#ifndef SSJ_MPJOIN_H
#define SSJ_MPJOIN_H

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

#include <algorithm>
#include <vector>
#include "classes.h"
#include "output.h"
#include "verify.h"
#include "data.h"
#include "mpjoinpolicies.h"
#include "groupjoin_indexes.h"
#include "lengthfilter.h"
#include "frequencysorting.h"
#include "ppjoinpolicies.h"
#include "candidateset.h"
#include "gpu_handler.h"

template <typename MpJoinSimilarity/* = Jaccard*/,
		typename MpJoinIndexStructurePolicy/* = MpJoinLinkedListIndexPolicy*/,
		typename MpJoinIndexingStrategyPolicy = IndexOnTheFlyPolicy,
		typename MpJoinLengthFilterPolicy = DefaultLengthFilterPolicy,
		typename MpJoinPostPrefixPolicy = DisabledPPFilterPolicy
>
class GroupJoin: public Algorithm {
protected:

	struct CandidateData {
		int count;
		unsigned int minoverlap;
		unsigned int recpos;
		unsigned int indrecpos;

		// default constructor
		CandidateData() : count(0) {}

		inline void reset() {
			count = 0;
		}
	};


public:
	typedef typename MpJoinSimilarity::threshold_type threshold_type;
	const threshold_type threshold;

	/* Terminology:
       ForeignRecords .. Only for foreign joins - contain sets to probe against index
       IndexedRecords .. Records that will be indexed - in case of self-joins, identical to probing set
       */
	typedef IntRecord BaseRecord;
	class ForeignRecord : public IntRecord {
	public:
		ForeignRecord() : nextgrouprecord(NULL) {}
		ForeignRecord * nextgrouprecord;
		unsigned int maxprefixsize;
	};

	typedef std::vector<ForeignRecord> ForeignRecords;

	class IndexedRecord : public ForeignRecord {
	public:
		typename MpJoinPostPrefixPolicy::IndexedRecordData postprefixfilterdata;
		IndexedRecord() {}
		inline void cleanup() {
			postprefixfilterdata.cleanup();
		}
	};
	typedef std::vector<IndexedRecord> IndexedRecords;

	struct GroupForeignRecord {
		typedef ForeignRecord RecordType;
		GroupForeignRecord() : firstgrouprecord(NULL) {}
		ForeignRecord * firstgrouprecord;
		unsigned int groupsize;
		unsigned int maxprefixsize;
		unsigned int size;
	};

	struct GroupIndexedRecord {
		typedef IndexedRecord RecordType;
		GroupIndexedRecord() : firstgrouprecord(NULL) {}
		ForeignRecord * firstgrouprecord;
		unsigned int groupsize;
		unsigned int maxprefixsize;
		unsigned int size;
		CandidateData candidateData;
		typename MpJoinIndexStructurePolicy::IndexStructureRecordData structuredata;
	};

	typedef std::vector<GroupIndexedRecord> GroupIndexedRecords;
	typedef std::vector<GroupForeignRecord> GroupForeignRecords;


private:
	ForeignRecords foreignrecords;
	IndexedRecords indexedrecords;

	GroupForeignRecords groupforeignrecords;
	GroupIndexedRecords groupindexedrecords;

	struct LexSortGroup {
		static unsigned int sortprefixsize(const ForeignRecord & rec) {
			return rec.maxprefixsize;
		}
	};

	algo_handle_records_freq_sort<IndexedRecords, ForeignRecords, LexSortGroup> inputhandler;


	GPUHandler* _gpuHandler;
	size_t result = 0;

public:
	typedef GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinPostPrefixPolicy> self_type;

	typedef std::vector<IntRecord> Records;
	typedef IntRecord Record;

	typedef CandidateSet<CandidateData, GroupIndexedRecords> CandidateSet_;
	typedef MpJoinSimilarity Similarity;

	// Minimum information we expect to be in an index list entry
	typedef GroupGenericIndexListEntry GenericIndexListEntry;

	typedef MpJoinIndexStructurePolicy IndexStructurePolicy;
	typedef typename IndexStructurePolicy::template IndexStructure<self_type, GroupIndexedRecord> IndexStructure;
	typedef MpJoinLengthFilterPolicy LengthFilterPolicy;
	typedef MpJoinPostPrefixPolicy PostPrefixPolicy;
	typedef MpJoinIndexingStrategyPolicy IndexingStrategyPolicy;
	typedef typename MpJoinIndexingStrategyPolicy::template Index<self_type> Index;
	typedef typename PostPrefixPolicy::template PostPrefixFilter<self_type> PostPrefixFilter;

	Index index;
	PostPrefixFilter postprefixfilter;

	typedef GetProbeRecord<Index::SELF_JOIN,  GroupForeignRecords, GroupIndexedRecords,
			GroupIndexedRecord, typename Index::GroupProbeRecord> GetGroupProbeRecord;
	typedef GetProbeRecord<Index::SELF_JOIN,  ForeignRecords, IndexedRecords,
			IndexedRecord, typename Index::ProbeRecord> _GetProbeRecord;
	typedef GetProbeRecords<Index::SELF_JOIN, GroupForeignRecords, GroupIndexedRecords> GetGroupProbeRecords;
	typedef GetProbeRecords<Index::SELF_JOIN, ForeignRecords, IndexedRecords> _GetProbeRecords;

public:
	//constructor
	GroupJoin(threshold_type threshold) : threshold(threshold), inputhandler(indexedrecords, foreignrecords), postprefixfilter(this) {}

	//addrecord and addforeignrecord must use swap to get the integer vector from record,
	//such that record refers to an empty record after the respective calls
	void addrecord(IntRecord & record);
	void addforeignrecord(IntRecord & record);

	//addrawrecord and addrawforeignrecord must use swap to get the integer vector from record,
	//such that record refers to an empty record after the respective calls
	void addrawrecord(IntRecord & record);
	void addrawforeignrecord(IntRecord & record);

	//multi-step process to measure the individual steps from outside
	void preparerecords();
	void prepareforeignrecords();
	void preparefinished();
	void doindex(GPUHandler* gpuHandler);
	void dojoin(
			HandleOutput * handleoutput);

	virtual ~GroupJoin();

	size_t getResult() { return result; }

private:
	inline size_t groupproberecordssize() {
		if(Index::SELF_JOIN) {
			return groupindexedrecords.size();
		} else {
			return groupforeignrecords.size();
		}
	}
	GetGroupProbeRecord getgroupproberecord;
	_GetProbeRecord getproberecord;
	GetGroupProbeRecords getgroupproberecords;
	_GetProbeRecords getproberecords;
};


template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrecord(IntRecord & record) {
	IndexedRecord & rec = inputhandler.addrecord(record);
	rec.maxprefixsize = Similarity::maxprefix(rec.tokens.size(), threshold);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addforeignrecord(IntRecord & record) {
	ForeignRecord & rec = inputhandler.addforeignrecord(record);
	rec.maxprefixsize = Similarity::maxprefix(rec.tokens.size(), threshold);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrawrecord(IntRecord & record) {
	IndexedRecord & rec = inputhandler.addrawrecord(record);
	rec.maxprefixsize = Similarity::maxprefix(rec.tokens.size(), threshold);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrawforeignrecord(IntRecord & record) {
	ForeignRecord & rec = inputhandler.addrawforeignrecord(record);
	rec.maxprefixsize = Similarity::maxprefix(rec.tokens.size(), threshold);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::preparerecords() {
	inputhandler.prepareindex(true);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::prepareforeignrecords() {
	inputhandler.prepareforeign(true);

}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::preparefinished() {
	inputhandler.cleanup();
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::doindex(GPUHandler* gpuHandler) {
	_gpuHandler = gpuHandler;

	index.largest_tokenid(inputhandler.get_largest_tokenid());

	index.group_records_index(groupindexedrecords, indexedrecords, threshold);

	index.index_records(groupindexedrecords, threshold);

	// Tell bitmap handler maximum set size
	postprefixfilter.largest_index_set_size(indexedrecords[indexedrecords.size() - 1].tokens.size());

	// Create bitmaps for indexed records
	postprefixfilter.create_for_indexed_records(indexedrecords);
    for (auto& rec : indexedrecords) {
        _gpuHandler->addIndexedRecord(rec.tokens);
    }
    _gpuHandler->transferInputCollection();
    if (!Index::SELF_JOIN) {
        for (auto& rec : foreignrecords) {
            _gpuHandler->addForeignRecord(rec.tokens);
        }
        _gpuHandler->transferForeignInputCollection();
    }
    _gpuHandler->reserveCandidateSpace(Index::SELF_JOIN ? indexedrecords.size() : foreignrecords.size());
}

template < class GroupProbeRecord, class RecordType, class Similarity, class Statistics>
void inline verify_within_group(
		GroupProbeRecord & grecord,
		unsigned int greclen,
		typename Similarity::threshold_type threshold,
		HandleOutput * handleoutput,
		Statistics & statistics,
        size_t * count) {
	// check every combination within group if self-join
	// An Ernst Jandl - Arrow :-)
	unsigned int groupsize = grecord.groupsize;
	if(groupsize != 1) {
		unsigned int gminoverlap = Similarity::minoverlap(greclen, greclen, threshold);
		assert(grecord.maxprefixsize == Similarity::maxprefix(greclen, threshold));
		RecordType * recpl1 = grecord.firstgrouprecord;
		while(recpl1 != NULL) {
			RecordType * recpl2 = recpl1->nextgrouprecord;
            while(recpl2 != NULL) {
				statistics.candidatesVery.inc();
                if(verifypair(recpl1->tokens, recpl2->tokens, gminoverlap, grecord.maxprefixsize, grecord.maxprefixsize, grecord.maxprefixsize)) {
                    //handleoutput->addPair(*recpl1,  *recpl2);
                    (*count)++;
                }
                recpl2 = recpl2->nextgrouprecord;
			}
			recpl1 = recpl1->nextgrouprecord;
		}
	}
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::dojoin(
		HandleOutput * handleoutput) {
	CandidateSet_ candidateSet(groupindexedrecords);

	std::vector<unsigned int> minoverlapcache;
	unsigned int lastprobesize = 0;

	index.group_records_join(getgroupproberecords(groupindexedrecords, groupforeignrecords), getproberecords(indexedrecords, foreignrecords), threshold);

//	_gpuHandler->initOffsets(getgroupproberecords(groupindexedrecords, groupforeignrecords).size());
    auto count = new size_t;
    *count = 0;
    size_t sec_count = 0;
	// foreach record...
	for (unsigned grecind = 0; grecind < groupproberecordssize(); ++grecind) {

		typename Index::GroupProbeRecord & grecord = getgroupproberecord(groupindexedrecords, groupforeignrecords, grecind);
		unsigned int greclen = grecord.size;

		//Minimum size of records in index
		unsigned int minsize = Similarity::minsize(greclen, threshold);

		// Check whether cache is to renew
		if(lastprobesize != greclen) {
			lastprobesize = greclen;
			unsigned int maxel = Index::SELF_JOIN ? greclen : Similarity::maxsize(greclen, threshold);
			minoverlapcache.resize(maxel + 1);
			for(unsigned int i = minsize; i <= maxel; ++i) {
				minoverlapcache[i] = Similarity::minoverlap(greclen, i, threshold);
			}
		}

		// Length of probing prefix
		unsigned int maxprefix = grecord.maxprefixsize;
		assert(maxprefix == grecord.maxprefixsize);
        _gpuHandler->updateMaxSetSize(greclen);

		typename MpJoinIndexingStrategyPolicy::template maxsizechecker<self_type>
				maxsizechecker(greclen, threshold);

		//  Hook for postprefix filter - would need unrolling of group
		// postprefixfilter.probe_record_compute(record, maxprefix);

		// foreach elem in probing prefix
		for (unsigned recpos = 0; recpos < maxprefix; ++recpos) {
			unsigned int token = grecord.firstgrouprecord->tokens[recpos];

			// get iterator and do min length filtering at the start
			typename Index::iterator ilit = index.getiterator(token);
			statistics.lookups.inc();

			maxsizechecker.updateprobepos(recpos);

			// First, apply min-length filter
			while(!ilit.end()) {
				// record from index
				GroupIndexedRecord & gindexrecord = groupindexedrecords[ilit->grouprecordid];
				unsigned int gindreclen = gindexrecord.size;

				//Length filter
				if(ilit.lengthfilter(gindreclen, minsize)) {
					break;
				}
				//Note: the iterator is increased by lengthfilter
			}

			// for each record in inverted list
			while(!ilit.end() ) {

				if(!MpJoinIndexingStrategyPolicy::recindchecker::istocheck(grecind, ilit->grouprecordid)) {
					break;
				}

				// record from index
				GroupIndexedRecord & gindexrecord = groupindexedrecords[ilit->grouprecordid];
				unsigned int gindreclen = gindexrecord.size;

				//Length filter 2 - maxlength above tighter length filter (if enabled)
				if(maxsizechecker.isabove(gindreclen)) {
					break;
				}

				statistics.indexEntriesSeen.inc();

				//TODO: Bitmap filter that doesn't let the candidate into the candidate hash map

				// position of token in indexrecord
				unsigned int gindrecpos = ilit->position;

				// Remove if stale element (mpjoin only)
				if(ilit.mpjoin_removestale1(gindexrecord, gindrecpos)) {
					continue;
				}

				// search for candidate in candidate set
				CandidateData & candidateData = candidateSet.getCandidateData(ilit->grouprecordid);

				if(candidateData.count == 0) {

					unsigned int minoverlap = minoverlapcache[gindreclen];

					// pos filter
					if(!LengthFilterPolicy::posfilter(greclen, gindreclen, recpos, gindrecpos, minoverlap)) {

						// Needs to be done for each index entry, independent of positional filter
						ilit.mpjoin_updateprefsize(gindexrecord, gindreclen, minoverlap);

						// mpjoin_removestale determines whether the current entry is dead. IF yes, it removes it and
						// increases the iterator. Otherwise, we have to increase the iterator ourselves.
						if(!ilit.mpjoin_removestale2(gindexrecord, gindreclen, gindrecpos, minoverlap)) {
							ilit.next();
						}

						continue;
					}

					candidateSet.addRecord(ilit->grouprecordid);
                    _gpuHandler->insertCandidate(grecord.firstgrouprecord->recordid, groupindexedrecords[ilit->grouprecordid].firstgrouprecord->recordid);

					candidateData.minoverlap = minoverlap;

				}

				candidateData.count += 1;
				candidateData.recpos = recpos;
				candidateData.indrecpos = gindrecpos;

				//Update prefsize
				// Needs to be done for each index entry, independent of positional filter
				ilit.mpjoin_updateprefsize(gindexrecord, gindreclen, candidateData.minoverlap);

				ilit.next();
			}
		}

		// Candidate set after prefix filter
		statistics.candidatesP1.add(candidateSet.size());
        _gpuHandler->updateCandidateOffset(grecord.firstgrouprecord->recordid);

		//Now, verify candidates

		// First check every combination within group if self-join
#ifndef CAND_ONLY
		if(Index::SELF_JOIN) {
			verify_within_group<typename Index::GroupProbeRecord, ForeignRecord, Similarity, _Statistics>(
					grecord, greclen, threshold, handleoutput, statistics, count);
		}
#endif

		//Verification loop
		typename CandidateSet_::iterator candit = candidateSet.begin();
//        verify_loop_count = 0;
		for( ; candit != candidateSet.end(); ++candit) {

			CandidateData & candidateData = candidateSet.getCandidateData(*candit);
#ifdef CAND_ONLY
			statistics.candidatesVery.inc();
#else
			// record from index
			GroupIndexedRecord gindexrecord = groupindexedrecords[*candit];
			unsigned int gindreclen = gindexrecord.size;

			unsigned int recpos = candidateData.recpos;
			unsigned int indrecpos = candidateData.indrecpos;

			//First position after last position by index lookup in indexed record
			unsigned int lastposind = IndexStructure::verify_indexrecord_start(gindexrecord, gindreclen, this);

			//First position after last position by index lookup in probing record
			unsigned int lastposprobe = LengthFilterPolicy::verify_record_start(greclen, maxprefix, candidateData.minoverlap);

			unsigned int recpreftoklast = grecord.firstgrouprecord->tokens[lastposprobe - 1];
			unsigned int indrecpreftoklast = gindexrecord.firstgrouprecord->tokens[lastposind - 1];


			if(recpreftoklast > indrecpreftoklast) {
				recpos += 1;
				//first position after minprefix / lastposind
				indrecpos = lastposind;
			} else {
				// First position after maxprefix / lastposprobe
				recpos = lastposprobe;
				indrecpos += 1;
			}

			ForeignRecord * recpl1 = grecord.firstgrouprecord;
            bool first = true;
			while( recpl1 != NULL) {
				ForeignRecord * recpl2 = gindexrecord.firstgrouprecord;
				while(recpl2 != NULL) {
                    if (first) {
                        first = false;
                        recpl2 = recpl2->nextgrouprecord;
                        continue;
                    }

					statistics.candidatesVery.inc();
                    if(verifypair(recpl1->tokens, recpl2->tokens, candidateData.minoverlap,      recpos, indrecpos, candidateData.count)) {
                        //handleoutput->addPair(*recpl1, *recpl2);
                        (*count)++;
                    }
					recpl2 = recpl2->nextgrouprecord;
				}
				recpl1 = recpl1->nextgrouprecord;
			}

#endif
			candidateData.reset();

		}
		candidateSet.clear();

		index.index_record(grecord, grecind, greclen, threshold);

		// Add computed bitmap index to record
		// postprefixfilter.probe_to_index(record);

	}
//    std::cout << "CPU: " << *count << std::endl;
	_gpuHandler->flush();
	result = _gpuHandler->getResult() + *count;
	_gpuHandler->free();
}
template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
GroupJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::~GroupJoin() {
	postprefixfilter.cleanup_postprefixfilterdata(indexedrecords);
	//inputhandler.cleanup();

}


#endif
