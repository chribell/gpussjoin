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
#include "indexes.h"
#include "mpjoinpolicies.h"
#include "lengthfilter.h"
#include "frequencysorting.h"
#include "ppjoinpolicies.h"
#include "candidateset.h"
#include "cpucycles.h"
#include "gpu_handler.h"

template <typename MpJoinSimilarity/* = Jaccard*/,
		 typename MpJoinIndexStructurePolicy/* = MpJoinLinkedListIndexPolicy*/,
		 typename MpJoinIndexingStrategyPolicy = IndexOnTheFlyPolicy,
		 typename MpJoinLengthFilterPolicy = DefaultLengthFilterPolicy,
		 typename MpJoinPostPrefixPolicy = DisabledPPFilterPolicy
		 >
class MpJoin: public Algorithm {
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
		typedef IntRecord ForeignRecord;
		typedef IntRecords ForeignRecords;

		class IndexedRecord : public ForeignRecord {
			public:
				typename MpJoinIndexStructurePolicy::IndexStructureRecordData structuredata;
				typename MpJoinPostPrefixPolicy::IndexedRecordData postprefixfilterdata;
				IndexedRecord() {}
				inline void cleanup() {
					postprefixfilterdata.cleanup();
				}
				CandidateData candidateData;
		};
		typedef std::vector<IndexedRecord> IndexedRecords;
	private:
		ForeignRecords foreignrecords;
		IndexedRecords indexedrecords;
		algo_handle_records_freq_sort<IndexedRecords, ForeignRecords> inputhandler;


		GPUHandler* _gpuHandler;
		size_t result = 0;
	public:
		typedef MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinPostPrefixPolicy> self_type;

		typedef std::vector<IntRecord> Records;
		typedef IntRecord Record;

		typedef CandidateSet<CandidateData, IndexedRecords> CandidateSet_;
		typedef MpJoinSimilarity Similarity;

		// Minimum information we expect to be in an index list entry
		typedef GlobalGenericIndexListEntry GenericIndexListEntry;

		typedef MpJoinIndexStructurePolicy IndexStructurePolicy;
		typedef typename IndexStructurePolicy::template IndexStructure<self_type> IndexStructure;
		typedef MpJoinLengthFilterPolicy LengthFilterPolicy;
		typedef MpJoinPostPrefixPolicy PostPrefixPolicy;
		typedef MpJoinIndexingStrategyPolicy IndexingStrategyPolicy;
		typedef typename MpJoinIndexingStrategyPolicy::template Index<self_type> Index;
		typedef typename PostPrefixPolicy::template PostPrefixFilter<self_type> PostPrefixFilter;

		Index index;
		PostPrefixFilter postprefixfilter;

	public:
		//constructor
		MpJoin(threshold_type threshold) : threshold(threshold), inputhandler(indexedrecords, foreignrecords), postprefixfilter(this) {}

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

		virtual ~MpJoin();

	size_t getResult() { return result; }

	private:
		inline size_t proberecordssize() {
			if(Index::SELF_JOIN) {
				return indexedrecords.size();
			} else {
				return foreignrecords.size();
			}
		}
		GetProbeRecord<Index::SELF_JOIN,  ForeignRecords, IndexedRecords,
			IndexedRecord, typename Index::ProbeRecord> getproberecord;
};


template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrecord(IntRecord & record) {
	inputhandler.addrecord(record);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addforeignrecord(IntRecord & record) {
	inputhandler.addforeignrecord(record);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrawrecord(IntRecord & record) {
	inputhandler.addrawrecord(record);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::addrawforeignrecord(IntRecord & record) {
	inputhandler.addrawforeignrecord(record);
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::preparerecords() {
	inputhandler.prepareindex();
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::prepareforeignrecords() {
	inputhandler.prepareforeign();

}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::preparefinished() {
	inputhandler.cleanup();
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::doindex(GPUHandler* gpuHandler) {
    _gpuHandler = gpuHandler;
	index.largest_tokenid(inputhandler.get_largest_tokenid());
	index.index_records(indexedrecords, threshold);

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
    _gpuHandler->reserveCandidateSpace();
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
void MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::dojoin(
				HandleOutput * handleoutput) {
#if CYCLE_COUNT
#define CC_BUCKETS 10000
#define CC_BUCKETSIZE 1
	unsigned long truecycles[CC_BUCKETS];
	unsigned long falsecycles[CC_BUCKETS];
	unsigned long truesfcycles[CC_BUCKETS];
	unsigned long falsesfcycles[CC_BUCKETS];
	for(unsigned int i = 0; i < CC_BUCKETS; ++i) {
		truecycles[i] = falsecycles[i] = truesfcycles[i] = falsesfcycles[i] = 0;
	}
#endif
	CandidateSet_ candidateSet(indexedrecords);

	std::vector<unsigned int> minoverlapcache;
	unsigned int lastprobesize = 0;

	// foreach record...
	for (unsigned recind = 0; recind < proberecordssize(); ++recind) {
		typename Index::ProbeRecord & record = getproberecord(indexedrecords, foreignrecords, recind);
		unsigned int reclen = record.tokens.size();

		//Minimum size of records in index
		unsigned int minsize = Similarity::minsize(reclen, threshold);

		_gpuHandler->insertProbe(recind);

		// Check whether cache is to renew
		if(lastprobesize != reclen) {
			lastprobesize = reclen;
			unsigned int maxel = Index::SELF_JOIN ? reclen : Similarity::maxsize(reclen, threshold);
			minoverlapcache.resize(maxel + 1);
			for(unsigned int i = minsize; i <= maxel; ++i) {
				minoverlapcache[i] = Similarity::minoverlap(reclen, i, threshold);
			}
		}

		// Length of probing prefix
		unsigned int maxprefix = Similarity::maxprefix(reclen, threshold);

		typename MpJoinIndexingStrategyPolicy::template maxsizechecker<self_type> 
			maxsizechecker(reclen, threshold);

		// Compute bitmap for record
		postprefixfilter.probe_record_compute(record, maxprefix);
		_gpuHandler->updateMaxSetSize(reclen);

		// foreach elem in probing prefix
		for (unsigned recpos = 0; recpos < maxprefix; ++recpos) {
			unsigned int token = record.tokens[recpos];

			// get iterator and do min length filtering at the start
			typename Index::iterator ilit = index.getiterator(token);
			statistics.lookups.inc();

			maxsizechecker.updateprobepos(recpos);

			// First, apply min-length filter
			while(!ilit.end()) {
				// record from index
				IndexedRecord & indexrecord = indexedrecords[ilit->recordid];
				unsigned int indreclen = indexrecord.tokens.size();

				//Length filter - check whether minsize is satisfied
				if(ilit.lengthfilter(indreclen, minsize)) {
					break;
				}
				//Note: the iterator is increased by lengthfilter
			}

			// for each record in inverted list 
			while(!ilit.end() ) {

				if(!MpJoinIndexingStrategyPolicy::recindchecker::istocheck(recind, ilit->recordid)) {
					break;
				}

				// record from index 
				IndexedRecord & indexrecord = indexedrecords[ilit->recordid];
				unsigned int indreclen = indexrecord.tokens.size();

				//Length filter 2 - maxlength above tighter length filter (if enabled)
				if(maxsizechecker.isabove(indreclen)) {
					break;
				}

				//TODO: Bitmap filter that doesn't let the candidate into the candidate hash map

				// position of token in indexrecord
				unsigned int indrecpos = ilit->position;

				// Remove if stale element (mpjoin only)
				if(ilit.mpjoin_removestale1(indexrecord, indrecpos)) {
					continue;
				}

				statistics.indexEntriesSeen.inc();

				// search for candidate in candidate set
				CandidateData & candidateData = candidateSet.getCandidateData(ilit->recordid);

				if(candidateData.count == 0) {
					// Not seen before
					
					unsigned int minoverlap = minoverlapcache[indreclen];

					if(!LengthFilterPolicy::posfilter(reclen, indreclen, recpos, indrecpos, minoverlap)) {

						// Needs to be done for each index entry, independent of positional filter
						ilit.mpjoin_updateprefsize(indexrecord, indreclen, minoverlap);

						// mpjoin_removestale determines whether the current entry is dead. IF yes, it removes it and
						// increases the iterator. Otherwise, we have to increase the iterator ourselves.
						if(!ilit.mpjoin_removestale2(indexrecord, indreclen, indrecpos, minoverlap)) {
							ilit.next();
						}

						continue;

					} else if (!postprefixfilter.check_probe_against_index(
					            record, indexrecord, recind,
					            minoverlap,
					            reclen, recpos,
					            indreclen, indrecpos, 1 /*candidateData.count + 1*/ )) {

						// Needs to be done for each index entry, independent of positional filter
						ilit.mpjoin_updateprefsize(indexrecord, indreclen, minoverlap);
						ilit.next();

						continue;
					}
					candidateSet.addRecord(ilit->recordid);
					candidateData.minoverlap = minoverlapcache[indreclen];
					_gpuHandler->insertCandidate(ilit->recordid);

				}

				candidateData.count += 1;
				candidateData.recpos = recpos;
				candidateData.indrecpos = indrecpos;
				
				//Update prefsize
				// Needs to be done for each index entry, independent of positional filter
				ilit.mpjoin_updateprefsize(indexrecord, indreclen, candidateData.minoverlap);

				ilit.next();
			}
		}

		// Candidate set after prefix filter
		statistics.candidatesP1.add(candidateSet.size());
		_gpuHandler->updateProbeNumberOfCandidates();

		candidateSet.reset();
		candidateSet.clear();
	
		index.index_record(record, recind, reclen, threshold);
		
		// Add computed bitmap index to record	
		postprefixfilter.probe_to_index(record);
	}

	_gpuHandler->flush();

	result = _gpuHandler->getResult();
	_gpuHandler->free();
}

template <typename MpJoinSimilarity, typename MpJoinIndexStructurePolicy, class MpJoinIndexingStrategyPolicy, class MpJoinLengthFilterPolicy, typename MpJoinBitfilterPolicy>
MpJoin<MpJoinSimilarity, MpJoinIndexStructurePolicy, MpJoinIndexingStrategyPolicy, MpJoinLengthFilterPolicy, MpJoinBitfilterPolicy>::~MpJoin() {
	postprefixfilter.cleanup_postprefixfilterdata(indexedrecords);
	//inputhandler.cleanup();

}


#endif
