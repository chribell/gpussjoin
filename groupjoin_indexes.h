#ifndef SSJ_INDEXES_H
#define SSJ_INDEXES_H

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

#include <vector>
#include "mpltricks.h"

struct GroupGenericIndexListEntry {
	int grouprecordid;
	GroupGenericIndexListEntry(unsigned int grouprecordid) : grouprecordid(grouprecordid) {}
};

template <class AlgoIndexListEntry>
struct GenericIndexListHeader {
	typedef std::vector<AlgoIndexListEntry > AlgoIndexList;

	AlgoIndexList indexlist;
};

template < class Record > 
bool inline equalprefix(Record * rec1, Record * rec2, unsigned int maxprefix) {
	if(rec1->tokens.size() != rec2->tokens.size()) {
		return false;
	}
	assert( maxprefix <= rec1->tokens.size() && maxprefix <= rec2->tokens.size());

	for(unsigned int recpos = 0; recpos < maxprefix; ++recpos) {
		if(rec1->tokens[recpos] != rec2->tokens[recpos]) {
			return false;
		}
	}
	return true;
}

template < class GroupRecords, class Records, class GroupRecord, class Record, class Similarity>
void group_records (GroupRecords & grouprecords, Records & records, typename Similarity::threshold_type threshold) {
	if(records.size() == 0) {
		return;
	}
	// Create first group, add first record
	grouprecords.push_back(GroupRecord());
	GroupRecord * curgrouprecord = &grouprecords.back();
	Record * lastrec = &records[0];
	curgrouprecord->firstgrouprecord = lastrec;
	curgrouprecord->groupsize = 1;
	curgrouprecord->size = lastrec->tokens.size();
	curgrouprecord->maxprefixsize = lastrec->maxprefixsize;
	assert(lastrec->maxprefixsize ==
			Similarity::maxprefix(curgrouprecord->size, threshold));

	// Loop over records
	for(unsigned int i = 1; i < records.size(); ++i) {

		Record * currec = &records[i];
		if(equalprefix(lastrec, currec, curgrouprecord->maxprefixsize )) {

			// Add record to current group
			curgrouprecord->groupsize += 1;
			currec->nextgrouprecord = curgrouprecord->firstgrouprecord;
			curgrouprecord->firstgrouprecord = currec;

		} else {

			// Create new group, add current record
			grouprecords.push_back(GroupRecord());
			curgrouprecord = &grouprecords.back();
			lastrec = currec;
			curgrouprecord->firstgrouprecord = lastrec;
			curgrouprecord->groupsize = 1;
			curgrouprecord->size = lastrec->tokens.size();
			curgrouprecord->maxprefixsize = lastrec->maxprefixsize;
			assert(lastrec->maxprefixsize ==
					Similarity::maxprefix(curgrouprecord->size, threshold));
		}
	}
}

class IndexOnTheFlyPolicy {
	public:

		template <class Algorithm>
		struct Index {
			enum {
				SELF_JOIN = true,
				INDEXFIRST = false
			};
			typedef typename Algorithm::IndexStructure IndexStructure;
			typedef typename Algorithm::IndexStructure::iterator iterator;
			typedef typename Algorithm::Similarity Similarity;
			typedef typename Algorithm::IndexedRecords IndexedRecords;
			typedef typename Algorithm::IndexedRecord IndexedRecord;
			typedef typename Algorithm::GroupIndexedRecords GroupIndexedRecords;
			typedef typename Algorithm::GroupIndexedRecord GroupIndexedRecord;
			typedef typename Algorithm::BaseRecord BaseRecord;

			typedef typename Algorithm::IndexedRecord ProbeRecord;
			typedef typename Algorithm::GroupIndexedRecord GroupProbeRecord;

			IndexStructure index;

			typedef typename Similarity::threshold_type threshold_type;

			void largest_tokenid(unsigned int size_universe) {
				index.largest_tokenid(size_universe);
			}
			
			void inline group_records_index(GroupIndexedRecords & grouprecords, IndexedRecords & indexedrecords, threshold_type threshold) {}

			void inline group_records_join(GroupIndexedRecords & grouprecords, IndexedRecords & indexedrecords, threshold_type threshold) {
				group_records<GroupIndexedRecords,
					IndexedRecords,
					GroupIndexedRecord,
					IndexedRecord,
					Similarity>(grouprecords, indexedrecords, threshold);
			}

			void inline index_records(const GroupIndexedRecords & records, threshold_type threshold) {}

			inline void index_record(GroupIndexedRecord & grouprecord, unsigned int recind, unsigned int reclen, threshold_type threshold) {
				//Put record to index
				unsigned int midprefix = Similarity::midprefix(reclen, threshold);
				grouprecord.structuredata.setInitialPrefSize(midprefix);
				assert(midprefix <= grouprecord.maxprefixsize);
				for(unsigned int recpos = 0; recpos < midprefix; ++recpos) {
					unsigned int token = grouprecord.firstgrouprecord->tokens[recpos];
					index.addtoken(token, recind, recpos);
				}
			}

			inline iterator getiterator(unsigned int token) {
				return index.getiterator(token);
			}

			inline iterator end() {
				return index.end();
			}
		};

		template <class Algorithm>
		struct maxsizechecker {
			unsigned int maxlen;
			unsigned int curlen;

			typedef typename Algorithm::Similarity::threshold_type threshold_type;
			threshold_type threshold;
			inline maxsizechecker(unsigned int curlen, threshold_type threshold) : curlen(curlen), threshold(threshold) {
				if(Algorithm::LengthFilterPolicy::POS) {
					maxlen = Algorithm::Similarity::maxsize(curlen, threshold);
				}
			}

			inline bool isabove(unsigned int len) {
				if(Algorithm::LengthFilterPolicy::POS) {
					return len > maxlen;
				} else {
					return false;
				}
			}
			
			inline void updateprobepos(unsigned int pos) {
				if(Algorithm::LengthFilterPolicy::POS) {
					maxlen = Algorithm::Similarity::maxsize(curlen, pos, threshold);
				}
			}

		};
		
		struct recindchecker {
			inline static bool istocheck(unsigned int reclen, unsigned int indreclen) {
				return true;
			}
		};


};

template <bool foreign>
class IndexFirstPolicy {
	public:

		template <class Algorithm>
		struct Index {
			enum {
				SELF_JOIN = !foreign,
				INDEXFIRST = true
			};
			typedef typename Algorithm::IndexStructure IndexStructure;
			typedef typename Algorithm::IndexStructure::iterator iterator;
			typedef typename Algorithm::Similarity Similarity;
			typedef typename Algorithm::IndexedRecords IndexedRecords;
			typedef typename Algorithm::IndexedRecord IndexedRecord;
			typedef typename Algorithm::ForeignRecord ForeignRecord;
			typedef typename Algorithm::BaseRecord BaseRecord;
			typedef typename Algorithm::GroupIndexedRecords GroupIndexedRecords;
			typedef typename Algorithm::GroupIndexedRecord GroupIndexedRecord;

			IndexStructure index;

			typedef typename Similarity::threshold_type threshold_type;

			// Template metaprogramming trickery to select right types
			typedef typename IF<foreign, typename Algorithm::ForeignRecord, typename Algorithm::IndexedRecord>::RET ProbeRecord;
			typedef typename IF<foreign, typename Algorithm::ForeignRecords, typename Algorithm::IndexedRecords>::RET ProbeRecords;
			typedef typename IF<foreign, typename Algorithm::GroupForeignRecord, typename Algorithm::GroupIndexedRecord>::RET GroupProbeRecord;
			typedef typename IF<foreign, typename Algorithm::GroupForeignRecords, typename Algorithm::GroupIndexedRecords>::RET GroupProbeRecords;

			void largest_tokenid(unsigned int size_universe) {
				index.largest_tokenid(size_universe);
			}

			void inline group_records_index(GroupIndexedRecords & grouprecords, IndexedRecords & indexedrecords, threshold_type threshold) {
				group_records<GroupIndexedRecords,
					IndexedRecords,
					GroupIndexedRecord,
					IndexedRecord,
					Similarity>(grouprecords, indexedrecords, threshold);
			}

			void inline group_records_join(GroupProbeRecords & grouprecords, ProbeRecords & indexedrecords, threshold_type threshold) {
				if(!SELF_JOIN) {
					group_records<GroupProbeRecords,
						ProbeRecords,
						GroupProbeRecord,
						ProbeRecord,
						Similarity>(grouprecords, indexedrecords, threshold);
				}
			}

			inline unsigned int indexPrefixSize(unsigned int reclen, threshold_type threshold) const {
			 return foreign ? Similarity::maxprefix(reclen, threshold) : Similarity::midprefix(reclen, threshold);
			}

			void index_records(GroupIndexedRecords & grouprecords, threshold_type threshold) {
				unsigned int recind = 0;
				for(; recind < grouprecords.size(); ++recind) {
					GroupIndexedRecord & grouprecord = grouprecords[recind];
					const ForeignRecord & record = *(grouprecords[recind].firstgrouprecord);
					unsigned int reclen = grouprecord.size;
					unsigned int indexprefixsize = indexPrefixSize(reclen, threshold);
					grouprecord.structuredata.setInitialPrefSize(indexprefixsize);
					for(unsigned int recpos = 0; recpos < indexprefixsize; ++recpos) {
						unsigned int token = record.tokens[recpos];
						index.addtoken(token, recind, recpos);
					}
				}
			}

			inline void index_record(const GroupProbeRecord & record, unsigned int recind, 
					unsigned int reclen, threshold_type threshold) {}

			inline iterator getiterator(unsigned int token) {
				return index.getiterator(token);
			}

			inline iterator end() {
				return index.end();
			}
		};
		
		template <class Algorithm>
		struct maxsizechecker {
			unsigned int curlen;

			typedef typename Algorithm::Similarity::threshold_type threshold_type;
			threshold_type threshold;

			unsigned int maxlen;
			inline maxsizechecker(unsigned int curlen, threshold_type threshold) : curlen(curlen), threshold(threshold) {
				maxlen = Algorithm::Similarity::maxsize(curlen, threshold);
			}

			inline void updateprobepos(unsigned int pos) {
				if(Algorithm::LengthFilterPolicy::POS) {
					maxlen = Algorithm::Similarity::maxsize(curlen, pos, threshold);
				}
			}

			inline bool isabove(unsigned int len) {
				return len > maxlen;
			}
		};

		struct recindchecker {
			inline static bool istocheck(unsigned int recid, unsigned int indrecid) {
				return foreign ? true : recid > indrecid;
			}
		};


};

#endif
