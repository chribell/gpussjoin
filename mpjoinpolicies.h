#ifndef SSJ_MPJOINPOLICIES_H
#define SSJ_MPJOINPOLICIES_H

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

#include "inv_index.h"

class PPJoinIndexPolicy {

	public:

		struct IndexStructureRecordData {
			unsigned int prefsize;
			inline IndexStructureRecordData() : prefsize(INT_MAX) {}
			inline void setInitialPrefSize(unsigned int prefsize) {
				this->prefsize = prefsize;
			}
		};

		template <typename Algorithm, typename IndexedRecord_ = typename Algorithm::IndexedRecord>
		struct IndexStructure {

			struct IndexListEntry : Algorithm::GenericIndexListEntry {
				typedef typename Algorithm::GenericIndexListEntry GenericIndexListEntry;
				int position;
				IndexListEntry(unsigned int recordid, int position) : GenericIndexListEntry(recordid), position(position) {}
			};

			struct IndexListHeader {
				std::vector<IndexListEntry> ilist;
				int firsttocheck;
				IndexListHeader() : firsttocheck(0) {}
			};

			typedef IndexedRecord_ IndexedRecord;
			typedef InvIndexVector::template IndexMap<IndexListHeader> Index;
			Index index;

			struct iterator {
				unsigned int curelem;
				unsigned int elemcount;

				IndexListHeader * ilheader;

				// Empty iterator if there are no entries
				inline iterator() : curelem(0), elemcount(0) {}

				inline iterator(IndexListHeader * ilheader) :
					ilheader(ilheader) {
						curelem = ilheader->firsttocheck;
						elemcount = ilheader->ilist.size();
					}

				inline bool end() const {
					return curelem == elemcount;
				}

				inline void next() {
					assert(curelem >= 0);
					curelem += 1;
				}

				inline bool lengthfilter(unsigned int indreclen, unsigned int minsize) {
					if(indreclen < minsize) {
						curelem += 1;
						ilheader->firsttocheck = curelem;
						return false;
					}
					return true;
				}

				inline void next_deletecur() {
					next();
				}

				inline bool mpjoin_removestale1(IndexedRecord & indexedrecord, unsigned int position) {
					assert(position == ilheader->ilist[curelem].position);
					return false;
				}

				inline bool mpjoin_removestale2(IndexedRecord & indexedrecord, unsigned int indreclen, unsigned int position, unsigned int minoverlap) {
					return false;
				}

				inline void mpjoin_updateprefsize(IndexedRecord & indexrecord, unsigned int indreclen, unsigned int minoverlap) {}

				inline IndexListEntry & operator*() {
					assert(curelem >= 0);
					return ilheader->ilist[curelem];
				}

				inline IndexListEntry * operator->() {
					assert(curelem >= 0);
					return &ilheader->ilist[curelem];
				}

			};


			inline void addtoken(unsigned int token, unsigned int recind, unsigned int recpos) {
				typename Index::value_type & ilhead = index.get_list_create(token);
				ilhead.ilist.push_back(IndexListEntry(recind, recpos));
			}

			inline iterator getiterator(unsigned int token) {
				typename Index::value_type * ilhead = index.get_list(token);
				if( ilhead != NULL ) {
					return iterator(ilhead);
					//If the header is null, we are the end
				} else {
					return iterator();
				}
			}

			static inline unsigned int verify_indexrecord_start(const IndexedRecord & indexrecord, unsigned int indreclen, Algorithm * algo) {
				return indexrecord.structuredata.prefsize;
			}

			void largest_tokenid(unsigned int tokenid) {
				index.resize(tokenid + 1);
			}
		};
};

template <typename IndexStructure>
struct MpJoinCommonMethods {
	typedef typename IndexStructure::iterator IndStructIt;
	typedef typename IndexStructure::IndexedRecord IndexedRecord;

	inline static bool lengthfilter(IndStructIt & chldpnt, unsigned int indreclen, unsigned int minsize) {
		if(indreclen < minsize) {
			chldpnt.next_deletecur();
			return false;
		}
		return true;
	}

	inline static bool mpjoin_removestale1(IndStructIt & chldpnt, IndexedRecord & indexedrecord,
			unsigned int position) {
		if(indexedrecord.structuredata.prefsize < position) {
			chldpnt.next_deletecur();
			return true;
		}
		return false;

	}

	inline static void mpjoin_updateprefsize(IndexedRecord & indexrecord,
			unsigned int indreclen, unsigned int minoverlap) {
		
		indexrecord.structuredata.prefsize = indreclen - minoverlap + 1;
	}

	inline static bool mpjoin_removestale2(IndStructIt & chldpnt, IndexedRecord & indexedrecord,
			unsigned int indreclen, unsigned int position, unsigned int minoverlap) {

		if(1 + indreclen - position < minoverlap) {
			chldpnt.next_deletecur();
			mpjoin_updateprefsize(indexedrecord, indreclen, minoverlap);
			return true;
		}
		return false;

	}

	inline static unsigned int verify_indexrecord_start(const IndexedRecord & indexrecord, unsigned int indreclen, Algorithm * algo) {
		return indexrecord.structuredata.prefsize;
	}

};

class MpJoinLinkedListIndexPolicy {

	public:

		struct IndexStructureRecordData {
			unsigned int prefsize;
			IndexStructureRecordData() : prefsize(INT_MAX) {}
			inline void setInitialPrefSize(unsigned int prefsize) {}
		};

		template <typename Algorithm, typename IndexedRecord_ = typename Algorithm::IndexedRecord>
		struct IndexStructure {

			struct IndexListEntry : Algorithm::GenericIndexListEntry {
				typedef typename Algorithm::GenericIndexListEntry GenericIndexListEntry;
				unsigned int position;
				IndexListEntry * next;
				IndexListEntry(unsigned int recordid, unsigned int position) : GenericIndexListEntry(recordid), position(position), next(NULL) {}
			};

			struct IndexListHeader {
				IndexListEntry * head;
				IndexListEntry * last;
				IndexListHeader() : head(NULL), last(NULL) {}
			};

			typedef IndexedRecord_ IndexedRecord;
			typedef InvIndexVector::template IndexMap<IndexListHeader> Index;
			typedef IndexStructure<Algorithm, IndexedRecord> self_type;
			Index index;

			struct iterator {
				IndexListHeader * curlistheader;
				IndexListEntry * curlistentry;

				IndexListEntry * predlistentry;

				typedef MpJoinCommonMethods<self_type> commonMethods;

				inline iterator(IndexListHeader * curlistheader, IndexListEntry * curlistentry) :
					curlistheader(curlistheader), curlistentry(curlistentry), predlistentry(NULL) {}

				inline bool end() const {
					return curlistentry == NULL;
				}

				inline void next() {
					assert(curlistentry != NULL);
					predlistentry = curlistentry;
					curlistentry = curlistentry->next;
				}

				inline bool lengthfilter(unsigned int indreclen, unsigned int minsize) {
					return commonMethods::lengthfilter(*this, indreclen, minsize);
				}

				inline void next_deletecur() {
					assert(curlistentry != NULL);
					// Is first element . so change header of list
					// in case it is called by length filter, this condition always applies so 
					// assist the compiler at inlining and avoiding the branch...
					if(predlistentry == NULL) {
						assert(curlistheader->head == curlistentry);
						curlistheader->head = curlistentry->next;
					} else {
						// otherwise, change pointer in preceeding entry
						predlistentry->next = curlistentry->next;
					}
					// If current element is last element, change "last" pointer in header
					if( curlistentry->next == NULL) {
						assert(curlistheader->last == curlistentry);
						curlistheader->last = predlistentry;
					} else {
						assert(curlistheader->last != curlistentry);
					}
					// Change pointer to point to next element, delete current element
					IndexListEntry * tmp = curlistentry;
					curlistentry = curlistentry->next;
					delete tmp;
				}

				inline bool mpjoin_removestale1(IndexedRecord & indexedrecord, unsigned int position) {
					return commonMethods::mpjoin_removestale1(*this, indexedrecord, position);
				}

				inline bool mpjoin_removestale2(IndexedRecord & indexedrecord, 
						unsigned int indreclen, unsigned int position, unsigned int minoverlap) {

					return commonMethods::mpjoin_removestale2(*this, indexedrecord, indreclen, position, minoverlap);
				}

				inline void mpjoin_updateprefsize(IndexedRecord & indexrecord, unsigned int indreclen, unsigned int minoverlap) {
					return commonMethods::mpjoin_updateprefsize(indexrecord, indreclen, minoverlap);
				}


				inline IndexListEntry & operator*() {
					assert(curlistentry != NULL);
					return *curlistentry;
				}

				inline IndexListEntry * operator->() {
					assert(curlistentry != NULL);
					return curlistentry;
				}

			};


			inline static unsigned int verify_indexrecord_start(const IndexedRecord & indexrecord, unsigned int indreclen, Algorithm * algo) {
				return MpJoinCommonMethods<IndexStructure>::verify_indexrecord_start(indexrecord, indreclen, algo);
			}

			inline void addtoken(unsigned int token, unsigned int recind, unsigned int recpos) {
				typename Index::value_type & ilhead = index.get_list_create(token);
				IndexListEntry * newentry = new IndexListEntry(recind, recpos);
				if(ilhead.last != NULL) {
					// append after last entry
					ilhead.last->next = newentry;
				} else {
					// First entry to insert in list
					ilhead.head = newentry;
				}
				ilhead.last = newentry;

			}

			inline iterator getiterator(unsigned int token) {
				typename Index::value_type * ilhead = index.get_list(token);
				if(ilhead != NULL) {
					return iterator(ilhead, ilhead->head);
				} else {
					return iterator(NULL, NULL);
				}
			}


			~IndexStructure() {
				typename Index::iterator indit = index.begin();
				for(; indit != index.end(); ++indit) {
					IndexListEntry * ilent = Index::it2value(indit).head;
					while(ilent != NULL) {
						IndexListEntry * tmp = ilent;
						ilent = ilent->next;
						delete tmp;
					}
				}

			}
			void largest_tokenid(unsigned int tokenid) {
				index.resize(tokenid + 1);
			}
		};

};

class MpJoinArrayIndexPolicy {

	public:
		struct IndexStructureRecordData {
			unsigned int prefsize;
			IndexStructureRecordData() : prefsize(INT_MAX) {}
			inline void setInitialPrefSize(unsigned int prefsize) {}
		};

		template<typename Algorithm, typename IndexedRecord_ = typename Algorithm::IndexedRecord>
		struct IndexStructure {

			struct IndexListEntry : Algorithm::GenericIndexListEntry {
				typedef typename Algorithm::GenericIndexListEntry GenericIndexListEntry;
				int position;
				IndexListEntry(unsigned int recordid, int position) : GenericIndexListEntry(recordid), position(position) {}
			};

			struct IndexListHeader {
				std::vector<IndexListEntry> ilist;
				int firstvalid;
				IndexListHeader() : firstvalid(-1) {}
			};

			typedef IndexedRecord_ IndexedRecord;
			typedef InvIndexVector::template IndexMap< IndexListHeader> Index;
			typedef IndexStructure<Algorithm, IndexedRecord> self_type;
			Index index;

			struct iterator {
				int curelem;

				IndexListHeader * ilheader;

				typedef MpJoinCommonMethods<self_type> commonMethods;

				// Empty iterator if there are no entries
				inline iterator() : curelem(-1) {}

				inline iterator(IndexListHeader * ilheader) :
					ilheader(ilheader) {
						curelem = ilheader->firstvalid;
					}

				inline bool end() const {
					return curelem < 0;
				}

				inline void next() {
					assert(curelem >= 0);
					do {
						curelem += 1;
					} while(curelem < ilheader->ilist.size() && ilheader->ilist[curelem].position < 0);

					if(curelem == ilheader->ilist.size()) {
						curelem = -1;
					}		
				}

				inline void next_deletecur() {
					if(curelem == ilheader->firstvalid) {
						next();
						ilheader->firstvalid = curelem;
						return;
					}

					ilheader->ilist[curelem].position = -1;
					next();
				}

				inline bool lengthfilter(unsigned int indreclen, unsigned int minsize) {
					return commonMethods::lengthfilter(*this, indreclen, minsize);
				}


				inline bool mpjoin_removestale1(IndexedRecord & indexedrecord, unsigned int position) {
					return commonMethods::mpjoin_removestale1(*this, indexedrecord, position);
				}

				inline bool mpjoin_removestale2(IndexedRecord & indexedrecord, 
						unsigned int indreclen, unsigned int position, unsigned int minoverlap) {

					return commonMethods::mpjoin_removestale2(*this, indexedrecord, indreclen, position, minoverlap);
				}

				inline void mpjoin_updateprefsize(IndexedRecord & indexrecord, unsigned int indreclen, unsigned int minoverlap) {
					return commonMethods::mpjoin_updateprefsize(indexrecord, indreclen, minoverlap);
				}

				inline IndexListEntry & operator*() {
					assert(curelem >= 0);
					return ilheader->ilist[curelem];
				}

				inline IndexListEntry * operator->() {
					assert(curelem >= 0);
					return &ilheader->ilist[curelem];
				}

			};


			inline static unsigned int verify_indexrecord_start(const IndexedRecord & indexrecord, unsigned int indreclen, Algorithm * algo) {
				return MpJoinCommonMethods<IndexStructure>::verify_indexrecord_start(indexrecord, indreclen, algo);
			}

			inline void addtoken(unsigned int token, unsigned int recind, unsigned int recpos) {
				typename Index::value_type & ilhead = index.get_list_create(token);
				if(ilhead.firstvalid == -1) {
					ilhead.firstvalid = ilhead.ilist.size();
				}
				ilhead.ilist.push_back(IndexListEntry(recind, recpos));
			}

			inline iterator getiterator(unsigned int token) {
				typename Index::value_type * ilhead = index.get_list(token);
				if( ilhead != NULL ) {
					return iterator(ilhead);
				} else {
					return iterator();
				}
			}
			void largest_tokenid(unsigned int tokenid) {
				index.resize(tokenid + 1);
			}
		};
};

class MpJoinArrayLinkedIndexPolicy {

	public:
		struct IndexStructureRecordData {
			unsigned int prefsize;
			IndexStructureRecordData() : prefsize(INT_MAX) {}
			inline void setInitialPrefSize(unsigned int prefsize) {}
		};

		template <typename Algorithm, typename IndexedRecord_ = typename Algorithm::IndexedRecord>
		struct IndexStructure {

			struct IndexListEntry : Algorithm::GenericIndexListEntry {
				typedef typename Algorithm::GenericIndexListEntry GenericIndexListEntry;
				int position;
				IndexListEntry(unsigned int recordid, int position) : GenericIndexListEntry(recordid), position(position) {}
				IndexListEntry() : GenericIndexListEntry(0) {}
			};

			struct IndexListHeader {
				std::vector<IndexListEntry> ilist;
				int firstvalid;
				IndexListHeader() : firstvalid(0) {}
			};

			typedef IndexedRecord_ IndexedRecord;
			typedef InvIndexVector::template IndexMap< IndexListHeader > Index;
			typedef IndexStructure<Algorithm, IndexedRecord> self_type;
			Index index;

			IndexListHeader dummyil;

			struct iterator {
				int curelem;
				int lastptr;

				IndexListHeader * ilheader;
				
				typedef MpJoinCommonMethods<self_type> commonMethods;

				// Empty iterator if there are no entries
				inline iterator() : curelem(0), lastptr(-1) {}

				inline iterator(IndexListHeader * ilheader) :
					ilheader(ilheader), lastptr(-1) {
						curelem = ilheader->firstvalid;
					}

				inline bool end() const {
					return curelem == ilheader->ilist.size();
				}

				inline void next() {
					assert(curelem != ilheader->ilist.size());
					curelem +=1;
					lastptr = curelem;

					if(curelem == ilheader->ilist.size()) {
						return;
					}

					int pos = ilheader->ilist[curelem].position;

					if(pos < 0) {
						curelem -= pos;
					}
				}

				inline void next_deletecur() {
					assert(curelem != ilheader->ilist.size());
					//Covers all firstvalid cases
					if(curelem == ilheader->firstvalid) {
						next();
						if(curelem == ilheader->ilist.size()) {
							// 3a
							ilheader->ilist.resize(0);
							curelem = 0;
						}
						ilheader->firstvalid = curelem;
						return;
					}

					assert(lastptr >= 0);

					//3b
					if(curelem + 1 == ilheader->ilist.size()) {
						int reducesize = 1;
						if(ilheader->ilist[lastptr].position < 0) {
							//3bb
							reducesize -= ilheader->ilist[lastptr].position;
						}
						ilheader->ilist.resize(curelem - reducesize + 1);
						curelem = ilheader->ilist.size();
						return;
					}

					// 2b
					if(ilheader->ilist[curelem + 1].position >= 0) {
						if(ilheader->ilist[lastptr].position >= 0) {
							//2ba
							ilheader->ilist[curelem].position = -1;
						} else {
							//2bb
							ilheader->ilist[lastptr].position -= 1;
						}
						curelem += 1;
						return;
					}

					// 1b
					if(ilheader->ilist[curelem + 1].position < 0) {
						int reducesize = ilheader->ilist[curelem + 1].position;
						if(ilheader->ilist[lastptr].position < 0) {
							//1bb
							ilheader->ilist[lastptr].position += reducesize - 1;
						} else {
							//1ba
							ilheader->ilist[curelem].position = reducesize - 1;
						}
						curelem += -reducesize + 1;
					}
				}

				inline bool lengthfilter(unsigned int indreclen, unsigned int minsize) {
					return commonMethods::lengthfilter(*this, indreclen, minsize);
				}


				inline bool mpjoin_removestale1(IndexedRecord & indexedrecord, unsigned int position) {
					return commonMethods::mpjoin_removestale1(*this, indexedrecord, position);
				}

				inline bool mpjoin_removestale2(IndexedRecord & indexedrecord, 
						unsigned int indreclen, unsigned int position, unsigned int minoverlap) {

					return commonMethods::mpjoin_removestale2(*this, indexedrecord, indreclen, position, minoverlap);
				}

				inline void mpjoin_updateprefsize(IndexedRecord & indexrecord, unsigned int indreclen, unsigned int minoverlap) {
					return commonMethods::mpjoin_updateprefsize(indexrecord, indreclen, minoverlap);
				}

				inline IndexListEntry & operator*() {
					assert(curelem >= 0);
					assert(curelem < ilheader->ilist.size());
					assert(ilheader->ilist[curelem].position >= 0);
					return ilheader->ilist[curelem];
				}

				inline IndexListEntry * operator->() {
					assert(curelem >= 0);
					assert(curelem < ilheader->ilist.size());
					assert(ilheader->ilist[curelem].position >= 0);
					return &ilheader->ilist[curelem];
				}

			};


			inline static unsigned int verify_indexrecord_start(const IndexedRecord & indexrecord, unsigned int indreclen, Algorithm * algo) {
				return MpJoinCommonMethods<IndexStructure>::verify_indexrecord_start(indexrecord, indreclen, algo);
			}

			inline void addtoken(unsigned int token, unsigned int recind, unsigned int recpos) {
				typename Index::value_type & ilhead = index.get_list_create(token);
				if(ilhead.firstvalid == -1) {
					ilhead.firstvalid = ilhead.ilist.size();
				}
				ilhead.ilist.push_back(IndexListEntry(recind, recpos));
			}

			inline iterator getiterator(unsigned int token) {
				typename Index::value_type * ilhead = index.get_list(token);
				if( ilhead != NULL ) {
					return iterator(ilhead);
				} else {
					return iterator(&dummyil);
				}
			}

			void largest_tokenid(unsigned int tokenid) {
				index.resize(tokenid + 1);
			}
		};
};



#endif
