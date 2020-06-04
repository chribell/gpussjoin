/**
 * A stepping stone header file for both CPU & GPU
 */
#ifndef GPUSSJOIN_DEFINITIONS_H
#define GPUSSJOIN_DEFINITIONS_H

#include <vector>

template <class elementtype>
class Record {
public:
    typedef std::vector<elementtype> Tokens;
    typedef unsigned int recordid_type;
    recordid_type recordid;
    Tokens tokens;
};


#endif //GPUSSJOIN_DEFINITIONS_H
