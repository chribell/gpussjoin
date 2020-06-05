#include "device_timing.hxx"
#include <algorithm>

DeviceTiming::EventPair* DeviceTiming::add(const std::string & argName, cudaStream_t const& argStream) {

    EventPair* pair = new EventPair(argName, argStream);

    cudaEventCreate(&(pair->start));
    cudaEventCreate(&(pair->end));

    cudaEventRecord(pair->start, argStream);

    pairs.push_back(pair);
    return pair;
}

void DeviceTiming::finish(EventPair* pair) {
    cudaEventRecord(pair->end, pair->stream);
}

float DeviceTiming::total() const {
    float total = 0.0;
    std::vector<EventPair*>::const_iterator it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        float millis = 0.0;
        cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
        total += millis;
    }
    return total;
}


float DeviceTiming::sum(std::string const &argName) const {
    float total = 0.0;
    std::vector<EventPair*>::const_iterator it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        if ((*it)->name == argName) {
            float millis = 0.0;
            cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
            total += millis;
        }
    }
    return total;
}

DeviceTiming::~DeviceTiming() {
    std::vector<EventPair*>::iterator it = pairs.begin();
    for(; it != pairs.end(); ++it) {
        cudaEventDestroy((*it)->start);
        cudaEventDestroy((*it)->end);
        delete *it;
    }
}

std::ostream & operator<<(std::ostream & os, const DeviceTiming& timer) {
    std::vector<std::string> distinctNames;
    for(auto& pair : timer.pairs) {
        if (std::find(distinctNames.begin(), distinctNames.end(), pair->name) == distinctNames.end())
            distinctNames.push_back(pair->name);
    }
    for(auto& name : distinctNames) {
        os << name << std::setw(11) << timer.sum(name) << " ms" << std::endl;
    }

    return os;
}
