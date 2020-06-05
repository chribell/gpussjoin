#ifndef GPUSSJOIN_DEVICE_TIMING_HXX
#define GPUSSJOIN_DEVICE_TIMING_HXX

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>


class DeviceTiming {
public:
    struct EventPair {
        std::string name;
        cudaEvent_t start;
        cudaEvent_t end;
        cudaStream_t stream;
        EventPair(std::string const& argName, cudaStream_t const& argStream) : name(argName), stream(argStream)  {}
    };
protected:
    std::vector<EventPair*> pairs;
public:
    EventPair* add(std::string const& argName, cudaStream_t const& argStream);
    float sum(std::string const& argName) const;
    float total() const;
    void finish(EventPair* pair);
    ~DeviceTiming();
    friend std::ostream & operator<<(std::ostream& os, const DeviceTiming& timer);
};


#endif // GPUSSJOIN_DEVICE_TIMING_HXX
