#ifndef GPUSSJOIN_STRUCTS_HXX
#define GPUSSJOIN_STRUCTS_HXX

#pragma once
#include <numeric>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline void __gpuAssert(cudaError_t stat, int line, std::string file) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "Error %s at line %d in file %s\n",
                cudaGetErrorString(stat), line, file.c_str());
        exit(1);
    }
}

#define gpuAssert(value)  __gpuAssert((value),(__LINE__),(__FILE__))

struct Collection
{
    typedef std::vector<unsigned int> Tokens;
    typedef std::vector<unsigned int> Sizes;
    typedef std::vector<unsigned int> Starts;
    Tokens tokens;
    Sizes sizes;
    Starts starts;

    Collection() = default;

    inline void addToken(unsigned int token)
    {
        tokens.push_back(token);
    }
    inline void addStart(unsigned int start)
    {
        starts.push_back(start);
    }
    inline void addSize(unsigned int size)
    {
        sizes.push_back(size);
    }
};

template <class T>
struct HostCandidates
{
    typedef std::vector<T> Candidates;
    typedef std::vector<T> Probes;
    typedef std::vector<unsigned int> Starts;
    typedef std::vector<unsigned int> Sizes;
    Candidates candidates;
    Probes probes;
    Starts starts;
    Sizes sizes;

    HostCandidates() = default;

    HostCandidates(size_t numOfCandidates, size_t numOfProbes) {
        candidates.reserve(numOfCandidates);
        probes.reserve(numOfProbes);
        starts.reserve(numOfProbes);
        sizes.reserve(numOfProbes);
    }
    HostCandidates(HostCandidates<T>& hostCandidates) {
        candidates.swap(hostCandidates.candidates);
        probes.swap(hostCandidates.probes);
        starts.swap(hostCandidates.starts);
        sizes.swap(hostCandidates.sizes);
        hostCandidates.clear(); // lazy
    }

    inline void clear() {
        candidates.clear();
        probes.clear();
        starts.clear();
        sizes.clear();
    }
};

template <class T>
struct DeviceArray {
    T* array;
    size_t length = 0;

    inline void init(size_t len, T* arrayPtr)
    {
        length = len;
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemcpy(array, arrayPtr, length * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void init(std::vector<T>& vector)
    {
        length = vector.size();
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemcpy(array, &vector[0], length * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void init(size_t len)
    {
        length = len;
        gpuAssert(cudaMalloc((void**) &array, length * sizeof(T)));
        gpuAssert(cudaMemset(array, 0, length * sizeof(T)));
    }

    inline void zero()
    {
        gpuAssert(cudaMemset(array, 0, length * sizeof(T)));
    }

    inline void free()
    {
        gpuAssert(cudaFree(array));
        length = 0;
    }

    __forceinline__ __host__ __device__ T& operator[] (unsigned int i)
    {
        return array[i];
    }
};


struct DeviceCollection {
    DeviceArray<unsigned int> tokens;
    DeviceArray<unsigned int> starts;
    DeviceArray<unsigned int> sizes;

    size_t numberOfSets = 0;
    size_t numberOfTokens = 0;

    inline void init(Collection& collection)
    {
        numberOfSets    = collection.sizes.size();
        numberOfTokens  = collection.tokens.size();

        sizes.init(collection.sizes);
        starts.init(collection.starts);
        tokens.init(collection.tokens);
    }

    __forceinline__ __device__ unsigned int* tokenAt(size_t pos)
    {
        return tokens.array + pos;
    }

    inline void free() {
        sizes.free();
        starts.free();
        tokens.free();
    }

    inline size_t requiredMemory()
    {
        size_t memory = 0;
        memory += sizes.length * sizeof(unsigned int);
        memory += starts.length * sizeof(unsigned int);
        memory += tokens.length * sizeof(unsigned int);
        return memory;
    }
};

template <class T>
struct DeviceCandidates {
    DeviceArray<T> candidates;
    DeviceArray<T> probes;
    DeviceArray<unsigned int> starts;
    DeviceArray<unsigned int> sizes;

    inline void init(HostCandidates<T>& hostCandidates) {
        candidates.init(hostCandidates.candidates);
        probes.init(hostCandidates.probes);
        starts.init(hostCandidates.starts);
        sizes.init(hostCandidates.sizes);
    }

    inline void free() {
        candidates.free();
        probes.free();
        starts.free();
        sizes.free();
    }

};

template <class T>
struct HostArray {
    typedef std::vector<T> Array;
    Array array;

    HostArray() = default;

    inline void init(DeviceArray<T>& deviceArray) {
        array.reserve(deviceArray.length);
        gpuAssert(cudaMemcpy(&array[0], deviceArray.array, deviceArray.length * sizeof(T), cudaMemcpyDeviceToHost));
    }
};


#endif // GPUSSJOIN_STRUCTS_HXX
