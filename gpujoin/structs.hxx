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
class HostArray
{
private:
    typedef T* Array;
    typedef unsigned int* Offsets;
    Array _array;
    Offsets _offsets;
    size_t _arraySize = 0;
    size_t _offsetsSize = 0;
public:
    HostArray(size_t arraySize, size_t offsetsSize){
        _array = new T[arraySize];
        _offsets = new unsigned int[offsetsSize * 2]; // fat offsets
        _arraySize = 0;
        _offsetsSize = 0;
    }
    HostArray(HostArray<T>& hostArray){
        std::swap(_array, hostArray._array);
        std::swap(_arraySize, hostArray._arraySize);
        std::swap(_offsets, hostArray._offsets);
        std::swap(_offsetsSize , hostArray._offsetsSize);
    }

    inline void addElement(T element) {
        _array[_arraySize++] = element;
    }
    inline void addOffset(unsigned int offset) {
        _offsets[_offsetsSize++] = offset;
    }
    inline Array getArray() { return _array; }
    inline Offsets getOffsets() { return _offsets; }
    inline size_t getArraySize() { return _arraySize; }
    inline size_t getOffsetsSize() { return _offsetsSize; }

    ~HostArray(){
        delete[] _array;
        delete[] _offsets;
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
    DeviceArray<unsigned int> candidates;
    DeviceArray<unsigned int> offsets;

    inline void init(HostArray<T>& hostArray) {
        candidates.init(hostArray.getArraySize(), hostArray.getArray());
        offsets.init(hostArray.getOffsetsSize(), hostArray.getOffsets());
    }

    inline void free() {
        candidates.free();
        offsets.free();
    }

};




#endif // GPUSSJOIN_STRUCTS_HXX
