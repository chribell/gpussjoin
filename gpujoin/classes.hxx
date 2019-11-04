#ifndef GPU_JOIN_CLASSES_HXX
#define GPU_JOIN_CLASSES_HXX

#pragma once
#include <numeric>
#include <cuda/api_wrappers.hpp>

#include "gputimer.hxx"

template <class T>
class HostVector
{
private:
    typedef std::vector<T> Vector;
    typedef std::vector<unsigned int> Offsets;
    Vector _vector;
    Offsets _offsets;
public:
    HostVector(){}
    HostVector(const HostVector<T>& array)
            : _vector(array._vector),
              _offsets(array._offsets) {}

    inline void addElement(T element) {
        _vector.push_back(element);
    }
    inline void addOffset(unsigned int offset) {
        _offsets.push_back(offset);
    }
    inline Vector& getVector() { return _vector; }
    inline Offsets& getOffsets() { return _offsets; }
    inline void clear() {
        _vector.clear();
        _vector.shrink_to_fit();
        _offsets.clear();
        _offsets.shrink_to_fit();
    }
    inline void swap(HostVector& hostVector) {
        _vector.swap(hostVector.getVector());
        _offsets.swap(hostVector.getOffsets());
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
class DeviceArray
{
private:
    typedef T Array;
    typedef unsigned int Offsets;
    Array* _array;
    Offsets* _offsets;
public:
    DeviceArray(){}
    inline void init(int deviceID, HostArray<T>& hostArray) {
        _array   =  (T*) cuda::device::get(deviceID).memory().allocate(hostArray.getArraySize() * sizeof(T));
        _offsets =  (unsigned int*) cuda::device::get(deviceID).memory().allocate(hostArray.getOffsetsSize() * sizeof(unsigned int));
        cuda::memory::copy(_array, &hostArray.getArray()[0], hostArray.getArraySize() * sizeof(T));
        cuda::memory::copy(_offsets, &hostArray.getOffsets()[0], hostArray.getOffsetsSize() * sizeof(unsigned int));
    }

    inline void init(int deviceID, HostVector<T>& hostVector) {
        _array   =  (T*) cuda::device::get(deviceID).memory().allocate(hostVector.getVector().size() * sizeof(T));
        _offsets =  (unsigned int*) cuda::device::get(deviceID).memory().allocate(hostVector.getOffsets().size() * sizeof(unsigned int));
        cuda::memory::copy(_array, &hostVector.getVector()[0], hostVector.getVector().size() * sizeof(T));
        cuda::memory::copy(_offsets, &hostVector.getOffsets()[0], hostVector.getOffsets().size() * sizeof(unsigned int));
    }

    inline Array* getArray() { return _array; }
    inline Offsets* getOffsets() { return _offsets; }

    inline void free() {
        cuda::memory::device::free(_array);
        cuda::memory::device::free(_offsets);
    }
};

template <class T>
class DeviceResult
{
private:
    typedef T Result;
    unsigned int _length;
    Result* _result;
public:
    DeviceResult(){}

    inline void init(int deviceID, unsigned int length) {
        _length = length;
        _result = (T*) cuda::device::get(deviceID).memory().allocate(_length  * sizeof(T));
        cuda::memory::device::set(_result, 0, _length  * sizeof(T));
    }

    inline Result* getResult() { return _result; }
    inline unsigned int getLength() { return _length; }

    inline void free() {
        cuda::memory::device::free(_result);
    }
};

template <class T>
class HostResult
{
private:
    typedef T Result;
    unsigned int _length;
    Result* _result;
public:
    HostResult(){}
    inline void init(int deviceID, DeviceResult<T> deviceResult, unsigned int length) {
        _length = length;
        _result = new T[length];
        cuda::memory::copy(_result, deviceResult.getResult(), length * sizeof(T));
    }

    inline Result* getResult() { return _result; }
    inline unsigned int getLength() { return _length; }

    inline void free() {
        delete[] _result;
    }

};


typedef DeviceResult<unsigned int> DeviceCountResult;
typedef DeviceResult<bool> DevicePairResult;
typedef HostResult<unsigned int> HostCountResult;
typedef HostResult<bool> HostPairResult;


typedef struct gpu_info
{
    unsigned int offset;
    unsigned int numberOfRecords;
    unsigned int numberOfCandidates;
    unsigned int maxSetSize;
    bool completeChunk;

    gpu_info(unsigned int off, unsigned int numOfRecords, unsigned int numOfCandidates,
             unsigned int setSize, bool cChunk)
            : offset(off),
              numberOfRecords(numOfRecords),
              numberOfCandidates(numOfCandidates),
              maxSetSize(setSize),
              completeChunk(cChunk) {}

    inline friend std::ostream & operator<<(std::ostream & os, const gpu_info & offsets){
        os << std::setw(25) << "Offset: " << std::setw(16) << offsets.offset << std::endl;
        os << std::setw(25) << "Max set size: " << std::setw(16) << offsets.maxSetSize << std::endl;
        os << std::setw(25) << "Number of records: " << std::setw(16) << offsets.numberOfRecords << std::endl;
        os << std::setw(25) << "Candidates end: " << std::setw(16) << offsets.numberOfCandidates << std::endl;
        return os;
    }
} gpu_info;

#endif //GPU_JOIN_CLASSES_HXX