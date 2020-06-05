#ifndef GPUSSJOIN_HANDLER_H
#define GPUSSJOIN_HANDLER_H

#pragma once
#include <thread>
#include <unordered_map>
#include "gpujoin/structs.hxx"
#include "gpujoin/device_timing.hxx"
#include "definitions.h"

class GPUHandler
{
public:
    GPUHandler(unsigned scenario, bool aggregate,
               double threshold, unsigned int threads,
               size_t maxNumberOfCandidates, bool binaryJoin, const std::string& outFile) :
            _scenario(scenario),
            _aggregate(aggregate),
            _threshold(threshold),
            _threads(threads),
            _maxNumberOfCandidates(maxNumberOfCandidates),
            _binaryJoin(binaryJoin),
            _outFile(outFile) {};

    void addIndexedRecord(std::vector<unsigned int> tokens);
    void addForeignRecord(std::vector<unsigned int> tokens);

    void transferInputCollection();
    void transferForeignInputCollection();

    void insertProbe(unsigned int probe);

    void insertCandidate(unsigned int candidate);

    void updateProbeNumberOfCandidates();

    void updateMaxSetSize(unsigned int maxSetSize);
    void reserveCandidateSpace();
    void flush();
    void free();

    size_t getResult();
    float  getGPUTotalTime();
private:

    void freeInputCollection();
    void freeForeignInputCollection();

    void makeLaunchParameters();
    void doJoin();
    void invokeGPU(unsigned int numOfRecords);
    void transferCandidatesToDevice();
    void freeCandidatesFromDevice();

    DeviceTiming _deviceTimings;

    // constructor input
    unsigned int _scenario;
    bool _aggregate = true;
    double _threshold;
    unsigned int _threads;
    size_t _maxNumberOfCandidates = 0;

    // kernel specific
    unsigned int _blocks;
    size_t _sharedMemory = 0;

    bool _binaryJoin = false;
    unsigned int _maxSetSize = 0;
    size_t _numberOfRecords = 0;
    size_t _numberOfCandidates = 0;
    unsigned int _candidateCounter = 0;

    std::string _outFile;

    size_t _result = 0;

    std::thread _gpuThread;

    Collection _hostInputCollection;
    Collection _hostForeignInputCollection;

    DeviceCollection _deviceInputCollection;
    DeviceCollection _deviceForeignCollection;

    HostCandidates<unsigned int>* _hostCandidates;
    HostCandidates<unsigned int>* _chunkCandidates;

    DeviceCandidates<unsigned int> _deviceCandidates;
    DeviceArray<unsigned int> _deviceResults;
};

#endif // GPUSSJOIN_HANDLER_H
