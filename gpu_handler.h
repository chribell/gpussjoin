#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#pragma once
#include <thread>
#include <unordered_map>
#include "gpujoin/classes.hxx"


class GPUHandler
{
public:
    // 1 allpairs, 2 ppjoin, 3 groupjoin
    GPUHandler(unsigned int algorithm, unsigned scenario, bool aggregate,
               double threshold, unsigned int numOfThreads, size_t maxNumberOfCandidates) :
            _scenario(scenario),
            _aggregate(aggregate),
            _threshold(threshold),
            _numOfThreads(numOfThreads),
            _maxNumberOfCandidates(maxNumberOfCandidates) {
        std::string s = "A";
        switch (_scenario) {
            case 2:
                s = "B";
                break;
            case 3:
                s = "C";
                break;
            default:
                break;
        }
        if (algorithm == 3) {
            _isGroupJoin = true;
        }
        _scenarioKey = s;
        _deviceID = cuda::device::current::get().id();
    };

    template <class T>
    struct LaunchParameters {
        unsigned int blocks = 0;
        unsigned int threads = 0;
        cuda::shared_memory_size_t sharedMemory = 0;

        LaunchParameters(unsigned int scenario, bool count, unsigned int threads,
                         unsigned int numberOfRecords , unsigned int maxSetSize) : threads(threads){
            switch (scenario) {
                case 1:
                    blocks = (numberOfRecords + threads - 1 ) / threads;
                    sharedMemory = count ? threads * sizeof(unsigned int) : 0;
                    break;
                case 2:
                    blocks = numberOfRecords;
                    sharedMemory = maxSetSize * sizeof(T) + (count ? threads * sizeof(unsigned int) : 0);
                    break;
                case 3:
                    blocks = numberOfRecords;
                    sharedMemory = 2 * maxSetSize * sizeof(T) + threads * sizeof(unsigned int);
                    break;
                default:
                    break;
            }
        };

        inline cuda::launch_configuration_t makeConfig() {
            return cuda::make_launch_config(blocks, threads, sharedMemory);
        }

    };
    typedef LaunchParameters<unsigned int> DefaultLaunchParameters;

    void insertToken(unsigned int token);
    void insertTokenOffset(unsigned int offset);
    void insertCandidate(unsigned int probe, unsigned int candidate);
    void updateCandidateOffset(unsigned int probe);
    void updateMaxSetSize(unsigned int maxSetSize);
    void reserveCandidateSpace(size_t size);


    void updateCandidateMap(unsigned int probeRecordID, unsigned int candidateID);

    void transferTokensToDevice();
    void freeTokensFromDevice();
    void flush();

    size_t getResult();
    float  getGPUJoinTime();
    float  getGPUTransferTime();
private:
    void doJoin();
    void invokeGPU(unsigned int numOfRecords);
    void transferCandidatesToDevice();
    void freeCandidatesFromDevice();

    void serializeMap();

    int _deviceID;
    GPUTimer _gpuTimer;
    bool _isGroupJoin = false;
    bool _aggregate = false;

    HostVector<unsigned int> _hostTokens;
    HostArray<unsigned int>* _hostCandidates;
    HostArray<unsigned int>* _chunkCandidates;

    DeviceArray<unsigned int> _deviceTokens;
    DeviceArray<unsigned int> _deviceCandidates;

    DeviceCountResult _deviceCounts;
    DevicePairResult _devicePairs;
    HostPairResult _hostPairs;

    size_t _maxNumberOfCandidates = 0;
    size_t _numberOfRecords = 0;


    std::thread _gpuThread;

    size_t _result = 0;

    double _threshold;
    unsigned int _numOfThreads;
    std::string _scenarioKey;

    unsigned int _scenario = 1;

    float gpuTime = 0.0;
    float transferTime = 0.0;

    // used in groupjoin
    std::unordered_map<unsigned int, std::vector<unsigned int>> _candidateMap;
    size_t _numOfCandidates = 0;
    unsigned int _maxSetSize = 0;

};

#endif
