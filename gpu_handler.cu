#include "gpu_handler.h"
#include "gpujoin/classes.hxx"
#include "gpujoin/scenarios.hxx"
#include <iostream>
#include <ostream>
#include <fstream>


#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>

/**
 * Transfer dataset tokens to device
 */
void GPUHandler::transferTokensToDevice() {
//    GPUTimer::EventPair* tokensTransfer = _gpuTimer.add("allocate_transfer_tokens", nullptr);
    _deviceTokens.init(_deviceID, _hostTokens);
//    _gpuTimer.finish(tokensTransfer);
}

/**
 * Free device memory used for tokens
 */
void GPUHandler::freeTokensFromDevice() {
//    GPUTimer::EventPair* freeTokens = _gpuTimer.add("free_tokens", nullptr);
    _deviceTokens.free();
//    _gpuTimer.finish(freeTokens);
}

/**
 * Transfer candidates to device
 */
void GPUHandler::transferCandidatesToDevice() {
//    auto start = cuda::event::create(_deviceID, true);
//    auto stop = cuda::event::create(_deviceID, true);
//    start.record(cuda::stream::default_stream_id);
    _deviceCandidates.init(_deviceID, *_chunkCandidates);
//    stop.record(cuda::stream::default_stream_id);
//    stop.synchronize();
//    transferTime += cuda::event::milliseconds_elapsed_between(start, stop);
}

/**
 * Free device memory used for candidates
 */
void GPUHandler::freeCandidatesFromDevice() {
//    GPUTimer::EventPair* freeCandidates = _gpuTimer.add("free_candidates", nullptr);
    _deviceCandidates.free();
//    _gpuTimer.finish(freeCandidates);
}

void GPUHandler::reserveCandidateSpace(size_t size) {
    _numberOfRecords = size;
    _hostCandidates = new HostArray<unsigned int>(_maxNumberOfCandidates, _numberOfRecords);
}

/**
 * Post-process data and launch device context
 */
void GPUHandler::doJoin()
{
    // wait for gpu thread to finish (if it's running)
    if (_gpuThread.joinable()) _gpuThread.join();

    //if (_isGroupJoin) this->serializeMap();

    // move candidates to another array, so the cpu can continue to
    // use the main canidates array
    _chunkCandidates = new HostArray<unsigned int>(*_hostCandidates);
    _hostCandidates = new HostArray<unsigned int>(_maxNumberOfCandidates, _numberOfRecords);

    unsigned int numOfRecords = _chunkCandidates->getOffsetsSize() / 2;

    // start the gpu join process
    _gpuThread = std::thread(&GPUHandler::invokeGPU, this, numOfRecords);

    _numOfCandidates = 0;
}

void GPUHandler::invokeGPU(unsigned int numOfRecords)
{
    this->transferCandidatesToDevice();
    DefaultLaunchParameters launchParameters(_scenario, _aggregate, _numOfThreads, numOfRecords, _maxSetSize);

    if (_aggregate) {
        _deviceCounts.init(_deviceID, launchParameters.blocks);
    } else {
        _devicePairs.init(_deviceID, _chunkCandidates->getArraySize());
    }
    // TODO refactor for a more elegant gpu call :D

    typedef void (*AggregateKernel)( unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int, double);
    typedef void (*PairsPairKernel)( unsigned int*, unsigned int*, unsigned int*, unsigned int*, bool*, unsigned int, double);

    std::unordered_map<std::string, AggregateKernel> aggregateMap;
    aggregateMap["A"] = &scenarioA<true, unsigned int, unsigned int, unsigned int>;
    aggregateMap["B"] = &scenarioB<true, unsigned int, unsigned int, unsigned int>;
    aggregateMap["C"] = &scenarioC<true, unsigned int, unsigned int, unsigned int>;

    std::unordered_map<std::string, PairsPairKernel> pairsMap;
    pairsMap["A"] = &scenarioA<false, unsigned int, unsigned int, bool>;
    pairsMap["B"] = &scenarioB<false, unsigned int, unsigned int, bool>;
    pairsMap["C"] = &scenarioC<false, unsigned int, unsigned int, bool>;


    auto start_event = cuda::event::create(_deviceID, true);
    auto stop_event = cuda::event::create(_deviceID, true);

    start_event.record(cuda::stream::default_stream_id);
    if (_aggregate) {
        cuda::enqueue_launch(
                aggregateMap.at(_scenarioKey), cuda::stream::default_stream_id,
                launchParameters.makeConfig(),
                _deviceTokens.getArray(), _deviceTokens.getOffsets(),
                _deviceCandidates.getArray(), _deviceCandidates.getOffsets(),
                _deviceCounts.getResult(), _maxSetSize, _threshold
        );
    } else {
        cuda::enqueue_launch(
                pairsMap.at(_scenarioKey), cuda::stream::default_stream_id,
                launchParameters.makeConfig(),
                _deviceTokens.getArray(), _deviceTokens.getOffsets(),
                _deviceCandidates.getArray(), _deviceCandidates.getOffsets(),
                _devicePairs.getResult(), _maxSetSize, _threshold
        );
    }
    cuda::device::get(_deviceID).synchronize();

    stop_event.record(cuda::stream::default_stream_id); // record in stream-0, to ensure that all previous CUDA calls have completed
    stop_event.synchronize();

    auto kernel_time = cuda::event::time_elapsed_between(start_event, stop_event);
    gpuTime += kernel_time.count();

    if (_aggregate) {
        _result += thrust::reduce(thrust::device, _deviceCounts.getResult(), _deviceCounts.getResult() + _deviceCounts.getLength());
        _deviceCounts.free();
    } else {
        std::ofstream of;
        std::ostream * os = &std::cout;
        of.open("test", std::ofstream::out | std::ofstream::app);

        os = &of;
        _hostPairs.init(_deviceID, _devicePairs, _devicePairs.getLength());
        _devicePairs.free();
        for (int i = 0; i < _chunkCandidates->getOffsetsSize(); i+=2) {
            unsigned int probeID = _chunkCandidates->getOffsets()[i];
            unsigned int candidatesStart = i > 0 ? _chunkCandidates->getOffsets()[i - 1] : 0;
            unsigned int candidatesEnd = _chunkCandidates->getOffsets()[i + 1];
            for (int j = candidatesStart; j < candidatesEnd; j++)
                if (_hostPairs.getResult()[j]) {
                    *os << probeID << " " << _chunkCandidates->getArray()[j] << std::endl;
                }
        }
        _hostPairs.free();
    }

    this->freeCandidatesFromDevice();
    _chunkCandidates->~HostArray();
}


void GPUHandler::insertToken(unsigned int token) { _hostTokens.addElement(token); }

void GPUHandler::insertTokenOffset(unsigned int offset) { _hostTokens.addOffset(offset); }

void GPUHandler::insertCandidate(unsigned int probe, unsigned int candidate)
{
    _hostCandidates->addElement(candidate);

    // we deal with any candidates which are not mapped to a probe record
    // in the doJoin function
    _numOfCandidates++;
    if (_numOfCandidates == _maxNumberOfCandidates) {
        this->updateCandidateOffset(probe);
        this->doJoin();
    }
}

void GPUHandler::updateCandidateOffset(unsigned int probe)
{
    _hostCandidates->addOffset(probe);
    _hostCandidates->addOffset(_hostCandidates->getArraySize());
}

void GPUHandler::updateMaxSetSize(unsigned int maxSetSize)
{
    if (maxSetSize > _maxSetSize) _maxSetSize = maxSetSize;
}

void GPUHandler::flush()
{
    if (_numOfCandidates > 0) this->doJoin();
}

size_t GPUHandler::getResult()
{
    _gpuThread.join();
    return _result;
}

void GPUHandler::updateCandidateMap(unsigned int probeRecordID, unsigned int candidateID)
{
    _candidateMap[probeRecordID].push_back(candidateID);
    _numOfCandidates++;
    if (_numOfCandidates == _maxNumberOfCandidates) this->doJoin();
}

void GPUHandler::serializeMap()
{
    for (auto it : _candidateMap) {
        for (auto v = it.second.begin(); v != it.second.end(); v++)
            _hostCandidates->addElement(*v);
        ;
        _hostCandidates->addOffset(it.first);
        _hostCandidates->addOffset(_hostCandidates->getArraySize());

        it.second.clear();
        it.second.shrink_to_fit();
    }
    _candidateMap.clear();
}

float GPUHandler::getGPUJoinTime()
{
//    return _gpuTimer.sum("GPUJoin");
    return gpuTime;
}

float GPUHandler::getGPUTransferTime()
{
    return transferTime;
}