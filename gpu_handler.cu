#include "gpu_handler.h"
#include "gpujoin/scenarios.hxx"
#include <ostream>
#include <fstream>


#include <thrust/reduce.h>
#include <thrust/device_vector.h>


/**
 * Create kernel launch parameters
 */
void GPUHandler::makeLaunchParameters()
{
    if (_scenario == 1) {
        _blocks = (_deviceCandidates.probes.length + _threads - 1) / _threads;
        _sharedMemory = _aggregate ? _threads * sizeof(unsigned int) : 0;
    } else if (_scenario == 2) {
        _blocks = _deviceCandidates.probes.length;
        _sharedMemory = _maxSetSize * sizeof(unsigned int) + (_aggregate ? _threads * sizeof(unsigned int) : 0);
    } else {
        _blocks = _deviceCandidates.probes.length;
        _sharedMemory = 2 * _maxSetSize * sizeof(unsigned int) + _threads * sizeof(unsigned int);
    }
}

/**
 * Transfer input collection to device
 */
void GPUHandler::transferInputCollection() {
    DeviceTiming::EventPair* inputTransfer = _deviceTimings.add( "Transfer input collection to device (allocate + copy)", 0);
    _deviceInputCollection.init(_hostInputCollection);
    _deviceTimings.finish(inputTransfer);
}

/**
 * Free device memory used for input collection
 */
void GPUHandler::freeInputCollection() {
    DeviceTiming::EventPair* inputFree = _deviceTimings.add("Free input collection from device memory", 0);
    _deviceInputCollection.free();
    _deviceTimings.finish(inputFree);
}

/**
 * Transfer foreign input collection to device
 */
void GPUHandler::transferForeignInputCollection() {
    DeviceTiming::EventPair* foreignTransfer = _deviceTimings.add( "Transfer foreign input collection to device (allocate + copy)", 0);
    _deviceForeignCollection.init(_hostForeignInputCollection);
    _deviceTimings.finish(foreignTransfer);
}

/**
 * Free device memory used for foreign input collection
 */
void GPUHandler::freeForeignInputCollection() {
    DeviceTiming::EventPair* inputFree = _deviceTimings.add("Free foreign input collection from device memory", 0);
    _deviceForeignCollection.free();
    _deviceTimings.finish(inputFree);
}

/**
 * Transfer candidates to device
 */
void GPUHandler::transferCandidatesToDevice() {

    DeviceTiming::EventPair* candidatesTransfer = _deviceTimings.add( "Transfer candidates to device (allocate + copy)", 0);
    _deviceCandidates.init(*_chunkCandidates);
    _deviceTimings.finish(candidatesTransfer);
}

/**
 * Free device memory used for candidates
 */
void GPUHandler::freeCandidatesFromDevice() {
    DeviceTiming::EventPair* candidatesFree = _deviceTimings.add("Free candidates from device memory", 0);
    _deviceCandidates.free();
    _deviceTimings.finish(candidatesFree);
}

void GPUHandler::reserveCandidateSpace() {
    _numberOfRecords = _binaryJoin ? _hostForeignInputCollection.sizes.size() : _hostInputCollection.sizes.size();
    _hostCandidates = new HostCandidates<unsigned int>(_maxNumberOfCandidates, _numberOfRecords);
}

void GPUHandler::addIndexedRecord(std::vector<unsigned int> tokens)
{
    _hostInputCollection.addStart(_hostInputCollection.tokens.size());
    for (const auto& x : tokens) {
        _hostInputCollection.addToken(x);
	}
    _hostInputCollection.addSize(tokens.size());
}

void GPUHandler::addForeignRecord(std::vector<unsigned int> tokens)
{
    _hostForeignInputCollection.addStart(_hostForeignInputCollection.tokens.size());
    for (const auto& x : tokens) {
        _hostForeignInputCollection.addToken(x);
    }
    _hostForeignInputCollection.addSize(tokens.size());
}

/**
 * Post-process data and launch device context
 */
void GPUHandler::doJoin()
{
    // wait for gpu thread to finish (if it's running)
    if (_gpuThread.joinable()) _gpuThread.join();

    // move candidates to another array, so the cpu can continue to
    // use the main candidates array
    _chunkCandidates = new HostCandidates<unsigned int>(*_hostCandidates);
    _hostCandidates = new HostCandidates<unsigned int>(_maxNumberOfCandidates, _numberOfRecords);

    unsigned int numOfRecords = _chunkCandidates->probes.size();

    // start the gpu join process
    _gpuThread = std::thread(&GPUHandler::invokeGPU, this, numOfRecords);

    _numberOfCandidates = 0;
}

void GPUHandler::invokeGPU(unsigned int numOfRecords)
{
    transferCandidatesToDevice();
    makeLaunchParameters();

    if (_aggregate) {
        _deviceResults.init(_blocks);
    } else {
        _deviceResults.init(_chunkCandidates->candidates.size());
    }

    DeviceTiming::EventPair* joinTime = _deviceTimings.add("Conduct GPU join", 0);
    if (_scenario == 1) {
        scenarioA <<<_blocks, _threads, _sharedMemory >>>(
                _deviceInputCollection,
                _binaryJoin ? _deviceForeignCollection : _deviceInputCollection,
                _deviceCandidates,
                _deviceResults,
                _threshold,
                _aggregate);
    } else if (_scenario == 2) {
        scenarioB <<<_blocks, _threads, _sharedMemory >>>(
                _deviceInputCollection,
                 _binaryJoin ? _deviceForeignCollection : _deviceInputCollection,
                 _deviceCandidates,
                 _deviceResults,
                 _threshold,
                 _maxSetSize,
                 _aggregate);
    } else {
        scenarioC <<<_blocks, _threads, _sharedMemory >>>(
                _deviceInputCollection,
                _binaryJoin ? _deviceForeignCollection : _deviceInputCollection,
                _deviceCandidates,
                _deviceResults,
                _threshold,
                _maxSetSize,
                _aggregate);
    }

    if (_aggregate) {
        thrust::device_ptr<unsigned int> thrustCount(_deviceResults.array);
        _result += thrust::reduce(thrust::device, thrustCount, thrustCount + _deviceResults.length);
        _deviceResults.free();
    } else {
        std::ofstream of;
        std::ostream * os = &std::cout;
        of.open(_outFile, std::ofstream::out);

        os = &of;
        HostArray<unsigned int> results;
        results.init(_deviceResults);
        for (unsigned int i = 0; i < _chunkCandidates->probes.size(); ++i){
            unsigned int probeID = _chunkCandidates->probes[i];
            unsigned int candidateStart = _chunkCandidates->starts[i];
            unsigned int candidateEnd = candidateStart + _chunkCandidates->sizes[i];
            for (unsigned int j = candidateStart; j < candidateEnd; ++j) {
                if (results.array[j]) {
                    *os << probeID << " " << _chunkCandidates->candidates[j] << std::endl;
                }
            }
        }
        of.close();
    }

    _deviceTimings.finish(joinTime);

    this->freeCandidatesFromDevice();
    _chunkCandidates->clear();
}

void GPUHandler::insertProbe(unsigned int probe)
{
    _hostCandidates->probes.push_back(probe);
    _hostCandidates->starts.push_back(_hostCandidates->sizes.size() > 0
        ? _hostCandidates->starts[_hostCandidates->starts.size() - 1] + _hostCandidates->sizes[_hostCandidates->sizes.size() - 1]
        : 0);
    _candidateCounter = 0;
}

void GPUHandler::updateProbeNumberOfCandidates()
{
    _hostCandidates->sizes.push_back(_candidateCounter);
}

void GPUHandler::insertCandidate(unsigned int candidate)
{
    _hostCandidates->candidates.push_back(candidate);
    _candidateCounter++;

    // we deal with any candidates which are not mapped to a probe record
    // in the doJoin function
    _numberOfCandidates++;
    if (_numberOfCandidates == _maxNumberOfCandidates) {
        this->updateProbeNumberOfCandidates();
        _candidateCounter = 0;
        this->doJoin();
    }
}

void GPUHandler::updateMaxSetSize(unsigned int maxSetSize)
{
    if (maxSetSize > _maxSetSize) _maxSetSize = maxSetSize;
}

void GPUHandler::flush()
{
    if (_numberOfCandidates > 0) this->doJoin();
}

size_t GPUHandler::getResult()
{
    _gpuThread.join();
    return _result;
}

float GPUHandler::getGPUTotalTime()
{
    return _deviceTimings.total();
}

void GPUHandler::free()
{
    freeInputCollection();
    if (_binaryJoin) {
        freeForeignInputCollection();
    }
}