#include <cuda.h>
#include <cuda_runtime.h>
#include "structs.hxx"
#include "functions.hxx"

__global__ void scenarioA(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, bool aggregate);

__global__ void scenarioB(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, unsigned int maxSetSize, bool aggregate);

__global__ void scenarioC(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, unsigned int maxSetSize, bool aggregate);
