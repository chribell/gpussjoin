#include "scenarios.hxx"

__global__ void scenarioA(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, bool aggregate)
{
    extern __shared__ unsigned int counts[];

    // too serial, seriously
    unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x; // global thread id 1D grid of 1D blocks

    unsigned int recordStart = 0;
    unsigned int recordEnd = 0;

    unsigned int candidatesStart = 0;
    unsigned int candidatesEnd = 0;

    // candidateOffsets[globalID * 2] gives us the probe set id
    unsigned int probeSetID = candidates.offsets[globalID * 2];

    recordStart = probeCollection.starts[probeSetID];
    recordEnd = recordStart + probeCollection.sizes[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = globalID > 0 ? candidates.offsets[(globalID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidates.offsets[(globalID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    unsigned int recordLength = recordEnd - recordStart;
    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart; offset < candidatesEnd; ++offset) {
        unsigned int candidate = candidates.candidates[offset];

        unsigned int candidateRecordStart = indexedCollection.starts[candidate];
        unsigned int candidateRecordEnd = indexedCollection.starts[candidate] + indexedCollection.sizes[candidate];

        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;

        double withoutCeil = (recordLength + candidateLength) * threshold / (1 + threshold);
        unsigned int jaccardEqovelarp = myMin(candidateLength, myMin(recordLength, (unsigned int)(ceil(withoutCeil))));

        if ( intersection(
                probeCollection.tokens.array,
                indexedCollection.tokens.array,
                recordStart, recordEnd,
                candidateRecordStart, candidateRecordEnd)  >= jaccardEqovelarp ) {
            if (aggregate) {
                counter++;
            } else {
                results[offset] = 1;
            }
        }
    }
    if (aggregate) {
        counts[threadIdx.x] = counter;

        reduce(threadIdx.x, blockDim.x, counts, &counter);

        if (threadIdx.x == 0) results[blockIdx.x] = counter;
    }
}

__global__ void scenarioB(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, unsigned int maxSetSize, bool aggregate)
{
    extern __shared__ unsigned int shared[];
    unsigned int* querySet = shared;
    unsigned int* counts = (unsigned int*) &querySet[maxSetSize];

    unsigned int blockID = blockIdx.x;

    unsigned int recordStart = 0;
    unsigned int recordEnd = 0;

    unsigned int candidatesStart = 0;
    unsigned int candidatesEnd = 0;

    // candidateOffsets[globalID * 2] gives us the probe set id
    unsigned int probeSetID = candidates.offsets[blockID * 2];

    recordStart = probeCollection.starts[probeSetID];
    recordEnd = recordStart + probeCollection.sizes[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = blockID > 0 ? candidates.offsets[(blockID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidates.offsets[(blockID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    // load query set to shared memory
    for(int i = threadIdx.x; i  < maxSetSize && i < recordEnd - recordStart; i += blockDim.x) {
        querySet[i] = probeCollection.tokens[recordStart + i];
    }
    __syncthreads();

    unsigned int recordLength = recordEnd - recordStart;

    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart + threadIdx.x; offset < candidatesEnd; offset += blockDim.x) {
        unsigned int candidate = candidates.candidates[offset];

        unsigned int candidateRecordStart = indexedCollection.starts[candidate];
        unsigned int candidateRecordEnd = indexedCollection.starts[candidate] + indexedCollection.sizes[candidate];

        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;

        double withoutCeil = (recordLength + candidateLength) * threshold / (1 + threshold);
        unsigned int jaccardEqovelarp = myMin(candidateLength, myMin(recordLength, (unsigned int)(ceil(withoutCeil))));

        if ( intersection(
                querySet,
                indexedCollection.tokens.array,
                0, recordLength,
                candidateRecordStart, candidateRecordEnd)  >= jaccardEqovelarp ) {
            if (aggregate) {
                counter++;
            } else {
                results[offset] = 1;
            }
        }
    }
    if (aggregate) {
        counts[threadIdx.x] = counter;

        reduce(threadIdx.x, blockDim.x, counts, &counter);

        if (threadIdx.x == 0) results[blockIdx.x] = counter;
    }
}

__global__ void scenarioC(DeviceCollection indexedCollection, DeviceCollection probeCollection,
                          DeviceCandidates<unsigned int> candidates, DeviceArray<unsigned int> results,
                          double threshold, unsigned int maxSetSize, bool aggregate)
{
    extern __shared__ unsigned int shared[];
    unsigned int* querySet = shared;
    unsigned int* candidateSet = (unsigned int*) &querySet[maxSetSize];
    unsigned int* intersections = (unsigned int*) &candidateSet[maxSetSize];

    unsigned int blockID = blockIdx.x;

    unsigned int recordStart = 0;
    unsigned int recordEnd = 0;

    unsigned int candidatesStart = 0;
    unsigned int candidatesEnd = 0;

    // candidateOffsets[blockID * 2] gives us the probe set id
    unsigned int probeSetID = candidates.offsets[blockID * 2];

    // first record should be handle differently
    recordStart = probeCollection.starts[probeSetID];
    recordEnd = recordStart + probeCollection.sizes[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = blockID > 0 ? candidates.offsets[(blockID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidates.offsets[(blockID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    // load query set to shared memory
    for(int i = threadIdx.x; i  < maxSetSize && i < recordEnd - recordStart; i += blockDim.x) {
        querySet[i] = probeCollection.tokens[recordStart + i];
    }

    __syncthreads();

    unsigned int recordLength = recordEnd - recordStart;

    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart; offset < candidatesEnd; ++offset) {
        unsigned int candidate = candidates.candidates[offset];

        unsigned int candidateRecordStart = indexedCollection.starts[candidate];
        unsigned int candidateRecordEnd = indexedCollection.starts[candidate] + indexedCollection.sizes[candidate];

        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;
        unsigned int vt = ((recordLength + candidateLength + 1) / blockDim.x) + 1;


        for(int i = threadIdx.x; i  < maxSetSize && i < candidateLength; i += blockDim.x) {
            candidateSet[i] = indexedCollection.tokens[candidateRecordStart + i];
        }

        __syncthreads();

        unsigned int diag = threadIdx.x * vt;

        int mp = merge_path(querySet, recordEnd - recordStart, candidateSet, candidateRecordEnd - candidateRecordStart, diag);

        unsigned int isection = serial_intersect(querySet,
                                                 mp,
                                                 recordLength,
                                                 candidateSet,
                                                 diag - mp,
                                                 candidateLength,
                                                 vt);

        intersections[threadIdx.x] = isection;

        reduce(threadIdx.x, blockDim.x, intersections, &isection);

        __syncthreads();

        if (threadIdx.x == 0) {
            double withoutCeil = (recordLength + candidateLength) * threshold / (1 + threshold);
            unsigned int jaccardEqovelarp = myMin(candidateLength, myMin(recordLength, (unsigned int)(ceil(withoutCeil))));

            if (isection >= jaccardEqovelarp) {
                if (aggregate) {
                    counter++;
                } else {
                    results[offset] = true;
                }
            }
        }


    }
    if (aggregate && threadIdx.x == 0) {
        results[blockIdx.x] = counter;
    }
}