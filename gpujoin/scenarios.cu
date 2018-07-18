#include "scenarios.hxx"
#include "functions.hxx"

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioA(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold)
{
    extern __shared__ unsigned int counts[];

    // too serial, seriously
    unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x; // global thread id 1D grid of 1D blocks

    unsigned int recordStart = 0;
    unsigned int recordEnd = 0;

    unsigned int candidatesStart = 0;
    unsigned int candidatesEnd = 0;

    // candidateOffsets[blockID * 2] gives us the probe set id
    unsigned int probeSetID = candidateOffsets[globalID * 2];

    // first record should be handle differenlty
    recordStart = probeSetID > 0 ? tokenOffsets[probeSetID - 1] : 0;
    recordEnd = tokenOffsets[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = globalID > 0 ? candidateOffsets[(globalID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidateOffsets[(globalID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    unsigned int recordLength = recordEnd - recordStart;
    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart; offset < candidatesEnd; ++offset) {
        unsigned int candidate = candidates[offset];

        unsigned int candidateRecordStart = candidate > 0 ? tokenOffsets[candidate - 1] : 0;
        unsigned int candidateRecordEnd = tokenOffsets[candidate];

        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;

        double withoutCeil = (recordLength + candidateLength) * threshold / (1 + threshold);
        unsigned int jaccardEqovelarp = myMin(candidateLength, myMin(recordLength, (unsigned int)(ceil(withoutCeil))));

        if ( intersection(tokens, recordStart, recordEnd, candidateRecordStart, candidateRecordEnd)  >= jaccardEqovelarp ) {
            if (aggregate) {
                counter++;
            } else {
                results[offset] = true;
            }
        }
    }
    if (aggregate) {
        counts[threadIdx.x] = counter;

        reduce(threadIdx.x, blockDim.x, counts, &counter);

        if (threadIdx.x == 0) results[blockIdx.x] = counter;
    }
}

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioB(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold)
{
    extern __shared__ unsigned int shared[];
    unsigned int* querySet = shared;
    unsigned int* counts = (unsigned int*) &querySet[maxSetSize];

    unsigned int blockID = blockIdx.x;

    unsigned int recordStart = 0;
    unsigned int recordEnd = 0;

    unsigned int candidatesStart = 0;
    unsigned int candidatesEnd = 0;

    // candidateOffsets[blockID * 2] gives us the probe set id
    unsigned int probeSetID = candidateOffsets[blockID * 2];

    // first record should be handle differenlty
    recordStart = probeSetID > 0 ? tokenOffsets[probeSetID - 1] : 0;
    recordEnd = tokenOffsets[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = blockID > 0 ? candidateOffsets[(blockID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidateOffsets[(blockID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    // load query set to shared memory
    for(int i = threadIdx.x; i  < maxSetSize && i < recordEnd - recordStart; i += blockDim.x) {
        querySet[i] = tokens[recordStart + i];
    }
    __syncthreads();

    unsigned int recordLength = recordEnd - recordStart;

    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart + threadIdx.x; offset < candidatesEnd; offset += blockDim.x) {
        unsigned int candidate = candidates[offset];

        unsigned int candidateRecordStart = candidate > 0 ? tokenOffsets[candidate - 1] : 0;
        unsigned int candidateRecordEnd = tokenOffsets[candidate];


        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;

        double withoutCeil = (recordLength + candidateLength) * threshold / (1 + threshold);
        unsigned int jaccardEqovelarp = myMin(candidateLength, myMin(recordLength, (unsigned int)(ceil(withoutCeil))));

        if ( intersection_shared(tokens, querySet, recordStart, recordEnd, candidateRecordStart, candidateRecordEnd)  >= jaccardEqovelarp ) {
            if (aggregate) {
                counter++;
            } else {
                results[offset] = true;
            }

        }
    }
    if (aggregate) {
        counts[threadIdx.x] = counter;

        reduce(threadIdx.x, blockDim.x, counts, &counter);

        if (threadIdx.x == 0) results[blockIdx.x] = counter;
    }
}

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioC(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold)
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
    unsigned int probeSetID = candidateOffsets[blockID * 2];

    // first record should be handle differenlty
    recordStart = probeSetID > 0 ? tokenOffsets[probeSetID - 1] : 0;
    recordEnd = tokenOffsets[probeSetID];

    // candidateOffsets[(blockID * 2) + 1]
    candidatesStart = blockID > 0 ? candidateOffsets[(blockID * 2) - 1] : 0;

    // candidateOffsets[(blockID * 2) + 1] gives us the current's probe set candidates offset
    candidatesEnd = candidateOffsets[(blockID * 2) + 1];

    if (candidatesEnd - candidatesStart <= 0) return; // has no candidates

    // load query set to shared memory
    for(int i = threadIdx.x; i  < maxSetSize && i < recordEnd - recordStart; i += blockDim.x) {
        querySet[i] = tokens[recordStart + i];
    }

    __syncthreads();

    unsigned int recordLength = recordEnd - recordStart;

    unsigned int counter = 0;
    for (unsigned int offset = candidatesStart; offset < candidatesEnd; ++offset) {
        unsigned int candidate = candidates[offset];

        unsigned int candidateRecordStart = candidate > 0 ? tokenOffsets[candidate - 1] : 0;
        unsigned int candidateRecordEnd = tokenOffsets[candidate];

        unsigned int candidateLength = candidateRecordEnd - candidateRecordStart;
        unsigned int vt = ((recordLength + candidateLength + 1) / blockDim.x) + 1;


        for(int i = threadIdx.x; i  < maxSetSize && i < candidateLength; i += blockDim.x) {
            candidateSet[i] = tokens[candidateRecordStart + i];
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


// aggregate scenarios
template __global__ void scenarioA<true, unsigned int, unsigned int, unsigned int>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int, double);
template __global__ void scenarioB<true, unsigned int, unsigned int, unsigned int>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int, double);
template __global__ void scenarioC<true, unsigned int, unsigned int, unsigned int>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned int, double);

// general join scenarios
template __global__ void scenarioA<false, unsigned int, unsigned int, bool>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, bool*, unsigned int, double);
template __global__ void scenarioB<false, unsigned int, unsigned int, bool>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, bool*, unsigned int, double);
template __global__ void scenarioC<false, unsigned int, unsigned int, bool>(unsigned int*, unsigned int*, unsigned int*, unsigned int*, bool*, unsigned int, double);
