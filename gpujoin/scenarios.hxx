#include <cuda.h>
#include <cuda_runtime.h>
#include "functions.hxx"

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioA(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold);

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioB(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold);

template<bool aggregate, class Token, class Candidate, class Result>
__global__ void scenarioC(Token* tokens, unsigned int* tokenOffsets,
                           Candidate* candidates, unsigned int* candidateOffsets,
                           Result* results, unsigned int maxSetSize, double threshold);