#ifndef GPU_JOIN_FUNCTIONS_HXX
#define GPU_JOIN_FUNCTIONS_HXX

__forceinline__ __device__  unsigned int intersection(
        unsigned int* foreignTokens, unsigned int* inputTokens,
        unsigned int probeStart, unsigned int probeEnd,
        unsigned int candidateStart, unsigned int candidateEnd)
{
    unsigned int count = 0;
    unsigned int i = probeStart, j = candidateStart;
    while (i < probeEnd && j < candidateEnd)
    {
        if (foreignTokens[i] < inputTokens[j])
            i++;
        else if (inputTokens[j] < foreignTokens[i])
            j++;
        else // intersect
        {
            count++;
            i++;
        }
    }
    return count;
}

__forceinline__ __device__  int myMax(int a, int b)
{
    return (a < b) ? b : a;
}

__forceinline__ __device__  int myMin(int a, int b)
{
    return (a > b) ? b : a;
}


__forceinline__ __device__  int merge_path(unsigned int* a, unsigned int aSize,
                                unsigned int* b, unsigned int bSize, unsigned int diag)
{
    int begin = myMax(0, diag -  bSize);
    int end   = myMin(diag, aSize);

    while (begin < end) {
        int mid = (begin + end) / 2;

        if (a[mid] < b[diag - 1 - mid]) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

__forceinline__ __device__ void reduce(unsigned int tid, unsigned blockSize, unsigned int* counts, unsigned int* counter)
{
    __syncthreads();

    if ((blockSize >= 1024) && (tid < 512))
    {
        counts[tid] = *counter = *counter + counts[tid + 512];
    }

    __syncthreads();

    if ((blockSize >= 512) && (tid < 256))
    {
        counts[tid] = *counter = *counter + counts[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128))
    {
        counts[tid] = *counter = *counter + counts[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64))
    {
        counts[tid] = *counter = *counter + counts[tid + 64];
    }

    __syncthreads();

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) *counter += counts[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = 16; offset > 0; offset /= 2)
        {
            *counter += __shfl_down_sync(0xFFFFFFFF, *counter, offset);
        }
    }
}
__device__ inline unsigned int serial_intersect(unsigned int* a, unsigned int aBegin, unsigned int aEnd,
                                       unsigned int* b, unsigned int bBegin, unsigned int bEnd, unsigned int vt)
{
    unsigned int count = 0;

    // vt parameter must be odd integer
    for (int i = 0; i < vt; i++)
    {
        bool p = false;
        if ( aBegin >= aEnd ) p = false; // a, out of bounds
        else if ( bBegin >= bEnd ) p = true; //b, out of bounds
        else {
            if (a[aBegin] < b[bBegin]) p = true;
            if (a[aBegin] == b[bBegin]) count++;
        }
        if(p) aBegin++;
        else bBegin++;

    }

    __syncthreads();
    return count;
}


#endif //GPU_JOIN_FUNCTIONS_HXX