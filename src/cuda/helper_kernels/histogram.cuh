#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "utils.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


/// the multistep kernel for the histogram
template<typename UInt>
__global__ void
multiStepGenericKernel (
                const UInt* inp_vals     // Input values
                , volatile uint32_t* hist      // Histogram counts
                , const uint32_t N         // Number of input elements
                , const uint32_t LB        // Lower bound
                , const uint32_t UB        // Upper bound
) {
    // Ensure UInt is less than or equal to 64 bytes
    // this will not work for larger uints
    static_assert(sizeof(UInt) <= 64, "UInt must be 64 bytes or less");

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < N) {
        // this differs from the original kernel in the way that 
        // we can derive the index directly from the value
        UInt ind = inp_vals[gid];

        if(ind < UB && ind >= LB) {
            // we increment by one
            // where in the original histogram implementation 
            // we were summing by the values
            atomicAdd((uint32_t*)&hist[ind], uint32_t(1));
        }
    }
}

// Corresponding multiStepHisto function
// we just make it generic over any unsigned integer type
template<typename UInt, int B>
void multiStepGenericHisto (
                    const UInt* d_inp_vals
                    , uint32_t* d_hist
                    , const uint32_t N
                    , const uint32_t H // number of bins
                    , const uint32_t LLC
) {
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = (LLC_FRAC * LLC) / sizeof(UInt);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    cudaMemset(d_hist, 0, H * sizeof(UInt));
    for (uint32_t k = 0; k < num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k * CHUNK;
        uint32_t upp_bound = min((k + 1) * CHUNK, H);

        uint32_t grid = (N + B - 1) / B;
        multiStepGenericKernel<UInt><<<grid,B>>>(d_inp_vals, d_hist, N, low_bound, upp_bound);
    }
}

#endif
