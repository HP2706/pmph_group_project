#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "utils.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "../traits.h"


/// the multistep kernel for the histogram
template<typename AnyUInt>
__global__ void
multiStepGenericKernel (
                const AnyUInt* inp_vals     // Input values
                , volatile uint32_t* hist      // Histogram counts
                , const uint32_t N         // Number of input elements
                , const uint32_t LB        // Lower bound
                , const uint32_t UB        // Upper bound
) {
    // Ensure UInt is less than or equal to 64 bytes
    // this will not work for larger uints
    static_assert(is_zero_extendable_to_uint32<AnyUInt>::value, "AnyUInt must be zero-extendable to uint32_t");
    static_assert(sizeof(AnyUInt) <= 64, "AnyUInt must be 64 bytes or less");

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < N) {
        // this differs from the original kernel in the way that 
        // we can derive the index directly from the value
        
        // we cast to uint32_t (DONT KNOW WHAT THE PERF IMPLICATIONS ARE)
        uint32_t ind = static_cast<uint32_t>(inp_vals[gid]);

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
template<typename AnyUInt, int B>
void multiStepGenericHisto (
                    const AnyUInt* d_inp_vals
                    , uint32_t* d_hist
                    , const uint32_t N
                    , const uint32_t H // number of bins
                    , const uint32_t LLC
) {
    static_assert(is_zero_extendable_to_uint32<AnyUInt>::value, "AnyUInt must be zero-extendable to uint32_t");
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = (LLC_FRAC * LLC) / sizeof(AnyUInt);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    cudaMemset(d_hist, 0, H * sizeof(AnyUInt));
    for (uint32_t k = 0; k < num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k * CHUNK;
        uint32_t upp_bound = min((k + 1) * CHUNK, H);

        uint32_t grid = (N + B - 1) / B;
        multiStepGenericKernel<AnyUInt><<<grid,B>>>(d_inp_vals, d_hist, N, low_bound, upp_bound);
    }
}

#endif
