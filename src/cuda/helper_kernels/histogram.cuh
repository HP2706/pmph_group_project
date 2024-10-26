#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include "../traits.h"  // Adjust the path based on your directory structure
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


// we dont expect to process more than 8 bits for the radix sort
#define MaxH (1 << 8) // 256

template<typename AnyUInt, uint32_t LGH>  // Make lgH a template parameter
__global__ void
RadixHistoKernel(
    const AnyUInt* inp_vals,        // Input values
    uint32_t* hist,                 // Removed volatile
    const uint32_t N,               // Total number of elements
    const uint32_t bit_pos,         // Starting bit position to examine
    const uint32_t Q                // Elements per thread
) {
    static_assert(is_zero_extendable_to_uint32<AnyUInt>::value,
        "UInt must be an unsigned type of 32 bits or less that can be zero-extended to uint32_t");

    // Now this static_assert works because LGH is known at compile time
    static_assert(LGH <= 8, "LGH must be less than or equal to 8 as otherwise shared memory will overflow");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // Use LGH instead of lgH
    const uint32_t H = 1 << LGH;
    const uint32_t hist_offset = bid * H;
    
    // Initialize local histogram in shared memory
    __shared__ uint32_t local_hist[MaxH]; 

    if (tid < H) {
        local_hist[tid] = 0;
    }
    __syncthreads();
    
    for (int q = 0; q < Q; q++) {
        int idx = hist_offset + bid * Q + q;
        if (idx < N) {
            AnyUInt val = inp_vals[idx];
            uint32_t bits = (static_cast<uint32_t>(val) >> bit_pos) & (H - 1);
            atomicAdd(&local_hist[bits], 1u);
        }
    }
    __syncthreads();

    if (tid < H) {
        hist[hist_offset + tid] = local_hist[tid];
    }
}


#endif
