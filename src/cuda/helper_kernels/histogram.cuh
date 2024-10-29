#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "../helper.h"

// we dont expect to process more than 8 bits for the radix sort

template<class P>  
__global__ void
Histo(
    typename P::ElementType* inp_vals,        // Input values
    typename P::UintType* hist,                 // array of length num_bins * gridDim.x
    const uint32_t N,                         // Total number of elements
    const uint32_t bit_pos                    // Starting bit position to examine
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // Offset into global histogram for this block
    uint32_t* block_hist = hist + (bid * P::H);

    __shared__ typename P::UintType local_hist[P::H]; 
    if (tid < P::H) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Calculate the starting position for this thread
    uint32_t start_pos = bid * blockDim.x * P::Q + tid;
    const uint32_t stride = blockDim.x;

    // Process Q elements per thread with proper striding
    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t pos = start_pos + q * stride;
        if (pos < N) {
            typename P::ElementType val = inp_vals[pos];
            uint32_t bits = (static_cast<uint32_t>(val) >> bit_pos) & (P::H - 1);
            atomicAdd(&local_hist[bits], 1);  // Need atomicAdd here for shared memory
        }
    }
    __syncthreads();

    // Write to global memory
    if (tid < P::H) {
        block_hist[tid] = local_hist[tid];
    }
}


#endif



