#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include "../constants.cuh"
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
    
    // allocate shared memory for local histogram
    __shared__ typename P::UintType local_hist[P::H]; 
    if (tid < P::H) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // the global offset is the blockidx times the number of elements per block 
    // Q(elms per thread)*B(threads per block)
    const uint32_t global_offset = bid * P::QB;  

    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t local_position = q * P::BLOCK_SIZE + tid;
        uint64_t global_position = global_offset + local_position;
        if (global_position < N) {

            typename P::ElementType val = inp_vals[global_position];
            uint32_t bits = (static_cast<uint32_t>(val) >> bit_pos) & (P::H - 1);
            atomicAdd(&local_hist[bits], 1);  // Need atomicAdd here for shared memory
        }
    }
    __syncthreads();

    // Write to global memory
    if (tid < P::H) {
        uint32_t global_offset = bid * P::H;
        hist[global_offset + tid] = local_hist[tid];
    }
}


#endif



