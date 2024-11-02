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

template<class P, class T>  
__global__ void
Histo(
    typename P::ElementType* inp_vals,        // Input values
    T* hist,                 // array of length num_bins * GRID_SIZE
    const uint32_t N,                         // Total number of elements
    const uint32_t bit_pos                    // Starting bit position to examine
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    
    // allocate shared memory for histogram
    __shared__ uint32_t shmem_hist[P::H]; 

    // initialize the histogram to 0
    for (uint32_t i = tid; i < P::H; i += P::BLOCK_SIZE) {
        shmem_hist[i] = 0;
    }
    __syncthreads();

    // the global offset is the blockidx times the number of elements per block 
    // Q(elms per thread)*B(threads per block)
    const uint32_t global_offset = bid * (P::Q * P::BLOCK_SIZE);  

    // each thread processes Q elements and writes to shared memory
    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t local_position = q * P::BLOCK_SIZE + tid;
        uint64_t global_position = global_offset + local_position;
        if (global_position < N) {

            typename P::ElementType val = inp_vals[global_position];
            uint32_t pos = getDigit<typename P::ElementType, P::lgH>(bit_pos, val);
            atomicAdd(&shmem_hist[pos], 1);  // increment the histogram bucket by 1
        }
    }
    __syncthreads();


    // write the local histogram to global memory
    uint32_t global_mem_offset = bid * P::H;

    for (uint32_t i = tid; i < P::H; i += P::BLOCK_SIZE) {
        hist[global_mem_offset + i] = shmem_hist[i];
    }
    
}


#endif



