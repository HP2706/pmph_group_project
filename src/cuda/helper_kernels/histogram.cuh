#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


// we dont expect to process more than 8 bits for the radix sort
#define MaxH (1 << 8) 


template<typename ElTp, uint32_t LGH, uint32_t Q>  
__global__ void
HistoKernel1(
    const ElTp* inp_vals,        // Input values
    uint32_t* hist,                 // array of length num_bins * gridDim.x
    const uint32_t N,               // Total number of elements
    const uint32_t bit_pos         // Starting bit position to examine
) {

    static_assert(LGH <= 8, "LGH must be less than or equal to 8 as otherwise shared memory will overflow");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t H = 1 << LGH;
    
    // Offset into global histogram for this block
    uint32_t* block_hist = hist + (bid * H);

    __shared__ uint32_t local_hist[MaxH]; 
    if (tid < H) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Calculate the starting position for this thread
    uint32_t start_pos = bid * blockDim.x * Q + tid;
    const uint32_t stride = blockDim.x;

    // Process Q elements per thread with proper striding
    for (uint32_t q = 0; q < Q; q++) {
        uint32_t pos = start_pos + q * stride;
        if (pos < N) {
            ElTp val = inp_vals[pos];
            uint32_t bits = (static_cast<uint32_t>(val) >> bit_pos) & (H - 1);
            atomicAdd(&local_hist[bits], 1);  // Need atomicAdd here for shared memory
        }
    }
    __syncthreads();

    // Write to global memory
    if (tid < H) {
        block_hist[tid] = local_hist[tid];
    }
}

template<typename ElTp, uint32_t LGH>  
__global__ void
HistoKernel2(
    const ElTp* inp_vals,        // Input values
    uint32_t* hist,                 // array of length num_bins instead of num_bins X BLOCK_SIZE as in HistoKernel1
    const uint32_t N,               // Total number of elements
    const uint32_t bit_pos,         // Starting bit position to examine
    const uint32_t Q                // Elements per thread
) {
    static_assert(LGH <= 8, "LGH must be less than or equal to 8 as otherwise shared memory will overflow");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t idx = bid * blockDim.x + tid;
    const uint32_t H = 1 << LGH;

    __shared__ uint32_t local_hist[MaxH]; 
    if (tid < H) {
        local_hist[tid] = 0;
    }
    __syncthreads();

    // Process Q elements per thread in a coalesced manner
    #pragma unroll
    for (uint32_t q = 0; q < Q; q++) {
        uint32_t pos = (blockIdx.x * blockDim.x * Q) + (tid * Q) + q;
        if (pos < N) {
            ElTp val = inp_vals[pos];
            uint32_t bits = (static_cast<uint32_t>(val) >> bit_pos) & (H - 1);
            atomicAdd(&local_hist[bits], 1u);
        }
    }
    __syncthreads();

    // Accumulate results using atomic operations to global memory
    if (tid < H) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}



#endif



