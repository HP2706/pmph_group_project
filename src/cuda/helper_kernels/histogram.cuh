#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include "../traits.h"  // Adjust the path based on your directory structure
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#define MaxH (1 << 8) 

template<typename AnyUInt, uint32_t LGH>
__global__ void
RadixHistoKernel(const AnyUInt* inp_vals,        // Input values
                 uint32_t* hist,                 // array of length num_bins instead of num_bins X BLOCK_SIZE as in HistoKernel1
                 const uint32_t N,               // Total number of elements
                 const uint32_t bit_pos,         // Starting bit position to examine
                 const uint32_t Q                // Elements per thread
) {
    static_assert(is_zero_extendable_to_uint32<AnyUInt>::value,
        "UInt must be an unsigned type of 32 bits or less that can be zero-extended to uint32_t");
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
            AnyUInt val = inp_vals[pos];
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

template<typename AnyUInt, uint32_t LGH>
__global__ void TransposeHistoKernel(uint32_t* inputHist, uint32_t* outputMatrix, uint32_t numChunks, uint32_t numBlocks)
{
    // Calculate row and column indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure within bounds
    if (row < numChunks && col < numBlocks) 
    {
        // Transpose: move element from inputHist to outputMatrix in transposed position
        outputMatrix[col * numChunks + row] = inputHist[row * numBlocks + col];
    }
}

#endif
