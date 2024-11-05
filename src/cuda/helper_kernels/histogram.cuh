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
    
    // allocate shared memory for local histogram
    __shared__ uint32_t shmem_hist[P::H]; 

    // initialize the local histogram to 0
    if (tid < P::H) {
        shmem_hist[tid] = 0;
    }
    __syncthreads();

    // the global offset is the blockidx times the number of elements per block 
    // Q(elms per thread)*B(threads per block)
    const uint32_t global_offset = bid * P::QB;  

    // each thread processes Q elements and writes to shared memory
    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t local_position = q * P::BLOCK_SIZE + tid;
        uint64_t global_position = global_offset + local_position;
        if (global_position < N) {

            typename P::ElementType val = inp_vals[global_position];
            uint32_t pos = getBits<typename P::ElementType, P::lgH>(bit_pos, val);
            atomicAdd(&shmem_hist[pos], 1);  // increment the histogram bucket by 1
        }
    }
    __syncthreads();


    // write the local histogram to global memory
    uint32_t global_mem_offset = bid * P::H;
    // we need a strided version for when P::BLOCK_SIZE < P::H
    for (uint32_t i = tid; i < P::H; i += P::BLOCK_SIZE) {
        hist[global_mem_offset + i] = shmem_hist[i];
    }
    
}
template<class P, class T>  
__global__ void
HistoWarp(
    typename P::ElementType* inp_vals,
    T* hist,
    const uint32_t N,
    const uint32_t bit_pos
) {
    static_assert(is_params<P>::value, "P must be an instance of Params");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;
    
    // shared memory for warp-level histograms
    // each warp gets its own histogram
    __shared__ uint32_t warp_hists[P::WARPS_PER_BLOCK][P::H];

    // initialize warp histograms to 0
    if (lane_id < P::H) {
        warp_hists[warp_id][lane_id] = 0;
    }
    __syncwarp();

    // global offset calculation
    const uint32_t global_offset = bid * P::QB;

    // each thread processes Q elements but updates warp-private histogram
    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t local_position = q * P::BLOCK_SIZE + tid;
        uint64_t global_position = global_offset + local_position;
        
        if (global_position < N) {
            typename P::ElementType val = inp_vals[global_position];
            uint32_t pos = getBits<typename P::ElementType, P::lgH>(bit_pos, val);
            // atomic add to warp-private histogram
            atomicAdd(&warp_hists[warp_id][pos], 1);
        }
    }
    __syncthreads();

    // combine warp histograms into final block histogram
    if (warp_id == 0 && lane_id < P::H) {
        uint32_t sum = 0;
        for (uint32_t w = 0; w < P::WARPS_PER_BLOCK; w++) {
            sum += warp_hists[w][lane_id];
        }
        // write to global memory
        hist[bid * P::H + lane_id] = sum;
    }
}

template<class P, class T>  
__global__ void
HistoWarpReg(
    typename P::ElementType* inp_vals,
    T* hist,
    const uint32_t N,
    const uint32_t bit_pos
) {
    static_assert(is_params<P>::value, "P must be an instance of Params");

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;
    
    // Each thread maintains a portion of the histogram in registers
    // Each thread in the warp handles H/WARP_SIZE bins
    uint32_t reg_hist[P::H / WARP_SIZE + (P::H % WARP_SIZE ? 1 : 0)] = {0};
    
    // global offset calculation
    const uint32_t global_offset = bid * P::QB;

    // Process elements
    for (uint32_t q = 0; q < P::Q; q++) {
        uint32_t local_position = q * P::BLOCK_SIZE + tid;
        uint64_t global_position = global_offset + local_position;
        
        if (global_position < N) {
            typename P::ElementType val = inp_vals[global_position];
            uint32_t pos = getBits<typename P::ElementType, P::lgH>(bit_pos, val);
            
            // Use warp voting to increment the correct thread's register
            uint32_t target_lane = pos / (P::H / WARP_SIZE);
            uint32_t local_pos = pos % (P::H / WARP_SIZE);
            
            // Broadcast to all threads which bin needs incrementing
            uint32_t vote = __ballot_sync(0xffffffff, target_lane == lane_id);
            if (target_lane == lane_id) {
                reg_hist[local_pos] += __popc(vote);
            }
        }
    }
    
    __syncthreads();

    // Combine register histograms across warps using shared memory
    __shared__ uint32_t block_hist[P::H];
    
    // Initialize shared memory
    if (tid < P::H) {
        block_hist[tid] = 0;
    }
    __syncthreads();
    
    // Each thread writes its register counts to shared memory
    for (uint32_t i = 0; i < (P::H / WARP_SIZE + (P::H % WARP_SIZE ? 1 : 0)); i++) {
        uint32_t global_bin = lane_id * (P::H / WARP_SIZE) + i;
        if (global_bin < P::H) {
            atomicAdd(&block_hist[global_bin], reg_hist[i]);
        }
    }
    
    __syncthreads();

    // Write final results to global memory
    if (tid < P::H) {
        hist[bid * P::H + tid] = block_hist[tid];
    }
}

#endif



