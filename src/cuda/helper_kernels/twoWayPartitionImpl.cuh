#pragma once

#include <algorithm>
#include <iostream>
#include "../constants.cuh"
#include "../helper.h"


template<class P>
__device__ void TwoWayPartition(
    typename P::ElementType reg[P::Q], // Q elements per thread in registers
    typename P::ElementType* shmem, // shared memory
    uint16_t* local_histo, // local histogram in shared memory
    uint32_t bit_offs, // position of the bit in the element
    uint32_t N // number of elements
){
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint16_t thread_offset = tid * P::Q;

    for (int bit = 0; bit < P::lgH; bit++) {

        // Sequential reduce of Q elements in registers
        uint16_t accum = 0;
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            // we shift by bit to get the bit value and we mask it with 1 to get the boolean value
            // we first shift by the bit_offs offset and then by the current bit from 0 to lgH-1
            uint16_t res = isBitUnset<uint>(bit_offs + bit, reg[q_idx]);
            accum += res;
        }
        
        // the thread local count is stored to shared memory
        local_histo[tid] = accum;
        __syncthreads();
        // we compute the inclusive scan across shared memory
        // to get the cumulative sum of unset bits up to our threadIdx
        uint16_t res = scanIncBlock<Add<uint16_t>>(local_histo, tid);
        local_histo[tid] = res;
        __syncthreads();

        // we get the prefix sum for the entire block
        // which is the cumulative sum of unset bits up to the last thread in the block
        int16_t prefix_sum = local_histo[P::BLOCK_SIZE-1];
        // each thread loads to its registers from shared memory
        // the prefix sum of the previous thread 
        // IE would be the same as readin local_histogram as an exclusive scan
        if (tid == 0) {
            accum = 0; 
        } else {
            accum = local_histo[tid-1];
        }
        __syncthreads();


        // each thread sequentially rearranges its Q elements
        // and write them to their new position in shared memory
        
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            uint val = reg[q_idx];
            uint16_t bit_val = (uint16_t) isBitUnset<uint>(bit_offs + bit, val);
            
            accum += bit_val;

            int newpos;

            if (bit_val == uint(1)) {
                // if bit is unset(it is 0) we 
                // keep the same position
                newpos = accum -1 ;
            } else {
                // if bit is set(it is 1) we compute the new position
                // as: 
                // prefix_sum : the cumulative sum of unset elements up to the previous thread
                // thread_offset + q_idx : the original position of the element in the block
                // we subtract accum the cumulative sum of unset elements for 
                // that thread within its Q elements
                newpos = prefix_sum + thread_offset + q_idx - accum;
            }
            // we write the element to the new position
            shmem[newpos] = val;
        }
        // wait for all threads to have written their elements
        __syncthreads();

        // each thread loads its new elements from shared memory to its registers
        if (bit < P::lgH-1) {
            // if we are not at the last bit
            // in the new position
            for (int q_idx = 0; q_idx < P::Q; q_idx++) {
                reg[q_idx] = shmem[thread_offset + q_idx];
            }
        } else {
            // if we are at the last bit
            for (int q_idx = 0; q_idx < P::Q; q_idx++) {
                // we iterate with a stride of BLOCK_SIZE
                // but this is not a big deal as uncoalesced 
                // access in shared memory does not affect performance
                uint local_pos = q_idx*P::BLOCK_SIZE + tid;
                reg[q_idx] = shmem[local_pos];
            }
        }
        __syncthreads();
    }
}


template<class T>
inline void TwoWayPartitionCpu(
    uint32_t bit_offs,
    uint32_t N,
    T* arr_inp,
    T* arr_out,
    bool debug = false
) {
    // Copy input to output first
    std::copy(arr_inp, arr_inp + N, arr_out);
    
    if (debug) {
        // Print original array
        std::cout << "Original array (showing bit " << bit_offs << "):\n";
        for(uint32_t i = 0; i < N; i++) {
        bool bit = isBitUnset(bit_offs, arr_inp[i]);
            std::cout << arr_inp[i] << "(" << (bit ? "0" : "1") << ") ";
        }
        std::cout << "\n\n";
    }
    
    printf("calling std::partition\n");
    // Use std::partition to do the partitioning
    std::partition(arr_out, arr_out + N, 
        [bit_offs](const T& val) {
            return isBitUnset(bit_offs, val);
        }
    );

    printf("std::partition done\n");

    if (debug) {
        // Print partitioned array
        std::cout << "Partitioned array:\n";
        for(uint32_t i = 0; i < N; i++) {
        bool bit = isBitUnset(bit_offs, arr_out[i]);
        std::cout << arr_out[i] << "(" << (bit ? "0" : "1") << ") ";
        }
        std::cout << "\n";
    }
    
    // Verify partition
    uint32_t partition_point = 0;
    while (partition_point < N && isBitUnset(bit_offs, arr_out[partition_point])) {
        partition_point++;
    }
    if (debug) {
        std::cout << "\nPartition point: " << partition_point << "\n";
        std::cout << "First part (0s): " << partition_point << " elements\n";
        std::cout << "Second part (1s): " << N - partition_point << " elements\n";
    }

    // Add debugging for each bit position
    for (uint32_t bit = 0; bit < 8; bit++) {  // Assuming 8 bits like in GPU version
        if (debug) {
            std::cout << "Examining bit position " << (bit_offs + bit) << ":\n";
            for(uint32_t i = 0; i < N; i++) {
                bool is_unset = isBitUnset(bit_offs + bit, arr_out[i]);
                std::cout << arr_out[i] << "(" << (is_unset ? "0" : "1") << ") ";
            }
            std::cout << "\n\n";
        }
    }
}