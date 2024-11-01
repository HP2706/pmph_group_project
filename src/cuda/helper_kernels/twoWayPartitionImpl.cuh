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

        // Phase 1: Sequential reduction of Q elements
        // we can use uint16_t because the maximum number of elements is 1 << lgH = 2^lgH = 2^8 = 256
        uint16_t accum = 0;
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            // we shift by bit to get the bit value and we mask it with 1 to get the boolean value
            // we first shift by the bit_offs offset and then by the current bit from 0 to lgH-1
            uint16_t res = isBitUnset<uint>(bit_offs + bit, reg[q_idx]);
            accum += res;
        }
        
        // the thread local count is stored in the local histogram
        local_histo[tid] = accum;

        // we scan the local histogram
        uint16_t res = scanIncBlock<Add<uint16_t>>(local_histo, tid);
        __syncthreads();
        local_histo[tid] = res;
        __syncthreads();

        // we get the prefix of the thread
        // TODO: check if this is correct
        int16_t prefix_thread = local_histo[P::BLOCK_SIZE-1];

        // we reset the accumumulator
        // or we take the previous value
        if (tid == 0) {
            accum = 0; 
        } else {
            accum = local_histo[tid-1];
        }
        __syncthreads();

        // we rearrange the elements based on the bit value
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            uint val = reg[q_idx];
            uint16_t bit_val = (uint16_t) isBitUnset<uint>(bit_offs + bit, val);
            
            accum += bit_val;

            int newpos;
            if (bit_val == uint(1)) {
                newpos = accum -1 ;
            } else {
                // we add the prefix of the thread to the local histogram position
                // and we subtract the accumumulator to get the new position
                // we offset by threadIdx.x*Q 
                newpos = prefix_thread + thread_offset + q_idx - accum;
            }
            // we write the element to the new position
            shmem[newpos] = val;
        }
        // wait for all threads to have written their elements
        __syncthreads();

        if (bit < P::lgH-1) {
            // if we are not at the last bit
            // we copy the elements from the shared memory to the registers
            // in the new position
            for (int q_idx = 0; q_idx < P::Q; q_idx++) {
                reg[q_idx] = shmem[thread_offset + q_idx];
            }
        } else {
            // if we are at the last bit
            for (int q_idx = 0; q_idx < P::Q; q_idx++) {
                // we iterate with a stride of BLOCK_SIZE
                uint local_pos = q_idx*P::BLOCK_SIZE + tid;
                reg[q_idx] = shmem[local_pos];
            }
        }
        __syncthreads();


        #if 0
        // After rearrangement and syncthreads, add debug print
        if (tid == 0) {
            printf("debugging shmem after bit_offs: %d, bit: %d\n", bit_offs, bit);
            debugPartitionCorrectness<P>(
                shmem, 
                min(N, P::BLOCK_SIZE * P::Q),
                bit_offs + bit
            );
        }
        __syncthreads(); 

        debugPartitionCorrectness<P>(
            reg, 
            P::Q, 
            bit_offs + bit
        );
        __syncthreads();
        #endif
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