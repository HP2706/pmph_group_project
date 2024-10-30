#ifndef RANK_PERMUTE_CUH
#define RANK_PERMUTE_CUH

#include "utils.cuh"
#include "../constants.cuh"
#include "../helper.h"
#include <limits>
/// we use UintType as we are manipulating 
/// the histograms not the input array elements
template<class T, int Q, int BLOCK_SIZE, int MAXNUMERIC>
__device__ void GlbToReg(
    uint64_t N, // n elements
    T* shmem,  // shared memory
    T* arr,  // global memory
    T reg[Q]  // registers with Q elements
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t QB = BLOCK_SIZE * Q;

    // 1. Read from global to shared memory
    const uint64_t glb_offs = blockIdx.x * QB;
    
    
    // this causes an "illegal memory access" error
    copyFromGlb2ShrMem<T, Q>(
        glb_offs,
        N,
        MAXNUMERIC,  // the identity value, but in our case to be sure of correct ordering we set it to the maximum value
        arr,
        shmem  // Updated variable name
    );

    __syncthreads(); 

    // 2. Read from shared memory to registers
    for (int i = 0; i < Q; i++) {
        reg[i] = shmem[Q*threadIdx.x + i];
    }
    __syncthreads();
}


template<class T>
__device__ T isBitSet(T val, int bitpos, int bit, T compar = T(1)) {
    return (val >> (bitpos + bit)) & compar;
}


// this is not a kernel but a compound function that calls many kernels
template<class P>
__global__ void RankPermuteKer(
    typename P::UintType* d_hist,
    typename P::UintType* d_hist_transposed_scanned_transposed, // as we are using scan we have higher integer values and thus need to use uint64_t
    uint32_t bitpos, 
    uint32_t N,
    typename P::ElementType* arr_inp, // this is either uint8_t, uint16_t or uint32_t or uint64_t
    typename P::ElementType* arr_out
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    using uint = typename P::ElementType;


    extern __shared__ uint64_t sh_mem_uint64[];
    
    uint* shmem = (uint*) sh_mem_uint64;

    uint64_t* global_histo = (uint64_t*) sh_mem_uint64;
    uint16_t* local_histo = (uint16_t*) (global_histo + P::H);

    // we allocate the registers
    // Q elements per thread
    uint reg[P::Q];

    // illegal memory access here
    GlbToReg<
        uint, 
        P::Q,  
        P::BLOCK_SIZE,
        P::MAXNUMERIC_ElementType
    >(N, shmem, arr_inp, reg);

    /* Step 2: Two-way partitioning loop
     * For each bit position (lgH bits total):
     * 1. Each thread sequentially reduces its Q elements in registers
     * 2. Perform parallel block-level scan across all threads
     * 3. Each thread applies prefix to rearrange its elements
    */

    uint16_t thread_offset = tid * P::Q;

    #pragma unroll
    for (int bit = 0; bit < P::lgH; bit++) {
        // Phase 1: Sequential reduction of Q elements
        // we can use uint16_t because the maximum number of elements is 1 << lgH = 2^lgH = 2^8 = 256
        uint16_t accum = 0;
        #pragma unroll
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            // we shift by bit to get the bit value and we mask it with 1 to get the boolean value
            // we first shift by the bitpos offset and then by the current bit from 0 to lgH-1
            uint16_t res = isBitSet<uint>(reg[q_idx], bitpos, bit);
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

        // wait for all threads to have accum
        __syncthreads();
        // we rearrange the elements based on the bit value
        for (int q_idx = 0; q_idx < P::Q; q_idx++) {
            uint val = reg[q_idx];
            uint bit_val = isBitSet<uint>(val, bitpos, bit);
            
            accum += bit_val;
            uint newpos;
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
    }

    /* Step 3: Final output generation
     * 1. Copy original and scanned histograms from global to shared memory
     * 2. Scan in place the original histogram for this block
     * 3. Each thread writes its Q elements to their final positions in global memory
     *    using the histogram information
    */
    
    // step 3.1: we copy the histogram from global to shared memory

    // the offset for each block in the global histogram
    uint32_t global_histo_offs = bid * P::H;

    // we write the histogram from global to shared memory
    for (int i = threadIdx.x; i < P::H; i += blockDim.x) {
        uint16_t local_val = d_hist[global_histo_offs + i];
        uint64_t global_val = d_hist_transposed_scanned_transposed[global_histo_offs + i];
        // we compute the difference between the global and local values
        global_histo[i] = global_val - local_val;
        local_histo[i] = local_val;
    }
    __syncthreads();

    // step 3.2: we perform an exclusive scan on the histogram
    // this can be simulated via an inclusive scan and a shift by one element
    int res = scanIncBlock<Add<uint16_t>>(local_histo, tid);
    if (tid < P::H-1) {
        global_histo[tid+1] -= res;
    }
    __syncthreads();

    // step 3.3: we write the elements to their final positions 
    // in global memory q elements per thread
    for (int q_idx = 0; q_idx < P::Q; q_idx++) {
        uint elm = reg[q_idx];
        // we extract the position via the bitpos offset
        uint32_t pos = isBitSet<uint>(elm, bitpos, 0, ((1 << P::lgH) - 1));
        uint32_t offs = global_histo[pos];
        uint32_t pos_out = offs+(q_idx*blockDim.x + tid);

        if (pos_out < N) {
            arr_out[pos_out] = elm;
        }
    }
}

#endif // RANK_PERMUTE_CUH
