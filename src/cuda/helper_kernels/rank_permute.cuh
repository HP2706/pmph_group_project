#ifndef RANK_PERMUTE_CUH
#define RANK_PERMUTE_CUH

#include "utils.cuh"
#include "../constants.cuh"
#include "../helper.h"
/// we use UintType as we are manipulating 
/// the histograms not the input array elements
template<class P>
__device__ void GlbToReg(
    uint64_t N, // n elements
    typename P::UintType* shmem,  // shared memory
    typename P::UintType* arr,  // global memory
    typename P::UintType* reg  // registers
) {

    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    const uint32_t tid = threadIdx.x;

    // 1. Read from global to shared memory
    const uint64_t glb_offs = blockIdx.x * P::QB;

    copyFromGlb2ShrMem<typename P::UintType, P::QB>(
        glb_offs, 
        N, 
        P::H,  // the identity value, but in our case to be sure of correct ordering we set it to the maximum value
        arr, 
        shmem  // Updated variable name
    );
    
    // 2. Read from shared memory to registers
    for (int i = 0; i < P::Q; i++) {
        reg[i] = shmem[P::Q*threadIdx.x + i];
    }
    __syncthreads();
}



// this is not a kernel but a compound function that calls many kernels
template<
    class P
>
__global__ void RankPermuteKer(
    typename P::UintType* d_hist,
    uint32_t bitpos, 
    uint32_t N,
    typename P::ElementType* arr_inp,
    typename P::ElementType* arr_out
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");


    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    // we allocate shared memory for the elements and registers
    __shared__ typename P::UintType shmem[P::QB];
    typename P::UintType reg[P::Q];
    uint16_t reg_flags[P::Q];

    // step 1: read from global to shared memory and registers
    GlbToReg<P>(N, d_hist, shmem, reg);

    /* Step 2: Two-way partitioning loop
     * For each bit position (lgH bits total):
     * 1. Each thread sequentially reduces its Q elements in registers
     * 2. Perform parallel block-level scan across all threads
     * 3. Each thread applies prefix to rearrange its elements
     */
    #pragma unroll
    for (int bit = 0; bit < P::lgH; bit++) {
        // Phase 1: Sequential reduction of Q elements
        // we can use uint16_t because the maximum number of elements is 1 << lgH = 2^lgH = 2^8 = 256
        uint16_t local_count = 0;
        #pragma unroll
        for (int i = 0; i < P::Q; i++) {
            // we shift by bit to get the bit value and we mask it with 1 to get the boolean value
            // we first shift by the bitpos offset and then by the current bit from 0 to lgH-1
            reg_flags[i] = ((reg[i] >> (bitpos + bit)) & 1);
            local_count += reg_flags[i];
        }
        
        // Store local count to shared memory
        if (tid < P::Q) {
            shmem[tid] = local_count;
        }
        __syncthreads();

        // we scan the shared memory array
        uint32_t res = scanIncBlock<Add<uint32_t>>(shmem, tid);
        __syncthreads();

        // Get prefix for current thread
        uint32_t prefix = (tid > 0) ? shmem[tid-1] : 0;
        
        // Rearrange elements based on their bits
        #pragma unroll
        for (int i = 0; i < P::Q; i++) {
            if (reg_flags[i]) {
                // we add the prefix to the element
                reg[i] += prefix;
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
    uint32_t global_histo_offs = blockIdx.x * P::H;

    // we write the histogram from global to shared memory
    for (int i = threadIdx.x; i < P::H; i += blockDim.x) {
        shmem[i] = d_hist[global_histo_offs + i];
    }
    __syncthreads();

    // step 3.2: we perform an exclusive scan on the histogram
    // this simulates an exclusive scan
    // with an inclusive scan and then subtracting the result from the next element
    int res = scanIncBlock<Add<uint32_t>>(shmem, tid);
    if (tid < P::H-1) {
        shmem[tid+1] -= res;
    }
    __syncthreads();

    // step 3.3: we write the elements to their final positions 
    // in global memory q elements per thread
    for (int q = 0; q < P::Q; q++) {
        typename P::ElementType elm = reg[q];
        // we extract the position via the bitpos offset
        uint32_t pos = (elm >> bitpos) & ((1 << P::lgH) - 1);
        uint32_t offs = shmem[pos];
        uint32_t pos_out = offs+(q*blockDim.x + tid);

        if (pos_out < N) {
            arr_out[pos_out] = elm;
        }
    }
}

#endif // RANK_PERMUTE_CUH
