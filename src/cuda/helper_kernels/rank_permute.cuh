#ifndef RANK_PERMUTE_CUH
#define RANK_PERMUTE_CUH

#include "../constants.cuh"
#include "../helper.h"
#include "utils.cuh"
#include <limits>
#include "twoWayPartitionImpl.cuh"


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



template<class P>
__device__ void WriteOutput(
    typename P::ElementType reg[P::Q],
    uint64_t* global_histo,
    uint16_t* local_histo,
    uint16_t* d_hist,
    uint64_t* d_hist_transposed_scanned_transposed,
    uint32_t bitpos,
    uint32_t N,
    typename P::ElementType* arr_out
) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    // step 3.1: copy histogram from global to shared memory
    uint32_t global_histo_offs = bid * P::H;
    for (int i = threadIdx.x; i < P::H; i += P::BLOCK_SIZE) {
        uint16_t local_val = d_hist[global_histo_offs + i];
        uint64_t global_val = d_hist_transposed_scanned_transposed[global_histo_offs + i];
        global_histo[i] = global_val - local_val;
        local_histo[i] = local_val;
    }
    __syncthreads();

    // step 3.2: perform exclusive scan on the histogram
    int res = scanIncBlock<Add<uint16_t>>(local_histo, tid);
    if (tid < P::H-1) {
        global_histo[tid+1] -= res;
    }
    __syncthreads();

    // step 3.3: write elements to their final positions
    for (int q_idx = 0; q_idx < P::Q; q_idx++) {
        typename P::ElementType elm = reg[q_idx];
        int bin = getBits<uint, P::lgH>(bitpos, elm);
        uint32_t global_offs = global_histo[bin];
        uint32_t global_pos = global_offs + (q_idx*P::BLOCK_SIZE + tid);

        if (global_pos < N) {
            arr_out[global_pos] = elm;
        }
    }
}




template<class P>
__launch_bounds__(P::BLOCK_SIZE, 1024/P::BLOCK_SIZE)
__global__ void RankPermuteKer(
    uint16_t* d_hist,
    uint64_t* d_hist_transposed_scanned_transposed, // as we are using scan we have higher integer values and thus need to use uint64_t
    uint32_t bit_offs, 
    uint32_t N,
    typename P::ElementType* arr_inp, // this is either uint8_t, uint16_t or uint32_t or uint64_t
    typename P::ElementType* arr_out
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    using uint = typename P::ElementType;

    // Calculate shared memory layout:
    // 1. Element buffer: BLOCK_SIZE * Q * sizeof(ElementType)
    // 2. Global histogram: H * sizeof(uint64_t)
    // 3. Local histogram: BLOCK_SIZE * sizeof(uint16_t)
    
    extern __shared__ uint64_t sh_mem_uint64[];
    
    // Ensure proper alignment for all types
    // the interval [0, QB] is reserved for the element buffer
    uint* shmem = (uint*) sh_mem_uint64;
    // the interval [QB, QB+H] is reserved for the global histogram
    uint64_t* global_histo = (uint64_t*)(shmem + P::H);
    // the interval [QB+H, QB+H+BLOCK_SIZE] is reserved for the local histogram
    uint16_t* local_histo = (uint16_t*)(global_histo + P::BLOCK_SIZE);
    
    // we allocate the registers
    // Q elements per thread
    uint reg[P::Q];

    GlbToReg<
        uint, 
        P::Q,  
        P::BLOCK_SIZE,
        P::MAXNUMERIC_ElementType
    >(N, shmem, arr_inp, reg);

    __syncthreads();

    /* Step 2: Two-way partitioning loop
     * For each bit position (lgH bits total):
     * 1. Each thread sequentially reduces its Q elements in registers
     * 2. Perform parallel block-level scan across all threads
     * 3. Each thread applies prefix to rearrange its elements
    */

    TwoWayPartition<P>(
        reg, 
        shmem, 
        local_histo,
        bit_offs,
        N
    );
    __syncthreads();

    if (tid == 0) {
        printf("debugging after two way partition\n bit_offs: %d\n", bit_offs + P::lgH - 1);
        debugPartitionCorrectness<P>(
            shmem, 
            min(N, P::BLOCK_SIZE * P::Q),
            bit_offs + P::lgH - 1
        );
    }
    __syncthreads();

    /* Step 3: Final output generation
     * 1. Copy original and scanned histograms from global to shared memory
     * 2. Scan in place the original histogram for this block
     * 3. Each thread writes its Q elements to their final positions in global memory
     *    using the histogram information
    */

    WriteOutput<P>(
        reg,
        global_histo,
        local_histo,
        d_hist,
        d_hist_transposed_scanned_transposed,
        bit_offs,
        N,
        arr_out
    );

    if (tid == 0) {
        printf("debugging after write output\n bit_offs: %d\n", bit_offs + P::lgH - 1);
        debugPartitionCorrectness<P>(
            arr_out, 
            min(N, P::BLOCK_SIZE * P::Q),
            bit_offs + P::lgH - 1
        );
    }
    __syncthreads();

}








#endif // RANK_PERMUTE_CUH
