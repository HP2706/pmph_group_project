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
    uint32_t bit_offs,
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
        int bin = getBits<uint, P::lgH>(bit_offs, elm);
        uint32_t global_offs = global_histo[bin];
        uint32_t local_pos = q_idx*P::BLOCK_SIZE + tid;
        uint32_t global_pos = global_offs + local_pos;

        if (global_pos < N) {
            arr_out[global_pos] = elm;
        }
    }
}




template<class P>
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
    
    const int elms_buf_size = P::BLOCK_SIZE * P::Q;

    __shared__ uint shmem[elms_buf_size];
    __shared__ uint64_t global_histo[P::H];
    __shared__ uint16_t local_histo[P::BLOCK_SIZE];
    
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
        reg, // registers
        shmem, // shared memory
        local_histo, // shared memory
        bit_offs,
        N
    );
    __syncthreads();

    int debug_bit_offs = bit_offs + P::lgH - 1;
    

    /* Step 3: Final output generation
     * 1. Copy original and scanned histograms from global to shared memory
     * 2. Scan in place the original histogram for this block
     * 3. Each thread writes its Q elements to their final positions in global memory
     *    using the histogram information
    */

    WriteOutput<P>(
        reg, // registers
        global_histo, // shared memory
        local_histo, // shared memory
        d_hist, // global memory
        d_hist_transposed_scanned_transposed, // global memory
        bit_offs, // integer
        N, // integer
        arr_out // global memory
    );
}








#endif // RANK_PERMUTE_CUH
