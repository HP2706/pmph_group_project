#ifndef RANK_PERMUTE_CUH
#define RANK_PERMUTE_CUH

#include "utils.cuh"

// this is not a kernel but a compound function that calls many kernels
template<class ElTp, int Q, int lgH>
__global__ void RankPermuteKer(
    ElTp* d_hist_transposed,
    ElTp* d_out,
    int N
) {
    extern __shared__ char sh_mem[];
    volatile ElTp* shmem_inp = (ElTp*)sh_mem;
    volatile ElTp* shmem_hist = (ElTp*)(sh_mem + Q * blockDim.x * sizeof(ElTp));
    
    ElTp reg_vals[Q];
    uint32_t reg_flags[Q];
    
    uint32_t tid = threadIdx.x;
    uint32_t num_elems_per_iter = Q * blockDim.x;
    uint32_t gid = blockIdx.x * num_elems_per_iter + tid;

    for (int seq = 0; seq < N; seq += num_elems_per_iter) {
        // Step 1: Load Q elements per thread into registers
        #pragma unroll
        for (int i = 0; i < Q; i++) {
            if (gid + i * blockDim.x < N) {
                reg_vals[i] = d_hist_transposed[gid + i * blockDim.x];
            }
        }

        // Step 2: Two-way partitioning loop for lgH bits
        #pragma unroll
        for (int bit = 0; bit < lgH; bit++) {
            // Phase 1: Sequential reduction of Q elements
            uint32_t local_count = 0;
            #pragma unroll
            for (int i = 0; i < Q; i++) {
                reg_flags[i] = ((reg_vals[i] >> bit) & 1);
                local_count += reg_flags[i];
            }
            
            // Store local count to shared memory
            shmem_inp[tid] = local_count;
            __syncthreads();

            // we scan the shared memory array
            local_count = scanIncBlock<Add<uint32_t>>(shmem_inp, tid);
            __syncthreads();

            // Get prefix for current thread
            uint32_t prefix = (tid > 0) ? shmem_inp[tid-1] : 0;
            
            // Rearrange elements based on their bits
            #pragma unroll
            for (int i = 0; i < Q; i++) {
                if (reg_flags[i]) {
                    reg_vals[i] += prefix;
                }
            }
            __syncthreads();
        }

        // Step 3: Write results back to global memory
        #pragma unroll
        for (int i = 0; i < Q; i++) {
            if (gid + i * blockDim.x < N) {
                d_out[gid + i * blockDim.x] = reg_vals[i];
            }
        }
        __syncthreads();
    }
}

#endif // RANK_PERMUTE_CUH
