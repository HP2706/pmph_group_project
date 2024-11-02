#ifndef RANK_PERMUTE_CUH
#define RANK_PERMUTE_CUH

#include "../constants.cuh"
#include "../helper.h"
#include "pbb_kernels.cuh"
#include <limits>

template<class T, int B, int Q, int largest>
void __device__ cpGlb2Reg(
    uint64_t N,
    T* shmem,
    T* arr,
    T reg[Q]
){
    const uint32_t M = B*Q;

    const uint64_t glb_offs = blockIdx.x * M;

    // each thread loads q elements into shared memory
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t loc_pos = i*B + threadIdx.x;
        uint64_t glb_pos = glb_offs + loc_pos;


        T el = largest;
        if (glb_pos < N) {
            el = arr[glb_pos];
        }
        shmem[loc_pos] = el;
    }
    __syncthreads();

    // each thread saves q elements into its register
    for (int i = 0; i < Q; i++) {
        reg[i] = shmem[Q*threadIdx.x + i];
    }
    __syncthreads();

}


template <class P>
__global__ void rank_permute_ker(
    typename P::ElementType* d_in,
    typename P::ElementType* d_out,
    uint64_t* d_histogram,
    uint64_t* d_histogram_t_scanned_t,
    uint32_t input_size,
    uint32_t bit_offs
) { 

    using uint = typename P::ElementType;
    // check p is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    __shared__ uint64_t shared_buf[P::BLOCK_SIZE*P::Q]; // we have QB elements for each block
    uint* elmshm = (uint*) shared_buf;
    uint64_t* global_histogram = (uint64_t*) shared_buf;
    uint64_t* local_histogram = (uint64_t*) global_histogram + P::H;

    // allocate registers

    uint reg[P::Q]; // each thread has a register with q elements

    cpGlb2Reg<uint, P::BLOCK_SIZE, P::Q, P::MAXNUMERIC_ElementType>(
        input_size, 
        elmshm, 
        d_in, 
        reg
    );

    uint32_t thread_offs = threadIdx.x * P::Q; 

    // we do lgH passes
    for (uint16_t i = 0; i < P::lgH; i++) {

        // each thread computes an inclusive scan across its registers checking 
        // if the bit_offs + i'th bit is set or not
        uint32_t accum = 0; 
        for (uint16_t q = 0; q < P::Q; q++) {
            accum += isBitUnset<uint>(P::lgH + i, reg[q]);
        }

        // we store the result in the global histogram
        // note accum is the last element in the register scan
        global_histogram[threadIdx.x] = accum;
        __syncthreads();
        // wait for everyone to finish

        uint16_t block_res = scanIncBlock<Add<uint64_t>>(
            global_histogram, 
            threadIdx.x
        );
        __syncthreads();
        // we write the result in the local histogram
        local_histogram[threadIdx.x] = block_res;

        // the partition point is the last element in the local histogram
        int64_t partition_point = local_histogram[P::BLOCK_SIZE - 1];

        if (threadIdx.x == 0) {
            accum = 0;
        } else {
            accum = local_histogram[threadIdx.x - 1];
        }

        __syncthreads();

        for (uint16_t q = 0; q < P::Q; q++) {
            uint el = reg[q];
            uint16_t is_unset = isBitUnset<uint>(P::lgH + i, el);
            accum += is_unset;

            int pos;
            if (is_unset) {
                pos = accum - 1;
            } else {
                pos = partition_point + thread_offs + q - accum;
            }

            // we write the result to shared memory
            elmshm[pos] = el;
        }
        __syncthreads();

        // we update the registers with the new values from shared memory
        if (i < P::lgH - 1) {
            for (uint16_t q = 0; q < P::Q; q++) {
                reg[q] = elmshm[thread_offs + q];
            }
        } else {
            for (uint16_t q = 0; q < P::Q; q++) {
                // fetch the pos in shared memory
                int pos = q*P::BLOCK_SIZE + threadIdx.x;
                reg[q] = elmshm[pos];
            }
        }

        __syncthreads();
    }

    // write the result to global memory
    int global_offs = blockIdx.x * P::H;
    for (int t = threadIdx.x; t < P::H; t += P::BLOCK_SIZE) {
        uint64_t local = d_histogram[global_offs + t];
        uint64_t global = d_histogram_t_scanned_t[global_offs + t];

        global_histogram[t] = global - local;
        local_histogram[t] = local;
    }

    int result = scanIncBlock<Add<uint64_t>>(local_histogram, threadIdx.x);
    // shift each element by one to convert inclusive scan to exclusive scan
    if (threadIdx.x < P::H-1) {
        global_histogram[threadIdx.x+1] = result;
    }
    __syncthreads();


    // write the result to global memory
    for (uint16_t q = 0; q < P::Q; q++) {
        uint64_t el = reg[q];
        int pos = getDigit<uint64_t, P::lgH>(bit_offs+q, el);
        uint64_t global_radix_offset = global_histogram[pos];
        uint64_t global_pos = global_radix_offset + q*P::BLOCK_SIZE + threadIdx.x;
        if (global_pos < input_size) {
            d_out[global_pos] = el;
        }
    }

}

#endif // RANK_PERMUTE_CUH
