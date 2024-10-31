#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cstdint>
#include <vector>

#include "../constants.cuh"
#include "../helper.h"
#include "../helper_kernels/utils.cuh"
#include "../helper_kernels/rank_permute.cuh"
#include "../kernels.cuh"
#include "../cub_kernel.cuh"
#include <cuda_runtime.h>



template<class P>
__global__ void testGlbToRegKernel(
    typename P::ElementType* arr_in,
    typename P::ElementType* arr_out,
    uint64_t N                       // Total number of elements
) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    using uint = typename P::ElementType;

    // Allocate shared memory
    extern __shared__ uint64_t sh_mem_uint64[];
    uint* shmem = (uint*) sh_mem_uint64;

    // Allocate local registers (Q elements per thread)
    uint reg[P::Q];

    // Call GlbToReg function
    GlbToReg<uint, P::Q, P::BLOCK_SIZE, P::MAXNUMERIC_ElementType>(N, shmem, arr_in, reg);

    __syncthreads();

    // 4. Write from shared memory back to global output array
    const uint32_t QB = P::BLOCK_SIZE * P::Q;
    const uint64_t glb_offs = bid * QB + tid * P::Q;

    for (int i = 0; i < P::Q; i++) {
        if (glb_offs + i < N) {
            arr_out[glb_offs + i] = shmem[P::Q * tid + i];
        }
    }
}

template<typename P>
bool verifyGlbToReg(const typename P::ElementType* arr_in, const typename P::ElementType* arr_out, uint64_t N) 
{
    bool success = true;
    const uint32_t QB = P::BLOCK_SIZE * P::Q;

    for (uint32_t bid = 0; bid < P::GRID_SIZE; ++bid) 
    {
        for (uint32_t tid = 0; tid < P::BLOCK_SIZE; ++tid) 
        {
            const uint32_t thread_start = bid * QB + tid * P::Q;
            for (int i = 0; i < P::Q; i++) 
            {
                uint64_t idx = thread_start + i;
                typename P::ElementType expected_value = (idx < N) ? arr_in[idx] : P::MAXNUMERIC_ElementType;

                if (arr_out[idx] != expected_value) 
                {
                    printf("Mismatch at index %d: expected %d, got %d\n", idx, expected_value, arr_out[idx]);
                    success = false;
                }
            }
        }
    }
    return success;
}

using ElementType = uint32_t;

template<typename P>
__host__ void test_glb_to_reg_ker(
    uint32_t N
)
{
    typename P::ElementType* h_in;
    typename P::ElementType* d_in;
    
    typename P::ElementType* h_out;
    typename P::ElementType* d_out;

    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &h_in, 
        &d_in, 
        N,
        true
    );

    for (int i = 0; i < N; ++i)
    {
        printf("Num: %d\n", h_in[i]);
    }

    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &h_out, 
        &d_out, 
        N,
        false
    );

    //Set shared memory size (assuming it fits the necessary elements)
    size_t shared_mem_size = P::BLOCK_SIZE * P::Q * sizeof(ElementType);

    //Launch kernel
    testGlbToRegKernel<P><<<P::GRID_SIZE, P::BLOCK_SIZE, P::QB*sizeof(typename P::ElementType)>>> 
    (
        d_in,
        d_out,
        N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("Test kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    //Copy result from device to host
    cudaMemcpy(h_out, d_out, N * sizeof(ElementType), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
    {
        printf("Expected vs what I got: %d vs %d\n", h_in[i], h_out[i]);
    }

/*    verifyGlbToReg<P>(h_in, h_out, N);

    //Verify output
    bool test_passed = true;
    for (int i = 0; i < N; i++) 
    {
        if (h_out[i] != h_in[i]) 
        {
            printf("Mismatch at index %d: expected %d, got %d\n", i, h_in[i], h_out[i]);
            test_passed = false;
            break;
        }
    }

    if (test_passed) 
    {
        std::cout << "Test passed: GlbToReg function works correctly." << std::endl;
    } 
    else 
    {
        std::cout << "Test failed: GlbToReg function has errors." << std::endl;
    }*/

    //Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}
