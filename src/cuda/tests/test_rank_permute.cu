#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cstdint>

#include "../constants.cuh"
#include "../helper.h"
#include "../helper_kernels/utils.cuh"
#include "../helper_kernels/rank_permute.cuh"
#include "../kernels.cuh"
#include "../cub_kernel.cuh"
#include <cuda_runtime.h>


/// NOTE this does not check correctness but it just checks that the kernel can be called
/// without errors
template<typename P>
__host__ void test_call_rank_permute_ker(
    uint32_t input_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");

    typename P::UintType* h_hist;
    typename P::UintType* d_hist;
    
    typename P::UintType* h_hist_transposed_scanned_transposed;
    typename P::UintType* d_hist_transposed_scanned_transposed;

    typename P::ElementType* h_in;
    typename P::ElementType* d_in;
    
    typename P::ElementType* h_out;
    typename P::ElementType* d_out;

    allocateAndInitialize<typename P::UintType>(
        &h_hist, 
        &d_hist, 
        input_size,
        true
    );

    allocateAndInitialize<typename P::UintType>(
        &h_hist_transposed_scanned_transposed, 
        &d_hist_transposed_scanned_transposed, 
        input_size,
        true
    );

    allocateAndInitialize<typename P::ElementType>(
        &h_in, 
        &d_in, 
        input_size,
        true
    );

    allocateAndInitialize<typename P::ElementType>(
        &h_out, 
        &d_out, 
        input_size,
        false
    );


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    RankPermuteKer<P><<<P::GRID_SIZE, P::BLOCK_SIZE, P::QB*sizeof(typename P::ElementType)>>> 
    (
        d_hist,
        d_hist_transposed_scanned_transposed,
        0,
        input_size,
        d_in,
        d_out
    );
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("RankPermuteKer took %f milliseconds\n", milliseconds);

    //cudaMemcpy(h_out, d_out, sizeof(typename P::ElementType) * input_size, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Rank and permute kernel failed: %s\n", cudaGetErrorString(err));
        return;
    } else {
        printf("RankPermuteKer done no cuda error\n");
    }
}