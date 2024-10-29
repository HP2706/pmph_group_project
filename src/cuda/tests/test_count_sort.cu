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
#include "../helper_kernels/pbb_kernels.cuh"
#include "../helper_kernels/prefix_sum.cuh"
#include "../kernels.cuh"
#include "../cub_kernel.cuh"
#include <cuda_runtime.h>

template<typename P>
__host__ void test_count_sort(
    uint32_t input_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");

    // for this test we use uint8_t and thus count sort should fully sort the input
    // assuming lgH is 8 

    assert(P::lgH == 8);
    assert(sizeof(typename P::ElementType) * 8 == P::lgH && "Element type bits must match lgH ");

    uint32_t hist_size = P::H * P::GRID_SIZE;

    // ptr allocations
    typename P::ElementType* d_in;
    typename P::ElementType* h_in;

    typename P::ElementType* d_out;
    typename P::ElementType* h_out;

    typename P::UintType* d_hist;
    typename P::UintType* d_hist_transposed;
    typename P::UintType* d_hist_transposed_scanned_transposed;
    typename P::UintType* d_hist_transposed_scanned;
    typename P::UintType* d_tmp;

    // cub allocations
    typename P::ElementType* cub_d_in;
    typename P::ElementType* cub_d_out;
    typename P::ElementType* cub_h_out;

    printf("allocating memory\n");

    allocateAndInitialize<typename P::ElementType>(
        &h_in,
        &d_in,
        input_size,
        true // we initialize to random values
    );

    allocateAndInitialize<typename P::ElementType>(
        &h_out,
        &d_out,
        input_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        nullptr,
        &d_hist,
        hist_size,
        false // we initialize to random values
    );

    allocateAndInitialize<typename P::UintType>(
        nullptr,
        &d_hist_transposed,
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        nullptr,
        &d_hist_transposed_scanned,
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        nullptr,
        &d_hist_transposed_scanned_transposed,
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        nullptr,
        &d_tmp,
        hist_size,
        false // we initialize to 0
    );

    CountSort<P>(
        d_in, 
        d_out, 
        d_hist, 
        d_hist_transposed, 
        d_hist_transposed_scanned_transposed,
        d_hist_transposed_scanned, 
        d_tmp, 
        input_size, 
        0 // we use a bit position of 0 for this test
    );

    
    cudaMemcpy(h_out, d_out, sizeof(typename P::ElementType) * input_size, cudaMemcpyDeviceToHost);

    {
        printf("freeing memory\n");
        // we free some memory
        cudaFree(d_out);
        cudaFree(d_hist);
        cudaFree(d_hist_transposed);
        cudaFree(d_hist_transposed_scanned);
        cudaFree(d_hist_transposed_scanned_transposed);
        cudaFree(d_tmp);
    }

    printf("cub allocations\n");
    
    // cub allocations
    cudaMalloc((void**)&cub_d_in, sizeof(typename P::ElementType) * input_size);
    // we keep an array of the input in host memory to compare the results
    cudaMemcpy(cub_d_in, h_in, sizeof(typename P::ElementType) * input_size, cudaMemcpyHostToDevice);

    printf("allocating cub_h_out, cub_d_out\n");
    allocateAndInitialize<typename P::ElementType>(
        &cub_h_out,
        &cub_d_out,
        input_size,
        false 
    );


    printf("computing cub kernel\n");
    CUBSortKernel<
        typename P::ElementType, 
        P::BLOCK_SIZE, 
        P::Q
    ><<<P::GRID_SIZE, P::BLOCK_SIZE>>>
    (
        cub_d_in,
        cub_d_out,
        input_size
    );

    cudaMemcpy(cub_h_out, cub_d_out, sizeof(typename P::ElementType) * input_size, cudaMemcpyDeviceToHost);
    


    printf("validating results\n");
    assert(validate(cub_h_out, h_out, input_size));

    for (uint32_t i = 0; i < input_size; i++) {
        printf("%d: %d, %d\n", i, h_out[i], cub_h_out[i]);
    }

    {
        // we free some gpu-memory
        cudaFree(cub_d_in);
        cudaFree(cub_d_out);

        // we free some host-memory
        free(cub_h_out);
        free(h_in);
        free(h_out);
    }
   
   
}


