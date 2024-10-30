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
    typename P::UintType* h_hist;
    

    typename P::UintType* d_hist_transposed;
    typename P::UintType* h_hist_transposed;

    typename P::UintType* d_hist_transposed_scanned_transposed;
    typename P::UintType* h_hist_transposed_scanned_transposed;

    typename P::UintType* d_hist_transposed_scanned;
    typename P::UintType* h_hist_transposed_scanned;

    typename P::UintType* d_tmp;

    // cub allocations
    typename P::ElementType* cub_d_in;
    typename P::ElementType* cub_h_in;
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
        &h_hist,
        &d_hist,
        hist_size,
        false // we initialize to random values
    );

    allocateAndInitialize<typename P::UintType>(
        &h_hist_transposed,
        &d_hist_transposed,
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        &h_hist_transposed_scanned,
        &d_hist_transposed_scanned,
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        &h_hist_transposed_scanned_transposed,
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

    allocateAndInitialize<typename P::ElementType>(
        &cub_h_out,
        &cub_d_out,
        input_size,
        false 
    ); 

    // cub allocations
    allocateAndInitialize<typename P::ElementType>(
        &cub_h_in,
        &cub_d_in,
        input_size,
        false // we initialize to 0 and copy from h_in
    );

    //we copy the host input to the cub array
    memcpy(cub_h_in, h_in, sizeof(typename P::ElementType) * input_size);

    /*
    // we check that the host input and cub input are the same
    assert(validate(cub_h_in, h_in, input_size));

    // we copy the host input to the cub array
    cudaMemcpy(cub_d_in, cub_h_in, sizeof(typename P::ElementType) * input_size, cudaMemcpyHostToDevice);

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
    */

    /* CountSort<P>(
        d_in, 
        d_out, 
        d_hist, 
        d_hist_transposed, 
        d_hist_transposed_scanned_transposed,
        d_hist_transposed_scanned, 
        d_tmp, 
        input_size, 
        0 // we use a bit position of 0 for this test
    ); */


    // TODO
    // we should use different uint types for the histogram and the scanned histogram
    // for a faster kernel

    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");


    uint32_t bit_pos = 0;
    uint32_t N = input_size;

    Histo<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Histogram kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_hist, d_hist, sizeof(typename P::UintType) * hist_size, cudaMemcpyDeviceToHost);
    checkAllZeros(h_hist, hist_size);
    printf("Histogram kernel done and checked all zeros\n");

    tiled_transpose_kernel<typename P::UintType, P::T>(
        d_hist,
        d_hist_transposed,
        P::H,
        P::GRID_SIZE
    );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("First transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemcpy(h_hist_transposed, d_hist_transposed, sizeof(typename P::UintType) * hist_size, cudaMemcpyDeviceToHost);
    checkAllZeros(h_hist_transposed, hist_size);
    printf("First transpose kernel done and checked all zeros\n");

    // in the future use something like FusedAddCast<uint16_t, uint32_t>
    scanInc<Add<typename P::UintType>>(
        P::BLOCK_SIZE, // Block size
        P::HB,          // Histogram size
        d_hist_transposed_scanned,     // output: scanned histogram
        d_hist_transposed,
        d_tmp              // temporary storage
    );

    cudaMemcpy(h_hist_transposed_scanned, d_hist_transposed_scanned, sizeof(typename P::UintType) * hist_size, cudaMemcpyDeviceToHost);
    checkAllZeros(h_hist_transposed_scanned, hist_size);
    printf("Scan kernel done and checked all zeros\n");
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Scan kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_hist_transposed_scanned_transposed, d_hist_transposed_scanned_transposed, sizeof(typename P::UintType) * hist_size, cudaMemcpyDeviceToHost);
    checkAllZeros(h_hist_transposed_scanned_transposed, hist_size);
    printf("Second transpose kernel done and checked all zeros\n");

    // we do inclusive scan here
    // transpose back to original histogram
    tiled_transpose_kernel<typename P::UintType, P::T>(
        d_hist_transposed_scanned,
        d_hist_transposed_scanned_transposed,
        P::GRID_SIZE,
        P::H
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Second transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_hist_transposed_scanned_transposed, d_hist_transposed_scanned_transposed, sizeof(typename P::UintType) * hist_size, cudaMemcpyDeviceToHost);
    checkAllZeros(h_hist_transposed_scanned_transposed, hist_size);
    printf("Third transpose kernel done and checked all zeros\n");

    RankPermuteKer<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_hist,
        d_hist_transposed_scanned_transposed,
        bit_pos,
        N,
        d_in,
        d_out
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Rank and permute kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out, d_out, sizeof(typename P::ElementType) * input_size, cudaMemcpyDeviceToHost);
    printf("Rank and permute kernel done\n");

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CountSort Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }


    /* cudaMemcpy(h_out, d_out, sizeof(typename P::ElementType) * input_size, cudaMemcpyDeviceToHost);

    // we check that the host output and cub output are the same
    assert(validate(cub_h_out, h_out, input_size)); */



    {
        printf("freeing memory\n");
        // we free some gpu-memory
        cudaFree(cub_d_in);
        cudaFree(cub_d_out);

        // we free some memory
        cudaFree(d_out);
        cudaFree(d_hist);
        cudaFree(d_hist_transposed);
        cudaFree(d_hist_transposed_scanned);
        cudaFree(d_hist_transposed_scanned_transposed);
        cudaFree(d_tmp);

        // we free some host-memory
        free(cub_h_out);
        free(h_in);
        free(h_out);
    }
   
   
}


