#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cstdint>
#include <fstream>

#include "../constants.cuh"
#include "../helper.h"
#include "../helper_kernels/utils.cuh"
#include "../helper_kernels/pbb_kernels.cuh"
#include "../helper_kernels/prefix_sum.cuh"
#include "../kernels.cuh"
#include "../cub_kernel.cuh"
#include <cuda_runtime.h>

template<typename P>
__host__ void test_radix_sort_ker(
    uint32_t input_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");

    // declare ptrs
    typename P::ElementType* cub_h_in;
    typename P::ElementType* cub_d_in;
    typename P::ElementType* cub_h_out;
    typename P::ElementType* cub_d_out;
    
    typename P::ElementType* d_in;
    typename P::ElementType* h_in;
    typename P::ElementType* d_out;
    typename P::ElementType* h_out;



    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &h_out,
        &d_out,
        input_size,
        false
    );

    // allocate memory
    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &cub_h_in,
        &cub_d_in,
        input_size,
        true
    );

    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &cub_h_out,
        &cub_d_out,
        input_size,
        false
    );


    cudaMalloc((typename P::ElementType**) &d_in, input_size * sizeof(typename P::ElementType));
    h_in = (typename P::ElementType*) malloc(input_size * sizeof(typename P::ElementType));

    // copy cub_h_in to h_in
    memcpy(h_in, cub_h_in, input_size * sizeof(typename P::ElementType));

    // copy h_in to d_in
    cudaMemcpy(d_in, h_in, input_size * sizeof(typename P::ElementType), cudaMemcpyHostToDevice);

    // check that thecub and radix sort inputs are the same
    printf("checking that cub and radix sort inputs are the same\n");
    assert(validate(cub_h_in, h_in, input_size));


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

    // copy results back to host
    cudaMemcpy(cub_h_out, cub_d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }


    RadixSortKer<P>(
        d_in,
        d_out,
        input_size
    );


    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("radix sort kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out, d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);

    #if 0
    // Find and print the exact location where sorting fails
    for (uint32_t i = 1; i < input_size; i++) {
        if (h_out[i] < h_out[i-1]) {
            printf("\nRadixSort failure at index %d: %d > %d\n", 
                i, h_out[i-1], h_out[i]);
            // Print surrounding elements for context
            printf("Elements around failure point:\n");
            for (int j = max(0, (int)i-5); j < min(input_size, i+5); j++) {
                printf("h_out[%d]: %d\n", j, h_out[j]);
            }
            break;
        }
    }

    if (!checkSorted(h_out, input_size)) {
        printf("RadixSortKer is not sorted\n");
        // print top 100 elements
        printf("top 100 elements:\n");
        for (uint32_t i = 0; i < 100; i++) {
            printf("h_out[%d]: %d\n", i, h_out[i]);
        }
    } else {
        printf("RadixSortKer is sorted\n");
    }
    #endif

    // if file already exists, delete it
    std::remove("radix_sort_output.txt");

    // Write RadixSort output to file
    std::ofstream radix_file("radix_sort_output.txt");
    if (radix_file.is_open()) {
        for (uint32_t i = 0; i < input_size; i++) {
            radix_file << h_out[i] << "\n";
        }
        radix_file.close();
    } else {
        printf("Error: Could not open radix_sort_output.txt\n");
    }

    printf("writing cub sort output to file\n");

    std::remove("cub_sort_output.txt");

    // Write CUBSort output to file
    std::ofstream cub_file("cub_sort_output.txt");
    if (cub_file.is_open()) {
        for (uint32_t i = 0; i < input_size; i++) {
            cub_file << cub_h_out[i] << "\n";
        }
        cub_file.close();
    } else {
        printf("Error: Could not open cub_sort_output.txt\n");
    }

    printf("checking cub matches radix\n");
    assert(validate(cub_h_out, h_out, input_size));

}