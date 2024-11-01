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


    typename P::ElementType* d_out_debug;
    cudaMalloc((typename P::ElementType**) &d_out_debug, input_size * sizeof(typename P::ElementType));

    RadixSortKer<P>(
        d_in,
        d_out,
        d_out_debug,
        input_size
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("radix sort kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out, d_out_debug, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_size; i++) {
        printf("line 113 h_out[%d]: %u\n", i, h_out[i]);
    }


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


    for (int i = 0; i < input_size; i++) {
        printf("cub_h_in[%d]: %u\n", i, cub_h_in[i]);
    }
    // Calculate grid size based on input size
    int elements_per_block = P::BLOCK_SIZE * P::Q;
    int grid_size = (input_size + elements_per_block - 1) / elements_per_block;
    
    CUBSortKernel<typename P::ElementType, P::BLOCK_SIZE, P::Q>
    <<<grid_size, P::BLOCK_SIZE>>>
    (
        cub_d_in, 
        cub_d_out, 
        input_size
    );


    cudaDeviceSynchronize();
    cudaError_t cub_err = cudaGetLastError();
    if (cub_err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(cub_err));
        return;
    }
    // copy results back to host
    cudaMemcpy(cub_h_out, cub_d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < input_size; i++) {
        printf("cub_h_out[%d]: %u\n", i, cub_h_out[i]);
    }

    
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

    //printf("checking cub matches radix\n");
    //assert(validate(cub_h_out, h_out, input_size));


}