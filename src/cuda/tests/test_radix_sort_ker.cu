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

template<class T>
void quickvalidatesortedarray(
    uint32_t N,
    T* arr_inp
) {
    bool success = true;
    for (uint32_t i = 0; i < N-1; ++i)
    {
        if (arr_inp[i] > arr_inp[i+1])
        {
            printf("Element %d: %d is larger than element %d: %d in our array, thus it is not sorted correctly.\n", i, arr_inp[i], i+1, arr_inp[i+1]);
            success = false;
        }
    }
    if (success)
    {
        printf("Sorted entire array successfully with our implementation! (No errors)\n");
    }
    else
    {
        printf("Sorted entire array incorrectly! (Errors)\n");
    }
}

template<class T>
void quickvalidatecub(
    uint32_t N,
    T* arr_inp
) {
    bool success = true;
    for (uint32_t i = 0; i < N-1; ++i)
    {
        if (arr_inp[i] > arr_inp[i+1])
        {
            printf("Element %d: %d is larger than element %d: %d in our CUB array, thus it is not sorted correctly.\n", i, arr_inp[i], i+1, arr_inp[i+1]);
            success = false;
        }
    }
    if (success)
    {
        printf("Sorted entire array successfully with CUB! (No errors)\n");
    }
    else
    {
        printf("Sorted entire array incorrectly with cub! (Errors)\n");
    }
}

template<class T>
void assertCubIsEqualToOurImplementation(
    uint32_t N,
    T* arr_inp,
    T* arr_inpOurs
) {
    bool success = true;
    for (uint32_t i = 0; i < N-1; ++i)
    {
        if (arr_inp[i] != arr_inpOurs[i])
        {
            printf("Element %d: %d from ours implementation is not equal to %d in our CUB array, thus it is not sorted correctly.\n", i, arr_inpOurs[i], arr_inp[i]);
            success = false;
        }
    }
    if (success)
    {
        printf("CUB sorted array matches our sorted array!\n");
    }
    else
    {
        printf("CUB sorted array does NOT match our sorted array!\n");
    }
}

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

    /* for (int i = 0; i < input_size; i++) {
        printf("line 113 h_out[%d]: %u\n", i, h_out[i]);
    } */

    quickvalidatesortedarray<uint32_t>(input_size, h_out);

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



    // Calculate grid size based on input size
    int elements_per_block = P::BLOCK_SIZE * P::Q;
    int grid_size = (input_size + elements_per_block - 1) / elements_per_block;
    
    void* mem = NULL;
    size_t len = 0;
    /*CUBSortKernel<typename P::ElementType, P::BLOCK_SIZE, P::Q>
    <<<grid_size, P::BLOCK_SIZE>>>
    (
        cub_d_in, 
        cub_d_out, 
        input_size
    );*/
    //Get mem and length
    
    uint32_t startBit = 0;
    uint32_t endBit = 32;
    
    cub::DeviceRadixSort::SortKeys(mem, len, cub_d_in, cub_d_out, input_size, startBit, endBit);
    cudaMalloc(&mem, len);

    cudaDeviceSynchronize();
    cudaError_t cub_err = cudaGetLastError();
    if (cub_err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(cub_err));
        return;
    }

    cub::DeviceRadixSort::SortKeys(mem, len, cub_d_in, cub_d_out, input_size, startBit, endBit);
    cudaDeviceSynchronize();
    
    cub_err = cudaGetLastError();
    if (cub_err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(cub_err));
        return;
    }
    
    
    // copy results back to host
    cudaMemcpy(cub_h_out, cub_d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);
    quickvalidatecub<uint32_t>(input_size, cub_h_out);
    assertCubIsEqualToOurImplementation<uint32_t>(input_size, h_out, cub_h_out);
    
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
