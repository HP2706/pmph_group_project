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
__host__ void testRadixSortKer(
    uint32_t input_size,
    int grid_size
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

    allocateAndInitialize<typename P::ElementType>(
        &h_out,
        &d_out,
        input_size,
        false,
        P::MAXNUMERIC_ElementType
    );

    // allocate memory
    allocateAndInitialize<typename P::ElementType>(
        &cub_h_in,
        &cub_d_in,
        input_size,
        true,
        P::MAXNUMERIC_ElementType
    );

    allocateAndInitialize<typename P::ElementType>(
        &cub_h_out,
        &cub_d_out,
        input_size,
        false,
        P::MAXNUMERIC_ElementType
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

    RadixSortKer<P>(
        d_in,
        d_out,
        input_size
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("radix sort kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out, d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);
    quickvalidatesortedarray<uint32_t>(input_size, h_out);

    deviceRadixSortKernel<typename P::ElementType>(
        cub_d_in,
        cub_d_out,
        input_size
    );
    
    // copy results back to host
    cudaMemcpy(cub_h_out, cub_d_out, input_size * sizeof(typename P::ElementType), cudaMemcpyDeviceToHost);
    quickvalidatecub<uint32_t>(input_size, cub_h_out);
    assertCubIsEqualToOurImplementation<uint32_t>(input_size, h_out, cub_h_out);
}
