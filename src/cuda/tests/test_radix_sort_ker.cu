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

    // allocate memory
    allocateAndInitialize<typename P::ElementType>(
        &cub_h_in,
        &cub_d_in,
        input_size,
        true
    );

    allocateAndInitialize<typename P::ElementType>(
        &cub_h_out,
        &cub_d_out,
        input_size,
        false
    );


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

    // print top 100 elements in reverse order
    for (int i = 0; i < 1000; i++) {
        printf("%d\n", cub_h_out[input_size - i - 1]);
    }


}