#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cstdint>

#include "../helper.h"
#include "../constants.cuh"
#include "../helper_kernels/utils.cuh"
#include "../helper_kernels/pbb_kernels.cuh"
#include "../helper_kernels/prefix_sum.cuh"
#include "../kernels.cuh"

#if 1

template<typename T>
void transposeCPU(
    T* input, 
    T* output, 
    uint32_t height, 
    uint32_t width
) 
{
    for (uint32_t i = 0; i < width; ++i) 
    {
        for (uint32_t j = 0; j < height; ++j) 
        { 
            T inputVal = input[i * height + j];
            output[j * width + i] = inputVal; 
        }
    }
}


template<typename T>
void verifyTranspose(
    T* cpuInput, 
    T* cpuOutput, 
    T* gpuOutput, 
    uint32_t height, 
    uint32_t width
)
{
    transposeCPU<T>(cpuInput, cpuOutput, height, width);
    bool success = true;
    uint32_t mismatchCount = 0;
    for (uint32_t i = 0; i < width * height; ++i) 
    {
        if (cpuOutput[i] != gpuOutput[i]) 
        {
            success = false;
            std::cout << "Mismatch at index " << i << ": CPU " << cpuOutput[i] 
                      << " != GPU " << gpuOutput[i] << "\n";
            ++mismatchCount;
        }
    }
    
    if (success) 
    {
        std::cout << "Transpose verification succeeded.\n";
    } 
    else 
    {
        std::cout << "Transpose verification failed.\n";
        std::cout << "Mismatch count: " << mismatchCount << " of " << width * height << " elements.\n";
    }
}

//Should probably be moved to separate file...
template<typename P>
void test_verify_transpose(
    uint32_t input_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");
    // Calculate grid size based on input size and elements per thread

    uint32_t hist_size = P::H * P::NUM_BLOCKS;
    
    // ptr allocations
    
    using UintType = typename P::UintType;
    UintType* h_histogram_transposed = nullptr;
    UintType* d_histogram_transposed = nullptr;

    UintType* h_histogram = nullptr;
    UintType* d_histogram = nullptr;

    UintType* h_histogram_transposed_2 = nullptr;
    UintType* d_histogram_transposed_2 = nullptr;

    UintType* cpu_h_histogram_transposed = nullptr;
    

    allocateAndInitialize<UintType, P::MAXNUMERIC_UintType>(
        &h_histogram_transposed, 
        &d_histogram_transposed, 
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<UintType, P::MAXNUMERIC_UintType>(
        &h_histogram, 
        &d_histogram, 
        hist_size,
        true // we do not initialize to 0
    );

    allocateAndInitialize<UintType, P::MAXNUMERIC_UintType>(
        &h_histogram_transposed_2, 
        &d_histogram_transposed_2, 
        hist_size,
        false // we initialize to 0
    );

    // initialize cpu histogram transposed to 0
    cpu_h_histogram_transposed = (UintType*) malloc(sizeof(UintType) * hist_size);
    for (int i = 0; i < hist_size; i++) {
        cpu_h_histogram_transposed[i] = 0;
    }

    tiledTranspose<UintType, P::TILE_SIZE>(
        d_histogram,
        d_histogram_transposed,
        P::NUM_BLOCKS,
        P::H
    );

    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(UintType) * hist_size, cudaMemcpyDeviceToHost);
    verifyTranspose<UintType>(
        h_histogram, 
        cpu_h_histogram_transposed, 
        h_histogram_transposed,
        P::H, // width
        P::NUM_BLOCKS // height
    );

    tiledTranspose<UintType, P::TILE_SIZE>(
        d_histogram_transposed,
        d_histogram_transposed_2,
        P::H, //width
        P::NUM_BLOCKS //height
    );

    cudaMemcpy(h_histogram_transposed_2, d_histogram_transposed_2, sizeof(UintType) * hist_size, cudaMemcpyDeviceToHost);
    printf("\n");

    printf("check double transpose equals original\n");
    // check if the double transposed histogram is the same as the original histogram
    validate<UintType>(h_histogram_transposed_2, h_histogram, hist_size);
    
    free(cpu_h_histogram_transposed);
    free(h_histogram_transposed);
    free(h_histogram_transposed_2);
    cudaFree(d_histogram_transposed);
    cudaFree(d_histogram_transposed_2);
}

#endif