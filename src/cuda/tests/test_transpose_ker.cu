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



template<typename T>
void transposeCPU(T* input, T* output, int numRows, int numCols) 
{
    for (int i = 0; i < numRows; ++i) 
    {
        for (int j = 0; j < numCols; ++j) 
        { 
            T inputVal = input[i * numCols + j];
            output[j * numRows + i] = inputVal; 
        }
    }
}


template<typename T>
void verifyTranspose(
    T* cpuInput, 
    T* cpuOutput, 
    T* gpuOutput, 
    int numRows, 
    int numCols
)
{
    transposeCPU<T>(cpuInput, cpuOutput, numRows, numCols);
    bool success = true;
    uint32_t mismatchCount = 0;
    for (int i = 0; i < numRows * numCols; ++i) 
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
        std::cout << "Mismatch count: " << mismatchCount << " of " << numRows * numCols << " elements.\n";
    }
}

//Should probably be moved to separate file...
template<typename P>
void testTransposeKer(
    uint32_t input_size,
    int grid_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");
    // Calculate grid size based on input size and elements per thread

    uint32_t hist_size = P::H * grid_size;
    
    // ptr allocations
    
    using UintType = typename P::UintType;
    UintType* h_histogram_transposed = nullptr;
    UintType* d_histogram_transposed = nullptr;

    UintType* h_histogram = nullptr;
    UintType* d_histogram = nullptr;

    UintType* h_histogram_transposed_2 = nullptr;
    UintType* d_histogram_transposed_2 = nullptr;

    UintType* cpu_h_histogram_transposed = nullptr;
    

    allocateAndInitialize<UintType>(
        &h_histogram_transposed, 
        &d_histogram_transposed, 
        hist_size,
        false, // we initialize to 0,
        P::MAXNUMERIC_UintType
    );

    allocateAndInitialize<UintType>(
        &h_histogram, 
        &d_histogram, 
        hist_size,
        true, // we do not initialize to 0,
        P::MAXNUMERIC_UintType
    );

    allocateAndInitialize<UintType>(
        &h_histogram_transposed_2, 
        &d_histogram_transposed_2, 
        hist_size,
        false, // we initialize to 0,
        P::MAXNUMERIC_UintType
    );

    // initialize cpu histogram transposed to 0
    cpu_h_histogram_transposed = (UintType*) malloc(sizeof(UintType) * hist_size);
    for (int i = 0; i < hist_size; i++) {
        cpu_h_histogram_transposed[i] = 0;
    }

    tiled_transpose_kernel<UintType, P::T>(
        d_histogram,
        d_histogram_transposed,
        P::H,
        grid_size
    );

    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(UintType) * hist_size, cudaMemcpyDeviceToHost);
    verifyTranspose<UintType>(
        h_histogram, 
        cpu_h_histogram_transposed, 
        h_histogram_transposed, 
        P::H, 
        grid_size
    );

    tiled_transpose_kernel<UintType, P::T>(
        d_histogram_transposed,
        d_histogram_transposed_2,
        grid_size,
        P::H
    );

    cudaMemcpy(h_histogram_transposed_2, d_histogram_transposed_2, sizeof(UintType) * hist_size, cudaMemcpyDeviceToHost);
    printf("\n");

    // check if the double transposed histogram is the same as the original histogram
    validate<UintType>(h_histogram_transposed_2, h_histogram, hist_size);
    
    free(cpu_h_histogram_transposed);
    free(h_histogram_transposed);
    free(h_histogram_transposed_2);
    cudaFree(d_histogram_transposed);
    cudaFree(d_histogram_transposed_2);
}
