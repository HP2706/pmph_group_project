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



void transposeCPU(uint32_t* input, uint32_t* output, int numRows, int numCols) 
{
    for (int i = 0; i < numRows; ++i) 
    {
        for (int j = 0; j < numCols; ++j) 
        { 
            uint32_t inputVal = input[i * numCols + j];
            output[j * numRows + i] = inputVal; 
        }
    }
}



void verifyTranspose(uint32_t* cpuInput, uint32_t* cpuOutput, uint32_t* gpuOutput, int numRows, int numCols)
{
    transposeCPU(cpuInput, cpuOutput, numRows, numCols);
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
void test_verify_transpose(
    uint32_t input_size
)
{
    static_assert(is_params<P>::value, "P must be a Params instance");
    // Calculate grid size based on input size and elements per thread

    uint32_t hist_size = P::H * P::GRID_SIZE;
    
    // ptr allocations
    
    typename P::UintType* h_histogram_transposed;
    typename P::UintType* d_histogram_transposed;

    typename P::UintType* h_histogram;
    typename P::UintType* d_histogram;

    typename P::UintType* cpu_h_histogram_transposed;
    

    allocateAndInitialize<typename P::UintType>(
        &h_histogram_transposed, 
        &d_histogram_transposed, 
        hist_size,
        false // we initialize to 0
    );

    allocateAndInitialize<typename P::UintType>(
        &h_histogram, 
        &d_histogram, 
        hist_size,
        true // we do not initialize to 0
    );


    // initialize cpu histogram transposed to 0
    cpu_h_histogram_transposed = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    for (int i = 0; i < hist_size; i++) {
        cpu_h_histogram_transposed[i] = 0;
    }

    transpose_kernel<P>(
        d_histogram,
        d_histogram_transposed
    );

    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

    // verify
    verifyTranspose(h_histogram, cpu_h_histogram_transposed, h_histogram_transposed, P::H, P::GRID_SIZE);
}
