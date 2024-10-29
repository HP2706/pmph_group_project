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
template<uint32_t BLOCK_SIZE>
void test_verify_transpose(
    uint32_t input_size
)
{
    uint32_t* h_in;
    uint32_t* d_in;
    uint32_t* d_out;
    uint32_t* d_histogram;
    uint32_t* h_histogram;
    uint32_t* d_histogram_transposed;
    uint32_t* d_hist_out;
    uint32_t* h_histogram_transposed;
    uint32_t* tranposedHistogramCPU;
    

    uint32_t NUM_BINS = 1 << 8;
    const uint32_t Q = 22;
    const uint32_t lgH = 8;
    
    // Calculate grid size based on input size and elements per thread
    const uint32_t grid_size = (input_size + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);
    uint32_t hist_size = NUM_BINS * grid_size;


    PrepareMemory<uint32_t, BLOCK_SIZE>(
        &h_in, 
        &d_in, 
        &d_histogram, 
        &h_histogram,
        NUM_BINS,
        input_size,
        hist_size
    );
    
    uint32_t* h_hist_out = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    
    // initialize h_histogram_transposed to 0
    h_histogram_transposed = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    tranposedHistogramCPU = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    for (int i = 0; i < hist_size; i++) {
        h_histogram_transposed[i] = 0;
        h_hist_out[i] = 0;
    }
    
    cudaMalloc((uint32_t**) &d_hist_out, sizeof(uint32_t) * hist_size);
    cudaMemcpy(d_hist_out, h_hist_out, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
    cudaMemset(d_hist_out, 0, sizeof(uint32_t) * hist_size);


    cudaMalloc((uint32_t**) &d_histogram_transposed, sizeof(uint32_t) * hist_size);
    cudaMemcpy(d_histogram_transposed, h_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram_transposed, 0, sizeof(uint32_t) * hist_size);
    
    constexpr uint32_t GRID_SIZE = 256;
    constexpr uint32_t BLOCK_DIM = 256;
    using SortParams = Params<uint32_t, uint32_t, Q, lgH, GRID_SIZE, BLOCK_DIM, 32, 16>;
    

    cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    transpose_kernel<SortParams>(
        d_histogram,
        d_histogram_transposed
    );
    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    
    verifyTranspose(h_histogram, tranposedHistogramCPU, h_histogram_transposed, NUM_BINS, BLOCK_DIM);
}
