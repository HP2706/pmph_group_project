#include "kernels.cuh"
#include "helper.h"
#include "traits.h"
#include "helper_kernels/prefix_sum.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// Define the LLC size (Last Level Cache)
#define LLC 41943040 // number taken from assignment 3-4
#define Q 22
// Define the size of your input

template <typename UInt, int BLOCK_SIZE>
void PrepareMemory(
    UInt** h_in, 
    UInt** d_in, 
    uint32_t** d_hist, 
    uint32_t** h_hist,
    uint32_t** tranposedHistogram,
    uint32_t** tranposedHistogramCPU,
    uint32_t** h_tranposedHistogram,
    uint32_t num_bins,
    uint32_t SIZE,
    uint32_t hist_size
) {
    static_assert(is_zero_extendable_to_uint32<UInt>::value, "UInt must be zero-extendable to uint32_t");

    *h_in = (UInt*) malloc(sizeof(UInt) * SIZE);
    *h_hist = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    *tranposedHistogramCPU = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    *h_tranposedHistogram = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    // initialize h_hist to 0
    for (int i = 0; i < hist_size; i++) {
        (*h_hist)[i] = 0;
    }


    // 2. allocate device memory
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((uint32_t**) d_hist, sizeof(uint32_t) * hist_size);

    cudaMalloc((uint32_t**) tranposedHistogram, sizeof(uint32_t) * hist_size);


    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*tranposedHistogram, *h_hist, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
}

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
    printf("E\n");
    transposeCPU(cpuInput, cpuOutput, numRows, numCols);
    printf("F\n");
    bool success = true;
    for (int i = 0; i < numRows * numCols; ++i) 
    {
        if (cpuOutput[i] != gpuOutput[i]) 
        {
            success = false;
            std::cout << "Mismatch at index " << i << ": CPU " << cpuOutput[i] 
                      << " != GPU " << gpuOutput[i] << "\n";
        }
    }
    printf("G\n");
    
    if (success) 
    {
        std::cout << "Transpose verification succeeded.\n";
    } 
    else 
    {
        std::cout << "Transpose verification failed.\n";
    }
}

int main() {
    srand(2006);
 
    const int N_BITS = 8;
    const unsigned int num_bins = 1 << N_BITS; // 2^8
    const unsigned int SIZE = pow(10, 8);
    
    
    uint32_t* h_in;
    uint32_t* d_in;
    uint32_t* d_hist;
    uint32_t* h_hist; 
    
    uint32_t* tranposedHistogram;
    uint32_t* h_tranposedHistogram;
    uint32_t* tranposedHistogramCPU;    
    const int BLOCK_SIZE = 256;

    const uint32_t hist_size = num_bins * BLOCK_SIZE;

    PrepareMemory<uint32_t, BLOCK_SIZE>(
        &h_in, 
        &d_in, 
        &d_hist,
        &h_hist,
        &tranposedHistogram,
        &tranposedHistogramCPU,
        &h_tranposedHistogram,
        num_bins,
        SIZE,
        hist_size
    );

    printf("Copying h_in to d_in\n");
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * SIZE, cudaMemcpyHostToDevice);

    printf("Starting multiStepGenericHisto with size %d\n", SIZE);
    // Calculate grid dimensions
    const uint32_t elems_per_block = BLOCK_SIZE * Q;
    dim3 block(BLOCK_SIZE);
    dim3 grid((SIZE + elems_per_block - 1) / elems_per_block);  // Ceiling division to ensure all elements are processed
    

    // print the number of data ops we will make 
    // Calculate the total number of threads
    uint32_t total_threads = grid.x * block.x;
    printf("Total number of threads : %u\n", total_threads);
    printf("number of ops per thread X threads: %u diff from SIZE: %u\n", Q * total_threads, Q * total_threads - SIZE);
    

    RadixHistoKernel<uint32_t, N_BITS><<<grid, block>>>(
        d_in,                
        d_hist, 
        SIZE,
        0,
        Q
    );
    
    
    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_hist, d_hist, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

    printf("Hist size: %d\n", hist_size);
    uint32_t sum = 0;
    for (int b = 0; b < hist_size; b++) {
        /*if (h_hist[b] > 0)
        {
            printf("Final bin for one block %d: %d\n", b, h_hist[b]);
        }*/
        sum += h_hist[b];
    }


    printf("sum: %d vs SIZE: %d\n", sum, SIZE);
    
    //Transpose kernel
    TransposeHistoKernel<uint32_t, N_BITS><<<grid, block>>>(d_hist, tranposedHistogram, num_bins, BLOCK_SIZE);
    cudaMemcpy(h_tranposedHistogram, tranposedHistogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    verifyTranspose(h_hist, tranposedHistogramCPU, h_tranposedHistogram, num_bins, BLOCK_SIZE);


    // there might be a bug in the histogram code as the sum and SIZE are not equal
    // HOWEVER, this is also the case for the assignment 3-4 code, which we expect to work
    // its a bit weird but we will take it up with cosmin
    // for (int i = 0; i < num_bins; i++) {
    //     printf("num bin %d: %d\n", i, h_hist[i]);
    // }
    free(h_in);
    free(h_hist);
    cudaFree(d_in);
    cudaFree(d_hist);
}
