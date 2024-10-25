#include "kernels.cuh"
#include "helper.h"
#include "helper_kernels/prefix_sum.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Define the LLC size (Last Level Cache)
#define LLC 41943040 // number taken from assignment 3-4

// Define the size of your input

template <typename UInt>
void PrepareMemory(
    UInt** h_in, 
    UInt** d_in, 
    uint32_t** d_hist, 
    uint32_t** h_hist,
    uint32_t num_bins,
    uint32_t SIZE
) {
    *h_in = (UInt*) malloc(sizeof(UInt) * SIZE);
    *h_hist = (uint32_t*) malloc(sizeof(uint32_t) * num_bins);
 
    // 2. allocate device memory
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((UInt**) d_hist, sizeof(uint32_t) * num_bins);


    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(uint32_t) * num_bins, cudaMemcpyHostToDevice);
}


int main() {
    srand(2006);
 
    unsigned int num_bins = 1 << 8; // 2^8
    unsigned int SIZE = pow(10, 8);
    uint8_t* h_in;
    uint8_t* d_in;
    uint32_t* d_hist;
    uint32_t* h_hist; 
    
    PrepareMemory<uint8_t>(
        &h_in, 
        &d_in, 
        &d_hist, 
        &h_hist, // Add this argument to the function call
        num_bins,
        SIZE
    );

    printf("Copying h_in to d_in\n");
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * SIZE, cudaMemcpyHostToDevice);

    printf("Starting multiStepGenericHisto\n");
    const int BLOCK_SIZE = 256; // Add this line to define block size
    multiStepGenericHisto<uint8_t, BLOCK_SIZE>(
        (const uint8_t*)d_in,
        d_hist, 
        SIZE, 
        num_bins, 
        LLC
    );

    cudaMemcpy(h_hist, d_hist, sizeof(uint32_t) * num_bins, cudaMemcpyDeviceToHost);

    // check it sums to the size
    uint32_t sum = 0;
    for (int i = 0; i < num_bins; i++) {
        
        printf("num bin %d: %d\n", i, h_hist[i]);
        sum += h_hist[i];
    }
    printf("sum: %d vs SIZE: %d\n", sum, SIZE);

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
