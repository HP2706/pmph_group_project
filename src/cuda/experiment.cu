#include "kernels.cuh"
#include "helper.h"
#include "traits.h"
#include "helper_kernels/prefix_sum.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
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
    uint32_t num_bins,
    uint32_t SIZE,
    uint32_t hist_size
) {
    static_assert(is_zero_extendable_to_uint32<UInt>::value, "UInt must be zero-extendable to uint32_t");

    *h_in = (UInt*) malloc(sizeof(UInt) * SIZE);
    *h_hist = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);

    // initialize h_hist to 0
    for (int i = 0; i < hist_size; i++) {
        (*h_hist)[i] = 0;
    }


    // 2. allocate device memory
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((uint32_t**) d_hist, sizeof(uint32_t) * hist_size);

    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
}


int main() {
    srand(2006);
 
    const int N_BITS = 8;
    const unsigned int NUM_BINS = 1 << N_BITS; // 2^8
    const unsigned int SIZE = pow(10, 8);
        // Example launch configuration
    const uint32_t BLOCK_SIZE = 256;

    
    
    uint32_t* h_in;
    uint32_t* d_in;
    uint32_t* d_histogram;
    uint32_t* h_histogram; 

    // Calculate grid size based on input size and elements per thread
    const uint32_t grid_size = (SIZE + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);
    // Change the histogram size calculation
    const uint32_t hist_size = NUM_BINS * grid_size; // This needs to be calculated before PrepareMemory

    PrepareMemory<uint32_t, BLOCK_SIZE>(
        &h_in, 
        &d_in, 
        &d_histogram, 
        &h_histogram,
        NUM_BINS,
        SIZE,
        hist_size
    );



    // Launch kernel
    HistoKernel1<uint32_t, N_BITS><<<grid_size, BLOCK_SIZE>>>(
        d_in, d_histogram, SIZE, 0, Q);
        
    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

    printf("Hist size: %d\n", hist_size);
    uint32_t sum = 0;
    for (int b = 0; b < hist_size; b++) {
        //printf("Final bin for one block %d: %d\n", b, h_histogram[b]);
        sum += h_histogram[b];
    }


    printf("sum: %d vs SIZE: %d\n", sum, SIZE);

    // there might be a bug in the histogram code as the sum and SIZE are not equal
    // HOWEVER, this is also the case for the assignment 3-4 code, which we expect to work
    // its a bit weird but we will take it up with cosmin
    // for (int i = 0; i < num_bins; i++) {
    //     printf("num bin %d: %d\n", i, h_hist[i]);
    // }
    free(h_in);
    free(h_histogram);
    cudaFree(d_in);
    cudaFree(d_histogram);
}
