#include "kernels.cuh"
#include "helper.h"
#include "helper_kernels/prefix_sum.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Define the LLC size (Last Level Cache)
#define LLC 41943040 // number taken from assignment 3-4
#define Q 22
// Define the size of your input


int main() {
    srand(2006);
 

    const int N_RUNS = 10;
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

    printf("grid size: %d\n", grid_size);
    printf("hist size: %d\n", hist_size);

    PrepareMemory<uint32_t, BLOCK_SIZE>(
        &h_in, 
        &d_in, 
        &d_histogram, 
        &h_histogram,
        NUM_BINS,
        SIZE,
        hist_size
    );

    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < N_RUNS; i++) {
        // Launch kernel
        HistoKernel1<uint32_t, N_BITS, Q><<<grid_size, BLOCK_SIZE>>>(
        d_in, d_histogram, SIZE, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);


    elapsed /= N_RUNS;
    // Input: SIZE elements * (N_BITS/8) bytes (since we only process N_BITS per number)
    // Output: NUM_BINS elements * 4 bytes (uint32_t histogram bins)
    float gigaBytesPerSec = (SIZE * (N_BITS/8.0f) + 4.0f * NUM_BINS) * 1.0e-3f / elapsed;
    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }


    printf("gigaBytesPerSec: %f\n", gigaBytesPerSec);
    cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

    printf("Hist size: %d\n", hist_size);
    uint32_t sum = 0;
    for (int b = 0; b < hist_size; b++) {
        //printf("Final bin for one block %d: %d\n", b, h_histogram[b]);
        if (h_histogram[b] == 0) {
            printf("zero entry in bin %d: %d\n", b, h_histogram[b]);
        }
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
