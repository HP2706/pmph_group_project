#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "helper.h"

#define Q 22

int main() {

    uint32_t* h_in;
    uint32_t* d_in;
    uint32_t* d_out;
    uint32_t* d_histogram;
    uint32_t* h_histogram;
    
    uint32_t* d_histogram_transposed;
    uint32_t* h_histogram_transposed;
    

    const uint32_t SIZE = 1000000;
    const uint32_t NUM_BINS = 1 << 8;
    const uint32_t BLOCK_SIZE = 1024;
    
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

    // initialize h_histogram_transposed to 0
    h_histogram_transposed = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    for (int i = 0; i < hist_size; i++) {
        h_histogram_transposed[i] = 0;
    }

    cudaMalloc((uint32_t**) &d_histogram_transposed, sizeof(uint32_t) * hist_size);
    cudaMemcpy(d_histogram_transposed, h_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram_transposed, 0, sizeof(uint32_t) * hist_size);

    // initialize d_out to 0
    cudaMalloc((uint32_t**) &d_out, sizeof(uint32_t) * SIZE);
    cudaMemset(d_out, 0, sizeof(uint32_t) * SIZE);




    CountSort<uint32_t, grid_size, BLOCK_SIZE>(
        d_in,
        d_out,
        d_histogram,
        d_histogram_transposed,
        SIZE,
        0
    );

    return 0;

}