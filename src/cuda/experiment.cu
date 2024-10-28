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
    uint32_t* d_hist_out;
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

    uint32_t* h_hist_out = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
    // initialize h_histogram_transposed to 0
    h_histogram_transposed = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);
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

    // initialize d_out to 0
    cudaMalloc((uint32_t**) &d_out, sizeof(uint32_t) * SIZE);
    cudaMemset(d_out, 0, sizeof(uint32_t) * SIZE);

    uint32_t* h_out = (uint32_t*) malloc(sizeof(uint32_t) * SIZE);

    uint32_t height = NUM_BINS;
    uint32_t width = grid_size;

    randomInit<uint32_t>(h_histogram, hist_size, NUM_BINS);
    cudaMemcpy(d_histogram, h_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);

    printf("SIZE: %d, height: %d, width: %d, height times width: %d\n", hist_size, height, width, height * width);

    printf("height: %d, width: %d\n", height, width);
    // Update the kernel call with correct dimensions

    CountSort<uint32_t, grid_size, BLOCK_SIZE, 16>(
        d_in,
        d_out,
        d_histogram,
        d_histogram_transposed,
        d_hist_out,
        SIZE,
        0
    );

    // Add after kernel call
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }



    cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_out, d_hist_out, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

    printf("h_histogram: ");
    for (int i = 0; i < hist_size; i++) {
        if (0 == h_histogram[i]) {
            printf("%d ", h_histogram[i]);
        }
    }
    printf("\n");

    
    printf("\n");

   

    /* printf("h_hist_out post exclusive scan: ");
    for (int i = 0; i < hist_size; i++) {
        printf("%d ", h_hist_out[i]);
    }
    printf("\n"); */

    /* cudaMemcpy(h_out, d_out, sizeof(uint32_t) * SIZE, cudaMemcpyDeviceToHost);  

    for (int i = 0; i < SIZE; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl; */



    


    return 0;

}
