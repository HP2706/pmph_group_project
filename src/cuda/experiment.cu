#include "kernels.cuh"
#include "helper.h"
#include "helper_kernels/prefix_sum.cuh"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
// Define the LLC size (Last Level Cache)
#define LLC 41943040 // number taken from assignment 3-4

// Define the size of your input
#define SIZE 1024 // Adjust this value as needed

int main() {
    srand(2006);
 
    uint32_t num_bins = 1 << 8; // 2^8

    uint8_t* h_in = (uint8_t*) malloc(sizeof(uint8_t) * SIZE);
    uint8_t* h_hist = (uint8_t*) malloc(sizeof(uint8_t) * num_bins);
    uint32_t* h_inp_inds = (uint32_t*) malloc(sizeof(uint32_t) * SIZE);
 
    // 2. allocate device memory
    uint32_t* d_inp_inds;
    uint32_t* d_in; // Change the type from uint8_t* to uint32_t*
    uint32_t* d_hist;
    cudaMalloc((uint32_t**) &d_inp_inds, sizeof(uint32_t) * SIZE);
    cudaMalloc((uint32_t**) &d_in, sizeof(uint32_t) * SIZE); // Update the cudaMalloc call
    cudaMalloc((uint32_t**) &d_hist, sizeof(uint32_t) * SIZE);
 
    // 3. initialize host memory
    randomInit<uint8_t>(h_in, SIZE);


    // initialize inds array
    for (int i = 0; i < SIZE; i++) {
        h_inp_inds[i] = i;
    }
    
    // 4. copy host memory to device
    cudaMemcpy(d_in, h_in, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp_inds, h_inp_inds, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice);

    printf("Starting multiStepGenericHisto\n");
    multiStepGenericHisto<uint32_t, 256>(
        d_inp_inds, 
        d_in, 
        d_hist, 
        SIZE, 
        num_bins, 
        LLC
    );


    cudaMemcpy(h_hist, d_hist, sizeof(uint32_t) * num_bins, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_bins; i++) {
        printf("num bin %d: %d\n", i, h_hist[i]);
    }
    free(h_in);
    free(h_hist);
    cudaFree(d_in);
    cudaFree(d_hist);
}
