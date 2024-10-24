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
    uint32_t** h_inp_inds, 
    uint32_t** d_inp_inds, 
    UInt** d_hist, 
    UInt** h_hist,
    uint32_t num_bins,
    uint32_t SIZE
) {
    *h_in = (UInt*) malloc(sizeof(UInt) * SIZE);
    *h_hist = (UInt*) malloc(sizeof(UInt) * num_bins);
    *h_inp_inds = (uint32_t*) malloc(sizeof(uint32_t) * SIZE);
 
    // 2. allocate device memory
    cudaMalloc((uint32_t**) d_inp_inds, sizeof(uint32_t) * SIZE);
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((UInt**) d_hist, sizeof(UInt) * num_bins);
 
    FILE* debug_file = fopen("debug.txt", "w");
    if (debug_file == NULL) {
        fprintf(stderr, "Failed to open debug.txt for writing\n");
        return;
    }


    fprintf(debug_file, "h_in before randomInit\n");
    for (int i = 0; i < SIZE; i++) {
        fprintf(debug_file, "h_in[%d]: %d\n", i, (*h_in)[i]);
    }
    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);
    // initialize inds array
    for (int i = 0; i < SIZE; i++) {
        (*h_inp_inds)[i] = i; // Corrected dereferencing
    }
    
    fprintf(debug_file, "h_in after randomInit\n");
    for (int i = 0; i < SIZE; i++) {
        fprintf(debug_file, "h_in[%d]: %d\n", i, (*h_in)[i]);
    }
    
    fclose(debug_file);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(UInt) * num_bins, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_inp_inds, *h_inp_inds, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice);
}


int main() {
    srand(2006);
 
    uint32_t num_bins = 1 << 8; // 2^8
    uint32_t SIZE = pow(10, 8);
    uint32_t* h_in;
    uint32_t* d_in;
    uint32_t* h_inp_inds;
    uint32_t* d_inp_inds;
    uint32_t* d_hist;
    uint32_t* h_hist; // Add this line to declare h_hist
    
    PrepareMemory<uint32_t>(
        &h_in, 
        &d_in, 
        &h_inp_inds, 
        &d_inp_inds, 
        &d_hist, 
        &h_hist, // Add this argument to the function call
        num_bins,
        SIZE
    );

    printf("Copying h_in to d_in\n");
    cudaMemcpy(d_in, h_in, sizeof(uint32_t) * SIZE, cudaMemcpyHostToDevice);

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

    // check it sums to the size
    uint32_t sum = 0;
    for (int i = 0; i < num_bins; i++) {
        sum += h_hist[i];
    }
    printf("sum: %d\n", sum);

    if (sum != SIZE) {
        printf("sum is not equal to SIZE\n");
        std::exit(1);
    }


    for (int i = 0; i < num_bins; i++) {
        printf("num bin %d: %d\n", i, h_hist[i]);
    }
    free(h_in);
    free(h_hist);
    cudaFree(d_in);
    cudaFree(d_hist);
}
