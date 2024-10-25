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
    *h_hist = (uint32_t*) malloc(sizeof(uint32_t) * num_bins);
 
    // 2. allocate device memory
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((UInt**) d_hist, sizeof(uint32_t) * hist_size);


    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
}


int main() {
    srand(2006);
 
    const int N_BITS = 2;
    const unsigned int num_bins = 1 << N_BITS; // 2^8
    const unsigned int SIZE = pow(10, 8);
    
    // using uint2_t = UInt<2>;  // Remove or comment this line
    
    UInt<N_BITS>* h_in;              // Changed from uint2_t
    UInt<N_BITS>* d_in;              // Changed from uint2_t
    uint32_t* d_hist;
    uint32_t* h_hist; 
    const int BLOCK_SIZE = 256;

    const uint32_t hist_size = num_bins ;//* BLOCK_SIZE;

    PrepareMemory<UInt<N_BITS>, BLOCK_SIZE>(
        &h_in, 
        &d_in, 
        &d_hist, 
        &h_hist,
        num_bins,
        SIZE,
        hist_size
    );

    printf("Copying h_in to d_in\n");
    cudaMemcpy(d_in, h_in, sizeof(unsigned int) * SIZE, cudaMemcpyHostToDevice);

    printf("Starting multiStepGenericHisto\n");
    multiStepGenericHisto<UInt<N_BITS>, BLOCK_SIZE>(
        (const UInt<N_BITS>*)d_in,
        d_hist, 
        SIZE, 
        num_bins, 
        LLC
    );

    cudaMemcpy(h_hist, d_hist, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);

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
