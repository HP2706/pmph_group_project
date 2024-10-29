#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "constants.cuh"
#include "helper.h"
#include "constants.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/pbb_kernels.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "kernels.cuh"


void initArray(int32_t* inp_arr, const uint32_t N, const int R) {
    const uint32_t M = 2*R+1;
    for(uint32_t i=0; i<N; i++) {
        inp_arr[i] = (rand() % M) - R;
    }
}

/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 */
int bandwidthMemcpy( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t   N     // length of the input array
                   , int* d_in            // device input  of length N
                   , int* d_out           // device result of length N
) {
    // dry run to exercise the d_out allocation!
    const uint32_t num_blocks = (N + B - 1) / B;
    naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);

    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL);

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);

        CUDASSERT(cudaPeekAtLastError());

        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive Memcpy GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }

    return 0;
}

int scanIncAddI32( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                 , const size_t   N     // length of the input array
                 , int* h_in            // host input    of size: N * sizeof(int)
                 , int* d_in            // device input  of size: N * sizeof(ElTp)
                 , int* d_out           // device result of size: N * sizeof(int)
) {
    const size_t mem_size = N * sizeof(int);
    int* d_tmp;
    int* h_out = (int*)malloc(mem_size);
    int* h_ref = (int*)malloc(mem_size);
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(int));
    cudaMemset(d_out, 0, N*sizeof(int));

    // dry run to exercise d_tmp allocation
    scanInc< Add<int> > ( B, N, d_out, d_in, d_tmp );
    CUDASSERT(cudaPeekAtLastError());

    // time the GPU computation
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i<RUNS_GPU; i++) {
        scanInc<Add<int>> ( B, N, d_out, d_in, d_tmp );
    }
    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    CUDASSERT(cudaPeekAtLastError());

    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
    double gigaBytesPerSec = N  * (2*sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
    printf("Scan Inclusive AddI32 GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n"
          , elapsed, gigaBytesPerSec);


    { // sequential computation
        gettimeofday(&t_start, NULL);
        for(int i=0; i<RUNS_CPU; i++) {
            int acc = 0;
            for(uint32_t i=0; i<N; i++) {
                acc += h_in[i];
                h_ref[i] = acc;
            }
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_CPU;
        double gigaBytesPerSec = N * (sizeof(int) + sizeof(int)) * 1.0e-3f / elapsed;
        printf("Scan Inclusive AddI32 CPU Sequential runs in: %lu microsecs, GB/sec: %.2f\n"
              , elapsed, gigaBytesPerSec);
    }

    { // Validation
        CUDASSERT(cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost));
        for(uint32_t i = 0; i<N; i++) {
            if(h_out[i] != h_ref[i]) {
                printf("!!!INVALID!!!: Scan Inclusive AddI32 at index %d, dev-val: %d, host-val: %d\n"
                      , i, h_out[i], h_ref[i]);
                exit(1);
            }
        }
        printf("Scan Inclusive AddI32: VALID result!\n\n");
    }

    free(h_out);
    free(h_ref);
    cudaFree(d_tmp);

    return 0;
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
void test_verify_transpose()
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
    

    const uint32_t SIZE = 1000000;
    const uint32_t NUM_BINS = 1 << 8;
    const uint32_t BLOCK_SIZE = 1024;
    const uint32_t Q = 22;
    const uint32_t lgH = 8;
    
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
    
    using SortParams = Params<uint32_t, uint32_t, Q, lgH, grid_size, BLOCK_SIZE, 32, 16>;
    
    Histo<SortParams><<<SortParams::GRID_SIZE, SortParams::BLOCK_SIZE>>>(
        d_in,
        d_histogram, 
        SIZE,
        uint32_t(0)
    );

    cudaMemcpy(h_histogram, d_histogram, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    transpose_kernel<SortParams>(
        d_histogram,
        d_histogram_transposed
    );
    cudaMemcpy(h_histogram_transposed, d_histogram_transposed, sizeof(uint32_t) * hist_size, cudaMemcpyDeviceToHost);
    
    verifyTranspose(h_histogram, tranposedHistogramCPU, h_histogram_transposed, NUM_BINS, BLOCK_SIZE);

}

int main() {
   
    test_verify_transpose();

    initHwd();

    const uint32_t N = 100000000;
    const uint32_t B = 256;

    printf("Testing parallel basic blocks for input length: %d and CUDA-block size: %d\n\n\n", N, B);

    const size_t mem_size = N*sizeof(int);
    int* h_in    = (int*) malloc(mem_size);
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in ,   mem_size);
    cudaMalloc((void**)&d_out,   mem_size);

    initArray(h_in, N, 13);
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    CUDASSERT(cudaPeekAtLastError());

    // computing a "realistic/achievable" bandwidth figure
    bandwidthMemcpy(B, N, d_in, d_out);


    { // inclusive scan and segmented scan with int addition
        scanIncAddI32   (B, N, h_in, d_in, d_out);
    }


    // cleanup memory
    free(h_in);
    cudaFree(d_in );
    cudaFree(d_out);
}
