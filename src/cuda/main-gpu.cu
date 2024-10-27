#include <iostream>
#include <sys/time.h>
#include "cub_kernel.cuh"
#include "helper.h"
using namespace std;

#define GPU_RUNS    50
#define ERR          0.000005


template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__host__ void runCub(
    T* d_in, 
    T* d_out, 
    int size
) {

    // Calculate grid size
    int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    int grid_size = (size + items_per_block - 1) / items_per_block;

    // Dry run
    CUBSortKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // Benchmark
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < GPU_RUNS; i++) {
        CUBSortKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(d_in, d_out, size);
    }
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / GPU_RUNS;

    printf("CUB Block Sort Kernel runs in: %lu microsecs\n", elapsed);

    // Calculate and print bandwidth and latency
    double gigabytes = (double)(size * sizeof(T)) / (1024 * 1024 * 1024);
    double seconds = elapsed / 1e6;
    double bandwidth = gigabytes / seconds;
    printf("GB processed: %.2f\n", gigabytes);
    printf("Bandwidth: %.2f GB/sec\n", bandwidth);
    printf("Latency: %.2f microsecs\n", elapsed);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runRadixSort(
    T* d_in, 
    T* d_out, 
    int size
) {
    const int B = 256;
    const int Q = 22;
    const int lgH = 4;
    const int H = pow(2, lgH);
    
    int blocksAmount = 1;
    
    uint32_t* histogram;
    cudaSucceeded(cudaMalloc((void**) &histogram, blocksAmount * H * sizeof(uint32_t)));
    
    makeHistogram<<<numBlocks, threadsPerBlock>>>(d_in, histogram, H, Q, B);
    
    uint32_t* transposedMatrix;
    cudaSucceeded(cudaMalloc((void**) &histogram, blocksAmount * H * sizeof(uint32_t)));
    transposeKernel<<<numBlocks, threadsPerBlock>>>(histogram, transposedMatrix, H, Q, B);
}


template<class T, int TL, int REG>
void runAll ( int size ) {

    srand(2006);
 
    // 1. allocate host memory for the two arrays
    unsigned long long mem_size = sizeof(T) * size;
    T* h_in = (T*) malloc(mem_size);
    T* h_out = (T*) malloc(mem_size);
 
    // 2. allocate device memory
    T* d_in;
    T* d_out;
    cudaMalloc((void**) &d_in, mem_size);
    cudaMalloc((void**) &d_out, mem_size);
 
    // 3. initialize host memory
    randomInit<T>(h_in, size);
    
    // 4. copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, mem_size, cudaMemcpyHostToDevice);


    printf("Size is: %d\n", size);

    runCub<T, TL, REG>( d_in, d_out, size );

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s size\n", argv[0]);
        exit(1);
    }
    const int size = atoi(argv[1]);

    printf("Running GPU-Parallel Versions (Cuda) of MMM\n");

    runAll<float, 256, 4> ( size );
}



