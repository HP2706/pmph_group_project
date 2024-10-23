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
    BlockSortKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // Benchmark
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < GPU_RUNS; i++) {
        BlockSortKernel<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(d_in, d_out, size);
    }
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / GPU_RUNS;

    printf("CUB Block Sort Kernel runs in: %lu microsecs\n", elapsed);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
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

    runAll<float, 16, 5> ( size );
}



