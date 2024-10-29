#include <iostream>
#include <sys/time.h>
#include "cub_kernel.cuh"
#include "kernels.cuh"
#include "helper.h"
using namespace std;

#define GPU_RUNS    50
#define ERR          0.000005
#define Q            22

template<typename T, int GRID_SIZE, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__host__ void runCub(
    T* d_in, 
    T* d_out, 
    int size
) {

    // Dry run
    CUBSortKernel<
        T, 
        BLOCK_SIZE, 
        ITEMS_PER_THREAD
    >
    <<<GRID_SIZE, BLOCK_SIZE>>>
    (
        d_in, 
        d_out, 
        size
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // Benchmark
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < GPU_RUNS; i++) {
        CUBSortKernel<T, BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<GRID_SIZE, BLOCK_SIZE>>>(
                d_in, 
                d_out, 
                size
        );
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
}


// ... existing code ...

template<typename T, int GRID_SIZE, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__host__ void runOurImpl(
    T* d_in, 
    T* d_out, 
    int size
) {
    // Calculate grid size based on input size and elements per thread
    const uint32_t NUM_BINS = 1 << 8;
    const uint32_t hist_size = NUM_BINS * GRID_SIZE;

    // Allocate histogram memory
    uint32_t* d_histogram;
    uint32_t* d_histogram_transposed;
    uint32_t* d_hist_out;
    
    cudaMalloc((uint32_t**)&d_histogram, sizeof(uint32_t) * hist_size);
    cudaMalloc((uint32_t**)&d_histogram_transposed, sizeof(uint32_t) * hist_size);
    cudaMalloc((uint32_t**)&d_hist_out, sizeof(uint32_t) * hist_size);
    
    cudaMemset(d_histogram, 0, sizeof(uint32_t) * hist_size);
    cudaMemset(d_histogram_transposed, 0, sizeof(uint32_t) * hist_size);
    cudaMemset(d_hist_out, 0, sizeof(uint32_t) * hist_size);

    // Dry run
    CountSort<T, GRID_SIZE, BLOCK_SIZE, 16>(
        d_in,
        d_out,
        d_histogram,
        d_histogram_transposed,
        d_hist_out,
        uint32_t(size),
        uint32_t(0)
    );
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // Benchmark
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < GPU_RUNS; i++) {
        CountSort<T, GRID_SIZE, BLOCK_SIZE, 16>(
            d_in,
            d_out,
            d_histogram,
            d_histogram_transposed,
            d_hist_out,
            uint32_t(size),
            uint32_t(0)
        );
    }

    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / GPU_RUNS;

    printf("Our Implementation runs in: %lu microsecs\n", elapsed);

    // Calculate and print bandwidth and latency
    double gigabytes = (double)(size * sizeof(T)) / (1024 * 1024 * 1024);
    double seconds = elapsed / 1e6;
    double bandwidth = gigabytes / seconds;
    printf("GB processed: %.2f\n", gigabytes);
    printf("Bandwidth: %.2f GB/sec\n", bandwidth);
    printf("Latency: %.2f microsecs\n", elapsed);

    // Clean up
   {
        cudaFree(d_histogram);
        cudaFree(d_histogram_transposed);
        cudaFree(d_hist_out);
    }
}


template<class T, int TL, int NELMS>
void runAll ( int SIZE ) {

    srand(2006);

    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);
 
    // 1. allocate host memory for the two arrays
    unsigned long long mem_size = sizeof(T) * SIZE;
    T* h_in = (T*) malloc(mem_size);
    T* h_out = (T*) malloc(mem_size);
 
    // 2. allocate device memory
    T* d_in;
    T* d_out;
    cudaMalloc((void**) &d_in, mem_size);
    cudaMalloc((void**) &d_out, mem_size);
 
    // 3. initialize host memory
    randomInit<T>(h_in, SIZE, 100);
    
    // 4. copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, mem_size, cudaMemcpyHostToDevice);


    printf("Size is: %d\n", SIZE);

    runCub<T, GRID_SIZE, BLOCK_SIZE, NELMS>( d_in, d_out, SIZE );

    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);



    // we allocate the new memory for our implementation
    T* d_out_2;
    cudaMalloc((void**)&d_out_2, mem_size);
    cudaMemset(d_out_2, 0, mem_size);

    
    // clear up memory
    cudaFree(d_out);

    // the last generic parameter means something different here
    runOurImpl<T, GRID_SIZE, BLOCK_SIZE, 32>( 
        d_in, 
        d_out_2, 
        SIZE
    );


    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out_2);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s size\n", argv[0]);
        exit(1);
    }
    const int SIZE = atoi(argv[1]);

    printf("Running GPU-Parallel Versions (Cuda) of MMM\n");

    runAll<uint8_t, 256, 22> ( SIZE );



    /* const uint32_t BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);
 
    uint32_t mem_size = sizeof(uint8_t) * SIZE;
    uint8_t* d_in;
    uint8_t* d_out_2;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out_2, mem_size);
    // the last generic parameter means something different here


    runOurImpl<uint8_t, GRID_SIZE, BLOCK_SIZE, 32>( 
        d_in, 
        d_out_2, 
        SIZE
    ); */
}



