#include <iostream>
#include <sys/time.h>
#include "cub_kernel.cuh"
#include "kernels.cuh"
#include "helper.h"
#include <unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "constants.cuh"
#include <iostream>
#include <cstdint>
#include "tests/test_transpose_ker.cu"
#include "tests/test_radix_sort_ker.cu"
#include "tests/test_histo_ker.cu"
#include "helper_kernels/prefix_sum.cuh"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;
#define GPU_RUNS    50
#define ERR          0.000005


enum class SortImplementation {
    CUB,
    OUR_IMPL
};

// benchmarks and returns the time taken in microseconds
template<class P>
double runSort(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    int size,
    SortImplementation impl,
    int grid_size
) {
    uint32_t *d_histogram = nullptr;
    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    // Benchmark
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);


    // warmup
    if (impl == SortImplementation::CUB) {
        deviceRadixSortKernel<typename P::ElementType>(
            d_in,
            d_out,
            size
        );
    } else {
        RadixSortKer<P>(d_in, d_out, size);
    }

    printf("Done with warmup\n");


    for (int i = 0; i < GPU_RUNS; i++) {
        if (impl == SortImplementation::CUB) {
            deviceRadixSortKernel<typename P::ElementType>(
                d_in,
                d_out,
                size
            );
        } else {
            RadixSortKer<P>(d_in, d_out, size);
        }
    }

    cudaDeviceSynchronize();
    gpuAssert(cudaPeekAtLastError());

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / GPU_RUNS;
    printf("Done with runs\n");
    printf("%s runs in: %lu microsecs\n", 
        impl == SortImplementation::CUB ? "CUB Block Sort Kernel" : "Our Implementation",
        elapsed);

    // Calculate and print bandwidth and latency
    double gigabytes = (double)(size * sizeof(typename P::ElementType)) / (1024 * 1024 * 1024);
    double seconds = elapsed / 1e6;
    double bandwidth = gigabytes / seconds;
    printf("GB processed: %.2f\n", gigabytes);
    printf("Bandwidth: %.2f GB/sec\n", bandwidth);
    printf("Latency: %.2f microsecs\n", elapsed);


    return elapsed;
}

template<typename T>
void runWithType(
    uint32_t SIZE, 
    int MAX_VAL,
    uint32_t impl
) {

    // Run CUB implementation
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t T_val = 32;
    const uint32_t lgH = 8;
    const uint32_t Q = 22;

    using P = Params<
        T, 
        uint32_t, 
        Q, 
        lgH,
        BLOCK_SIZE, 
        T_val
    >;

    
    T* h_in;
    T* d_in;
    
    T* h_out;
    T* d_out;


    allocateAndInitialize<T>(
        &h_in, 
        &d_in, 
        SIZE,
        true,
        MAX_VAL
    );


    allocateAndInitialize<T>(
        &h_out, 
        &d_out, 
        SIZE,
        false,
        MAX_VAL
    );

    const int mem_size = sizeof(T) * SIZE;
    
    const int GRID_SIZE = (SIZE + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);
    const SortImplementation radix_impl = impl == 0 ? SortImplementation::CUB : SortImplementation::OUR_IMPL;
    double elapsed = runSort<P>(
        d_in, 
        d_out, 
        SIZE, 
        radix_impl,
        GRID_SIZE
    );
   
    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    
    printf("%f\n", elapsed);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////



 
int main(int argc, char* argv[]) {
    initHwd();
    if (argc != 5) {
        printf("Usage: %s data_type: [u64|u32|u16|u8] size impl: [1|0] max_val\n", argv[0]);
        exit(1);
    }
    
    const std::string input_type = argv[1];
    const int SIZE = atoi(argv[2]);
    const int impl = atoi(argv[3]);
    const int MAX_VAL = atoi(argv[4]);


    printf("Running with type: %s\n", input_type.c_str());
    if (input_type == "u64") {
        runWithType<uint64_t>(SIZE, MAX_VAL, impl);
    } else if (input_type == "u32") {
        runWithType<uint32_t>(SIZE, MAX_VAL, impl);
    } else if (input_type == "u16") {
        runWithType<uint16_t>(SIZE, MAX_VAL, impl);
    } else if (input_type == "u8") {
        runWithType<uint8_t>(SIZE, MAX_VAL, impl);
    } else {
        printf("Invalid data type: %s\n", input_type.c_str());
        exit(1);
    }

    return 0;
}



