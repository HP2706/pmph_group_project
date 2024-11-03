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

using namespace std;

#define GPU_RUNS    50
#define ERR          0.000005




enum class SortImplementation {
    CUB,
    OUR_IMPL
};

struct ArrayData {
    void* data;
    size_t size;
    std::string type;
};

std::unordered_map<std::string, std::string> TYPE_MAP = {
    {"u32", "uint32_t"},
    {"u64", "uint64_t"},
    {"u16", "uint16_t"},
    {"u8", "uint8_t"}
};

template<typename T>
ArrayData readDataFile(const char* filename) {
    FILE* input_file = fopen(filename, "r");
    if (!input_file) {
        throw std::runtime_error("Failed to open input file");
    }

    // Read header
    char b[2], type[10];
    if (fscanf(input_file, "%1s %s", b, type) != 2) {
        fclose(input_file);
        throw std::runtime_error("Failed to read header");
    }

    // Verify header format
    if (b[0] != 'b') {
        fclose(input_file);
        throw std::runtime_error("Invalid file format: should start with 'b'");
    }

    // Count numbers in file
    size_t size = 0;
    T value;
    while (fscanf(input_file, "%u", &value) == 1) {
        size++;
    }

    // Reset file position after header
    fseek(input_file, 0, SEEK_SET);
    fscanf(input_file, "%1s %s", b, type);  // Skip header

    // Allocate and read data
    T* data = (T*)malloc(size * sizeof(T));
    if (!data) {
        fclose(input_file);
        throw std::runtime_error("Failed to allocate memory");
    }

    for (size_t i = 0; i < size; i++) {
        if (fscanf(input_file, "%u", &data[i]) != 1) {
            free(data);
            fclose(input_file);
            throw std::runtime_error("Failed to read data");
        }
    }

    fclose(input_file);
    return ArrayData{(void*)data, size, std::string(type)};
}

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
    uint32_t *d_histogram_transposed = nullptr;
    uint32_t *d_hist_out = nullptr;

    if (impl == SortImplementation::OUR_IMPL) {
        // Initialize histogram memory for our implementation
        const uint32_t NUM_BINS = 1 << 8;
        const uint32_t hist_size = NUM_BINS * grid_size;
        
        cudaMalloc((uint32_t**)&d_histogram, sizeof(uint32_t) * hist_size);
        cudaMalloc((uint32_t**)&d_histogram_transposed, sizeof(uint32_t) * hist_size);
        cudaMalloc((uint32_t**)&d_hist_out, sizeof(uint32_t) * hist_size);
        
        cudaMemset(d_histogram, 0, sizeof(uint32_t) * hist_size);
        cudaMemset(d_histogram_transposed, 0, sizeof(uint32_t) * hist_size);
        cudaMemset(d_hist_out, 0, sizeof(uint32_t) * hist_size);
    }

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


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s input.data [u32|u64|u16|u8] [size]\n", argv[0]);
        exit(1);
    }

    std::string input_type = argv[2];
    const int SIZE = atoi(argv[3]);
    ArrayData array_data;
    
    if (TYPE_MAP.find(input_type) == TYPE_MAP.end()) {
        printf("Invalid data type: %s\n", input_type.c_str());
        exit(1);
    }

    if (input_type == "u32") {
        array_data = readDataFile<uint32_t>(argv[1]);
    } else if (input_type == "u64") {
        array_data = readDataFile<uint64_t>(argv[1]);
    } else if (input_type == "u16") {
        array_data = readDataFile<uint16_t>(argv[1]);
    } else if (input_type == "u8") {
        array_data = readDataFile<uint8_t>(argv[1]);
    }

    printf("Running GPU-Parallel Versions (Cuda) of Radix Sort\n");
    printf("Input size: %ld\n", array_data.size);

    // Allocate device memory
    uint32_t mem_size = sizeof(array_data.type) * array_data.size;
    uint32_t* d_in;
    uint32_t* d_out;
    cudaMalloc((void**)&d_in, mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // Copy input data to device
    cudaMemcpy(d_in, array_data.data, mem_size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, mem_size);

    // Run CUB implementation
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t T = 32;
    const uint32_t lgH = 8;
    const uint32_t Q = 22;

    using P = Params<
        uint32_t, 
        uint32_t, 
        Q, 
        lgH,
        BLOCK_SIZE, 
        T
    >;

    const int GRID_SIZE = (SIZE + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);

    double elapsed = runSort<P>(
        d_in, 
        d_out, 
        SIZE, 
        SortImplementation::CUB,
        GRID_SIZE
    );

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(array_data.data);
    
    printf("%f\n", elapsed);
    return 0;
}



