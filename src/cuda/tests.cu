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


void run_tests() {
    initHwd();

    // setup params

    const uint32_t input_size = 100000;
    const uint32_t Q = 22; // 22
    const uint32_t lgH = 8;
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t T = 32;
    const uint32_t GRID_SIZE = (input_size + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);

    printf("total number of threads used: %u\n", GRID_SIZE * BLOCK_SIZE);
    printf("QB: %u\n", Q * BLOCK_SIZE);
    printf("grid size : %u\n", GRID_SIZE);

    using P = Params<
        uint32_t, 
        uint32_t, 
        Q, 
        lgH, 
        BLOCK_SIZE, 
        T
    >;

    testTransposeKer<P>(input_size, GRID_SIZE);
    testHistoKer<P>(input_size, GRID_SIZE);
    testRadixSortKer<P>(input_size, GRID_SIZE);
}

int main() {
    run_tests();
    return 0;
}