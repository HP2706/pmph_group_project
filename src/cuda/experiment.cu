#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "constants.cuh"
#include <iostream>
#include <cstdint>
#include "tests/test_transpose_ker.cu"
#include "tests/test_scan_inc.cu"
#include "tests/test_count_sort.cu"
#include "tests/test_rank_permute.cu"
#include "tests/test_histo_ker.cu"
#include <cuda_runtime.h>

int main() {
    initHwd();

    // setup params

    const uint32_t input_size = 10000000;
    const uint32_t Q = 22;
    const uint32_t lgH = 8;
    const uint32_t BLOCK_SIZE = 256;
    const uint32_t T = 6;
    const uint32_t ELEMS_PER_THREAD_SCAN = 22;
    const uint32_t GRID_SIZE = (input_size + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);

    printf("grid size : %u\n", GRID_SIZE);

    using P = Params<
        uint8_t, 
        uint32_t, 
        Q, 
        lgH, 
        GRID_SIZE, 
        BLOCK_SIZE, 
        T, 
        ELEMS_PER_THREAD_SCAN
    >;


    //test_verify_transpose<P>(input_size);
    //test_call_rank_permute_ker<P>(input_size);

    test_count_sort<P>(input_size);
    //printf("CountSort done\n");

    //test_histo_ker<P>(input_size);

}
