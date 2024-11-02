#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "constants.cuh"
#include <iostream>
#include <cstdint>
#include "tests/test_transpose_ker.cu"
#include "tests/test_radix_sort_ker.cu"
#include "tests/test_histo_ker.cu"
#include "tests/test_glb_to_reg.cu"
#include "tests/test_two_way_partition.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>


int main() {
    initHwd();

    // setup params

    const uint32_t input_size = 10000;
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
        GRID_SIZE, 
        BLOCK_SIZE, 
        T
    >;

    //TestTwoWayPartition<P>();
    
    //test_verify_transpose<P>(input_size);
    //test_call_rank_permute_ker<P>(input_size);

    //test_glb_to_reg_ker<P>(input_size);
    //test_count_sort<P>(input_size);
    //printf("CountSort done\n");

    test_histo_ker<P>(input_size);
    //test_radix_sort_ker<P>(input_size);

}
