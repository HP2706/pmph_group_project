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
#include "tests/test_scan_buckets.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>



void debug_scaninc() {
    const uint32_t input_size = 100;
    const uint32_t Q = 22;
    const uint32_t BLOCK_SIZE = 256;

    // allocate ptrs
    uint32_t* d_scanned;
    uint32_t* h_scanned;
    
    uint32_t* h_in;
    uint32_t* d_in;
    
    uint32_t* d_tmp;

    allocateAndInitialize<uint32_t, 100000>(
        &h_in,
        &d_in,
        input_size,
        true
    );
    
    allocateAndInitialize<uint32_t, 100000>(
        &h_scanned, 
        &d_scanned, 
        input_size,
        false
    );

    cudaMalloc(&d_tmp, sizeof(uint32_t) * 1024);


    printf("printing h_in\n");
    for (int i = 0; i < input_size; i++) {
        printf("h_in[%d] = %d\n", i, h_in[i]);
    }

    scanInc<Add<uint32_t>, Q>(
        BLOCK_SIZE,
        input_size,
        d_scanned,
        d_in,
        d_tmp
    );

    cudaMemcpy(h_scanned, d_scanned, sizeof(uint32_t) * input_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < input_size; i++) {
        printf("scanned[%d] = %d\n", i, h_scanned[i]);
    }

    free(h_in);
    free(h_scanned);
    cudaFree(d_in);
    cudaFree(d_scanned);
    cudaFree(d_tmp);

}


int main() {
    initHwd();

    // setup params

    const uint32_t input_size = 1000;
    const uint32_t Q = 22; // 22
    const uint32_t lgH = 8;
    const uint32_t BLOCK_SIZE = 16;
    const uint32_t TILE_SIZE = 32;


    
    const uint32_t NUM_BLOCKS = (input_size + (BLOCK_SIZE * Q - 1)) / (BLOCK_SIZE * Q);

    printf("total number of blocks: %u\n", NUM_BLOCKS);

    using P = Params<
        uint32_t, 
        uint32_t, 
        Q, 
        lgH, 
        NUM_BLOCKS, 
        BLOCK_SIZE, 
        TILE_SIZE
    >;

    //test_verify_transpose<P>(input_size);
    test_scan_buckets<P>(input_size);

    //TestTwoWayPartition<P>();
    //test_call_rank_permute_ker<P>(input_size);
    //test_glb_to_reg_ker<P>(input_size);
    //test_count_sort<P>(input_size);
    //printf("CountSort done\n");
    //test_histo_ker<P>(input_size);
    //test_radix_sort_ker<P>(input_size);

}
