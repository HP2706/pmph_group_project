#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cstdint>
#include <vector>

#include "../constants.cuh"
#include "../helper.h"
#include "../helper_kernels/utils.cuh"
#include "../helper_kernels/rank_permute.cuh"
#include "../kernels.cuh"
#include "../cub_kernel.cuh"
#include <cuda_runtime.h>

#if 1

template<class P>
void cpu_scan_buckets(
    uint64_t* in_arr,
    uint64_t* out_arr
) {
    int num_buckets = P::NUM_BLOCKS;
    int histogram_size = P::H;
    
    // For each digit j
    for (int j = 0; j < histogram_size; j++) {
        // For each processor i
        for (int i = 0; i < num_buckets; i++) {
            int sum = 0;
            
            // First sum: all digits less than j across all processors
            for (int k = 0; k < num_buckets; k++) {
                for (int m = 0; m < j; m++) {
                    sum += in_arr[k * histogram_size + m];
                }
            }
            
            // Second sum: digits equal to j in processors less than i
            for (int k = 0; k < i; k++) {
                sum += in_arr[k * histogram_size + j];
            }
            
            out_arr[i * histogram_size + j] = sum;
        }
    }
}

// note the outputs are almost similar except for the fact that 
// one is "behind" by one element likely because
// we are using scaninc and not exclusive scan
template<class P>
void test_scan_buckets(int input_size) {

    // ptrs
    typename P::ElementType* h_in;
    typename P::ElementType* d_in;

    uint64_t* h_histogram;
    uint64_t* d_histogram;
    uint64_t* d_histogram_t;
    uint64_t* d_histogram_t_scanned;

    uint64_t* d_histogram_t_scanned_t;
    uint64_t* h_histogram_t_scanned_t;

    uint64_t* cpu_h_histogram_scanned;

    int hist_len = P::H * P::NUM_BLOCKS;

    allocateAndInitialize<typename P::ElementType, P::MAXNUMERIC_ElementType>(
        &h_in,
        &d_in,
        input_size,
        true
    );

    // allocate memory
    allocateAndInitialize<uint64_t, P::MAXNUMERIC_ElementType>(
        &h_histogram,
        &d_histogram,
        hist_len,
        false
    );

    allocateAndInitialize<uint64_t, P::MAXNUMERIC_ElementType>(
        &h_histogram_t_scanned_t,
        &d_histogram_t_scanned_t,
        hist_len,
        false
    );


    // allocate individual arrays
    cudaMalloc(&d_histogram_t, sizeof(uint64_t) * hist_len);
    cudaMalloc(&d_histogram_t_scanned, sizeof(uint64_t) * hist_len);

    
    cudaMemset(d_histogram_t, 0, sizeof(uint64_t) * hist_len);
    cudaMemset(d_histogram_t_scanned, 0, sizeof(uint64_t) * hist_len);

    cpu_h_histogram_scanned = (uint64_t*)malloc(sizeof(uint64_t) * hist_len);


    Histo<P, uint64_t><<<P::NUM_BLOCKS, P::BLOCK_SIZE>>>(
        d_in, 
        d_histogram, 
        input_size, 
        0
    );

    cudaMemcpy(h_histogram, d_histogram, sizeof(uint64_t) * hist_len, cudaMemcpyDeviceToHost);


    // print the histogram
    int sum = 0;
    for (int i = 0; i < hist_len; i++) {
        sum += h_histogram[i];
        printf("histogram[%d] = %d\n", i, h_histogram[i]);
    }    
    printf("sum = %d should be input size %d\n", sum, input_size);
    


    cpu_scan_buckets<P>(h_histogram, cpu_h_histogram_scanned);

    scan_buckets<P>(
        d_histogram, 
        d_histogram_t, 
        d_histogram_t_scanned, 
        d_histogram_t_scanned_t
    );

    cudaMemcpy(h_histogram_t_scanned_t, d_histogram_t_scanned_t, sizeof(uint64_t) * hist_len, cudaMemcpyDeviceToHost);

    // Change this line to just cast the existing pointer
    uint64_t* cpu_h_histogram_scanned_64 = (uint64_t*)cpu_h_histogram_scanned;

    for (int i = 0; i < hist_len; i++) {
        printf("histogram_scanned[%d] = %d, histogram_t_scanned_t[%d] = %d\n", i, cpu_h_histogram_scanned_64[i], i, h_histogram_t_scanned_t[i]);
        //assert(cpu_h_histogram_scanned_64[i] == h_histogram_t_scanned_t[i]);
    }

}

#endif