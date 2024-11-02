#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/rank_permute.cuh"
#include "helper.h"
#include <cassert>


template <class P>
__host__ void CountSort(
    typename P::ElementType* d_in,
    typename P::ElementType* d_out,
    uint64_t* d_histogram,
    uint64_t* d_histogram_t,
    uint64_t* d_histogram_t_scanned,
    uint64_t* d_histogram_t_scanned_t,
    uint32_t N,
    uint32_t bit_offs
) {

    Histo<P><<<P::NUM_BLOCKS, P::BLOCK_SIZE>>>(
        d_in,
        d_histogram,
        N,
        bit_offs
    );

    uint64_t* h_histogram = (uint64_t*) malloc(sizeof(uint64_t) * P::H * P::NUM_BLOCKS);
    cudaMemcpy(h_histogram, d_histogram, sizeof(uint64_t) * P::H * P::NUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < P::H * P::NUM_BLOCKS; i++) {
        printf("h_histogram[%d] = %lu\n", i, h_histogram[i]);
    }

    scan_buckets<P>(
        d_histogram, 
        d_histogram_t, 
        d_histogram_t_scanned, 
        d_histogram_t_scanned_t
    );

    uint64_t* h_histogram_t_scanned_t = (uint64_t*) malloc(sizeof(uint64_t) * P::H * P::NUM_BLOCKS);
    cudaMemcpy(h_histogram_t_scanned_t, d_histogram_t_scanned_t, sizeof(uint64_t) * P::H * P::NUM_BLOCKS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < P::H * P::NUM_BLOCKS; i++) {
        printf("h_histogram_t_scanned_t[%d] = %lu\n", i, h_histogram_t_scanned_t[i]);
    }

    rank_permute_ker<P><<<P::NUM_BLOCKS, P::BLOCK_SIZE>>>(
        d_in,
        d_out, // the new array
        d_histogram,
        d_histogram_t_scanned_t,
        N,
        bit_offs
    );

}


#endif
