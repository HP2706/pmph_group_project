#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/rank_permute.cuh"
#include "helper.h"


template<class P>
void transpose_kernel(
    typename P::UintType* input,          // renamed from d_hist
    typename P::UintType* output,         // renamed from d_hist_transposed
    const uint32_t height,
    const uint32_t width
) {
    static_assert(is_params<P>::value, "P must be an instance of Params");
    const uint32_t T = P::T;

    // 1. setup block and grid parameters
    int  dimy = (height+T-1) / T; 
    int  dimx = (width +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);

    coalsTransposeKer<typename P::UintType, T> <<<grid, block>>>(
        input,
        output,
        height,    // This should be NUM_BINS for your case
        width      // This should be SIZE/NUM_BINS for your case
    );
}


template<class P>
void CountSort(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    typename P::UintType* d_hist,
    typename P::UintType* d_hist_transposed,
    typename P::UintType* d_hist_scanned,
    typename P::UintType* d_tmp,
    uint32_t N,
    uint32_t bit_pos
) {
    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    Histo<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );

    transpose_kernel<P>(
        d_hist,
        d_hist_transposed,
        P::H,
        P::GRID_SIZE
    );


    scanInc<Add<typename P::UintType>>(
        P::BLOCK_SIZE, // Block size
        P::HB,          // Histogram size
        d_hist_scanned,     // output: scanned histogram
        d_hist_transposed,
        d_tmp              // temporary storage
    );

    // we do inclusive scan here
    // transpose back to original histogram
    transpose_kernel<P>(
        d_hist_scanned,
        d_hist_transposed, // we write back to the transposed histogram
        P::GRID_SIZE,
        P::H
    );

    RankPermuteKer<P> <<<P::GRID_SIZE, P::BLOCK_SIZE>>> 
    (
        d_hist_transposed,
        bit_pos,
        N,
        d_in,
        d_out
    );
}

template<class P>
void RadixSortKer(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    int size
) {
    typename P::UintType* d_hist;
    typename P::UintType* d_hist_transposed;
    typename P::UintType* d_hist_scanned;
    cudaMalloc((void**) &d_hist, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_hist_transposed, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_hist_scanned, sizeof(typename P::UintType) * P::HB);


    const int total_elements = P::H * P::GRID_SIZE;
    typename P::UintType* d_tmp;
    cudaMalloc((void**) &d_tmp, sizeof(typename P::UintType) * P::BLOCK_SIZE);
    

    int bit_pos = 0;
    while (bit_pos < P::lgH) {
        CountSort<P>(
            d_in, 
            d_out, 
            d_hist, 
            d_hist_transposed, 
            d_hist_scanned, 
            d_tmp,
            size, 
            bit_pos
        );

        // increment the bit position
        bit_pos += P::lgH;

        // swap the input and output arrays
        typename P::ElementType* tmp = d_in;
        d_in = d_out;
        d_out = tmp;

        // zero out the histogram arrays
        cudaMemset(d_hist, 0, sizeof(P::UintType) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_hist_transposed, 0, sizeof(P::UintType) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_hist_scanned, 0, sizeof(P::UintType) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_tmp, 0, sizeof(P::UintType) * P::BLOCK_SIZE);
    }

    cudaFree(d_hist);
    cudaFree(d_hist_transposed);
    cudaFree(d_hist_scanned);
    cudaFree(d_tmp);

}

#endif
