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
    typename P::UintType* output         // renamed from d_hist_transposed
) {
    static_assert(is_params<P>::value, "P must be an instance of Params");
    const uint32_t T = P::T;

    const uint32_t height = P::H;
    const uint32_t width = P::GRID_SIZE;
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
    uint32_t N,
    uint32_t bit_pos
) {
    Histo<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );

    transpose_kernel<P>(
        d_hist,
        d_hist_transposed
    );

    const int total_elements = P::H * P::GRID_SIZE;
    uint32_t* d_tmp;
    cudaMalloc((void**) &d_tmp, sizeof(uint32_t) * P::BLOCK_SIZE);
    
    // we do inclusive scan here
    scan3rdKernel<Add<uint32_t>, P::Q> <<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_hist_scanned,     // output: scanned histogram
        d_hist,  // input
        d_tmp,              // temporary storage
        total_elements, // hist size
        P::GRID_SIZE          // number of segments (one per bin)
    );
    cudaFree(d_tmp);

    // transpose back to original histogram
    transpose_kernel<P>(
        d_hist_scanned,
        d_hist_transposed // we write back to the transposed histogram
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
    uint32_t* d_hist;
    uint32_t* d_hist_transposed;
    uint32_t* d_hist_scanned;
    cudaMalloc((void**) &d_hist, sizeof(uint32_t) * P::BINS * P::BLOCK_SIZE);
    cudaMalloc((void**) &d_hist_transposed, sizeof(uint32_t) * P::BINS * P::BLOCK_SIZE);
    cudaMalloc((void**) &d_hist_scanned, sizeof(uint32_t) * P::BINS * P::BLOCK_SIZE);

    int bit_pos = 0;
    while (bit_pos < P::lgH) {
        CountSort<P>(
            d_in, 
            d_out, 
            d_hist, 
            d_hist_transposed, 
            d_hist_scanned, 
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
        cudaMemset(d_hist, 0, sizeof(uint32_t) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_hist_transposed, 0, sizeof(uint32_t) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_hist_scanned, 0, sizeof(uint32_t) * P::BINS * P::GRID_SIZE);
    }
}

#endif
