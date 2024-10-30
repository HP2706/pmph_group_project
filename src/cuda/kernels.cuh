#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/rank_permute.cuh"
#include "helper.h"


template<class T, int TILE_SIZE>
void tiled_transpose_kernel(
    T* input,          // renamed from d_hist
    T* output,         // renamed from d_hist_transposed
    const uint32_t height,
    const uint32_t width
) {
    // 1. setup block and grid parameters
    int  dimy = (height+TILE_SIZE-1) / TILE_SIZE; 
    int  dimx = (width +TILE_SIZE-1) / TILE_SIZE;
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid (dimx, dimy, 1);

    coalsTransposeKer<T, TILE_SIZE> <<<grid, block>>>(
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
    typename P::UintType* d_hist_transposed_scanned_transposed, // double transposed histogram
    typename P::UintType* d_hist_transposed_scanned,
    typename P::UintType* d_tmp,
    uint32_t N,
    uint32_t bit_pos
) {

    // TODO
    // we should use different uint types for the histogram and the scanned histogram
    // for a faster kernel

    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    Histo<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Histogram kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    tiled_transpose_kernel<typename P::UintType, P::T>(
        d_hist,
        d_hist_transposed,
        P::H,
        P::GRID_SIZE
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("First transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // in the future use something like FusedAddCast<uint16_t, uint32_t>
    scanInc<Add<typename P::UintType>>(
        P::BLOCK_SIZE, // Block size
        P::HB,          // Histogram size
        d_hist_transposed_scanned,     // output: scanned histogram
        d_hist_transposed,
        d_tmp              // temporary storage
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Scan kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // we do inclusive scan here
    // transpose back to original histogram
    tiled_transpose_kernel<typename P::UintType, P::T>(
        d_hist_transposed_scanned,
        d_hist_transposed_scanned_transposed,
        P::GRID_SIZE,
        P::H
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Second transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    RankPermuteKer<P><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_hist,
        d_hist_transposed_scanned_transposed,
        bit_pos,
        N,
        d_in,
        d_out
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Rank and permute kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

template<class P>
void RadixSortKer(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    int size
) {
    typename P::UintType* d_hist;
    typename P::UintType* d_hist_transposed;
    typename P::UintType* d_hist_transposed_scanned;
    typename P::UintType* d_hist_transposed_scanned_transposed;
    typename P::UintType* d_tmp;

    cudaMalloc((void**) &d_hist, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_hist_transposed, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_hist_transposed_scanned, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_hist_transposed_scanned_transposed, sizeof(typename P::UintType) * P::HB);
    cudaMalloc((void**) &d_tmp, sizeof(typename P::UintType) * P::BLOCK_SIZE);
    

    int bit_pos = 0;
    while (bit_pos < P::lgH) {
        CountSort<P>(
            d_in, 
            d_out, 
            d_hist, 
            d_hist_transposed, 
            d_hist_transposed_scanned_transposed,
            d_hist_transposed_scanned, 
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
        cudaMemset(d_hist_transposed_scanned, 0, sizeof(P::UintType) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_hist_transposed_scanned_transposed, 0, sizeof(P::UintType) * P::BINS * P::GRID_SIZE);
        cudaMemset(d_tmp, 0, sizeof(P::UintType) * P::BLOCK_SIZE);
    }

    cudaFree(d_hist);
    cudaFree(d_hist_transposed);
    cudaFree(d_hist_transposed_scanned);
    cudaFree(d_hist_transposed_scanned_transposed);
    cudaFree(d_tmp);

}

#endif
