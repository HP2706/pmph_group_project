#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/rank_permute.cuh"
#include "helper.h"
#include <cassert>

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
__host__ void CountSort(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    uint16_t* d_hist,
    uint16_t* d_hist_transposed,
    uint64_t* d_hist_transposed_scanned_transposed,
    uint64_t* d_hist_transposed_scanned,
    uint64_t* d_tmp,
    uint32_t N,
    uint32_t bit_offs,
    typename P::ElementType* h_out_debug = nullptr
) {

    // TODO
    // we should use different uint types for the histogram and the scanned histogram
    // for a faster kernel

    // check that P is an instance of Params
    static_assert(is_params<P>::value, "P must be an instance of Params");

    // we use uint16_t for d_hist
    Histo<P, uint16_t><<<P::GRID_SIZE, P::BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_offs
    );

    cudaError_t err;

    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("Histogram kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    tiled_transpose_kernel<uint16_t, P::T>(
        d_hist,
        d_hist_transposed,
        P::H,
        P::GRID_SIZE
    );
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("First transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // in the future use something like FusedAddCast<uint16_t, uint32_t>
    scanInc<FusedAddCast<uint16_t, uint64_t>>(
        P::BLOCK_SIZE, // Block size
        P::GRID_SIZE*P::H,  // Histogram size
        d_hist_transposed_scanned,     // output: scanned histogram
        d_hist_transposed,
        d_tmp              // temporary storage
    );
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Scan kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // we do inclusive scan here
    // transpose back to original histogram
    tiled_transpose_kernel<uint64_t, P::T>(
        d_hist_transposed_scanned,
        d_hist_transposed_scanned_transposed,
        P::GRID_SIZE,
        P::H
    );
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Second transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }


    int shm_size = 
        (P::Q*P::BLOCK_SIZE*sizeof(typename P::ElementType) 
        + P::H*sizeof(uint64_t)  // global histogram
        + P::BLOCK_SIZE*sizeof(uint16_t)); // local histogram

    RankPermuteKer<P>
    <<<
        P::GRID_SIZE, 
        P::BLOCK_SIZE
        //shm_size
    >>>
    (
        d_hist,
        d_hist_transposed_scanned_transposed,
        bit_offs,
        N,
        d_in,
        d_out
    );

    if (h_out_debug != nullptr) {
        int debug_bit_offs = bit_offs + P::lgH - 1;

        cudaMemcpy(h_out_debug, d_out, sizeof(typename P::ElementType) * N, cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++) {
            printf("h_out_debug[%d]: %u\n", i, h_out_debug[i]);
        }
        /* printf("debugging h_out_debug after rank and permute\n bit_offs: %d\n", debug_bit_offs);
        debugPartitionCorrectness<P>(
            h_out_debug, 
            min(N, P::BLOCK_SIZE * P::Q),
            debug_bit_offs,
            true
        ); */
    }

   
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Rank and permute kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

template<class P>
__host__ void RadixSortKer(
    typename P::ElementType* d_in, 
    typename P::ElementType* d_out, 
    typename P::ElementType* d_out_debug,
    int size
) {

    uint16_t* d_hist;
    uint16_t* d_hist_transposed;
    uint64_t* d_hist_transposed_scanned;
    uint64_t* d_hist_transposed_scanned_transposed;
    uint64_t* d_tmp;

    cudaMalloc((void**) &d_hist, sizeof(uint16_t) * P::H*P::GRID_SIZE);
    cudaMalloc((void**) &d_hist_transposed, sizeof(uint16_t) * P::H*P::GRID_SIZE);
    cudaMalloc((void**) &d_hist_transposed_scanned, sizeof(uint64_t) * P::H*P::GRID_SIZE);
    cudaMalloc((void**) &d_hist_transposed_scanned_transposed, sizeof(uint64_t) * P::H*P::GRID_SIZE);
    cudaMalloc((void**) &d_tmp, sizeof(uint64_t) * P::BLOCK_SIZE);
    
    // STRICTLY FOR DEBUGGING
    typename P::ElementType* h_out_debug;
    h_out_debug = (typename P::ElementType*) malloc(sizeof(typename P::ElementType) * size);
    
    // check divisibility by lgH
    assert((sizeof(typename P::ElementType)*8) % P::lgH == 0);
    
    int n_iter = (sizeof(typename P::ElementType)*8) / P::lgH;

    int bit_offs = 0;
    printf("n_iter: %d\n", n_iter);


    for (int i = 0; i < n_iter; i++) {
        CountSort<P>(
            d_in, 
            d_out, 
            d_hist, 
            d_hist_transposed, 
            d_hist_transposed_scanned_transposed,
            d_hist_transposed_scanned, 
            d_tmp, 
            size, 
            bit_offs,
            h_out_debug
        );

        // increment the bit position
        bit_offs += P::lgH;
        
        if (i < n_iter - 1) {  // Only swap for all but the last iteration
            printf("swapping d_in and d_out at bit_offs: %d\n", bit_offs);
            typename P::ElementType* tmp = d_in;
            d_in = d_out;
            d_out = tmp;
        }

        // zero out the histogram arrays
        cudaMemset(d_hist, 0, sizeof(uint16_t) * P::H * P::GRID_SIZE);
        cudaMemset(d_hist_transposed, 0, sizeof(uint16_t) * P::H * P::GRID_SIZE);
        cudaMemset(d_hist_transposed_scanned, 0, sizeof(uint64_t) * P::H * P::GRID_SIZE);
        cudaMemset(d_hist_transposed_scanned_transposed, 0, sizeof(uint64_t) * P::H * P::GRID_SIZE);
        cudaMemset(d_tmp, 0, sizeof(uint64_t) * P::BLOCK_SIZE);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("count sort kernel failed at iteration %d: %s\n", i, cudaGetErrorString(err));
            return;
        }

    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Radix sort kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out_debug, d_out, sizeof(typename P::ElementType) * size, cudaMemcpyDeviceToHost);
    /* printf("printing the final output array at radix sort end \n");
    for (int i = 0; i < size; i++) {
        printf("line 270 h_out_debug[%d]: %u\n", i, h_out_debug[i]);
    } */

    cudaMemcpy(d_out_debug, d_out, sizeof(typename P::ElementType) * size, cudaMemcpyDeviceToDevice);

    cudaFree(d_hist);
    cudaFree(d_hist_transposed);
    cudaFree(d_hist_transposed_scanned);
    cudaFree(d_hist_transposed_scanned_transposed);
    cudaFree(d_tmp);

}

#endif
