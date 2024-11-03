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
    const int grid_size
) {


    static_assert(is_params<P>::value, "P must be an instance of Params");

    // we use uint16_t for d_hist
    Histo<P, uint16_t><<<grid_size, P::BLOCK_SIZE>>>(
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
        grid_size,
        P::H
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
        grid_size*P::H,  // Histogram size
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
        P::H,
        grid_size
    );
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Second transpose kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }


    RankPermuteKer<P>
    <<<
        grid_size, 
        P::BLOCK_SIZE
    >>>
    (
        d_hist,
        d_hist_transposed_scanned_transposed,
        bit_offs,
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
__host__ void RadixSortKer(
    typename P::ElementType* d_in,
    typename P::ElementType* output,
    int size
) {
    // Allocate a separate buffer for intermediate results
    typename P::ElementType* d_buffer;
    cudaMalloc((void**) &d_buffer, sizeof(typename P::ElementType) * size);
    // Copy initial input to buffer
    cudaMemcpy(d_buffer, d_in, sizeof(typename P::ElementType) * size, cudaMemcpyDeviceToDevice);
    
    uint16_t* d_hist;
    uint16_t* d_hist_transposed;
    uint64_t* d_hist_transposed_scanned;
    uint64_t* d_hist_transposed_scanned_transposed;
    uint64_t* d_tmp;
    typename P::ElementType* tmp_swap;

    const int grid_size = (size + (P::BLOCK_SIZE * P::Q - 1)) / (P::BLOCK_SIZE * P::Q);

    cudaMalloc((void**) &tmp_swap, sizeof(typename P::ElementType) * size);
    cudaMalloc((void**) &d_hist, sizeof(uint16_t) * P::H*grid_size);
    cudaMalloc((void**) &d_hist_transposed, sizeof(uint16_t) * P::H*grid_size);
    cudaMalloc((void**) &d_hist_transposed_scanned, sizeof(uint64_t) * P::H*grid_size);
    cudaMalloc((void**) &d_hist_transposed_scanned_transposed, sizeof(uint64_t) * P::H*grid_size);
    cudaMalloc((void**) &d_tmp, sizeof(uint64_t) * P::BLOCK_SIZE);
    
    // check divisibility by lgH
    assert((sizeof(typename P::ElementType)*8) % P::lgH == 0);
    
    int n_iter = (sizeof(typename P::ElementType)*8) / P::lgH;

    int bit_offs = 0;

    typename P::ElementType* current_in = d_buffer;
    typename P::ElementType* current_out = output;

    for (int i = 0; i < n_iter; i++) {
        CountSort<P>(
            current_in,
            current_out,
            d_hist, 
            d_hist_transposed, 
            d_hist_transposed_scanned_transposed,
            d_hist_transposed_scanned, 
            d_tmp, 
            size, 
            bit_offs,
            grid_size
        );

        bit_offs += P::lgH;
        
        // Swap the buffers
        typename P::ElementType* temp = current_in;
        current_in = current_out;
        current_out = temp;

        cudaDeviceSynchronize();
        gpuAssert(cudaGetLastError());
    }

    // If we ended with the result in the wrong buffer, copy it to output
    if (current_in != output) {
        cudaMemcpy(output, current_in, sizeof(typename P::ElementType) * size, cudaMemcpyDeviceToDevice);
    }

    // Clean up
    cudaFree(d_buffer);
    cudaFree(tmp_swap);
    cudaFree(d_hist);
    cudaFree(d_hist_transposed);
    cudaFree(d_hist_transposed_scanned);
    cudaFree(d_hist_transposed_scanned_transposed);
    cudaFree(d_tmp);

}

#endif
