#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"
#include "helper_kernels/rank_permute.cuh"

#define lgH 8 // bits used for each counting sort step
#define Q 22 // elms processed per thread
#define BINS (1 << lgH) // number of bins in the histogram


template <class ElTp> 
__global__ void RadixSortKer(ElTp* d_in, ElTp* d_out, int size) {
    

}



template<int T>
void transpose_kernel(
    uint32_t* input,          // renamed from d_hist
    uint32_t* output,         // renamed from d_hist_transposed
    const uint32_t height,    // renamed from bins
    const uint32_t width      // renamed from GRID_SIZE
) {

    // 1. setup block and grid parameters
    int  dimy = (height+T-1) / T; 
    int  dimx = (width +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);

    coalsTransposeKer<uint32_t, T> <<<grid, block>>>(
        input,
        output,
        height,    // This should be NUM_BINS for your case
        width      // This should be SIZE/NUM_BINS for your case
    );
}

/// the code for the counting sort subroutine
/// modify to __device__ later when incorporated in the radix sort kernel
template <class ElTp, int GRID_SIZE, int BLOCK_SIZE, int T>
void CountSort(
    const ElTp* d_in, 
    ElTp* d_out, 
    uint32_t* d_hist,
    uint32_t* d_hist_transposed,
    uint32_t* d_hist_scanned,
    uint32_t N,
    uint32_t bit_pos
) {
    // Step 1: Compute histogram
    // Each block processes N/GRID_SIZE elements and produces local histogram
    // Result: d_hist is [GRID_SIZE][BINS] in row-major order
    Histo1<ElTp, lgH, Q><<<GRID_SIZE, BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );

    // Step 2: Transpose histogram matrix for coalesced memory access
    // Input: d_hist [GRID_SIZE][BINS]
    // Output: d_hist_transposed [BINS][GRID_SIZE]
    
    // 1. setup block and grid parameters
    int  dimy = (BINS+T-1) / T; 
    int  dimx = (GRID_SIZE +T-1) / T;
    dim3 block(T, T, 1);
    dim3 grid (dimx, dimy, 1);


    transpose_kernel<16>(
        d_hist,
        d_hist_transposed,
        BINS,
        GRID_SIZE
    );

    const int total_elements = BINS * GRID_SIZE;
    uint32_t* d_tmp;
    cudaMalloc((void**) &d_tmp, sizeof(uint32_t) * BLOCK_SIZE);
    
    ScanExc<Add<uint32_t>, Q> <<<GRID_SIZE, BLOCK_SIZE>>>(
        d_hist_scanned,     // output: scanned histogram
        d_hist_transposed,  // input
        d_tmp,              // temporary storage
        total_elements,
        GRID_SIZE          // number of segments (one per bin)
    );
    cudaFree(d_tmp);

    RankPermuteKer<ElTp, Q, lgH><<<GRID_SIZE, BLOCK_SIZE>>>(
        d_hist_scanned,
        d_out,
        N
    ); 

}


 /* // Step 3: Perform exclusive scan on transposed histogram
    // Each row contains counts for one bin across all blocks


    // Step 4: Use scanned histogram to permute elements to final positions
    */



#endif
