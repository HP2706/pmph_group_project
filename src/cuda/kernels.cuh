#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"
#include "helper_kernels/utils.cuh"


#define lgH 8 // bits used for each counting sort step
#define Q 22 // elms processed per thread
#define BINS (1 << lgH) // number of bins in the histogram


template <class ElTp> 
__global__ void RadixSortKer(ElTp* d_in, ElTp* d_out, int size) {
    
}


/// the code for the counting sort subroutine
template <class ElTp, int GRID_SIZE, int BLOCK_SIZE>
void CountSort(
    const ElTp* d_in, 
    ElTp* d_out, 
    uint32_t* d_hist, // we assume we are given a histogram array that is uninitialized
    uint32_t* d_hist_transposed, // we assume we are given a histogram array that is uninitialized
    uint32_t N, // number of elements in the array
    uint32_t bit_pos
) {

    
    /*
    
    1. Compute the histogram of the input array

    2. we now should have a 2d array of num blocks x num bins 

    this matrix which was in row major order where each row is the histogram of a block 
    we then need to transpose this and probably do two scans, one segmented scan to
    sum up the bins across blocks and then a prefix sum to get the starting index of each block in the sorted array
    
    */


    HistoKernel1<ElTp, lgH, Q><<<GRID_SIZE, BLOCK_SIZE>>>(
        d_in,
        d_hist, 
        N, 
        bit_pos
    );

    // we transpose the histogram matrix in chunks of 16x16
    const int T = 16;

    // Launch transpose kernel
    coalescedTransposeKer<ElTp, T><<<GRID_SIZE, BLOCK_SIZE>>>(
        d_hist,
        d_hist_transposed,
        BINS,
        N
    );
}

#endif
