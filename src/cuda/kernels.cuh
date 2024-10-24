#ifndef kernels_h
#define kernels_h
#include "helper_kernels/histogram.cuh"
#include "helper_kernels/prefix_sum.cuh"

#define RADIX_BITS 2

template <class ElTp> 
__global__ void RadixSortKer(ElTp* d_in, ElTp* d_out, int size) {
    
}

template <class ElTp>
__global__ void CountSortKer(ElTp* d_in, ElTp* d_out, int size, const int LLC) {
    multiStepHisto(d_in, d_out, size, RADIX_BITS, 1024, LLC);
    scanExc(d_out, d_out, size, 32);
}

#endif