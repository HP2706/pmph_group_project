#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "utils.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


/// the multistep kernel for the histogram
template<typename UInt>
__global__ void
multiStepGenericKernel ( const uint32_t* inp_inds  // Indices
                , const UInt* inp_vals     // Input values
                , volatile UInt* hist      // Histogram counts
                , const uint32_t N         // Number of input elements
                , const uint32_t LB        // Lower bound
                , const uint32_t UB        // Upper bound
) {
    // Ensure UInt is less than or equal to 64 bytes
    // this will not work for larger uints
    static_assert(sizeof(UInt) <= 64, "UInt must be 64 bytes or less");

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < N) {
        uint32_t ind = inp_inds[gid];
        if(ind < UB && ind >= LB) {
            UInt val = inp_vals[gid];
            atomicAdd((unsigned int*)&hist[ind], static_cast<unsigned int>(val)); // Cast val to unsigned int
        }
    }
}

// Corresponding multiStepHisto function
template<typename UInt, int B>
void multiStepGenericHisto ( const uint32_t* d_inp_inds
                    , const UInt* d_inp_vals
                    , UInt* d_hist
                    , const uint32_t N
                    , const uint32_t H // number of bins
                    , const uint32_t LLC
) {
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = (LLC_FRAC * LLC) / sizeof(UInt);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    cudaMemset(d_hist, 0, H * sizeof(UInt));
    for (uint32_t k = 0; k < num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k * CHUNK;
        uint32_t upp_bound = min((k + 1) * CHUNK, H);

        uint32_t grid = (N + B - 1) / B;
        multiStepGenericKernel<UInt><<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, low_bound, upp_bound);
    }
}

// for debugging
__global__ void
multiStepUnsignedIntKernel ( uint32_t* inp_inds
                , unsigned int*    inp_vals
                , volatile unsigned int* hist
                , const uint32_t N
                // the lower & upper bounds
                // of the current chunk
                , const uint32_t LB
                , const uint32_t UB
) {
    const uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < N) {
        uint32_t ind = inp_inds[gid];

        if(ind < UB && ind >= LB) {
            unsigned int val = inp_vals[gid];
            atomicAdd((unsigned int*)&hist[ind], val);
        }
    }
}


// for debugging
template<int B>
void multiStepUnsignedIntHisto ( uint32_t* d_inp_inds
                    , unsigned int*    d_inp_vals
                    , unsigned int*    d_hist
                    , const uint32_t N
                    , const uint32_t H
                    , const uint32_t LLC
) {
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = ( LLC_FRAC * LLC ) / sizeof(unsigned int);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;


    cudaMemset(d_hist, 0, H * sizeof(unsigned int));
    for(uint32_t k=0; k<num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k*CHUNK;
        uint32_t upp_bound = min( (k+1)*CHUNK, H );

        uint32_t grid = (N + B - 1) / B;
        {
            multiStepUnsignedIntKernel<<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, low_bound, upp_bound);
        }
    }
}


#endif
