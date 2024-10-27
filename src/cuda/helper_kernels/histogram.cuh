#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#pragma once
#include "utils.cuh"
#include "../traits.h"  // Adjust the path based on your directory structure
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>


template<typename AnyUInt, uint32_t LGH>
__global__ void
RadixHistoKernel(uint32_t* keysArray, uint32_t* histogramOut, uint32_t numberOfElems, uint32_t numberOfBits, uint32_t powNumberOfBits, uint32_t totalKeyArraySize)
{
/*    uint32_t mBuckets = blockDim.x*numberOfElems*numberOfBits//What size?//blockDim.x;

    //Init a bucket array of size blockDim*bits*elems processed per block:
    uint32_t bucketArray[];
    for (uint32_t i = 0; i < mBuckets; ++i)
    {
        //Init array
        bucketArray[i] = 0;
    }
    //int currentFourBits = (number >> i) & 0b111
    int currentBitsOffset = currentPass*numberOfBits;
    //We add this for-loop to allow for processing of multiple bits at once!
    for (uint32_t i = 0; i < numberOfBits; ++i)
    {
        uint32_t key = keysArray[blockDim.x * i + threadIdx.x + (blockIdx.x * blockDim.x * numberOfBits)];
        //Extract only four bits starting from the LSB
        uint32_t extractedBits = (key >> currentBitsOffset) & 0b1111;
        //Update our bucket array at the correct position
        bucketArray[extractedBits * blockDim.x * numberOfBits + blockDim.x * i + threadIdx.x]++;
    }  */
    
    //Shared memory for the histogram
    extern __shared__ uint32_t localHistogram[]; 
//    __shared__ uint32_t localHistogram[256]; 
    int tid = threadIdx.x;
    //Starting index for this block's elements
    int blockStart = blockIdx.x * blockDim.x + threadIdx.x;

    //Initialize shared memory histogram. We know from "produces a histogram of length H"
    //the length.
    for (int i = tid; i < powNumberOfBits; i += blockDim.x) 
    {
        localHistogram[i] = 0;
    }
    __syncthreads();

    int numChunks = numberOfBits;
    //Process Q elements per thread
    for (int i = 0; i < numberOfElems; ++i) 
    {
        if ((blockStart * numberOfElems + i) >= totalKeyArraySize)
        {
            return;
        }
        int element = keysArray[blockIdx.x * blockDim.x * numberOfElems + tid * numberOfElems + i];

        //Process the element in 4-bit chunks (from least significant to most)
        for (int j = 0; j < numChunks; ++j) 
        {
            int bitShift = j * numChunks;
            //Extract 4 bits at a time
            int extractedBits = (element >> bitShift) & (powNumberOfBits - 1);  

            //Update the histogram for the current chunk
            atomicAdd(&localHistogram[extractedBits], 1);
        }
    }
    //I am not sure if this __syncthreads is necessary, but I am 99% it is
    __syncthreads();

    //Write local histogram to global memory
    for (int i = tid; i < powNumberOfBits; i += blockDim.x) 
    {
        atomicAdd(&histogramOut[blockIdx.x * powNumberOfBits + i], localHistogram[i]);
    }
}


#endif
