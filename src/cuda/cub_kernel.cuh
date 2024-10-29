#pragma once

#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>


// Block-sorting CUDA kernel

// we use ElTp to make it generic over data types 
template <class ElTp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void CUBSortKernel(
    ElTp* d_in, 
    ElTp* d_out,
    int size // the length of the array
)
{
    using namespace cub;

    // Specialize BlockRadixSort, BlockLoad, and BlockStore for the given parameters
    typedef cub::BlockRadixSort<ElTp, BLOCK_THREADS, ITEMS_PER_THREAD>                     BlockRadixSort;
    typedef cub::BlockLoad<ElTp, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>    BlockLoad;
    typedef cub::BlockStore<ElTp, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>  BlockStore;

    // Allocate shared memory
    __shared__ union {
        typename BlockRadixSort::TempStorage  sort;
        typename BlockLoad::TempStorage       load;
        typename BlockStore::TempStorage      store;
    } temp_storage;

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);  // OffsetT for this block's segment

    // Obtain a segment of consecutive keys that are blocked across threads
    ElTp thread_keys[ITEMS_PER_THREAD];
    BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
    __syncthreads();

    // Collectively sort the keys
    BlockRadixSort(temp_storage.sort).Sort(thread_keys);
    __syncthreads();

    // Store the sorted segment
    BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

