#pragma once

#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <algorithm>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

// this sorts the entire array
// however we cannot specify num_items(Q) in the kernel launch
template<typename T>
void deviceRadixSortKernel(
    T* d_keys_in, 
    T* d_keys_out, 
    int SIZE
) {
    // No need for initial copy since d_keys_in already contains the input data
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Get size requirements for temporary storage
    // its a bit weird that we need to call this twice but 
    // it is necessary for cub to allocate the correct amount of memory
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, 
        temp_storage_bytes,
        d_keys_in, 
        d_keys_out,
        SIZE
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform the sort
    cub::DeviceRadixSort::SortKeys(
        d_temp_storage, 
        temp_storage_bytes,
        d_keys_in, 
        d_keys_out,
        SIZE
    );

    cudaFree(d_temp_storage);
}


// we use ElTp to make it generic over data types 
template <typename ElTp, int BLOCK_SIZE, int Q>
__global__ void CUBSortKernel(
    ElTp* d_in, 
    ElTp* d_out,    
    int size 
)
{
    using namespace cub;

    // Specialize BlockRadixSort, BlockLoad, and BlockStore for the given parameters
    typedef cub::BlockRadixSort<ElTp, BLOCK_SIZE, Q>                     BlockRadixSort;
    typedef cub::BlockLoad<ElTp, BLOCK_SIZE, Q, cub::BLOCK_LOAD_TRANSPOSE>    BlockLoad;
    typedef cub::BlockStore<ElTp, BLOCK_SIZE, Q, cub::BLOCK_STORE_TRANSPOSE>  BlockStore;

    // Allocate shared memory
    __shared__ union {
        typename BlockRadixSort::TempStorage  sort;
        typename BlockLoad::TempStorage       load;
        typename BlockStore::TempStorage      store;
    } temp_storage;

    int block_offset = blockIdx.x * (BLOCK_SIZE * Q);  // OffsetT for this block's segment

    // Obtain a segment of consecutive keys that are blocked across threads
    ElTp thread_keys[Q];
    BlockLoad(temp_storage.load).Load(d_in + block_offset, thread_keys);
    __syncthreads();

    // Collectively sort the keys
    BlockRadixSort(temp_storage.sort).Sort(thread_keys);
    __syncthreads();

    // Store the sorted segment
    BlockStore(temp_storage.store).Store(d_out + block_offset, thread_keys);
}


// Example to check cub sort global works
int callCubSort() {
    const int NUM_ITEMS = 1000000;
    
    // Allocate host arrays
    int* h_keys = new int[NUM_ITEMS];

    // Initialize data
    for (int i = 0; i < NUM_ITEMS; i++) {
        h_keys[i] = rand();
    }

    // Allocate device memory
    int *d_keys_in, *d_keys_out;
    cudaMalloc(&d_keys_in, NUM_ITEMS * sizeof(int));
    cudaMalloc(&d_keys_out, NUM_ITEMS * sizeof(int));

    cudaMemcpy(d_keys_in, h_keys, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);

    deviceRadixSortKernel(d_keys_in, d_keys_out, NUM_ITEMS);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_keys, d_keys_out, NUM_ITEMS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1000; i++) {
        printf("%d\n", h_keys[i]);
    }


    // Cleanup
    delete[] h_keys;
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);

    return 0;
}
