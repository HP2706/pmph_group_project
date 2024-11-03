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

    void* mem = NULL;
    size_t len = 0;
    
    uint32_t startBit = 0;
    uint32_t endBit = sizeof(T) * 8; // 8 bits per byte 
    
    cub::DeviceRadixSort::SortKeys(
        mem, 
        len, 
        d_keys_in, 
        d_keys_out, 
        SIZE, 
        startBit, 
        endBit
    );
    cudaMalloc(&mem, len);

    cudaDeviceSynchronize();
    cudaError_t cub_err = cudaGetLastError();
    if (cub_err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(cub_err));
        return;
    }

    cub::DeviceRadixSort::SortKeys(
        mem, 
        len, 
        d_keys_in, 
        d_keys_out, 
        SIZE, 
        startBit, 
        endBit
    );
    cudaDeviceSynchronize();
    
    cub_err = cudaGetLastError();
    if (cub_err != cudaSuccess) {
        printf("cub sort kernel failed: %s\n", cudaGetErrorString(cub_err));
        return;
    }
    cudaFree(mem);
}

