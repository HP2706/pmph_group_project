#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <sys/time.h>
#include <time.h>

#define DEBUG_INFO  true

#define RUNS_GPU            100
#define RUNS_CPU            5
#define NUM_BLOCKS_SCAN     1024
#define ELEMS_PER_THREAD    6 // this is for pbb_kernels.cuh
#define LLC                 41943040 // number taken from assignment 3-4
#define LLC_FRAC (3.0 / 7.0)
#define WARP_COUNT 32 // renamed from WARP to ensure compatibility with the cub library
#define lgWARP 5



typedef unsigned int uint32_t;
typedef int           int32_t;



uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        puts("====");
    }
}

#endif // CONSTANTS_H
