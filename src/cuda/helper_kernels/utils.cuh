#ifndef UTILS_CUH
#define UTILS_CUH

#define LLC_FRAC (3.0 / 7.0)
#define WARP 32
#define lgWARP 5

#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

uint32_t HWD;
uint32_t BLOCK_SZ;

void getDeviceInfo() {
{
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, 0);
        HWD = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
        const uint32_t BLOCK_SZ = prop.maxThreadsPerBlock;
        const uint32_t SH_MEM_SZ = prop.sharedMemPerBlock;
        
        {
            printf("Device name: %s\n", prop.name);
            printf("Number of hardware threads: %d\n", HWD);
            printf("Block size: %d\n", BLOCK_SZ);
            printf("Shared memory size: %d\n", SH_MEM_SZ);
            puts("====");
        }
    }
}

#endif
