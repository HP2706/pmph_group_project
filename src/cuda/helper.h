#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#if 0
typedef int        int32_t;
typedef long long  int64_t;
#endif

typedef unsigned int uint32_t;


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




int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
void randomInit(T* data, uint64_t size, uint32_t upper) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = static_cast<T>(rand() % upper); // Generate random integers
}


void randomInds(uint32_t* data, uint64_t size, uint32_t M) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand() % M;
}

template<class T>
bool validate(T* A, T* B, uint64_t sizeAB){
    for(uint64_t i = 0; i < sizeAB; i++) {
        if ( A[i] != B[i] ) {
            printf("INVALID RESULT at flat index %llu: %f vs %f\n", i, (float)A[i], (float)B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}



template <typename UInt, int BLOCK_SIZE>
void PrepareMemory(
    UInt** h_in, 
    UInt** d_in, 
    uint32_t** d_hist, 
    uint32_t** h_hist,
    uint32_t num_bins,
    uint32_t SIZE,
    uint32_t hist_size
) {

    *h_in = (UInt*) malloc(sizeof(UInt) * SIZE);
    *h_hist = (uint32_t*) malloc(sizeof(uint32_t) * hist_size);

    // initialize h_hist to 0
    for (int i = 0; i < hist_size; i++) {
        (*h_hist)[i] = 0;
    }


    // 2. allocate device memory
    cudaMalloc((UInt**) d_in, sizeof(UInt) * SIZE); // Update the cudaMalloc call
    cudaMalloc((uint32_t**) d_hist, sizeof(uint32_t) * hist_size);

    // 3. initialize host memory
    randomInit<UInt>(*h_in, SIZE, num_bins);

    // 4. copy host memory to device
    cudaMemcpy(*d_in, *h_in, sizeof(UInt) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_hist, *h_hist, sizeof(uint32_t) * hist_size, cudaMemcpyHostToDevice);
}


#endif
