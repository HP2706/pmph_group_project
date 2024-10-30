#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstdint>
#include <time.h>
#include <limits>
#if 0
typedef int        int32_t;
typedef long long  int64_t;
#endif

typedef unsigned int uint32_t;

uint32_t HWD;
uint32_t BLOCK_SZ;


// this is a struct that holds the generic parameters for the radix sort

template<
    typename ElTp, 
    typename UintTp, 
    int _Q, 
    int _lgH, 
    int _GRID_SIZE, 
    int _BLOCK_SIZE, 
    int _T
>
struct Params {
    static constexpr int Q = _Q; // number of elements per thread
    static constexpr int lgH = _lgH; // number of bits per iteration
    static constexpr int GRID_SIZE = _GRID_SIZE; // number of blocks per grid
    static constexpr int BLOCK_SIZE = _BLOCK_SIZE; // number of threads per block
    static constexpr int T = _T; // Tile size
    static constexpr int H = 1 << _lgH; // number of bins, this is simply 2^lgH 
    static constexpr int QB = _Q * _BLOCK_SIZE; // number of elements per block
    static constexpr int HB = H * _BLOCK_SIZE; // size of the histogram array
    static constexpr int MAXNUMERIC_ElementType = std::numeric_limits<ElTp>::max(); // the maximum value of the input array
    static constexpr int MAXNUMERIC_UintType = std::numeric_limits<UintTp>::max(); // the maximum value of the histogram array
    using ElementType = ElTp; // the type of the input array
    using UintType = UintTp; // the type of the histogram

    static_assert(lgH <= 8, "LGH must be less than or equal to 8 as otherwise shared memory will overflow");

};


// Type trait to check if T is an instance of Params
template<typename T>
struct is_params : std::false_type {};

template<class ElTp, class UintTp, int _Q, int _lgH, int _GRID_SIZE, int _BLOCK_SIZE, int _T>
struct is_params<Params<ElTp, UintTp, _Q, _lgH, _GRID_SIZE, _BLOCK_SIZE, _T>> : std::true_type {};


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

template<class T>
bool checkSorted(T* arr, uint64_t size) {
    for(uint64_t i = 1; i < size; i++) {
        if(arr[i-1] > arr[i]) {
            printf("Array not sorted at index %llu: %u > %u\n", i, arr[i-1], arr[i]);
            return false;
        }
    }
    printf("Array is sorted!\n");
    return true;
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




template <typename T, int MAX_VAL>
void allocateAndInitialize(
    T** h_ptr, 
    T** d_ptr, 
    uint32_t N, 
    bool initRnd = false
) {
    // Allocate and initialize host memory if h_ptr is provided
    if (h_ptr) {
        *h_ptr = (T*) malloc(sizeof(T) * N);
        
        // Initialize host memory
        if (initRnd) {
            randomInit<T>(*h_ptr, N, MAX_VAL); // Using 1000 as default upper bound
        } else {
            memset(*h_ptr, 0, sizeof(T) * N);
        }
    }

    // Allocate device memory if d_ptr is provided
    if (d_ptr) {
        cudaMalloc((void**)d_ptr, sizeof(T) * N);
        
        // If we have both pointers, copy host to device
        if (h_ptr) {
            cudaMemcpy(*d_ptr, *h_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
        } else {
            if (initRnd) {
                // raise an error
                throw std::runtime_error("Cannot randomly initialize device memory without host memory");
            } else {
                cudaMemset(*d_ptr, 0, sizeof(T) * N);
            }
        }
    }
}


template <typename T>
void checkAllZeros(T* ptr, uint32_t size) {
    bool all_zeros = true;
    for (uint32_t i = 0; i < size; i++) {
        if (ptr[i] != 0) {
            printf("Non-zero value at index %u: %f\n", i, (float)ptr[i]);
            all_zeros = false;
            break;
        }
    }
    if (all_zeros) {
        // raise an error
        throw std::runtime_error("All zeros");
    }
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
