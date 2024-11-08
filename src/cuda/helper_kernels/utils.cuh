#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "../constants.cuh"
/// helper kernels from assignment 2 and 3-4



// blockDim.y = T; blockDim.x = T
// each block transposes a square T

// blockDim.y = T; blockDim.x = T
// each block transposes a square T
template <class ElTp, int T> 
__global__ void
coalsTransposeKer(ElTp* A, ElTp* B, int heightA, int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;


  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}


uint32_t nextMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

/**
 * `N` is the input-array length
 * `B` is the CUDA block size
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024.
 */
template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, (N + min_elms_all_thds - 1) / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks <= MAX_BLOCK) {
        return num_blocks;
    } else {
        //printf("Warning: reduce/scan configuration does not allow the maximal concurrency supported by hardware.\n");
        const uint32_t num_blocks = 1024;
        const uint32_t num_thds   = num_blocks * B;
        const uint32_t num_conc_elems = num_thds * CHUNK;
        *num_chunks = (N + num_conc_elems - 1) / num_conc_elems;
        return num_blocks;
    }    
}

template<class T>
bool __host__ __device__ isBitUnset(int pos, T val) {
    return (val & (1 << pos)) == 0;
}



template<class T, int lgH>
int __device__ getBits(int pos, T val) {
    return (val >> pos) & ((1 << lgH) - 1);
}

#define CUDASSERT(code) { __cudassert((code), __FILE__, __LINE__); }
#define CUDACHECK(code) { __cudassert((code), __FILE__, __LINE__, false); }

void __cudassert(cudaError_t code,
                 const char *file,
                 int line,
                 bool do_abort_on_err = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error in %s (line %d): %s\n",
            file, line, cudaGetErrorString(code));
    if (do_abort_on_err)
        exit(1);
  }
}


/**
 * Generic Add operator with output type casting. Can be instantiated over
 * numeric types, with different input and output types.
 * For example: FusedAddCast<int, float> will add integers but return floats.
 */
template<class InT, class OutT>
class FusedAddCast {
  public:
    typedef InT InpElTp;   // Type of input elements
    typedef OutT RedElTp;  // Type of reduced/output elements
    static const bool commutative = true;
    
    static __device__ __host__ inline InT identInp() { 
        return (InT)0; 
    }
    
    static __device__ __host__ inline OutT mapFun(const InT& el) { 
        return (OutT)el; 
    }
    
    static __device__ __host__ inline OutT identity() { 
        return (OutT)0; 
    }
    
    static __device__ __host__ inline OutT apply(const OutT t1, const OutT t2) { 
        return t1 + t2; 
    }

    static __device__ __host__ inline bool equals(const OutT t1, const OutT t2) { 
        return (t1 == t2); 
    }
    
    static __device__ __host__ inline OutT remVolatile(volatile OutT& t) { 
        OutT res = t; 
        return res; 
    }
};


#endif
