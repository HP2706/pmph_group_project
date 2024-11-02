#ifndef PREFIX_SUM_CUH
#define PREFIX_SUM_CUH

#include "utils.cuh"
#include "pbb_kernels.cuh"
#include "../constants.cuh"
#include "../helper.h"
#include <cuda_runtime.h>

/**
 * Host Wrapper orchestraiting the execution of scan:
 * d_in  is the input array
 * d_out is the result array (result of scan)
 * t_tmp is a temporary array (used to scan in-place across the per-block results)
 * Implementation consist of three phases:
 *   1. elements are partitioned across CUDA blocks such that the number of
 *      spawned CUDA blocks is <= 1024. Each CUDA block reduces its elements
 *      and publishes the per-block partial result in `d_tmp`. This is
 *      implemented in the `redAssocKernel` kernel.
 *   2. The per-block reduced results are scanned within one CUDA block;
 *      this is implemented in `scan1Block` kernel.
 *   3. Then with the same partitioning as in step 1, the whole scan is
 *      performed again at the level of each CUDA block, and then the
 *      prefix of the previous block---available in `d_tmp`---is also
 *      accumulated to each-element result. This is implemented in
 *      `scan3rdKernel` kernel and concludes the whole scan.
 */
template<class OP, int Q>                     // element-type and associative operator properties
void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
) {
    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = Q*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);
    const size_t   shmem_size = B * max_tp_size * CHUNK;

    //
    redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_tmp, d_in, N, num_seq_chunks);

    {
        const uint32_t block_size = nextMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }

    scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>(d_out, d_in, d_tmp, N, num_seq_chunks);
}


/// from exercise 3-4 bmm 
template <class ElTp, int T> 
__global__ void matTransposeTiledKer(
    ElTp* A, // input matrix
    ElTp* A_tr, // output matrix
    const int heightA, // height of input matrix
    const int widthA // width of input matrix
) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x; 
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      A_tr[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}

template<class T, int TILE_SIZE>
void tiledTranspose(
    T* d_in, // input matrix
    T* d_out, // output matrix
    uint32_t height, // width of input matrix
    uint32_t width // height of input matrix
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    matTransposeTiledKer<T, TILE_SIZE><<<grid, block>>>(d_in, d_out, height, width);
}


// this kernel efficiently 
// computes the scan across buckets via transpose -> scan -> transpose 
template<class P>
void scan_buckets(
    uint16_t* arr,
    uint16_t* arr_t,
    uint64_t* arr_t_scanned, 
    uint64_t* arr_t_scanned_t // the end result is stored here
) {

    // we allocate a temporary buffer for the 
    uint64_t* d_tmp_buf;
    // NUM_BLOCKS_SCAN is 1024 the largest number of blocks we can have on cuda gpu
    cudaMalloc(&d_tmp_buf, NUM_BLOCKS_SCAN * sizeof(uint64_t));

    tiledTranspose<uint16_t, P::TILE_SIZE>(
        arr, 
        arr_t, 
        P::H, // width is the height of the input matrix
        P::NUM_BLOCKS // height is the number of blocks we can have on cuda gpu
    );

    // we add uint16_t elemens and store the result in uint64_t
    scanInc<FusedAddCast<uint16_t, uint64_t>, P::Q>(
        P::B,
        P::NUM_BLOCKS,
        arr_t_scanned, // output array
        arr_t, // input array
        d_tmp_buf // temporary buffer of size NUM_BLOCKS_SCAN
    );

    // we transpose the result back
    tiledTranspose<uint64_t, P::TILE_SIZE>(
        arr_t_scanned,
        arr_t_scanned_t,
        P::NUM_BLOCKS,
        P::H
    );


}


#endif
