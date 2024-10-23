#ifndef HELPER_KERNELS_CU_H
#define HELPER_KERNELS_CU_H

#define LLC_FRAC (3.0 / 7.0)

template<class T, uint32_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {

        //uint32_t loc_ind = threadIdx.x*CHUNK + i;
        uint32_t loc_ind = i * blockDim.x + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < N) { elm = d_inp[glb_ind]; }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads(); // leave this here at the end!
}

/**
 * This is very similar with `copyFromGlb2ShrMem` except
 * that you need to copy from shared to global memory, so
 * that consecutive threads write consecutive indices in
 * global memory in the same SIMD instruction.
 * `glb_offs` is the offset in global-memory array `d_out`
 *    where elements should be written.
 * `d_out` is the global-memory array
 * `N` is the length of `d_out`
 * `shmem_red` is the shared-memory of size
 *    `blockDim.x*CHUNK*sizeof(T)`
 */
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red
) {
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        //uint32_t loc_ind = threadIdx.x * CHUNK + i;
        uint32_t loc_ind = i * blockDim.x + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_red[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads(); // leave this here at the end!
}


/********************/
/*** Scan Kernels ***/
/********************/



/**
 * A warp of threads cooperatively scan with generic-binop `OP` a
 *   number of warp elements stored in shared memory (`ptr`).
 * No synchronization is needed because the threads in a warp execute
 *   in lockstep.
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`
*/
template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( 
    volatile typename OP::RedElTp* ptr, 
    const unsigned int idx 
) {
    const unsigned int lane = idx & (WARP-1);
    
    #pragma unroll
    for (int d = 0; d < 5; d++) {
        int h = 1 << d; // 2^d double the stride
        if (lane >= h) {
            ptr[idx] = OP::apply(ptr[idx - h], ptr[idx]);
        }
    }
    
    return OP::remVolatile(ptr[idx]);
}

/// Exclusive scan for a single warp
template<class OP>
__device__ inline typename OP::RedElTp
scanExcWarp(
    volatile typename OP::RedElTp* ptr,
    const unsigned int idx
) {
    const unsigned int lane = idx & (WARP-1);
    typename OP::RedElTp temp;

    #pragma unroll
    for (int d = 0; d < 5; d++) {
        int h = 1 << d; // 2^d double the stride
        if (lane >= h) {
            ptr[idx] = OP::apply(ptr[idx - h], ptr[idx]);
        }
    }

    // Shift right by one to make it exclusive
    // this is the only difference with scanIncWarp
    // if lane > 0, then we take the previous element
    // for the first element lane == 0
    // we take the identity element ie for integers 0
    temp = (lane > 0) ? OP::remVolatile(ptr[idx - 1]) : OP::identity();

    return temp;
}



/**
 * Kernel for scanning up to CUDA-block elements using
 *    CUDA-block threads.
 * `N` number of elements to be scanned (N < CUDA-block size)
 * `d_input` is the value array stored in shared memory
 *  This kernel operates in-place, i.e., the input is
 *  overwritten with the result.
*/
template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane    = idx & (WARP - 1);
    const unsigned int warpid  = idx >> lgWARP;
    const unsigned int n_warps = (blockDim.x + WARP - 1) >> lgWARP; // Total number of warps

    // 1. Perform scan at warp level.
    typename OP::RedElTp res = scanIncWarp<OP>(ptr, idx);
    __syncthreads();
    
    // Place the end-of-warp results into a separate location in memory.
    typename OP::RedElTp end = OP::remVolatile(ptr[idx]);
    // synchronize the threads so that every thread has stored 
    // the value in the memory location before we write
    __syncthreads();
    if (lane == (WARP - 1)) {
        ptr[warpid] = end;
    }

    __syncthreads();
    // 3. Let the first warp scan the per-warp sums.
    // 3. scan again the first warp.
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step.
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }


    return res;
}

template<class OP>
__global__ void
scan1Block( typename OP::RedElTp* d_inout, uint32_t N ) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp elm = OP::identity();
    if(threadIdx.x < N) {
        elm = d_inout[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    __syncthreads();
    elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    if (threadIdx.x < N) {
        d_inout[threadIdx.x] = elm;
    }
}

/**
 * This kernel assumes that the generic-associative binary operator
 *   `OP` is NOT-necessarily commutative. It implements the third
 *   stage of the scan (parallel prefix sum), which scans within
 *   a block.  (The first stage is a per block reduction with the
 *   `redAssocKernel` kernel, and the second one is the `scan1Block`
 *    kernel that scans the reduced elements of each CUDA block.)
 *
 * `N` is the length of the input array
 * `CHUNK` (the template parameter) is the number of elements to
 *    be processed sequentially by a thread in one go.
 * `num_seq_chunks` is used to sequentialize even more computation,
 *    such that the number of blocks is <= 1024.
 * `d_out` is the result array of length `N`
 * `d_in`  is the input  array of length `N`
 * `d_tmp` is the array holding the per-block scanned results.
 *         it has number-of-CUDA-blocks elements, i.e., element
 *         `d_tmp[i-1]` is the scanned prefix that needs to be
 *         accumulated to each of the scanned elements corresponding
 *         to block `i`.
 * This kernels scans the elements corresponding to the current block
 *   `i`---in number of num_seq_chunks*CHUNK*blockDim.x---and then it
 *   accumulates to each of them the prefix of the previous block `i-1`,
 *   which is stored in `d_tmp[i-1]`.
 */
template<class OP, int CHUNK>
__global__ void
scan3rdKernel ( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;

    // number of elments to be processed by an iteration of the
    // "virtualization" loop
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.
    typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];

    // register memory for storing the scanned elements.
    typename OP::RedElTp chunk[CHUNK];

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_in, shmem_inp);

        // 2. each thread sequentially scans its `CHUNK` elements
        //    and stores the result in the `chunk` array. The reduced
        //    result is stored in `tmp`.
        typename OP::RedElTp tmp = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            tmp = OP::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();

        // 3. Each thread publishes in shared memory the reduced result of its
        //    `CHUNK` elements
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 4. perform an intra-CUDA-block scan
        tmp = scanIncBlock<OP>(shmem_red, threadIdx.x);
        __syncthreads();

        // 5. write the scan result back to shared memory
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 6. the previous element is read from shared memory in `tmp`:
        //       it is the prefix of the previous threads in the current block.
        tmp   = OP::identity();
        if (threadIdx.x > 0)
            tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);
        // 7. the prefix of the previous blocks (and iterations) is hold
        //    in `accum` and is accumulated to `tmp`, which now holds the
        //    global prefix for the `CHUNK` elements processed by the current thread.
        tmp   = OP::apply(accum, tmp);

        // 8. `accum` is also updated with the reduced result of the current
        //    iteration, i.e., of the last thread in the block: `shmem_red[blockDim.x-1]`
        accum = OP::apply(accum, shmem_red[blockDim.x-1]);
        __syncthreads();

        // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
        //      locally processed by the current thread (i.e., the ones
        //      in `chunk` array hold in registers).
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
        }
        __syncthreads();

        // 5. write back from shared to global memory in coalesced fashion.
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs+seq, N, d_out, shmem_red);
    }
}

/// the multistep kernel for the histogram
__global__ void
multiStepKernel ( uint32_t* inp_inds
                , float*    inp_vals
                , volatile float* hist
                , const uint32_t N
                // the lower & upper bounds
                // of the current chunk
                , const uint32_t LB
                , const uint32_t UB
) {
    const uint32_t gid = blockIdx.x*blockDim.x + threadIdx.x;

    if(gid < N) {
        uint32_t ind = inp_inds[gid];
        if(ind < UB && ind >= LB) {
            float val = inp_vals[gid];
            atomicAdd((float*)&hist[ind], val);
        }
    }
}


template<int B>
void multiStepHisto ( uint32_t* d_inp_inds
                    , float*    d_inp_vals
                    , float*    d_hist
                    , const uint32_t N
                    , const uint32_t H
                    , const uint32_t LLC
) {
    // we use a fraction L of the last-level cache (LLC) to hold `hist`
    const uint32_t CHUNK = ( LLC_FRAC * LLC ) / sizeof(float);
    uint32_t num_partitions = (H + CHUNK - 1) / CHUNK;

    cudaMemset(d_hist, 0, H * sizeof(float));
    for (uint32_t k=0; k<num_partitions; k++) {
        // we process only the indices falling in
        // the integral interval [k*CHUNK, (k+1)*CHUNK)
        uint32_t low_bound = k*CHUNK;
        uint32_t upp_bound = min( (k+1)*CHUNK, H );

        uint32_t grid = (N + B - 1) / B;
        {
            multiStepKernel<<<grid,B>>>(d_inp_inds, d_inp_vals, d_hist, N, low_bound, upp_bound);
        }
    }
}


/// Exclusive scan for a single block
/// this is an implementation inspired by the scanIncBlock reference implementation 
/// given in the exercises for week 2 in "pbb_kernels.cuh"
template<class OP>
__device__ inline typename OP::RedElTp
scanExcBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane    = idx & (WARP - 1);
    const unsigned int warpid  = idx >> lgWARP;
    const unsigned int n_warps = (blockDim.x + WARP - 1) >> lgWARP; // Total number of warps

    // 1. Perform exclusive scan at warp level.
    typename OP::RedElTp res = scanExcWarp<OP>(ptr, idx);
    __syncthreads();
    
    // Place the end-of-warp results into a separate location in memory.
    typename OP::RedElTp end = OP::remVolatile(ptr[idx]);
    __syncthreads();
    if (lane == (WARP - 1)) {
        ptr[warpid] = end;
    }

    __syncthreads();
    // 3. Let the first warp scan the per-warp sums.
    if (warpid == 0) scanExcWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step.
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

    return res;
}

// Exclusive scan for a single block
template<class OP>
__global__ void
scanExc1Block(typename OP::RedElTp* d_inout, uint32_t N) {
    extern __shared__ char sh_mem[];
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;
    typename OP::RedElTp elm = OP::identity();
    if(threadIdx.x < N) {
        elm = d_inout[threadIdx.x];
    }
    shmem_red[threadIdx.x] = elm;
    __syncthreads();
    elm = scanExcBlock<OP>(shmem_red, threadIdx.x);
    if (threadIdx.x < N) {
        d_inout[threadIdx.x] = elm;
    }
}

/// scan3rdKernel but with exclusive scan instead
template<class OP, int CHUNK>
__global__ void
ScanExc ( typename OP::RedElTp* d_out
              , typename OP::InpElTp* d_in
              , typename OP::RedElTp* d_tmp
              , uint32_t N
              , uint32_t num_seq_chunks
) {
    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::InpElTp* shmem_inp = (typename OP::InpElTp*)sh_mem;

    // shared memory for the reduce-element type; it overlaps with the
    //   `shmem_inp` since they are not going to be used in the same time.
    volatile typename OP::RedElTp* shmem_red = (typename OP::RedElTp*)sh_mem;

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = num_seq_chunks * CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t inp_block_offs = num_elems_per_block * blockIdx.x;

    // number of elments to be processed by an iteration of the
    // "virtualization" loop
    uint32_t num_elems_per_iter  = CHUNK * blockDim.x;

    // accumulator updated at each iteration of the "virtualization"
    //   loop so we remember the prefix for the current elements.
    typename OP::RedElTp accum = (blockIdx.x == 0) ? OP::identity() : d_tmp[blockIdx.x-1];

    // register memory for storing the scanned elements.
    typename OP::RedElTp chunk[CHUNK];

    // virtualization loop of count `num_seq_chunks`. Each iteration processes
    //   `blockDim.x * CHUNK` elements, i.e., `CHUNK` elements per thread.
    for(int seq=0; seq<num_elems_per_block; seq+=num_elems_per_iter) {
        // 1. copy `CHUNK` input elements per thread from global to shared memory
        //    in coalesced fashion (for global memory)
        copyFromGlb2ShrMem<typename OP::InpElTp, CHUNK>
                  (inp_block_offs+seq, N, OP::identInp(), d_in, shmem_inp);

        // 2. each thread sequentially scans its `CHUNK` elements
        //    and stores the result in the `chunk` array. The reduced
        //    result is stored in `tmp`.
        typename OP::RedElTp tmp = OP::identity();
        uint32_t shmem_offset = threadIdx.x * CHUNK;
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            typename OP::InpElTp elm = shmem_inp[shmem_offset + i];
            typename OP::RedElTp red = OP::mapFun(elm);
            tmp = OP::apply(tmp, red);
            chunk[i] = tmp;
        }
        __syncthreads();

        // 3. Each thread publishes in shared memory the reduced result of its
        //    `CHUNK` elements
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 4. perform an intra-CUDA-block scan
        tmp = scanExcBlock<OP>(shmem_red, threadIdx.x);
        __syncthreads();

        // 5. write the scan result back to shared memory
        shmem_red[threadIdx.x] = tmp;
        __syncthreads();

        // 6. the previous element is read from shared memory in `tmp`:
        //       it is the prefix of the previous threads in the current block.
        tmp   = OP::identity();
        if (threadIdx.x > 0)
            tmp = OP::remVolatile(shmem_red[threadIdx.x-1]);
        // 7. the prefix of the previous blocks (and iterations) is hold
        //    in `accum` and is accumulated to `tmp`, which now holds the
        //    global prefix for the `CHUNK` elements processed by the current thread.
        tmp   = OP::apply(accum, tmp);

        // 8. `accum` is also updated with the reduced result of the current
        //    iteration, i.e., of the last thread in the block: `shmem_red[blockDim.x-1]`
        accum = OP::apply(accum, shmem_red[blockDim.x-1]);
        __syncthreads();

        // 9. the `tmp` prefix is accumulated to all the `CHUNK` elements
        //      locally processed by the current thread (i.e., the ones
        //      in `chunk` array hold in registers).
        #pragma unroll
        for (uint32_t i = 0; i < CHUNK; i++) {
            shmem_red[threadIdx.x*CHUNK + i] = OP::apply(tmp, chunk[i]);
        }
        __syncthreads();

        // 5. write back from shared to global memory in coalesced fashion.
        copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
                  (inp_block_offs+seq, N, d_out, shmem_red);
    }
}


#endif