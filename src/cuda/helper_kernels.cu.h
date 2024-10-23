#define LLC_FRAC (3.0 / 7.0)

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

/**
 * A CUDA-block of threads cooperatively scan with generic-binop `OP`
 *   a CUDA-block number of elements stored in shared memory (`ptr`).
 * `idx` is the local thread index within a cuda block (threadIdx.x)
 * Each thread returns the corresponding scanned element of type
 *   `typename OP::RedElTp`. Note that this is NOT published to shared memory!
 *
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