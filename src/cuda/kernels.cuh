#ifndef kernels_h
#define kernels_h

//Q = numberOfElems = 22
//lgH = numberOfBits = 4
//H = powNumberOfBits = 16
//Note: currentPass is the current rank we are looking at. i.e. if it is 0,
//we look at the lgH bits starting from least significant bits. And jumping with
//currentPass*lgH bits each iteration. 
template <class ElTp> 
__global__ void makeHistogram(uint32_t* keysArray, uint32_t* histogramOut, uint32_t numberOfElems, uint8_t numberOfBits, uint32_t powNumberOfBits, uint32_t currentPass)
{
/*    uint32_t mBuckets = blockDim.x*numberOfElems*numberOfBits//What size?//blockDim.x;

    //Init a bucket array of size blockDim*bits*elems processed per block:
    uint32_t bucketArray[];
    for (uint32_t i = 0; i < mBuckets; ++i)
    {
        //Init array
        bucketArray[i] = 0;
    }
    //int currentFourBits = (number >> i) & 0b111
    int currentBitsOffset = currentPass*numberOfBits;
    //We add this for-loop to allow for processing of multiple bits at once!
    for (uint32_t i = 0; i < numberOfBits; ++i)
    {
        uint32_t key = keysArray[blockDim.x * i + threadIdx.x + (blockIdx.x * blockDim.x * numberOfBits)];
        //Extract only four bits starting from the LSB
        uint32_t extractedBits = (key >> currentBitsOffset) & 0b1111;
        //Update our bucket array at the correct position
        bucketArray[extractedBits * blockDim.x * numberOfBits + blockDim.x * i + threadIdx.x]++;
    }  */
    
    //Shared memory for the histogram
    extern __shared__ int local_histogram[]; 
    
    int tid = threadIdx.x;
    //Starting index for this block's elements
    int block_start = blockIdx.x * blockDim.x * numberOfElems; 

    //Initialize shared memory histogram
    for (int i = tid; i < powNumberOfBits; i += blockDim.x) 
    {
        local_histogram[i] = 0;
    }
    __syncthreads();

    //Process Q elements per thread
    for (int i = 0; i < numberOfElems; ++i) 
    {
        if ((block_start + tid + i * blockDim.x))
        {
            break;
        }
        int element = keysArray[block_start + tid + i * blockDim.x];

        //Process the element in 4-bit chunks (from least significant to most)
        for (int j = 0; j < num_chunks; ++j) 
        {
            int bitShift = j * numberOfBits;
            //Extract 4 bits at a time
            int extractedBits = (element >> bitShift) & (powNumberOfBits - 1);  

            //Update the histogram for the current chunk
            atomicAdd(&local_histogram[extractedBits + j * powNumberOfBits], 1);
        }
    }
    //I am not sure if this __syncthreads is necessary, but I am 99% it is
    __syncthreads();

    //Write local histogram to global memory
    for (int i = tid; i < powNumberOfBits * num_chunks; i += blockDim.x) 
    {
        histogramOut[blockIdx.x * powNumberOfBits * num_chunks + i] = local_histogram[i];
    }
}
    
template <class ElTp> 
__global__ void transposeKernel(uint32_t* inputMatrix, uint32_t* outputMatrix, uint32_t N, uint32_t powNumberOfBits)
{
    int bid  = blockIdx.x;
    int tid = threadIdx.x;

    if (bid < num_blocks && tid < powNumberOfBits * num_chunks) 
    {
        int inIndex = bid * powNumberOfBits * num_chunks + tid;
        int outIndex = tid * num_blocks + bid;
        transposed_histograms[outIndex] = histograms[inIndex];
    }

    //Calculate the row and column indices of the element
/*    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Ensure we are within bounds
    if (row < N && col < N) 
    {
        //Transpose the element from inputMatrix to outputMatrix
        outputMatrix[col * N + row] = inputMatrix[row * N + col];
    }*/
}

template <class ElTp> 
__global__ void flattenKernel(int* histograms, int* flattened, int powNumberOfBits) 
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int index = bid * powNumberOfBits + tid;
    if (tid < powNumberOfBits) 
    {
        flattened[index] = histograms[bid * powNumberOfBits + tid];
    }
}

template <class ElTp> 
__global__ void scanKernel(uint32_t* input, uint32_t* output, int size) 
{
    //Allocate temporary storage for the CUB scan operation
    void* scanned = nullptr;
    size_t scanBytes = 0;

    //Determine the size of the temporary storage needed
    cub::DeviceScan::ExclusiveSum(scanned, scanBytes, input, output, size);

    //Allocate temporary storage
    cudaMalloc(&d_temp_storage, scanBytes);

    //Perform the exclusive scan
    cub::DeviceScan::ExclusiveSum(scanned, scanBytes, input, output, size);
    cudaFree(scanned);
}

__global__ void finalKernel(uint32_t* d_input, uint32_t* d_output, uint32_t* d_histograms, uint32_t numElements, uint32_t numberOfElems, uint32_t numberOfBits, uint32_t powNumberOfBits) 
{
    int blockSize = blockDim.x;
    
    __shared__ uint32_t sharedHistogram[H];
    __shared__ uint32_t sharedData[numberOfElems * blockSize];



    int tid = threadIdx.x;
    int globalTid = blockIdx.x * numberOfElems * blockSize + tid;

    //Step 1: Load Q * B elements into registers from global memory via shared memory
    for (int i = 0; i < numberOfElems; i++) 
    {
        if (globalTid + i * blockSize < numElements) 
        {
            sharedData[tid + i * blockSize] = d_input[globalTid + i * blockSize];
        }
    }
    __syncthreads();

    uint32_t regs[numberOfElems];
    for (int i = 0; i < numberOfElems; i++) 
    {
        regs[i] = sharedData[tid + i * blockSize];
    }
    //Step 2: Two-way partitioning loop
    for (int bit = 0; bit < numberOfBits; ++bit) 
    {
        //Determine partition predicate for the current bit
        uint32_t mask = (1 << bit);
        
        //Step (i): Sequential reduction of each thread's elements based on the bit
        uint32_t trueCnt = 0;
        uint32_t falseCnt = 0;
        
        for (int i = 0; i < numberOfElems; i++) 
        {
            if (regs[i] & mask) 
            {
                trueCnt++;
            } 
            else 
            {
                falseCnt++;
            }
        }

        //Step (ii): Parallel block-level scan in shared memory
        uint32_t prefix_sum;
        __shared__ uint32_t shared_true_counts[blockSize];
        shared_true_counts[tid] = trueCnt;
        __syncthreads();

        cub::BlockScan<uint32_t, blockSize>(shared_true_counts).InclusiveSum(shared_true_counts[tid], prefix_sum);
        __syncthreads();

        //Step (iii): Apply prefix to each thread’s elements in registers
        uint32_t startIdxT = prefix_sum;
        uint32_t startIdxF = startIdxT + shared_true_counts[tid];

        for (int i = 0; i < numberOfElems; i++) 
        {
            if (regs[i] & mask) 
            {
                sharedData[startIdxT + i] = regs[i];
            } 
            else 
            {
                sharedData[startIdxF + i] = regs[i];
            }
        }
        __syncthreads();
    }

    //Step 3: Copy histogram to shared memory and perform in-place scan
    if (tid < powNumberOfBits) 
    {
        sharedHistogram[tid] = d_histograms[blockIdx.x * powNumberOfBits + tid];
    }
    __syncthreads();

    cub::BlockScan<uint32_t, blockSize>(sharedHistogram).InclusiveSum(sharedHistogram[tid], sharedHistogram[tid]);
    __syncthreads();

    //Step 4: Write elements to final positions in global memory
    for (int i = 0; i < numberOfElems; i++) 
    {
        int outIdx = sharedHistogram[regs[i] & ((1 << numberOfBits) - 1)] + tid * numberOfElems + i;
        d_output[outIdx] = regs[i];
    }
}

template <class ElTp> 
__global__ void RadixSortKer(ElTp* d_in, ElTp* d_out, int size) {
    
}
#endif
