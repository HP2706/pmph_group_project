#include "../helper_kernels/twoWayPartitionImpl.cuh"
#include "../helper.h"
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <string>
#include <bitset>



// Kernel that calls TwoWayPartition
template<typename P>
__global__ void TwoWayPartitionKernel(
    typename P::ElementType* d_input,
    typename P::ElementType* d_output,
    uint32_t bitpos
) {
    __shared__ typename P::ElementType shmem[P::BLOCK_SIZE * P::Q];
    __shared__ uint16_t local_histo[P::BLOCK_SIZE];
    
    // Load input into registers
    typename P::ElementType reg[P::Q];
    uint32_t tid = threadIdx.x;
    for (int q = 0; q < P::Q; q++) {
        reg[q] = d_input[tid * P::Q + q];
    }
    
    // Call TwoWayPartition
    TwoWayPartition<P>(reg, shmem, local_histo, bitpos);

    // Store results back to global memory
    for (int q = 0; q < P::Q; q++) {
        d_output[tid * P::Q + q] = reg[q];
    }
}


template<class P>
void TestTwoWayPartition() {
    const uint32_t N = P::BLOCK_SIZE * P::Q;
    const uint32_t bitpos = 2; // Test with bit position 3

    // declare ptrs
    typename P::ElementType* h_input_gpu;
    typename P::ElementType* h_output_gpu;
    
    
    typename P::ElementType* h_output_cpu;
    typename P::ElementType* h_input_cpu;

    typename P::ElementType* d_input;
    typename P::ElementType* d_output;
    
    // allocate memory
    h_input_gpu = (typename P::ElementType*)malloc(N * sizeof(typename P::ElementType));
    h_output_gpu = (typename P::ElementType*)malloc(N * sizeof(typename P::ElementType));
    h_output_cpu = (typename P::ElementType*)malloc(N * sizeof(typename P::ElementType));
    h_input_cpu = (typename P::ElementType*)malloc(N * sizeof(typename P::ElementType));


    // allocate device memory
    cudaMalloc(&d_input, N * sizeof(typename P::ElementType));
    cudaMalloc(&d_output, N * sizeof(typename P::ElementType));

    // generate random input data
    for (uint32_t i = 0; i < N; i++) {
        h_input_gpu[i] = rand() % (P::MAXNUMERIC_ElementType);
    }

    memcpy(h_input_cpu, h_input_gpu, N * sizeof(typename P::ElementType));
    memset(h_output_cpu, 0, N * sizeof(typename P::ElementType));
    memset(h_output_gpu, 0, N * sizeof(typename P::ElementType));

    cudaMemcpy(d_input, h_input_gpu, N * sizeof(typename P::ElementType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output_gpu, N * sizeof(typename P::ElementType), cudaMemcpyHostToDevice);
    

    // Run CPU implementation
    TwoWayPartitionCpu<typename P::ElementType>(
        bitpos, N, h_input_cpu, h_output_cpu, false
    );
    
    printf("CPU done\n");

    printf("launching gpu kernel\n");
    // Launch kernel
    TwoWayPartitionKernel<P><<<1, P::BLOCK_SIZE>>>(d_input, d_output, bitpos);
    
    // Copy results back to host
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(typename P::ElementType), 
               cudaMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    uint32_t cpu_zeros = 0, gpu_zeros = 0;
    
    // Count zeros in both outputs
    for (uint32_t i = 0; i < N; i++) {
        cpu_zeros += isBitUnset(bitpos, h_output_cpu[i]) ? 1 : 0;
        gpu_zeros += isBitUnset(bitpos, h_output_gpu[i]) ? 1 : 0;
    }



    #if 0
    printf("cpu: ");
    for (uint32_t i = 0; i < N; i++) {
        printf("%d: %d ", i, isBitUnset(bitpos, h_output_cpu[i]));
    }
    printf("\n");

    printf("gpu: ");
    for (uint32_t i = 0; i < N; i++) {
        printf("%d: %d ", i, isBitUnset(bitpos, h_output_gpu[i]));
    }
    printf("\n");
    
    // Print results
    std::cout << "\nGPU Output:\n";
    for (uint32_t i = 0; i < N; i++) {
        bool bit = isBitUnset(bitpos, h_output_gpu[i]);
        std::cout << h_output_gpu[i] << "(" << (bit ? "0" : "1") << ") ";
    }
    std::cout << "\n\nNumber of zeros - CPU: " << cpu_zeros << ", GPU: " << gpu_zeros << "\n";
    
    if (cpu_zeros != gpu_zeros) {
        std::cout << "ERROR: Partition sizes don't match!\n";
        success = false;
    }
    
    // Verify that elements are properly partitioned in GPU output
    for (uint32_t i = 0; i < gpu_zeros; i++) {
        if (!isBitUnset(bitpos, h_output_gpu[i])) {
            std::cout << "ERROR: Found 1 in zeros section at position " << i << "\n";
            success = false;
            break;
        }
    }
    for (uint32_t i = gpu_zeros; i < N; i++) {
        if (isBitUnset(bitpos, h_output_gpu[i])) {
            std::cout << "ERROR: Found 0 in ones section at position " << i << "\n";
            success = false;
            break;
        }
    }
    std::cout << (success ? "Test PASSED!" : "Test FAILED!") << "\n";
    
    #endif
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

