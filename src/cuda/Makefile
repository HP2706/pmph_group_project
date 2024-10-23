CXX        = nvcc -O3

#-arch=sm_60

SRC_GPU = main-gpu.cu
HELPERS = helper.h
KERNELS = cub_kernel.cuh
EXEC_GPU= mmm-gpu


default: compile_gpu run_gpu

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU) 


# we call with inputs array size, block size, items per thread, n bits per key??
run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 100000000 

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU)
