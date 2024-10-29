CXX        = nvcc -O3

#-arch=sm_60

SRC_GPU = main-gpu.cu
HELPERS = helper*.h helper_kernels/*.cuh  
KERNELS = cub_kernel.cuh
EXEC_GPU= gpu

# New variables for experiment
EXPERIMENT_SRC = experiment.cu
EXPERIMENT_EXEC = experiment_binary

BENCH_HISTOGRAM_SRC = bench_histogram.cu
BENCH_HISTOGRAM_EXEC = bench_histogram_binary


default: compile_gpu run_gpu

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU) 


# New rule for compiling and running experiment
experiment: $(EXPERIMENT_EXEC)

histogram: $(BENCH_HISTOGRAM_EXEC)

$(BENCH_HISTOGRAM_EXEC): $(BENCH_HISTOGRAM_SRC) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(BENCH_HISTOGRAM_EXEC) $(BENCH_HISTOGRAM_SRC)
	./$(BENCH_HISTOGRAM_EXEC)
	
$(EXPERIMENT_EXEC): $(EXPERIMENT_SRC) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXPERIMENT_EXEC) $(EXPERIMENT_SRC)
	./$(EXPERIMENT_EXEC)

# we call with inputs array size, block size, items per thread, n bits per key??
run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) 100000000 

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) $(EXPERIMENT_EXEC) $(BENCH_HISTOGRAM_EXEC)
