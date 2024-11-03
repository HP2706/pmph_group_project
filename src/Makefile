CXX        = nvcc -O3

#-arch=sm_60

SRC_GPU = cuda/main-gpu.cu
HELPERS = cuda/helper*.h cuda/helper_kernels/*.cuh cuda/constants.cuh
KERNELS = cuda/cub_kernel.cuh
EXEC_GPU= gpu


BENCH_HISTOGRAM_SRC = cuda/bench_histogram.cu
BENCH_HISTOGRAM_EXEC = bench_histogram_binary

# Add these definitions near the top with other variables
TEST_SRC = cuda/tests.cu
TEST_EXEC = test_binary

default: compile_gpu run_gpu

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile_gpu: $(EXEC_GPU)

$(EXEC_GPU): $(SRC_GPU) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(EXEC_GPU) $(SRC_GPU) 

# New rule for compiling and running test
test: $(TEST_EXEC)

histogram: $(BENCH_HISTOGRAM_EXEC)

$(BENCH_HISTOGRAM_EXEC): $(BENCH_HISTOGRAM_SRC) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(BENCH_HISTOGRAM_EXEC) $(BENCH_HISTOGRAM_SRC)
	./$(BENCH_HISTOGRAM_EXEC)
	
$(TEST_EXEC): $(TEST_SRC) $(HELPERS) $(KERNELS)
	nvcc -O3 -o $(TEST_EXEC) $(TEST_SRC)
	./$(TEST_EXEC)

# we call with inputs array size, block size, items per thread, n bits per key??

SIZE ?= 10000
TYPE ?= u32
IMPL ?= 1
MAX_VAL ?= 10000
run_gpu: $(EXEC_GPU)
	./$(EXEC_GPU) $(TYPE) $(SIZE) $(IMPL) $(MAX_VAL)

clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) $(TEST_EXEC) $(BENCH_HISTOGRAM_EXEC)
	rm -f array.in
	rm -f bench_futhark


#futhark benchmarks
SIZE ?= 100
MAX ?= 10
TYPE ?= u32

#baaah
gen_data:
	futhark dataset --$(TYPE)-bounds=0:$(MAX) -b -g [$(SIZE)]$(TYPE) > array.in

TYPE ?= u32
run_futhark:
	@if [ "$(TYPE)" = "u32" ]; then \
		futhark cuda futhark/bench_u32.fut -o bench_futhark; \
	elif [ "$(TYPE)" = "u64" ]; then \
		futhark cuda futhark/bench_u64.fut -o bench_futhark; \
	elif [ "$(TYPE)" = "u16" ]; then \
		futhark cuda futhark/bench_u16.fut -o bench_futhark; \
	elif [ "$(TYPE)" = "u8" ]; then \
		futhark cuda futhark/bench_u8.fut -o bench_futhark; \
	fi
	@cat array.in | ./bench_futhark -t /dev/stderr > /dev/null
