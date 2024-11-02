#!/bin/bash


# Output file for results
OUTPUT_FILE_FUT="futhark_benchmarks.csv"

# Header for CSV file
echo "data_type,dataset_size,microseconds, max_value" > $OUTPUT_FILE_FUT

# Array of dataset sizes (powers of 2 for good measure)
sizes=(1000 10000 100000) #1000000 10000000 100000000)

# Fixed parameters
MAX_VALUE=100000
DATA_TYPE="u64"

for size in "${sizes[@]}"; do
    # Generate dataset
    make gen_data SIZE=$size MAX=$MAX_VALUE TYPE=$DATA_TYPE

    # Run benchmark and capture stderr (where timing info is written)
    # The benchmark outputs just the number directly
    runtime=$(make run_futhark TYPE=$DATA_TYPE 2>&1 | grep -v "make" | grep -v "futhark" | grep -v "cat" | tr -d '\n')
    
    # Save results to CSV
    echo "$DATA_TYPE,$size,$runtime,$MAX_VALUE" >> $OUTPUT_FILE_FUT
done

echo "Benchmarking complete. Results saved in $OUTPUT_FILE_FUT"

