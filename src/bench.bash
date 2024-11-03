#!/bin/bash

# Output files for results
OUTPUT_FILE="benchmarks.csv"

# Headers for CSV files
echo "data_type,dataset_size,microseconds,max_value,impl" > $OUTPUT_FILE


# Array of dataset sizes (powers of 2 for good measure)
sizes=(1000 10000 100000) #1000000 10000000 100000000)

# Fixed parameters
MAX_VALUE=100000
DATA_TYPE="u64"

# Determine which implementations to run
IMPL=${1:-"all"}  # Default to running all implementations

for size in "${sizes[@]}"; do
    # Generate dataset (needed for Futhark)
    make gen_data SIZE=$size MAX=$MAX_VALUE TYPE=$DATA_TYPE

    if [[ "$IMPL" == "all" || "$IMPL" == "futhark" ]]; then
        # Run Futhark benchmark
        runtime=$(make run_futhark TYPE=$DATA_TYPE 2>&1 | grep -v "make" | grep -v "futhark" | grep -v "cat" | tr -d '\n')
        echo "$DATA_TYPE,$size,$runtime,$MAX_VALUE,futhark" >> $OUTPUT_FILE
    fi
    
    if [[ "$IMPL" == "all" || "$IMPL" == "our" ]]; then
        # Run GPU implementation
        runtime=$(make run_gpu TYPE=$DATA_TYPE SIZE=$size IMPL=1 MAX_VAL=$MAX_VALUE | grep -v "make" | tail -n 1)
        echo "$DATA_TYPE,$size,$runtime,$MAX_VALUE,our" >> $OUTPUT_FILE
    fi
    
    if [[ "$IMPL" == "all" || "$IMPL" == "cub" ]]; then
        # Run CUB implementation
        runtime=$(make run_gpu TYPE=$DATA_TYPE SIZE=$size IMPL=0 MAX_VAL=$MAX_VALUE | grep -v "make" | tail -n 1)
        echo "$DATA_TYPE,$size,$runtime,$MAX_VALUE,cub" >> $OUTPUT_FILE
    fi
done

# Print completion messages based on what was run
if [[ "$IMPL" == "all" || "$IMPL" == "futhark" ]]; then
    echo "Futhark benchmarking complete. Results saved in $OUTPUT_FILE"
fi
if [[ "$IMPL" == "all" || "$IMPL" == "our" ]]; then
    echo "GPU benchmarking complete. Results saved in $OUTPUT_FILE"
fi
if [[ "$IMPL" == "all" || "$IMPL" == "cub" ]]; then
    echo "CUB benchmarking complete. Results saved in $OUTPUT_FILE"
fi

