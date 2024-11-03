#!/bin/bash

# Fixed parameters
# Modified check for DATA_TYPE variable
DATA_TYPE=${1:-"u64"}  # Use command line arg if env var not set
IMPL=${2:-"all"}  # Move IMPL to second argument

# Output files for results
OUTPUT_FILE="benchmarks_${DATA_TYPE}.csv"

# Headers for CSV files
echo "data_type,dataset_size,microseconds,max_value,impl" > $OUTPUT_FILE


# Array of dataset sizes (powers of 2 for good measure)
sizes=(1000 10000 100000 1000000 10000000 100000000 200000000 400000000 800000000 1000000000 2000000000)

MAX_VAL_U64=18446744073709551615
MAX_VAL_U32=4294967295
MAX_VAL_U16=65535
MAX_VAL_U8=255


if [ -z "$DATA_TYPE" ]; then
    echo "ERROR: DATA_TYPE must be provided either as environment variable or first argument"
    echo "Usage: bash bench.bash <data_type> [impl]"
    echo "Example: bash bench.bash u32 all"
    exit 1
fi

# Fix comparison operators
if [ "$DATA_TYPE" == "u64" ]; then
    MAX_VALUE=$MAX_VAL_U64
elif [ "$DATA_TYPE" == "u32" ]; then
    MAX_VALUE=$MAX_VAL_U32
elif [ "$DATA_TYPE" == "u16" ]; then
    MAX_VALUE=$MAX_VAL_U16
else
    MAX_VALUE=$MAX_VAL_U8
fi


for size in "${sizes[@]}"; do
    # Generate dataset (needed for Futhark)

    if [[ "$IMPL" == "all" || "$IMPL" == "futhark" ]]; then
        # Run Futhark benchmark
        # only run if size < 200000000 otherwise it will give an error
        if [ $size -lt 200000000 ]; then
            make gen_data SIZE=$size MAX=$MAX_VALUE TYPE=$DATA_TYPE
            runtime=$(make run_futhark TYPE=$DATA_TYPE 2>&1 | grep -v "make" | grep -v "futhark" | grep -v "cat" | tr -d '\n')
            echo "$DATA_TYPE,$size,$runtime,$MAX_VALUE,futhark" >> $OUTPUT_FILE
        fi
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

