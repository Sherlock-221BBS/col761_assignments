#!/bin/bash

apriori_path="$1"
fpgrowth_path="$2"
input_path=$3
output_path=$4

apriori_path=$(realpath "$apriori_path")
fpgrowth_path=$(realpath "$fpgrowth_path")
input_path=$(realpath "$input_path")
output_path=$(realpath "$output_path")

#=================================
log_path=$output_path/run_log.txt
mkdir -p $output_path
#delete log file if exist
if [ -f "$log_path" ]; then
    echo "File exists. Deleting..."
    rm "$log_path"
fi

# Create a new file
touch "$log_path"
echo "New file created: $log_path"
#=================================


threshold_list=(90 50 25 10 5)

for support in "${threshold_list[@]}"; do
    apriori_file="$output_path/ap$support"
    fpgrowth_file="$output_path/fp$support"
    touch $apriori_file
    touch $fpgrowth_file

    echo "Running Apriori for support $support%"
    start_time=$(date +%s)
    timeout 3600 "$apriori_path" -s$support "$input_path" "$apriori_file"
    status=$? || true
    end_time=$(date +%s)
    apriori_time=$((end_time - start_time))
    echo "Apriori runtime at $support% support: $apriori_time seconds" >> "$log_path"
    if [ $status -eq 124 ]; then
        echo "Apriori timeout 1hr" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 139 ]; then
        echo "Apriori crashed with segmentation fault (SIGSEGV)" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 137 ]; then
        echo "Apriori was killed (SIGKILL)" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 143 ]; then
        echo "Apriori was terminated (SIGTERM)" | tee -a "$log_path"
        > "$output_file"
    fi

    echo "Running FP-growth for support $support%"
    start_time=$(date +%s)
    timeout 3600 "$fpgrowth_path" -s$support "$input_path" "$fpgrowth_file"
    status=$? || true
    end_time=$(date +%s)
    fpgrowth_time=$((end_time- start_time))
    echo "FP-growth runtime at $support% support: $fpgrowth_time seconds" >> "$log_path"
    if [ $status -eq 124 ]; then
        echo "FP-growth timeout 1hr" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 139 ]; then
        echo "FP-growth crashed with segmentation fault (SIGSEGV)" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 137 ]; then
        echo "FP-growth was killed (SIGKILL)" | tee -a "$log_path"
        > "$output_file"
    elif [ $status -eq 143 ]; then
        echo "FP-growth was terminated (SIGTERM)" | tee -a "$log_path"
        > "$output_file"
    fi
done

python3 q1.py $output_path