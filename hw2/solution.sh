#!/bin/bash

start_time=$(date +%s)
# Activate Conda Environment
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "whocares"; then
    echo "Creating Conda environment 'whocares'..."
    conda create -n whocares python=3.10 -y
else
    echo "Conda environment 'whocares' already exists. Skipping creation."
fi

conda activate whocares
chmod +x ./src/infection

pip install torch torch_geometric scikit-learn numpy matplotlib networkx

DATASET_FILE="$1"
OUTPUT_FILE="$2"
K=$3

GLOBAL_BEST_SPREAD=0
GLOBAL_BEST_FILE="$OUTPUT_FILE"
rm -f "$GLOBAL_BEST_FILE" 

declare -A param_grids
param_grids["algo3"]="5"
param_grids["pagerank"]="0.8 1000 1e-6 0.8 1000 1e-5 0.85 1000 1e-6 0.85 1000 1e-5 0.9 1000 1e-6 0.9 1000 1e-5 0.95 1000 1e-6 0.95 1000 1e-5"
param_grids["betweenness"]="1000"
param_grids["imm"]="0.05 1e-3 0.05 1e-4 0.01 1e-3 0.01 1e-4"



for algo in "${!param_grids[@]}"; do
    echo "Running $algo..."

    params=(${param_grids[$algo]})
    index=0
    best_spread=0 
    best_file="best_${algo}.txt"
    temp_file="temp_${algo}.txt"

    rm -f "$best_file"

    params=(${param_grids[$algo]}) 
    index=0

    while [ $index -lt ${#params[@]} ]; do
        param_json=""

        case $algo in
            "algo3") 
                param_json="{\"top_k\": ${params[$index]}}" 
                index=$((index+1)) ;;
            "iim") 
                param_json="{\"iterations\": ${params[$index]}}"
                index=$((index+1)) ;;  
            "stop_and_stare") 
                param_json="{\"lookahead\": ${params[$index]}}"
                index=$((index+1)) ;;  
            "pagerank") 
                param_json="{\"alpha\": ${params[$index]}, \"max_iter\": ${params[$index+1]}, \"tol\": ${params[$index+2]}}"
                index=$((index+2)) ;;  
            "betweenness") 
                param_json="{\"k_sample\": ${params[$index]}}" 
                index=$((index+1)) ;;
            "imm") 
                param_json="{\"epsilon\": ${params[$index]}, \"delta\": ${params[$index+1]}}"
                index=$((index+1)) ;;    
        esac

        echo "Running $algo with hyperparameters: $param_json"
        echo "python3 ./src/main.py "$DATASET_FILE" "$algo" "$param_json" "$K" "$temp_file" "
        python3 ./src/main.py "$DATASET_FILE" "$algo" "$param_json" "$K" "$temp_file"

        status=$? || true
        if [ $status -eq 124 ]; then
            echo "Timeout occurred for $algo with params: $param_json"
        fi

        if [ -f "$temp_file" ]; then
            result=$(./src/infection "$DATASET_FILE" "$temp_file")
            spread=$(echo "$result" | grep "Spread:" | awk '{print $2}') 
            echo $result

            if [[ "$spread" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                if (( $(echo "$spread > $best_spread" | bc -l) )); then
                    best_spread=$spread
                    cp "$temp_file" "$best_file"
                    echo "Updated best spread for $algo: $best_spread"
                fi
                if (( $(echo "$spread > $GLOBAL_BEST_SPREAD" | bc -l) )); then
                    GLOBAL_BEST_SPREAD=$spread
                    cp "$best_file" "$GLOBAL_BEST_FILE"
                    echo "Updated GLOBAL best spread: $GLOBAL_BEST_SPREAD"
                fi
            else
                echo "Invalid spread value in $temp_file"
            fi

            

        fi

        
        echo "Completed $algo with hyperparameters: $param_json with spread: $spread"
        index=$((index+1))
    done

done

end_time=$(date +%s)
elapsed_time=$(( (end_time - start_time) / 60 ))  # Convert seconds to minutes

echo "Execution time: ${elapsed_time} minutes"

# Directory containing best seed files
BEST_SEED_DIR="./"
OUTPUT_DIR="./permuted_seeds"
mkdir -p "$OUTPUT_DIR"

# Output log file
RESULT_LOG="permutation_results.txt"
echo "Permutation Results" > "$RESULT_LOG"

# Get all best seed files
SEED_FILES=($(ls "$BEST_SEED_DIR"/best_*.txt))

# Read all seeds from best seed files
declare -A SEED_COUNT
declare -A SEED_SOURCE

for FILE in "${SEED_FILES[@]}"; do
    while read -r SEED; do
        SEED_COUNT["$SEED"]=$((SEED_COUNT["$SEED"] + 1))
        SEED_SOURCE["$SEED"]+="$FILE "
    done < "$FILE"
done

# Get common seeds (appearing in all files)
TOTAL_FILES=${#SEED_FILES[@]}
COMMON_SEEDS=()
DIFFERENT_SEEDS=()



echo "running stopandstare"
# Combine all seeds from all files
COMBINED_SEEDS=()
for FILE in "${SEED_FILES[@]}"; do
    while read -r SEED; do
        COMBINED_SEEDS+=("$SEED")
    done < "$FILE"
done

# Remove duplicates
UNIQUE_SEEDS=($(echo "${COMBINED_SEEDS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))


current_time=$(date +%s)
elapsed_time=$((current_time - start_time))

# Calculate remaining time from 45 minutes
remaining_time=$((2700 - elapsed_time))


# Call main.py with combined seeds
echo "Running stop_and_stare mode... with time limit $remaining_time"
param_json="{\"lookahead\": 20}"
python3 ./src/main.py "$DATASET_FILE" stop_and_stare "$param_json" 50 "$OUTPUT_FILE"


end_time=$(date +%s)  # Get end time in seconds

elapsed_time=$(( (end_time - start_time) / 60 ))  # Convert seconds to minutes

echo "Execution time: ${elapsed_time} minutes"