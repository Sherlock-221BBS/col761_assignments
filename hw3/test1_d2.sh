#!/bin/bash

TEST_DATA_DIR=$1
MODEL_PATH=$2
PRED_OUTPUT_PATH=$3

DEVICE="cuda" # or "cpu"

if [ -z "$TEST_DATA_DIR" ] || [ -z "$MODEL_PATH" ] || [ -z "$PRED_OUTPUT_PATH" ]; then echo "Usage: $0 <data_dir> <model_path> <pred_out_path>"; exit 1; fi
if [ ! -f "$MODEL_PATH" ]; then echo "Error: Model file not found: $MODEL_PATH"; exit 1; fi

# --- Execution ---
echo "Starting Task 1 (d2) testing (Submission)..."
echo "Test Data Directory: ${TEST_DATA_DIR}"
echo "Model Path: ${MODEL_PATH}"
echo "Prediction Output Path: ${PRED_OUTPUT_PATH}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="${SCRIPT_DIR}/src/test_task1_d2.py" # Ensure correct python script name if different
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: Python script not found: ${PYTHON_SCRIPT}"; exit 1; fi

python "${PYTHON_SCRIPT}" \
    --data_path "${TEST_DATA_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_path "${PRED_OUTPUT_PATH}" \
    --device "${DEVICE}"

if [ $? -eq 0 ]; then echo "Testing script finished successfully."; else echo "Error: Testing script failed."; exit 1; fi
echo "Testing finished. Predicted labels saved to ${PRED_OUTPUT_PATH}"