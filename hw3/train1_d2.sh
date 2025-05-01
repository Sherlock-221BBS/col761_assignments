#!/bin/bash

TRAIN_DATA_DIR=$1
MODEL_OUTPUT_PATH=$2

MODEL_TYPE="graphsage"
LEARNING_RATE="0.01"
HIDDEN_DIM="128"
EPOCHS="2500"
WEIGHT_DECAY="5e-4" 
DEVICE="cuda"

if [ -z "$TRAIN_DATA_DIR" ] || [ -z "$MODEL_OUTPUT_PATH" ]; then echo "Usage: $0 <data_dir> <model_out_path>"; exit 1; fi
if [ ! -d "$TRAIN_DATA_DIR" ]; then echo "Error: Data dir not found: $TRAIN_DATA_DIR"; exit 1; fi

echo "Starting Task 1 (d2) training..."
echo "Data Directory: ${TRAIN_DATA_DIR}"
echo "Model Output Path: ${MODEL_OUTPUT_PATH}"
echo "Using Model: ${MODEL_TYPE}, LR: ${LEARNING_RATE}, Hidden: ${HIDDEN_DIM}, Epochs: ${EPOCHS}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="${SCRIPT_DIR}/src/train_task1_d2.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: Python script not found: ${PYTHON_SCRIPT}"; exit 1; fi

python "${PYTHON_SCRIPT}" \
    --data_path "${TRAIN_DATA_DIR}" \
    --model_path "${MODEL_OUTPUT_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --lr "${LEARNING_RATE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --epochs "${EPOCHS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --device "${DEVICE}"

if [ $? -eq 0 ]; then echo "Training script finished successfully."; else echo "Error: Training script failed."; exit 1; fi
echo "Training finished."