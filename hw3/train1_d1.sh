#!/bin/bash

TRAIN_DATA_DIR=$1
MODEL_OUTPUT_PATH=$2

LEARNING_RATE="0.005"
HIDDEN_DIM="256"
NUM_LAYERS="6"
EPOCHS="1000"      
WEIGHT_DECAY="0"   
USE_LAYERNORM="True" 
DEVICE="cuda"      

if [ -z "$TRAIN_DATA_DIR" ] || [ -z "$MODEL_OUTPUT_PATH" ]; then echo "Usage: $0 <data_dir> <model_out_path>"; exit 1; fi
if [ ! -d "$TRAIN_DATA_DIR" ]; then echo "Error: Data dir not found: $TRAIN_DATA_DIR"; exit 1; fi

# --- Execution ---
echo "Starting Task 1 (d1) training (Submission)..."
echo "Data Directory: ${TRAIN_DATA_DIR}"
echo "Model Output Path: ${MODEL_OUTPUT_PATH}"
echo "Using Best Model: GraphSAGE, LR=${LEARNING_RATE}, Hidden=${HIDDEN_DIM}, Layers=${NUM_LAYERS}, Epochs=${EPOCHS}, LayerNorm=${USE_LAYERNORM}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="${SCRIPT_DIR}/src/train_task1_d1.py" 
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: Script not found: ${PYTHON_SCRIPT}"; exit 1; fi

python "${PYTHON_SCRIPT}" \
    --data_path "${TRAIN_DATA_DIR}" \
    --model_path "${MODEL_OUTPUT_PATH}" \
    --lr "${LEARNING_RATE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --num_layers "${NUM_LAYERS}" \
    --epochs "${EPOCHS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --use_layernorm "${USE_LAYERNORM}" \
    --device "${DEVICE}"

if [ $? -eq 0 ]; then echo "Training script finished successfully."; else echo "Error: Training script failed."; exit 1; fi
echo "Training finished."