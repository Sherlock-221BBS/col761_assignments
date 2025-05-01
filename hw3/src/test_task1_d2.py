# src/test_task1_d2.py (Submission Version)

import torch
import torch.nn.functional as F
# Only import GraphSAGE and LayerNorm (if used)
from torch_geometric.nn import SAGEConv, LayerNorm
import argparse
import os
import pandas as pd
import numpy as np
import time
import traceback

# Use the Task 1 utility which handles NaN labels for splitting
try:
    from utils_task1 import load_graph_data
except ImportError:
    print("Error: Cannot find utils_task1.py. Make sure it's in the same directory (src/).")
    exit(1)

# --- Model Definition (MUST MATCH SAVED MODEL) ---
# This definition needs to match the one used in training exactly
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes, num_layers=2, dropout_p=0.5, use_layer_norm=False):
        super().__init__()
        self.dropout_p = dropout_p # Store dropout even if not used in eval
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
             self.convs.append(SAGEConv(num_node_features, num_classes, aggr='mean'))
        else:
             self.convs.append(SAGEConv(num_node_features, num_hidden, aggr='mean'))
             if use_layer_norm: self.norms.append(LayerNorm(num_hidden))
             for _ in range(num_layers - 2):
                  self.convs.append(SAGEConv(num_hidden, num_hidden, aggr='mean'))
                  if use_layer_norm: self.norms.append(LayerNorm(num_hidden))
             self.convs.append(SAGEConv(num_hidden, num_classes, aggr='mean'))

    def forward(self, x, edge_index): # Expects x, edge_index
        if self.num_layers == 1:
            return self.convs[0](x, edge_index)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_layer_norm and self.norms is not None and i < len(self.norms):
                 x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training) # Inactive in eval mode
        x = self.convs[-1](x, edge_index) # Last layer
        return x

# --- Main Testing Function ---
def test(args):
    print("--- Task 1 / d2 Testing (Submission - Output Labels) ---")
    print(f"Test Data Path: {args.data_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Output CSV Path: {args.output_path}")
    print(f"Device: {args.device}")
    print("--------------------------------------------------------")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading test data...")
    start_time = time.time()
    # Load data - we only strictly need x, edge_index, and test_mask
    # load_graph_data also returns num_classes, which we can ignore as it's in saved model config
    data, _ = load_graph_data(args.data_path)
    if data is None: print("Error loading data. Exiting."); return
    if not hasattr(data, 'test_mask') or data.test_mask is None: print("Error: test_mask not found."); return

    data = data.to(device)
    load_time = time.time() - start_time
    test_mask_dev = data.test_mask
    num_test_nodes = test_mask_dev.sum().item()
    print(f"Data loaded in {load_time:.2f}s. Num test nodes: {num_test_nodes}")
    if num_test_nodes == 0: print("Warning: No test nodes found. Saving empty file."); pd.DataFrame().to_csv(args.output_path, index=False, header=False); return

    print("Loading model and config from file...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device) # Load checkpoint to target device
        if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
             raise KeyError("Saved checkpoint missing 'config' or 'model_state_dict' key.")

        config = checkpoint['config']
        print(f"Loaded model config: {config}")

        # Verify model type matches expectation
        if config.get('model_type', 'graphsage').lower() != 'graphsage':
             print(f"Warning: Loaded model config type is '{config.get('model_type')}', but expected 'graphsage'.")

        # Instantiate model using loaded config
        model = GraphSAGE(num_node_features=config['num_node_features'],
                          num_hidden=config['hidden_dim'],
                          num_classes=config['num_classes'],
                          num_layers=config.get('num_layers', 2), # Default to 2 if missing
                          use_layer_norm=config.get('use_layernorm', False) # Default to False if missing
                         ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully.")

    except FileNotFoundError: print(f"Error: Model file not found: {args.model_path}"); return
    except KeyError as e: print(f"Error: Missing key {e} in saved model file. Ensure config was saved correctly during training."); traceback.print_exc(); return
    except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc(); return

    model.eval() # Set to evaluation mode

    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        # Pass x and edge_index separately
        all_logits = model(data.x, data.edge_index)
        test_logits = all_logits[test_mask_dev] # Select test node logits
    infer_time = time.time() - start_time
    print(f"Inference completed in {infer_time:.2f} seconds.")
    print(f"Generated test logits shape: {test_logits.shape}")

    # --- Prepare Output (Predicted Labels for d2) ---
    predictions = test_logits.argmax(dim=1)
    print(f"Generated predicted labels shape: {predictions.shape}") # Should be (num_test_nodes,)

    print(f"Saving predicted labels to {args.output_path}...")
    try:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        predictions_np = predictions.cpu().numpy()
        # Ensure single column as requested by assignment text (despite d1 example)
        df_output = pd.DataFrame(predictions_np.reshape(-1, 1))
        df_output.to_csv(args.output_path, index=False, header=False)
        print(f"Predicted labels saved successfully. Shape: {df_output.shape}")
    except Exception as e: print(f"Error saving predictions: {e}"); traceback.print_exc()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GraphSAGE for Task 1 / d2 (Submission).')
    parser.add_argument('--data_path',type=str,required=True, help='Path to Task 1/d2 data dir')
    parser.add_argument('--model_path',type=str,required=True, help='Path to saved trained model (.pth including config)')
    parser.add_argument('--output_path',type=str,required=True, help='Path to save output CSV')
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    # No need to pass hidden_dim, num_layers etc. as they are loaded from the model file
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    test(args)
    print("\nTest script finished.")