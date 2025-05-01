# src/test_task1_d1.py (Submission for Task 1 / d1 - Output Logits - FIXED)

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm # Only need these
import argparse
import os
import pandas as pd
import numpy as np
import time
import traceback

try:
    from utils_task1 import load_graph_data
except ImportError:
    print("Error: Cannot find utils_task1.py. Make sure it's in the same directory (src/).")
    exit(1)

# --- Model Definition (MUST MATCH SAVED MODEL ARCHITECTURE) ---
# This definition needs to handle variable layers and layernorm correctly
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes, num_layers=3, dropout_p=0.5, use_layer_norm=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if use_layer_norm else None

        if num_layers < 1: raise ValueError("Number of layers must be at least 1")

        current_dim = num_node_features
        for i in range(num_layers - 1): # All layers except the last one
             self.convs.append(SAGEConv(current_dim, num_hidden, aggr='mean'))
             if use_layer_norm:
                 # Add LayerNorm except after the layer directly feeding the output layer
                 self.norms.append(LayerNorm(num_hidden))
             current_dim = num_hidden # Input for next layer is hidden_dim

        # Output layer
        self.convs.append(SAGEConv(current_dim, num_classes, aggr='mean'))

    def forward(self, x, edge_index): # Expects x, edge_index
        for i, conv in enumerate(self.convs[:-1]): # Iterate through hidden layers
            x = conv(x, edge_index)
            # Apply norm if it exists for this layer index
            if self.use_layer_norm and self.norms is not None and i < len(self.norms):
                 x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training) # Inactive in eval mode
        x = self.convs[-1](x, edge_index) # Apply final layer
        return x

# --- Main Testing Function ---
def test(args):
    print("--- Task 1 / d1 Testing (Submission - Output Logits) ---")
    print(f"Test Data Path: {args.data_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Output CSV Path: {args.output_path}")
    print(f"Device: {args.device}")
    print("--------------------------------------------------------")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    print("Loading test data..."); start_time = time.time()
    data, _ = load_graph_data(args.data_path)
    if data is None: print("Error loading data. Exiting."); return
    if not hasattr(data, 'test_mask') or data.test_mask is None: print("Error: test_mask not found."); return
    data = data.to(device); load_time = time.time() - start_time
    test_mask_dev = data.test_mask; num_test_nodes = test_mask_dev.sum().item()
    print(f"Data loaded in {load_time:.2f}s. Num test nodes: {num_test_nodes}")
    if num_test_nodes == 0: print("Warning: No test nodes. Saving empty file."); pd.DataFrame().to_csv(args.output_path, index=False, header=False); return

    print("Loading model and config from file...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'config' not in checkpoint or 'model_state_dict' not in checkpoint: raise KeyError("Missing keys")

        config = checkpoint['config']
        print(f"Loaded model config: {config}")

        # Verify model type matches expectation (optional but good)
        loaded_model_type = config.get('model_type', 'graphsage').lower()
        if loaded_model_type != 'graphsage': print(f"Warning: Loaded model type is '{loaded_model_type}', but expected 'graphsage'.")

        model = GraphSAGE(num_node_features=config['num_node_features'],
                          num_hidden=config['hidden_dim'],
                          num_classes=config['num_classes'],
                          num_layers=config.get('num_layers', 2),
                          use_layer_norm=config.get('use_layernorm', False)
                         ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model architecture instantiated from config and weights loaded successfully.")

    except FileNotFoundError: print(f"Error: Model file not found: {args.model_path}"); return
    except KeyError as e: print(f"Error: Missing key {e} in saved model file. Ensure config was saved correctly during training."); traceback.print_exc(); return
    except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc(); return

    model.eval()
    print("Running inference..."); start_time = time.time()
    with torch.no_grad():
        all_logits = model(data.x, data.edge_index)
        test_logits = all_logits[test_mask_dev]
    infer_time = time.time() - start_time; print(f"Inference completed in {infer_time:.2f} seconds.")
    print(f"Generated test logits shape: {test_logits.shape}")

    print(f"Saving logits to {args.output_path}...")
    try:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        predictions_np = test_logits.cpu().numpy()
        df_output = pd.DataFrame(predictions_np)
        df_output.to_csv(args.output_path, index=False, header=False, float_format='%.8f')
        print(f"Logits saved successfully. Shape: {df_output.shape}")
    except Exception as e: print(f"Error saving predictions: {e}"); traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GraphSAGE for Task 1 / d1 (Submission - Output Logits).')
    parser.add_argument('--data_path',type=str,required=True, help='Path to Task 1/d1 data dir')
    parser.add_argument('--model_path',type=str,required=True, help='Path to saved trained model (.pth including config)')
    parser.add_argument('--output_path',type=str,required=True, help='Path to save output CSV')
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    test(args)
    print("\nTest script finished.")