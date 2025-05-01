# src/train_task1_d2.py

import torch
import torch.nn.functional as F
# Import necessary layers
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, LayerNorm # Keep imports for flexibility if needed
import argparse
import os
import time
import traceback
import numpy as np # Needed by utils_task1


# Use the Task 1 utility which handles NaN labels for splitting
# Ensure this file exists in the same directory or adjust path
try:
    from utils_task1 import load_graph_data
except ImportError:
    print("Error: Cannot find utils_task1.py. Make sure it's in the same directory (src/).")
    exit(1)
  
# --- Model Definition (Only GraphSAGE needed based on train1_d2.sh) ---
class GraphSAGE(torch.nn.Module):
    # Using default parameters matching common practice (2 layers, no layernorm by default)
    # Make sure this matches the intended structure if num_layers/layernorm were tuned
    def __init__(self, num_node_features, num_hidden, num_classes, num_layers=2, dropout_p=0.5, use_layer_norm=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers # Store num_layers
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
        if self.num_layers == 1: # Handle single-layer case
            return self.convs[0](x, edge_index)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_layer_norm and self.norms is not None and i < len(self.norms):
                 x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.convs[-1](x, edge_index) # Last layer
        return x

# --- Main Training Function ---
def train(args):
    print("--- Task 1 / d2 Training (Submission) ---")
    print(f"Data Path: {args.data_path}")
    print(f"Save Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type}") # Should be graphsage
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Hidden Dimensions: {args.hidden_dim}")
    print(f"Num Layers: {args.num_layers}") # Added num_layers arg
    print(f"Use LayerNorm: {args.use_layernorm}") # Added layernorm arg
    print(f"Weight Decay: {args.weight_decay}")
    print("----------------------------------------")

    # --- Set Seed (Optional but good for reproducibility) ---
    # from your_seed_util import set_seed # If you have a seed utility
    # set_seed(42) # Example seed

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    start_time = time.time()
    data, num_classes = load_graph_data(args.data_path)
    if data is None: print("Error loading data. Exiting."); return
    if num_classes is None or num_classes <= 0: print(f"Error: Invalid classes ({num_classes}). Exiting."); return
    print(f"Detected {num_classes} classes for d2.")
    data = data.to(device)
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s. Nodes: {data.num_nodes}, Train nodes: {data.train_mask.sum()}")

    # --- Model Instantiation ---
    # Assume only GraphSAGE needed based on the shell script
    if args.model_type.lower() == 'graphsage':
        model = GraphSAGE(num_node_features=data.num_node_features,
                          num_hidden=args.hidden_dim,
                          num_classes=num_classes,
                          num_layers=args.num_layers, # Pass num_layers
                          use_layer_norm=args.use_layernorm # Pass layernorm flag
                          ).to(device)
    else:
        # If you need flexibility, add GCN/GAT here too
        print(f"Error: This script expects --model_type graphsage based on train1_d2.sh, but got {args.model_type}")
        return
    print("Model Architecture:"); print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    start_train_time = time.time()
    train_mask_dev = data.train_mask

    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad()
        # Pass features and edge_index separately
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask_dev], data.y[train_mask_dev])
        loss.backward(); optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == args.epochs - 1:
             print(f'Epoch {epoch+1:04d}/{args.epochs:04d} | Loss: {loss.item():.4f}')

    train_time = time.time() - start_train_time
    print(f"--- Training Finished in {train_time:.2f} seconds ---")
    print(f"Final Loss: {loss.item():.4f}")

    print(f"\nSaving final model state dictionary to {args.model_path}...")
    try:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        # Save config along with state_dict for easier loading in test script
        save_obj = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_type': 'graphsage',
                'num_node_features': data.num_node_features,
                'num_classes': num_classes,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'use_layernorm': args.use_layernorm
            }
        }
        torch.save(save_obj, args.model_path)
        print("Model and config saved successfully.")
    except Exception as e: print(f"Error saving model: {e}"); traceback.print_exc()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GraphSAGE for Task 1 / d2 (Submission).')
    parser.add_argument('--data_path',type=str,required=True, help='Path to Task 1/d2 data dir')
    parser.add_argument('--model_path',type=str,required=True, help='Path to save trained model')
    # Arguments passed from train1_d2.sh
    parser.add_argument('--model_type',type=str,default='graphsage', help='GNN model type (should be graphsage)')
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--epochs',type=int,default=2253, help='Training epochs') # Default from your sh script
    parser.add_argument('--lr',type=float,default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim',type=int,default=128, help='Hidden units')
    parser.add_argument('--weight_decay',type=float,default=5e-4, help='Weight decay')
    # Add arguments for model architecture details if they vary
    parser.add_argument('--num_layers', type=int, default=2, help='Num GNN layers (set default to match GraphSAGE if needed)')
    parser.add_argument('--use_layernorm', type=bool, default=False, help='Use LayerNorm (set default based on best config)')

    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'

    # Ensure model type is correct for this script
    if args.model_type.lower() != 'graphsage':
        print(f"Warning: This script is intended for GraphSAGE, but model_type '{args.model_type}' was provided.")
        # You could choose to exit, or proceed if the definition exists above

    train(args)
    print("\nTrain script finished.")