# src/train_task1_d1.py (Submission for Task 1 / d1)

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm # Only need these
import argparse
import os
import time
import traceback
import numpy as np

# Use the Task 1 utility which handles NaN labels for splitting
try:
    from utils_task1 import load_graph_data
except ImportError:
    print("Error: Cannot find utils_task1.py. Make sure it's in the same directory (src/).")
    exit(1)

# --- Model Definition (GraphSAGE - matching best config) ---
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden, num_classes, num_layers=3, dropout_p=0.5, use_layer_norm=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if use_layer_norm else None

        if num_layers == 1:
             self.convs.append(SAGEConv(num_node_features, num_classes, aggr='mean'))
        else:
             self.convs.append(SAGEConv(num_node_features, num_hidden, aggr='mean'))
             if use_layer_norm: self.norms.append(LayerNorm(num_hidden)) # Norm after first layer if LN enabled
             for _ in range(num_layers - 2):
                  self.convs.append(SAGEConv(num_hidden, num_hidden, aggr='mean'))
                  if use_layer_norm: self.norms.append(LayerNorm(num_hidden)) # Norms after hidden layers
             self.convs.append(SAGEConv(num_hidden, num_classes, aggr='mean')) # Final layer

    def forward(self, x, edge_index): # Expects x, edge_index
        if self.num_layers == 1:
            return self.convs[0](x, edge_index)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # Check norms existence based on index before applying
            if self.use_layer_norm and self.norms is not None and i < len(self.norms):
                 x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.convs[-1](x, edge_index) # Last layer
        return x

# --- Main Training Function ---
def train(args):
    print("--- Task 1 / d1 Training (Submission) ---")
    print(f"Data Path: {args.data_path}")
    print(f"Save Model Path: {args.model_path}")
    print(f"Using Model: GraphSAGE")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}, Hidden: {args.hidden_dim}, Layers: {args.num_layers}")
    print(f"LayerNorm: {args.use_layernorm}, Weight Decay: {args.weight_decay}")
    print("----------------------------------------")

    # --- Set Seed ---
    # from your_seed_util import set_seed # If you have a seed utility
    # set_seed(args.seed) # Use a fixed seed for reproducibility

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    start_time = time.time()
    data, num_classes = load_graph_data(args.data_path)
    if data is None: print("Error loading data. Exiting."); return
    if num_classes is None or num_classes <= 0: print(f"Error: Invalid classes ({num_classes}). Exiting."); return
    print(f"Detected {num_classes} classes for d1.")
    data = data.to(device)
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s. Nodes:{data.num_nodes}, Features:{data.num_node_features}, Train nodes:{data.train_mask.sum()}")

    # --- Instantiate the BEST model configuration ---
    model = GraphSAGE(num_node_features=data.num_node_features,
                      num_hidden=args.hidden_dim,
                      num_classes=num_classes,
                      num_layers=args.num_layers,
                      use_layer_norm=args.use_layernorm).to(device)
    print("Model Architecture:"); print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    start_train_time = time.time()
    train_mask_dev = data.train_mask

    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad()
        out = model(data.x, data.edge_index) # Pass x, edge_index
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
        # Save config along with state_dict
        save_obj = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_type': 'graphsage', # Hardcode since it's fixed for this script
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
    parser = argparse.ArgumentParser(description='Train GraphSAGE for Task 1 / d1 (Submission).')
    parser.add_argument('--data_path',type=str,required=True, help='Path to Task 1/d1 data dir')
    parser.add_argument('--model_path',type=str,required=True, help='Path to save trained model')
    # Use Fixed optimal parameters for submission
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--epochs',type=int,default=500, help='Training epochs') # Adjust based on best AUC epoch
    parser.add_argument('--lr',type=float,default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim',type=int,default=256, help='Hidden units')
    parser.add_argument('--num_layers',type=int,default=3, help='Num GNN layers')
    parser.add_argument('--use_layernorm',type=bool,default=False, help='Use LayerNorm')
    parser.add_argument('--weight_decay',type=float,default=0, help='Weight decay')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed') # Add if using set_seed
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    train(args)
    print("\nTrain script finished.")