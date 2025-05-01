# src/train_task2.py (Submission Version)
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import argparse
import os
import time
import numpy as np
import warnings
import traceback

# Import necessary components
from utils_task2 import load_task2_data_components_final
from model_task2 import HeteroGraphSAGE # Import the best model

def get_masks(data_dir, num_users, user_labels_tensor):
    """Determines train mask, prioritizing train_mask.npy then NaNs in labels."""
    train_mask_file = os.path.join(data_dir, 'train_mask.npy')
    train_mask = None
    test_mask = None # Not strictly needed for training, but good practice

    if os.path.exists(train_mask_file):
        print(f"Found {train_mask_file}, using it for training split.")
        train_mask_np = np.load(train_mask_file)
        if train_mask_np.shape != (num_users,) or train_mask_np.dtype != np.bool_:
             raise ValueError("train_mask.npy has incorrect shape or dtype.")
        train_mask = torch.from_numpy(train_mask_np).to(dtype=torch.bool)
        test_mask = ~train_mask # Define test as inverse
    elif user_labels_tensor is not None:
        print("train_mask.npy not found. Checking label.npy for NaNs to define split...")
        # Check row-wise for NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            is_nan_per_user = torch.isnan(user_labels_tensor).any(dim=1)
        if is_nan_per_user.any(): # If any NaNs were found
             test_mask = is_nan_per_user
             train_mask = ~test_mask
             print(f"Found {test_mask.sum()} test users (NaNs) and {train_mask.sum()} train users.")
        else:
             print("No NaNs found in label.npy and no train_mask.npy. Assuming ALL users are for training.")
             train_mask = torch.ones(num_users, dtype=torch.bool)
             test_mask = ~train_mask # No test users in this case
    else:
        print("No label.npy and no train_mask.npy found. Cannot determine training users.")
        raise FileNotFoundError("Missing label.npy or train_mask.npy needed for training.")

    if train_mask.sum() == 0:
        raise ValueError("Resulting training mask has zero users!")

    return train_mask, test_mask

def train(args):
    print("--- Task 2 Training (Submission) ---")
    print(f"Data Path: {args.data_path}")
    print(f"Save Model Path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Using BEST Params: LR={args.lr}, Hidden={args.hidden_dim}, Layers={args.num_layers}, WD={args.weight_decay}")
    print("------------------------------------")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data components...")
    start_time = time.time()
    components = load_task2_data_components_final(args.data_path)
    if components[0] is None: print("Error loading data. Exiting."); return
    user_feat, prod_feat, edge_idx_user_prod, user_labels, \
    num_users, num_products, num_labels = components
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s. Users: {num_users}, Prods: {num_products}, Labels: {num_labels}")

    # --- Determine Train/Test Masks ---
    try:
        train_mask, _ = get_masks(args.data_path, num_users, user_labels) # We only need train mask here
        print(f"Using {train_mask.sum()} users for training.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error determining training data: {e}"); return

    # --- Create HeteroData Object ---
    data = HeteroData()
    data['user'].x = user_feat
    data['product'].x = prod_feat
    # Store labels ONLY if they were loaded (needed for loss)
    if user_labels is not None:
         data['user'].y = user_labels
    else:
         print("Error: Label file (label.npy) required for training was not found or loaded.")
         return

    data['user','interacts_with','product'].edge_index = edge_idx_user_prod
    data['product','rev_interacts_with','user'].edge_index = edge_idx_user_prod.flip([0])
    # Don't add masks to the object itself for training, use the separate mask tensor
    data = data.to(device)
    train_mask = train_mask.to(device) # Move mask to device
    print("Data prepared and moved to device.")

    # --- Model, Optimizer, Loss ---
    model = HeteroGraphSAGE(hidden_channels=args.hidden_dim,
                            out_channels=num_labels,
                            num_layers=args.num_layers).to(device)
    print("Model Architecture:"); print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("\n--- Starting Training ---")
    start_train_time = time.time()
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad()
        out_logits = model(data.x_dict, data.edge_index_dict)
        # Use the determined train_mask
        loss = criterion(out_logits[train_mask], data['user'].y[train_mask])
        loss.backward(); optimizer.step()
        if (epoch + 1) % 50 == 0 or epoch == 0: print(f'Epoch {epoch+1:04d}/{args.epochs:04d} | Loss: {loss.item():.4f}')

    train_time = time.time() - start_train_time
    print(f"--- Training Finished in {train_time:.2f} seconds ---")
    print(f"Final Loss: {loss.item():.4f}")

    print(f"\nSaving final model state dictionary to {args.model_path}...")
    try:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_path)
        print("Model saved successfully.")
    except Exception as e: print(f"Error saving model: {e}"); traceback.print_exc()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HeteroGraphSAGE for Task 2 (Submission).')
    parser.add_argument('--data_path',type=str,required=True, help='Path to Task 2 data dir')
    parser.add_argument('--model_path',type=str,required=True, help='Path to save trained model')
    # Set defaults to best found params
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--epochs',type=int,default=500, help='Training epochs') # Adjust as needed
    parser.add_argument('--lr',type=float,default=0.005, help='Learning rate')
    parser.add_argument('--hidden_dim',type=int,default=128, help='Hidden units')
    parser.add_argument('--num_layers',type=int,default=2, help='Num GNN layers')
    parser.add_argument('--weight_decay',type=float,default=0, help='Weight decay')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    train(args)
    print("\nTrain script finished.")