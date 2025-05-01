# src/test_task2.py (Submission Version)
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import argparse
import os
import pandas as pd
import numpy as np
import time
import warnings
import traceback

from utils_task2 import load_task2_data_components_final
from model_task2 import HeteroGraphSAGE # Import the specific model used

def get_test_mask(data_dir, num_users, user_labels_tensor):
    """Determines test mask, prioritizing test_mask.npy, then NaNs, then all users."""
    test_mask_file = os.path.join(data_dir, 'test_mask.npy')
    test_mask = None

    if os.path.exists(test_mask_file):
        print(f"Found {test_mask_file}, using it for testing split.")
        test_mask_np = np.load(test_mask_file)
        if test_mask_np.shape != (num_users,) or test_mask_np.dtype != np.bool_:
             raise ValueError("test_mask.npy has incorrect shape or dtype.")
        test_mask = torch.from_numpy(test_mask_np).to(dtype=torch.bool)
    elif user_labels_tensor is not None:
        print("test_mask.npy not found. Checking label.npy for NaNs...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            is_nan_per_user = torch.isnan(user_labels_tensor).any(dim=1)
        if is_nan_per_user.any():
             test_mask = is_nan_per_user
             print(f"Using {test_mask.sum()} users with NaN labels as test set.")
        else:
             print("No test_mask.npy or NaNs found. Assuming ALL users are test users.")
             test_mask = torch.ones(num_users, dtype=torch.bool)
    else:
        print("No test_mask.npy or label.npy found. Assuming ALL users are test users.")
        test_mask = torch.ones(num_users, dtype=torch.bool)

    if test_mask.sum() == 0:
        print("Warning: Resulting test mask has zero users!")

    return test_mask


def test(args):
    print("--- Task 2 Testing (Submission) ---")
    print(f"Test Data Path: {args.data_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Output CSV Path: {args.output_path}")
    print(f"Using Model Config: Type={args.model_type}, Hidden={args.hidden_dim}, Layers={args.num_layers}")
    print(f"Device: {args.device}")
    print("------------------------------------")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data components...")
    start_time = time.time()
    components = load_task2_data_components_final(args.data_path)
    if components[0] is None: print("Error loading data. Exiting."); return
    user_feat, prod_feat, edge_idx_user_prod, user_labels, \
    num_users, num_products, num_labels_loaded = components # num_labels might be None
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s. Users: {num_users}, Prods: {num_products}")

    # --- Determine Test Mask ---
    try:
        test_mask = get_test_mask(args.data_path, num_users, user_labels)
        num_test_users = test_mask.sum().item()
        print(f"Identified {num_test_users} users for testing.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error determining test users: {e}"); return

    if num_test_users == 0:
        print("Warning: No test users identified. Saving empty predictions file.")
        pd.DataFrame().to_csv(args.output_path, index=False, header=False); return

    # --- Model Loading ---
    print("Loading model...")
    # Need to know output dimension (num_labels) if labels weren't loaded
    # Best practice: save num_labels with model, or require it as arg.
    # WORKAROUND: Infer from model state dict if possible, otherwise require arg.
    # Let's try loading state dict first to infer output size.
    try:
        state_dict = torch.load(args.model_path, map_location='cpu') # Load to CPU first
        # Infer num_labels from the final layer's weight/bias
        out_dim = None
        if 'lin.weight' in state_dict:
            out_dim = state_dict['lin.weight'].shape[0]
        elif 'lin.bias' in state_dict:
            out_dim = state_dict['lin.bias'].shape[0]

        if out_dim is None:
             raise RuntimeError("Could not infer output dimension (num_labels) from model state_dict.")
        if num_labels_loaded is not None and num_labels_loaded != out_dim:
             print(f"Warning: num_labels from data ({num_labels_loaded}) != model output dim ({out_dim}). Using model dim.")
        num_labels = out_dim # Use dimension from loaded model
        print(f"Inferred num_labels (l) = {num_labels} from model.")

        # Instantiate model with known config
        model = HeteroGraphSAGE(hidden_channels=args.hidden_dim,
                                out_channels=num_labels,
                                num_layers=args.num_layers).to(device) # Move model first
        model.load_state_dict(state_dict) # Load state dict now
        print("Model weights loaded successfully.")
    except FileNotFoundError: print(f"Error: Model file not found: {args.model_path}"); return
    except Exception as e: print(f"Error loading model: {e}"); traceback.print_exc(); return

    model.eval()

    # --- Create HeteroData for Inference ---
    # Use loaded components, don't need labels for inference
    data = HeteroData()
    data['user'].x = user_feat
    data['product'].x = prod_feat
    data['user','interacts_with','product'].edge_index = edge_idx_user_prod
    data['product','rev_interacts_with','user'].edge_index = edge_idx_user_prod.flip([0])
    data = data.to(device)
    test_mask = test_mask.to(device) # Move mask to device

    # --- Inference ---
    print("Running inference...")
    start_time = time.time()
    with torch.no_grad():
        all_user_logits = model(data.x_dict, data.edge_index_dict) # Shape (num_users, l)
        test_user_logits = all_user_logits[test_mask] # Shape (mt, l)
    infer_time = time.time() - start_time
    print(f"Inference completed in {infer_time:.2f} seconds.")
    print(f"Test user logits shape: {test_user_logits.shape}")

    # --- Prepare Output (Binary Predictions) ---
    test_probs = torch.sigmoid(test_user_logits)
    predictions = (test_probs > 0.5).int() # Threshold at 0.5 -> binary 0/1
    print(f"Generated binary predictions shape: {predictions.shape}") # Should be (mt, l)

    # --- Save Output ---
    print(f"Saving predictions to {args.output_path}...")
    try:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        predictions_np = predictions.cpu().numpy()
        df_output = pd.DataFrame(predictions_np) # Already (mt, l)
        df_output.to_csv(args.output_path, index=False, header=False)
        print(f"Predictions saved successfully. Shape: {df_output.shape}")
    except Exception as e: print(f"Error saving predictions: {e}"); traceback.print_exc()

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test HeteroGraphSAGE for Task 2 (Submission).')
    parser.add_argument('--data_path',type=str,required=True,help='Path to Task 2 test data dir')
    parser.add_argument('--model_path',type=str,required=True,help='Path to saved trained model')
    parser.add_argument('--output_path',type=str,required=True,help='Path to save output CSV')
    # Args needed to load correct model architecture
    parser.add_argument('--model_type',type=str,default='graphsage', help='Architecture type (should match saved model)')
    parser.add_argument('--hidden_dim',type=int,default=128, help='Hidden units used in training')
    parser.add_argument('--num_layers',type=int,default=2, help='Num GNN layers used in training')
    parser.add_argument('--device',type=str,default='cuda',choices=['cuda','cpu'])
    args = parser.parse_args()
    # Ensure model type matches script expectation
    if args.model_type.lower() != 'graphsage': print(f"Warning: This script expects model_type 'graphsage', got '{args.model_type}'")
    if args.device == 'cuda' and not torch.cuda.is_available(): args.device = 'cpu'
    test(args)
    print("\nTest script finished.")