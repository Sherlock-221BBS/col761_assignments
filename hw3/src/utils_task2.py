# src/utils_task2.py (FINAL Submission Version - Handles Global Indexing - Corrected)
import torch
import numpy as np
import os
import warnings
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] %(message)s')

def load_task2_data_components_final(data_dir):
    """
    Loads heterogeneous graph data components for Task 2 submission scripts.
    Detects edge indexing (Global 0-based or Local 0-based) and remaps if needed.

    Expected files: user_features.npy, product_features.npy,
                    user_product.npy. Optionally loads label.npy.

    Args:
        data_dir (str): Path to the directory containing data files.

    Returns:
        tuple: (user_features, product_features, edge_index_user_prod, user_labels_or_none)
               All are PyTorch tensors (user_labels_or_none is None if not loaded/found).
        int: Number of users (m), or None if error.
        int: Number of products (p), or None if error.
        int: Number of personality traits (l) if labels loaded, else None.
    """
    user_feat_file=os.path.join(data_dir,'user_features.npy');prod_feat_file=os.path.join(data_dir,'product_features.npy');label_file=os.path.join(data_dir,'label.npy');edge_file=os.path.join(data_dir,'user_product.npy')
    logging.info(f"--- Loading Task 2 Data Components from: {data_dir} ---")
    user_labels = None; num_labels = None

    try:
        # 1. Load Features
        user_features_np=np.load(user_feat_file).astype(np.float32);user_features=torch.from_numpy(user_features_np);num_users=user_features.shape[0]
        product_features_np=np.load(prod_feat_file).astype(np.float32);product_features=torch.from_numpy(product_features_np);num_products=product_features.shape[0]
        logging.info(f"Loaded user features: {user_features.shape}, product features: {product_features.shape}")
        assert user_features.ndim==2 and product_features.ndim==2, "Features not 2D"

        # 2. Load Labels (Optional)
        if os.path.exists(label_file):
            try:
                user_labels_np=np.load(label_file).astype(np.float32)
                assert user_labels_np.shape[0]==num_users and user_labels_np.ndim==2,"Label shape/count mismatch"
                num_labels=user_labels_np.shape[1]; user_labels=torch.from_numpy(user_labels_np)
                logging.info(f"Loaded user labels: {user_labels.shape}")
            except Exception as e: logging.warning(f"Could not load labels: {e}")
        else: logging.warning(f"Label file not found: {label_file}")

        # 3. Load Edges and Determine/Remap Indexing
        edges_np=np.load(edge_file); logging.info(f"Loaded edges: {edges_np.shape}"); assert edges_np.ndim==2 and edges_np.shape[1]==2, "Edges shape incorrect"

        src_user_indices = torch.tensor(edges_np[:, 0], dtype=torch.long)
        # !!! FIX: Read original product indices into a separate variable !!!
        prod_indices_read = torch.tensor(edges_np[:, 1], dtype=torch.long)

        # Validate user indices (should always be 0 to m-1)
        assert src_user_indices.min()>=0 and src_user_indices.max()<num_users, "User index in edges out of bounds [0, m-1]"

        # Check product index range to determine format using the read indices
        min_prod_idx = prod_indices_read.min().item()
        max_prod_idx = prod_indices_read.max().item()

        if min_prod_idx >= num_users and max_prod_idx < (num_users + num_products):
            # GLOBAL 0-based indexing (m to n-1) -> Remap needed
            logging.info("Detected GLOBAL 0-based product indexing in edges. Remapping...")
            # !!! FIX: Remap from the read indices !!!
            dst_product_indices = prod_indices_read - num_users
            assert dst_product_indices.min()>=0 and dst_product_indices.max()<num_products, "Remapped product index out of bounds [0, p-1]"
        elif min_prod_idx >= 0 and max_prod_idx < num_products:
            # LOCAL 0-based indexing (0 to p-1) -> Use directly
            logging.info("Detected LOCAL 0-based product indexing in edges. Using directly.")
            # !!! FIX: Assign the read indices directly !!!
            dst_product_indices = prod_indices_read
        else:
            # Indexing format is unclear or invalid
            raise ValueError(f"Product indices range [{min_prod_idx}, {max_prod_idx}] is inconsistent "
                             f"with num_users={num_users} and num_products={num_products}. "
                             f"Expected either [0, {num_products-1}] (Local) or "
                             f"[{num_users}, {num_users + num_products - 1}] (Global).")

        edge_index_user_prod = torch.stack([src_user_indices, dst_product_indices], dim=0)
        logging.info(f"Processed user->product edge index (local 0-based): {edge_index_user_prod.shape}")

        return user_features, product_features, edge_index_user_prod, user_labels, num_users, num_products, num_labels

    except FileNotFoundError as e: logging.error(f"Required data file not found: {e}"); traceback.print_exc()
    except AssertionError as e: logging.error(f"Data validation failed: {e}"); traceback.print_exc()
    except ValueError as e: logging.error(f"Data consistency/format error: {e}"); traceback.print_exc() # Catch the specific ValueError
    except Exception as e: logging.exception(f"ERROR during data loading: {e}")
    return None, None, None, None, None, None, None

# --- Example Usage (for testing the loader directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and inspect Task 2 data components (Flexible Indexing).')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Task 2 data directory')
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
         print(f"Error: Provided data path is not a valid directory: {args.data_path}")
         exit()

    # Call the loader function
    u_feat, p_feat, edge_idx, u_labels, n_users, n_prods, n_labels = load_task2_data_components_final(args.data_path)

    if u_feat is not None:
        print("\n--- Loaded Components Summary ---")
        print(f"User Features: Tensor, Shape={u_feat.shape}, Dtype={u_feat.dtype}")
        print(f"Product Features: Tensor, Shape={p_feat.shape}, Dtype={p_feat.dtype}")
        print(f"Edge Index (User->Prod, Local): Tensor, Shape={edge_idx.shape}, Dtype={edge_idx.dtype}")
        if u_labels is not None:
             print(f"User Labels: Tensor, Shape={u_labels.shape}, Dtype={u_labels.dtype}")
        else:
             print("User Labels: Not loaded.")
        print(f"Num Users (m): {n_users}")
        print(f"Num Products (p): {n_prods}")
        if n_labels is not None:
            print(f"Num Labels (l): {n_labels}")
    else:
        print("\nFailed to load data components.")