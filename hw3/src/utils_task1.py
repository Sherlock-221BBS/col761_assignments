# src/utils_task1.py
import torch
import numpy as np
import os
from torch_geometric.data import Data
import warnings
import traceback

def load_graph_data(data_dir):
    """ Loads graph data for Task 1 from .npy files """
    edge_file = os.path.join(data_dir, 'edges.npy')
    label_file = os.path.join(data_dir, 'label.npy')
    feature_file = os.path.join(data_dir, 'node_feat.npy')
    print(f"--- Loading Task 1 Data from: {data_dir} ---")
    try:
        node_features_np = np.load(feature_file).astype(np.float32)
        node_features = torch.from_numpy(node_features_np)
        num_nodes = node_features.shape[0]

        edges_np = np.load(edge_file)
        if edges_np.ndim != 2 or edges_np.shape[1] != 2: raise ValueError("Edges shape")
        # Task 1 d1/d2 had simple edge list, assumed 0-based, transpose needed
        edge_index = torch.from_numpy(edges_np).t().contiguous().to(torch.long)
        if edge_index.max() >= num_nodes: warnings.warn(f"Max edge index {edge_index.max()} >= num_nodes {num_nodes}")

        labels_np = np.load(label_file) # Keep as float for NaN check
        if labels_np.shape[0] != num_nodes: raise ValueError("Labels shape")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            test_mask_np = np.isnan(labels_np)
        train_mask_np = ~test_mask_np
        test_mask = torch.from_numpy(test_mask_np).to(dtype=torch.bool)
        train_mask = torch.from_numpy(train_mask_np).to(dtype=torch.bool)

        # Create y tensor (long type), replace NaN with -1 (placeholder)
        y = torch.full_like(torch.from_numpy(labels_np), fill_value=-1, dtype=torch.long)
        # Convert non-NaN original labels to long for the y tensor
        train_labels_np_int = labels_np[train_mask_np].astype(np.int64)
        y[train_mask] = torch.from_numpy(train_labels_np_int) # Assign using torch mask

        unique_train_labels = np.unique(train_labels_np_int)
        num_classes = len(unique_train_labels)
        if num_classes <= 0: raise ValueError("No classes found in training labels")

        data = Data(x=node_features, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.test_mask = test_mask
        # data.num_classes = num_classes # Store if needed, though train script recalculates

        print("Data loaded successfully.")
        return data, num_classes

    except FileNotFoundError as e: print(f"Error: File not found - {e}"); traceback.print_exc()
    except ValueError as e: print(f"Error: Data loading/processing issue - {e}"); traceback.print_exc()
    except Exception as e: print(f"An unexpected error during data loading: {e}"); traceback.print_exc()
    return None, None