import sys
import time
import subprocess
import os
import itertools
import json
import algorithms
import networkx as nx

NODE_ID_TYPE = str

# --- Utility Functions ---
def load_graph(file_path):
    graph = {}
    nodes = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            u, v, p_str = parts
            try:
                p = float(p_str)
            except ValueError:
                continue
            if u not in graph:
                graph[u] = []
            graph[u].append((v, p))
            nodes.update([u, v])
            if v not in graph:
                graph[v] = []
    return graph, list(nodes)

def create_nx_graph(graph_dict, nodes_list):
    print("Creating NetworkX graph...")
    G_nx = nx.DiGraph()
    typed_nodes = [NODE_ID_TYPE(n) for n in nodes_list]
    G_nx.add_nodes_from(typed_nodes)
    nodes_in_graph_set = set(G_nx.nodes())
    edge_count = 0
    for u_raw, neighbors in graph_dict.items():
        u = NODE_ID_TYPE(u_raw)
        if u in nodes_in_graph_set:
            for v_raw, p in neighbors:
                v = NODE_ID_TYPE(v_raw)
                if v in nodes_in_graph_set:
                    G_nx.add_edge(u, v, probability=p, weight=p)
                    edge_count += 1
    print(f"NetworkX graph created: {len(G_nx)} nodes, {edge_count} edges.")
    if len(G_nx) != len(nodes_list):
        print(f"Warning: Node count mismatch. Expected {len(nodes_list)}, NetworkX graph has {len(G_nx)}.")
    return G_nx


def log(message):
    print(message)  # Print all logs instead of writing to a file

def write_seed_file(seed_set, output_file):
    """Saves the selected seed nodes to the specified output file."""
    with open(output_file, "w") as f:
        for node in seed_set:
            f.write(f"{node}\n")
    log(f"Seed nodes saved to '{output_file}'.")


def run_algorithm(algo_name, graph, k, params, output_file,nxGraph,dataset_file,seed_nodes=None):
    """
    Runs the given algorithm with parameters and ensures it does not exceed 45 minutes.
    """
    start_time = time.time()
    best_seed_set = []

    try:

        seed_set = None
        while time.time() - start_time < 2700:  # 2700 seconds = 45 minutes
            if algo_name == "stop_and_stare":  # Stop and Stare
                seed_set = algorithms.stop_and_stare(nxGraph, k,seed_nodes,dataset_file,output_file, lookahead=params["lookahead"],initial_seed_set=collect_best_seeds(output_file))
            elif algo_name == "pagerank":
                seed_set = algorithms.calculate_pagerank_scores(nxGraph, k, alpha=params["alpha"],max_iter=params["max_iter"],tol=params["tol"])
            elif algo_name == "betweenness":
                seed_set = algorithms.calculate_betweenness_centrality(nxGraph, k, k_sample=params["k_sample"])
            elif algo_name == "algo3":
                seed_set = algorithms.algo3(graph, k)
            elif algo_name == "imm":
                seed_set = algorithms.imm_algorithm(nxGraph, k,epsilon=params["epsilon"],delta=params["delta"])        
            else:
                log(f"Error: Unknown algorithm '{algo_name}'")
                return
            
            if seed_set:
                break  

        if not seed_set:
            log(f"Warning: {algo_name} returned an empty seed set!")
            return

        write_seed_file(seed_set, output_file)
        elapsed_time = time.time() - start_time

    except Exception as e:
        log(f"Error in {algo_name}: {e}")


import os
import glob

def collect_unique_seeds(directory="."):
    """Collects unique seeds from all files starting with 'best_' in the given directory."""
    seed_set = set()
    
    # Get all files starting with 'best_' and ending with '.txt'
    seed_files = glob.glob(os.path.join(directory, "best_*.txt"))
    
    for file in seed_files:
        with open(file, "r") as f:
            for line in f:
                seed = line.strip()
                if seed:  # Avoid empty lines
                    seed_set.add(seed)
    
    return seed_set  # Convert back to list if neededc


def collect_best_seeds(file_path=None, directory="."):

    if file_path is None or not os.path.exists(file_path):
        file_path = os.path.join(directory, "best_pagerank.txt")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return set()

    seed_set = set()
    with open(file_path, "r") as f:
        for line in f:
            seed = line.strip()
            if seed:  # Avoid empty lines
                seed_set.add(seed)

    return seed_set  # Convert back to list if needed


# --- Main Execution ---
def main():
    if len(sys.argv) < 6:
        print("Usage: python main.py <dataset_file> <algo_name> <params_json> <budget> <output_file>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    algo_name = sys.argv[2]
    param_json = sys.argv[3]
    budget = int(sys.argv[4])
    output_file = sys.argv[5]
    
    # Load algorithm parameters
    try:
        params = json.loads(param_json)
    except json.JSONDecodeError:
        log("Error: Invalid JSON format for parameters.")
        sys.exit(1)


    seed_list = collect_unique_seeds(".")

    graph, nodes = load_graph(dataset_file)
    log(f"Graph loaded: {len(graph)} nodes, {sum(len(edges) for edges in graph.values())} edges.")

    nxGraph = create_nx_graph(graph, nodes)
    log(f"Starting {algo_name} with params {params} and budget {budget}...")
    run_algorithm(algo_name, graph, budget, params, output_file,nxGraph,dataset_file,seed_list)

if __name__ == '__main__':
    main()
