import random
import sys
import time
import torch
import torch_geometric
from torch_geometric.nn import Node2Vec
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import tempfile
import subprocess
import os

import random
import heapq
import time
import math
from collections import deque

def generate_rr_set(G, source):
    """Generate a Reverse Reachable (RR) set for a given source node."""
    rr_set = set()
    queue = deque([source])

    while queue:
        node = queue.popleft()  # Faster than pop(0)
        rr_set.add(node)

        for neighbor in G.predecessors(node):  # Reverse traversal
            if neighbor not in rr_set and random.random() < G[neighbor][node].get('weight', 1):
                queue.append(neighbor)

    return rr_set

def generate_rr_sets(G, num_rr_sets):
    """Generate multiple RR sets by randomly selecting source nodes."""
    rr_sets = []
    nodes = list(G.nodes())

    for _ in range(num_rr_sets):
        source = random.choice(nodes)
        rr_sets.append(generate_rr_set(G, source))

    return rr_sets

def select_seeds(rr_sets, budget):
    """Select the most influential seed nodes using a maximum coverage strategy."""
    seed_set = set()
    node_coverage = {}

    # Count occurrences of each node in RR sets
    for rr_set in rr_sets:
        for node in rr_set:
            node_coverage[node] = node_coverage.get(node, 0) + 1

    # Use max heap for efficient selection of top nodes
    max_heap = [(-count, node) for node, count in node_coverage.items()]
    heapq.heapify(max_heap)

    for _ in range(budget):
        if not max_heap:
            break
        _, best_node = heapq.heappop(max_heap)
        seed_set.add(best_node)

        # Remove RR sets containing the selected node
        rr_sets = [rr for rr in rr_sets if best_node not in rr]

        # Recalculate node coverage
        new_node_coverage = {}
        for rr_set in rr_sets:
            for node in rr_set:
                new_node_coverage[node] = new_node_coverage.get(node, 0) + 1

        max_heap = [(-count, node) for node, count in new_node_coverage.items()]
        heapq.heapify(max_heap)

    return seed_set

def imm_algorithm(G, budget, epsilon=0.1, delta=1e-3):
    """Improved IMM Algorithm with optimized RR set generation and seed selection."""
    start_time = time.time()
    
    n = len(G)
    num_rr_sets = int((8 + 2 / epsilon) * math.log(n) / (epsilon ** 2))
    num_rr_sets = max(min(num_rr_sets, 50000), 1000)  # Ensure reasonable range

    print(f"Generating {num_rr_sets} RR sets...")
    rr_sets = generate_rr_sets(G, num_rr_sets)

    print(f"Selecting {budget} seed nodes...")
    seeds = select_seeds(rr_sets, budget)

    elapsed_time = round((time.time() - start_time) / 60, 2)
    print(f"IMM Algorithm completed in {elapsed_time} minutes.")
    
    return seeds

def write_seed_file(seed_set, output_file):
    """Saves the selected seed nodes to the specified output file."""
    with open(output_file, "w") as f:
        for node in seed_set:
            f.write(f"{node}\n")
    print(f"Seed nodes saved to '{output_file}'.")


# # --- Stop and Stare Optimization Stage ---
def stop_and_stare(G,k, candidate_pool,dataset_file,output_file, lookahead,initial_seed_set):
    """
    Performs a local search ("stop and stare") to iteratively improve the seed set.
    Starting from initial_seed_set (a list of k seeds), for each seed in the set,
    it tries replacing it with a candidate from candidate_pool (union of seeds from all methods)
    and keeps the replacement if it improves the simulated spread.
    """
    print(f"set of pool : {candidate_pool}")
    print(f"initial seed length : {len(initial_seed_set)} and seeds {initial_seed_set}")
    best_seeds = set(initial_seed_set)
    print(f"best_seeds length : {len(best_seeds)} ")
    # best_spread = simulate_spread(G, best_seeds, lookahead)
    best_spread = get_spread(dataset_file, best_seeds)
    print(f"Initial spread for stop and stare: {best_spread}")
    improved = True

    while improved:
        improved = False
        # For each seed in the current best set, try replacing it with a candidate not in the set
        for seed in list(best_seeds):
            for candidate in candidate_pool:
                if candidate in best_seeds:
                    continue
                new_seed_set = (best_seeds - {seed}) | {candidate}

                # new_spread = simulate_spread(G, new_seed_set, lookahead)
                new_spread = get_spread(dataset_file,new_seed_set)
                if new_spread > best_spread:
                    print(f"Replacing seed {seed} with {candidate} improved spread from {best_spread} to {new_spread}")
                    write_seed_file(new_seed_set,output_file)
                    best_seeds = new_seed_set
                    best_spread = new_spread
                    improved = True
                    break  # Break inner loop to restart search
            if improved:
                break  # Restart outer loop if any improvement was found

    return list(best_seeds), best_spread


# --- Extended scoring functions ---
def calculate_pagerank_scores(G, number_of_seeds,  alpha=0.85, max_iter=100, tol=1.0e-6,):
    print("Calculating PageRank scores...")
    start_time = time.time()
    try:
        scores = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol, weight=None)
        print(f"PageRank calculation finished in {time.time() - start_time:.2f} seconds.")
        adjusted_scores = adjust_scores_by_component(G, scores)
        return select_top_k(adjusted_scores, number_of_seeds)
    except Exception as e:
        print(f"Error calculating PageRank: {e}")
        return {}


# Outdegree heuristic
def algo3(graph, k):
    degree = {u: len(graph[u]) for u in graph}
    sorted_nodes = sorted(degree, key=degree.get, reverse=True)
    return sorted_nodes[:k]   # Return list

def adjust_scores_by_component(G, node_scores):
    print("Adjusting scores based on component sizes...")
    total_nodes = G.number_of_nodes()
    comp_sizes = {}
    for comp in nx.weakly_connected_components(G):
        comp_size = len(comp)
        for node in comp:
            comp_sizes[node] = comp_size
    adjusted_scores = {node: score * (comp_sizes.get(node, 1) / total_nodes) for node, score in node_scores.items()}
    return adjusted_scores

# --- Provided helper functions ---
def select_top_k(node_scores, k):
    """Selects top k nodes based on scores."""
    if not node_scores:
        print("Warning: Node scores dictionary is empty, cannot select top K.")
        return []
    actual_k = min(k, len(node_scores))
    if actual_k <= 0:
        print("Warning: K is zero or negative, returning empty list.")
        return []
    try:
        sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        top_k_nodes = [node for node, score in sorted_nodes[:actual_k]]
        print(f"Selected top {len(top_k_nodes)} nodes.")
        return top_k_nodes
    except Exception as e:
        print(f"Error during top-k selection: {e}")
        return []

def calculate_betweenness_centrality(G, number_of_seeds, k_sample=None):
    nodes_count = len(G)
    if k_sample is not None and k_sample >= nodes_count:
         k_sample = None
    print(f"Calculating Betweenness Centrality scores{f' (approximated with k={k_sample})' if k_sample else ''}...")
    print("Warning: Betweenness Centrality can be very slow on large graphs.")
    start_time = time.time()
    try:
        scores = nx.betweenness_centrality(G, k=k_sample, normalized=True, weight=None)
        print(f"Betweenness Centrality finished in {time.time() - start_time:.2f} seconds.")
        adjusted_scores = adjust_scores_by_component(G, scores)
        return select_top_k(adjusted_scores, number_of_seeds)
        
    except Exception as e:
        print(f"Error calculating Betweenness Centrality: {e}")
        return {}




def get_spread(dataset_file, seed):
    """
    Invokes the external './infection' executable with <dataset_file> and either
    a seed file path or a list of seed nodes. Returns the float value found after 'Spread:'.

    :param dataset_file: Path to the dataset file
    :param seed: Either a string (path to seed file) or a list of seed nodes
    :return: Spread (float)
    """
    # 1. If 'seed' is a list, we'll create a temporary file to hold those seeds.
    temp_seed_file = None  # keep track so we can delete it later
    if isinstance(seed, set):
        # create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
            temp_seed_file = tmp.name
            for node in seed:
                tmp.write(f"{node}\n")
        seed_file = temp_seed_file
    elif isinstance(seed, str):
        # user passed an actual seed file path
        seed_file = seed
    else:
        print("Error: 'seed' must be either a string (seed file path) or a list of seeds.")
        return 0.0

    command = ["./src/infection", dataset_file, seed_file]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if line.startswith("Spread:"):
                return float(line.split(":")[1].strip())

        print("Error: 'Spread:' line not found in output.")
        return 0.0

    except subprocess.CalledProcessError as e:
        print(f"Error running ./infection: {e}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return 0.0
    except ValueError:
        print("Error: Could not parse spread value as float.")
        return 0.0
    except FileNotFoundError:
        print("Error: ./infection executable not found.")
        return 0.0
    finally:
        # 2. If we created a temporary file, remove it
        if temp_seed_file is not None and os.path.exists(temp_seed_file):
            os.remove(temp_seed_file)

def simulate_spread(G, initial_seeds, num_simulations=100):
    total_spread = 0
    if not initial_seeds:
        return 0
    valid_seeds = {seed for seed in initial_seeds if seed in G}
    if not valid_seeds:
        return 0
    for _ in range(num_simulations):
        infected_nodes = set(valid_seeds)
        newly_infected_queue = deque(valid_seeds)
        while newly_infected_queue:
            u = newly_infected_queue.popleft()
            for v in G.neighbors(u):
                if v not in infected_nodes:
                    try:
                        prob = G[u][v].get('probability', 0)
                        if random.random() < prob:
                            infected_nodes.add(v)
                            newly_infected_queue.append(v)
                    except KeyError:
                        pass
        total_spread += len(infected_nodes)
    return total_spread / num_simulations if num_simulations > 0 else 0


# Outdegree heuristic
def algo3(graph, k):
    degree = {u: len(graph[u]) for u in graph}
    sorted_nodes = sorted(degree, key=degree.get, reverse=True)
    return sorted_nodes[:k]   # Return list

def iim(graph, k, iterations=50):
    seed_set = set()
    for _ in range(k):
        best_node = max(graph, key=lambda node: sum(p for _, p in graph[node] if node not in seed_set))
        seed_set.add(best_node)
    return list(seed_set)

