import os
import numpy as np
import networkx as nx
from collections import defaultdict
import argparse
from multiprocessing import Pool, cpu_count


LABEL_MAPPING = {
    'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4,
    'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def saveGastonFormatInFile(graphs, filename):
    with open(filename, 'w') as f:
        for i, g in enumerate(graphs):
            f.write(f"t # {i}\n")
            for node in sorted(g.nodes(data=True)):
                f.write(f"v {node[0]} {node[1]['label']}\n")
            for u, v in g.edges():
                f.write(f"e {u} {v} {g[u][v]['label']}\n")

def loadGraphs(filename):
    graphs = []
    temp = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('t #') or line.startswith('#'):
                if temp is not None:
                    graphs.append(temp)
                temp = nx.Graph()
            elif line.startswith('v'):
                parts = line.split()
                if len(parts) == 3:
                    _, node_id, label = parts
                else:
                    node_id, label = parts[1], parts[2]
                temp.add_node(int(node_id), label=int(label))
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) == 4:
                    _, u, v, label = parts
                else:
                    u, v, label = parts[1], parts[2], parts[3]
                temp.add_edge(int(u), int(v), label=int(label))
    if temp is not None:
        graphs.append(temp)
    return graphs


def loadLabels(path):
    return np.loadtxt(path, dtype=int)

def executeGaston(input_file, maxVertex,support, totalGraphs):
    gastonThreshold = max(1, int(round(totalGraphs * support)))
    
    base_name = os.path.basename(input_file)
    outputFile = f"{base_name}.fp"
    
    command = f"./gaston -m {maxVertex} {gastonThreshold} {input_file} {outputFile}"
    print(f"Running Gaston command: {command}")
    os.system(command)
    
    return outputFile
    
def gastonParsing(file_path):
    subgraphs = []
    current_sg = None
    graphCount = 0
    
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                graphCount +=1
                if current_sg is not None:
                    subgraphs.append(current_sg)
                current_sg = None
            elif line.startswith('t'):
                current_sg = nx.Graph()
            elif line.startswith('v') and current_sg is not None:
                parts = line.split()
                node_id, label = parts[1], parts[2]
                current_sg.add_node(int(node_id), label=int(label))
            elif line.startswith('e') and current_sg is not None:
                parts = line.split()
                u, v, label = parts[1], parts[2], parts[3]
                current_sg.add_edge(int(u), int(v), label=int(label))
    
    if current_sg is not None:
        subgraphs.append(current_sg)
    
    return subgraphs, graphCount


def dynamicParameterSetting(totalGraphs):
    if totalGraphs < 5000:
        return 7,100
    elif totalGraphs < 20000:
        return 6,100
    elif totalGraphs < 35000:
        return 5,90
    elif totalGraphs < 45000:
        return 4,70
    elif totalGraphs < 50000:
        return 3,40
    else:
        return 2,20

def findDiscriminativeSubgraphs(args):

    graphs = loadGraphs(args.graphs)
    labels = loadLabels(args.labels)

    print("graphs reading done")

    saveGastonFormatInFile(graphs, "gaston_format.txt")
    totalGraphs = len(labels)
    maxVertex, maxSubgraphThreshold = dynamicParameterSetting(totalGraphs)
    supportThreshold = 0.10
    subGraphCount = 200

    while (subGraphCount > maxSubgraphThreshold and supportThreshold < 1):

        print(f"running gaston for threshold {supportThreshold}")

        filePathOfGastonOutput = executeGaston("gaston_format.txt",maxVertex,supportThreshold,totalGraphs)
        subgraphs, subGraphCount = gastonParsing(filePathOfGastonOutput)
        print(f"subgraph found {subGraphCount}")

        supportThreshold += 0.020

    saveGastonFormatInFile(subgraphs[:100], args.output)

def checkIfSubgraphOrNot(graph, subGraph):

    if len(subGraph) > len(graph):
        return False

    graphNodeLabels = [graph.nodes[n]['label'] for n in graph.nodes()]
    subGraphNodeLabels = [subGraph.nodes[n]['label'] for n in subGraph.nodes()]
    
    for label in set(subGraphNodeLabels):
        if graphNodeLabels.count(label) < subGraphNodeLabels.count(label):
            return False  

    matcher = nx.isomorphism.GraphMatcher(
        graph, subGraph,
        node_match=lambda x, y: x['label'] == y['label'],
        edge_match=lambda x, y: x['label'] == y['label']
    )
    return matcher.subgraph_is_isomorphic()


def checkSubgraphIntermediateCode(tupleForArguments):
    graph, subGraph = tupleForArguments
    return checkIfSubgraphOrNot(graph, subGraph)

def convertToFeature(args):
    try:
        graphs = loadGraphs(args.graphs)
        subgraphs = loadGraphs(args.subgraphs)
        
        if not subgraphs:
            print("No subgraphs found in discriminative file.")
            return

        tasks = [(graph, subGraph) for graph in graphs for subGraph in subgraphs]

        numberOfWorkers = min(cpu_count(), 4)
        with Pool(numberOfWorkers) as pool:
            results = pool.map(checkSubgraphIntermediateCode, tasks)

        features = np.array(results, dtype=int).reshape(len(graphs), len(subgraphs))

        np.save(args.output, features)
        print(f"Features saved to {args.output}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    

    mine_parser = subparsers.add_parser('findDiscriminativeSubgraphs')
    mine_parser.add_argument('--graphs', required=True)
    mine_parser.add_argument('--labels', required=True)
    mine_parser.add_argument('--output', required=True)
    mine_parser.set_defaults(func=findDiscriminativeSubgraphs)


    convert_parser = subparsers.add_parser('convertToFeature')
    convert_parser.add_argument('--graphs', required=True)
    convert_parser.add_argument('--subgraphs', required=True)
    convert_parser.add_argument('--output', required=True)
    convert_parser.set_defaults(func=convertToFeature)
    
    args = parser.parse_args()
    args.func(args)
