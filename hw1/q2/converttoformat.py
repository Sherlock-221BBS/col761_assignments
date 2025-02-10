import os
import sys

def convert_to_fsg(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        graph_id = -1
        node_id = 0
        edge_count = 0
        reading_edges = False
        
        for line in infile:
            line = line.strip()
            if line.startswith("#"):  
                graph_id += 1
                outfile.write(f"t # {graph_id}\n")  
                node_id = 0
                edge_count = 0
                reading_edges = False
            elif line.isdigit():  
                if node_id == 0:  
                    node_count = int(line)
                elif edge_count == 0:  
                    edge_count = int(line)
                    reading_edges = True  
            elif not reading_edges:  
                outfile.write(f"v {node_id} {line}\n")  
                node_id += 1
            else:
                parts = line.split()
                if len(parts) == 3:
                    node1, node2, label = parts
                    outfile.write(f"e {node1} {node2} {label}\n")

    print(f"Successfully written to {output_file}")
                    

def parse_graph_file(filename):
    """
    Parses the given .txt_graph file and returns a list of graphs.
    """
    graphs = []
    label_mapping = {'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10}

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    index = 0

    while index < len(lines):
        if not lines[index].startswith('#'):  
            index += 1
            continue
    
        graph_id = lines[index][1:]
        index += 1

        if index >= len(lines):
            print("Unexpected end of file after graph ID.")
            break

        try:
            num_nodes = int(lines[index])
        except ValueError:
            print(f"  Error: Invalid number of nodes at line {index}: {lines[index]}")
            break

        index += 1

        if index + num_nodes > len(lines):
            print(f"  Error: Not enough lines for {num_nodes} node labels.")
            break

        node_labels = []
        for i in range(num_nodes):
            label = lines[index]
            if label not in label_mapping:
                print(f"  Warning: Unknown label {label} at line {index}")
            node_labels.append(label_mapping.get(label, -1))
            index += 1

        if index >= len(lines):
            print("Unexpected end of file after node labels.")
            break

        try:
            num_edges = int(lines[index])
        except ValueError:
            print(f"  Error: Invalid number of edges at line {index}: {lines[index]}")
            break

        index += 1

        edges = []
        for i in range(num_edges):
            if index >= len(lines):
                print("  Unexpected end of file while reading edges.")
                break
            edge_data = lines[index].split()
            if len(edge_data) != 3:
                print(f"  Warning: Invalid edge format at line {index}: {lines[index]}")
            else:
                edges.append(tuple(map(int, edge_data)))
            index += 1

        graphs.append((graph_id, node_labels, edges))


    return graphs


def write_gspan_format(graphs, output_filename):
    """
    Writes graphs in gSpan format.
    """
    with open(output_filename, 'w') as f:
        for graph_id, node_labels, edges in graphs:
            f.write(f't # {graph_id}\n')
            for i, label in enumerate(node_labels):
                f.write(f'v {i} {label}\n')
            for src, dest, edge_type in edges:
                f.write(f'e {src} {dest} {edge_type}\n')

    print(f"Successfully written to {output_filename}")
            

def write_gaston_format(graphs, output_file):
    """
    Writes the parsed graphs into Gaston format.
    """
    with open(output_file, 'w') as f:
        for graph_id, node_labels, edges in graphs:
            f.write(f"t # {graph_id}\n")

            for i, label in enumerate(node_labels):
                f.write(f"v {i} {label}\n")

            for edge in edges:
                node1, node2, label = edge
                f.write(f"e {node1} {node2} {label}\n")

    print(f"Successfully written to {output_file}")




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 converttoformat.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"Processing file: {input_file}")
    graphs = parse_graph_file(input_file)
    write_gspan_format(graphs, "gspan.txt")
    write_gaston_format(graphs, "gaston.txt")
    convert_to_fsg(input_file, "fsg.txt")