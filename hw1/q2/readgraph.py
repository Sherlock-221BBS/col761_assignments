def read_graph_file(input_file):
    with open(input_file, 'r') as infile:
        print("Contents of the file:")
        for line in infile:
            print(line.strip())

# Provide the path to your file
file_path = "167.txt_graph"

# Read and display the file contents
read_graph_file(file_path)
