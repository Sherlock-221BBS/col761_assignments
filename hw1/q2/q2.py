import os
import sys
import matplotlib.pyplot as plt

def read_log(log_path):
    gspan_list = []
    fsg_list = []
    gaston_list = []
    support_list = []

    with open(log_path, "r") as file:
        for line in file:
            if "gSpan runtime" in line:
                time = int(line.split(":")[-1].strip().split()[0])
                gspan_list.append(time)
            elif "FSG runtime" in line:
                time = int(line.split(":")[-1].strip().split()[0])
                fsg_list.append(time)
            elif "Gaston runtime" in line:
                time = int(line.split(":")[-1].strip().split()[0])
                gaston_list.append(time)

            if "support" in line:
                support = int(line.split("support")[0].strip().split()[-1].replace('%', ''))
                if support not in support_list:
                    support_list.append(support)
            
    
    return support_list, gspan_list, fsg_list, gaston_list

def plot_results(support_list, gspan_list, fsg_list, gaston_list, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(support_list, gspan_list, label="gSpan", marker="o")
    plt.plot(support_list, fsg_list, label="FSG", marker="o")
    plt.plot(support_list, gaston_list, label="Gaston", marker="o")
    plt.xlabel("Minimum Support (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison of gSpan, FSG, and Gaston")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "plot.png"))
    plt.show()

if __name__ == "__main__":
    output_path = sys.argv[1]
    log_path = os.path.join(output_path, "run_log.txt")
    support_list, gspan_list, fsg_list, gaston_list = read_log(log_path)
    plot_results(support_list, gspan_list, fsg_list, gaston_list, output_path)
