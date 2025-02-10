import os
import sys
import matplotlib.pyplot as plt


def generate_plot(output_path):
    apriori_list = {}
    fpgrowth_list = {}
    logfile= os.path.join(output_path, "run_log.txt")
    if not os.path.exists(logfile):
        with open(logfile, "w") as f:
            pass

    with open(logfile, 'r') as f:
        for line in f:
            if "Apriori runtime" in line:
                support = int(line.split()[3][:-1])
                runtime = float(line.split()[-2])
                apriori_list[support] = runtime
            elif "FP-growth runtime" in line:
                support = int(line.split()[3][:-1])
                runtime = float(line.split()[-2])
                fpgrowth_list[support] = runtime

    support_array = [90, 50, 25, 10, 5]

    _, ax = plt.subplots()
    ax.plot(support_array, [apriori_list.get(s, 0) for s in support_array], label="Apriori", marker='o')
    ax.plot(support_array, [fpgrowth_list.get(s, 0) for s in support_array], label="FP-growth", marker='o')
    ax.set_xlabel('Support Threshold (%)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison of Apriori and FP-growth')
    ax.legend()

    plot_file = os.path.join(output_path, "plot.png")
    plt.savefig(plot_file)
    print(f"saved to {plot_file}")

if __name__ == "__main__":
    output_path = sys.argv[1]
    generate_plot(output_path)
