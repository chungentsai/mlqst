import sys, os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

markers = [
     "+", "1", "x", "*", "P", "v", "^", "<", ">", "D"
     ]
markers = cycle(markers)

linecolors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:blue','tab:gray', 'tab:olive', 'tab:cyan']
linecolors = cycle(linecolors)

filename = sys.argv[1]


def read_records(filename):
    approx_opt = inf

    results = dict()
    f = open(filename, "r")
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n","")
    n_line = len(lines)

    i = 0
    while i < n_line:
        alg_name = lines[i]
        n_epoch = int(lines[i+1])
        n_rate = int(lines[i+2])
        n_data = n_epoch * n_rate

        results[alg_name] = dict()
        results[alg_name]["n_epoch"] = np.zeros(n_data)
        results[alg_name]["fidelity"] = np.zeros(n_data)
        results[alg_name]["elapsed_time"] = np.zeros(n_data)
        results[alg_name]["opt_error"] = np.zeros(n_data)
        results[alg_name]["marker"] = next(markers)
        results[alg_name]["linecolor"] = next(linecolors)
        
        i = i + 3
        for j in range(n_data):
            data = lines[i + j].split("\t")
            results[alg_name]["n_epoch"][j] = float(data[0])
            results[alg_name]["fidelity"][j] =  float(data[1])
            results[alg_name]["elapsed_time"][j] = float(data[2])
            results[alg_name]["opt_error"][j] = float(data[3])
            approx_opt = min(results[alg_name]["opt_error"][j], approx_opt)
        i = i + n_data

    for alg_name in results.keys():
        results[alg_name]["opt_error"] = results[alg_name]["opt_error"] - approx_opt

    return results


def main():
    results = read_records(filename)
    algs= results.keys()

    directory = filename.replace("records", "figures") + "/"
    if not os.path.isdir(directory):
        os.makedirs(directory)

    plt.figure(1)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    #algs = ["BPG", "QEM", "SQSB", "Frank-Wolfe", "SQLBOMD", "d-sample LB-SDA", "1-sample LB-SDA"]
    for alg_name in algs:
        plt.semilogy(results[alg_name]["n_epoch"], results[alg_name]["opt_error"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
        plt.xlabel("Number of epochs")
        plt.ylabel("Approximate optimization error")
        plt.xlim((0, 200))
        plt.ylim((1e-5,  1e-1))
    plt.legend()
    plt.savefig(directory + "epoch-error.png", dpi=300)

    plt.figure(2)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    #algs = ["1-sample LB-SDA", "d-sample LB-SDA", "SQLBOMD", "Frank-Wolfe", "SQSB", "QEM", "BPG"]
    for alg_name in algs:
        plt.plot(results[alg_name]["n_epoch"], results[alg_name]["fidelity"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
        plt.xlabel("Number of epochs")
        plt.ylabel("Fidelity")
        plt.xlim((0, 200))
        plt.ylim((0.5,  1))
    plt.legend()
    plt.savefig(directory + "epoch-fidelity.png", dpi=300)

    plt.figure(3)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    #algs = ["BPG", "SQSB", "SQLBOMD", "QEM", "1-sample LB-SDA", "d-sample LB-SDA", "Frank-Wolfe"]
    for alg_name in algs:
        plt.loglog(results[alg_name]["elapsed_time"], results[alg_name]["opt_error"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
        plt.xlabel("Elapsed time")
        plt.ylabel("Approximate optimization error")
        #plt.xlim((1e1, 3e5))
        plt.ylim((1e-5,  1e-1))
    plt.legend()
    plt.savefig(directory + "time-error.png", dpi=300)

    plt.figure(4)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    #algs = ["d-sample LB-SDA", "Frank-Wolfe", "1-sample LB-SDA", "QEM", "SQLBOMD", "SQSB", "BPG"]
    for alg_name in algs:
        plt.semilogx(results[alg_name]["elapsed_time"], results[alg_name]["fidelity"], label=alg_name, marker=results[alg_name]["marker"], markevery=0.1, linewidth=1, color=results[alg_name]["linecolor"])
        plt.xlabel("Elapsed time")
        plt.ylabel("Fidelity")
        #plt.xlim((1e1, 3e5))
        plt.ylim((0.5,  1))
    plt.legend()
    plt.savefig(directory + "time-fidelity.png", dpi=300)


if __name__ == "__main__":
    main()