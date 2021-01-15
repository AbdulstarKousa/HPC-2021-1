# Script for processing the data generated to compare the sequential implementations of the
# Jacobi function and the Gauss-Seidel function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

first_data_point = 1 # Skip the first x data points when plotting
plot_names = ["Iterations per second", "Iterations to converge", "Wall time log", "Wall time"]

file_jac = r"sequential_comparison_JAC.dat"
file_gs = r"sequential_comparison_GS.dat"

files = [file_jac, file_gs]
names = ["Jacobi", "Gauss-Seidel"]

for (fi, file) in enumerate(files):
    name = names[fi]
    with open(file) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    content = np.array(content)

    # Remove run settings from data
    run_settings = content[-1]
    content = content[0:-1]

    # Parse sizes from settings
    sizes = run_settings.split("iterations")[0]
    sizes = sizes.split('sizes')[1]
    sizes = sizes.strip().split()
    sizes = np.array(sizes, dtype=int)

    # Split data into runs
    content = np.split(content, np.where(content == content[0])[0])
    content = content[1:]

    # Build data frame from data
    data = pd.DataFrame(columns=["run_time", "iterations", "norm"])

    for (i, run) in enumerate(content):
        time = float(run[1].split(' ')[-1])
        norm = float(run[2].split(' ')[-1])
        iterations = int(run[3].split(' ')[-1])
        row = np.array([time, iterations, norm])
        data.loc[i] = row

    data["iterations/second"] = data["iterations"]/data["run_time"]

    # plt.plot(sizes, data["iterations/second"], label = name)
    # Excluding size 10
    plt.figure(plot_names[0])
    plt.plot(sizes[first_data_point:], data["iterations/second"][first_data_point:], label = name)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Problem size: N")
    plt.ylabel("Iterations/second")

    plt.figure(plot_names[1])
    plt.plot(sizes[first_data_point:], data["iterations"][first_data_point:], label = name)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Problem size: N")
    plt.ylabel("Iterations to converge")

    plt.figure(plot_names[2])
    plt.plot(sizes[first_data_point:], data["run_time"][first_data_point:], label = name)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Problem size: N")
    plt.ylabel("Wall time [s]")

    plt.figure(plot_names[3])
    plt.plot(sizes[first_data_point:], data["run_time"][first_data_point:], label = name)
    # plt.yscale("log")
    plt.legend()
    plt.xlabel("Problem size: N")
    plt.ylabel("Wall time [s]")


for plot_name in plot_names:
    plt.figure(plot_name)
    plt.savefig("Sequential comparison - " + plot_name, bbox_inches='tight', dpi=300)

















#
