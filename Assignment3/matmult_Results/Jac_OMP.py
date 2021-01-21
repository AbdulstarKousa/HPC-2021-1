# Latice site updates = iterations*N^3
# Performance: (Lattice site updates for x core)/(Lattice site updates for 1 core)
# Plot performance as a function of core
# Plot wall time as a function of core

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_sets = [
    [r"Jac_OMP_GCC_Plain_300_1000.dat", r"Jac_OMP_GCC_O3_300_1000.dat", r"Jac_OMP_GCC_Ofast_300_1000.dat"],
    [r"Jac_NUMA_100_2000.dat", r"Jac_NUMA_150_1000.dat", r"Jac_NUMA_300_300.dat"],
    [r"Jac_simple_100_2000.dat", r"Jac_simple_150_1000.dat", r"Jac_simple_300_300.dat"],
    [r"Jac_code_sub_100_2000.dat", r"Jac_code_sub_150_1000.dat", r"Jac_code_sub_300_300.dat"],
    [r"Jac_Parallel_region_100_2000.dat", r"Jac_Parallel_region_150_1000.dat", r"Jac_Parallel_region_300_300.dat", "Jac_Parallel_region_500_100.dat"]
]

name_sets = [
    ["Plain", "O3", "Ofast"],
    ["N = 100", "N = 150", "N = 300"],
    ["N = 100", "N = 150", "N = 300"],
    ["N = 100", "N = 150", "N = 300"],
    ["N = 100", "N = 150", "N = 300", "N = 500"]
]

sets = ["Compiler flags", "NUMA", "Simple", "Code improvements", "Moved parallel region"]


for (set_number, set) in enumerate(sets):
    files = file_sets[set_number]
    names = name_sets[set_number]

    plot_names = ["Performance", "Wall time"]

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
        sizes = sizes.split('size')[1]
        sizes = sizes.strip().split()
        sizes = np.array(sizes, dtype=int)
        size = sizes[0]

        # Split data into runs
        content = np.split(content, np.where(content == content[0])[0])
        content = content[1:]

        # Reverse array to have run with 1 thread at the top
        content = np.flip(content, axis=0)

        # Build data frame from data
        data = pd.DataFrame(columns=["run_time", "iterations", "norm", "threads", "lattice_updates"])

        for (i, run) in enumerate(content):
            time = float(run[1].split(' ')[-1])
            norm = float(run[2].split(' ')[-1])
            iterations = int(run[3].split(' ')[-1])
            threads = int(run[4].split(' ')[-1])
            lattice_updates_per_second = float(iterations)*(float(size)**3)/float(time)

            row = np.array([time, iterations, norm, threads, lattice_updates_per_second])
            data.loc[i] = row

        data.iloc[0]["lattice_updates"]
        data["performance"] = data["lattice_updates"]/data.iloc[0]["lattice_updates"]

        plt.figure(plot_names[0] + set)
        if name == "Ofast":
            plt.plot(data["threads"], data["performance"], label = name, linestyle=":")
        else:
            plt.plot(data["threads"], data["performance"], label = name)
        plt.legend()
        plt.xlabel("Number of threads")
        plt.ylabel("Speed-up (rel. to 1 thread)")
        plt.title(set)

        plt.figure(plot_names[1] + set)
        if name == "Ofast":
            plt.plot(data["threads"], data["lattice_updates"], label = name, linestyle=":")
        else:
            plt.plot(data["threads"], data["lattice_updates"], label = name)
        plt.legend()
        plt.xlabel("Number of threads")
        plt.ylabel("Lattice-site updates per second")
        plt.title(set)

        ### Wall time
        # plt.figure(plot_names[1] + set)
        # if name == "Ofast":
        #     plt.plot(data["threads"], data["run_time"], label = name, linestyle=":")
        # else:
        #     plt.plot(data["threads"], data["run_time"], label = name)
        # plt.legend()
        # plt.xlabel("Number of threads")
        # plt.ylabel("Wall time [s]")
        # plt.title(set)

    # Add line for ideal performance scaling
    min_performance = 1
    min_threads = 1
    max_threads = np.max(data["threads"])
    max_performance = min_performance * max_threads

    threads = np.array([min_threads, max_threads])
    performance = ([min_performance, max_performance])

    plt.figure(plot_names[0] + set)
    plt.plot(threads, performance, label = "Linear")
    plt.legend()


for set in sets:
    for plot_name in plot_names:
        plt.figure(plot_name + set)
        plt.savefig("Jac_OMP - " + set + " - " + plot_name, bbox_inches='tight', dpi=300)
