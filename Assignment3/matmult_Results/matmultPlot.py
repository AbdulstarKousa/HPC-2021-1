import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open("datmatmult__gpu1.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/5)
# print(content)
# content = content[1:]

# # Reverse array to have run with 1 thread at the top
# content = np.flip(content, axis=0)

# # Build data frame from data
# data = pd.DataFrame(columns=["run_time", "iterations", "norm", "threads", "lattice_updates"])

# for (i, run) in enumerate(content):
#     time = float(run[1].split(' ')[-1])
#     norm = float(run[2].split(' ')[-1])
#     iterations = int(run[3].split(' ')[-1])
#     threads = int(run[4].split(' ')[-1])
#     lattice_updates_per_second = float(iterations)*(float(size)**3)/float(time)

#     row = np.array([time, iterations, norm, threads, lattice_updates_per_second])
#     data.loc[i] = row

# data.iloc[0]["lattice_updates"]
# data["performance"] = data["lattice_updates"]/data.iloc[0]["lattice_updates"]

# plt.figure(plot_names[0] + set)
# if name == "Ofast":
#     plt.plot(data["threads"], data["performance"], label = name, linestyle=":")
# else:
#     plt.plot(data["threads"], data["performance"], label = name)
# plt.legend()
# plt.xlabel("Number of threads")
# plt.ylabel("Speed-up (rel. to 1 thread)")
# plt.title(set)

# plt.figure(plot_names[1] + set)
# if name == "Ofast":
#     plt.plot(data["threads"], data["lattice_updates"], label = name, linestyle=":")
# else:
#     plt.plot(data["threads"], data["lattice_updates"], label = name)
# plt.legend()
# plt.xlabel("Number of threads")
# plt.ylabel("Lattice-site updates per second")
# plt.title(set)

# ### Wall time
# # plt.figure(plot_names[1] + set)
# # if name == "Ofast":
# #     plt.plot(data["threads"], data["run_time"], label = name, linestyle=":")
# # else:
# #     plt.plot(data["threads"], data["run_time"], label = name)
# # plt.legend()
# # plt.xlabel("Number of threads")
# # plt.ylabel("Wall time [s]")
# # plt.title(set)