import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("datmatmult_time_lib.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/5)

lib = pd.DataFrame(columns=["Problem Size[elm]", "Wall time[s]", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Wtime = (float(run[0].split(' ')[-1]) + float(run[1].split(' ')[-1]) + float(run[2].split(' ')[-1])) /3
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Wtime, Perm]
    lib.loc[i] = row


with open("datmatmult_time_gpu1.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/8)

data = pd.DataFrame(columns=["Problem Size[elm]", "Kernel wall time[s]", "Memory wall time[s]", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Wtime = float(float(run[0].split(' ')[-1]) + float(run[2].split(' ')[-1]) + float(run[4].split(' ')[-1])) /3
    Mtime = float(float(run[1].split(' ')[-1]) + float(run[3].split(' ')[-1]) + float(run[5].split(' ')[-1])) /3
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Wtime, Mtime, Perm]
    data.loc[i] = row

with open("datmatmult_time_gpu2.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/8)

data2 = pd.DataFrame(columns=["Problem Size[elm]", "Kernel wall time[s]", "Memory wall time[s]", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Wtime = float(float(run[0].split(' ')[-1]) + float(run[2].split(' ')[-1]) + float(run[4].split(' ')[-1])) /3
    Mtime = float(float(run[1].split(' ')[-1]) + float(run[3].split(' ')[-1]) + float(run[5].split(' ')[-1])) /3
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Wtime, Mtime, Perm]
    data2.loc[i] = row


sets = ["lib", "gpu1", "gpu2"]
plot_names = ["Execution time vs Problem size", "Bla"]

plt.figure(plot_names[0])
plt.plot(lib["Problem Size[elm]"], lib["Wall time[s]"], label = lib["Permutation"][1], linestyle="-")
plt.plot(data["Problem Size[elm]"], data["Kernel wall time[s]"] + data["Memory wall time[s]"], label = data["Permutation"][1], linestyle="--")
plt.plot(data2["Problem Size[elm]"], data2["Kernel wall time[s]"] + data2["Memory wall time[s]"], label = "gpu2 wall time", linestyle=":")
plt.plot(data2["Problem Size[elm]"], data2["Memory wall time[s]"], label = "gpu2 memory time", linestyle=":")
plt.legend()
plt.xlabel("Problem Size[elm]")
plt.ylabel("Total wall time [s]")
plt.title(plot_names[0])
plt.yscale("log")
plt.xscale("log")
plt.show()

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