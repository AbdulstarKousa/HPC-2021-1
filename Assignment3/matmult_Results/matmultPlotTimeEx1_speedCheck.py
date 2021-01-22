import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("datmatmult_time_lib_single.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)


# Split data into runs
content = np.split(content, 7)

lib = pd.DataFrame(columns=["Problem Size[elm]", "Wall time[s]", "Permutation"])
print(content[0])
for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Wtime = (float(run[0].split(' ')[-1]) + float(run[1].split(' ')[-1]) + float(run[2].split(' ')[-1])) /3
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Wtime, Perm]
    lib.loc[i] = row


with open("datmatmult_time_lib.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/5)

lib2 = pd.DataFrame(columns=["Problem Size[elm]", "Wall time[s]", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Wtime = (float(run[0].split(' ')[-1]) + float(run[1].split(' ')[-1]) + float(run[2].split(' ')[-1])) /3
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Wtime, Perm]
    lib2.loc[i] = row


plot_names = ["Single thread, CPU vs GPU", "Bla"]

plt.figure(plot_names[0])
plt.plot(lib["Problem Size[elm]"], lib["Wall time[s]"], label = lib["Permutation"][1], linestyle="-")
plt.plot(lib["Problem Size[elm]"], lib["Wall time[s]"], label = "libmulti", linestyle=":")
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