import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# with open("datmatmult_gpu2.dat") as f:
#     content = f.readlines()

# content = [x.strip() for x in content]
# content = np.array(content)

# # Split data into runs
# content = np.split(content, len(content)/5)

# lib = pd.DataFrame(columns=["Problem Size[elm]", "Wall time[s]", "Permutation"])

# for (i, run) in enumerate(content):
#     size = int(run[-1].split(' ')[-1])
#     Wtime = (float(run[0].split(' ')[-1]) + float(run[1].split(' ')[-1]) + float(run[2].split(' ')[-1])) /3
#     Perm = str(run[-1].split(' ')[-3])
#     row = [size, Wtime, Perm]
#     lib.loc[i] = row


with open("datmatmult_gpu2.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/2)

data2 = pd.DataFrame(columns=["Problem Size[elm]", "Speed", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Speed = float(run[0].split(' ')[1])
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Speed, Perm]
    data2.loc[i] = row

with open("datmatmult_gpu4.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/2)

data4 = pd.DataFrame(columns=["Problem Size[elm]", "Speed", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Speed = float(run[0].split(' ')[1])
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Speed, Perm]
    data4.loc[i] = row

with open("datmatmult_gpu5.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/2)

data5 = pd.DataFrame(columns=["Problem Size[elm]", "Speed", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Speed = float(run[0].split(' ')[1])
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Speed, Perm]
    data5.loc[i] = row

with open("datmatmult_gpulib.dat") as f:
    content = f.readlines()

content = [x.strip() for x in content]
content = np.array(content)

# Split data into runs
content = np.split(content, len(content)/2)

datalib = pd.DataFrame(columns=["Problem Size[elm]", "Speed", "Permutation"])

for (i, run) in enumerate(content):
    size = int(run[-1].split(' ')[-1])
    Speed = float(run[0].split(' ')[1])
    Perm = str(run[-1].split(' ')[-3])
    row = [size, Speed, Perm]
    datalib.loc[i] = row

plot_names = ["Execution time vs Problem size", "Bla"]

plt.figure(plot_names[0])
plt.plot(data2["Problem Size[elm]"], data2["Speed"], label = data2["Permutation"][1], linestyle=":")
plt.plot(data4["Problem Size[elm]"], data4["Speed"], label = data4["Permutation"][1], linestyle="-")
plt.plot(data5["Problem Size[elm]"], data5["Speed"], label = data5["Permutation"][1], linestyle="-.")
plt.plot(datalib["Problem Size[elm]"], datalib["Speed"], label = datalib["Permutation"][1], linestyle="--")
plt.legend()
plt.xlabel("Problem Size[elm]")
plt.ylabel("Speed[Mflops/s]")
plt.title(plot_names[0])
plt.yscale("log")
plt.xscale("log")
plt.show()