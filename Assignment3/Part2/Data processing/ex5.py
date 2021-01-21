import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = [r"../Results_ex5/Exercise5_CPU_100.dat", r"../Results_ex5/Exercise5_GPU_100.dat"]
names = ["CPU", "GPU"]

for (fi, file) in enumerate(files):
    name = names[fi]
    with open(file) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    content = np.array(content)

    runs = np.split(content, len(content)/2)

    data = pd.DataFrame(columns=["size", "time"])

    for (i, run) in enumerate(runs):
        time = float(run[0].split(' ')[-1])
        size = int(run[1].split(' ')[1])
        row = [size, time]
        data.loc[i] = row

    plt.plot(data["size"], data["time"], label = name)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Problem size: N")
    plt.ylabel("Total time")
    # plt.title("Title")
