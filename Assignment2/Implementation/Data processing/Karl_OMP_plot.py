import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parse the two files
file_a = r"datJacOMPReduction12Threads100x.dat"
file_b = r"datJacOMPReduction12Threads.dat"

with open(file_a) as f:
    content_a = f.readlines()

content_a = [x.strip() for x in content_a]
content_a = np.array(content_a)

with open(file_b) as f:
    content_b = f.readlines()

content_b = [x.strip() for x in content_b]
content_b = np.array(content_b)

# Use only one of the experiments from file b
content_b = np.split(content_b, np.where(content_b == content_b[-1])[0])
content_b = np.append(content_b[0], content_b[2])

# Collect the data from the two files
contents = [content_a, content_b]
content_names = ["100x", "Not 100x"]

for (problem, content) in enumerate(contents):
    run_settings = content[-1]
    cube_size = int(run_settings.split(' ')[1])

    content = content[0:-1]

    # Cleaning up data. Not the nicest way, but it works
    content = content[content != "Running Jacobi OMP"]
    # content = content[content != "static"]

    split_indices = np.where(content == "static")

    content = np.split(content, split_indices[0])

    for (i, block) in enumerate(content):
        content[i] = block[block != "static"]

    for i in range(len(content)):
        if len(content[i]) == 0:
            np.delete(content, i)

    valid_entries = np.array(list([len(x) > 0 for x in content]), dtype=bool)
    content = np.array(content)[valid_entries]

    # Put data into pandas data frame
    data = pd.DataFrame(columns=["Iterations", "run_time", "norm", "thread_count"])

    for (i, run) in enumerate(content):
        iterations = int(run[0].split(' ')[-1])
        run_time = float(run[1].split(' ')[-1])
        norm = float(run[2].split(' ')[-1])
        thread_count = int(run[3].split(' ')[-1])
        row = np.array([iterations, run_time, norm, thread_count])
        data.loc[i] = row

    # Add calculation as column to data
    data["value"] = data["Iterations"] * cube_size/data["run_time"]

    plt.plot(data["thread_count"], data["value"], label=content_names[problem])

plt.legend()











#
