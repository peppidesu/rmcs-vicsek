import matplotlib.pyplot as plt
import numpy as np
import json

data = None
with open("data.json") as file:
    data = json.load(file)

fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

xaxis = np.arange(0, data["timesteps"] + 1, data["sliceInterval"])

for groupName, groupData in data["results"].items():
    error = np.array(list(groupData["std"].values()))
    ax.errorbar(xaxis, groupData["mean"].values(), label=groupName, yerr=error)

ax.legend()
ax.set_xlabel('timesteps')
ax.set_ylabel('alive agents')
# xaxis = []
# final_data = []
# for groupName, groupData in data["results"].items():
#     final_data_group = []
#     xaxis.append(float(groupName[-3:]))
#     for sample in groupData["data"]:
#         final_data_group.append(list(sample.values())[-1])
#     final_data.append(final_data_group)

# ax.boxplot(final_data, positions=xaxis, widths=0.15, patch_artist=True)
# ax.set_xlabel('alpha')
# ax.set_ylabel('alive agents after 50k timesteps')
plt.show()