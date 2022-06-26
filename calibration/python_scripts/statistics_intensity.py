import numpy as np
from matplotlib import pyplot as plt

data_path = "/home/godm/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/outputs/"

intensity = np.loadtxt(data_path + "intensity.txt", delimiter="\n")
mean = np.mean(intensity)
sigma = np.std(intensity)
min = np.min(intensity)
max = np.max(intensity)

intensity = intensity[intensity < 30]

num_bins = 300
nums, bins, patches = plt.hist(intensity, bins=num_bins, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("range of variance")
plt.ylabel("number of blocks")
plt.title("histogram of variance")
plt.show()