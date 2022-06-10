import numpy as np
from matplotlib import pyplot as plt

data_path = "/home/godm/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/outputs/"

curvature = np.loadtxt(data_path + "curvature.txt", delimiter="\n")
mean = np.mean(curvature)
sigma = np.std(curvature)
min = np.min(curvature)
max = np.max(curvature)

curvature = curvature[curvature < 0.3]

num_bins = 300
nums, bins, patches = plt.hist(curvature, bins=num_bins, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel("range of variance")
plt.ylabel("number of blocks")
plt.title("histogram of variance")
plt.show()