import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
data_path = data_path = root_path + "/data/lh3_global/log"

# load files #
def load_data(tag1, tag2=None):
    if tag2 is None:
        output = np.loadtxt(data_path + "/" + tag1 + "_result.txt", delimiter="\t")
    else:
        output = np.loadtxt(data_path + "/" + tag1 + "_" + tag2 + "_result.txt", delimiter="\t")
    return output

# visualization #
def visualization(data):
    # print(data)
    plt.plot(data[:, 0], data[:, 1])
    plt.show()
    plt.close()

def visualization3D(data):
    # print(data)
    ax = Axes3D(plt.figure(figsize=(12,8)))
    x, y = np.meshgrid(np.unique(data[:, 0]), np.unique(data[:, 1]))
    z_size = int(np.sqrt(data[:, 2].size))
    ax.plot_surface(x, y, data[:, 2].reshape(z_size, z_size))
    plt.show()
    plt.close()

if __name__=="__main__":
    names = ["rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"]
    idx1 = 0
    idx2 = None
    if (len(sys.argv) > 1):
        idx1 = sys.argv[1]
    if (len(sys.argv) > 2):
        idx1 = sys.argv[1]
        idx2 = sys.argv[2]
    data = load_data(tag1=names[int(idx1)], tag2=names[int(idx2)])
    visualization3D(data)
