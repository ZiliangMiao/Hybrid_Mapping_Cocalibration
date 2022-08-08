from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, interpolate
import os, sys

dataset = "parking"
root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
data_path = root_path + "/data/" + dataset + "/log"

# load files #
def load_data(tag1, tag2=None, spot=0, bw=1):
    if tag2 is None:
        output = np.loadtxt(data_path + "/" + str(tag1) + "_spot_" + str(spot) + "_bw_" + str(bw) + "_result.txt", delimiter="\t")
    else:
        output = np.loadtxt(data_path + "/" + str(tag1) + "_" + str(tag2) + "_result.txt", delimiter="\t")
    return output

# visualization #
def visualization(data, name, bw):
    # print(data)
    scale = 1 / np.max(data[1:, 1])
    param = data[0, :]
    plt.plot(data[1:, 0], data[1:, 1] * scale, label=("bw="+str(bw)))
    func = np.poly1d(np.polyfit(data[1:, 0], data[1:, 1] * scale, 1))
    plt.scatter(param[0], func(param[0]))
    plt.scatter(param[1], func(param[1]))
    plt.title(name)


def visualization3D(data, name="2-axis", cubic_interp=False):
    # print(data)
    ax = Axes3D(plt.figure(figsize=(12,8)))
    x = np.unique(data[:, 0])
    y = np.unique(data[:, 1])
    z = data[:, 2]
    if cubic_interp:
        scale = 2
        x_dense = np.linspace(np.min(x), np.max(x), int(x.size * scale))
        y_dense = np.linspace(np.min(y), np.max(y), int(y.size * scale))
        f = interpolate.interp2d(x, y, z, kind='cubic')
        z = f(x_dense, y_dense)
        x = x_dense
        y = y_dense
        print(z.size)
        
    X, Y = np.meshgrid(x, y)
    Z = z.reshape(int(np.sqrt(z.size)), int(np.sqrt(z.size)))
    ax.plot_surface(X, Y, -Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'))
    plt.title(name)


if __name__=="__main__":
    names = ["rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"]
    idx1 = 0
    idx2 = None
    if (len(sys.argv) > 1):
        # idx1 = int(sys.argv[1])
        for idx1 in range(6):
            bw_list = [16, 4, 1]
            for bw in bw_list:
                data = load_data(tag1=names[idx1], bw=bw)
                visualization(param, data, names[idx1], bw)
            plt.legend()
            plt.show()
            plt.close()
    if (len(sys.argv) > 2):
        idx1 = int(sys.argv[1])
        idx2 = int(sys.argv[2])
        data = load_data(tag1=names[idx1], tag2=names[idx2])
        visualization3D(data, cubic_interp=True)

