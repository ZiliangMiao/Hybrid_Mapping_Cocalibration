import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpolate
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

def visualization3D(data, cubic_interp=False):
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
    plt.show()
    plt.close()

if __name__=="__main__":
    names = ["rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"]
    init_value = [
        0.00513968, 3.13105, 1.56417,
        0.250552, 0.0014601, 0.0765269,
        1020.0, 1198.0,
        1888.37, -536.802, -19.6401, -17.8592, 6.34771,
        0.996981, -0.00880807, 0.00981348
    ]
    idx1 = 0
    idx2 = None

    np.set_printoptions(precision=5)
    if (len(sys.argv) > 1 and len(sys.argv) <= 2):
        idx1 = int(sys.argv[1])
        data = load_data(tag1=names[idx1])
        if (np.any(data[:, 0] == 0)):
            print("Converting ... ")
            data[:, 0] = data[:, 0] + init_value[idx1]
            np.savetxt(names[idx1] + "_result.txt",
                        data, delimiter="\t")
        visualization(data)
    if (len(sys.argv) > 2):
        idx1 = int(sys.argv[1])
        idx2 = int(sys.argv[2])
        data = load_data(tag1=names[idx1], tag2=names[idx2])
        if (np.any(data[:, 0] == 0)):
            print("Converting ... ")
            data[:, 0] = data[:, 0] + init_value[idx1]
            data[:, 1] = data[:, 1] + init_value[idx2]
            np.savetxt(names[idx1] + "_" + names[idx2] + "_result.txt",
                        data, delimiter="\t")
        visualization3D(data, cubic_interp=True)
