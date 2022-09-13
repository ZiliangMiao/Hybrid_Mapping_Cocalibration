import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
import os, sys

root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
data_path = data_path = root_path + "/bin/outputs/lh3/2/"

# load files #
def load_data(type, tag):
    output = np.loadtxt(data_path + "/" + type + "_cost" + "/" + tag + "_result.txt", delimiter="\t")
    return output

# visualization #
def visualization(data, type):
    scale = 1 / np.max(data[:, 1] - np.min(data[:, 1]))
    interp_scale = 2
    f = interpolate.interp1d(data[:, 0], (data[:, 1] - np.min(data[:, 1])) * scale, kind='cubic')
    plot_x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), int(data[:, 0].size * interp_scale))
    plt.plot(plot_x, f(plot_x), linewidth=1.7, label=type)
    plt.axvline(data[np.argmax(data[:, 1]), 0], c='grey', alpha=0.35)
    

if __name__=="__main__":
    # names = ["rx", "ry", "rz",
    #         "tx", "ty", "tz"]
    names = ["rx", "rz",
            "tx", "tz"]
    # plt.style.use("seaborn-dark-palette")
    for idx in range(len(names)):
        plt.figure(figsize=(4.80, 3.20))
        plt.tick_params(labelsize=11)

        data = load_data(type="KDE", tag=names[int(idx)])
        visualization(data, type="KDE")
        data = load_data(type="MI", tag=names[int(idx)])
        visualization(data, type="MI")
        
        plt.title(names[int(idx)])
        plt.legend()
        plt.show()
        plt.close()
