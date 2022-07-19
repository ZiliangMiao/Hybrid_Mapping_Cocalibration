import numpy as np
import matplotlib.pyplot as plt
import os, sys

root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
data_path = data_path = root_path + "/data/lh3_global/log"

# load files #
def load_data():
    output = np.loadtxt(data_path + "/rz_results.txt", delimiter=",")
    return output

# visualization #
def visualization(data):
    print(data)
    plt.plot(data[:, 0], data[:, 1])
    plt.show()
    plt.close()

if __name__=="__main__":
    data = load_data()
    visualization(data)
