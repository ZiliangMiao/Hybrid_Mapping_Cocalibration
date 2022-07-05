import numpy as np
import matplotlib.pyplot as plt
import os, sys

root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
data_path = data_path = root_path + "/data/lh3/spot0/0/outputs"

########################################################################################################################
# load files #
def load_data(sample_shape, mode):
    # input_data = np.loadtxt(data_path + "outputs/" + mode + "EdgePix.txt", delimiter="\t")
    # _i = np.loadtxt(data_path + mode + "KDE.txt", delimiter="\t")
    if mode == "kde":
        _cam = np.loadtxt(data_path + "/fisheye_outputs/camKDE.txt", delimiter="\t")[:, 2]
        auto_scale = np.sqrt((sample_shape[0] * sample_shape[1] / np.size(_cam)))
        print("scale: " + str(1/auto_scale))
        _cam = _cam.reshape(((int)(sample_shape[0] / auto_scale), (int)(sample_shape[1] / auto_scale)))
        # _cam = np.loadtxt(data_path + mode + "Trans.txt", delimiter="\t")[:, 2].reshape(sample_shape)
        return np.flip(_cam, axis=0)
    elif mode == "cam":
        _cam = np.loadtxt(data_path + "/fisheye_outputs/camPixOut.txt", delimiter="\t").astype(int)
        return np.flip(_cam, axis=0)
    elif mode == "lid":
        _lid = np.loadtxt(data_path + "/lidar_outputs/lidTrans.txt", delimiter=",")[:100000]
        lid = np.zeros(sample_shape)
        for i in range(_lid.shape[0]):
            lid[np.clip(int(_lid[i, 0]), 0, sample_shape[0]-1), np.clip(int(_lid[i, 1]), 0, sample_shape[1]-1)] = 0.1
        return np.flip(lid, axis=0)

########################################################################################################################
# visualization #
def visualization(img_rows, img_cols, cam):

    fig, ax = plt.subplots(figsize=(42, 10))
    ax.scatter(cam[:, 0], cam[:, 1])
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.show()
    plt.close()


def joint_visualization(img_rows, img_cols, lid, kde):

    # fig, ax = plt.subplots(1, 2, figsize=(42, 10))
    # ax[0].imshow(input_a, cmap=plt.cm.gist_earth_r,
    #           interpolation='nearest',
    #           origin="upper")
    # ax[1].imshow(input_b, cmap=plt.cm.gist_earth_r,
    #           interpolation='nearest',
    #           origin="upper",
    #           extent=[0, img_cols-1, 0, img_rows-1])
    # ax[0].set_xlim([0, img_cols-1])
    # ax[0].set_ylim([0, img_rows-1])
    # ax[1].set_xlim([0, img_cols-1])
    # ax[1].set_ylim([0, img_rows-1])
    # plt.show()
    # plt.close()

    fig, ax = plt.subplots(figsize=(24.48, 20.48))
    
    ax.imshow(lid, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              )
    ax.imshow(kde, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              extent=[0, img_cols-1, 0, img_rows-1],
              alpha = 0.5,)
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.savefig("/home/halsey/Desktop/pytest.png")
    plt.show()


########################################################################################################################

if __name__=="__main__":
    # parameters of lidar #
    img_rows = 2048
    img_cols = 2448
    cam_sample_shape = (2048, 2448)
    lid_sample_shape = (2048, 2448)
    cam_mode = "cam"
    lid_mode = "lid"
    # load and visualize
    lid = load_data(lid_sample_shape, "lid")
    kde = load_data(cam_sample_shape, "kde")
    cam = load_data(cam_sample_shape, "cam")
    # visualization(img_rows, img_cols, cam)
    # visualization(img_rows, img_cols, lid)
    joint_visualization(img_rows, img_cols, lid, kde)
    visualization(img_rows, img_cols, cam)
