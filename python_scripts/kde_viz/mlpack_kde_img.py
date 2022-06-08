import numpy as np
import matplotlib.pyplot as plt

# data_path = "/home/halsey/Software/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/python_scripts/kde_viz/"
data_path = "/home/halsey/Software/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/outputs/"

########################################################################################################################
# load files #
def load_data(sample_shape, mode):
    # estimation = np.loadtxt(data_path + "outputs/" + mode + "KDE.txt", delimiter="\t").reshape(sample_shape)
    # input_data = np.loadtxt(data_path + "outputs/" + mode + "EdgePix.txt", delimiter="\t")
    # _i = np.loadtxt(data_path + mode + "KDE.txt", delimiter="\t")
    if mode == "cam":
        _cam = np.loadtxt(data_path + "camKDE.txt", delimiter="\t")[:, 2].reshape(sample_shape)
        # _cam = np.loadtxt(data_path + mode + "Trans.txt", delimiter="\t")[:, 2].reshape(sample_shape)
        return np.flip(_cam, axis=0)
    else:
        _lid = np.loadtxt(data_path + mode + "Trans.txt", delimiter=",")[:50000]
        lid = np.zeros(sample_shape)
        for i in range(_lid.shape[0]):
            lid[np.clip(int(_lid[i, 0]), 0, sample_shape[0]-1), np.clip(int(_lid[i, 1]), 0, sample_shape[1]-1)] = 0.1
        return np.flip(lid, axis=0)
########################################################################################################################
# visualization #
def visualization(img_rows, img_cols, input):

    fig, ax = plt.subplots(figsize=(42, 10))
    ax.imshow(input, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              extent=[0, img_cols-1, 0, img_rows-1])
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.show()
    plt.close()
    # fig, ax = plt.subplots(figsize=(42, 10))
    # ax.imshow(np.abs(estimation1 - estimation2), cmap=plt.cm.gist_earth_r,
    #           interpolation='nearest',
    #           origin="upper",
    #           extent=[0, img_cols-1, 0, img_rows-1])
    # # ax.imshow(estimation2, cmap=plt.cm.gist_earth_r,
    # #           interpolation='nearest',
    # #           origin="upper",
    # #           extent=[0, img_cols-1, 0, img_rows-1])
    # # ax.plot(reference_samples[1], img_rows-1-reference_samples[0], 'k.', markersize=0.6)
    # ax.set_xlim([0, img_cols-1])
    # ax.set_ylim([0, img_rows-1])
    # plt.show()
    # plt.close()

def joint_visualization(img_rows, img_cols, input_a, input_b):

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
    
    ax.imshow(input_a, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              )
    ax.imshow(input_b, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              extent=[0, img_cols-1, 0, img_rows-1],
              alpha = 0.5,)
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.savefig("/home/halsey/Desktop/output/pytest.png")
    # plt.show()
    # plt.close()


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
    lid = load_data(lid_sample_shape, lid_mode)
    cam = load_data(cam_sample_shape, cam_mode)
    # visualization(img_rows, img_cols, cam)
    # visualization(img_rows, img_cols, lid)
    joint_visualization(img_rows, img_cols, lid, cam)
