import numpy as np
import matplotlib.pyplot as plt

# data_path = "/home/halsey/Software/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/python_scripts/kde_viz/"
data_path = "/home/halsey/Software/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/outputs/"

########################################################################################################################
# load files #
def load_data(sample_shape, mode):
    # estimation = np.loadtxt(data_path + "outputs/" + mode + "KDE.txt", delimiter="\t").reshape(sample_shape)
    # input_data = np.loadtxt(data_path + "outputs/" + mode + "EdgePix.txt", delimiter="\t")
    estimation = np.loadtxt(data_path + mode + "KDE.txt", delimiter="\t").reshape(sample_shape)
    input_data = np.loadtxt(data_path + mode + "EdgePix.txt", delimiter="\t")
    input_data = np.unique(input_data, axis=0).T
    reference_samples = np.vstack((input_data[0, :], input_data[1, :]))
    sum = np.sum(estimation)
    print("sum of estimations: ", sum)
    return reference_samples, estimation
########################################################################################################################
# visualization #
def visualization(img_rows, img_cols, reference_samples, estimation1, estimation2):

    fig, ax = plt.subplots(figsize=(42, 10))
    ax.imshow(estimation1, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              extent=[0, img_cols-1, 0, img_rows-1])
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(42, 10))
    ax.imshow(estimation2, cmap=plt.cm.gist_earth_r,
              interpolation='nearest',
              origin="upper",
              extent=[0, img_cols-1, 0, img_rows-1])
    ax.set_xlim([0, img_cols-1])
    ax.set_ylim([0, img_rows-1])
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(42, 10))
    ax.imshow(np.abs(estimation1 - estimation2), cmap=plt.cm.gist_earth_r,
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
########################################################################################################################

if __name__=="__main__":
    # parameters of lidar #
    lid_img_rows = 1223
    lid_img_cols = 4000
    # lid_sample_shape = (140, 800)
    lid_sample_shape = (200, 800)
    lid_mode = "lid"
    # parameters of camera #
    cam_img_rows = 1223
    cam_img_cols = 4000
    cam_sample_shape = (200, 800)
    cam_mode = "cam"
    # load and visualize
    cam_reference_samples, cam_estimation = load_data(cam_sample_shape, cam_mode)
    # visualization(cam_img_rows, cam_img_cols, cam_reference_samples, cam_estimation)

    lid_reference_samples, lid_estimation = load_data(lid_sample_shape, lid_mode)
    visualization(lid_img_rows, lid_img_cols, lid_reference_samples, cam_estimation, lid_estimation)
