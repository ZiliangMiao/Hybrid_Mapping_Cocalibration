import os, sys
import numpy as np
import cv2

def contour_filter(contour, len_threshold=100):
    invalid_cam = 0
    for i in range(len(contour) - invalid_cam):
        dist = np.size(contour[i - invalid_cam])
        end_dist = np.sum(np.abs(contour[i - invalid_cam][0, 0, :] - contour[i - invalid_cam][-1, 0, :]))
        if (dist < len_threshold and 8 * end_dist < dist) or (dist < len_threshold / 8):
            contour = contour[:(i - invalid_cam)] + contour[(i - invalid_cam) + 1:]
            invalid_cam += 1
        if i - invalid_cam == len(contour) - 2:
            break
    return contour

def patch_image(image, mode=cv2.MORPH_CLOSE, size=3, iter=1):
    kernel = np.ones((size, size), dtype=np.uint8)
    image = cv2.morphologyEx(image, mode, kernel, iter)
    return image

def blur(edge_lid, h_u, h_l):
    if (len(edge_lid.shape) == 2):
        edge_lid = cv2.cvtColor(edge_lid, cv2.COLOR_GRAY2BGR)
    edge_lid_u = edge_lid[:int(0.15 * edge_lid.shape[0]), :, :]
    edge_lid_l = edge_lid[int(0.15 * edge_lid.shape[0]):, :, :]
    edge_lid_u = cv2.pyrMeanShiftFiltering(edge_lid_u, h_u, 2*h_u)
    edge_lid_l = cv2.pyrMeanShiftFiltering(edge_lid_l, h_l, 2*h_l)
    edge_lid_u = cv2.cvtColor(edge_lid_u, cv2.COLOR_BGR2GRAY)
    edge_lid_l = cv2.cvtColor(edge_lid_l, cv2.COLOR_BGR2GRAY)
    edge_lid_u = cv2.fastNlMeansDenoising(edge_lid_u, h=h_l, searchWindowSize=21, templateWindowSize=7)
    edge_lid_l = cv2.fastNlMeansDenoising(edge_lid_l, h=h_l, searchWindowSize=21, templateWindowSize=7)
    edge_lid = np.vstack((edge_lid_u, edge_lid_l))
    return edge_lid

def check_folder():
    dir_list = [
        data_path + "/edges/canny_outputs/",
        data_path + "/edges/hog_outputs/"
    ]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)

if __name__ == "__main__":
    kArgs = 4
    if not (len(sys.argv) >= kArgs):
        print("Edge extraction failed.")
    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
    dataset_path = sys.argv[1]
    mode = sys.argv[2]
    spot = sys.argv[3]
    
    data_path = dataset_path + "/spot" + str(spot) + "/0"
    print("Edge extraction in: " + data_path + " ... ", end="")

    dir_cam_original = data_path + "/outputs/omni_outputs/flat_image.bmp"
    dir_cam_mask = root_path + "/python_scripts/image_process/flat_image_mask.png"
    dir_cam_filtered = data_path + "/edges/canny_outputs/omni_1_filtered.png"
    dir_cam_canny = data_path + "/edges/canny_outputs/omni_2_canny.png"
    dir_cam_output = data_path + "/edges/omni_edge.png"

    dir_lid_original = data_path + "/outputs/lidar_outputs/flat_lidar_image.bmp"
    dir_lid_mask = root_path + "/python_scripts/image_process/flat_lidar_image_mask.png"
    dir_lid_filtered = data_path + "/edges/canny_outputs/lid_1_filtered.png"
    dir_lid_canny = data_path + "/edges/canny_outputs/lid_2_canny.png"
    dir_lid_output = data_path + "/edges/lidar_edge.png"

    check_folder()

    # -------- omnidirectional camera --------

    if (mode == "omni"):
        edge_cam_raw = cv2.imread(dir_cam_original)
        edge_cam_raw = cv2.cvtColor(edge_cam_raw, cv2.COLOR_BGR2GRAY)

        edge_cam = cv2.GaussianBlur(edge_cam_raw, sigmaX=1, sigmaY=1, ksize=(5, 5))
        # edge_cam = blur(edge_cam_raw, h_u=10, h_l=5)
        cv2.imwrite(dir_cam_filtered, edge_cam)

        # remove the black region
        edge_cam = cv2.Canny(image=edge_cam, threshold1=25, threshold2=50)
        mask_cam = cv2.imread(dir_cam_mask, cv2.IMREAD_GRAYSCALE)
        edge_cam = cv2.bitwise_and(edge_cam, mask_cam)
        cv2.imwrite(dir_cam_canny, cv2.bitwise_or(edge_cam_raw, edge_cam))

        # contour filter
        cnt_cam, hierarchy_cam = cv2.findContours(edge_cam, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt_cam_out = contour_filter(contour=cnt_cam, len_threshold=150)
        edge_cam_out = np.zeros(np.shape(edge_cam), np.uint8)
        cv2.drawContours(edge_cam_out, cnt_cam_out, -1, 255, 1)
        cv2.imwrite(dir_cam_output, edge_cam_out)
        print("done.")

    # -------- Lidar --------

    else:
        edge_lid_raw = cv2.imread(dir_lid_original)
        edge_lid_raw = cv2.cvtColor(edge_lid_raw, cv2.COLOR_BGR2GRAY)
        # edge_lid_raw = cv2.GaussianBlur(edge_lid_raw, sigmaX=0.5, sigmaY=0.5, ksize=(5, 5))
        edge_lid = blur(edge_lid_raw, h_u=20, h_l=10)
        cv2.imwrite(dir_lid_filtered, edge_lid)

        # mask to remove the upper and lower bound noise
        edge_lid = cv2.Canny(image=edge_lid, threshold1=25, threshold2=50)
        mask_lid = cv2.imread(dir_lid_mask, cv2.IMREAD_GRAYSCALE)
        edge_lid = cv2.bitwise_and(edge_lid, mask_lid)
        cv2.imwrite(dir_lid_canny, cv2.bitwise_or(edge_lid_raw, edge_lid))
        
        # contour filter
        cnt_lid, hierarchy_lid = cv2.findContours(edge_lid, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt_lid = contour_filter(contour=cnt_lid, len_threshold=150)
        edge_lid = np.zeros(edge_lid.shape, np.uint8)
        cv2.drawContours(edge_lid, cnt_lid, -1, 255, 1)

        cv2.imwrite(dir_lid_output, edge_lid)
        print("done.")

    if (len(sys.argv) >= kArgs):
        print("Edge extraction completed.")

 