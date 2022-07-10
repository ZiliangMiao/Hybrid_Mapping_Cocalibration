from logging import root
import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import _hog, _hoghistogram

def _hog_channel_gradient(channel, sobel=True, remove=True):
    g_row_side = np.empty(channel.shape, dtype=np.double)
    g_col_side = np.empty(channel.shape, dtype=np.double)
    if sobel:
        g_row_side = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        g_col_side = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        g_max = np.max([np.max(g_row_side), np.max(g_col_side)])
        g_col_side = g_col_side / g_max * 255
        g_row = g_row_side / g_max * 255
    else:
        g_row_side[1:-1, :] = channel[2:, :] - channel[:-2, :]
        g_col_side[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    if remove:
        bool = ((-np.sign(channel-127) + 1) * 0.5).astype(np.uint8)
        g_row_side = bool * g_row_side
        g_col_side = bool * g_col_side
        # remove the gradient which are less than 0
        g_row_side[g_row_side < 0] = 0
        g_col_side[g_col_side < 0] = 0
        g_row = np.zeros((g_row_side.shape[0], g_row_side.shape[1]), float)
        g_col = np.zeros((g_col_side.shape[0], g_col_side.shape[1]), float)
        g_row[1:, :] = g_row_side[:-1, :]
        g_col[:, 1:] = g_col_side[:, :-1]
        cv2.imwrite(data_path + "/edges/hog_outputs/g_row.png", g_row)
        cv2.imwrite(data_path + "/edges/hog_outputs/g_col.png", g_col)
    return g_row, g_col

def get_hog(img, orientations=9, size=32, visualize=True):
    pixels_per_cell = (size, size)
    cells_per_block = (1, 1)
    block_norm = 'L1'
    if len(img.shape) == 3:
        img = img[:, :, 0]
    elif len(img.shape) != 2:
        raise ValueError('invalid input channels')
    if img.dtype.kind == 'u':
        img = img.astype('float')

    g_row, g_col = _hog_channel_gradient(img)
    s_row, s_col = img.shape[:2]
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations))

    _hoghistogram.hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                                 n_cells_col, n_cells_row,
                                 orientations, orientation_histogram)

    # now compute the histogram for each cell
    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, orientations))

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = \
                _hog._hog_normalize_block(block, method=block_norm)
    normalized_blocks = normalized_blocks.squeeze()

    if visualize:
        from skimage import draw
        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
                np.pi * (orientations_arr + .5) / orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = draw.line(int(centre[0] - dc),
                                       int(centre[1] + dr),
                                       int(centre[0] + dc),
                                       int(centre[1] - dr))
                    hog_image[rr, cc] += normalized_blocks[r, c, o]

        return normalized_blocks, (255 * hog_image).astype(np.uint8)
    else:
        return normalized_blocks

def get_block_idx(var_bound, vars):
    vars[vars < var_bound] = 0
    idx = np.nonzero(vars)  # return the indices of non-zero elements
    idx_rows = idx[0]
    idx_cols = idx[1]
    return idx_rows, idx_cols

def get_pixel(block_rows, block_cols, block_size):
    pix_rows_lb = 0 + block_rows * block_size  # 0 32
    pix_rows_ub = (block_rows + 1) * block_size - 1  # 31 63
    pix_cols_lb = 0 + block_cols * block_size  # 0 32
    pix_cols_ub = (block_cols + 1) * block_size - 1  # 31 63
    return pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub

def get_roi(img, pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub):

    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(len(pix_rows_ub)):
        roi = img[pix_rows_lb[i]:pix_rows_ub[i] + 1, pix_cols_lb[i]:pix_cols_ub[i] + 1]
        output[pix_rows_lb[i]:pix_rows_ub[i] + 1, pix_cols_lb[i]:pix_cols_ub[i] + 1] = roi

    return output

def get_var_bound(hog_blocks, var_proportion, num_bins):
    vars = np.var(hog_blocks, axis=2)
    # get rid of the zeros in variances
    var = vars[vars != 0]
    # plot the histogram of variances
    nums, bins, patches = plt.hist(var.ravel(), bins=num_bins, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlabel("range of variance")
    # plt.ylabel("number of blocks")
    # plt.title("histogram of variance")
    # plt.show()
    # get the bound of bin which numbers accumulated to 70%
    accumulation = 0
    index = 0
    for i in range(len(nums)):
        accumulation += nums[i]
        proportion = accumulation / np.sum(nums)
        if proportion > var_proportion:
            index = i
            break

    bin_lb = bins[index]
    bin_ub = bins[index + 1]
    return vars, bin_lb, bin_ub

def hog_filter(img_cam, orien, block_size, var_proportion, num_bins):
    hog_blocks, img_hog = get_hog(img=img_cam, orientations=orien, size=block_size)
    # get the variance bound
    vars, bin_lb, bin_ub = get_var_bound(hog_blocks, var_proportion, num_bins)
    # get the index of target blocks
    idx_rows, idx_cols = get_block_idx(bin_lb, vars)
    # get the indices of target pixels
    pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub = get_pixel(idx_rows, idx_cols, block_size)
    # get the pixel region of interest
    img_filtered = get_roi(img_cam, pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub)
    return img_filtered, img_hog

def extremum_filter(img, orien, block_size):
    hog_blocks, img_hog = get_hog(img=img, orientations=orien, size=block_size)

    oriens = hog_blocks.reshape(-1, orien)
    oriens[np.all(oriens < 0.7, axis=1)] = 0

    hog = oriens.reshape(len(hog_blocks), -1, orien)
    hog_sum = np.sum(hog, axis=2)
    idx = np.nonzero(hog_sum)  # return the indices of non-zero elements
    idx_rows = idx[0]
    idx_cols = idx[1]
    # get the indices of target pixels
    pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub = get_pixel(idx_rows, idx_cols, block_size)
    # get the pixel region of interest
    img_filtered = get_roi(img, pix_rows_lb, pix_rows_ub, pix_cols_lb, pix_cols_ub)
    return img_filtered

def principal_component_filter(img, orien, block_size):
    ######## TBD ########
    hog_blocks, _ = get_hog(img=img, orientations=orien, size=block_size)
    if img.dtype.kind == 'u':
        img = img.astype('float')
    g_row, g_col = _hog_channel_gradient(img)
    direct = np.arctan2(g_row, g_col)
    principal_direct = np.argmax(hog_blocks, axis=2)

def contour_filter(contour, len_threshold=200):
    invalid_cam = 0
    for i in range(len(contour) - invalid_cam):
        dist = len(contour[i - invalid_cam])
        # dist = np.sum(np.abs(contour[i - invalid_cam][1:, 0, :] - contour[i - invalid_cam][:-1, 0, :]))
        end_dist = np.sum(np.abs(contour[i - invalid_cam][0, 0, :] - contour[i - invalid_cam][-1, 0, :]))
        # area = cv2.contourArea(contour[i - invalid_cam])
        if (dist < len_threshold and 8 * end_dist < dist) or (dist < len_threshold / 4):
            contour = contour[:(i - invalid_cam)] + contour[(i - invalid_cam) + 1:]
            invalid_cam += 1
        if i - invalid_cam == len(contour) - 2:
            break
    return contour

def patch_image(image, mode=cv2.MORPH_OPEN, size=3, iter=1):
    kernel = np.ones((size, size), dtype=np.uint8)
    image = cv2.morphologyEx(image, mode, kernel, iter)
    return image

def nlmeans(edge_lid, h_u, h_l):
    # h = 40/20 for normal lidar images;
    # h = 20/10 for full lidar image;
    edge_lid_u = edge_lid[:int(0.15 * edge_lid.shape[0]), :]
    edge_lid_l = edge_lid[int(0.15 * edge_lid.shape[0]):, :]
    edge_lid_u = cv2.fastNlMeansDenoising(edge_lid_u, h=h_u, searchWindowSize=21, templateWindowSize=7)
    edge_lid_l = cv2.fastNlMeansDenoising(edge_lid_l, h=h_l, searchWindowSize=21, templateWindowSize=7)
    edge_lid = np.vstack((edge_lid_u, edge_lid_l))
    return edge_lid

def black_region_removal(img, pix_rows_bound):
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    output[pix_rows_bound:, :] = img[pix_rows_bound:, :]
    return output

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

    dir_cam_original = data_path + "/outputs/fisheye_outputs/flatImage.bmp"
    dir_cam_mask = root_path + "/python_scripts/image_process/flatImage_mask.png"
    dir_cam_filtered = data_path + "/edges/canny_outputs/cam_1_filtered.png"
    dir_cam_canny = data_path + "/edges/canny_outputs/cam_2_canny.png"
    dir_cam_output = data_path + "/edges/camEdge.png"

    dir_lid_original = data_path + "/outputs/lidar_outputs/flatLidarImage.bmp"
    dir_lid_mask = root_path + "/python_scripts/image_process/flatLidarImage_mask.png"
    dir_lid_filtered = data_path + "/edges/canny_outputs/lid_1_filtered.png"
    dir_lid_canny = data_path + "/edges/canny_outputs/lid_2_canny.png"
    dir_lid_output = data_path + "/edges/lidEdge.png"

    check_folder()

    # -------- fisheye camera --------

    if(mode == "fisheye"):
        edge_cam = cv2.imread(dir_cam_original)
        edge_cam = cv2.cvtColor(edge_cam, cv2.COLOR_BGR2GRAY)

        edge_cam = cv2.GaussianBlur(edge_cam, sigmaX=1.5, sigmaY=1.5, ksize=(5, 5))
        cv2.imwrite(dir_cam_filtered, edge_cam)

        # remove the black region
        # edge_cam = black_region_removal(edge_cam, pix_rows_bound=435)

        edge_cam = cv2.Canny(image=edge_cam, threshold1=30, threshold2=50)
        mask_cam = cv2.imread(dir_cam_mask, cv2.IMREAD_GRAYSCALE)
        edge_cam = cv2.bitwise_and(edge_cam, mask_cam)
        cv2.imwrite(dir_cam_canny, edge_cam)

        # contour filter
        cnt_cam, hierarchy_cam = cv2.findContours(edge_cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_cam = contour_filter(contour=cnt_cam, len_threshold=200)
        edge_cam = np.zeros(edge_cam.shape, np.uint8)
        cv2.drawContours(edge_cam, cnt_cam, -1, 255, 1)
        cv2.imwrite(dir_cam_output, edge_cam)
        print("done.")

    # -------- Lidar --------

    else:
        edge_lid = cv2.imread(dir_lid_original)
        edge_lid = cv2.cvtColor(edge_lid, cv2.COLOR_BGR2GRAY)
        # edge_lid = cv2.fastNlMeansDenoising(edge_lid, h=10, searchWindowSize=21, templateWindowSize=7)
        edge_lid = nlmeans(edge_lid, h_u=40, h_l=20)
        cv2.imwrite(dir_lid_filtered, edge_lid)

        # mask to remove the upper and lower bound noise
        edge_lid = cv2.Canny(image=edge_lid, threshold1=30, threshold2=50)
        mask_lid = cv2.imread(dir_lid_mask, cv2.IMREAD_GRAYSCALE)
        edge_lid = cv2.bitwise_and(edge_lid, mask_lid)
        cv2.imwrite(dir_lid_canny, edge_lid)
        
        # contour filter
        cnt_lid, hierarchy_lid = cv2.findContours(edge_lid, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnt_lid = contour_filter(contour=cnt_lid, len_threshold=200)
        edge_lid = np.zeros(edge_lid.shape, np.uint8)
        cv2.drawContours(edge_lid, cnt_lid, -1, 255, 1)
        cv2.imwrite(dir_lid_output, edge_lid)
        print("done.")

    # patch
    # edge_cam = patch_image(image=edge_cam, mode=cv2.MORPH_CLOSE, size=8, iter=2)
    # edge_lid = patch_image(image=edge_lid, mode=cv2.MORPH_CLOSE, size=8, iter=2)

    # edge_cam = fill_hole(img=edge_cam, hole_color=0, bkg_color=255)
    # edge_lidar = fill_hole(img=edge_lidar, hole_color=0, bkg_color=255)

    # cv2.imwrite(data_path + "edges/canny_outputs/lidar_3_canny_patch.png", edge_lidar)
    # cv2.imwrite(data_path + "edges/canny_outputs/cam_3_canny_patch.png", edge_cam)

    # # contour filter
    # cnt_cam, hierarchy_cam = cv2.findContours(edge_cam, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cnt_lid, hierarchy_lid = cv2.findContours(edge_lid, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cnt_cam = contour_filter(contour=cnt_cam, len_threshold=200)
    # cnt_lid = contour_filter(contour=cnt_lid, len_threshold=200)
    # edge_cam = np.zeros(edge_cam.shape, np.uint8)
    # edge_lid = np.zeros(edge_lid.shape, np.uint8)
    # cv2.drawContours(edge_cam, cnt_cam, -1, 255, 1)
    # cv2.drawContours(edge_lid, cnt_lid, -1, 255, 1)

    # cv2.imwrite(dir_lid_output, edge_lid)
    # cv2.imwrite(dir_cam_output, edge_cam)

    if (len(sys.argv) >= kArgs):
        print("Edge extraction completed.")

    # ################################################################################################
    # # LiDAR # define the hyper-params of LiDAR
    # orien_lid = 3 
    # block_size_lid = 5  
    # var_proportion_lid = 0.10
    # num_bins_lid = 300
    # ################################################################################################
    # # first filter (big blocks)
    # img_lid = cv2.imread(data_path + "/edges/canny_outputs/lid_3_contour.png", cv2.IMREAD_GRAYSCALE)
    # img_filtered_lid, img_hog_lid = hog_filter(img_lid, orien_lid, block_size_lid, var_proportion_lid, num_bins_lid)
    # # img_filtered_lid_ex = extremum_filter(img_lid, orien_lid, block_size_lid)
    # # store the image files
    # # cv2.imwrite(data_path + "edges/hog_outputs/img_hog_lid.png", img_hog_lid)
    # cv2.imwrite(data_path + "/edges/hog_outputs/img_filtered_lid.png", img_filtered_lid)
    # # cv2.imwrite(data_path + "edges/lidEdge.png", img_filtered_lid_ex)
    # ################################################################################################
    # # counter filter
    # cnt_lidar, hierarchy_lidar = cv2.findContours(img_filtered_lid, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cnt_lidar = contour_filter(contour=cnt_lidar, len_threshold=200)
    # image_lidar = np.zeros(img_lid.shape, np.uint8)
    # cv2.drawContours(image_lidar, cnt_lidar, -1, 255, 1)
    # cv2.imwrite(data_path + "/edges/lidEdge.png", image_lidar)
    # ################################################################################################
    # # fisheye # define the hyper-params of the first hog filter
    # orien_cam = 5  
    # block_size_cam = 50  
    # var_proportion_cam = 0.20
    # num_bins_cam = 300
    # # define the params of second filter
    # # orien_cam_2 = 4
    # # block_size_cam_2 = 48
    # # var_proportion_cam_2 = 0.3
    #
    # ################################################################################################
    # # variance filter (big blocks)
    # img_cam = cv2.imread(data_path + "/edges/canny_outputs/cam_3_contour.png", cv2.IMREAD_GRAYSCALE) 
    # img_cam_filtered, img_hog_cam = hog_filter(img_cam, orien_cam, block_size_cam, var_proportion_cam, num_bins_cam)
    # # extremum filter
    # # img_cam_filtered_ex = extremum_filter(img_cam, orien_cam, block_size_cam)
    # # principal_component_filter (small blocks)
    # principal_component_filter(img_cam, orien_cam, block_size_cam)
    # # store the image files
    # # cv2.imwrite(data_path + "edges/hog_outputs/img_hog_cam.png", img_hog_cam)
    # cv2.imwrite(data_path + "/edges/hog_outputs/img_filtered_cam.png", img_cam_filtered)
    # # cv2.imwrite(data_path + "edges/camEdge.png", img_cam_filtered_ex)
    # ################################################################################################
    # # counter filter
    # cnt_cam, hierarchy_cam = cv2.findContours(img_cam_filtered, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cnt_cam = contour_filter(contour=cnt_cam, len_threshold=400)
    # edge_cam = np.zeros(img_cam.shape, np.uint8)
    # cv2.drawContours(edge_cam, cnt_cam, -1, 255, 1)
    # cv2.imwrite(data_path + "/edges/camEdge.png", edge_cam)
