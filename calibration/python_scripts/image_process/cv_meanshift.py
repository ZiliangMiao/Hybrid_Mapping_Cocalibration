import numpy as np
from cv2 import cv2
from skimage.feature import _hog, _hoghistogram

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


def patch_img(img, mode=cv2.MORPH_OPEN, size=3, iter=1):
    kernel = np.ones((size, size), dtype=np.uint8)
    img = cv2.morphologyEx(img, mode, kernel, iter)
    return img


def fill_hole(img, hole_color=0, bkg_color=255, size_threshold=200, size_lim=False):

    # 复制 im_in 图像
    img_floodfill = img.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    seedPoint = (1, 1)
    for i in range(img_floodfill.shape[0]):
        for j in range(img_floodfill.shape[1]):
            if (img_floodfill[i][j] == hole_color):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(img_floodfill, mask, seedPoint, bkg_color)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(img_floodfill)

    if size_lim:
        im_floodfill_inv_copy = im_floodfill_inv.copy()

        # 函数findContours获取轮廓
        _, contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= size_threshold and hierarchy[0, num, 2] == -1):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], hole_color)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    return img | im_floodfill_inv


# def highlight_removal(img):
#     _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
#     rep = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
#     mask = cv2.resize(mask, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
#     # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     # dst = cv2.inpaint(rep, mask, 10, cv2.INPAINT_TELEA)
#     dst = cv2.illuminationChange(rep, mask, alpha=0.2, beta=1.5)
#     return dst


def _hog_channel_gradient(channel, sobel=True, remove=True):
    g_row = np.empty(channel.shape, dtype=np.double)
    g_col = np.empty(channel.shape, dtype=np.double)
    if sobel:
        from cv2 import cv2
        g_row = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
        g_col = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
        g_max = np.max([np.max(g_row), np.max(g_col)])
        g_col = g_col / g_max * 256
        g_row = g_row / g_max * 256
    else:
        g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
        g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    if remove:
        bool = ((-np.sign(channel-127) + 1) * 0.5).astype(np.uint8)
        g_row = bool * g_row
        g_col = bool * g_col

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

    if visualize:
        from skimage import draw

        hog_image = None
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
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

        return normalized_blocks, 10 * hog_image
    else:
        return normalized_blocks

dir_cam = "./images/flatImage.bmp"
dir_lidar = "./images/flatLidarImageRNN.bmp"
img_cam = cv2.imread(dir_cam)
img_lidar = cv2.imread(dir_lidar)

# img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY)
# img_cam = cv2.bilateralFilter(img_cam, d=8, sigmaColor=76, sigmaSpace=8)
# cv2.imwrite(dir_root + "cam_1f_lowfil.png", img_cam)

img_cam_meanshift = np.zeros(img_cam.shape, np.uint8)
cv2.pyrMeanShiftFiltering(src=img_cam, dst=img_cam_meanshift, sp=40, sr=40, maxLevel=2)
img_cam_meanshift = cv2.cvtColor(img_cam_meanshift, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./canny_outputs/cam_1f_meanshift.png", img_cam_meanshift)
img_cam = img_cam_meanshift

img_lidar = cv2.cvtColor(img_lidar, cv2.COLOR_BGR2GRAY)

img_cam = cv2.GaussianBlur(img_cam, sigmaX=3, sigmaY=3, ksize=(5, 5))
img_lidar = cv2.fastNlMeansDenoising(img_lidar, h=40, searchWindowSize=21, templateWindowSize=7)
# cv2.imwrite(dir_root + "cam_1f_filtered.png", img_cam)
# cv2.imwrite(dir_root + "lidar_1f_filtered.png", img_lidar)

edge_cam = cv2.Canny(image=img_cam, threshold1=20, threshold2=100)
edge_lidar = cv2.Canny(image=img_lidar, threshold1=20, threshold2=100)

cv2.imwrite("./canny_outputs/lidar_2_canny_original.png", edge_lidar)
cv2.imwrite("./canny_outputs/cam_2_canny_original.png", edge_cam)

# edge_cam = patch_image(image=edge_cam, mode=cv2.MORPH_CLOSE, size=7, iter=2)
# edge_lidar = patch_image(image=edge_lidar, mode=cv2.MORPH_CLOSE, size=3, iter=2)

# 这个是填充区域的
# edge_cam = fill_hole(img=edge_cam, hole_color=0, bkg_color=255)
# edge_lidar = fill_hole(img=edge_lidar, hole_color=0, bkg_color=255)

# cv2.imwrite("./canny_outputs/lidar_3_canny_patch.png", edge_lidar)
# cv2.imwrite("./canny_outputs/cam_3_canny_patch.png", edge_cam)

# cnt_cam, hierarchy_cam = cv2.findContours(edge_cam, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# cnt_lidar, hierarchy_lidar = cv2.findContours(edge_lidar, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
# cnt_cam = contour_filter(contour=cnt_cam, len_threshold=300)
# cnt_lidar = contour_filter(contour=cnt_lidar, len_threshold=50)
#
# image_cam = np.zeros(image_cam.shape, np.uint8)
# image_lidar = np.zeros(image_lidar.shape, np.uint8)
#
# cv2.drawContours(image_cam, cnt_cam, -1, 255, 1)
# cv2.drawContours(image_lidar, cnt_lidar, -1, 255, 1)
#
# cv2.imwrite("./canny_outputs/lidar_4_contour.png", image_lidar)
# cv2.imwrite("./canny_outputs/cam_4_contour.png", image_cam)

# image_lidar = cv2.imread(dir_root + "lidar_test_2.png", cv2.IMREAD_GRAYSCALE)
# image_lidar = np.vstack((np.zeros((300, image_lidar.shape[1])).astype(np.uint8), image_lidar))
# kp_lidar = cv2.xfeatures2d.SIFT_create().detect(image_lidar)
# img_lidar_sift = cv2.drawKeypoints(image_lidar, kp_lidar, image_lidar, color=[0,255,255])
# cv2.imwrite(dir_root + "lidar_5_sift.png", img_lidar_sift)
#
# sift_cam = cv2.xfeatures2d.SIFT_create()
# kp_cam = sift_cam.detect(image_cam, None)
# img_cam_sift = cv2.drawKeypoints(image_cam, kp_cam, image_cam, color=[0,0,255])
# cv2.imwrite(dir_root + "cam_5_edge_sift.png", img_cam_sift)