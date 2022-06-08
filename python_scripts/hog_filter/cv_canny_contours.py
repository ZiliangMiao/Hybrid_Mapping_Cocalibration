import cv2
import numpy as np

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
        contours, hierarchy = cv2.findContours(im_floodfill_inv_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for num in range(len(contours)):
            if (cv2.contourArea(contours[num]) >= size_threshold and hierarchy[0, num, 2] == -1):
                cv2.fillConvexPoly(im_floodfill_inv, contours[num], hole_color)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    return img | im_floodfill_inv

def black_region_removal(img, pix_rows_bound):
    output = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    output[pix_rows_bound:, :] = img[pix_rows_bound:, :]
    return output

# def highlight_removal(img):
#     _, mask = cv2.threshold(img, int(0.8*img.max()+0.2*img.min()), 255, cv2.THRESH_BINARY)
#     rep = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
#     mask = cv2.resize(mask, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
#     dst = cv2.illuminationChange(rep, mask, alpha=1, beta=1)
#     return dst

data_path = "/home/godm/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/"
dir_cam = data_path + "outputs/flatImage.bmp"
dir_lidar = data_path + "outputs/byIntensity/flatLidarImage.bmp"
image_cam = cv2.imread(dir_cam)
image_lidar = cv2.imread(dir_lidar)

image_cam = cv2.cvtColor(image_cam, cv2.COLOR_BGR2GRAY)
image_lidar = cv2.cvtColor(image_lidar, cv2.COLOR_BGR2GRAY)
image_cam = cv2.GaussianBlur(image_cam, sigmaX=1, sigmaY=1, ksize=(5, 5))
image_lidar = cv2.fastNlMeansDenoising(image_lidar, h=40, searchWindowSize=21, templateWindowSize=7)
cv2.imwrite(data_path + "edges/cannyOutputs/cam_1_filtered.png", image_cam)
cv2.imwrite(data_path + "edges/cannyOutputs/lid_1_filtered.png", image_lidar)

edge_cam = cv2.Canny(image=image_cam, threshold1=20, threshold2=100)
edge_lidar = cv2.Canny(image=image_lidar, threshold1=20, threshold2=100)

# remove the black region
pix_rows_bound = 435
edge_cam = black_region_removal(edge_cam, pix_rows_bound)
cv2.imwrite(data_path + "edges/cannyOutputs/lid_2_canny_original.png", edge_lidar)
cv2.imwrite(data_path + "edges/cannyOutputs/cam_2_canny_original.png", edge_cam)

# edge_cam = patch_image(image=edge_cam, mode=cv2.MORPH_CLOSE, size=7, iter=2)
# edge_lidar = patch_image(image=edge_lidar, mode=cv2.MORPH_CLOSE, size=3, iter=2)

# 这个是填充区域的
# edge_cam = fill_hole(img=edge_cam, hole_color=0, bkg_color=255)
# edge_lidar = fill_hole(img=edge_lidar, hole_color=0, bkg_color=255)

# cv2.imwrite(data_path + "edges/cannyOutputs/lidar_3_canny_patch.png", edge_lidar)
# cv2.imwrite(data_path + "edges/cannyOutputs/cam_3_canny_patch.png", edge_cam)

_, cnt_cam, hierarchy_cam = cv2.findContours(edge_cam, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
_, cnt_lidar, hierarchy_lidar = cv2.findContours(edge_lidar, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cnt_cam = contour_filter(contour=cnt_cam, len_threshold=150)
cnt_lidar = contour_filter(contour=cnt_lidar, len_threshold=90)

image_cam = np.zeros(image_cam.shape, np.uint8)
image_lidar = np.zeros(image_lidar.shape, np.uint8)

cv2.drawContours(image_cam, cnt_cam, -1, 255, 1)
cv2.drawContours(image_lidar, cnt_lidar, -1, 255, 1)

cv2.imwrite(data_path + "edges/cannyOutputs/lid_4_contour.png", image_lidar)
cv2.imwrite(data_path + "edges/cannyOutputs/cam_4_contour.png", image_cam)

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