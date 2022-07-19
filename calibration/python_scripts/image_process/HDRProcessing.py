import cv2 as cv2
import numpy as np
import os, sys

def ACESToneMapping(hdr_image, exposure):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    for i in range(3):
        channel = hdr_image[:, :, i] * exposure
        channel = np.clip(channel, 0, 32.767)
        channel2 = np.square(channel)
        hdr_image[:, :, i] = cv2.divide((channel * A + channel2 * B), (channel * C + channel2 * D + E))
    aces_8bit = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)
    return aces_8bit

def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,:,:] = cv2.LUT(np.array(imgs[i,:,:], dtype = np.uint8), table)
    return new_imgs

if __name__ == "__main__":

    num_spots = 5
    num_views = 5
    default_name = "lh3_global"
    response = []
    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))

    for spot_idx in range(num_spots):
        for view_idx in range(num_views):
            # view_angle = -int(sys.argv[3]) + int(sys.argv[3]) * view_idx
            view_angle = -50 + 25 * view_idx
            # data_path = sys.argv[2] + "/spot" + str(spot_idx) + "/" + view_angle + "/images/"
            data_path = root_path + "/data" + "/" + default_name + "/spot" + str(spot_idx) + "/" + str(view_angle) + "/images"

            # Loading exposure images into a list
            # exposure_times = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0,
            #     1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            #     20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
            #     ], dtype=np.float32)
            exposure_times = np.array([0.5, 1.0, 5, 10, 20, 50, 100], dtype=np.float32)
            img_list = []
            
            for t in exposure_times:
                if (t >= 5):
                    num = int(t)
                else:
                    num = t
                img_list.append(cv2.imread(data_path + "/grab_" + str(num) + ".bmp"))
            
            # Obtain Camera Response Function (CRF)
            # print("Calculating Camera Response Function (CRF) ... ")
            if (spot_idx == 0 and view_idx == 0):
                calibrate = cv2.createCalibrateDebevec()
                response = calibrate.process(img_list, exposure_times)

            # Merge exposures to HDR image
            print("Merging images into one HDR image ... ")
            merge_mertens = cv2.createMergeMertens()
            res_mertens = merge_mertens.process(img_list, exposure_times, response)
            res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
            # cv2.imwrite(data_path + "/grab_0_mertens.bmp", res_mertens_8bit)
            clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
            res_gamma = res_mertens_8bit
            weight = 0.25
            for i in range(3):
                res_gamma[:, :, i] = clahe.apply(res_mertens_8bit[:, :, i]) * weight + res_mertens_8bit[:, :, i] * (1 - weight)
            cv2.imwrite(data_path + "/grab_0.bmp", res_gamma)

            
            # mergeDebevec = cv2.createMergeDebevec()
            # hdrDebevec = mergeDebevec.process(img_list, exposure_times, response)
            
            ############# Tone map ###############
            # # Tonemap using Reinhard's method to obtain 24-bit color image
            # print("Tonemaping using Reinhard's method ... ")
            # tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 1)
            # ldrReinhard = tonemapReinhard.process(hdrDebevec)
            # ldrReinhard_8bit = np.clip(ldrReinhard * 255, 0, 255).astype('uint8')
            # # cv2.imwrite(data_path + "/grab_0.bmp", ldrReinhard_8bit)
            # # print("saved ldr-Reinhard.bmp")
            # res_gamma = ldrReinhard_8bit
            # for i in range(3):
            #     res_gamma[:, :, i] = clahe.apply(res_gamma[:, :, i])
            # cv2.imwrite(data_path + "/grab_0_reinhard.bmp", res_gamma)


            print("Spot Index: " + str(spot_idx) + " View Index: " + str(view_idx) + " HDR Finished!")
