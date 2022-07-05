import cv2 as cv2
import numpy as np
import os, sys

if __name__ == "__main__":

    spot_range = 4
    default_name = "floor5"

    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))

    for spot in range(spot_range):
        skip = False
        if (len(sys.argv) > 1):
                data_path = root_path + "/data/" + sys.argv[1] + "/" + str(spot) + "/0"
        else:
            data_path = root_path + "/data/" + default_name + "/" + str(spot) + "/0"
        # Loading exposure images into a list
        exposure_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
            1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
            20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
            ], dtype=np.float32)
        img_list = []
        f = os.walk(data_path)
        for dirpath, dirnames, filenames in f:
            for filename in filenames:
                if not filename.__contains__("grab_0.bmp"):
                    img_list.append(cv2.imread(data_path + filename))
                else:
                    skip = True
        if skip:
            continue
        # Merge exposures to HDR image
        # merge_debevec = cv2.createMergeDebevec()
        # hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
        merge_robertson = cv2.createMergeRobertson()
        hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
        # Tonemap HDR image
        # tonemap1 = cv2.createTonemap(gamma=2.2)
        # res_debevec = tonemap1.process(hdr_debevec.copy())
        # Exposure fusion using Mertens
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        # Convert datatype to 8-bit and save
        # res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
        res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
        # cv2.imwrite(data_path + "grab_0.bmp", res_debevec_8bit)
        cv2.imwrite(data_path + "grab_0.bmp", res_mertens_8bit)