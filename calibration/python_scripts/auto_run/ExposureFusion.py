#coding=utf-8
import os, sys
from time import time
import numpy as np
import cv2 as cv2

def ExposureFusion(image_output_path):
	exposure_times = np.array([0.5, 1, 5, 10, 20, 50, 100], dtype=np.float32)
	try:
		real_exp_time = np.loadtxt(image_output_path + "/exposure.txt", delimiter="\t")
	except:
		print("Exposure time record file not found, set to default values.")
		real_exp_time = np.vstack((exposure_times, exposure_times)).T
	img_list = []
	time_list = []
	
	for exp_time in exposure_times:
		t = exp_time if (exp_time - int(exp_time) != 0) else int(exp_time)
		img_list.append(cv2.imread(image_output_path + "/grab_" + str(t) + ".bmp"))
		real_t = real_exp_time[:, 1].flat[np.abs(real_exp_time[:, 0] - t).argmin()]
		time_list.append(real_t)

	time_list = np.asarray(time_list).astype(np.float32)
	print(time_list)
	# Obtain Camera Response Function (CRF)
	print("Calculating Camera Response Function (CRF) ... ")
	calibrate = cv2.createCalibrateDebevec()
	response = calibrate.process(img_list, time_list)

	# Merge exposures to HDR image
	print("Merging images into one HDR image ... ")
	merge_mertens = cv2.createMergeMertens()
	res_mertens = merge_mertens.process(img_list, time_list, response)
	res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
	clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
	res_clahe = res_mertens_8bit
	weight = 0.25
	for i in range(3):
		res_clahe[:, :, i] = clahe.apply(res_mertens_8bit[:, :, i]) * weight + res_mertens_8bit[:, :, i] * (1 - weight)
	cv2.imwrite(image_output_path + "/grab_0.bmp", res_clahe)

if __name__ == "__main__":
	image_output_path = sys.argv[1]
	ExposureFusion(image_output_path)