#coding=utf-8
import os, sys
from time import time
import numpy as np
import cv2 as cv2

def ExposureFusion(image_output_path):
	exposure_times = np.array([0.5, 1, 5, 10, 20, 50, 100], dtype=np.float32)
	real_exp_time = np.loadtxt(image_output_path + "/exposure.txt", delimiter="\t")
	try:
		real_exp_time = np.loadtxt(image_output_path + "/exposure.txt", delimiter="\t")
	except:
		print("Exposure time record missing, set to default values")
	img_list = []
	time_list = []
	
	for t in exposure_times:
		if (t - int(t) == 0):
			num = int(t)
		else:
			num = t
		img_list.append(cv2.imread(image_output_path + "/grab_" + str(num) + ".bmp"))
		real_t = real_exp_time[:, 1].flat[np.abs(real_exp_time[:, 0] - num).argmin()]
		time_list.append(real_t)

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