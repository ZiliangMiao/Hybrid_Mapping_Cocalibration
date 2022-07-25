#coding=utf-8
import mvsdk
import os, sys
from datetime import datetime
import numpy as np
import cv2 as cv2

# TriggerMode: 0: 连续; 1: 软件触发; 2: 硬件触发
# ROI: 0: 2448X2048 Max; 1: 1920X1080 ROI; 2: 1600X1200 ROI; 3: 1280X1024 ROI; 4: 640X480 ROI
# (?): 0:D65; 1: 5500K(纯白光源); 2: 阴天、室内
# (?): 0: Bayer BG 8bit (1Bpp); 1: Bayer BG 12bit Packed (1.5Bpp); 
# (?): 0: Low; 1: Mid; 2: High
# (?): 0: Knee 1; 1: Knee 2; 2: Knee 3; 3: Line

def Capture(image_output_path):

	exposure_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
				 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
				 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
				 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
				 120, 140, 160, 180, 200]

	DevList = mvsdk.CameraEnumerateDevice()
	nDev = len(DevList)
	if nDev < 1:
		print("No camera was found!")
		return
		
	for i, DevInfo in enumerate(DevList):
		print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
	i = 0 if nDev == 1 else int(input("Select camera: "))
	DevInfo = DevList[i]
	# print(DevInfo)

	# 打开相机
	hCamera = 0
	try:
		hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
	except mvsdk.CameraException as e:
		print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
		return

		# 获取相机特性描述
	cap = mvsdk.CameraGetCapability(hCamera)
	# PrintCapbility(cap)

	# 判断是黑白相机还是彩色相机
	monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

	# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
	if monoCamera:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

	# 相机模式切换成连续采集
	mvsdk.CameraSetTriggerMode(hCamera, 0)
	mvsdk.CameraSetAeState(hCamera, 0)

	# 让SDK内部取图线程开始工作
	mvsdk.CameraPlay(hCamera)
	print("Saving images ...")

	for t in exposure_times:
		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
		FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

		# 分配RGB buffer，用来存放ISP输出的图像
		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
		pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

		# 手动曝光，曝光时间30ms
		mvsdk.CameraSetExposureTime(hCamera, t * 1000)

		# 从相机取一帧图片
		try:
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
			
			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 该示例中我们只是把图片保存到硬盘文件中
			status = mvsdk.CameraSaveImage(hCamera, image_output_path + "/grab_" + str(t) + ".bmp", pFrameBuffer, FrameHead, mvsdk.FILE_BMP, 100)
			if status == mvsdk.CAMERA_STATUS_SUCCESS:
				# print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth, FrameHead.iHeight) )
				pass
			else:
				print("Save image failed. err={}".format(status) )
		except mvsdk.CameraException as e:
			print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

	print("done.")

	# 关闭相机
	mvsdk.CameraUnInit(hCamera)

	# 释放帧缓存
	mvsdk.CameraAlignFree(pFrameBuffer)

def PrintCapbility(cap):
	for i in range(cap.iTriggerDesc):
		desc = cap.pTriggerDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iImageSizeDesc):
		desc = cap.pImageSizeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iClrTempDesc):
		desc = cap.pClrTempDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iMediaTypeDesc):
		desc = cap.pMediaTypeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iFrameSpeedDesc):
		desc = cap.pFrameSpeedDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPackLenDesc):
		desc = cap.pPackLenDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPresetLut):
		desc = cap.pPresetLutDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmSwDesc):
		desc = cap.pAeAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmHdDesc):
		desc = cap.pAeAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmSwDesc):
		desc = cap.pBayerDecAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmHdDesc):
		desc = cap.pBayerDecAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )

def ExposureFusion(image_output_path):
	exposure_times = np.array([0.5, 1, 5, 10, 20, 50, 100], dtype=np.float32)
	img_list = []
	
	for t in exposure_times:
		if (t - int(t) == 0):
			num = int(t)
		else:
			num = t
		img_list.append(cv2.imread(image_output_path + "/grab_" + str(num) + ".bmp"))
	
	# Obtain Camera Response Function (CRF)
	print("Calculating Camera Response Function (CRF) ... ")
	calibrate = cv2.createCalibrateDebevec()
	response = calibrate.process(img_list, exposure_times)

	# Merge exposures to HDR image
	print("Merging images into one HDR image ... ")
	merge_mertens = cv2.createMergeMertens()
	res_mertens = merge_mertens.process(img_list, exposure_times, response)
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