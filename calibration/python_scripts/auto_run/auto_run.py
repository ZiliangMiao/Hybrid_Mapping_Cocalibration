#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, signal, atexit
import rospy
from geometry_msgs.msg import Twist
# import mvsdk
import subprocess
import numpy as np
from threading import Timer

dataset_name = "parking"
num_gimbal_step = 25
num_views = 5
num_spots = 6

script_path = os.path.join(os.path.abspath(__file__))
data_path = script_path.split("/catkin_ws/src")[0] + "/catkin_ws/data"
root_path = data_path + "/" + dataset_name

lidar_broadcast_cmd = "roslaunch livox_ros_driver livox_lidar_rviz.launch"
lidar_msg_cmd = "roslaunch livox_ros_driver livox_lidar_msg.launch"
lidar_sync_cmd = "roslaunch calibration lidar_sync.launch"
fisheye_auto_capture_cmd = "roslaunch mindvision mindvision.launch"

process_pids = []
hCamera = 0

def _checkFolder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def CheckFolders():
    view_folders = ["bags", "all_pcds", "dense_pcds", "icp_pcds",
                    "images", "edges", "outputs", "results"]
    _checkFolder(data_path)
    _checkFolder(root_path)
    for spot_idx in range(num_spots):
        spot_path = GetFolderPath(spot_idx)
        _checkFolder(spot_path)
        for view_idx in range(num_views):
            view_path = GetFolderPath(spot_idx, view_idx)
            _checkFolder(view_path)
            for folder in view_folders:
                _checkFolder(view_path + "/" + folder)
        _checkFolder(spot_path + "/fullview_recon")

def GetFolderPath(spot_idx, view_idx=None, return_angle=False):
    path = root_path + "/spot" + str(spot_idx)
    if view_idx is not None:
        angle = int((-(num_views - 1) / 2 + view_idx) * num_gimbal_step)
        path = path + "/" + str(angle)
        if return_angle:
            return path, angle
    return path

def GetFisheyeCmd(spot_idx, view_idx):
    cmd = "python3" + " " + os.path.abspath(os.path.join(os.path.abspath(__file__), "../exposure_fusion.py")) \
        + " " + GetFolderPath(spot_idx, view_idx) + "/images"
    return cmd

def GetFisheyeCapturePath(spot_idx, view_idx):
    return GetFolderPath(spot_idx, view_idx) + "/images"

def GetLidarStaticBagCmd(spot_idx, view_idx, duration=60):
    folder_path, angle = GetFolderPath(spot_idx, view_idx, return_angle=True)
    name = dataset_name + "_spot" + str(spot_idx) + "_" + str(angle) + ".bag"
    filepath = folder_path + "/bags/" + name
    cmd = "rosbag record -a -o" + " " + filepath + " " +  "--duration=" + str(duration)
    return cmd

def GetLidarLioBagCmd(source_spot_idx, target_spot_idx, duration=30):
    folder_path = GetFolderPath(source_spot_idx)
    name = dataset_name + "_spot" + str(source_spot_idx) + "_spot" + str(target_spot_idx) + ".bag"
    filepath = folder_path + "/" + name
    cmd = "rosbag record -a -o" + " " + filepath + " " + "--duration=" + str(duration)
    return cmd

# def Capture(image_output_path):

# 	exposure_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
#                     1.5, 2, 2.5, 3, 3.5, 4, 4.5,
#                     5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
#                     20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
#                     120, 140, 160, 180, 200]

# 	pre_sample_size = 10
# 	target_median_exp = 20 * 1e3
# 	isp_auto_exp = 20 * 1e3
# 	real_exposure_time = []

# 	DevList = mvsdk.CameraEnumerateDevice()
# 	nDev = len(DevList)
# 	if nDev < 1:
# 		print("No camera was found!")
# 		return
		
# 	for i, DevInfo in enumerate(DevList):
# 		print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
# 	i = 0 if nDev == 1 else int(input("Select camera: "))
# 	DevInfo = DevList[i]

# 	# 打开相机
# 	global hCamera
# 	try:
# 		hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
# 	except mvsdk.CameraException as e:
# 		print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
# 		return

# 		# 获取相机特性描述
# 	cap = mvsdk.CameraGetCapability(hCamera)

# 	# 判断是黑白相机还是彩色相机
# 	monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

# 	# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
# 	if monoCamera:
# 		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

# 	# 相机模式切换成连续采集
# 	mvsdk.CameraSetTriggerMode(hCamera, 0)
# 	mvsdk.CameraSetAeState(hCamera, 1)
# 	mvsdk.CameraSetAeTarget(hCamera, 10)

# 	# 让SDK内部取图线程开始工作
# 	mvsdk.CameraPlay(hCamera)

# 	for t in range(pre_sample_size):
# 		FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
# 		pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

# 		try:
# 			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
# 			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
# 			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
# 			isp_auto_exp = mvsdk.CameraGetExposureTime(hCamera)

# 		except mvsdk.CameraException as e:
# 			print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

# 	mvsdk.CameraSetAeState(hCamera, 0)
# 	print("Auto exposure time = " + str(isp_auto_exp))
# 	scale = isp_auto_exp / target_median_exp

# 	for t in exposure_times:

# 		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
# 		exp_time = int(t * scale) if ( t - int(t * scale) == 0) else t * scale
# 		FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

# 		# 分配RGB buffer，用来存放ISP输出的图像
# 		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
# 		pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

# 		# 曝光时间
# 		mvsdk.CameraSetExposureTime(hCamera, exp_time * 1000)

# 		# 从相机取一帧图片
# 		try:
# 			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
# 			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
# 			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
			
# 			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
# 			# 该示例中我们只是把图片保存到硬盘文件中
# 			status = mvsdk.CameraSaveImage(hCamera, image_output_path + "/grab_" + str(t) + ".bmp", pFrameBuffer, FrameHead, mvsdk.FILE_BMP, 100)
# 			if status == mvsdk.CAMERA_STATUS_SUCCESS:
# 				# print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth, FrameHead.iHeight) )\
# 				real_exposure_time.append(mvsdk.CameraGetExposureTime(hCamera) * 1e-3)
# 				pass
# 			else:
# 				print("Save image failed. err={}".format(status) )
# 		except mvsdk.CameraException as e:
# 			print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

# 	# 关闭相机
# 	mvsdk.CameraUnInit(hCamera)

# 	exposure_output = np.vstack((np.array(exposure_times), np.array(real_exposure_time))).T
# 	np.savetxt(image_output_path + '/exposure.txt', exposure_output, delimiter='\t')

# 	print("Capture terminated.")

# 	# 释放帧缓存
# 	mvsdk.CameraAlignFree(pFrameBuffer)

def GimbalPublisher(view_idx=6, time_interval=15, send_interval=0.1):
    if view_idx == 'center':
        view_idx = (num_views - 1) / 2
    center_mode = 6
    control_mode = int(center_mode + (num_views - 1) / 2 - view_idx)
    control_msg = [
    "vertical -50 degrees",
    "vertical -25 degrees",
    "vertical 0 degree",
    "vertical 25 degrees",
    "vertical 50 degrees",
    ]
    pub = rospy.Publisher('/cmd_instruction', Twist, queue_size=5)
    rospy.loginfo('Gimbal: Rotate to %s , mode = %d', control_msg[view_idx], control_mode)
    twist = Twist()
    twist.linear.x = control_mode
    for i in range(int(time_interval / send_interval)):
        # Timer(send_interval, pub.publish, args=(twist,)).start()
        pub.publish(twist)
        time.sleep(send_interval)

def CreateProcess(cmd, t_process=0, t_output=1):
    proc = subprocess.Popen(cmd, shell=True,
    # proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("New subprocess (pid = %d) is created, terminate in %d seconds." %(proc.pid, t_process))
    print("Command: %s" %(cmd))
    process_pids.append(proc.pid)
    if (t_process > 0):
        proc_timer = Timer(t_process, KillProcess, [proc.pid])
        proc_timer.start()
    if (t_output > 0):
        output_timer = Timer(t_output, Output, [proc])
        output_timer.start()

def Output(proc):
    outs, _ = proc.communicate()
    # print('== subprocess exited with rc =', proc.returncode)
    print(outs.decode('utf-8'))

def KillProcess(proc_pid):
    try:
        os.killpg(proc_pid, signal.SIGTERM)
        os.killpg(proc_pid, signal.SIGKILL)
        print("Subprocess %d terminated." %(proc_pid))
    except OSError as e:
        print("Subprocess %d is already terminated." %(proc_pid))
    process_pids.remove(proc_pid)

def Exiting():
	# Cleanup subprocess is important!
	print("Cleaning ... ")
	for pid in process_pids:
		KillProcess(proc_pid=pid)
	# try:
	# 	mvsdk.CameraUnInit(hCamera)
	# except:
	# 	pass

atexit.register(Exiting)

if __name__ == "__main__":
    enable_real_time_fast_lio = True
    rospy.init_node('auto_run')
    CheckFolders()
    CreateProcess(cmd=lidar_sync_cmd, t_process=num_spots * num_views * 150)
    # reset gimbal to center position (maximum 20s)
    GimbalPublisher(view_idx='center', time_interval=20)
    for spot_idx in range(num_spots):
        # broadcast LiDAR pointclouds to ROS
        CreateProcess(cmd=lidar_broadcast_cmd, t_process=num_views*90)
        for view_idx in range(num_views):
            # rotate gimbal (maximum 20s)
            if (view_idx == 0 or view_idx == num_views - 1):
                GimbalPublisher(view_idx=view_idx, time_interval=20)
            else:
                GimbalPublisher(view_idx=view_idx, time_interval=10)
            # record rosbag (default 60s + delay 10s)
            CreateProcess(cmd=GetLidarStaticBagCmd(spot_idx, view_idx, duration=60),
                                            t_process=60)
            # capture images (default 60s + delay 10s)
            # Capture(GetFisheyeCapturePath(spot_idx, view_idx))
            CreateProcess(cmd=GetFisheyeCmd(spot_idx, view_idx),
                                            t_process=10)
            time.sleep(65)
        if (spot_idx < num_spots - 1):
            # broadcast LiDAR messages to ROS (delay 20s)
            CreateProcess(cmd=fisheye_auto_capture_cmd, t_process=60)
            CreateProcess(cmd=lidar_msg_cmd, t_process=60)
        # reset gimbal to center position (maximum 20s)
        GimbalPublisher(view_idx='center', time_interval=20)
        if (spot_idx < num_spots - 1):
            # record rosbag (default 30s + delay 10s)
            CreateProcess(cmd=GetLidarLioBagCmd(source_spot_idx=spot_idx, target_spot_idx=spot_idx+1, duration=30),
                                                t_process=30)
            time.sleep(40)
