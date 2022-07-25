#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, signal, atexit
import rospy
from geometry_msgs.msg import Twist
import subprocess
from threading import Timer
from ExposureFusion import Capture

dataset_name = "ceres"
num_gimbal_step = 25
num_views = 5
num_spots = 5
terminal_output = True

script_path = os.path.join(os.path.abspath(__file__))
data_path = script_path.split("/catkin_ws/src")[0] + "/catkin_ws/data"
root_path = data_path + "/" + dataset_name

lidar_broadcast_cmd = "roslaunch livox_ros_driver livox_lidar_rviz.launch"
lidar_liomsg_cmd = "roslaunch livox_ros_driver livox_lidar_msg.launch"
lidar_sync_cmd = "python3 sync.py"
fisheye_fusion_cmd_prefix = "python3 ExposureFusion.py"
fisheye_auto_capture_cmd = "roslaunch mindvision mindvision.launch"

process_pids = []

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
    cmd = fisheye_fusion_cmd_prefix + " " + GetFolderPath(spot_idx, view_idx) + "/images"
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

def CreateProcess(cmd, t_process, t_output=1):
    proc = subprocess.Popen(cmd, shell=True,
    # proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("New subprocess (pid = %d) is created, terminate in %d seconds." %(proc.pid, t_process))
    print("Command: %s" %(cmd))
    process_pids.append(proc.pid)
    proc_timer = Timer(t_process, KillProcess, [proc.pid])
    proc_timer.start()
    if terminal_output:
        output_timer = Timer(t_output, Output, [proc])
        output_timer.start()

def Output(proc):
    outs, _ = proc.communicate()
    # print('== subprocess exited with rc =', proc.returncode)
    print(outs.decode('utf-8'))

def KillProcess(proc_pid):
    try:
        os.killpg(proc_pid, signal.SIGTERM)
        print("Subprocess %d terminated." %(proc_pid))
    except OSError as e:
        print("Subprocess %d is already terminated." %(proc_pid))
    process_pids.remove(proc_pid)

def Exiting():
    # Cleanup subprocess is important!
    print("Cleaning ... ")
    for pid in process_pids:
        KillProcess(proc_pid=pid)

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

atexit.register(Exiting)

if __name__ == "__main__":
    rospy.init_node('auto_run')
    CheckFolders()
    # reset gimbal to center position (maximum 20s)
    GimbalPublisher(view_idx='center', time_interval=20)
    for spot_idx in range(num_spots):
        # broadcast LiDAR pointclouds to ROS
        CreateProcess(cmd=lidar_broadcast_cmd, t_process=num_views*90)
        for view_idx in range(num_views):
            # rotate gimbal (maximum 20s)
            GimbalPublisher(view_idx=view_idx, time_interval=20)
            # record rosbag (default 60s + delay 10s)
            record_view_cmd = GetLidarStaticBagCmd(spot_idx, view_idx, duration=60)
            CreateProcess(cmd=record_view_cmd, t_process=60)
            # capture images (default 60s + delay 10s)
            fisheye_capture_path = GetFisheyeCapturePath(spot_idx, view_idx)
            Capture(fisheye_capture_path)
            fisheye_fusion_cmd = GetFisheyeCmd(spot_idx, view_idx)
            CreateProcess(cmd=fisheye_fusion_cmd, t_process=10)
            time.sleep(70)
        # broadcast LiDAR messages to ROS (delay 20s)
        CreateProcess(cmd=lidar_liomsg_cmd, t_process=60, t_output=10)
        CreateProcess(cmd=fisheye_auto_capture_cmd, t_process=60)
        # reset gimbal to center position (maximum 20s)
        GimbalPublisher(view_idx='center', time_interval=20)
        # record rosbag (default 30s + delay 10s)
        record_lio_cmd = GetLidarLioBagCmd(source_spot_idx=spot_idx, target_spot_idx=spot_idx+1, duration=30)
        CreateProcess(cmd=record_lio_cmd, t_process=30)
        time.sleep(40)
