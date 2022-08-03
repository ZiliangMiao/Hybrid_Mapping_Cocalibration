#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, signal, atexit
import rospy
import subprocess
from threading import Timer

lidar_fast_lio_cmd = "roslaunch fast_lio mapping_mid360.launch"
lidar_msg_cmd = "roslaunch livox_ros_driver livox_lidar_msg.launch"
lidar_sync_cmd = "roslaunch calibration lidar_sync.launch"
fisheye_auto_capture_cmd = "roslaunch mindvision mindvision.launch delay:=0"
record_cmd = "rosbag record -a -o /home/isee/catkin_ws/fast_lio_record.bag"

process_pids = []

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

atexit.register(Exiting)

if __name__ == "__main__":
    rospy.init_node('auto_run_fast_lio')
    CreateProcess(cmd=fisheye_auto_capture_cmd)
    CreateProcess(cmd=lidar_sync_cmd)
    time.sleep(5)
    CreateProcess(cmd=lidar_fast_lio_cmd)
    CreateProcess(cmd=lidar_msg_cmd)
    CreateProcess(cmd=record_cmd)
    