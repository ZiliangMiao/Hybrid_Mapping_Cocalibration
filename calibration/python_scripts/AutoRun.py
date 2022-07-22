import os, sys, time
import signal, warnings
import psutil
import subprocess
from threading import Timer

dataset_name = "ceres"
num_gimbal_step = 25
num_views = 5
num_spots = 5

script_path = os.path.join(os.path.abspath(__file__))
data_path = script_path.split("/catkin_ws/src")[0] + "/catkin_ws/data"
root_path = data_path + "/" + dataset_name

gimbal_cmd = "roslaunch gimbal gimbal.launch"
lidar_broadcast_cmd = "roslaunch livox_ros_driver livox_lidar.launch"
lidar_liomsg_cmd = "roslaunch livox_ros_driver livox_lidar_msg.launch"
fisheye_capture_cmd = "python3 grab.py"
fisheye_hdr_cmd = "python3 grab_hdr.py"
record_prefix = "rosbag record -a -o "
record_suffix = " --duration=60s"

def check_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check_data_folders():
    
    view_folders = ["bags", "all_pcds", "dense_pcds", "icp_pcds",
                    "images", "edges", "outputs", "results"]
    check_folder(data_path)
    check_folder(root_path)
    for spot_idx in range(num_spots):
        spot_path = get_folder_path(spot_idx)
        check_folder(spot_path)
        for view_idx in range(num_views):
            view_path = get_folder_path(spot_idx, view_idx)
            check_folder(view_path)
            for folder in view_folders:
                check_folder(view_path + "/" + folder)
        check_folder(spot_path + "/fullview_recon")

def get_folder_path(spot_idx, view_idx=None, return_angle=False):
    
    path = root_path + "/spot" + str(spot_idx)
    if view_idx is not None:
        angle = (-(num_views - 1) / 2 + view_idx) * num_gimbal_step
        path = path + "/" + str(int(angle))
        if return_angle:
            return path, angle
    return path
        

def get_view_bag_cmd(spot_idx, view_idx):
    path, angle = get_folder_path(spot_idx, view_idx, return_angle=True)
    name = dataset_name + "_spot" + str(spot_idx) + "_" + str(int(angle)) + ".bag"
    res = path + "/bags/" + name
    cmd = record_prefix + res + record_suffix
    return cmd

def get_lio_bag_cmd(source_spot_idx, target_spot_idx):
    path = get_folder_path(source_spot_idx)
    name = dataset_name + "_spot" + str(source_spot_idx) + "_spot" + str(target_spot_idx) + ".bag"
    res = path + "/" + name
    cmd = record_prefix + res + record_suffix
    return cmd

def timer_process(cmd, t_interval):
    proc = subprocess.Popen(cmd, start_new_session=True, shell=True, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("new subprocess (pid = %d) is created, terminate in %d seconds." %(proc.pid, t_interval))
    print(cmd)
    proc_timer = Timer(t_interval, kill, [proc.pid])
    proc_timer.start()
    output_timer = Timer(t_interval/10, output, [proc])
    output_timer.start()

def output(proc):
    outs, _ = proc.communicate()
    # print('== subprocess exited with rc =', proc.returncode)
    print(outs.decode('utf-8'))


def kill(proc_pid):
    try:
        os.killpg(proc_pid, signal.SIGTERM)
        print("Process %d terminated." %(proc_pid))
    except OSError as e:
        print("Process %d is already terminated." %(proc_pid))

if __name__ == "__main__":
    check_data_folders()
    
    # reset gimbal, 15s (not implemented)
    # 
    for view_idx in range(5):
        # gimbal.launch, 15s
        timer_process(cmd=gimbal_cmd, t_interval=15)
        # livox_lidar.launch, 90s
        timer_process(cmd=lidar_broadcast_cmd, t_interval=90)
        time.sleep(15)
        # rosbag record, 60s
        record_view_cmd = get_view_bag_cmd(spot_idx=0, view_idx=view_idx)
        timer_process(cmd=record_view_cmd, t_interval=60)
        # grab.py, 40s
        timer_process(cmd=fisheye_capture_cmd, t_interval=40)
        time.sleep(40)
        # HDR, 30s
        timer_process(cmd=fisheye_hdr_cmd, t_interval=30)
        time.sleep(30)
    # reset gimbal, 15s (not implemented)
    # 
    # livox_lidar_msg.launch, 90s
    timer_process(cmd=lidar_liomsg_cmd, t_interval=90)
    time.sleep(15)
    # rosbag record, 60s
    record_lio_cmd = get_lio_bag_cmd(source_spot_idx=0, target_spot_idx=1)
    timer_process(cmd=record_lio_cmd, t_interval=60)
    time.sleep(15)
