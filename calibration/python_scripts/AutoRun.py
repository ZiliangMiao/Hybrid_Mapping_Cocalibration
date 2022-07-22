import os, sys, time, signal, atexit
import select, termios, tty
import subprocess
from threading import Timer

dataset_name = "ceres"
num_gimbal_step = 25
num_views = 5
num_spots = 5
terminal_output = False

script_path = os.path.join(os.path.abspath(__file__))
data_path = script_path.split("/catkin_ws/src")[0] + "/catkin_ws/data"
root_path = data_path + "/" + dataset_name

gimbal_transmit_cmd = "roslaunch gimbal gimbal.launch"
gimbal_publish_cmd_prefix = "python3 gimbal_cmd_simple.py "
lidar_transmit_cmd = "roslaunch livox_ros_driver livox_lidar_rviz.launch"
lidar_liomsg_cmd = "roslaunch livox_ros_driver livox_lidar_msg.launch"
fisheye_cmd_prefix = "python3 FisheyeCapture.py"
record_cmd_prefix = "rosbag record -a -o "
record_cmd_suffix = " --duration=60s"

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
    cmd = fisheye_cmd_prefix + " " + GetFolderPath(spot_idx, view_idx) + "/images"
    return cmd

def GetGimbalPublisherCmd(view_idx):
    # mapping view_idx in [0,1,2,3,4] (from -(2*num_gimbal_step) to (2*num_gimbal_step)) 
    # to control mode [8,7,6,5,4] (center is 6)
    if view_idx == 'center':
        view_idx = (num_views - 1) / 2
    center_mode = 6
    view_mode = center_mode + (num_views - 1) / 2 - view_idx
    cmd = gimbal_publish_cmd_prefix + str(view_mode)
    return cmd

def GetViewBagCmd(spot_idx, view_idx):
    path, angle = GetFolderPath(spot_idx, view_idx, return_angle=True)
    name = dataset_name + "_spot" + str(spot_idx) + "_" + str(angle) + ".bag"
    res = path + "/bags/" + name
    cmd = record_cmd_prefix + res + record_cmd_suffix
    return cmd

def GetLioBagCmd(source_spot_idx, target_spot_idx):
    path = GetFolderPath(source_spot_idx)
    name = dataset_name + "_spot" + str(source_spot_idx) + "_spot" + str(target_spot_idx) + ".bag"
    res = path + "/" + name
    cmd = record_cmd_prefix + res + record_cmd_suffix
    return cmd

def GetKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def CreateProcess(cmd, t_process=0, t_output=3):
    proc = subprocess.Popen(cmd, start_new_session=True, shell=True, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("New subprocess (pid = %d) is created, terminate in %d seconds." %(proc.pid, t_process))
    print("Command: %s" %(cmd))
    process_pids.append(proc.pid)
    if (t_process != 0):
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
    for pid in process_pids:
        KillProcess(proc_pid=pid)

atexit.register(Exiting)

if __name__ == "__main__":
    if(len(sys.argv) > 1):
        terminal_output = True
    CheckFolders()

    for spot_idx in range(num_spots):
        # gimbal & lidar data transmitter
        CreateProcess(cmd=gimbal_transmit_cmd, t_process=5*75, t_output=5)
        CreateProcess(cmd=lidar_transmit_cmd, t_process=5*75, t_output=10)
        # reset gimbal, 15s
        gimbal_publish_cmd = GetGimbalPublisherCmd(view_idx='center')
        CreateProcess(cmd=gimbal_publish_cmd, t_process=15)
        for view_idx in range(num_views):
            # gimbal.launch, 15s
            gimbal_publish_cmd = GetGimbalPublisherCmd(view_idx)
            CreateProcess(cmd=gimbal_publish_cmd, t_process=15)
            time.sleep(15)
            # rosbag record + FisheyeCapture.py, 60s
            record_view_cmd = GetViewBagCmd(spot_idx, view_idx)
            CreateProcess(cmd=record_view_cmd, t_process=60)
            fisheye_cmd = GetFisheyeCmd(spot_idx, view_idx)
            CreateProcess(cmd=fisheye_cmd, t_process=60, t_output=60)
            time.sleep(60)
        # reset gimbal, 15s
        gimbal_publish_cmd = GetGimbalPublisherCmd(view_idx='center')
        CreateProcess(cmd=gimbal_publish_cmd, t_process=15)
        # livox_lidar_msg.launch, 60s
        CreateProcess(cmd=lidar_liomsg_cmd, t_process=60, t_output=10)
        time.sleep(15)
        # rosbag record, 30s
        record_lio_cmd = GetLioBagCmd(source_spot_idx=spot_idx, target_spot_idx=spot_idx+1)
        CreateProcess(cmd=record_lio_cmd, t_process=30)
        time.sleep(45)
