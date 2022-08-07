import os, sys, time, atexit
import shutil
import numpy as np
from auto_run import CreateProcess, Exiting

view_path_list = []
dataset = "conf1"

def ReformatBags(path, filename):
    if ("spot" in filename) and (".bag" in filename):
        print(filename)
        filename_list = filename.split(dataset)[1].split(".")[0].split("_")
        if len(filename_list) == 3:
            [_, spot_name, view_name] = filename_list
        else:
            [_, spot_name, view_name, _] = filename_list
        full_path = os.path.join(path, filename).replace('\\', '/')
        rename_path = os.path.join(path, dataset + "_" + spot_name + "_" + view_name + ".bag").replace('\\', '/')
        os.rename(full_path, rename_path)
        if ("spot" not in view_name):
            BagToPcd(rename_path, path)
            view_path = os.path.abspath(path)
            view_path_list.append(view_path)

def BagToPcd(input_file, path):
    cmd_prefix = "rosrun pcl_ros bag_to_pcd"
    topic = "/livox/lidar"
    output_path = os.path.abspath(os.path.join(path, "../all_pcds/"))
    print(output_path)
    cmd = cmd_prefix + " " + input_file + " " + topic + " " + output_path
    CreateProcess(cmd, t_process=15)

def DataCleaner():
    dir = os.path.abspath("/home/isee/catkin_ws/src/Livox_Fisheye_Fusion/calibration/data/" + dataset)
    # dir = os.path.abspath("/home/isee/catkin_ws/data/" + dataset)
    for path, _, files in os.walk(dir):
        for filename in files:
            ReformatBags(path, filename)

def MovePcds(path):
    target_length = 500
    source_path = os.path.abspath(os.path.join(path, "../all_pcds/"))
    target_path = os.path.abspath(os.path.join(path, "../dense_pcds/"))
    print(source_path)
    for _, _, files in os.walk(source_path):
        start_idx = int((len(files) - target_length) / 2)
        end_idx = start_idx + target_length
        file_list = np.sort(files)
        for filename in file_list[start_idx:end_idx]:
            source_file = source_path + "/" + filename
            target_file = target_path + "/" + filename
            print(source_file)
            print(target_file)
            shutil.copyfile(source_file, target_file)

atexit.register(Exiting)

if __name__ == "__main__":
    DataCleaner()
    # time.sleep(20)
    for spot_path in view_path_list:
        MovePcds(spot_path)