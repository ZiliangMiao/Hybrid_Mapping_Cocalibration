import os, sys

from AutoRun import CreateProcess

def ReformatBags(path, filename):
    if ("spot" in filename) and (".bag" in filename):
        print(filename)
        filename_list = filename.split("_")
        if len(filename_list) == 3:
            [dataset_name, spot_name, view_name] = filename_list
        else:
            [dataset_name, spot_name, view_name, _] = filename_list
        full_path = os.path.join(path, filename).replace('\\', '/')
        rename_path = os.path.join(path, dataset_name + "_" + spot_name + "_" + view_name + ".bag").replace('\\', '/')
        os.rename(full_path, rename_path)
        if ("spot" not in view_name):
            BagToPcd(rename_path, path)

def BagToPcd(input_file, path):
    cmd_prefix = "rosrun pcl_ros bag_to_pcd"
    topic = "/livox/lidar"
    output_path = os.path.abspath(os.path.join(path, "../all_pcds/"))
    print(output_path)
    cmd = cmd_prefix + " " + input_file + " " + topic + " " + output_path
    CreateProcess(cmd, t_process=20)

def DataCleaner():
    dir = os.path.abspath("/home/isee/catkin_ws/data/hesuan")
    for path, _, files in os.walk(dir):
        for filename in files:
            ReformatBags(path, filename)

DataCleaner()