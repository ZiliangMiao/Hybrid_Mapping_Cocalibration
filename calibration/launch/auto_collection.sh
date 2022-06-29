gnome-terminal -t "livox" -x bash -c "roslaunch calibration livox_driver.launch"
gnome-terminal -t "gimbal3" -x bash -c "roslaunch gimbal gimbal3.launch"
gnome-terminal -t "record3" -x bash -c "rosbag record /livox/lidar -o /home/godm/software/data/data_pose1_60.bag --duration=10s"


