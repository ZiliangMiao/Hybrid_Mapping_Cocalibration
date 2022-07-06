
gnome-terminal -t "gimbal3" -x bash -c "roslaunch gimbal gimbal3.launch" sleep 10	
gnome-terminal -t "gimbal4" -x bash -c "roslaunch gimbal gimbal4.launch" sleep 10


gnome-terminal -t "gimbal5" -x bash -c "roslaunch gimbal gimbal5.launch; sleep 10s;" sleep 10s
gnome-terminal -t "gimbal6" -x bash -c "roslaunch gimbal gimbal6.launch; sleep 10s;" sleep 10s
gnome-terminal -t "gimbal3" -x bash -c "roslaunch gimbal gimbal7.launch; sleep 10s;" sleep 10s

gnome-terminal -t "livox" -x bash -c "roslaunch calibration livox_driver.launch"
gnome-terminal -t "record3" -x bash -c "rosbag record /livox/lidar -o /home/isee/software/data/data_pose1_60.bag --duration=10s"
