#!/bin/bash

ROS_LAUNCH=$(pwd)/pcd_view.launch
CFG_FILE=$(pwd)/pcd_view.rviz
roslaunch ${ROS_LAUNCH} rviz_config:=${CFG_FILE} pcd_file:=$1

