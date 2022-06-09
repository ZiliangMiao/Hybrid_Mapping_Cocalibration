# Livox_Fisheye_Fusion (LFF)
LFF project is an automatic calibration method for Livox mid-360 LiDAR and Fisheye Camera. The package is developed by MIAO Ziliang, He Buwei, Xie Wenya, WANG Zhenhu (ISEE Lab, SDIM, SUSTech), directed by Prof.HONG Xiaoping (ISEE Lab, SDIM, SUSTech).


## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 18.04.

ROS Melodic. Please follow [ROS Installation](http://wiki.ros.org/ROS/Installation)
### 1.2. **Ceres-Solver**
Please follow [Ceres-Solver Installation](http://ceres-solver.org/installation.html).
### 1.3. **PCL**
Please follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).
### 1.3. **OpenCV**
Please follow [OpenCV Installation](https://opencv.org/).
### 1.4. **mlpack**
Please follow [mlpack Installation](https://mlpack.org/).

## 2. Build 
Clone the repository and catkin_make:

```
    cd ~/catkin_ws/src/
    git clone https://github.com/SDIM-Fisheye/Livox_Fisheye_Fusion.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```
note: package name is data_process


## 3. Run
### Only for Livox Mid-360

```
    roslaunch loam_livox livox.launch
    roslaunch livox_ros_driver livox_lidar.launch
```

## 4. Results
### 4.1. Hardwares

<div align="center">
    <img src="readme_pics/robot.jpg" width=45% >
    <img src="readme_pics/robot.jpg" width=45% >
</div>

 <table>
	<tr>
	    <th>Hardware Platform</th>
	    <th>Item Names</th>
	    <th>Pictures</th>  
	    <th>Shopping Links</th> 
	</tr >
	<tr >
            <td rowspan="4"><img src="readme_pics/robot.jpg" /></td>
	    <td>Livox Mid-360 </td>
	    <td align="center" valign="middle"><img src=  "readme_pics/robot.jpg" width=25% /></td>
            <td align="center" valign="middle">  <a href ="https://www.livoxtech.com/horizon"> Lidar </a> </td>
	</tr>
	<tr>
	    <td> MV-Fisheye Camera</td>
	    <td align="center" valign="middle"><img src="readme_pics/robot.jpg" width=19% /></td>
	    <td align="center" valign="middle">  <a href ="https://en.hikrobotics.com/vision/visioninfo.htm?type=42&oid=2451"> Camera </a> </td>
	</tr>
	<tr>
	    <td>Morefine S500+</td>
	    <td align="center" valign="middle"><img src="readme_pics/robot.jpg" width=22% /></td>
            <td align="center" valign="middle">  <a href ="https://www.dji.com/cn/manifold-2"> Onboard-Computer </a> </td>
	</tr>
	<tr>
	    <td> Scout-mini </td>
	    <td align="center" valign="middle"><img src="readme_pics/robot.jpg" width=28% /></td>
	    <td align="center" valign="middle">  <a href ="http://www.agilex.ai/index/product/id/3?lang=zh-cn"> Robot Chassis </a> </td>
	</tr>
</table>


## 5. Acknowledgements
Thanks for [CamVox](https://github.com/ISEE-Technology/CamVox), [Livox-SDK](https://github.com/Livox-SDK/livox_camera_lidar_calibration). [OCamCalib MATLAB Toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab).
Thanks for the guidance of [ISEE-Lab](https://isee.technology/) and Prof.HONG Xiaoping.
