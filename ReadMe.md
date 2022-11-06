# Coarse-to-fine Hybrid 3D Mapping System with Co-calibrated Omnidirectional Camera and Non-repetitive LiDAR
The project is an automatic calibration method for Livox mid-360 LiDAR and Fisheye Camera. The package is developed by Ziliang Miao, Buwei He, Wenya Xie, supervised by Prof.Xiaoping Hong ([ISEE-Lab](https://isee.technology/), SDIM, SUSTech).

Pre-print Paper: to be uploaded.

Demo Video: https://www.youtube.com/watch?v=Uh0C9VL9YEQ

## 0. Introduction

 <table>
	<tr>
	    <th>Hardware Platform</th>
	    <th>Item Names</th>
	    <th>Pictures</th>
	    <th>Shopping Links</th> 
	</tr >
	<tr >
            <td rowspan="4"><img src="readme_pics/robot.png" /></td>
	    <td>Livox Mid-360 </td>
	    <td align="center" valign="middle"><img src=  "readme_pics/mid360.JPG" width=60%/></td>
            <td align="center" valign="middle">  <a href ="https://www.livoxtech.com"> LiDAR </a> </td>
	</tr>
	<tr>
	    <td> MV-Fisheye Camera</td>
	    <td align="center" valign="middle"><img src="readme_pics/fisheye.JPG"width=60% /></td>
	    <td align="center" valign="middle">  <a href ="https://en.hikrobotics.com/vision/visioninfo.htm?type=42&oid=2451"> Camera </a> </td>
	</tr>
	<tr>
	    <td>Morefine S500+</td>
	    <td align="center" valign="middle"><img src="readme_pics/morefine.png" width=60% /></td>
            <td align="center" valign="middle">  <a href =https://morefines.com/products/mini-pc-s500-enclosure> Mini-Computer </a> </td>
	</tr>
	<tr>
	    <td> Scout-mini </td>
	    <td align="center" valign="middle"><img src="readme_pics/robot.png" width=60% /></td>
	    <td align="center" valign="middle">  <a href ="http://www.agilex.ai/index/product/id/3?lang=zh-cn"> Robot Chassis </a> </td>
	</tr>
</table>

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Version: Ubuntu 18.04.

Version: ROS Melodic. 

Please follow [ROS Installation](http://wiki.ros.org/ROS/Installation) to install.
### 1.2. **ceres-solver**
Version: ceres-solver 2.1.0

Please follow [Ceres-Solver Installation](http://ceres-solver.org/installation.html) to install.
### 1.3. **PCL**
Version: PCL 1.7.4

Version: Eigen 3.3.4

Please follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html) to install.
### 1.4. **OpenCV**
Version: OpenCV 3.2.0

Please follow [OpenCV Installation](https://opencv.org/) to install.
### 1.5. **mlpack**
Version: mlpack 3.4.2

Please follow [mlpack Installation](https://mlpack.org/) to install.

### 1.6 Livox SDK and Livox ROS Driver (Optional)
The SDK and driver is used for dealing with Livox LiDAR.
Remenber to install [Livox SDK](https://github.com/Livox-SDK/Livox-SDK) before [Livox ROS Driver](https://github.com/Livox-SDK/livox_ros_driver).

### 1.7 MindVision SDK (Optional)
The SDK of the fisheye camera is in [MindVision SDK](http://www.mindvision.com.cn/rjxz/list_12.aspx?lcid=138).




## 7. Acknowledgements
Thanks for [CamVox](https://github.com/ISEE-Technology/CamVox), [Livox-SDK](https://github.com/Livox-SDK/livox_camera_lidar_calibration). [OCamCalib MATLAB Toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab), thanks to the help of Wenquan Zhao, Xiao Huang, Jian Bai.
