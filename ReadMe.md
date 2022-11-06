# Coarse-to-fine Hybrid 3D Mapping System with Co-calibrated Omnidirectional Camera and Non-repetitive LiDAR
The project is an automatic calibration method for Livox mid-360 LiDAR and Fisheye Camera. The package is developed by Ziliang Miao, Buwei He, Wenya Xie, supervised by Prof.Xiaoping Hong ([ISEE-Lab](https://isee.technology/), SDIM, SUSTech).

Pre-print Paper: to be uploaded.

Demo Video: https://www.youtube.com/watch?v=Uh0C9VL9YEQ

## 0. Introduction
We presents a novel omnidirectional field-of-view (FoV) 3D scanning sensor suite composed of a non-repetitive scanning LiDAR, a fisheye camera, and a gimbal mount. Thanks to the non-repetitive nature of the LiDAR, an automatic and targetless co-calibration method with simultaneous intrinsic calibration for the fisheye camera and extrinsic calibration for the sensor suite is proposed, which is a crucial step in combining the color images with the 3D point clouds. Analyses and comparisons are made to target-based intrinsic calibration and mutual information (MI) based extrinsic calibration, respectively. Contrary to sensors based on the conventional LiDARs, this sensor suite permits a coarse-to-fine approach in robotic 3D scanning by obtaining the coarse global map with odometry/SLAM-based methods first, generating scanning viewpoints from the global map, and obtaining finer and more precise 3D scanning of the region-of-interest (ROI) through stationary non-repetitive scanning at these respective viewpoints. The still scan results are registered together to a fine map of ROI, then stitched with the global map. More accurate and robust scanning results are obtained compared to odometry/SLAM-only methods.

 <table>
	<tr>
	    <th>Mapping System</th>
	    <th>Sensor Suite</th>
	</tr >
	<tr >
	    <td align="center" valign="middle"><img src=  "readme_pics/robot.png" width=60%/></td>
	    <td align="center" valign="middle"><img src=  "readme_pics/sensor_suite.png" width=60%/></td>
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

## n. Acknowledgements
Thanks for [CamVox](https://github.com/ISEE-Technology/CamVox), [Livox-SDK](https://github.com/Livox-SDK/livox_camera_lidar_calibration). [OCamCalib MATLAB Toolbox](https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab), thanks to the help of Wenquan Zhao, Xiao Huang, Jian Bai.
