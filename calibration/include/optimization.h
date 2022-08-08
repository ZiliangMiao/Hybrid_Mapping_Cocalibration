// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
// eigen
#include <Eigen/Core>
// ros
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <ros/package.h>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
// ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
// pcl
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
// headings
#include "fisheye_process.h"
#include "lidar_process.h"

using namespace std;

static const int kExtrinsics = 7;
static const int kIntrinsics = 10;

inline double getDouble(double x) {
    return static_cast<double>(x);
}

template <typename SCALAR, int N>
inline double getDouble(const ceres::Jet<SCALAR, N> &x) {
    return static_cast<double>(x.a);
}

void Visualization2D(FisheyeProcess &fisheye, LidarProcess &lidar, std::vector<double> &params, double bandwidth);

void Visualization3D(FisheyeProcess &fisheye, LidarProcess &lidar, std::vector<double> &params);

std::vector<double> QuaternionCalib(FisheyeProcess &fisheye,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<int> spot_vec,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub);

void CorrelationAnalysis(FisheyeProcess &fisheye,
                        LidarProcess &lidar,
                        std::vector<int> spot_vec,
                        std::vector<double> init_params_vec,
                        std::vector<double> result_vec,
                        double bandwidth);