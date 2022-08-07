/** basic **/
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
/** opencv **/
#include <opencv2/opencv.hpp>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
/** pcl **/
#include <pcl/common/io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
/** heading **/
#include "optimization.h"
#include "utils.h"
/** namespace **/
using namespace std;
using namespace cv;
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointCloud<PointT>::Ptr CloudPtr;

/** params service **/
//    ros::param::get("~param_test", param_test_1);
//    ros::NodeHandle nh("~");
//    nh.getParam("param_test", param_test_1);
//    /** get the parameters from ros parameters server **/
//    bool param_get1 = ros::param::get("param_test", param_test_1);
//    bool param_get = nh.getParam("param_test", param_test_1);
//    /** set the value of parameter to ros parameters server **/
//    ros::param::set("param_test", 520.00);
//    if (param_get) {
//        for (int i = 0; i < 10; ++i) {
//            cout << param_test_1 << endl;
//        }
//    }

int main(int argc, char** argv) {
    /***** ROS Initialization *****/
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    /***** ROS Parameters Server *****/
    bool kFisheyeFlatProcess = false;
    bool kLidarFlatProcess = false;
    bool kCreateDensePcd = false;
    bool kViewRegistration = false;
    bool kFullViewMapping = false;
    bool kFullViewColorization = false;
    bool kSpotRegistration = false;
    bool kGlobalMapping = false;
    bool kGlobalColoredMapping = false;
    bool kCeresOptimization = false;
    bool kParamsAnalysis = false;
    int kOneSpot = 0; /** -1 means run all the spots, other means run a specific spot **/

    nh.param<bool>("switch/kLidarFlatProcess", kLidarFlatProcess, false);
    nh.param<bool>("switch/kFisheyeFlatProcess", kFisheyeFlatProcess, false);
    nh.param<bool>("switch/kCreateDensePcd", kCreateDensePcd, false);
    nh.param<bool>("switch/kViewRegistration", kViewRegistration, false);
    nh.param<bool>("switch/kFullViewMapping", kFullViewMapping, false);
    nh.param<bool>("switch/kFullViewColorization", kFullViewColorization, false);
    nh.param<bool>("switch/kSpotRegistration", kSpotRegistration, false);
    nh.param<bool>("switch/kGlobalMapping", kGlobalMapping, false);
    nh.param<bool>("switch/kGlobalColoredMapping", kGlobalColoredMapping, false);
    nh.param<bool>("switch/kCeresOptimization", kCeresOptimization, false);
    nh.param<bool>("switch/kParamsAnalysis", kParamsAnalysis, false);
    nh.param<int>("spot/kOneSpot", kOneSpot, -1);

    /***** Initial Parameters *****/
    std::vector<double> params_init = {
        M_PI, 0.00, -M_PI/2, /** Rx Ry Rz **/
        0.27, 0.00, 0.03, /** tx ty tz **/
        1023.0, 1201.0, /** u0 v0 **/
        616.7214056132 * M_PI, -616.7214056132, 0.0, 0.0, 0.0,
        1, 0, 0 /** c, d, e **/
    }; /** the two sensors are parallel on y axis **/
    std::vector<double> params_calib(params_init);
    std::vector<double> dev = {
        1e-1, 1e-1, 1e-1,
        5e-2, 5e-2, 5e-2,
        5e+0, 5e+0,
        160e+0, 80e+0, 40e+0, 20+0, 10e+0,
        1e-2, 1e-2, 1e-2
    };

    /***** Class Object Initialization *****/
    FisheyeProcess fisheye;
    fisheye.SetIntrinsic(params_init);
    LidarProcess lidar;
    lidar.SetExtrinsic(params_init);

    /***** Data Folder Check **/
    for (int i = 0; i < lidar.num_spots; ++i) {
        string spot_path = lidar.kDatasetPath + "/spot" + to_string(i);
        CheckFolder(spot_path);
        string log_path = lidar.kDatasetPath + "/log";
        CheckFolder(log_path);
        for (int j = 0; j < lidar.num_views; ++j) {
            int view_degree = lidar.view_angle_init + lidar.view_angle_step * j;
            string view_path = spot_path + "/" + to_string(view_degree);
            string fullview_path = spot_path + "/fullview_recon";
            CheckFolder(view_path);
            CheckFolder(view_path + "/bags");
            CheckFolder(view_path + "/all_pcds");
            CheckFolder(view_path + "/dense_pcds");
            CheckFolder(view_path + "/icp_pcds");
            CheckFolder(view_path + "/images");
            CheckFolder(view_path + "/edges");
            CheckFolder(view_path + "/outputs");
            CheckFolder(view_path + "/outputs/fisheye_outputs");
            CheckFolder(view_path + "/outputs/lidar_outputs");
            CheckFolder(view_path + "/results");
            CheckFolder(fullview_path);
        }
    }

    /***** Registration, Colorization and Mapping *****/
    /** view **/
    if (kCreateDensePcd) {
        cout << "----------------- Merge Dense Point Cloud ---------------------" << endl;
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    lidar.SetViewIdx(j);
                    lidar.CreateDensePcd();
                }
            }
        }
    }
    
    if (kViewRegistration) {
        cout << "----------------- View Registration ---------------------" << endl;
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    if (j == lidar.fullview_idx) {
                        continue;
                    }
                    lidar.SetViewIdx(j);
                    lidar.ViewRegistration();
                }
            }
        }
    }
    /** full view **/
    if (kFullViewMapping) {
        cout << "----------------- Full View Mapping ---------------------" << endl;
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                lidar.FullViewMapping(); /** generate fullview pcds **/
            }
        }
    }

    /***** Data Process *****/
    if (kLidarFlatProcess) {
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                lidar.SetViewIdx(lidar.fullview_idx);
                CloudPtr cart_cloud(new CloudT);
                CloudPtr polar_cloud(new CloudT);
                lidar.LidarToSphere(cart_cloud, polar_cloud);
                lidar.SphereToPlane(cart_cloud, polar_cloud);
                lidar.EdgeExtraction();
                lidar.EdgeToPixel();
                lidar.PixLookUp(cart_cloud);
            }
        }
    }

    if (kFisheyeFlatProcess) {
        for (int i = 0; i < fisheye.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                fisheye.SetSpotIdx(i); /** spot idx **/
                fisheye.SetViewIdx(fisheye.fullview_idx);
                std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye.FisheyeImageToSphere();
                RGBCloudPtr fisheye_polar_cloud;
                RGBCloudPtr fisheye_pixel_cloud;
                std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                fisheye.SphereToPlane(fisheye_polar_cloud);
                fisheye.EdgeExtraction();
                fisheye.EdgeToPixel();
                fisheye.PixLookUp(fisheye_pixel_cloud);
            }
        }
    }

    /***** Calibration and Optimization Cost Analysis *****/
    if (kCeresOptimization) {
        cout << "----------------- Ceres Optimization ---------------------" << endl;
        std::vector<double> lb(dev.size()), ub(dev.size());
        std::vector<double> bw = {16, 8, 4, 2, 1};
        for (int i = 0; i < dev.size(); ++i) {
            ub[i] = params_init[i] + dev[i];
            lb[i] = params_init[i] - dev[i];
        }
        Eigen::Matrix<double, 3, 17> params_mat;
        params_mat.row(0) = Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_init.data());
        params_mat.row(1) = params_mat.row(0) - Eigen::Map<Eigen::Matrix<double, 1, 17>>(dev.data());
        params_mat.row(2) = params_mat.row(0) + Eigen::Map<Eigen::Matrix<double, 1, 17>>(dev.data());

        /********* Initial Visualization *********/
        std::vector<int> spot_vec;

        for (int spot = 0; spot < lidar.num_spots; ++spot) {
            if (kOneSpot == -1 || kOneSpot == spot) {
                fisheye.SetSpotIdx(spot);
                lidar.SetSpotIdx(spot);
                fisheye.SetViewIdx(fisheye.fullview_idx);
                lidar.SetViewIdx(lidar.fullview_idx);
                spot_vec = {spot};
                Visualization2D(fisheye, lidar, params_init, 0); /** 0 - invalid bandwidth to initialize the visualization **/
                string record_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].result_folder_path
                                    + "/result_spot" + to_string(lidar.spot_idx) + ".txt";
                SaveResults(record_path, params_init, 0, 0, 0);

                for (int i = 0; i < bw.size(); i++) {
                    double bandwidth = bw[i];
                    params_calib = QuaternionCalib(fisheye, lidar, bandwidth, spot_vec, params_calib, lb, ub);
                }
            }
        }
    }

    /***** Registration, Colorization and Mapping *****/
    /** spot **/
    if (kSpotRegistration) {
        cout << "----------------- Spot Registration ---------------------" << endl;
        for (int i = lidar.num_spots - 1; i > 0; --i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                lidar.SpotRegistration();
            }
        }
    }

    if (kFullViewColorization) {
        cout << "----------------- Full View Cloud Colorization ---------------------" << endl;
        // Current Best:
        params_calib = {
                0.00326059, 3.13658, 1.56319, /** Rx Ry Rz **/
                0.277415, -0.0112217, 0.046939, /** tx ty tz **/
                1022.53, 1198.45, /** u0, v0 **/
                1880.36, -536.721, -12.9298, -18.0154, 5.6414,
                1.00176, -0.00863924, 0.00846056
        };
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                fisheye.SetSpotIdx(i);
                lidar.SetSpotIdx(i);
                fisheye.SetViewIdx(lidar.fullview_idx);
                lidar.SetViewIdx(lidar.fullview_idx);
                Visualization3D(fisheye, lidar, params_calib);
            }
        }
    }

    if (kGlobalMapping) {
        cout << "----------------- Global Mapping ---------------------" << endl;
        lidar.GlobalMapping();
    }

    if (kGlobalColoredMapping) {
        cout << "----------------- Global Colored Mapping ---------------------" << endl;
        lidar.GlobalColoredMapping();
    }

    return 0;
}
