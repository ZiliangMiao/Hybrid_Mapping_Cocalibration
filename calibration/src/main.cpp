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

#include "glog/logging.h"
/** heading **/
#include "optimization.h"
#include "common_lib.h"
/** namespace **/
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    /***** ROS Initialization *****/
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    /***** ROS Parameters Server *****/
    bool kGenerateOmniEdge = false;
    bool kGenerateLidarEdge = false;
    bool kGenerateViewCloud = false;
    bool kStitchViewCloud = false;
    bool kGenerateSpotCloud = false;
    bool kSpotColorization = false;
    bool kStitchSpotCloud = false;
    bool kGlobalMapping = false;
    bool kGlobalColoredMapping = false;
    bool kCeresOptimization = false;
    bool kMultiSpotsOptimization = false;
    bool kParamsAnalysis = false;
    bool kUniformSampling = false;
    int kOneSpot = 0; /** -1 means run all the spots, other means run a specific spot **/

    nh.param<bool>("switch/kGenerateLidarEdge", kGenerateLidarEdge, false);
    nh.param<bool>("switch/kGenerateOmniEdge", kGenerateOmniEdge, false);
    nh.param<bool>("switch/kGenerateViewCloud", kGenerateViewCloud, false);
    nh.param<bool>("switch/kStitchViewCloud", kStitchViewCloud, false);
    nh.param<bool>("switch/kGenerateSpotCloud", kGenerateSpotCloud, false);
    nh.param<bool>("switch/kSpotColorization", kSpotColorization, false);
    nh.param<bool>("switch/kStitchSpotCloud", kStitchSpotCloud, false);
    nh.param<bool>("switch/kGlobalMapping", kGlobalMapping, false);
    nh.param<bool>("switch/kGlobalColoredMapping", kGlobalColoredMapping, false);
    nh.param<bool>("switch/kCeresOptimization", kCeresOptimization, false);
    nh.param<bool>("switch/kMultiSpotsOptimization", kMultiSpotsOptimization, false);
    nh.param<bool>("switch/kParamsAnalysis", kParamsAnalysis, false);
    nh.param<bool>("switch/kUniformSampling", kUniformSampling, false);
    nh.param<int>("spot/kOneSpot", kOneSpot, -1);

    google::InitGoogleLogging(argv[0]);

    /***** Initial Parameters *****/
    std::vector<double> params_init = {
        M_PI + 0.02, 0.02, -M_PI/2, /** Rx Ry Rz **/
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
    OmniProcess omnicam;
    LidarProcess lidar;
    lidar.ext_ = Eigen::Map<Param_D>(params_init.data()).head(6);
    omnicam.int_ = Eigen::Map<Param_D>(params_init.data()).tail(K_INT);

    /***** Data Folder Check **/
    for (int i = 0; i < lidar.num_spots; ++i) {
        string spot_path = lidar.kDatasetPath + "/spot" + to_string(i);
        CheckFolder(spot_path);
        string log_path = lidar.kDatasetPath + "/log";
        CheckFolder(log_path);
        for (int j = 0; j < lidar.num_views; ++j) {
            int view_degree = lidar.view_angle_init + lidar.view_angle_step * j;
            string view_path = spot_path + "/" + to_string(view_degree);
            string center_view_path = spot_path + "/recon";
            CheckFolder(view_path);
            CheckFolder(view_path + "/bags");
            CheckFolder(view_path + "/images");
            CheckFolder(view_path + "/edges");
            CheckFolder(view_path + "/outputs");
            CheckFolder(view_path + "/outputs/omni_outputs");
            CheckFolder(view_path + "/outputs/lidar_outputs");
            CheckFolder(view_path + "/results");
            CheckFolder(center_view_path);
        }
    }

    /***** Registration, Colorization and Mapping *****/
    for (int i = 0; i < lidar.num_spots; ++i) {
        if (kOneSpot == -1 || kOneSpot == i) {
            lidar.setSpot(i);
            for (int j = 0; j < lidar.num_views; ++j) {
                lidar.setView(j);
                if (kGenerateViewCloud) {
                    lidar.generateViewCloud();
                }
            }
            for (int j = 0; j < lidar.num_views; ++j) {
                lidar.setView(j);
                if (kStitchViewCloud && j != lidar.center_view_idx) {
                    lidar.stitchViewCloud();
                }
            }
            if (kGenerateSpotCloud) {
                lidar.generateSpotCloud();
            }
        }
    }

    /***** Data Process *****/
    
    for (int i = 0; i < lidar.num_spots; ++i) {
        if (kOneSpot == -1 || kOneSpot == i) {
            if (kGenerateLidarEdge) {
                CloudI::Ptr lidar_cart_cloud(new CloudI);
                CloudI::Ptr lidar_polar_cloud(new CloudI);
                lidar.setSpot(i);
                lidar.setView(lidar.center_view_idx);
                lidar.lidarToSphere(lidar_cart_cloud, lidar_polar_cloud);
                lidar.sphereToPlane(lidar_polar_cloud);
                lidar.edgeExtraction();
                lidar.generateEdgeCloud(lidar_cart_cloud);
            }
            if (kGenerateOmniEdge) {
                omnicam.setSpot(i);
                omnicam.setView(omnicam.fullview_idx);
                omnicam.loadImage(true);
                omnicam.edgeExtraction();
                omnicam.generateEdgeCloud(); 

            }
        }
    }
    

    /***** Calibration and Optimization Cost Analysis *****/
    if (kCeresOptimization) {
        cout << "----------------- Ceres Optimization ---------------------" << endl;
        std::vector<double> lb(dev.size()), ub(dev.size());
        std::vector<double> bw = {32, 16, 4, 1};
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
                omnicam.setSpot(spot);
                lidar.setSpot(spot);
                omnicam.setView(omnicam.fullview_idx);
                lidar.setView(lidar.center_view_idx);
                omnicam.ReadEdge();
                lidar.ReadEdge();
                
                project2Image(omnicam, lidar, params_init, 0); /** 0 - invalid bandwidth to initialize the visualization **/
                string record_path = lidar.file_path_vec[lidar.spot_idx][lidar.view_idx].result_folder_path
                                    + "/result_spot" + to_string(lidar.spot_idx) + ".txt";
                saveResults(record_path, params_init, 0, 0, 0);
            }
        }

        bool kParamsAnalysis = false;
        ros::param::get("switch/kAnalysis", kParamsAnalysis);

        for (int spot = 0; spot < lidar.num_spots; ++spot) {
            if (kOneSpot == -1 || kOneSpot == spot) {
                if (kMultiSpotsOptimization && kOneSpot == -1) {
                    vector<int> spot_init_vec(lidar.num_spots);
                    std::iota(spot_init_vec.begin(), spot_init_vec.end(), 0);
                    spot_vec = spot_init_vec;
                }
                else {
                    spot_vec = {spot};
                }

                for (int i = 0; i < bw.size(); i++) {
                    double bandwidth = bw[i];
                    vector<double> init_params_vec(params_calib);
                    params_calib = QuaternionCalib(omnicam, lidar, bandwidth, spot_vec, params_calib, lb, ub, false);
                    // if (i == bw.size() - 1) {
                    //     params_calib = QuaternionCalib(fisheye, lidar, bandwidth, spot_vec, params_calib, lb, ub, true);
                    // }
                    if (kParamsAnalysis) {
                        costAnalysis(omnicam, lidar, spot_vec, init_params_vec, params_calib, bandwidth);
                    }
                }

                if (kMultiSpotsOptimization) { break;}
            }
        }
    }

    /***** Registration, Colorization and Mapping *****/
    /** spot **/
        if (kSpotColorization) {
        cout << "----------------- Spot Cloud Colorization ---------------------" << endl;
        // lh3_global:
        // params_calib = {
        //         0.000472, -3.139975, 1.563091, /** Rx Ry Rz **/
        //         0.274670, -0.012239, 0.034630, /** tx ty tz **/
        //         1022.412883, 1199.429484,       /** u0, v0 **/
        //         1995.940476, -696.447201, 27.426648, 2.044011, -1.568044, 
        //         0.999972, -0.008120, 0.007628
        // }
        // parking:
        // params_calib = {
        //         0.001335, -3.139391, 1.559892,
        //         0.281820, -0.006560, 0.044851,
        //         1024.081111, 1197.734465,
        //         1986.768694, -691.831611, 37.178636, -6.742971, 0.362401,
        //         1.000177, -0.005878, 0.006144
        // };
        // bs_hall:
        params_calib = {
                0.000990, -3.138274, 1.560729,
                0.291672, -0.005141, 0.038923,
                1021.553425, 1196.789762,
                1995.359807, -666.475065, -13.940682, 20.000000, -4.049823,
                0.998262, -0.005735, 0.005314
        };
        
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                omnicam.setSpot(i);
                lidar.setSpot(i);
                omnicam.setView(lidar.center_view_idx);
                lidar.setView(lidar.center_view_idx);
                SpotColorization(omnicam, lidar, params_calib);
            }
        }
    }

    if (kStitchSpotCloud) {
        cout << "----------------- Spot Registration ---------------------" << endl;
        for (int i = lidar.num_spots - 1; i > 0; --i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.setSpot(i);
                lidar.stitchSpotCloud();
            }
        }
        
    }

    if (kGlobalMapping) {
        cout << "----------------- Global Mapping ---------------------" << endl;
        lidar.generateFineMap(kUniformSampling);
    }

    if (kGlobalColoredMapping) {
        cout << "----------------- Global Colored Mapping ---------------------" << endl;
        lidar.generateColoredFineMap(kUniformSampling);
    }

    return 0;
}
