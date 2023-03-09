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
#include "common_lib.h"
/** namespace **/
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    /***** ROS Initialization *****/
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    /***** ROS Parameters Server *****/
    bool kCeresOpt = false;
    bool kMultiSpotOpt = false;
    bool kParamsAnalysis = false;
    bool kUniformSampling = false;
    nh.param<bool>("switch/kCeresOpt", kCeresOpt, false);
    nh.param<bool>("switch/kMultiSpotOpt", kMultiSpotOpt, false);
    nh.param<bool>("switch/kParamsAnalysis", kParamsAnalysis, false);
    nh.param<bool>("switch/kUniformSampling", kUniformSampling, false);
    /** Initialization **/
    std::vector<double> bw;
    nh.param<vector<double>>("cocalib/bw", bw, {32, 16, 8, 4, 2, 1});
    // Extrinsic Params
    double rx, ry, rz, tx, ty, tz;
    double rx_range, ry_range, rz_range, tx_range, ty_range, tz_range;
    nh.param<double>("cocalib/rx", rx, 0.00);
    nh.param<double>("cocalib/ry", ry, 0.00);
    nh.param<double>("cocalib/rz", rz, 0.00);
    nh.param<double>("cocalib/tx", tx, 0.00);
    nh.param<double>("cocalib/ty", ty, 0.00);
    nh.param<double>("cocalib/tz", tz, 0.00);
    nh.param<double>("cocalib/rx_range", rx_range, 0.00);
    nh.param<double>("cocalib/ry_range", ry_range, 0.00);
    nh.param<double>("cocalib/rz_range", rz_range, 0.00);
    nh.param<double>("cocalib/tx_range", tx_range, 0.00);
    nh.param<double>("cocalib/ty_range", ty_range, 0.00);
    nh.param<double>("cocalib/tz_range", tz_range, 0.00);
    // Intrinsic Params
    double u0, v0, a0, a1, a2, a3, a4, c, d, e;
    double u0_range, v0_range, a0_range, a1_range, a2_range, a3_range, a4_range, c_range, d_range, e_range;
    nh.param<double>("cocalib/u0", u0, 1024);
    nh.param<double>("cocalib/v0", v0, 1201);
    nh.param<double>("cocalib/a0", a0, 0.00);
    nh.param<double>("cocalib/a1", a1, 0.00);
    nh.param<double>("cocalib/a2", a2, 0.00);
    nh.param<double>("cocalib/a3", a3, 0.00);
    nh.param<double>("cocalib/a4", a4, 0.00);
    nh.param<double>("cocalib/c", c, 1.00);
    nh.param<double>("cocalib/d", d, 0.00);
    nh.param<double>("cocalib/e", e, 0.00);
    nh.param<double>("cocalib/u0_range", u0_range, 0.00);
    nh.param<double>("cocalib/v0_range", v0_range, 0.00);
    nh.param<double>("cocalib/a0_range", a0_range, 0.00);
    nh.param<double>("cocalib/a1_range", a1_range, 0.00);
    nh.param<double>("cocalib/a2_range", a2_range, 0.00);
    nh.param<double>("cocalib/a3_range", a3_range, 0.00);
    nh.param<double>("cocalib/a4_range", a4_range, 0.00);
    nh.param<double>("cocalib/c_range", c_range, 0.00);
    nh.param<double>("cocalib/d_range", d_range, 0.00);
    nh.param<double>("cocalib/e_range", e_range, 0.00);

    std::vector<double> params_init = {
        rx, ry, rz, tx, ty, tz,
        u0, v0, a0, a1, a2, a3, a4, c, d, e};
    std::vector<double> params_cocalib(params_init);
    std::vector<double> params_range = {
        rx_range, ry_range, rz_range, tx_range, ty_range, tz_range,
        u0_range, v0_range,
        a0_range, a1_range, a2_range, a3_range, a4_range,
        c_range, d_range, e_range};

    cout << "CHECK ROS PARAMS!" << rx << " " << ry << " " << rz << " " << a0 << endl;

    /***** Class Object Initialization *****/
    OmniProcess omni;
    LidarProcess lidar;
    lidar.ext_ = Eigen::Map<Param_D>(params_init.data()).head(6);
    omni.int_ = Eigen::Map<Param_D>(params_init.data()).tail(K_INT);

    /***** Folder Check **/
    CheckFolder(lidar.DATASET_PATH);
    CheckFolder(lidar.COCALIB_PATH);
    CheckFolder(lidar.EDGE_PATH);
    CheckFolder(lidar.RESULT_PATH);

    /***** Calibration and Optimization Cost Analysis *****/
    if (kCeresOpt) {
        std::vector<double> lb(params_range.size()), ub(params_range.size());
        for (int i = 0; i < params_range.size(); ++i) {
            ub[i] = params_init[i] + params_range[i];
            lb[i] = params_init[i] - params_range[i];
        }
        Eigen::Matrix<double, 3, 17> params_mat;
        params_mat.row(0) = Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_init.data());
        params_mat.row(1) = params_mat.row(0) - Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_range.data());
        params_mat.row(2) = params_mat.row(0) + Eigen::Map<Eigen::Matrix<double, 1, 17>>(params_range.data());
        /********* Pre Processing *********/
        cout << "----------------- Ocam Processing ---------------------" << endl;
        omni.loadCocalibImage();
        omni.edgeExtraction();
        omni.generateEdgeCloud();
        cout << "----------------- LiDAR Processing ---------------------" << endl;
        lidar.cartToSphere();
        lidar.sphereToPlane();
        lidar.edgeExtraction();
        lidar.generateEdgeCloud();
        /********* Init Viz *********/
        std::string fusion_image_path_init = omni.RESULT_PATH + "/fusion_image_init.bmp";
        std::string cocalib_result_path_init = lidar.RESULT_PATH + "/cocalib_init.txt";
        double proj_error = project2Image(omni, lidar, params_init, fusion_image_path_init, 0); // 0 - invalid bandwidth to initialize the visualization
        saveResults(cocalib_result_path_init, params_init, 0, 0, 0, proj_error);
        
        std::vector<int> spot_vec;
        if (lidar.NUM_SPOT == 1) {
            if (kMultiSpotOpt && lidar.NUM_SPOT != 1) {
                vector<int> spot_init_vec(lidar.NUM_SPOT);
                std::iota(spot_init_vec.begin(), spot_init_vec.end(), 0);
                spot_vec = spot_init_vec;
            }
            else {
                spot_vec = {0};
            }
            cout << "----------------- Ceres Optimization ---------------------" << endl;
            for (int i = 0; i < bw.size(); i++) {
                double bandwidth = bw[i];
                vector<double> init_params_vec(params_cocalib);
                params_cocalib = QuaternionCalib(omni, lidar, bandwidth, spot_vec, params_cocalib, lb, ub, false);
                if (kParamsAnalysis) {
                    costAnalysis(omni, lidar, spot_vec, init_params_vec, params_cocalib, bandwidth);
                }
            }
        }
    }
    return 0;
}
