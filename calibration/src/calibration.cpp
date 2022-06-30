// basic
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
// opencv
#include <opencv2/opencv.hpp>
// ros
#include <ros/ros.h>
#include <ros/package.h>
// pcl
#include <pcl/common/io.h>
// heading
#include "ceresMultiScenes.cpp"

using namespace std;
using namespace cv;

const bool kFisheyeFlatProcess = false;
const bool kFisheyeEdgeProcess = false;
const bool kLidarFlatProcess = false;
const bool kLidarEdgeProcess = false;
const bool kCeresOptimization = false;
const bool k3DViz = false;
const bool kCreateDensePcd = false;
const bool kCreateFullViewPcd = true;

/********* Directory Path of ROS Package *********/
string GetPkgPath() {
    std::string pkg_path = ros::package::getPath("calibration");
    return pkg_path;
}
string pkg_path = GetPkgPath();

bool checkFolder(string folder_path){
    if(opendir(folder_path.c_str()) == NULL){                 // The first parameter of 'opendir' is char *
        int ret = mkdir(folder_path.c_str(), (S_IRWXU | S_IRWXG | S_IRWXO));       // 'mkdir' used for creating new directory
        if(ret == 0){
            cout << "Successfully create file folder!" << endl;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "calibration");
    ros::NodeHandle nh;

    string pkg_path = GetPkgPath();
    if(!checkFolder(pkg_path)){
        return -1;
    }

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

    /** fisheye intrinsics calibrated by chessboard **/
    vector<double> params_calib = {
        0.001, 0.0197457, 0.13,  0.00891695, 0.00937508, 0.14,
        606.16, -0.000558783, -2.70908E-09, -1.17573E-10,
        1.00014, -0.000177, 0.000129, 1023, 1201
    };

    cout << "----------------- Camera Processing ---------------------" << endl;
    FisheyeProcess fisheye_process(pkg_path);
    fisheye_process.SetIntrinsic(params_calib);

    if (kFisheyeFlatProcess) {
        for (int idx = 0; idx < fisheye_process.num_scenes; idx++) {
            fisheye_process.SetSceneIdx(idx);
            std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye_process.FisheyeImageToSphere();
            RGBCloudPtr fisheye_polar_cloud;
            RGBCloudPtr fisheye_pixel_cloud;
            std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
            fisheye_process.SphereToPlane(fisheye_polar_cloud);
        }
    }
    else if (kFisheyeEdgeProcess) {
        for (int idx = 0; idx < fisheye_process.num_scenes; idx++) {
            fisheye_process.SetSceneIdx(idx);
            std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye_process.FisheyeImageToSphere();
            RGBCloudPtr fisheye_polar_cloud;
            RGBCloudPtr fisheye_pixel_cloud;
            std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
            fisheye_process.SphereToPlane(fisheye_polar_cloud);
            fisheye_process.EdgeToPixel();
            fisheye_process.PixLookUp(fisheye_pixel_cloud);
        }
    }

    cout << endl;
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    LidarProcess lidar_process(pkg_path);
    lidar_process.SetExtrinsic(params_calib);
    ROS_ASSERT_MSG(lidar_process.num_scenes == fisheye_process.num_scenes, "num_scenes in FisheyeProcess and LidarProcess is not equal!");
    /********* Create Dense Pcd for All Scenes *********/
    if (kCreateDensePcd) {
        for (int idx = 0; idx < lidar_process.num_scenes; idx++) {
            lidar_process.SetSceneIdx(idx);
            lidar_process.CreateDensePcd();
        }
    }
    if (kCreateFullViewPcd) {
        /** generate full view pcds **/
        string full_pcds_path = lidar_process.scenes_path_vec[3] + "/full_pcds";
        cout << full_pcds_path << endl;
        lidar_process.CreateDensePcd(full_pcds_path);
    }
    if (kLidarFlatProcess) {
        for (int idx = 0; idx < lidar_process.num_scenes; idx++) {
            lidar_process.SetSceneIdx(idx);
            std::tuple<IntensityCloudPtr, IntensityCloudPtr> lidResult = lidar_process.LidarToSphere();
            IntensityCloudPtr lidCartesianCloud;
            IntensityCloudPtr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
        }
    }
    else if (kLidarEdgeProcess) {
        for (int idx = 0; idx < lidar_process.num_scenes; idx++) {
            lidar_process.SetSceneIdx(idx);
            std::tuple<IntensityCloudPtr, IntensityCloudPtr> lidResult = lidar_process.LidarToSphere();
            IntensityCloudPtr lidCartesianCloud;
            IntensityCloudPtr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
            lidar_process.EdgeToPixel();
            lidar_process.PixLookUp(lidCartesianCloud);
        }
    }
   
    cout << endl;
    cout << "----------------- Ceres Optimization ---------------------" << endl;
    if (kCeresOptimization) {
        /** a0, a1, a2, a3, a4; size of params = 13 **/
//         vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a2", "a3", "a4"};
//         vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.09, 1023.0, 1201.0, 0.80541495, 594.42999235, 44.92838635, -54.82428857, 20.81519032};
//         vector<double> dev = {5e-2, 5e-2, 2e-2, 1e-2, 1e-2, 3e-2, 2e+0, 2e+0, 2e+0, 2e+1, 15e+0, 10e+0, 5e+0};

        /** a0, a1, a2, a3, a4; size of params = 13 **/
        vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a2", "a3", "a4"};
        vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.09, 1023.0, 1201.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
        vector<double> dev = {5e-2, 5e-2, 2e-2, 1e-2, 1e-2, 3e-2, 2e+0, 2e+0, 5e+0, 100e+0, 100e+0, 80+0, 30e+0};

        /** a0, a1, a2, a3, a4; size of params = 13 **/
        // vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a2", "a3", "a4"};
        // vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.12, 1023.0, 1201.0, 0.80541495, 594.42999235, 44.92838635, -54.82428857, 20.81519032};
        // vector<double> dev = {5e-2, 5e-2, M_PI/300, 1e-2, 1e-2, 5e-2, 2e+0, 2e+0, 2e+0, 2e+1, 8e+0, 4e+0, 2e+0};

        /** a0, a1, a3, a5; size of params = 12 **/
//        vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a3", "a5"};
//        vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.09, 1023.0, 1201.0, 0.0, 609.93645006, -7.48070567, 3.22415532};
//        vector<double> dev = {2e-2, 2e-2, 4e-2, 1e-2, 1e-2, 3e-2, 5e+0, 5e+0, 1e+1, 2e+1, 4e+0, 2e+0};

        /** a0, a1, a3, a5; size of params = 12 **/
//        vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a3", "a5"};
//        vector<double> params_init = {0.0, 0.0, 0.1175, 0.0, 0.0, 0.09, 1023.0, 1201.0, 0.0, 616.7214056132, -1, 1};
//        vector<double> dev = {5e-2, 5e-2, 2e-2, 1e-2, 1e-2, 3e-2, 2e+0, 2e+0, 5e+0, 2e+1, 10e+0, 5e+0};

        /** a1, a3, a5; size of params = 11 **/
        // vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a1", "a3", "a5"};
        // vector<double> params_init = {0.0, 0.0, 0.1175, 0.0, 0.0, 0.16, 1023.0, 1201.0, 609.93645006, -7.48070567, 3.22415532};
        // vector<double> dev = {5e-2, 5e-2, M_PI/300, 1e-2, 1e-2, 5e-2, 5e+0, 5e+0, 2e+1, 6e+0, 2e+0};

        vector<double> params = params_init;
        vector<double> lb(dev.size()), ub(dev.size());
        vector<double> bw = {32, 32, 16, 8, 4, 2};

        for (int i = 0; i < dev.size(); ++i) {
            ub[i] = params_init[i] + dev[i];
            lb[i] = params_init[i] - dev[i];
            if (i == dev.size() - 2){
                ub[i] = params_init[i];
                lb[i] = params_init[i] - dev[i];
            }
            if (i == dev.size() - 1 || i == dev.size() - 3){
                ub[i] = params_init[i] + dev[i];
                lb[i] = params_init[i];
            }
        }

        /********* Initial Visualization *********/
        for (int idx = 0; idx < fisheye_process.num_scenes; idx++) {
            lidar_process.SetSceneIdx(idx);
            fisheye_process.SetSceneIdx(idx);
            lidar_process.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            fisheye_process.ReadEdge();
            vector<vector<double>> edge_fisheye_projection = lidar_process.EdgeCloudProjectToFisheye(params_init);
            cout << "Edge Trans Txt Path:" << lidar_process.scenes_files_path_vec[idx].edge_fisheye_projection_path << endl;
            fusionViz(fisheye_process, lidar_process.scenes_files_path_vec[idx].edge_fisheye_projection_path, edge_fisheye_projection, 88); /** 88 - invalid bandwidth to initialize the visualization **/
        }

        for (int i = 0; i < bw.size(); i++) {
            double bandwidth = bw[i];
            cout << "Round " << i << endl;
            /**
             * kDisabledBlock = 0 -> enable all the params
             * kDisabledBlock = 1 -> enable intrinsics only
             * kDisabledBlock = 2 -> enable extrinsics only
             * **/
            if (i == 0) {
                int kDisabledBlock = 2;
                params = ceresMultiScenes(fisheye_process, lidar_process, bandwidth, params, name, lb, ub, kDisabledBlock);
//                kDisabledBlock = 1;
//                params = ceresMultiScenes(FisheyeProcess, LidarProcess, bandwidth, params, name, lb, ub, kDisabledBlock);
            }
            else {
                int kDisabledBlock = 0;
                params = ceresMultiScenes(fisheye_process, lidar_process, bandwidth, params, name, lb, ub, kDisabledBlock);
            }
        }
    }

    if (k3DViz) {
        lidar_process.SetSceneIdx(1);
        fisheye_process.SetSceneIdx(1);
        vector<double> test_params = {-0.0131396, 0.0179037, 0.116701, 0.01, 0.00374594, 0.118988, 1021.0, 1199.0, 2.79921, 606.544, 48.3143, -54.8969, 17.7703};
        int step = 5;
        fusionViz3D(fisheye_process, lidar_process, test_params, 5);
    }
    return 0;
}
