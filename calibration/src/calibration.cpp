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
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointCloud<PointT>::Ptr CloudPtr;

const bool kFisheyeFlatProcess = false;
const bool kFisheyeEdgeProcess = false;
const bool kLidarFlatProcess = false;
const bool kLidarEdgeProcess = false;
const bool kCeresOptimization = false;
const bool kCreateDensePcd = false;
const bool kInitialIcp = false;
const bool kCreateFullViewPcd = true;
const bool kReconstruction = false;

/********* Directory Path of ROS Package *********/
string pkg_path = ros::package::getPath("calibration");

bool checkFolder(string folder_path){
    if(opendir(folder_path.c_str()) == NULL){                 // The first parameter of 'opendir' is char *
        int ret = mkdir(folder_path.c_str(), (S_IRWXU | S_IRWXG | S_IRWXO));       // 'mkdir' used for creating new directory
        if(ret == 0){
            cout << "Successfully create file folder!" << endl;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "calibration");
    ros::NodeHandle nh;

    if(!checkFolder(pkg_path)){
        return -1;
    }

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
        for (int i = 0; i < fisheye_process.num_spots; ++i) {
            fisheye_process.SetSpotIdx(i); /** spot idx **/
            for (int j = 0; j < fisheye_process.num_views; ++j) {
                fisheye_process.SetViewIdx(j); /** view idx **/
                std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye_process.FisheyeImageToSphere();
                RGBCloudPtr fisheye_polar_cloud;
                RGBCloudPtr fisheye_pixel_cloud;
                std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                fisheye_process.SphereToPlane(fisheye_polar_cloud);
            }
        }
    }
    else if (kFisheyeEdgeProcess) {
        for (int i = 0; i < fisheye_process.num_spots; ++i) {
            fisheye_process.SetSpotIdx(i); /** spot idx **/
            for (int j = 0; j < fisheye_process.num_views; ++j) {
                fisheye_process.SetViewIdx(j); /** view idx **/
                std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye_process.FisheyeImageToSphere();
                RGBCloudPtr fisheye_polar_cloud;
                RGBCloudPtr fisheye_pixel_cloud;
                std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                fisheye_process.SphereToPlane(fisheye_polar_cloud);
                fisheye_process.EdgeToPixel();
                fisheye_process.PixLookUp(fisheye_pixel_cloud);
            }
        }
    }

    cout << endl;
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    LidarProcess lidar_process(pkg_path);
    lidar_process.SetExtrinsic(params_calib);
    /********* Create Dense Pcd for All Scenes *********/
    if (kCreateDensePcd) {
        for (int i = 0; i < lidar_process.num_spots; ++i) {
            lidar_process.SetSpotIdx(i);
            for (int j = 0; j < lidar_process.num_views; ++j) {
                lidar_process.SetViewIdx(j);
                lidar_process.CreateDensePcd();
            }
        }
    }
    if (kInitialIcp) {
        lidar_process.SetSpotIdx(0);
        for (int i = 0; i < lidar_process.num_views; ++i) {
            if (i == (lidar_process.num_views-1)/2) {
                continue;
            }
            lidar_process.SetViewIdx(i);
            lidar_process.ICP();
        }
    }
    if (kInitialIcp && kCreateFullViewPcd) {
        /** generate full view pcds **/
        lidar_process.CreateFullviewPcd();
        /** pcl viewer visualization **/
        CloudPtr full_view(new CloudT);
        string fullview_cloud_path = lidar_process.fullview_rec_folder_path + "/fullview_sparse_cloud.pcd";
        pcl::io::loadPCDFile(fullview_cloud_path, *full_view);
        pcl::visualization::CloudViewer viewer("Viewer");
        viewer.showCloud(full_view);
        while (!viewer.wasStopped()) {

        }
        cv::waitKey();
    }
    if (kLidarFlatProcess) {
        // check
        for (int idx = 0; idx < lidar_process.num_views; idx++) {
            lidar_process.SetViewIdx(idx);
            std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
            CloudPtr lidCartesianCloud;
            CloudPtr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
        }
    }
    else if (kLidarEdgeProcess) {
        for (int i = 0; i < lidar_process.num_spots; ++i) {
            lidar_process.SetSpotIdx(i);
            lidar_process.SetViewIdx((lidar_process.num_views-1)/2);
            std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
            CloudPtr lidCartesianCloud;
            CloudPtr lidPolarCloud;
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
        vector<double> params_init = {0.0, 0.0, -M_PI/2, -0.25, 0.0, -0.05, 1023.0, 1201.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
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
        fisheye_process.SetSpotIdx(0);
        lidar_process.SetSpotIdx(0);
        for (int i = 0; i < fisheye_process.num_views; i++) {
            lidar_process.SetViewIdx(i);
            fisheye_process.SetViewIdx(i);
            lidar_process.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            fisheye_process.ReadEdge();
            fusionViz(fisheye_process, lidar_process, params_init, 88); /** 88 - invalid bandwidth to initialize the visualization **/
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

    if (kReconstruction) {
        // check
        int target_view_idx = 1; /** degree 0 **/
        fisheye_process.SetSpotIdx(0);
        lidar_process.SetSpotIdx(0);
        lidar_process.SetViewIdx(target_view_idx);
        fisheye_process.SetViewIdx(target_view_idx);
        vector<double> calib_params = {0.0, 0.0, M_PI/2, +0.25, 0.0, -0.05, 1026.0, 1200.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
        fusionViz3D(fisheye_process, lidar_process, calib_params);
    }
    return 0;
}
