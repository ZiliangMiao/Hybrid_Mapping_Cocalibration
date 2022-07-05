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
/** heading **/
#include "ceresMultiScenes.cpp"
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

/** switch **/
const bool kFisheyeFlatProcess = false;
const bool kFisheyeEdgeProcess = false;
const bool kLidarFlatProcess = false;
const bool kLidarEdgeProcess = false;
const bool kCeresOptimization = false;
const bool kCreateDensePcd = false;
const bool kInitialIcp = false;
const bool kCreateFullViewPcd = false;
const bool kReconstruction = true;
const int kOneSpot = 0; /** -1 means run all the spots, other means run a specific spot **/

int CheckFolder(string spot_path) {
    int md = 0; /** 0 means the folder is already exist or has been created successfully **/
    if (0 != access(spot_path.c_str(), 0)) {
        /** if this folder not exist, create a new one **/
        md = mkdir(spot_path.c_str(), S_IRWXU);
    }
    return md;
}

int main(int argc, char** argv) {
    /** ros initialization **/
    ros::init(argc, argv, "calibration");
    ros::NodeHandle nh;

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

    /** class object generation **/
    FisheyeProcess fisheye;
    LidarProcess lidar;
    /** data folder check **/
    for (int i = 0; i < lidar.num_spots; ++i) {
        string spot_path = lidar.kDatasetPath + "/spot" + to_string(i);
        CheckFolder(spot_path);
        for (int j = 0; j < lidar.num_views; ++j) {
            int view_degree = -lidar.view_angle_step + lidar.view_angle_step * j;
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
        }
    }

    vector<double> initial_params = {
        0.001, 0.0197457, 0.13,  0.00891695, 0.00937508, 0.14,
        606.16, -0.000558783, -2.70908E-09, -1.17573E-10,
        1.00014, -0.000177, 0.000129, 1023, 1201
    }; /** fisheye intrinsics here are calibrated by chessboard **/

    cout << "----------------- Fisheye Processing ---------------------" << endl;
    fisheye.SetIntrinsic(initial_params);

    if (kFisheyeFlatProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < fisheye.num_spots; ++i) {
                fisheye.SetSpotIdx(i); /** spot idx **/
                for (int j = 0; j < fisheye.num_views; ++j) {
                    fisheye.SetViewIdx(j); /** view idx **/
                    std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye.FisheyeImageToSphere();
                    RGBCloudPtr fisheye_polar_cloud;
                    RGBCloudPtr fisheye_pixel_cloud;
                    std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                    fisheye.SphereToPlane(fisheye_polar_cloud);
                }
            }
        }
        else {
            fisheye.SetSpotIdx(kOneSpot); /** spot idx **/
            for (int j = 0; j < fisheye.num_views; ++j) {
                fisheye.SetViewIdx(j); /** view idx **/
                std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye.FisheyeImageToSphere();
                RGBCloudPtr fisheye_polar_cloud;
                RGBCloudPtr fisheye_pixel_cloud;
                std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                fisheye.SphereToPlane(fisheye_polar_cloud);
            }
        }
//        fisheye.EdgeExtraction(0);
    }
    else if (kFisheyeEdgeProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < fisheye.num_spots; ++i) {
                fisheye.SetSpotIdx(i); /** spot idx **/
                for (int j = 0; j < fisheye.num_views; ++j) {
                    fisheye.SetViewIdx(j); /** view idx **/
                    std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye.FisheyeImageToSphere();
                    RGBCloudPtr fisheye_polar_cloud;
                    RGBCloudPtr fisheye_pixel_cloud;
                    std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                    fisheye.SphereToPlane(fisheye_polar_cloud);
                    fisheye.EdgeToPixel();
                    fisheye.PixLookUp(fisheye_pixel_cloud);
                }
            }
        }
        else {
            fisheye.SetSpotIdx(kOneSpot); /** spot idx **/
            for (int j = 0; j < fisheye.num_views; ++j) {
                fisheye.SetViewIdx(j); /** view idx **/
                std::tuple<RGBCloudPtr, RGBCloudPtr> fisheye_clouds = fisheye.FisheyeImageToSphere();
                RGBCloudPtr fisheye_polar_cloud;
                RGBCloudPtr fisheye_pixel_cloud;
                std::tie(fisheye_polar_cloud, fisheye_pixel_cloud) = fisheye_clouds;
                fisheye.SphereToPlane(fisheye_polar_cloud);
                fisheye.EdgeToPixel();
                fisheye.PixLookUp(fisheye_pixel_cloud);
            }
        }
    }

    cout << "----------------- LiDAR Processing ---------------------" << endl;
    lidar.SetExtrinsic(initial_params);
    /********* Create Dense Pcd for All Scenes *********/
    if (kCreateDensePcd) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    lidar.SetViewIdx(j);
                    lidar.CreateDensePcd();
                }
            }
        }
        else {
            lidar.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar.num_views; ++j) {
                lidar.SetViewIdx(j);
                lidar.CreateDensePcd();
            }
        }
    }
    if (kInitialIcp) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    if (j == (lidar.num_views - 1) / 2) {
                        continue;
                    }
                    lidar.SetViewIdx(j);
                    lidar.ICP();
                }
            }
        }
        else {
            lidar.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar.num_views; ++j) {
                if (j == (lidar.num_views - 1) / 2) {
                    continue;
                }
                lidar.SetViewIdx(j);
                lidar.ICP();
            }
        }
    }
    if (kCreateFullViewPcd) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                lidar.SetSpotIdx(i);
                lidar.CreateFullviewPcd(); /** generate full view pcds **/
            }
        }
        else {
            lidar.SetSpotIdx(kOneSpot);
            lidar.CreateFullviewPcd();
        }
        /** pcl viewer visualization **/
        string fullview_cloud_path;
        if (lidar.kDenseCloud) {
           fullview_cloud_path = lidar.poses_files_path_vec[0][0].fullview_dense_cloud_path;
        }
        else {
            fullview_cloud_path = lidar.poses_files_path_vec[0][0].fullview_sparse_cloud_path;
        }
        CloudPtr full_view(new CloudT);
        pcl::io::loadPCDFile(fullview_cloud_path, *full_view);
        pcl::visualization::CloudViewer viewer("Viewer");
        viewer.showCloud(full_view);
        while (!viewer.wasStopped()) {

        }
        cv::waitKey();
    }
    if (kLidarFlatProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                lidar.SetSpotIdx(i);
                lidar.SetViewIdx((lidar.num_views - 1) / 2);
                std::tuple<CloudPtr, CloudPtr> lidResult = lidar.LidarToSphere();
                CloudPtr lidCartesianCloud;
                CloudPtr lidPolarCloud;
                std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                lidar.SphereToPlane(lidPolarCloud, lidCartesianCloud);
            }
        }
        else {
            lidar.SetSpotIdx(kOneSpot);
            lidar.SetViewIdx((lidar.num_views - 1) / 2);
            std::tuple<CloudPtr, CloudPtr> lidResult = lidar.LidarToSphere();
            CloudPtr lidCartesianCloud;
            CloudPtr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidar.SphereToPlane(lidPolarCloud, lidCartesianCloud);
        }
    }
    else if (kLidarEdgeProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    lidar.SetViewIdx(j);
                    std::tuple<CloudPtr, CloudPtr> lidResult = lidar.LidarToSphere();
                    CloudPtr lidCartesianCloud;
                    CloudPtr lidPolarCloud;
                    std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                    lidar.SphereToPlane(lidPolarCloud, lidCartesianCloud);
                    lidar.EdgeToPixel();
                    lidar.PixLookUp(lidCartesianCloud);
                }
            }
        }
        else {
            lidar.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar.num_views; ++j) {
                lidar.SetViewIdx(j);
                std::tuple<CloudPtr, CloudPtr> lidResult = lidar.LidarToSphere();
                CloudPtr lidCartesianCloud;
                CloudPtr lidPolarCloud;
                std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                lidar.SphereToPlane(lidPolarCloud, lidCartesianCloud);
                lidar.EdgeToPixel();
                lidar.PixLookUp(lidCartesianCloud);
            }
        }
    }


    if (kCeresOptimization) {
        cout << "----------------- Ceres Optimization ---------------------" << endl;
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
        fisheye.SetSpotIdx(0);
        lidar.SetSpotIdx(0);
        for (int i = 0; i < fisheye.num_views; i++) {
            lidar.SetViewIdx(i);
            fisheye.SetViewIdx(i);
            lidar.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            fisheye.ReadEdge();
            fusionViz(fisheye, lidar, params_init, 88); /** 88 - invalid bandwidth to initialize the visualization **/
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
                params = ceresMultiScenes(fisheye, lidar, bandwidth, params, name, lb, ub, kDisabledBlock);
//                kDisabledBlock = 1;
//                params = ceresMultiScenes(FisheyeProcess, LidarProcess, bandwidth, params, name, lb, ub, kDisabledBlock);
            }
            else {
                int kDisabledBlock = 0;
                params = ceresMultiScenes(fisheye, lidar, bandwidth, params, name, lb, ub, kDisabledBlock);
            }
        }
    }

    if (kReconstruction) {
        cout << "----------------- RGB Reconstruction ---------------------" << endl;
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar.num_spots; ++i) {
                int target_view_idx = 1; /** degree 0 **/
                fisheye.SetSpotIdx(i);
                lidar.SetSpotIdx(i);
                lidar.SetViewIdx(target_view_idx);
                fisheye.SetViewIdx(target_view_idx);
                vector<double> calib_params = {0.0, 0.0, M_PI/2, +0.25, 0.0, -0.05, 1026.0, 1200.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
                fusionViz3D(fisheye, lidar, calib_params);
            }
        }
        else {
            fisheye.SetSpotIdx(kOneSpot);
            lidar.SetSpotIdx(kOneSpot);
            lidar.SetViewIdx(lidar.fullview_idx);
            fisheye.SetViewIdx(lidar.fullview_idx);
            vector<double> calib_params = {0.0, 0.0, M_PI/2, +0.25, 0.0, -0.05, 1026.0, 1200.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
            fusionViz3D(fisheye, lidar, calib_params);
        }
    }
    return 0;
}
