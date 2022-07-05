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

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointCloud<PointT>::Ptr CloudPtr;

const bool kFisheyeFlatProcess = false;
const bool kFisheyeEdgeProcess = false;
const bool kLidarFlatProcess = true;
const bool kLidarEdgeProcess = false;
const bool kCeresOptimization = false;
const bool kCreateDensePcd = true;
const bool kInitialIcp = false;
const bool kCreateFullViewPcd = true;
const bool kReconstruction = false;
const int kOneSpot = 3; /** -1 means run all the spots, other means run a specific spot **/

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

    /** data path **/
    string pkg_path = ros::package::getPath("calibration");
    string dataset_path = pkg_path + "/data/floor5";
    /** class object generation **/
    FisheyeProcess fisheye_process(pkg_path);
    LidarProcess lidar_process(pkg_path);
    /** data folder check **/
    for (int i = 0; i < lidar_process.num_spots; ++i) {
        string spot_path = dataset_path + "/spot" + to_string(i);
        CheckFolder(spot_path);
        for (int j = 0; j < lidar_process.num_views; ++j) {
            int view_degree = -lidar_process.view_angle_step + lidar_process.view_angle_step * j;
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

    cout << "----------------- Camera Processing ---------------------" << endl;
    fisheye_process.SetIntrinsic(initial_params);

    if (kFisheyeFlatProcess) {
        if (kOneSpot == -1) {
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
        else {
            fisheye_process.SetSpotIdx(kOneSpot); /** spot idx **/
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
        if (kOneSpot == -1) {
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
        else {
            fisheye_process.SetSpotIdx(kOneSpot); /** spot idx **/
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

    cout << "----------------- LiDAR Processing ---------------------" << endl;
    lidar_process.SetExtrinsic(initial_params);
    /********* Create Dense Pcd for All Scenes *********/
    if (kCreateDensePcd) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                lidar_process.SetSpotIdx(i);
                for (int j = 0; j < lidar_process.num_views; ++j) {
                    lidar_process.SetViewIdx(j);
                    lidar_process.CreateDensePcd();
                }
            }
        }
        else {
            lidar_process.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar_process.num_views; ++j) {
                lidar_process.SetViewIdx(j);
                lidar_process.CreateDensePcd();
            }
        }
    }
    if (kInitialIcp) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                lidar_process.SetSpotIdx(i);
                for (int j = 0; j < lidar_process.num_views; ++j) {
                    if (j == (lidar_process.num_views-1)/2) {
                        continue;
                    }
                    lidar_process.SetViewIdx(j);
                    lidar_process.ICP();
                }
            }
        }
        else {
            lidar_process.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar_process.num_views; ++j) {
                if (j == (lidar_process.num_views-1)/2) {
                    continue;
                }
                lidar_process.SetViewIdx(j);
                lidar_process.ICP();
            }
        }
    }
    if (kCreateFullViewPcd) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                lidar_process.SetSpotIdx(i);
                lidar_process.CreateFullviewPcd(); /** generate full view pcds **/
            }
        }
        else {
            lidar_process.SetSpotIdx(kOneSpot);
            lidar_process.CreateFullviewPcd();
        }
        /** pcl viewer visualization **/
//        CloudPtr full_view(new CloudT);
//        string fullview_cloud_path = lidar_process.poses_files_path_vec[3][0].fullview_dense_cloud_path;
//        pcl::io::loadPCDFile(fullview_cloud_path, *full_view);
//        pcl::visualization::CloudViewer viewer("Viewer");
//        viewer.showCloud(full_view);
//        while (!viewer.wasStopped()) {
//
//        }
//        cv::waitKey();
    }
    if (kLidarFlatProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                lidar_process.SetSpotIdx(i);
                lidar_process.SetViewIdx((lidar_process.num_views-1)/2);
                std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
                CloudPtr lidCartesianCloud;
                CloudPtr lidPolarCloud;
                std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
            }
        }
        else {
            lidar_process.SetSpotIdx(kOneSpot);
            lidar_process.SetViewIdx((lidar_process.num_views-1)/2);
            std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
            CloudPtr lidCartesianCloud;
            CloudPtr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
        }
    }
    else if (kLidarEdgeProcess) {
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                lidar_process.SetSpotIdx(i);
                for (int j = 0; j < lidar_process.num_views; ++j) {
                    lidar_process.SetViewIdx(j);
                    std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
                    CloudPtr lidCartesianCloud;
                    CloudPtr lidPolarCloud;
                    std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                    lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
                    lidar_process.EdgeToPixel();
                    lidar_process.PixLookUp(lidCartesianCloud);
                }
            }
        }
        else {
            lidar_process.SetSpotIdx(kOneSpot);
            for (int j = 0; j < lidar_process.num_views; ++j) {
                lidar_process.SetViewIdx(j);
                std::tuple<CloudPtr, CloudPtr> lidResult = lidar_process.LidarToSphere();
                CloudPtr lidCartesianCloud;
                CloudPtr lidPolarCloud;
                std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                lidar_process.SphereToPlane(lidPolarCloud, lidCartesianCloud);
                lidar_process.EdgeToPixel();
                lidar_process.PixLookUp(lidCartesianCloud);
            }
        }
    }

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
        fisheye_process.SetSpotIdx(0);
        lidar_process.SetSpotIdx(0);
        for (int i = 0; i < fisheye_process.num_views; i++) {
            lidar_process.SetViewIdx(i);
            fisheye_process.SetViewIdx(i);
            lidar_process.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            fisheye_process.ReadEdge();
            vector<vector<double>> edge_fisheye_projection = lidar_process.EdgeCloudProjectToFisheye(params_init);
            cout << "Edge Trans Txt Path:" << lidar_process.poses_files_path_vec[0][i].edge_fisheye_projection_path << endl;
            fusionViz(fisheye_process, lidar_process.poses_files_path_vec[0][i].edge_fisheye_projection_path,
                      edge_fisheye_projection, 88); /** 88 - invalid bandwidth to initialize the visualization **/
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
        if (kOneSpot == -1) {
            for (int i = 0; i < lidar_process.num_spots; ++i) {
                int target_view_idx = 1; /** degree 0 **/
                fisheye_process.SetSpotIdx(i);
                lidar_process.SetSpotIdx(i);
                lidar_process.SetViewIdx(target_view_idx);
                fisheye_process.SetViewIdx(target_view_idx);
                vector<double> calib_params = {0.0, 0.0, M_PI/2, +0.25, 0.0, -0.05, 1026.0, 1200.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
                fusionViz3D(fisheye_process, lidar_process, calib_params);
            }
        }
        else {
            int target_view_idx = 1; /** degree 0 **/
            fisheye_process.SetSpotIdx(kOneSpot);
            lidar_process.SetSpotIdx(kOneSpot);
            lidar_process.SetViewIdx(target_view_idx);
            fisheye_process.SetViewIdx(target_view_idx);
            vector<double> calib_params = {0.0, 0.0, M_PI/2, +0.25, 0.0, -0.05, 1026.0, 1200.0, 0.0, 616.7214056132, 1.0, -1.0, 1.0};
            fusionViz3D(fisheye_process, lidar_process, calib_params);
        }
    }
    return 0;
}
