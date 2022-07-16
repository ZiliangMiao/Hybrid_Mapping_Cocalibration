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
const bool kFisheyeFlatProcess = true;
const bool kFisheyeEdgeProcess = false;

const bool kCreateDensePcd = true;
const bool kInitialIcp = true;
const bool kCreateFullViewPcd = true;

const bool kLidarFlatProcess = true;
const bool kLidarEdgeProcess = false;

const bool kCeresOptimization = true;
const bool kReconstruction = false;
const bool kSpotRegistration = false;
const bool kGlobalColoredRecon = true;
const int kOneSpot = -1; /** -1 means run all the spots, other means run a specific spot **/

int main(int argc, char** argv) {
    /** ros initialization **/
    ros::init(argc, argv, "calibration");
    ros::NodeHandle nh;

    vector<double> init_proj_params = {
            M_PI, 0.00, -M_PI/2, /** Rx, Ry, Rz **/
            0.27, 0.00, 0.03, /** tx ty tz **/
            -606.16, 0.000558783, 2.70908E-09, 1.17573E-10, /** a0, a2, a3, a4 **/
            1, 0, 0, /** c, d, e **/
            1023, 1201 /** u0, v0 **/
    }; /** fisheye intrinsics here are calibrated by chessboard **/

    std::vector<double> params_calib;

    /** class object generation **/
    FisheyeProcess fisheye;
    fisheye.SetIntrinsic(init_proj_params);
    LidarProcess lidar;
    lidar.SetExtrinsic(init_proj_params);
    /** data folder check **/
    for (int i = 0; i < lidar.num_spots; ++i) {
        string spot_path = lidar.kDatasetPath + "/spot" + to_string(i);
        CheckFolder(spot_path);
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
    cout << "----------------- Fisheye Processing ---------------------" << endl;

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
    
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    /********* Create Dense Pcd for All Scenes *********/

    if (kCreateDensePcd) {
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
    
    if (kInitialIcp) {
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                for (int j = 0; j < lidar.num_views; ++j) {
                    if (j == lidar.fullview_idx) {
                        continue;
                    }
                    lidar.SetViewIdx(j);
                    lidar.ICP();
                }
            }
        }
        //            vector <Eigen::Matrix4f> local_trans_vec(lidar.num_views);
//            vector <Eigen::Matrix4f> global_trans_vec(lidar.num_views);
//            Eigen::Matrix4f local_trans;
//            Eigen::Matrix4f global_trans;
//
//            /** local pair transformation matrix **/
//            for (int j = 0; j < lidar.num_views; ++j) {
//                int view_idx_tgt;
//                if (j == lidar.fullview_idx) {
//                    local_trans_vec[j] = Eigen::Matrix4f::Identity();
//                    continue;
//                }
//                else if (j < lidar.fullview_idx) {
//                    view_idx_tgt = j + 1;
//                    lidar.SetViewIdx(j);
//                    local_trans = lidar.ICP2(view_idx_tgt);
//                    local_trans_vec[j] = local_trans;
//                }
//                else if (j > lidar.fullview_idx) {
//                    view_idx_tgt = j - 1;
//                    lidar.SetViewIdx(j);
//                    local_trans = lidar.ICP2(view_idx_tgt);
//                    local_trans_vec[j] = local_trans;
//                }
//            }
//
//            /** global transformation matrix to target view **/
//            for (int j = 0; j < lidar.num_views; ++j) {
//                if (j == lidar.fullview_idx) {
//                    global_trans = Eigen::Matrix4f::Identity();
//                    global_trans_vec[j] = global_trans;
//                    std::ofstream mat_out;
//                    mat_out.open(lidar.poses_files_path_vec[lidar.spot_idx][j].pose_trans_mat_path);
//                    mat_out << global_trans << endl;
//                    mat_out.close();
//                    cout << "ICP Transformation Matrix" << " View " << j << ":\n" << global_trans << endl;
//                    continue;
//                }
//                else if (j < lidar.fullview_idx) {
//                    global_trans = Eigen::Matrix4f::Identity();
//                    for (int k = j; k <= lidar.fullview_idx; ++k) {
//                        /** note: take care of the matrix multiplication,  **/
//                        global_trans = local_trans_vec[j] * global_trans;
//                    }
//                    global_trans_vec[j] = global_trans;
//                    std::ofstream mat_out;
//                    mat_out.open(lidar.poses_files_path_vec[lidar.spot_idx][j].pose_trans_mat_path);
//                    mat_out << global_trans << endl;
//                    mat_out.close();
//                    cout << "ICP Transformation Matrix" << " View " << j << ":\n" << global_trans << endl;
//                }
//                else if (j > lidar.fullview_idx) {
//                    global_trans = Eigen::Matrix4f::Identity();
//                    for (int k = j; k >= lidar.fullview_idx; --k) {
//                        global_trans = local_trans_vec[j] * global_trans;
//                    }
//                    global_trans_vec[j] = global_trans;
//                    std::ofstream mat_out;
//                    mat_out.open(lidar.poses_files_path_vec[lidar.spot_idx][j].pose_trans_mat_path);
//                    mat_out << global_trans << endl;
//                    mat_out.close();
//                    cout << "ICP Transformation Matrix" << " View " << j << ":\n" << global_trans << endl;
//                }
//            }
    }
        
    if (kCreateFullViewPcd) {
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                lidar.CreateFullviewPcd(); /** generate full view pcds **/
            }
        }
        /** pcl viewer visualization **/
        // string fullview_cloud_path;
        // fullview_cloud_path = lidar.poses_files_path_vec[0][0].fullview_sparse_cloud_path;
        // CloudPtr full_view(new CloudT);
        // pcl::io::loadPCDFile(fullview_cloud_path, *full_view);
        // pcl::visualization::CloudViewer viewer("Viewer");
        // viewer.showCloud(full_view);
        // while (!viewer.wasStopped()) {

        // }
        // cv::waitKey();
    }
    
    if (kLidarFlatProcess) {
        for (int i = 0; i < lidar.num_spots; ++i) {
            if (kOneSpot == -1 || kOneSpot == i) {
                lidar.SetSpotIdx(i);
                lidar.SetViewIdx(lidar.fullview_idx);
                std::tuple<CloudPtr, CloudPtr> lidResult = lidar.LidarToSphere();
                CloudPtr lidCartesianCloud;
                CloudPtr lidPolarCloud;
                std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
                lidar.SphereToPlane(lidPolarCloud, lidCartesianCloud);
                lidar.EdgeExtraction();
                lidar.EdgeToPixel();
                lidar.PixLookUp(lidCartesianCloud);
            }
        }
    }

    if (kCeresOptimization) {
        cout << "----------------- Ceres Optimization ---------------------" << endl;
        /** the two sensors are parallel on y axis **/
        std::vector<const char*> name = {
                "rx", "ry", "rz",
                "tx", "ty", "tz",
                "u0", "v0",
                "a0", "a1", "a2", "a3", "a4",
                "c", "d", "e"};
        std::vector<double> params_init = {
                M_PI, 0.00, -M_PI/2, /** Rx Ry Rz **/
                0.27, 0.00, 0.03, /** tx ty tz **/
                1023.0, 1201.0, /** u0 v0 **/
                616.7214056132 * M_PI, -616.7214056132, 0.0, 0.0, 0.0,
                1, 0, 0 /** c, d, e **/
        }; /** initial parameters **/
        std::vector<double> dev = {
                1e-1, 1e-1, 1e-1,
                3e-2, 3e-2, 5e-2,
                3e+0, 3e+0,
                160e+0, 80e+0, 40e+0, 20+0, 10e+0,
                1e-2, 1e-2, 1e-2
                };
        std::vector<double> lb(dev.size()), ub(dev.size());
        std::vector<double> bw = {32, 24, 16, 8, 4, 2};

        for (int i = 0; i < dev.size(); ++i) {
            ub[i] = params_init[i] + dev[i];
            lb[i] = params_init[i] - dev[i];
            // if (i == dev.size() - 1 || i == dev.size() - 3){
            //     ub[i] = params_init[i];
            //     lb[i] = params_init[i] - dev[i];
            // }
            // if (i == dev.size() - 2){
            //     ub[i] = params_init[i] + dev[i];
            //     lb[i] = params_init[i];
            // }
        }

        /********* Initial Visualization *********/
        std::vector<int> spot_vec{2, 3, 4};
        fisheye.SetViewIdx(fisheye.fullview_idx);
        lidar.SetViewIdx(lidar.fullview_idx);
        for (int &spot_idx : spot_vec)
        {
            fisheye.SetSpotIdx(spot_idx);
            lidar.SetSpotIdx(spot_idx);
            lidar.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            fisheye.ReadEdge();
            Visualization2D(fisheye, lidar, params_init, 88); /** 88 - invalid bandwidth to initialize the visualization **/
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
                int kDisabledBlock = 0;
                // params_calib = ceresMultiScenes(fisheye, lidar, bandwidth, params_init, name, lb, ub, kDisabledBlock);
                params_calib = ceresQuaternion(fisheye, lidar, bandwidth, spot_vec, params_init, name, lb, ub, kDisabledBlock);
            }
            else {
                int kDisabledBlock = 0;
                // params_calib = ceresMultiScenes(fisheye, lidar, bandwidth, params_calib, name, lb, ub, kDisabledBlock);
                params_calib = ceresQuaternion(fisheye, lidar, bandwidth, spot_vec, params_calib, name, lb, ub, kDisabledBlock);
            }
        }
    }

    if (kReconstruction) {
        cout << "----------------- RGB Reconstruction ---------------------" << endl;
        // params_calib = {
        //     0.00513968, 3.13105, 1.56417, /** Rx Ry Rz **/
        //     0.250552, 0.0264601, 0.0765269, /** tx ty tz **/
        //     1020.0, 1198.0,
        //     1888.37, -536.802, -19.6401, -17.8592, 6.34771,
        //     0.996981, -0.00880807, 0.00981348
        // };
        // Final   rx: 0.00513968 ry: 3.13105 rz: 1.56417 tx: 0.250552 ty: 0.0264601 tz: 0.0765269 u0: 1020 v0: 1198 a0: 1888.37 a1: -536.802 a2: -19.6401 a3: -17.8592 a4: 6.34771 c: 0.996981 d: -0.00880807 e: 0.00981348 
        // params_calib = {
        //     0.00713431, -3.13089, 1.5521, /** Rx Ry Rz **/
        //     0.296866, -0.0272627, 0.0571168, /** tx ty tz **/
        //     1026.0, 1201.79, /** u0, v0 **/
        //     1879.81, -550.701, -11.7394, -11.7073, 3.82408,
        //     1, 0, 0 /** c, d, e **/
        // };
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

    if (kSpotRegistration) {
        cout << "----------------- Spot Registration ---------------------" << endl;
        lidar.SpotRegistration();
    }

    if (kGlobalColoredRecon) {
        cout << "----------------- Global Reconstruction ---------------------" << endl;
        lidar.GlobalColoredRecon();
    }

    return 0;
}
