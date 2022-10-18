/** basic **/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <time.h>
/** opencv **/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/** pcl **/
#include <pcl/common/common.h>
#include <Eigen/Core>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
/** mlpack **/
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

/** headings **/
#include <define.h>

using namespace std;

/** typedef **/
typedef pcl::PointXYZI PointI;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointI> CloudI;
typedef pcl::PointCloud<PointRGB> CloudRGB;
typedef pcl::PointCloud<pcl::PointXYZ> EdgeCloud;

class FisheyeProcess{
public:
    /** essential params **/
    string kPkgPath = ros::package::getPath("calibration");
    string dataset_name;
    string kDatasetPath;
    int spot_idx = 0;
    int view_idx = 0;
    int num_spots;
    int num_views;
    int view_angle_init;
    int view_angle_step;
    int fullview_idx;
    
    /** Omnidirectional image settings **/
    Pair kImageSize = {2048, 2448};
    Pair kEffectiveRadius = {325, 1125};
    int kExcludeRadius = 175;

    /** coordinates of edge pixels in fisheye images **/
    vector<vector<EdgeCloud::Ptr>> edge_cloud_vec;

    /***** Intrinsic Parameters *****/
    Int_D int_;

    /********* File Path of the Specific Pose *********/
    struct PoseFilePath {
        PoseFilePath () = default;
        PoseFilePath (const string &pose_folder_path) {
            this->output_folder_path = pose_folder_path + "/outputs/fisheye_outputs";
            this->fusion_folder_path = pose_folder_path + "/results";
            this->hdr_img_path = pose_folder_path + "/images/grab_0.bmp";
            this->edge_img_path = pose_folder_path + "/edges/cam_edge.png";
            this->flat_img_path = output_folder_path + "/flat_image.bmp";
            this->edge_cloud_path = output_folder_path + "/edge_image.pcd";
            this->kde_samples_path = output_folder_path + "/kde_image.txt";
            this->fusion_img_path = fusion_folder_path + "/fusion.bmp";
        }
        string output_folder_path;
        string fusion_folder_path;
        string fusion_img_path;
        string hdr_img_path;
        string edge_img_path;
        string flat_img_path;
        string edge_cloud_path;
        string kde_samples_path;
    };

    vector<vector<string>> folder_path_vec;
    vector<vector<struct PoseFilePath>> file_path_vec;

    /** Degree Map **/
    std::map<int, int> degree_map;

public:
    FisheyeProcess();

    cv::Mat LoadImage();
    void ReadEdge();
    void EdgeToPixel();
    std::vector<double> Kde(double bandwidth, double scale);
    void EdgeExtraction();

    void SetSpotIdx(int spot_idx) {
        this->spot_idx = spot_idx;
    }

    void SetViewIdx(int view_idx) {
        this->view_idx = view_idx;
    }

};
