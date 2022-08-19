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
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/principal_curvatures.h>
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
#include <spline.h>
#include <define.h>

using namespace std;
using namespace tk;

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
    vector<vector<string>> poses_folder_path_vec;

    /** original data - images **/
    int kFisheyeRows = 2048;
    int kFisheyeCols = 2448;
    const int kFlatRows = int((double)110 / 90 * 1000) + 1;
    const int kFlatCols = 4000;
    const float kRadPerPix = (M_PI * 2) / 4000;

    /** coordinates of edge pixels in flat images **/
    typedef vector<vector<int>> EdgePixels;
    vector<vector<EdgePixels>> edge_pixels_vec;

    /** coordinates of edge pixels in fisheye images **/
    // typedef vector<vector<double>> EdgeFisheyePixels;
    vector<vector<EdgeCloud::Ptr>> edge_fisheye_pixels_vec;

    /** tagsmap container **/
    typedef struct Tags {
        vector<int> pts_indices = {};
    }Tags; /** "Tags" here is a struct type, equals to "struct Tags", LidarProcess::Tags **/
    typedef vector<vector<Tags>> TagsMap;
    vector<vector<TagsMap>> tags_map_vec; /** container of tagsMaps of each pose **/

    /***** Intrinsic Parameters *****/
    Int_D int_;

    /********* File Path of the Specific Pose *********/
    struct PoseFilePath {
        PoseFilePath () = default;
        PoseFilePath (const string &pose_folder_path) {
            this->output_folder_path = pose_folder_path + "/outputs/fisheye_outputs";
            this->fusion_result_folder_path = pose_folder_path + "/results";
            this->fisheye_hdr_img_path = pose_folder_path + "/images/grab_0.bmp";
            this->edge_img_path = pose_folder_path + "/edges/camEdge.png";
            this->flat_img_path = this -> output_folder_path + "/flatImage.bmp";
            this->edge_fisheye_pixels_path = this -> output_folder_path + "/camPixOut.txt";
            this->kde_samples_path = this -> output_folder_path + "/camKDE.txt";
            this->fusion_img_path = this -> fusion_result_folder_path + "/fusion.bmp";
        }
        string output_folder_path;
        string fusion_result_folder_path;
        string fusion_img_path;
        string fisheye_hdr_img_path;
        string edge_img_path;
        string flat_img_path;
        string edge_fisheye_pixels_path;
        string kde_samples_path;
    };
    vector<vector<struct PoseFilePath>> poses_files_path_vec;

    /** Degree Map **/
    std::map<int, int> degree_map;

public:
    FisheyeProcess();
    /** Fisheye Pre-Processing **/
    cv::Mat ReadFisheyeImage(string fisheye_hdr_img_path);
    void FisheyeImageToSphere(CloudRGB::Ptr &pixel_cloud, CloudRGB::Ptr &polar_cloud);
    void FisheyeImageToSphere(CloudRGB::Ptr &pixel_cloud, CloudRGB::Ptr &polar_cloud, cv::Mat &image, Int_D intrinsic);
    void SphereToPlane(CloudRGB::Ptr &fisheye_polar_cloud);
    void SphereToPlane(CloudRGB::Ptr &fisheye_polar_cloud, double bandwidth);

    /** Edge Related **/
    void ReadEdge();
    void EdgeToPixel();
    void PixLookUp(CloudRGB::Ptr &fisheye_pixel_cloud);
    std::vector<double> Kde(double bandwidth, double scale);
    void EdgeExtraction();

    void SetSpotIdx(int spot_idx) {
        this->spot_idx = spot_idx;
    }

    void SetViewIdx(int view_idx) {
        this->view_idx = view_idx;
    }

    static bool cmp(const std::vector<double>& a, const std::vector<double>& b) {
        return a.back() < b.back();
    }

    tk::spline InverseSpline(Int_D intrinsic) {
        MatD(5, 1) a_ = intrinsic.head(7).tail(5);
        const int theta_ub = 180;
        const int extend = 2;

        // extend the range to get a stable cubic spline
        std::vector<std::vector<double>> r_theta_pts(theta_ub + extend * 2, std::vector<double>(2));

        for (double theta = 0; theta < theta_ub + extend * 2; ++theta) {
            double theta_rad = (theta - extend) * M_PI / 180;
            r_theta_pts[theta][0] = theta_rad;
            r_theta_pts[theta][1] = a_(0) + a_(1) * theta_rad + a_(2) * pow(theta_rad, 2) + a_(3) * pow(theta_rad, 3) + a_(4) * pow(theta_rad, 4);
        }
        sort(r_theta_pts.begin(), r_theta_pts.end(), cmp);
        std::vector<double> input_radius, input_theta;
        for (int i = 0; i < r_theta_pts.size(); i++) {
            input_theta.push_back(r_theta_pts[i][0]);
            input_radius.push_back(r_theta_pts[i][1]);
        }
        // default cubic spline (C^2) with natural boundary conditions (f''=0)
        tk::spline spline(input_radius, input_theta);			// X needs to be strictly increasing
        return spline;
    }
};
