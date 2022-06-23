#ifndef _IMAGEEDGE_H
#define _IMAGEEDGE_H
#include "CustomMsg.h"
#include "common.h"
#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sstream>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include <unordered_map>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <armadillo>

class imageProcess{
public:
    /** original data - images **/
    int orgRows = 2048;
    int orgCols = 2448;
    int flatRows = int((double)110/90 * 1000) + 1;
    int flatCols = 4000;
    double radPerPix = (M_PI/2) / 1000;

    /** coordinates of edge pixels in flat images **/
    typedef vector<vector<int>> EdgePixels;
    EdgePixels edge_pixels;
    vector<EdgePixels> edge_pixels_vec;

    /** coordinates of edge pixels in fisheye images **/
    typedef vector<vector<double>> EdgeFisheyePixels;
    EdgeFisheyePixels edge_fisheye_pixels;
    vector<EdgeFisheyePixels> edge_fisheye_pixels_vec;

    /** tagsmap container **/
    typedef vector<vector<vector<int>>> TagsMap;
    TagsMap tags_map;
    vector<TagsMap> tags_map_vec; /** container of tagsMaps of each scene **/

    /***** Intrinsic Parameters *****/
    struct Intrinsic {
        double a1 = 0;
        double a0 = 6.073762e+02;
        double a2 = -5.487830e-04;
        double a3 = -2.809080e-09;
        double a4 = -1.175734e-10;
        double c = 1.000143;
        double d = -0.000177;
        double e = 0.000129;
        double u0 = 1022.973079;
        double v0 = 1200.975472;
    } intrinsic;

    /***** Data of Multiple Scenes *****/
    int scene_idx = 0;
    int num_scenes = 5;
    vector<string> scenes_path_vec;

    /********* File Path of the Specific Scene *********/
    struct SceneFilePath
    {
        SceneFilePath(string ScenePath) {
            this -> OutputPath = ScenePath + "/outputs";
            this -> ResultPath = ScenePath + "/results";
            this -> HdrImgPath = ScenePath + "/images/grab_0.bmp";
            this -> EdgeImgPath = ScenePath + "/edges/camEdge.png";
            this -> FlatImgPath = this -> OutputPath + "/flatImage.bmp";
            this -> EdgeTxtPath = this -> OutputPath + "/camEdgePix.txt";
            this -> EdgeOrgTxtPath = this -> OutputPath + "/camPixOut.txt";
            this -> KdeTxtPath = this -> OutputPath + "/camKDE.txt";
            this -> FusionImgPath = this -> ResultPath + "/fusion.bmp";
        }
        string OutputPath;
        string ResultPath;
        string FusionImgPath;
        string HdrImgPath;
        string EdgeImgPath;
        string FlatImgPath;
        string EdgeTxtPath;
        string EdgeOrgTxtPath;
        string KdeTxtPath;
    };
    vector<struct SceneFilePath> scenes_files_path_vec;

public:
    imageProcess(string pkgPath);
    /***** Fisheye Pre-Processing *****/
    cv::Mat readOrgImage();
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fisheyeImageToSphere();
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fisheyeImageToSphere(cv::Mat image);
    void SphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar);
    void SphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar, double bandwidth);


    /***** Edge Related *****/
    void ReadEdge();
    void EdgeToPixel();
    void PixLookUp(pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud);
    std::vector<double> Kde(double bandwidth, double scale, bool polar);


    /***** Get and Set Methods *****/
    void SetIntrinsic(vector<double> parameters) {
        // polynomial params
        this->intrinsic.a0 = parameters[6];
        this->intrinsic.a2 = parameters[7];
        this->intrinsic.a3 = parameters[8];
        this->intrinsic.a4 = parameters[9];
        // expansion and distortion
        this->intrinsic.c = parameters[10];
        this->intrinsic.d = parameters[11];
        this->intrinsic.e = parameters[12];
        // center
        this->intrinsic.u0 = parameters[13];
        this->intrinsic.v0 = parameters[14];
    }
    void SetSceneIdx(int scene_idx) {
        this -> scene_idx = scene_idx;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud;
};

#endif