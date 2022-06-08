#ifndef _EDGE_H
#define _EDGE_H
#include "CustomMsg.h"
#include "common.h"
#include "ceres/ceres.h"
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
#include <dirent.h>
#include <iomanip>
using namespace std;
class edge{
    public:
        void checkFolder(string outputDir);
        void extractionAndProjection();
        void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map);
        void calcLine(const std::vector<Plane> &plane_list, const double voxel_size, const Eigen::Vector3d origin,std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list, pcl::PointCloud<pcl::PointXYZI>::Ptr lidarEdgeCloud);
        pcl::PointCloud<pcl::PointXYZI>::Ptr lidarEdgeExtraction(const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,const float ransac_dis_thre, const int plane_size_threshold,pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d);
        pcl::PointCloud<pcl::PointXYZI>::Ptr lidarEdgeProjection();
        pcl::PointCloud<pcl::PointXYZI>::Ptr projectToPlane();
        
    public:
        float voxel_size_ = 1.0;
        float ransac_dis_threshold_ = 0.02;
        float plane_size_threshold_ = 60;
        float theta_min_ = cos(DEG2RAD(45));
        float theta_max_ = cos(DEG2RAD(135));
        float min_line_dis_threshold_ = 0.03;
        float max_line_dis_threshold_ = 0.06;
        int line_number_ = 0;
        int plane_max_size_ = 8;
        int lidarDenseCloudSize;
        vector<int> plane_line_number_;
        string lidarAndEdgeFile;
        string lidarEdgeFile;
        string lidarEdgeSphereFile;

        vector<VoxelGrid> voxel_list;
        unordered_map<VOXEL_LOC, Voxel *>voxel_map;
        pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
}