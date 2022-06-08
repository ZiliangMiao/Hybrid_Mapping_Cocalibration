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
        // construction method of the class
        imageProcess(string dataPath, int thresh);
        void readEdge();
        cv::Mat readOrgImage();
        void setIntrinsic(vector<double> parameters);
        // core methods: tranform the fisheye image to a sphere, transform the sphere image to a flat image
        std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fisheyeImageToSphere();
        std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fisheyeImageToSphere(cv::Mat image);
        vector< vector< vector<int> > > sphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar);
        vector< vector< vector<int> > > sphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar, double bandwidth);
        vector< vector<double> > edgeTransform();
        vector< vector<int> > edgeToPixel();
        void pixLookUp(vector< vector <int> > edgePixels, vector< vector< vector<int> > > tagsMap, pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud);
        // void meanShiftCluster();
        std::vector<double> kdeBlur(double bandwidth, double scale, bool polar);
        // methods about the edges of image
        // cv::Mat edgeDetect();
        // void extractAndProject();
        // pcl::PointCloud<pcl::PointXYZ>::Ptr edgeProjection(cv::Mat img);


        struct intrinsic;
    public:
        int orgRows = 2048;
        int orgCols = 2448;
        // caution: the int division need a type coercion to float, otherwise the result would be zero
        int flatRows = int((double)110/90 * 1000) + 1;
        int flatCols = 4000;
        int kdeRows = 2048;
        int kdeCols = 2448;
        double radPerPix = (M_PI/2) / 1000;

        struct intrinsic {
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

        vector< vector<double> > camEdgeOrg;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud;

        string outputFolder;
        string dataPath;
        string imageFile;
        string imageEdgeFile;
        string imageGrayFile;
        string flatImageFile;
        string imageMirrorFile;
        string camEdgePixFile;
        string camPixOutFile;
        string camKdeFile;
        string camTransFile;
};

#endif