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
        imageProcess(string pkgPath);
        void readEdge();
        cv::Mat readOrgImage();
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

    public:
        int orgRows = 2048;
        int orgCols = 2448;
        int flatRows = int((double)110/90 * 1000) + 1;
        int flatCols = 4000;
        int kdeRows = 2048;
        int kdeCols = 2448;
        double radPerPix = (M_PI/2) / 1000;

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

        void setIntrinsic(vector<double> parameters) {
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

        /********* 所有Path定义中的/留前不留后 *********/
        /********* Data Path of Multi-Scenes *********/
        int scIdx = 0;
        void setSceneIdx(int scIdx) {
            this -> scIdx = scIdx;
        }

        static const int numScenes = 5;
        struct ScenesPath
        {
            ScenesPath(string pkgPath) {
                this -> sc1 = pkgPath + "/data/runYangIn";
                this -> sc2 = pkgPath + "/data/huiyuan2";
                this -> sc3 = pkgPath + "/data/12";
                this -> sc4 = pkgPath + "/data/conferenceF1";
                this -> sc5 = pkgPath + "/data/conferenceF2-P1";
            }
            string sc1;
            string sc2;
            string sc3;
            string sc4;
            string sc5;
        };

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
        vector<struct SceneFilePath> scenesFilePath;

        vector< vector<double> > edgeOrgTxtVec;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud;
};

#endif