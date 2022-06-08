#ifndef _LIDAREDGE_H
#define _LIDAREDGE_H
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
#include <string>
#include <vector>
#include <time.h>
#include <armadillo>

using namespace std;

class lidarProcess{
    public:
        lidarProcess(string dataPath, bool byIntensity);
        
        void readEdge();
        void setExtrinsic(vector<double> parameters);
        void bagToPcd(string bagFile);
        bool createDenseFile();
        void calculateMaxIncidence();

        void checkFolder(string outputDir);
        int readFileList(const std::string &folderPath, std::vector<std::string> &vFileList);

        std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidarToSphere();
        vector< vector< vector<int> > > sphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar);
        vector< vector<double> > edgeTransform();
        vector< vector<double> > edgeVizTransform(vector<double> params);
        vector< vector <int> > edgeToPixel();
        void pixLookUp(vector< vector <int> > edgePixels, vector< vector< vector<int> > > tagsMap, pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesian);
        void edgePixCheck(vector< vector<int> > edgePixels);
        vector<double> kdeFit(vector< vector <double> > edgePixels, int row_samples, int col_samples);
        void setImageAngle(float camThetaMin, float camThetaMax);
        // void lidFlatImageShift();

        struct extrinsic;
    public:
        bool byIntensity = true;
        int flatRows = int((double)110/90 * 1000) + 1;
        int flatCols = 4000;
        double radPerPix = (M_PI/2) / 1000;

        struct extrinsic {
            double rx = 0;
            double ry = 0;
            double rz = 0;
            double tx = 0;
            double ty = 0;
            double tz = 0;
        } extrinsic;

        pcl::PointCloud<pcl::PointXYZ>::Ptr lidEdgeOrg;

        string topicName = "/livox/lidar";
        string dataPath;
        string pcdsFolder;
        string outputFolder;
        string projectionFolder;
        string bagFile;
        string lidarDenseFile; // time integral of point clouds
        string lidPolar; // original point cloud in polar coordinates, without filtering
        string lidCartesian; // filtered point cloud in polar coordinates
        string flatLidarFile;
        string flatLidarShiftFile;
        string lidEdgeFile;
        string lidEdgePixFile;
        string edgeCheckFile;
        string lid3dOutFile;
        string lidTransFile;
};
#endif