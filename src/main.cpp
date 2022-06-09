// basic
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
// opencv
#include <opencv2/opencv.hpp>
// ros
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <ros/package.h>
// pcl
#include <pcl/common/io.h>
// heading
#include "imageProcess.h"
#include "lidarProcess.h"
#include "ceresMultiScenes.cpp"
#include "visualization.cpp"

using namespace std;
using namespace cv;

/********* Directory Path of ROS Package *********/
string getPkgPath() {
    std::string pkgPath = ros::package::getPath("data_process");
    return pkgPath;
}
string pkgPath = getPkgPath();

bool checkFolder(string FolderPath){
    if(opendir(FolderPath.c_str()) == NULL){             // The first parameter of 'opendir' is char *
        int ret = mkdir(FolderPath.c_str(), (S_IRWXU | S_IRWXG | S_IRWXO));       // 'mkdir' used for creating new directory
        if(ret == 0){
            cout << "Successfully create file folder!" << endl;
        }
    }
    return 1;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "mainNode");
    ros::NodeHandle nh;
    string pkgPath = getPkgPath();
    if(!checkFolder(pkgPath)){
        return -1;
    }

    vector<double> params_calib = {
        0.001, 0.0197457, 0.13,  0.00891695, 0.00937508, 0.14,
        606.16, -0.000558783, -2.70908E-09, -1.17573E-10,
        1.00014, -0.000177, 0.000129, 1023, 1201};

    cout << "----------------- Camera Processing ---------------------" << endl;
    imageProcess imageProcess(pkgPath);
    imageProcess.setIntrinsic(params_calib);
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult = imageProcess.fisheyeImageToSphere();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    vector< vector< vector<int> > > camtagsMap = imageProcess.sphereToPlane(camOrgPolarCloud);
    /********* Interrupt *********/
    vector< vector<int> > edgePixels = imageProcess.edgeToPixel();
    imageProcess.pixLookUp(edgePixels, camtagsMap, camOrgPixelCloud);

    cout << endl;
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    bool byIntensity = true;
    lidarProcess lidarProcess(pkgPath, byIntensity);
    lidarProcess.setExtrinsic(params_calib);

    lidarProcess.createDenseFile();

    std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidResult = lidarProcess.lidarToSphere();
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesianCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolarCloud;
    std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
    vector< vector< vector<int> > > lidTagsMap = lidarProcess.sphereToPlaneRNN(lidPolarCloud);
    vector< vector <int> > lidEdgePixels = lidarProcess.edgeToPixel();
    lidarProcess.pixLookUp(lidEdgePixels, lidTagsMap, lidCartesianCloud);

    cout << endl;
    cout << "----------------- Ceres Optimization ---------------------" << endl;
    /** a0, a1, a2, a3, a4; size of params = 13 **/
    // vector<char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a2", "a3", "a4"};
    // vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.12, 1023.0, 1201.0, 0.80541495, 594.42999235, 44.92838635, -54.82428857, 20.81519032};
    // vector<double> dev = {5e-2, 5e-2, M_PI/300, 1e-2, 1e-2, 5e-2, 2e+0, 2e+0, 2e+0, 2e+1, 8e+0, 4e+0, 2e+0};

    /** a0, a1, a3, a5; size of params = 12 **/
    vector<const char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a0", "a1", "a3", "a5"};
    vector<double> params_init = {0.0, 0.0, 0.1175, 0.0, 0.0, 0.16, 1023.0, 1201.0, 0.0, 609.93645006, -7.48070567, 3.22415532};
    vector<double> dev = {5e-2, 5e-2, 5e-3, 1e-2, 1e-2, 5e-2, 5e+0, 5e+0, 1e+1, 2e+1, 6e+0, 2e+0};

    /** a1, a3, a5; size of params = 11 **/
    // vector<char*> name = {"rx", "ry", "rz", "tx", "ty", "tz", "u0", "v0", "a1", "a3", "a5"};
    // vector<double> params_init = {0.0, 0.0, 0.1175, 0.0, 0.0, 0.16, 1023.0, 1201.0, 609.93645006, -7.48070567, 3.22415532};
    // vector<double> dev = {5e-2, 5e-2, M_PI/300, 1e-2, 1e-2, 5e-2, 5e+0, 5e+0, 2e+1, 6e+0, 2e+0};

    vector<double> params = params_init;
    // vector<double> params_init = {-0.03, 0.03, 0.1158, 0.0, 0.0, 0.21, 1023.0, 1201.0, 629.93645006, -8.48070567, 3.82415532};

    vector<double> lb(dev.size()), ub(dev.size());
    vector<double> bw = {32,24,16};
    Eigen::Matrix2d distortion;
    distortion << 1.000143, -0.000177, 0.000129, 1.000000;

    string lidEdgeTransTxtPath = lidarProcess.scenesFilePath[lidarProcess.scIdx].EdgeTransTxtPath;
    for (int i = 0; i < bw.size(); i++)
    {
        double bandwidth = bw[i];
        if (i == 0){
            /** generate initial projection image **/
            vector<vector<double>> lidProjection = lidarProcess.edgeVizTransform(params_init, distortion);
            fusionViz(imageProcess, lidEdgeTransTxtPath, lidProjection, 88);
            /** enable rx, ry, rz for the first round **/
            for (int j = 0; j < dev.size(); j++)
            {
                if (j < 3){
                    lb[j] = params_init[j] - dev[j];
                    ub[j] = params_init[j] + dev[j];
                }
                else{
                    lb[j] = params_init[j] - 1e-3;
                    ub[j] = params_init[j] + 1e-3;
                }
            }
        }
        else{
            /** enable all the params for other rounds **/
            for (int j = 3; j < dev.size(); j++)
            {
                lb[j] = params_init[j] - dev[j];
                ub[j] = params_init[j] + dev[j];
            }
        }
        cout << "Round " << i << endl;
        params = ceresMultiScenes(imageProcess, lidarProcess, bandwidth, distortion, params, name, lb, ub);

        cout << endl;
        cout << "----------------- Ceres Optimization ---------------------" << endl;
        int scVizIdx = 0; /** visualize the first scene **/
        lidarProcess.setSceneIdx(scVizIdx);
        imageProcess.setSceneIdx(scVizIdx);
        vector<vector<double>> lidProjection = lidarProcess.edgeVizTransform(params, distortion);
        lidarProcess.readEdge();
        imageProcess.readEdge();
        fusionViz(imageProcess, lidEdgeTransTxtPath, lidProjection, bandwidth);
    }
    return 0;
}
