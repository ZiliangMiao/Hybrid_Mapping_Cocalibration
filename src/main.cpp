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

using namespace std;
using namespace cv;

const bool fisheyeFlatProcess = false;
const bool fisheyeEdgeProcess = false;
const bool lidarFlatProcess = true;
const bool lidarEdgeProcess = false;
const bool ceresOpt = false;
const bool viz3D = false;
const bool denseFile = false;

/********* Directory Path of ROS Package *********/
string getPkgPath() {
    std::string pkgPath = ros::package::getPath("data_process");
    return pkgPath;
}
string pkgPath = getPkgPath();

bool checkFolder(string FolderPath){
    if(opendir(FolderPath.c_str()) == NULL){                 // The first parameter of 'opendir' is char *
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

    /** fisheye intrinsics calibrated by chessboard **/
    vector<double> params_calib = {
        0.001, 0.0197457, 0.13,  0.00891695, 0.00937508, 0.14,
        606.16, -0.000558783, -2.70908E-09, -1.17573E-10,
        1.00014, -0.000177, 0.000129, 1023, 1201
    };

    cout << "----------------- Camera Processing ---------------------" << endl;
    imageProcess imageProcess(pkgPath);
    imageProcess.setIntrinsic(params_calib);

    if (fisheyeFlatProcess) {
        for (int idx = 0; idx < imageProcess.numScenes; idx++) {
            imageProcess.setSceneIdx(idx);
            std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult = imageProcess.fisheyeImageToSphere();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
            std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
            vector< vector< vector<int> > > camtagsMap = imageProcess.sphereToPlane(camOrgPolarCloud);
        }
    }
    else if (fisheyeEdgeProcess) {
        for (int idx = 0; idx < imageProcess.numScenes; idx++) {
            imageProcess.setSceneIdx(idx);
            std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult = imageProcess.fisheyeImageToSphere();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
            std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
            vector< vector< vector<int> > > camtagsMap = imageProcess.sphereToPlane(camOrgPolarCloud);
            vector< vector<int> > edgePixels = imageProcess.edgeToPixel();
            imageProcess.pixLookUp(edgePixels, camtagsMap, camOrgPixelCloud);
        }
    }

    cout << endl;
    cout << "----------------- LiDAR Processing ---------------------" << endl;
    bool byIntensity = true;
    lidarProcess lidarProcess(pkgPath, byIntensity);
    lidarProcess.setExtrinsic(params_calib);
    ROS_ASSERT_MSG(lidarProcess.num_scenes == imageProcess.numScenes, "num_scenes in imageProcess and lidarProcess is not equal!");
    /********* Create Dense Pcd for All Scenes *********/
    if (denseFile) {
        for (int idx = 0; idx < lidarProcess.num_scenes; idx++) {
            lidarProcess.setSceneIdx(idx);
            lidarProcess.createDenseFile();
        }
    }
    if (lidarFlatProcess) {
        for (int idx = 0; idx < lidarProcess.num_scenes; idx++) {
            lidarProcess.setSceneIdx(idx);
            std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidResult = lidarProcess.LidarToSphere();
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesianCloud;
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidarProcess.SphereToPlaneRNN(lidPolarCloud, lidCartesianCloud);
        }
    }
    else if (lidarEdgeProcess) {
        for (int idx = 0; idx < lidarProcess.num_scenes; idx++) {
            lidarProcess.setSceneIdx(idx);
            std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidResult = lidarProcess.LidarToSphere();
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesianCloud;
            pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolarCloud;
            std::tie(lidPolarCloud, lidCartesianCloud) = lidResult;
            lidarProcess.SphereToPlaneRNN(lidPolarCloud, lidCartesianCloud);
            lidarProcess.EdgeToPixel();
            lidarProcess.PixLookUp(lidCartesianCloud);
        }
    }
   
    cout << endl;
    cout << "----------------- Ceres Optimization ---------------------" << endl;
    if (ceresOpt) {
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
        vector<double> bw = {32, 24, 16, 8, 4, 2};

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
        for (int idx = 0; idx < imageProcess.numScenes; idx++) {
            lidarProcess.setSceneIdx(idx);
            imageProcess.setSceneIdx(idx);
            lidarProcess.ReadEdge(); /** this is the only time when ReadEdge method appears **/
            imageProcess.readEdge();
            vector<vector<double>> edge_fisheye_projection = lidarProcess.EdgeCloudProjectToFisheye(params_init);
            fusionViz(imageProcess, lidarProcess.scenesFilePath[idx].EdgeTransTxtPath, edge_fisheye_projection, 88); /** 88 - invalid bandwidth to initialize the visualization **/
        }

        for (int i = 0; i < bw.size(); i++) {
            double bandwidth = bw[i];
            cout << "Round " << i << endl;
            /**
             * setConstant = 0 -> enable all the params
             * setConstant = 1 -> enable intrinsics only
             * setConstant = 2 -> enable extrinsics only
             * **/
            if (i == 0) {
                int setConstant = 2;
                params = ceresMultiScenes(imageProcess, lidarProcess, bandwidth, params, name, lb, ub, setConstant);
//                setConstant = 1;
//                params = ceresMultiScenes(imageProcess, lidarProcess, bandwidth, params, name, lb, ub, setConstant);
            }
            else {
                int setConstant = 0;
                params = ceresMultiScenes(imageProcess, lidarProcess, bandwidth, params, name, lb, ub, setConstant);
            }
        }
    }

    if (viz3D) {
        lidarProcess.setSceneIdx(1);
        imageProcess.setSceneIdx(1);
        vector<double> test_params = {-0.0131396, 0.0179037, 0.116701, 0.01, 0.00374594, 0.118988, 1021.0, 1199.0, 2.79921, 606.544, 48.3143, -54.8969, 17.7703};
        fusionViz3D(imageProcess, lidarProcess, test_params);
    }
    return 0;
}
