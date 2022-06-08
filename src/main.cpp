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
// pcl
#include <pcl/common/io.h>
// heading
#include "imageProcess.h"
#include "lidarProcess.h"
#include "feature.h"
#include "ceresAutoDiff.cpp"
// #include "ceresOpt.cpp"

using namespace std;
using namespace cv;

string getDataPath(){
    char *dir = NULL;
    dir = (char *)get_current_dir_name();
    string dir_string = dir;
    // string dataPath = dir_string + "/data/lycheeHill/";
    string dataPath = "/home/isee/software/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/runYangIn/";
    return dataPath;
}

bool checkFolder(string FolderPath){
    if(opendir(FolderPath.c_str()) == NULL){             // The first parameter of 'opendir' is char *
        int ret = mkdir(FolderPath.c_str(), (S_IRWXU | S_IRWXG | S_IRWXO));       // 'mkdir' used for creating new directory
        if(ret == 0){
            cout << "Successfully create file folder!" << endl;
        }
    }
    return 1;
}

void fusionViz(imageProcess cam, string lidPath, vector< vector<double> > lidProjection, double bandwidth){
    cv::Mat image = cam.readOrgImage();
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat lidarRGB = cv::Mat::zeros(rows, cols, CV_8UC3);
    double pixPerRad = 1000 / (M_PI/2);

    ofstream outfile;

    outfile.open(lidPath, ios::out);

    for(int i = 0; i < lidProjection[0].size(); i++){
        double theta = lidProjection[0][i];
        double phi = lidProjection[1][i];
        // int u = (int)pixPerRad * theta;
        // int v = (int)pixPerRad * phi;
        int u = std::clamp(lidarRGB.rows - 1 - theta, (double)0.0, (double)(lidarRGB.rows-1));
        int v = std::clamp(phi, (double)0.0, (double)(lidarRGB.cols-1));;
        int b = 255;
        int g = 0;
        int r = 0;
        lidarRGB.at<Vec3b>(u, v)[0] = b;
        lidarRGB.at<Vec3b>(u, v)[1] = g;
        lidarRGB.at<Vec3b>(u, v)[2] = r;
        outfile << u << "," << v << endl;
    }

    outfile.close();

    cv::Mat imageShow = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::addWeighted(image, 1, lidarRGB, 0.8, 0, imageShow);
    
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult = 
        cam.fisheyeImageToSphere(imageShow);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    vector< vector< vector<int> > > camtagsMap = 
        cam.sphereToPlane(camOrgPolarCloud, bandwidth);

    char o_[64];
    sprintf(o_, "%s%f%s", "/home/isee/Desktop/output/fusionViz_", bandwidth, ".png");
    outfile << o_;
    cv::imwrite(o_, imageShow);
//    cv::imshow("show", imageShow);
//    waitKey();
}

int main(int argc, char** argv){
    ros::init(argc, argv, "mainNode");
    ros::NodeHandle nh;
    string dataPath = getDataPath();
    if(!checkFolder(dataPath)){
        return -1;
    }

    vector<double> params_calib = {
        0.001, 0.0197457, 0.13,  0.00891695, 0.00937508, 0.14,
        606.16, -0.000558783, -2.70908E-09, -1.17573E-10,
        1.00014, -0.000177, 0.000129, 1023, 1201};

    cout << "----------------- Camera Processing ---------------------" << endl;
    imageProcess imageProcess(dataPath, 20);

    // // set intrinsic params
    imageProcess.setIntrinsic(params_calib);

    // imageProcess.intrinsic.a0 = 606.16;
    // imageProcess.intrinsic.a2 = -0.000558783;
    // imageProcess.intrinsic.a3 = -2.70908E-09;
    // imageProcess.intrinsic.a4 = -1.17573E-10;
    // imageProcess.intrinsic.c = 1.00014;
    // imageProcess.intrinsic.d = -0.000177;
    // imageProcess.intrinsic.e = 0.000129;
    // imageProcess.intrinsic.u0 = 1023;
    // imageProcess.intrinsic.v0 = 1201;

    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult = imageProcess.fisheyeImageToSphere();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    vector< vector< vector<int> > > camtagsMap = imageProcess.sphereToPlane(camOrgPolarCloud);
    vector< vector<int> > edgePixels = imageProcess.edgeToPixel();
    imageProcess.pixLookUp(edgePixels, camtagsMap, camOrgPixelCloud);

    cout << endl;
    cout << "----------------- LiDAR Processing ---------------------" << endl;

    bool byIntensity = true;
    lidarProcess lidarProcess(dataPath, byIntensity);
    lidarProcess.setExtrinsic(params_calib);

    // // set extrinsic params
    // lidarProcess.extrinsic.rx = 0.001;
    // lidarProcess.extrinsic.ry = 0.0197457;
    // lidarProcess.extrinsic.rz = 0.13;
    // lidarProcess.extrinsic.tx = 0.00891695;
    // lidarProcess.extrinsic.ty = 0.00937508;
    // lidarProcess.extrinsic.tz = 0.14;

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
    // vector<double> params_init = {0.0, 0.0, 0.1175, 0.0, 0.0, 0.16, 1023.0, 1201.0, 609.93645006, -7.48070567, 3.22415532};
    vector<double> params_init = {0.0, 0.0, 0.115, 0.0, 0.0, 0.16, 1023.0, 1201.0, 609.93645006, -7.48070567, 3.22415532};
    vector<double> params = params_init;
    // vector<double> params_init = {-0.03, 0.03, 0.1158, 0.0, 0.0, 0.21, 1023.0, 1201.0, 629.93645006, -8.48070567, 3.82415532};
    vector<double> dev = {5e-2, 5e-2, M_PI/200, 1e-2, 1e-2, 5e-2, 2e+0, 2e+0, 2e+1, 4e+0, 1e+0};
    vector<double> lb(dev.size()), ub(dev.size());
    vector<double> bw = {32, 24,16,12,8,6,4};
    for (unsigned int i = 0; i < params_init.size(); i++)
    {
        lb[i] = params_init[i] - dev[i];
        ub[i] = params_init[i] + dev[i];
    }
    lidarProcess.readEdge();
    imageProcess.readEdge();
    for (int i = 0; i < bw.size(); i++)
    {
        double bandwidth = bw[i];
        if (i == 0){
            vector<vector<double>> lidProjection = lidarProcess.edgeVizTransform(params_init);
            fusionViz(imageProcess, lidarProcess.lidTransFile, lidProjection, 88);
            for (int j = 3; j < dev.size(); j++)
            {
                lb[j] = params_init[j] - 1e-3;
                ub[j] = params_init[j] + 1e-3;
            }
            
        }
        else{
            // lb[0] = params_init[0] - 1e-2;
            // ub[0] = params_init[0] + 1e-2;
            // lb[1] = params_init[1] - 1e-2;
            // ub[1] = params_init[1] + 1e-2;
            // lb[2] = params[2] - 1e-3;
            // ub[2] = params[2] + 1e-3;
            for (int j = 3; j < dev.size(); j++)
            {
                lb[j] = params_init[j] - dev[j];
                ub[j] = params_init[j] + dev[j];
            }
        }
        cout << "Round " << i << endl;
        params = ceresAutoDiff(imageProcess, lidarProcess, bandwidth, params, lb, ub);
        vector<vector<double>> lidProjection = lidarProcess.edgeVizTransform(params);
        fusionViz(imageProcess, lidarProcess.lidTransFile, lidProjection, bandwidth);
    }
    return 0;
}
