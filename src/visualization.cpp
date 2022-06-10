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

using namespace std;
using namespace cv;

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
        int b = 0;
        int g = 0;
        int r = 255;
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
}