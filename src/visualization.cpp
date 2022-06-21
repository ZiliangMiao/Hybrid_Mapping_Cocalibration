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
#include <Eigen/Core>
#include <Eigen/Dense>
// heading
#include "imageProcess.h"
#include "lidarProcess.h"

using namespace std;
using namespace cv;
using namespace Eigen;

void fusionViz(imageProcess cam, string edge_proj_txt_path, vector< vector<double> > lidProjection, double bandwidth){
    cv::Mat image = cam.readOrgImage();
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat lidarRGB = cv::Mat::zeros(rows, cols, CV_8UC3);
    double pixPerRad = 1000 / (M_PI/2);

    /** write the edge points projected on fisheye to .txt file **/
    ofstream outfile;
    outfile.open(edge_proj_txt_path, ios::out);
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

void fusionViz3D(imageProcess cam, lidarProcess lid, vector<double> _p){

    Eigen::Matrix<double, 3, 1> eulerAngle(_p[0], _p[1], _p[2]);
    Eigen::Matrix<double, 3, 1> t{_p[3], _p[4], _p[5]};
    Eigen::Matrix<double, 2, 1> uv_0{_p[6], _p[7]};
    Eigen::Matrix<double, 6, 1> a_;
    switch (_p.size())
    {
        case 13:
            a_ << _p[8], _p[9], _p[10], _p[11], _p[12], double(0);
            break;
        case 12:
            a_ << _p[8], _p[9], double(0), _p[10], double(0), _p[11];
            break;
        default:
            a_ << double(0), _p[8], double(0), _p[9], double(0), _p[10];
            break;
    }

    double phi, theta;
    double inv_r, r;
    double res, val;

    // extrinsic transform
    Eigen::Matrix<double, 3, 3> R;
    Eigen::AngleAxisd xAngle(Eigen::AngleAxisd(eulerAngle(0), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitZ()));
    R = zAngle * yAngle * xAngle;

    Eigen::Matrix<double, 3, 1> p_;
    Eigen::Matrix<double, 3, 1> p_trans;
    Eigen::Matrix<double, 2, 1> S;
    Eigen::Matrix<double, 2, 1> p_uv;

    string lidDensePcdPath = lid.scenesFilePath[lid.scene_idx].LidDensePcdPath;
    string lidPro2DPath = lid.scenesFilePath[lid.scene_idx].LidPro2DPath;
    string lidPro3DPath = lid.scenesFilePath[lid.scene_idx].LidPro3DPath;
    string HdrImgPath = cam.scenesFilePath[cam.scIdx].HdrImgPath;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidRaw(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(lidDensePcdPath, *lidRaw);
    cv::Mat image = cv::imread(HdrImgPath, cv::IMREAD_UNCHANGED);
    int pixelThresh = 10;
    int rows = image.rows;
    int cols = image.cols;
    ofstream outfile2D;
    ofstream outfile3D;
    outfile2D.open(lidPro2DPath, ios::out);
    outfile3D.open(lidPro3DPath, ios::out);
    if (!outfile2D.is_open())
    {
        cout << "Open file failure" << endl;
    }
    if (!outfile3D.is_open())
    {
        cout << "Open file failure" << endl;
    }

    pcl::PointXYZRGB pt;
    vector<double> ptLoc(3, 0);
    vector<vector<vector<double>>> dict(rows, vector<vector<double>>(cols, ptLoc));
    cout << "---------------Save 2D and 3D txt file------------" << endl;
    for(int i = 0; i < lidRaw -> points.size(); i++){
        p_ << lidRaw -> points[i].x, lidRaw -> points[i].y, lidRaw -> points[i].z;
        p_trans = R * p_ + t;
        theta = acos(p_trans(2) / sqrt(pow(p_trans(0), 2) + pow(p_trans(1), 2) + pow(p_trans(2), 2)));
        inv_r = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        S = {inv_r * p_trans(0) / r, -inv_r * p_trans(1) / r};
        p_uv = S + uv_0;

        int u = int(p_uv(0));
        int v = int(p_uv(1));
        if(0 <= u && u < rows && 0 <=v && v < cols){

            dict[u][v][0] = p_(0);
            dict[u][v][1] = p_(1);
            dict[u][v][2] = p_(2);
            outfile2D << u << "\t" << v << endl;
            outfile3D << p_(0) << "\t" << p_(1) << "\t" << p_(2) << endl;
        }
        if(i % 100000 == 0){
            cout << i << " / " << lidRaw -> points.size() << " points written" << endl;
        }
    }
    outfile2D.close();
    outfile3D.close();
    cout << "---------------Coloring------------" << endl;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(image.at<cv::Vec3b>(i, j)[0] > pixelThresh || image.at<cv::Vec3b>(i, j)[1] > pixelThresh || image.at<cv::Vec3b>(i, j)[2] > pixelThresh){
                if(dict[i][j][0] != 0 && dict[i][j][1] != 0 && dict[i][j][2] != 0){
                    pt.x = dict[i][j][0];
                    pt.y = dict[i][j][1];
                    pt.z = dict[i][j][2];
                    pt.b = image.at<cv::Vec3b>(i, j)[0];
                    pt.g = image.at<cv::Vec3b>(i, j)[1];
                    pt.r = image.at<cv::Vec3b>(i, j)[2];
                    showCloud -> points.push_back(pt);
                }
            }
        }
    }
    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);

    while(!viewer.wasStopped()){

    }
    cv::waitKey();
}
