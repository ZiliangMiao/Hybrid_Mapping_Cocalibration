#include "CustomMsg.h"
#include "common.h"
// #include "ceres/ceres.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
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
#include <dirent.h>
#include "feature.h"
using namespace std;
using namespace cv;

#define HEIGHT 2048
#define WIDTH 2048
#define HALF_HEIGHT HEIGHT/2
#define HALF_WIDTH WIDTH/2

#define PI 3.14159

feature::feature(string dataPath, bool byIntensity){
    this -> byIntensity = byIntensity;
    this -> dataPath = dataPath;

    // this -> imageEdgeSphereFile = dataPath + "imageEdgeSphere.pcd";
    // this -> lidarFile = dataPath + "lidarEdgeSphere.pcd";
    // this -> lidarDenseFile = this -> dataPath + "lidarDense.pcd";
    // this -> lidarMarkFile = dataPath + "lidarMark.txt";
    // this -> imageMarkFile = dataPath + "imageMark.txt";
    // this -> rawImageFile = dataPath + "rawImage/grab12.bmp";
    // this -> imagePressedFile = dataPath + "imagePressed/grab12.bmp";
    this -> outputFolder = this -> dataPath + "outputs/";
    if(byIntensity){
        this -> projectionFolder = this -> outputFolder + "byIntensity/";
    }
    else{
        this -> projectionFolder = this -> outputFolder + "byDepth/";
    }
    this -> lidarFlatFile = this -> projectionFolder + "flatLidarImageRNNShift.bmp";
    this -> imageFlatFile = outputFolder + "flatImage.bmp";

    // this -> imageCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // this -> lidarCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // lidarDenseCloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::io::loadPCDFile(this->imageEdgeSphereFile, *imageCloud);
    // pcl::io::loadPCDFile(this->lidarFile, *lidarCloud);
    
}

void feature::showCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    int imagePointCount = cloud1 -> points.size();
    int lidarPointCount = cloud2 -> points.size();
    pcl::PointXYZRGB pt;
    for(int i = 0; i < imagePointCount; i++){
        pt.x = cloud1 -> points[i].x;
        pt.y = cloud1 -> points[i].y;
        pt.z = cloud1 -> points[i].z;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        showCloud -> points.push_back(pt);
        
    }
    for(int i = 0; i < lidarPointCount; i++){
        pt.x = cloud2 -> points[i].x;
        pt.y = cloud2 -> points[i].y;
        pt.z = cloud2 -> points[i].z;
        pt.r = 0;
        pt.g = 0;
        pt.b = 255;
        showCloud -> points.push_back(pt);
    }


    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);
    
    while(!viewer.wasStopped()){
        
    }
    cv::waitKey();
    
}
void feature::rotate(int n){
    int count = 180;
    float rad = M_PI / count;
    float theta = 180/count;
    Eigen::Affine3f transformation = Eigen::Affine3f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // float minimumDist = 1000;
    // int minimumIdx = 0;
    // for(int i = 0; i < count * 2; i++){
    //     transformation.rotate(Eigen::AngleAxisf(rad * i, Eigen::Vector3f::UnitZ()));
    //     pcl::transformPointCloud(*this->lidarCloud, *transformed_cloud, transformation);
    //     float dist = calculateDist(this -> imageCloud, transformed_cloud);
    //     if(dist < minimumDist){
    //         minimumDist = dist;
    //         minimumIdx = i;
    //     }
    //     // distance.push_back(dist);
    //     cout << "After " << i* theta << "'rotation, distance is: " << dist << endl;
    // }
    // cout << "Minimum Distance is when theta = " << minimumIdx * theta << endl;
    // cout << "Minimum Distance is = " << minimumDist << endl;

    
    transformation.rotate(Eigen::AngleAxisf(rad * n, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*this->lidarCloud, *transformed_cloud, transformation);
    showCloud(this -> imageCloud, transformed_cloud);
    calculateDist(this -> imageCloud, transformed_cloud);

}
void feature::visualRotate(){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ldCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cv::Mat im = cv::imread(this -> imageEdgeSphereFile, cv::IMREAD_UNCHANGED);
    pcl::io::loadPCDFile(this -> lidarFile, *ldCloud);
    int rows = 2048;
    int cols = 2448;
    float radius;
    pcl::PointXYZRGB pt;
    for(int i = 0 ; i < rows; i++){
        for(int j = 0; j < cols ; j++){
            if(pow((i-1024),2) + pow((j-1124),2) <= pow(1024,2) && im.at<uchar>(i,j) == 255){
                pt.x = i-1024;
                pt.y = j-1124;
                pt.z = 0;
                pt.r = 255;
                pt.b = 0;
                pt.g = 0;
                showCloud->points.push_back(pt);
            }
        }
            
    }
    for(int i = 0; i < ldCloud-> points.size(); i++){
        radius = sqrt(pow(ldCloud -> points[i].x,2) + pow(ldCloud -> points[i].y,2) + pow(ldCloud -> points[i].z,2));
        pt.x = ldCloud -> points[i].x / radius * 1024;
        pt.y = ldCloud -> points[i].y / radius * 1024;
        pt.z = 0;
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
        showCloud -> points.push_back(pt);
    }
    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);
    
    while(!viewer.wasStopped()){
        
    }
    cv::waitKey();
}
float feature::calculateDist(pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr lidarCloud){
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
    kdtree->setInputCloud(imageCloud);
    int K = 1;
    vector<int> pointIdx(K);
    vector<float> distance(K);
    float average_dis = 0;
    int lidarCount = lidarCloud -> points.size();
    for(int i = 0; i < lidarCount; i++){
        pcl::PointXYZ pt = lidarCloud -> points[i];
        if(kdtree -> nearestKSearch(pt, K, pointIdx, distance) > 0){
            for(int j = 0; j < K; j++){
                float dis = sqrt(
                    pow(pt.x - imageCloud->points[pointIdx[j]].x, 2) +
                    pow(pt.y - imageCloud->points[pointIdx[j]].y, 2)
                );
                average_dis += dis;
            }
        }
    }
    average_dis /= lidarCount;
    cout << "Distance is: " << average_dis << endl;
    return average_dis;

}

// void poseCalculation(vector<Point2f> points1, vector<Point2f> points2){
//     ifstream lidarMarks(this -> lidarMarkFile);
//     string line;
//     while(getline(lidarMarks, line)){
//         stringstream ss(line);
        
//     }
// }

// void feature::txtToCloud(){

// }
void feature::roughMatch(int degree, int x_offset, int y_offset){
    ifstream lidarMarks(this -> lidarMarkFile);
    ifstream imageMarks(this -> imageMarkFile);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr lidarKPCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr imageKPCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB point;
    string line;
    float radius;
    int x;
    int y;
    // vector<point2i> lidarKeyPoints;
    // vector<point2i> imageKeyPoints;
    // point2i pt;

    // Get lidar keypoints
    while(getline(lidarMarks, line)){
        stringstream ss(line);
        int i = 0;
        vector<float> lidarMark(3);
        while(ss >> lidarMark[i]){
            i++;
        }
        radius = sqrt(pow(lidarMark[0],2) + pow(lidarMark[1],2) + pow(lidarMark[2],2));
        // int x = (int)(lidarMark[0] * HALF_WIDTH / radius + HALF_WIDTH);
        // int y = (int)lidarMark[1] * HALF_HEIGHT / radius + HALF_HEIGHT);
        int x = (int)(lidarMark[0] * HALF_WIDTH / radius);
        int y = (int)(lidarMark[1] * HALF_HEIGHT / radius);

        point.x = x;
        point.y = y;
        point.z = 0;
        point.b = 0;
        point.g = 255;
        point.r = 0;

        lidarKPCloud -> points.push_back(point); 
        // if (x >= 0 && x <= WIDTH && y >= 0 && y <= HEIGHT){
        //     pt.x = x;
        //     pt.y = y;

        //     point.x = x;
        //     point.y = y;
        //     point.z = 0;
        //     point.b = 0;
        //     point.g = 255;
        //     point.r = 0;
        //     lidarKeyPoints.push_back(pt);
        //     lidarKPCloud -> points.push_back(point); 
        // }  
    }

    // Get image keypoints
    while(getline(imageMarks, line)){
        stringstream ss(line);
        int i = 0;
        vector<int> imageMark(2);
        while(ss >> imageMark[i]){
            i++;
        }
        // pt.x = imageMark[0];
        // pt.y = imageMark[1];

        // point.x = imageMark[0]*2 - HALF_WIDTH;
        // point.y = imageMark[1]*2 - HALF_HEIGHT;
        point.x = imageMark[0]*2 - x_offset;
        point.y = imageMark[1]*2 - y_offset;
        point.z = 0;
        point.b = 0;
        point.g = 0;
        point.r = 255;
        imageKPCloud -> points.push_back(point); 
        // imageKeyPoints.push_back(imageMark);
    }

    float rad = M_PI / 180;
    float theta = 1;
    Eigen::Affine3f transformation = Eigen::Affine3f::Identity();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    transformation.rotate(Eigen::AngleAxisf(rad * degree, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*lidarKPCloud, *transformed_cloud, transformation);
    
    for(int i = 0; i < transformed_cloud -> points.size(); i++){
        showCloud -> points.push_back(transformed_cloud -> points[i]);
        showCloud -> points.push_back(imageKPCloud -> points[i]);
    }
    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);
    while(!viewer.wasStopped()){
        
    }
    cv::waitKey();
}


void feature::imageLongitudeAndLatitude(){
    cv::Mat img = cv::imread(this -> rawImageFile, cv::IMREAD_UNCHANGED);
    int height = 2048;
    int width = 2448;
    float center_x = width / 2;
    float center_y = height / 2;
    float radius = center_y;
    
    float theta_unit = (M_PI * 2) / (2048 * 3.14);
    float r_min = 300;
    float r_max = 1024;
    float r_unit = 1;

    int x;
    int y;
    int i;
    int j;
    float theta;
    float r;
    cv::Mat imageOutput = cv::Mat::zeros(342 * 3, 2144 * 3, CV_8UC3);
    for(j = 0, r = 0; r < 1024; r += r_unit, j++){
        for(i = 0, theta = 0; theta < (M_PI * 2); theta += theta_unit, i++){
            
                x = int(r * cos(theta)) + center_x;
                y = int(r * sin(theta)) + center_y;

                if(sqrt(pow(x - center_x, 2) + pow(y - center_y, 2)) > 300 && sqrt(pow(x - center_x, 2) + pow(y - center_y, 2)) < 1024 && x >= 0 && x <=2447 && y >= 0 && y <= 2047 ){
                    imageOutput.at<cv::Vec3b>(j, i)[0] = img.at<cv::Vec3b>(y, x)[0];
                    imageOutput.at<cv::Vec3b>(j, i)[1] = img.at<cv::Vec3b>(y, x)[1];
                    imageOutput.at<cv::Vec3b>(j, i)[2] = img.at<cv::Vec3b>(y, x)[2];
                }
        }
    }
    // cout << i << "," << j << endl;
    cv::imwrite(dataPath +"imageLatitude.png", imageOutput);
    // cv::waitKey(0);
    // cv::destoryAllWindows();
}

void feature::lidarLongitudeAndLatitude(){
    pcl::PointCloud<pcl::PointXYZI>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI pt;
    pcl::io::loadPCDFile(this->lidarDenseFile, *lidarDenseCloud);

    float theta;
    float phi;
    float radius;
    float x, y, z;
    float phi_max = 0.9;
    float phi_unit = 0.015;
    float theta_max = M_PI * 2;
    float theta_unit = M_PI * 2 / 360;
    int longitude, latitude;
    
    int lidarCount = lidarDenseCloud -> points.size();
    for(int i = 0; i < lidarCount; i++){
        x = lidarDenseCloud -> points[i].x;
        y = lidarDenseCloud -> points[i].y;
        z = lidarDenseCloud -> points[i].z;
        radius = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        phi = asin(z / radius);
        theta = atan(y / x);
        longitude = int(phi / phi_unit);
        latitude = int (theta / theta_unit);
        pt.x = latitude;
        pt.y = longitude;
        pt.z = 0;
        pt.intensity = lidarDenseCloud -> points[i].intensity;
        showCloud -> points.push_back(pt);
    }
    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);
    
    while(!viewer.wasStopped()){
        
    }
    cv::waitKey();
    // pcl::io::savePCDFileBinary(dataPath + "1.pcd", *showCloud);

}

void feature::visualComparism(){
    Mat image = imread(this -> imageFlatFile, -1);
    Mat lidarImage = imread(this -> lidarFlatFile, -1);
    int lidRows = lidarImage.rows;
    int lidCols = lidarImage.cols;
    int imRows = image.rows;
    int imCols = image.cols; 
    Mat lidarRGB = Mat::zeros(imRows, imCols, CV_8UC3);
    int b;
    int g;
    int r;
    int visibility;
    int shift = 0;
    // lidarRGB = image;
    int count = 0;
    for(int i = 0; i < lidRows; i++){
        for(int j = 0; j < lidCols; j++){
            if(lidarImage.at<uchar>(i, j) < 50){
                b = 255;
                g = 0;
                r = 0;
                lidarRGB.at<Vec3b>((i+shift) % imRows, j)[0] = b;
                lidarRGB.at<Vec3b>((i+shift) % imRows, j)[1] = g;
                lidarRGB.at<Vec3b>((i+shift) % imRows, j)[2] = r;
                count ++;
            }
            
        }
    }
    Mat imageShow = Mat::zeros(imRows, imCols, CV_8UC3);
    addWeighted(image, 1, lidarRGB, 0.8, 0, imageShow);
    imwrite("/home/xwy/1.png", imageShow);
    imshow("show", imageShow);
    waitKey();
}