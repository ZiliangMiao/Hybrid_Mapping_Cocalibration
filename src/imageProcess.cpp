// include headings
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
#include <time.h>
// include packages
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
// include pcl package
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
// include ros
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
// include ros message
#include <std_msgs/Header.h>
// include mlpack
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
// include other files
#include "imageProcess.h"

using namespace std;
using namespace cv;

// using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;

using namespace arma;

imageProcess::imageProcess(string pkgPath) {
    struct ScenesPath scenesPath(pkgPath);
    struct SceneFilePath SC1(scenesPath.sc1);
    struct SceneFilePath SC2(scenesPath.sc2);
    struct SceneFilePath SC3(scenesPath.sc3);
    struct SceneFilePath SC4(scenesPath.sc4);
    struct SceneFilePath SC5(scenesPath.sc5);
    this -> scenesFilePath.push_back(SC1);
    this -> scenesFilePath.push_back(SC2);
    this -> scenesFilePath.push_back(SC3);
    this -> scenesFilePath.push_back(SC4);
    this -> scenesFilePath.push_back(SC5);
}

void imageProcess::readEdge()
{
    string edgeOrgTxtPath = this -> scenesFilePath[this -> scIdx].EdgeOrgTxtPath;
    ifstream infile(edgeOrgTxtPath);
    string line;
    vector<vector<double>> edgeOrgTxtVec;
    while (getline(infile, line))
    {
        stringstream ss(line);
        string tmp;
        vector<double> v;
        while (getline(ss, tmp, '\t')) // split string with "\t"
        {                           
            v.push_back(stod(tmp)); // string -> double
        }
        if (v.size() == 2)
        {
            edgeOrgTxtVec.push_back(v);
        }
    }
    ROS_ASSERT_MSG(edgeOrgTxtVec.size() != 0, "Fisheye Read Edge Fault! Scene Index: %d", this -> numScenes);
    cout << "Imported Fisheye Edge Points: " << edgeOrgTxtVec.size() << endl;
    /********* Remove Dumplicated Points *********/
    std::sort(edgeOrgTxtVec.begin(), edgeOrgTxtVec.end());
    edgeOrgTxtVec.erase(unique(edgeOrgTxtVec.begin(), edgeOrgTxtVec.end()), edgeOrgTxtVec.end());
    cout << "Dumplicated Fisheye Edge Points: " << edgeOrgTxtVec.size() << endl;
    this -> edgeOrgTxtVec = edgeOrgTxtVec;
}

cv::Mat imageProcess::readOrgImage(){
    string HdrImgPath = this -> scenesFilePath[this -> scIdx].HdrImgPath;
    cv::Mat image = cv::imread(HdrImgPath, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG(((image.rows != 0 && image.cols != 0) || (image.rows < 16384 || image.cols < 16384)), "Size of original fisheye image is 0, check the path and filename! Scene Index: %d", this -> numScenes);
    ROS_ASSERT_MSG((image.rows == this->orgRows || image.cols == this->orgCols), "Size of original fisheye image is incorrect! Scene Index: %d", this -> numScenes);
    return image;
}

std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> imageProcess::fisheyeImageToSphere()
{
    // read the origin fisheye image and check the image size
    cv::Mat image = readOrgImage();
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;
    result = fisheyeImageToSphere(image);
    return result;
}

std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> imageProcess::fisheyeImageToSphere(cv::Mat image)
{
    // color space
    int r, g, b;   
    // cartesian coordinates (3d vector)
    double X, Y, Z;    
    // radius of each pixel point
    double radius;
    // angle with u-axis (rows-axis, x-axis)
    double phi;
    // angle with z-axis
    double theta;
    // intrinsic parameters
    double a0, a2, a3, a4;
    double c, d, e;
    double u0, v0;
    // theta range
    double thetaMin = M_PI, thetaMax = -M_PI;
    double phiMin = M_PI, phiMax = -M_PI;

    // intrinsic params
    a0 = this->intrinsic.a0;
    a2 = this->intrinsic.a2;
    a3 = this->intrinsic.a3;
    a4 = this->intrinsic.a4;
    c = this->intrinsic.c;
    d = this->intrinsic.d;
    e = this->intrinsic.e;
    u0 = this->intrinsic.u0;
    v0 = this->intrinsic.v0;

    pcl::PointXYZRGB ptPixel;
    pcl::PointXYZRGB ptPolar;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    ROS_ASSERT_MSG((image.rows == this->orgRows || image.cols == this->orgCols), "Size of original fisheye image is incorrect! Scene Index: %d", this -> numScenes);

    for (int u = 0; u < this->orgRows; u++)
    {
        for (int v = 0; v < this->orgCols; v++)
        {
            X = c * u + d * v - u0;
            Y = e * u + 1 * v - v0;
            radius = sqrt(pow(X, 2) + pow(Y, 2));
            if (radius != 0)
            {
                Z = a0 + a2 * pow(radius, 2) + a3 * pow(radius, 3) + a4 * pow(radius, 4);
                // spherical coordinates
                // caution: the default range of phi is -pi to pi, we need to modify this range to 0 to 2pi
                phi = atan2(Y, X) + M_PI; // note that atan2 is defined as Y/X
                theta = acos(Z / sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)));

                ROS_ASSERT_MSG((theta != 0), "Theta equals to zero! Scene Index: %d", this -> numScenes);

                // point cloud with origin polar coordinates
                ptPolar.x = theta;
                ptPolar.y = phi;
                ptPolar.z = 0;
                ptPolar.b = image.at<cv::Vec3b>(u, v)[0];
                ptPolar.g = image.at<cv::Vec3b>(u, v)[1];
                ptPolar.r = image.at<cv::Vec3b>(u, v)[2];
                camOrgPolarCloud->points.push_back(ptPolar);
                // point cloud with origin pixel coordinates
                ptPixel.x = u;
                ptPixel.y = v;
                ptPixel.z = 0;
                ptPixel.b = image.at<cv::Vec3b>(u, v)[0];
                ptPixel.g = image.at<cv::Vec3b>(u, v)[1];
                ptPixel.r = image.at<cv::Vec3b>(u, v)[2];
                camOrgPixelCloud->points.push_back(ptPixel);

                if (theta > thetaMax)
                {
                    thetaMax = theta;
                }
                if (theta < thetaMin)
                {
                    thetaMin = theta;
                }
                if (phi > phiMax)
                {
                    phiMax = phi;
                }
                if (phi < phiMin)
                {
                    phiMin = phi;
                }
            }
        }
    }

    cout << "Min theta of camera: " << thetaMin << endl;
    cout << "Max theta of camera: " << thetaMax << endl;

    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;
    result = std::make_tuple(camOrgPolarCloud, camOrgPixelCloud);

    return result;
}

vector<vector<vector<int>>> imageProcess::sphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar)
{
    vector<vector<vector<int>>> tagsMap = sphereToPlane(sphereCloudPolar, -1.0);
    return tagsMap;
}

vector<vector<vector<int>>> imageProcess::sphereToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphereCloudPolar, double bandwidth)
{
    double flatRows = this -> flatRows;
    double flatCols = this -> flatCols;
    cv::Mat flatImage = cv::Mat::zeros(flatRows, flatCols, CV_8UC3); // define the flat image

    // define the tag list
    vector<int> tagsList;
    vector<vector<vector<int>>> tagsMap(flatRows, vector<vector<int>>(flatCols, tagsList));

    // define the variables of KDTree search
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(sphereCloudPolar);

    // define the variables of KDTree search
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree2;
    kdtree2.setInputCloud(sphereCloudPolar);

    int invalidSearch = 0; // search invalid count
    int invalidIndex = 0;  // index invalid count
    double radPerPix = this -> radPerPix;
    double searchRadius = radPerPix / 2;
    // use KDTree to search the spherical point cloud
    for (int u = 0; u < flatRows; ++u)
    {
        // upper bound and lower bound of the current theta unit
        float theta_lb = u * radPerPix;
        float theta_ub = (u + 1) * radPerPix;
        float theta_center = (theta_ub + theta_lb) / 2;
        for (int v = 0; v < flatCols; ++v)
        {
            // upper bound and lower bound of the current phi unit
            float phi_lb = v * radPerPix;
            float phi_ub = (v + 1) * radPerPix;
            float phi_center = (phi_ub + phi_lb) / 2;
            // assign the theta and phi center to the searchPoint
            pcl::PointXYZRGB searchPoint;
            searchPoint.x = phi_center;
            searchPoint.y = theta_center;
            searchPoint.z = 0;
            // define the vector container for storing the info of searched points
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            // radius search
            int numRNN = kdtree.radiusSearch(searchPoint, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance); // number of the radius nearest neighbors
            // if the corresponding points are found in the radius neighborhood
            if (numRNN == 0) // no point found
            {
                // assign the theta and phi center to the searchPoint
                pcl::PointXYZRGB searchPoint;
                searchPoint.x = theta_center;
                searchPoint.y = phi_center;
                searchPoint.z = 0;
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                int numSecondSearch = 0;
                float scale = 1;
                while (numSecondSearch == 0)
                {
                    scale = scale + 0.05;
                    numSecondSearch = kdtree2.radiusSearch(searchPoint, scale * searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                    if (scale > 2)
                    {
                        flatImage.at<cv::Vec3b>(u, v)[0] = 0; // b
                        flatImage.at<cv::Vec3b>(u, v)[1] = 0; // g
                        flatImage.at<cv::Vec3b>(u, v)[2] = 0; // r
                        invalidSearch = invalidSearch + 1;
                        // add tags
                        tagsMap[u][v].push_back(0);
                        break;
                    }
                }
                if (numSecondSearch != 0) // there are more points found than one
                {
                    int B = 0, G = 0, R = 0; // mean value of RGB channels
                    for (int i = 0; i < pointIdxRadiusSearch.size(); ++i)
                    {
                        B = B + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].b;
                        G = G + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].g;
                        R = R + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].r;
                        // add tags
                        tagsMap[u][v].push_back(pointIdxRadiusSearch[i]);
                    }
                    flatImage.at<cv::Vec3b>(u, v)[0] = int(B / numSecondSearch); // b
                    flatImage.at<cv::Vec3b>(u, v)[1] = int(G / numSecondSearch); // g
                    flatImage.at<cv::Vec3b>(u, v)[2] = int(R / numSecondSearch); // r
                }
            }

            else // more than one points found
            {
                int B = 0, G = 0, R = 0; // mean value of RGB channels
                for (int i = 0; i < pointIdxRadiusSearch.size(); ++i)
                {
                    if (pointIdxRadiusSearch[i] > sphereCloudPolar->points.size() - 1)
                    {
                        // caution: a bug is hidden here, index of the searched point is bigger than size of the whole point cloud
                        flatImage.at<cv::Vec3b>(u, v)[0] = 0; // b
                        flatImage.at<cv::Vec3b>(u, v)[1] = 0; // g
                        flatImage.at<cv::Vec3b>(u, v)[2] = 0; // r
                        invalidIndex = invalidIndex + 1;
                        continue;
                    }
                    B = B + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].b;
                    G = G + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].g;
                    R = R + (*sphereCloudPolar)[pointIdxRadiusSearch[i]].r;
                    // add tags
                    tagsMap[u][v].push_back(pointIdxRadiusSearch[i]);
                }
                flatImage.at<cv::Vec3b>(u, v)[0] = int(B / numRNN); // b
                flatImage.at<cv::Vec3b>(u, v)[1] = int(G / numRNN); // g
                flatImage.at<cv::Vec3b>(u, v)[2] = int(R / numRNN); // r
            }
        }
    }

    cout << "number of invalid searches:" << invalidSearch << endl;
    cout << "number of invalid indices:" << invalidIndex << endl;

    string flatImgPath = this -> scenesFilePath[this -> scIdx].FlatImgPath;
    string fusionImgPath = this -> scenesFilePath[this -> scIdx].FusionImgPath;
    string resultPath = this -> scenesFilePath[this -> scIdx].ResultPath;

    /********* Image Generation *********/
    if (bandwidth < 0){
        cv::imwrite(flatImgPath, flatImage); /** flat image generation **/
    }
    else{
        string fusionImgPath = resultPath + "/sc_" + to_string(this -> scIdx) + "_fusion_bw_" + to_string(int(bandwidth)) + ".bmp";
        cv::imwrite(fusionImgPath, flatImage); /** fusion image generation **/
    }
    
    return tagsMap;
}

vector<vector<int>> imageProcess::edgeToPixel()
{
    string edgeImgPath = this -> scenesFilePath[this -> scIdx].EdgeImgPath;
    cv::Mat edgeImage = cv::imread(edgeImgPath, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG(((edgeImage.rows != 0 && edgeImage.cols != 0) || (edgeImage.rows < 16384 || edgeImage.cols < 16384)), "Size of original fisheye image is 0, check the path and filename! Scene Index: %d", this -> numScenes);
    ROS_ASSERT_MSG((edgeImage.rows == this->flatRows || edgeImage.cols == this->flatCols), "Size of original fisheye image is incorrect! Scene Index: %d", this -> numScenes);

    vector<vector<int>> edgePixels;
    for (int u = 0; u < edgeImage.rows; ++u)
    {
        for (int v = 0; v < edgeImage.cols; ++v)
        {
            if (edgeImage.at<uchar>(u, v) > 127)
            {
                vector<int> pixel{u, v};
                edgePixels.push_back(pixel);
            }
        }
    }

    /********* write the coordinates into txt file *********/
    string edgeTxtPath = this -> scenesFilePath[this -> scIdx].EdgeTxtPath;
    ofstream outfile;
    outfile.open(edgeTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < edgePixels.size(); ++i)
    {
        outfile << edgePixels[i][0] << "\t" << edgePixels[i][1] << endl;
    }
    outfile.close();

    return edgePixels;
}

void imageProcess::pixLookUp(vector<vector<int>> edgePixels, vector<vector<vector<int>>> tagsMap, pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud)
{
    vector<vector<double>> pixUV;
    int invalidLookUp = 0;
    for (int i = 0; i < edgePixels.size(); ++i)
    {
        int u = edgePixels[i][0]; // u-axis, v-axis of pix points
        int v = edgePixels[i][1];

        double x = 0;
        double y = 0;

        int size = tagsMap[u][v].size();
        if (size == 0)
        {
            invalidLookUp = invalidLookUp + 1;
            x = 0;
            y = 0;
            continue;
        }
        else
        {
            for (int j = 0; j < size; ++j)
            {
                pcl::PointXYZRGB pt = (*camOrgPixelCloud)[tagsMap[u][v][j]];
                x = x + pt.x;
                y = y + pt.y;
            }
            // assign mean values to the output
            x = x / tagsMap[u][v].size();
            y = y / tagsMap[u][v].size();

            vector<double> pixel{x, y};
            pixUV.push_back(pixel);
        }
    }

    this -> edgeOrgTxtVec = pixUV;
    cout << "number of invalid lookups(image): " << invalidLookUp << endl;

    string edgeOrgTxtPath = this -> scenesFilePath[this -> scIdx].EdgeOrgTxtPath;
    /********* write the coordinates into txt file *********/
    ofstream outfile;
    outfile.open(edgeOrgTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < pixUV.size(); ++i)
    {
        outfile << pixUV[i][0] << "\t" << pixUV[i][1] << endl;
    }
    outfile.close();
}

vector<vector<double>> imageProcess::edgeTransform()
{
    vector<vector<double>> edgeOrgTxtVec = this -> edgeOrgTxtVec;
    vector<double> camEdgeRows(edgeOrgTxtVec.size());
    vector<double> camEdgeCols(edgeOrgTxtVec.size());

    double radius;
    double phi;
    double theta;
    double X, Y, Z;

    // intrinsic parameters
    double a0, a2, a3, a4;
    double c, d, e;
    double u0, v0;
    // intrinsic params
    a0 = this->intrinsic.a0;
    a2 = this->intrinsic.a2;
    a3 = this->intrinsic.a3;
    a4 = this->intrinsic.a4;
    c = this->intrinsic.c;
    d = this->intrinsic.d;
    e = this->intrinsic.e;
    u0 = this->intrinsic.u0;
    v0 = this->intrinsic.v0;

    for (int i = 0; i < edgeOrgTxtVec.size(); i++)
    {
        double u = edgeOrgTxtVec[i][0];
        double v = edgeOrgTxtVec[i][1];
        X = c * u + d * v - u0;
        Y = e * u + 1 * v - v0;
        radius = sqrt(pow(X, 2) + pow(Y, 2));

        Z = a0 + a2 * pow(radius, 2) + a3 * pow(radius, 3) + a4 * pow(radius, 4);
        phi = atan2(Y, X) + M_PI; // note that atan2 is defined as Y/X
        theta = acos(Z / sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2)));

        camEdgeRows[i] = theta;
        camEdgeCols[i] = phi;
    }

    vector<vector<double>> camEdgePolar(2);
    camEdgePolar[0] = camEdgeRows;
    camEdgePolar[1] = camEdgeCols;

    return camEdgePolar;
}

// convert cv::Mat to arma::mat (static and stable method)
static void cv_cast_arma(const cv::Mat &cv_mat_in, arma::mat &arma_mat_out)
{
    // convert unsigned int cv::Mat to arma::Mat<double>
    for (int r = 0; r < cv_mat_in.rows; r++)
    {
        for (int c = 0; c < cv_mat_in.cols; c++)
        {
            arma_mat_out(r, c) = cv_mat_in.data[r * cv_mat_in.cols + c] / 255.0;
        }
    }
}

// convert arma::mat to Eigen::Matrix (static and stable method)
static Eigen::MatrixXd arma_cast_eigen(arma::mat arma_A)
{

    Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                          arma_A.n_rows,
                                                          arma_A.n_cols);

    return eigen_B;
}

// create static blur image for autodiff ceres optimization
// the "scale" and "polar" option is implemented but not tested/supported in optimization.
std::vector<double> imageProcess::kdeBlur(double bandwidth, double scale, bool polar)
{
    clock_t start_time = clock();
    const double relError = 0.05;
    const int n_rows = scale * this->orgRows;
    const int n_cols = scale * this->orgCols;

    arma::mat query;

    // number of rows equal to number of dimensions, query.n_rows == reference.n_rows is required
    const int ref_size = this -> edgeOrgTxtVec.size();
    arma::mat reference(2, ref_size);
    for (int i = 0; i < ref_size; ++i)
    {
        reference(0, i) = (double)this -> edgeOrgTxtVec[i][0];
        reference(1, i) = (double)this -> edgeOrgTxtVec[i][1];
    }

    if (!polar)
    {
        query = arma::mat(2, n_cols * n_rows);
        arma::vec rows = arma::linspace(0, this->orgRows - 1, n_rows);
        arma::vec cols = arma::linspace(0, this->orgCols - 1, n_cols);

        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                query(0, i * n_cols + j) = rows.at(n_rows - 1 - i);
                query(1, i * n_cols + j) = cols.at(j);
            }
        }
    }
    else
    {
        query = arma::mat(2, n_cols * n_rows);
        arma::vec r_q = arma::linspace(1, this->flatRows, n_rows);
        arma::vec sin_q = arma::linspace(0, (2 * M_PI) * (1 - 1 / n_cols), n_cols);
        arma::vec cos_q = sin_q;
        sin_q.for_each([](mat::elem_type &val)
                       { val = sin(val); });
        cos_q.for_each([](mat::elem_type &val)
                       { val = cos(val); });

        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                query(0, i * n_cols + j) = r_q.at(i) * cos_q.at(j) + this->intrinsic.u0;
                query(1, i * n_cols + j) = r_q.at(i) * sin_q.at(j) + this->intrinsic.v0;
            }
        }
    }

    arma::vec kdeEstimations;
    mlpack::kernel::EpanechnikovKernel kernel(bandwidth);
    mlpack::metric::EuclideanDistance metric;
    mlpack::kde::KDE<EpanechnikovKernel, mlpack::metric::EuclideanDistance, arma::mat> kde(relError, 0.00, kernel);
    kde.Train(reference);
    kde.Evaluate(query, kdeEstimations);

    std::vector<double> img = arma::conv_to<std::vector<double>>::from(kdeEstimations);


    string kdeTxtPath = this -> scenesFilePath[this -> scIdx].KdeTxtPath;
    ofstream outfile;
    outfile.open(kdeTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < n_rows; ++i)
    {
        for (int j = 0; j < n_cols; j++)
        {
            int index = i * n_cols + j;
            outfile << query.at(0, index) << "\t"
                    << query.at(1, index) << "\t"
                    << kdeEstimations(index) << endl;
            // img(i, j) = kdeEstimations(index);
        }
    }
    double kde_sum = arma::sum(kdeEstimations);
    double kde_max = arma::max(kdeEstimations);
    outfile.close();
    cout << "New kde image generated with sum = " << kde_sum << " and max = " << kde_max << endl;
    cout << "The run time is: " <<(double)(clock() - start_time) / CLOCKS_PER_SEC << "s, bandwidth = " << bandwidth << endl;
    return img;
}
