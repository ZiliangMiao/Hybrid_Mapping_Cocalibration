/** basic **/
#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
/** opencv **/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/** pcl **/
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/principal_curvatures.h>
#include <Eigen/Core>
/** ros **/
#include <ros/ros.h>
/** mlpack **/
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
/** headings **/
#include "FisheyeProcess.h"
/** namespace **/
using namespace std;
using namespace cv;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;
using namespace arma;

FisheyeProcess::FisheyeProcess(string pkg_path) {
    cout << "----- Fisheye: ImageProcess -----" << endl;
    this -> num_scenes = 3;
    /** reserve the memory for vectors stated in LidarProcess.h **/
    this -> scenes_files_path_vec.reserve(this -> num_scenes);
    this -> edge_pixels_vec.reserve(this -> num_scenes);
    this -> edge_fisheye_pixels_vec.reserve(this -> num_scenes);
    this -> tags_map_vec.reserve(this -> num_scenes);

    /** push the data directory path into vector **/
    this -> scenes_path_vec.push_back(pkg_path + "/data/sanjiao_pose0/-40");
    this -> scenes_path_vec.push_back(pkg_path + "/data/sanjiao_pose0/0");
    this -> scenes_path_vec.push_back(pkg_path + "/data/sanjiao_pose0/40");

    for (int idx = 0; idx < num_scenes; ++idx) {
        struct SceneFilePath sc(scenes_path_vec[idx]);
        this -> scenes_files_path_vec.push_back(sc);
    }
    cout << endl;
}

void FisheyeProcess::ReadEdge() {
    cout << "----- Fisheye: ReadEdge -----" << endl;
    cout << "Scene Index in Fisheye ReadEdge: " << this -> scene_idx << endl;
    string edge_fisheye_txt_path = this -> scenes_files_path_vec[this -> scene_idx].edge_fisheye_pixels_path;

    ifstream infile(edge_fisheye_txt_path);
    string line;
    EdgeFisheyePixels edge_fisheye_pixels;
    while (getline(infile, line)) {
        stringstream ss(line);
        string tmp;
        vector<double> v;
        while (getline(ss, tmp, '\t')) {
            v.push_back(stod(tmp)); /** split string with "\t" **/
        }
        if (v.size() == 2) {
            edge_fisheye_pixels.push_back(v);
        }
    }
    ROS_ASSERT_MSG(!edge_fisheye_pixels.empty(), "Fisheye Read Edge Fault! Scene Index: %d", this -> num_scenes);
    cout << "Imported Fisheye Edge Points: " << edge_fisheye_pixels.size() << endl;

    /** remove duplicated points **/
    std::sort(edge_fisheye_pixels.begin(), edge_fisheye_pixels.end());
    edge_fisheye_pixels.erase(unique(edge_fisheye_pixels.begin(), edge_fisheye_pixels.end()), edge_fisheye_pixels.end());
    cout << "Fisheye Edge Points after Duplicated Removed: " << edge_fisheye_pixels.size() << endl;
    this -> edge_fisheye_pixels_vec.push_back(edge_fisheye_pixels);
    cout << endl;
}

cv::Mat FisheyeProcess::ReadFisheyeImage() {
    cout << "----- Fisheye: ReadOrgImage -----" << endl;
    string fisheye_hdr_img_path = this -> scenes_files_path_vec[this -> scene_idx].fisheye_hdr_img_path;
    cv::Mat fisheye_hdr_image = cv::imread(fisheye_hdr_img_path, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG(((fisheye_hdr_image.rows != 0 && fisheye_hdr_image.cols != 0) || (fisheye_hdr_image.rows < 16384 || fisheye_hdr_image.cols < 16384)), "size of original fisheye image is 0, check the path and filename! Scene Index: %d", this -> num_scenes);
    ROS_ASSERT_MSG((fisheye_hdr_image.rows == this->kFisheyeRows || fisheye_hdr_image.cols == this->kFisheyeCols), "size of original fisheye image is incorrect! Scene Index: %d", this -> num_scenes);
    cout << endl;
    return fisheye_hdr_image;
}

std::tuple<RGBCloudPtr, RGBCloudPtr> FisheyeProcess::FisheyeImageToSphere() {
    cout << "----- Fisheye: FisheyeImageToSphere2 -----" << endl;
    /** read the original fisheye image and check the image size **/
    cv::Mat image = ReadFisheyeImage();
    std::tuple<RGBCloudPtr, RGBCloudPtr> result;
    result = FisheyeImageToSphere(image);
    cout << endl;
    return result;
}

std::tuple<RGBCloudPtr, RGBCloudPtr> FisheyeProcess::FisheyeImageToSphere(cv::Mat &image) {
    cout << "----- Fisheye: FisheyeImageToSphere -----" << endl;
    int r, g, b;
    float x, y, z;
    float radius, theta, phi;
    /** intrinsic parameters **/
    float a0, a2, a3, a4;
    float c, d, e;
    float u0, v0;
    a0 = this->intrinsic.a0;
    a2 = this->intrinsic.a2;
    a3 = this->intrinsic.a3;
    a4 = this->intrinsic.a4;
    c = this->intrinsic.c;
    d = this->intrinsic.d;
    e = this->intrinsic.e;
    u0 = this->intrinsic.u0;
    v0 = this->intrinsic.v0;

    pcl::PointXYZRGB pixel_pt;
    pcl::PointXYZRGB polar_pt;
    RGBCloudPtr fisheye_pixel_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    RGBCloudPtr polar_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    ROS_ASSERT_MSG((image.rows == this->kFisheyeRows || image.cols == this->kFisheyeCols), "size of original fisheye image is incorrect! Scene Index: %d", this -> num_scenes);

    for (int u = 0; u < this -> kFisheyeRows; u++) {
        for (int v = 0; v < this -> kFisheyeCols; v++) {
            x = c * u + d * v - u0;
            y = e * u + 1 * v - v0;
            radius = sqrt(pow(x, 2) + pow(y, 2));
            if (radius != 0) {
                z = a0 + a2 * pow(radius, 2) + a3 * pow(radius, 3) + a4 * pow(radius, 4);
                /** spherical coordinates **/
                /** caution: the default range of phi is -pi to pi, we need to modify this range to 0 to 2pi **/
                phi = atan2(y, x); // note that atan2 is defined as Y/X
                theta = acos(z / sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));

                /** point cloud with origin polar coordinates **/
                polar_pt.x = theta;
                polar_pt.y = phi;
                polar_pt.z = 0;
                polar_pt.b = image.at<cv::Vec3b>(u, v)[0];
                polar_pt.g = image.at<cv::Vec3b>(u, v)[1];
                polar_pt.r = image.at<cv::Vec3b>(u, v)[2];
                polar_cloud->points.push_back(polar_pt);
                
                /** point cloud with origin pixel coordinates **/
                pixel_pt.x = u;
                pixel_pt.y = v;
                pixel_pt.z = 0;
                pixel_pt.b = image.at<cv::Vec3b>(u, v)[0];
                pixel_pt.g = image.at<cv::Vec3b>(u, v)[1];
                pixel_pt.r = image.at<cv::Vec3b>(u, v)[2];
                fisheye_pixel_cloud->points.push_back(pixel_pt);
            }
        }
    }
    std::tuple<RGBCloudPtr, RGBCloudPtr> result;
    result = std::make_tuple(polar_cloud, fisheye_pixel_cloud);
    cout << endl;
    return result;
}

void FisheyeProcess::SphereToPlane(RGBCloudPtr polar_cloud) {
    cout << "----- Fisheye: SphereToPlane2 -----" << endl;
    SphereToPlane(polar_cloud, -1.0);
    cout << endl;
}

void FisheyeProcess::SphereToPlane(RGBCloudPtr polar_cloud, double bandwidth) {
    cout << "----- Fisheye: SphereToPlane -----" << endl;
    double flat_rows = this -> kFlatRows;
    double flat_cols = this -> kFlatCols;
    cv::Mat flat_image = cv::Mat::zeros(flat_rows, flat_cols, CV_8UC3); // define the flat image

    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    // define the variables of KDTree search
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(polar_cloud);
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree2;
    kdtree2.setInputCloud(polar_cloud);

    int invalid_search = 0;
    int invalid_index = 0;
    double rad_per_pix = this -> kRadPerPix;
    double search_radius = rad_per_pix / 2;

    // use KDTree to search the spherical point cloud
    for (int u = 0; u < flat_rows; ++u) {
        // upper bound and lower bound of the current theta unit
        float theta_lb = u * rad_per_pix;
        float theta_ub = (u + 1) * rad_per_pix;
        float theta_center = (theta_ub + theta_lb) / 2;
        for (int v = 0; v < flat_cols; ++v) {
            // upper bound and lower bound of the current phi unit
            float phi_lb = -M_PI + v * rad_per_pix;
            float phi_ub = -M_PI + (v + 1) * rad_per_pix;
            float phi_center = (phi_ub + phi_lb) / 2;
            // assign the theta and phi center to the searchPoint
            pcl::PointXYZRGB search_point;
            search_point.x = theta_center;
            search_point.y = phi_center;
            search_point.z = 0;
            // define the vector container for storing the info of searched points
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            // radius search
            int num_RNN = kdtree.radiusSearch(search_point, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance); // number of the radius nearest neighbors
            // if the corresponding points are found in the radius neighborhood
            if (num_RNN == 0) {
                // assign the theta and phi center to the searchPoint
                pcl::PointXYZRGB searchPoint;
                searchPoint.x = theta_center;
                searchPoint.y = phi_center;
                searchPoint.z = 0;
                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                int numSecondSearch = 0;
                float scale = 1;
                while (numSecondSearch == 0) {
                    scale = scale + 0.05;
                    numSecondSearch = kdtree2.radiusSearch(searchPoint, scale * search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
                    if (scale > 2) {
                        flat_image.at<cv::Vec3b>(u, v)[0] = 0; // b
                        flat_image.at<cv::Vec3b>(u, v)[1] = 0; // g
                        flat_image.at<cv::Vec3b>(u, v)[2] = 0; // r
                        invalid_search = invalid_search + 1;
                        tags_map[u][v].pts_indices.push_back(0);
                        break;
                    }
                }
                if (numSecondSearch != 0) {
                    int B = 0, G = 0, R = 0; // mean value of RGB channels
                    for (int i = 0; i < pointIdxRadiusSearch.size(); ++i) {
                        B = B + (*polar_cloud)[pointIdxRadiusSearch[i]].b;
                        G = G + (*polar_cloud)[pointIdxRadiusSearch[i]].g;
                        R = R + (*polar_cloud)[pointIdxRadiusSearch[i]].r;
                        tags_map[u][v].pts_indices.push_back(pointIdxRadiusSearch[i]);
                    }
                    flat_image.at<cv::Vec3b>(u, v)[0] = int(B / numSecondSearch); // b
                    flat_image.at<cv::Vec3b>(u, v)[1] = int(G / numSecondSearch); // g
                    flat_image.at<cv::Vec3b>(u, v)[2] = int(R / numSecondSearch); // r
                }
            }
            else {
                int B = 0, G = 0, R = 0; // mean value of RGB channels
                for (int i = 0; i < pointIdxRadiusSearch.size(); ++i) {
                    if (pointIdxRadiusSearch[i] > polar_cloud->points.size() - 1) {
                        // caution: a bug is hidden here, index of the searched point is bigger than size of the whole point cloud
                        flat_image.at<cv::Vec3b>(u, v)[0] = 0; // b
                        flat_image.at<cv::Vec3b>(u, v)[1] = 0; // g
                        flat_image.at<cv::Vec3b>(u, v)[2] = 0; // r
                        invalid_index = invalid_index + 1;
                        continue;
                    }
                    B = B + (*polar_cloud)[pointIdxRadiusSearch[i]].b;
                    G = G + (*polar_cloud)[pointIdxRadiusSearch[i]].g;
                    R = R + (*polar_cloud)[pointIdxRadiusSearch[i]].r;
                    tags_map[u][v].pts_indices.push_back(pointIdxRadiusSearch[i]);
                }
                flat_image.at<cv::Vec3b>(u, v)[0] = int(B / num_RNN); // b
                flat_image.at<cv::Vec3b>(u, v)[1] = int(G / num_RNN); // g
                flat_image.at<cv::Vec3b>(u, v)[2] = int(R / num_RNN); // r
            }
        }
    }
    this -> tags_map_vec.push_back(tags_map);
    cout << "number of invalid searches:" << invalid_search << endl;
    cout << "number of invalid indices:" << invalid_index << endl;

    string flat_img_path = this -> scenes_files_path_vec[this -> scene_idx].flat_img_path;
    string fusion_img_path = this -> scenes_files_path_vec[this -> scene_idx].fusion_img_path;
    string result_folder_path = this -> scenes_files_path_vec[this -> scene_idx].fusion_result_folder_path;

    /********* Image Generation *********/
    if (bandwidth < 0) {
        cv::imwrite(flat_img_path, flat_image); /** flat image generation **/
    }
    else {
        string fusionImgPath = result_folder_path + "/sc_" + to_string(this -> scene_idx) + "_fusion_bw_" + to_string(int(bandwidth)) + ".bmp";
        cv::imwrite(fusionImgPath, flat_image); /** fusion image generation **/
    }
    cout << endl;
}

void FisheyeProcess::EdgeToPixel() {
    cout << "----- Fisheye: EdgeToPixel -----" << endl;
    string edge_img_path = this -> scenes_files_path_vec[this -> scene_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nScene Index: %d \nPath: %s", this -> num_scenes, edge_img_path.data());
    ROS_ASSERT_MSG((edge_img.rows == this->kFlatRows || edge_img.cols == this->kFlatCols), "Size of original fisheye image is (%d, %d), Scene Index: %d", edge_img.rows, edge_img.cols, this -> num_scenes);
    EdgePixels edge_pixels;
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                vector<int> pixel{u, v};
                edge_pixels.push_back(pixel);
            }
        }
    }
    this -> edge_pixels_vec.push_back(edge_pixels);
    cout << endl;
}

void FisheyeProcess::PixLookUp(pcl::PointCloud<pcl::PointXYZRGB>::Ptr fisheye_pixel_cloud) {
    cout << "----- Fisheye: PixLookUp -----" << endl;
    int invalid_edge_pix = 0;
    EdgeFisheyePixels edge_fisheye_pixels;
    EdgePixels edge_pixels = this -> edge_pixels_vec[this -> scene_idx];
    TagsMap tags_map = this -> tags_map_vec[this -> scene_idx];
    for (auto & edge_pixel : edge_pixels) {
        int u = edge_pixel[0];
        int v = edge_pixel[1];
        double x = 0;
        double y = 0;

        int size = tags_map[u][v].pts_indices.size();
        if (size == 0) {
            invalid_edge_pix++;
        }
        else {
            for (int j = 0; j < size; ++j) {
                pcl::PointXYZRGB pt = (*fisheye_pixel_cloud)[tags_map[u][v].pts_indices[j]];
                x = x + pt.x;
                y = y + pt.y;
            }
            x = x / tags_map[u][v].pts_indices.size();
            y = y / tags_map[u][v].pts_indices.size();
            vector<double> pixel{x, y};
            edge_fisheye_pixels.push_back(pixel);
        }
    }
    this -> edge_fisheye_pixels_vec.push_back(edge_fisheye_pixels);
    cout << "number of invalid lookups(image): " << invalid_edge_pix << endl;

    string edgeOrgTxtPath = this -> scenes_files_path_vec[this -> scene_idx].edge_fisheye_pixels_path;
    /********* write the coordinates into txt file *********/
    ofstream outfile;
    outfile.open(edgeOrgTxtPath, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (auto & edge_fisheye_pixel : edge_fisheye_pixels) {
        outfile << edge_fisheye_pixel[0] << "\t" << edge_fisheye_pixel[1] << endl;
    }
    outfile.close();
    cout << endl;
}

// create static blur image for autodiff ceres optimization
// the "scale" and "polar" option is implemented but not tested/supported in optimization.
std::vector<double> FisheyeProcess::Kde(double bandwidth, double scale, bool polar) {
    cout << "----- Fisheye: Kde -----" << endl;
    clock_t start_time = clock();
    const double relError = 0.05;
    const int n_rows = scale * this -> kFisheyeRows;
    const int n_cols = scale * this -> kFisheyeCols;
    arma::mat query;
    // number of rows equal to number of dimensions, query.n_rows == reference.n_rows is required
    EdgeFisheyePixels edge_fisheye_pixels = this -> edge_fisheye_pixels_vec[this -> scene_idx];
    const int ref_size = edge_fisheye_pixels.size();
    arma::mat reference(2, ref_size);
    for (int i = 0; i < ref_size; ++i) {
        reference(0, i) = edge_fisheye_pixels[i][0];
        reference(1, i) = edge_fisheye_pixels[i][1];
    }

    if (!polar) {
        query = arma::mat(2, n_cols * n_rows);
        arma::vec rows = arma::linspace(0, this -> kFisheyeRows - 1, n_rows);
        arma::vec cols = arma::linspace(0, this -> kFisheyeCols - 1, n_cols);

        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                query(0, i * n_cols + j) = rows.at(n_rows - 1 - i);
                query(1, i * n_cols + j) = cols.at(j);
            }
        }
    }
    else {
        query = arma::mat(2, n_cols * n_rows);
        arma::vec r_q = arma::linspace(1, this -> kFlatRows, n_rows);
        arma::vec sin_q = arma::linspace(0, (2 * M_PI) * (1 - 1 / n_cols), n_cols);
        arma::vec cos_q = sin_q;
        sin_q.for_each([](mat::elem_type &val) { val = sin(val); });
        cos_q.for_each([](mat::elem_type &val) { val = cos(val); });

        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                query(0, i * n_cols + j) = r_q.at(i) * cos_q.at(j) + this -> intrinsic.u0;
                query(1, i * n_cols + j) = r_q.at(i) * sin_q.at(j) + this -> intrinsic.v0;
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
    string kdeTxtPath = this -> scenes_files_path_vec[this -> scene_idx].kde_samples_path;
    ofstream outfile;
    outfile.open(kdeTxtPath, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; j++) {
            int index = i * n_cols + j;
            outfile << query.at(0, index) << "\t"
                    << query.at(1, index) << "\t"
                    << kdeEstimations(index) << endl;
        }
    }
    // double kde_sum = arma::sum(kdeEstimations);
    // double kde_max = arma::max(kdeEstimations);
    outfile.close();
    cout << "New kde image generated with size (" << n_rows << ", " << n_cols << ") in "
         <<(double)(clock() - start_time) / CLOCKS_PER_SEC << "s, bandwidth = " << bandwidth << endl;
    cout << endl;
    return img;
}

// convert cv::Mat to arma::mat (static and stable method)
static void cv_cast_arma(const cv::Mat &cv_mat_in, arma::mat &arma_mat_out) {
    // convert unsigned int cv::Mat to arma::Mat<double>
    for (int r = 0; r < cv_mat_in.rows; r++) {
        for (int c = 0; c < cv_mat_in.cols; c++) {
            arma_mat_out(r, c) = cv_mat_in.data[r * cv_mat_in.cols + c] / 255.0;
        }
    }
}

// convert arma::mat to Eigen::Matrix (static and stable method)
static Eigen::MatrixXd arma_cast_eigen(arma::mat arma_A) {
    Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                          arma_A.n_rows,
                                                          arma_A.n_cols);
    return eigen_B;
}
