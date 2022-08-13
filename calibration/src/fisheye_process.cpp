/** basic **/
#include <stdio.h>
#include <stdlib.h>
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
#include "fisheye_process.h"
#include "common_lib.h"
/** namespace **/
using namespace std;
using namespace cv;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;
using namespace arma;
using namespace tk;

FisheyeProcess::FisheyeProcess() {
    /** parameter server **/
    ros::param::get("essential/kDatasetName", this->dataset_name);
    this->kDatasetPath = this->kPkgPath + "/data/" + this->dataset_name;
    ros::param::get("essential/kNumSpots", this->num_spots);
    ros::param::get("essential/kNumViews", this->num_views);
    ros::param::get("essential/kFisheyeRows", this->kFisheyeRows);
    ros::param::get("essential/kFisheyeCols", this->kFisheyeCols);
    ros::param::get("essential/kAngleInit", this->view_angle_init);
    ros::param::get("essential/kAngleStep", this->view_angle_step);
    this->fullview_idx = (this->num_views - 1) / 2;

    cout << "----- Fisheye: ImageProcess -----" << endl;
    /** create objects, initialization **/
    string pose_folder_path_temp;
    PoseFilePath pose_file_path_temp;
    EdgePixels edge_pixels_temp;
    EdgeFisheyePixels edge_fisheye_pixels_temp;
    TagsMap tags_map_temp;
    for (int i = 0; i < num_spots; ++i) {
        vector<string> poses_path_vec_temp;
        vector<PoseFilePath> poses_file_path_vec_temp;
        vector<EdgePixels> edge_pixels_vec_temp;
        vector<TagsMap> tags_map_vec_temp;
        vector<EdgeFisheyePixels> edge_fisheye_pixels_vec_temp;
        for (int j = 0; j < num_views; ++j) {
            poses_path_vec_temp.push_back(pose_folder_path_temp);
            poses_file_path_vec_temp.push_back(pose_file_path_temp);
            edge_pixels_vec_temp.push_back(edge_pixels_temp);
            tags_map_vec_temp.push_back(tags_map_temp);
            edge_fisheye_pixels_vec_temp.push_back(edge_fisheye_pixels_temp);
        }
        this->poses_folder_path_vec.push_back(poses_path_vec_temp);
        this->poses_files_path_vec.push_back(poses_file_path_vec_temp);
        this->edge_pixels_vec.push_back(edge_pixels_vec_temp);
        this->tags_map_vec.push_back(tags_map_vec_temp);
        this->edge_fisheye_pixels_vec.push_back(edge_fisheye_pixels_vec_temp);
    }

    /** degree map **/
    for (int i = 0; i < this->num_spots; ++i) {
        for (int j = 0; j < this->num_views; ++j) {
            int v_degree = this->view_angle_init + this->view_angle_step * j;
            this -> degree_map[j] = v_degree;
            this -> poses_folder_path_vec[i][j] = this->kDatasetPath + "/spot" + to_string(i) + "/" + to_string(v_degree);
        }
    }

    for (int i = 0; i < this -> num_spots; ++i) {
        for (int j = 0; j < this -> num_views; ++j) {
            struct PoseFilePath pose_file_path(poses_folder_path_vec[i][j]);
            this -> poses_files_path_vec[i][j] = pose_file_path;
        }
    }
}

void FisheyeProcess::ReadEdge() {
    cout << "----- Fisheye: ReadEdge -----" << " Spot Index: " << this->spot_idx << endl;
    string edge_fisheye_txt_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_fisheye_pixels_path;

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
    ROS_ASSERT_MSG(!edge_fisheye_pixels.empty(), "Fisheye Read Edge Fault! View Index: %d", this->view_idx);
    cout << "Imported Fisheye Edge Points: " << edge_fisheye_pixels.size() << endl;

    /** remove duplicated points **/
    std::sort(edge_fisheye_pixels.begin(), edge_fisheye_pixels.end());
    edge_fisheye_pixels.erase(unique(edge_fisheye_pixels.begin(), edge_fisheye_pixels.end()), edge_fisheye_pixels.end());
    cout << "Fisheye Edge Points after Duplicated Removed: " << edge_fisheye_pixels.size() << endl;
    this->edge_fisheye_pixels_vec[this->spot_idx][this->view_idx] =  edge_fisheye_pixels;
}

cv::Mat FisheyeProcess::ReadFisheyeImage(string fisheye_hdr_img_path) {
    cout << "----- Fisheye: ReadFisheyeImage -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    cv::Mat fisheye_hdr_image = cv::imread(fisheye_hdr_img_path, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((fisheye_hdr_image.rows != 0 && fisheye_hdr_image.cols != 0),
                   "size of original fisheye image is 0, check the path and filename! View Index: %d", this->view_idx);
    ROS_ASSERT_MSG((fisheye_hdr_image.rows == this->kFisheyeRows || fisheye_hdr_image.cols == this->kFisheyeCols),
                   "size of original fisheye image is incorrect! View Index: %d", this->view_idx);
    return fisheye_hdr_image;
}

std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> FisheyeProcess::FisheyeImageToSphere() {
    cout << "----- Fisheye: FisheyeImageToSphere2 -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    /** read the original fisheye image and check the image size **/
    string fisheye_hdr_img_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fisheye_hdr_img_path;
    cv::Mat image = ReadFisheyeImage(fisheye_hdr_img_path);
    std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> result;
    std::vector<double> intrinsic_ = 
        {intrinsic.u0, intrinsic.v0, intrinsic.a0, intrinsic.a1, intrinsic.a2, intrinsic.a3, intrinsic.a4, intrinsic.c, intrinsic.d, intrinsic.e};
    tk::spline spline = InverseSpline(intrinsic_);
    result = FisheyeImageToSphere(image, spline);
    return result;
}

std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> FisheyeProcess::FisheyeImageToSphere(cv::Mat &image, tk::spline spline) {
    cout << "----- Fisheye: FisheyeImageToSphere -----"  << " Spot Index: " << this->spot_idx << endl;

    /** intrinsic parameters **/
    float c, d, e, u0, v0;
    c = this->intrinsic.c;
    d = this->intrinsic.d;
    e = this->intrinsic.e;
    u0 = this->intrinsic.u0;
    v0 = this->intrinsic.v0;

    PointRGB pixel_pt;
    PointRGB polar_pt;
    CloudRGB::Ptr fisheye_pixel_cloud(new CloudRGB);
    CloudRGB::Ptr fisheye_polar_cloud(new CloudRGB);
    ROS_ASSERT_MSG((image.rows == this->kFisheyeRows || image.cols == this->kFisheyeCols),
                   "size of original fisheye image is incorrect! View Index: %d", this->view_idx);

    float theta_min = M_PI, theta_max = -M_PI;

    for (int u = 0; u < this->kFisheyeRows; u++) {
        for (int v = 0; v < this->kFisheyeCols; v++) {
            float x = c * u + d * v - u0;
            float y = e * u + 1 * v - v0;
            float radius = sqrt(pow(x, 2) + pow(y, 2));
            if (radius > 1e-6) {
                /** spherical coordinates **/
                float phi = atan2(y, x); // note that atan2 is defined as Y/X
                float theta = spline(radius);

                /** point cloud with origin polar coordinates **/
                polar_pt.x = theta;
                polar_pt.y = phi;
                polar_pt.z = 0;
                polar_pt.b = image.at<cv::Vec3b>(u, v)[0];
                polar_pt.g = image.at<cv::Vec3b>(u, v)[1];
                polar_pt.r = image.at<cv::Vec3b>(u, v)[2];
                fisheye_polar_cloud->points.push_back(polar_pt);

                /** point cloud with origin pixel coordinates **/
                pixel_pt.x = u;
                pixel_pt.y = v;
                pixel_pt.z = 0;
                pixel_pt.b = image.at<cv::Vec3b>(u, v)[0];
                pixel_pt.g = image.at<cv::Vec3b>(u, v)[1];
                pixel_pt.r = image.at<cv::Vec3b>(u, v)[2];
                fisheye_pixel_cloud->points.push_back(pixel_pt);
                if (theta > theta_max) { theta_max = theta; }
                else if (theta < theta_min) { theta_min = theta; }
            }
        }
    }

    cout << "Fisheye polar cloud generated. \ntheta_min = " << theta_min << " theta_max = " << theta_max << endl;

    std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> result;
    result = std::make_tuple(fisheye_pixel_cloud, fisheye_polar_cloud);
    return result;
}

void FisheyeProcess::SphereToPlane(CloudRGB::Ptr polar_cloud) {
    cout << "----- Fisheye: SphereToPlane2 -----" << " Spot Index: " << this->spot_idx << endl;
    SphereToPlane(polar_cloud, -1.0);
}

void FisheyeProcess::SphereToPlane(CloudRGB::Ptr polar_cloud, double bandwidth) {
    cout << "----- Fisheye: SphereToPlane -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    cv::Mat flat_image = cv::Mat::zeros(kFlatRows, kFlatCols, CV_8UC3); // define the flat image

    vector<vector<Tags>> tags_map(kFlatRows, vector<Tags>(kFlatCols));

    // define the variables of KDTree search
    pcl::KdTreeFLANN<PointRGB> kdtree;
    kdtree.setInputCloud(polar_cloud);
    pcl::KdTreeFLANN<PointRGB> kdtree2;
    kdtree2.setInputCloud(polar_cloud);

    int invalid_search = 0;
    int invalid_index = 0;
    double search_radius = kRadPerPix / 2;

    /** Multiprocessing test **/
    #pragma omp parallel for num_threads(16)

    // use KDTree to search the spherical point cloud
    for (int u = 0; u < kFlatRows; ++u) {
        // upper bound and lower bound of the current theta unit
        float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;
        for (int v = 0; v < kFlatCols; ++v) {
            // upper bound and lower bound of the current phi unit
            float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;
            // assign the theta and phi center to the searchPoint
            PointRGB search_point;
            search_point.x = theta_center;
            search_point.y = phi_center;
            search_point.z = 0;
            // define the vector container for storing the info of searched points
            std::vector<int> search_idx;
            std::vector<float> dist_L2;
            // radius search
            int num_RNN = kdtree.radiusSearch(search_point, search_radius, search_idx, dist_L2); // number of the radius nearest neighbors
            float scale = sqrt(2);
            // if the corresponding points are found in the radius neighborhood
            if (num_RNN == 0) {
                // assign the theta and phi center to the searchPoint
                PointRGB searchPoint;
                searchPoint.x = theta_center;
                searchPoint.y = phi_center;
                searchPoint.z = 0;
                std::vector<int> search_idx_expand;
                std::vector<float> dist_L2_expand;
                int numSecondSearch = 0;
                scale = (2.0f - ((float)u / kFlatRows)) * sqrt(2);
                while (numSecondSearch == 0) {
                    scale = scale + 0.1f;
                    numSecondSearch = kdtree2.radiusSearch(searchPoint, scale * search_radius, search_idx_expand, dist_L2_expand);
                    if (scale > 3) {
                        flat_image.at<cv::Vec3b>(u, v)[0] = 0; // b
                        flat_image.at<cv::Vec3b>(u, v)[1] = 0; // g
                        flat_image.at<cv::Vec3b>(u, v)[2] = 0; // r
                        invalid_search += 1;
                        tags_map[u][v].pts_indices = {0};
                        break;
                    }
                }
                if (numSecondSearch != 0) {
                    int B = 0, G = 0, R = 0; // mean value of RGB channels
                    for (int &idx : search_idx_expand) {
                        B = B + polar_cloud->points[idx].b;
                        G = G + polar_cloud->points[idx].g;
                        R = R + polar_cloud->points[idx].r;
                    }
                    tags_map[u][v].pts_indices = search_idx_expand;
                    flat_image.at<cv::Vec3b>(u, v)[0] = int(B / numSecondSearch); // b
                    flat_image.at<cv::Vec3b>(u, v)[1] = int(G / numSecondSearch); // g
                    flat_image.at<cv::Vec3b>(u, v)[2] = int(R / numSecondSearch); // r
                }
            }
            else {
                int B = 0, G = 0, R = 0; // mean value of RGB channels
                for (int &idx : search_idx) {
                    B = B + polar_cloud->points[idx].b;
                    G = G + polar_cloud->points[idx].g;
                    R = R + polar_cloud->points[idx].r;
                }
                tags_map[u][v].pts_indices = search_idx;
                flat_image.at<cv::Vec3b>(u, v)[0] = int(B / num_RNN); // b
                flat_image.at<cv::Vec3b>(u, v)[1] = int(G / num_RNN); // g
                flat_image.at<cv::Vec3b>(u, v)[2] = int(R / num_RNN); // r
            }
        }
    }
    this->tags_map_vec[this->spot_idx][this->view_idx] = tags_map;
    cout << "number of invalid searches:" << invalid_search << endl;
    cout << "number of invalid indices:" << invalid_index << endl;

    string flat_img_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].flat_img_path;
    string result_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fusion_result_folder_path;

    /********* Image Generation *********/
    if (bandwidth < 0) {
        cv::imwrite(flat_img_path, flat_image); /** flat image generation **/
    }
    else {
        string fusion_img_path = result_path + "/spot_" + to_string(this->spot_idx) +
                                 "_fusion_bw_" + to_string(int(bandwidth)) + ".bmp";
        cv::imwrite(fusion_img_path, flat_image); /** fusion image generation **/
    }
}

void FisheyeProcess::EdgeToPixel() {
    cout << "----- Fisheye: EdgeToPixel -----" << " Spot Index: " << this->spot_idx << endl;
    string edge_img_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0),
                   "size of original fisheye image is 0, check the path and filename! View Index: %d", this->view_idx);
    ROS_ASSERT_MSG((edge_img.rows == this->kFlatRows || edge_img.cols == this->kFlatCols),
                   "size of original fisheye image is incorrect! View Index: %d", this->view_idx);
    EdgePixels edge_pixels;

    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                edge_pixels.push_back(vector<int>{u, v});
            }
        }
    }
    this -> edge_pixels_vec[this->spot_idx][this->view_idx] = edge_pixels;
}

void FisheyeProcess::PixLookUp(CloudRGB::Ptr fisheye_pixel_cloud) {
    cout << "----- Fisheye: PixLookUp -----" << " Spot Index: " << this->spot_idx << endl;
    int invalid_edge_pix = 0;
    EdgeFisheyePixels edge_fisheye_pixels;
    EdgePixels edge_pixels = this->edge_pixels_vec[this->spot_idx][this->view_idx];
    TagsMap tags_map = this->tags_map_vec[this->spot_idx][this->view_idx];
    for (auto &edge_pixel : edge_pixels) {
        int u = round(edge_pixel[0]);
        int v = round(edge_pixel[1]);
        float x = 0;
        float y = 0;

        int num_pts = tags_map[u][v].pts_indices.size();
        if (num_pts == 0) {
            invalid_edge_pix++;
        }
        else {
            for (auto &idx : tags_map[u][v].pts_indices) {
                PointRGB &pt = fisheye_pixel_cloud->points[idx];
                x += pt.x;
                y += pt.y;
            }
            x = x / num_pts;
            y = y / num_pts;
            vector<double> pixel{x, y};
            edge_fisheye_pixels.push_back(pixel);
        }
    }
    this->edge_fisheye_pixels_vec[this->spot_idx][this->view_idx] = edge_fisheye_pixels;
    cout << "number of invalid lookups(image): " << invalid_edge_pix << endl;

    string edge_org_txt_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_fisheye_pixels_path;
    /********* write the coordinates into txt file *********/
    ofstream outfile;
    outfile.open(edge_org_txt_path, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (auto &edge_fisheye_pixel : edge_fisheye_pixels) {
        outfile << edge_fisheye_pixel[0] << "\t" << edge_fisheye_pixel[1] << endl;
    }
    outfile.close();
}

// create static blur image for autodiff ceres optimization
// the "polar" option is implemented but not tested/supported in optimization.
std::vector<double> FisheyeProcess::Kde(double bandwidth, double scale) {
    cout << "----- Fisheye: Kde -----"  << " Spot Index: " << this->spot_idx << endl;
    clock_t start_time = clock();
    const double default_rel_error = 0.05;
    const int n_rows = scale * this->kFisheyeRows;
    const int n_cols = scale * this->kFisheyeCols;
    arma::mat query;
    // number of rows equal to number of dimensions, query.n_rows == reference.n_rows is required
    EdgeFisheyePixels edge_fisheye_pixels = this -> edge_fisheye_pixels_vec[this->spot_idx][this->view_idx];
    const int ref_size = edge_fisheye_pixels.size();
    arma::mat reference(2, ref_size);
    for (int i = 0; i < ref_size; ++i) {
        reference(0, i) = edge_fisheye_pixels[i][0];
        reference(1, i) = edge_fisheye_pixels[i][1];
    }

    query = arma::mat(2, n_cols * n_rows);
    arma::vec rows = arma::linspace(0, this->kFisheyeRows - 1, n_rows);
    arma::vec cols = arma::linspace(0, this->kFisheyeCols - 1, n_cols);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            query(0, i * n_cols + j) = rows.at(i);
            query(1, i * n_cols + j) = cols.at(j);
        }
    }

    arma::vec kde_estimations;
    mlpack::kernel::EpanechnikovKernel kernel(bandwidth);
    mlpack::metric::EuclideanDistance metric;
    mlpack::kde::KDE<EpanechnikovKernel, mlpack::metric::EuclideanDistance, arma::mat> kde(default_rel_error, 0.00, kernel);
    kde.Train(std::move(reference));
    kde.Evaluate(query, kde_estimations);

    std::vector<double> img = arma::conv_to<std::vector<double>>::from(kde_estimations);
    // string kde_txt_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].kde_samples_path;
    // ofstream outfile;
    // outfile.open(kde_txt_path, ios::out);
    // if (!outfile.is_open()) {
    //     cout << "Open file failure" << endl;
    // }
    // for (int i = 0; i < n_rows; ++i) {
    //     for (int j = 0; j < n_cols; j++) {
    //         int index = i * n_cols + j;
    //         outfile << query.at(0, index) << "\t"
    //                 << query.at(1, index) << "\t"
    //                 << kde_estimations(index) << endl;
    //     }
    // }
    // outfile.close();
    cout << "New kde image generated with size (" << n_rows << ", " << n_cols << ") in "
         <<(double)(clock() - start_time) / CLOCKS_PER_SEC << "s, bandwidth = " << bandwidth << endl;
    return img;
}

void FisheyeProcess::EdgeExtraction() {
    std::string script_path = this->kPkgPath + "/python_scripts/image_process/edge_extraction.py";
    std::string kSpots = to_string(this->spot_idx);
    std::string cmd_str = "python3 " 
        + script_path + " " + this->kDatasetPath + " " + "fisheye" + " " + kSpots;
    system(cmd_str.c_str());
}
