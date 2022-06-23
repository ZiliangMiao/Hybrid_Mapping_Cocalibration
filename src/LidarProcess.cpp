// basic
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <numeric>
// ros
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Header.h>
#include <ros/package.h>
// pcl
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
// opencv
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
// include mlpack
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
// heading
#include "LidarProcess.h"

using namespace std;
using namespace cv;

// using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;

using namespace arma;
using namespace Eigen;

LidarProcess::LidarProcess(string pkgPath, bool byIntensity)
{
    cout << "----- LiDAR: LidarProcess -----" << endl;
    this -> num_scenes = 5;
    /** reserve the memory for vectors stated in LidarProcess.h **/
    this -> scenes_files_path_vec.reserve(this -> num_scenes);
    this -> edge_pixels_vec.reserve(this -> num_scenes);
    this -> edge_cloud_vec.reserve(this -> num_scenes);
    this -> edge_pts_vec.reserve(this -> num_scenes);
    this -> tags_map_vec.reserve(this -> num_scenes);

    /** push the data directory path into vector **/
    this -> scenes_path_vec.push_back(pkgPath + "/data/runYangIn");
    this -> scenes_path_vec.push_back(pkgPath + "/data/huiyuan2");
    this -> scenes_path_vec.push_back(pkgPath + "/data/12");
    this -> scenes_path_vec.push_back(pkgPath + "/data/conferenceF2-P1");
    this -> scenes_path_vec.push_back(pkgPath + "/data/conferenceF2-P2");

    /** define the initial projection mode - by intensity or by depth **/
    this -> byIntensity = byIntensity;

    for (int idx = 0; idx < num_scenes; ++idx) {
        struct SceneFilePath sc(scenes_path_vec[idx]);
        this -> scenes_files_path_vec.push_back(sc);
    }
    cout << endl;
}

std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> LidarProcess::LidarToSphere() {
    cout << "----- LiDAR: LidarToSphere -----" << endl;
    double X, Y, Z;
    double radius;
    double theta, phi;
    double projProp;
    double thetaMin = M_PI, thetaMax = -M_PI;

    bool byIntensity = this->byIntensity;
    string lidDensePcdPath = this -> scenes_files_path_vec[this -> scene_idx].dense_pcd_path;
    // original cartesian point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr lid3dOrg(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(lidDensePcdPath, *lid3dOrg);

    // check the original point cloud size
    int cloudSizeOrg = lid3dOrg->points.size();
    cout << "size of original cloud:" << cloudSizeOrg << endl;

    /********* PCL Filter - Pass Through *********/
    pcl::PassThrough<pcl::PointXYZI> disFlt;
    disFlt.setFilterFieldName("x");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lid3dOrg);
    disFlt.filter(*lid3dOrg);
    pcl::PassThrough<pcl::PointXYZI> disFlt1;
    disFlt.setFilterFieldName("y");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lid3dOrg);
    disFlt.filter(*lid3dOrg);

    // check the pass through filtered point cloud size
    int cloudSizeIntFlt = lid3dOrg->points.size();
    cout << "size of cloud after a pass through filter:" << cloudSizeIntFlt << endl;

    // statistical filter: outlier removal
    // pcl::PointCloud<pcl::PointXYZI>::Ptr lidStaFlt(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZI> outlierFlt;
    // outlierFlt.setInputCloud(lidIntFlt);
    // outlierFlt.setMeanK(20);
    // outlierFlt.setStddevMulThresh(0.8);
    // outlierFlt.setNegative(false);
    // outlierFlt.filter(*lidStaFlt);

    // filter: radius outlier removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidStaFlt(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlierFlt;
    outlierFlt.setInputCloud(lid3dOrg);
    outlierFlt.setRadiusSearch(0.1);
    outlierFlt.setMinNeighborsInRadius(5);
    outlierFlt.setNegative(false);
    outlierFlt.filter(*lidStaFlt);

    // new container
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI pt_polar;
    pcl::PointXYZI pt_cartesian;

    // cloud size check
    int cloudSizeStaFlt = lidStaFlt->points.size();
    cout << "statistically filtered cloud size:" << cloudSizeStaFlt << endl;

    for (int i = 0; i < cloudSizeStaFlt; i++) {
        // assign the cartesian coordinate to pcl point cloud
        X = lidStaFlt->points[i].x;
        Y = lidStaFlt->points[i].y;
        Z = lidStaFlt->points[i].z;
        projProp = lidStaFlt->points[i].intensity;
        if (!byIntensity) {
            radius = projProp;
        }
        else {
            radius = sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2));
        }

        // assign the polar coordinate to pcl point cloud
        phi = M_PI - atan2(Y, X);
        theta = acos(Z / radius);
        pt_polar.x = theta;
        pt_polar.y = phi;
        pt_polar.z = 0;
        pt_polar.intensity = projProp;
        lidPolar->points.push_back(pt_polar);

        if (theta > thetaMax) {
            thetaMax = theta;
        }
        if (theta < thetaMin) {
            thetaMin = theta;
        }
    }

    // output the important features of the point cloud
    cout << "polar cloud size:" << lidPolar->points.size() << endl;

    // save to pcd files and create tuple return
    string polarPcdPath = this -> scenes_files_path_vec[this -> scene_idx].polar_pcd_path;
    string cartPcdPath = this -> scenes_files_path_vec[this -> scene_idx].cart_pcd_path;
    pcl::io::savePCDFileBinary(cartPcdPath, *lidStaFlt);
    pcl::io::savePCDFileBinary(polarPcdPath, *lidPolar);
    std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> result;
    result = std::make_tuple(lidPolar, lidStaFlt);
    cout << endl;
    return result;
}

void LidarProcess::SphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr polar_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cart_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << endl;
    double flatRows = this -> flatRows;
    double flatCols = this -> flatCols;

    /** define the data container **/
    cv::Mat flatImage = cv::Mat::zeros(flatRows, flatCols, CV_32FC1); /** define the flat image **/
    vector<vector<Tags>> tagsMap (flatRows, vector<Tags>(flatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be setted before the loop **/
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(polar_cloud);

    /** define the invalid search parameters **/
    int invalidSearch = 0; /** search invalid count **/
    int invalidIndex = 0; /** index invalid count **/
    double radPerPix = this -> radPerPix;
    double scale = 2;
    double searchRadius = scale * (radPerPix / 2);

    /** std range to generate weights **/
    double stdMax = 0;
    double stdMin = 1;

    for (int u = 0; u < flatRows; ++u) {
        float theta_lb = u * radPerPix;
        float theta_ub = (u + 1) * radPerPix;
        float theta_center = (theta_ub + theta_lb) / 2;

        for (int v = 0; v < flatCols; ++v) {
            float phi_lb = v * radPerPix;       // upper bound of the current phi unit
            float phi_ub = (v + 1) * radPerPix; // upper bound of the current phi unit
            float phi_center = (phi_ub + phi_lb) / 2;

            // assign the theta and phi center to the searchPoint
            pcl::PointXYZI searchPoint;
            searchPoint.x = theta_center;
            searchPoint.y = phi_center;
            searchPoint.z = 0;

            // define the vector container for storing the info of searched points
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance; // type of distance vector has to be float
            // use kdtree to search (radius search) the spherical point cloud
            int numRNN = kdtree.radiusSearch(searchPoint, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance); // number of the radius nearest neighbors
            if (numRNN == 0) {
                flatImage.at<float>(u, v) = 160; // intensity
                invalidSearch = invalidSearch + 1;
                // add tags
                tagsMap[u][v].label = 0;
                tagsMap[u][v].num_pts = 0;
                tagsMap[u][v].pts_indices.push_back(0);
                tagsMap[u][v].mean = 0;
                tagsMap[u][v].sigma = 0;
                tagsMap[u][v].weight = 0;
            }
            else { /** corresponding points are found in the radius neighborhood **/
                vector<double> Intensity;
                vector<double> Theta;
                vector<double> Phi;
                for (int i = 0; i < numRNN; ++i) {
                    if (pointIdxRadiusSearch[i] > polar_cloud->points.size() - 1) {
                        // caution: a bug is hidden here, index of the searched point is bigger than size of the whole point cloud
                        flatImage.at<float>(u, v) = 160; // intensity
                        invalidIndex = invalidIndex + 1;
                        continue;
                    }
                    Intensity.push_back((*polar_cloud)[pointIdxRadiusSearch[i]].intensity);
                    Theta.push_back((*polar_cloud)[pointIdxRadiusSearch[i]].x);
                    Phi.push_back((*polar_cloud)[pointIdxRadiusSearch[i]].y);
                    /** add tags **/
                    tagsMap[u][v].num_pts = numRNN;
                    tagsMap[u][v].pts_indices.push_back(pointIdxRadiusSearch[i]);
                }

                int numHidden = 0;
                /***** Hidden Points Filter *****/
                for (int i = 0; i < numRNN; ++i) {
                    if (i > 0 && i < (numRNN - 1)) {

                        pcl::PointXYZI pt = (*cart_cloud)[tagsMap[u][v].pts_indices[i]];
                        pcl::PointXYZI pt_former = (*cart_cloud)[tagsMap[u][v].pts_indices[i - 1]];
                        pcl::PointXYZI pt_latter = (*cart_cloud)[tagsMap[u][v].pts_indices[i + 1]];

                        float disFormer = sqrt(pt_former.x * pt_former.x + pt_former.y * pt_former.y + pt_former.z * pt_former.z);
                        float dis = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                        float disLatter = sqrt(pt_latter.x * pt_latter.x + pt_latter.y * pt_latter.y + pt_latter.z * pt_latter.z);

                        if (disFormer > disLatter) {
                            float neighborDiffXF = pt.x - pt_former.x;
                            float neighborDiffYF = pt.y - pt_former.y;
                            float neighborDiffZF = pt.z - pt_former.z;
                            float neighborDiffFormer = sqrt(neighborDiffXF * neighborDiffXF + neighborDiffYF * neighborDiffYF + neighborDiffZF * neighborDiffZF);
                            if (neighborDiffFormer > 0.1 * dis && dis > disFormer) {
                                /** Erase the hidden points **/
                                vector<double>::iterator intensity_iter = Intensity.begin() + i;
                                Intensity.erase(intensity_iter);
                                vector<double>::iterator theta_iter = Theta.begin() + i;
                                Theta.erase(theta_iter);
                                vector<double>::iterator phi_iter = Phi.begin() + i;
                                Phi.erase(phi_iter);
                                vector<int>::iterator idx_iter = tagsMap[u][v].pts_indices.begin() + i;
                                tagsMap[u][v].pts_indices.erase(idx_iter);
                                tagsMap[u][v].num_pts = tagsMap[u][v].num_pts - 1;
                                numHidden++;
                            }
                        }
                        else {
                            float neighborDiffXL = pt_latter.x - pt.x;
                            float neighborDiffYL = pt_latter.y - pt.y;
                            float neighborDiffZL = pt_latter.z - pt.z;
                            float neighborDiffLatter = sqrt(neighborDiffXL * neighborDiffXL + neighborDiffYL * neighborDiffYL + neighborDiffZL * neighborDiffZL);
                            if (neighborDiffLatter > 0.1 * dis && dis > disLatter) {
                                /** Erase the hidden points **/
                                vector<double>::iterator intensity_iter = Intensity.begin() + i;
                                Intensity.erase(intensity_iter);
                                vector<double>::iterator theta_iter = Theta.begin() + i;
                                Theta.erase(theta_iter);
                                vector<double>::iterator phi_iter = Phi.begin() + i;
                                Phi.erase(phi_iter);
                                vector<int>::iterator idx_iter = tagsMap[u][v].pts_indices.begin() + i;
                                tagsMap[u][v].pts_indices.erase(idx_iter);
                                tagsMap[u][v].num_pts = tagsMap[u][v].num_pts - 1;
                                numHidden++;
                            }
                        }
                    }
                }
                tagsMap[u][v].num_hidden_pts = numHidden;

                /** Check the size of vectors **/
                ROS_ASSERT_MSG((Theta.size() == Phi.size()) && (Phi.size() == Intensity.size()) && (Intensity.size() == tagsMap[u][v].pts_indices.size()) && (tagsMap[u][v].pts_indices.size() == tagsMap[u][v].num_pts), "size of the vectors in a pixel region is not the same!");

                if (tagsMap[u][v].num_pts == 1) {
                    /** only one point in the theta-phi sub-region of a pixel **/
                    tagsMap[u][v].label = 1;
                    tagsMap[u][v].mean = 0;
                    tagsMap[u][v].sigma = 0;
                    tagsMap[u][v].weight = 1;
                    double intensityMean = Intensity[0];
                    flatImage.at<float>(u, v) = intensityMean;
                }
                else if (tagsMap[u][v].num_pts == 0) {
                    /** no points in a pixel **/
                    tagsMap[u][v].label = 0;
                    tagsMap[u][v].mean = 99;
                    tagsMap[u][v].sigma = 99;
                    tagsMap[u][v].weight = 0;
                    flatImage.at<float>(u, v) = 160;
                }
                else if (tagsMap[u][v].num_pts >= 2) {
                    /** Gaussian Distribution Parameters Estimation **/
                    double thetaMean = std::accumulate(std::begin(Theta), std::end(Theta), 0.0) / tagsMap[u][v].num_pts; /** central position calculation **/
                    double phiMean = std::accumulate(std::begin(Phi), std::end(Phi), 0.0) / tagsMap[u][v].num_pts;
                    vector<double> Distance(tagsMap[u][v].num_pts);
                    double distance = 0.0;
                    for (int i = 0; i < tagsMap[u][v].num_pts; i++) {
                        if ((Theta[i] > thetaMean && Phi[i] >= phiMean) || (Theta[i] < thetaMean && Phi[i] <= phiMean)) {
                            /** consider these two conditions as positive distance **/
                            distance = sqrt(pow((Theta[i] - thetaMean), 2) + pow((Phi[i] - phiMean), 2));
                            Distance[i] = distance;
                        }
                        else if ((Theta[i] >= thetaMean && Phi[i] < phiMean) || (Theta[i] <= thetaMean && Phi[i] > phiMean)) {
                            /** consider these two conditions as negative distance **/
                            distance = - sqrt(pow((Theta[i] - thetaMean), 2) + pow((Phi[i] - phiMean), 2));
                            Distance[i] = distance;
                        }
                        else if (Theta[i] == thetaMean && Phi[i] == phiMean) {
                            Distance[i] = 0;
                        }
                    }

                    double distanceMean = std::accumulate(std::begin(Distance), std::end(Distance), 0.0) / tagsMap[u][v].num_pts;
                    double distanceVar = 0.0;
                    std::for_each (std::begin(Distance), std::end(Distance), [&](const double distance) {
                        distanceVar += (distance - distanceMean) * (distance - distanceMean);
                    });
                    distanceVar = distanceVar / tagsMap[u][v].num_pts;
                    double distanceStd = sqrt(distanceVar);

                    if (distanceStd > stdMax) {
                        stdMax = distanceStd;
                    }
                    else if (distanceStd < stdMin) {
                        stdMin = distanceStd;
                    }

                    tagsMap[u][v].label = 1;
                    tagsMap[u][v].mean = distanceMean;
                    tagsMap[u][v].sigma = distanceStd;
                    double intensityMean = std::accumulate(std::begin(Intensity), std::end(Intensity), 0.0) / tagsMap[u][v].num_pts;
                    flatImage.at<float>(u, v) = intensityMean;
                }
            }
        }
    }

    double weightMax = 1;
    double weightMin = 0.7;
    cout << stdMin << " " << stdMax << endl;
    for (int u = 0; u < flatRows; ++u) {
        for (int v = 0; v < flatCols; ++v) {
            if (tagsMap[u][v].num_pts > 1) {
                tagsMap[u][v].weight = (stdMax - tagsMap[u][v].sigma) / (stdMax - stdMin) * (weightMax - weightMin) + weightMin;
            }
        }
    }

    /** add the tags_map of this specific scene to maps **/
    this -> tags_map_vec.push_back(tagsMap);
    string tagsMapTxtPath = this -> scenes_files_path_vec[this -> scene_idx].tags_map_path;
    ofstream outfile;
    outfile.open(tagsMapTxtPath, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }


    for (int u = 0; u < flatRows; ++u) {
        for (int v = 0; v < flatCols; ++v) {
            bool idxPrint = false;
            if (idxPrint) {
                for (int k = 0; k < tagsMap[u][v].pts_indices.size(); ++k) {
                    /** k is the number of lidar points that the [u][v] pixel contains **/
                    if (k == tagsMap[u][v].pts_indices.size() - 1) {
                        cout << tagsMap[u][v].pts_indices[k] << endl;
                        outfile << tagsMap[u][v].pts_indices[k] << "\t" << "*****" << "\t" << tagsMap[u][v].pts_indices.size() << endl;
                    }
                    else {
                        cout << tagsMap[u][v].pts_indices[k] << endl;
                        outfile << tagsMap[u][v].pts_indices[k] << "\t";
                    }
                }
            }
            outfile << "Lable: " << tagsMap[u][v].label << "\t" << "size: " << tagsMap[u][v].num_pts << "\t" << "weight: " << tagsMap[u][v].weight << endl;
        }
    }
    outfile.close();

    cout << "number of invalid searches:" << invalidSearch << endl;
    cout << "number of invalid indices:" << invalidIndex << endl;
    string flatImgPath = this -> scenes_files_path_vec[this -> scene_idx].flat_img_path;
    cout << "LiDAR flat image path: " << flatImgPath << endl;
    cv::imwrite(flatImgPath, flatImage);
    cout << "LiDAR flat image generated successfully! Scene Index: " << this -> scene_idx << endl;
    cout << endl;
}

void LidarProcess::EdgeToPixel() {
    /** generate edge_pixels and push back into edge_pixels_vec **/
    cout << "----- LiDAR: EdgeToPixel -----" << endl;
    string edgeImgPath = this -> scenes_files_path_vec[this -> scene_idx].edge_img_path;
    cv::Mat edgeImage = cv::imread(edgeImgPath, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG(((edgeImage.rows != 0 && edgeImage.cols != 0) || (edgeImage.rows < 16384 || edgeImage.cols < 16384)), "size of original fisheye image is 0, check the path and filename! Scene Index: %d", this -> num_scenes);
    ROS_ASSERT_MSG((edgeImage.rows == this->flatRows || edgeImage.cols == this->flatCols), "size of original fisheye image is incorrect! Scene Index: %d", this -> num_scenes);

    EdgePixels edge_pixels;
    for (int u = 0; u < edgeImage.rows; ++u) {
        for (int v = 0; v < edgeImage.cols; ++v) {
            if (edgeImage.at<uchar>(u, v) > 127) {
                vector<int> pixel{u, v};
                edge_pixels.push_back(pixel);
            }
        }
    }
    this -> edge_pixels_vec.push_back(edge_pixels);

    /***** Write the Pixel Coordinates into .txt File *****/
    string edgeTxtPath = this -> scenes_files_path_vec[this -> scene_idx].edge_flat_pixels_path;
    ofstream outfile;
    outfile.open(edgeTxtPath, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < edge_pixels.size(); ++i) {
        outfile << edge_pixels[i][0] << "\t" << edge_pixels[i][1] << endl;
    }
    outfile.close();
    cout << endl;
}

void LidarProcess::EdgePixCheck() {
    cout << "----- LiDAR: EdgePixCheck -----" << endl;
    int flatCols = this->flatCols;
    int flatRows = this->flatRows;
    cv::Mat edge_check_img = cv::Mat::zeros(flatRows, flatCols, CV_8UC1);
    EdgePixels edge_pixels = this -> edge_pixels_vec[this -> scene_idx];
    for (int i = 0; i < edge_pixels.size(); ++i)
    {
        int u = edge_pixels[i][0];
        int v = edge_pixels[i][1];
        edge_check_img.at<uchar>(u, v) = 255;
    }
    string edge_check_img_path = this -> scenes_files_path_vec[this -> scene_idx].EdgeCheckImgPath;
    cv::imwrite(edge_check_img_path, edge_check_img);
    cout << endl;
}

void LidarProcess::PixLookUp(pcl::PointCloud<pcl::PointXYZI>::Ptr cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << endl;
    int invalid_pixel_space = 0;
    EdgePixels edge_pixels = this -> edge_pixels_vec[this -> scene_idx];
    TagsMap tags_map = this -> tags_map_vec[this -> scene_idx];
    EdgePts edge_pts;
    EdgeCloud edge_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < edge_pixels.size(); ++i) {
        int u = edge_pixels[i][0];
        int v = edge_pixels[i][1];
        int num_pts = tags_map[u][v].num_pts;

        if (tags_map[u][v].label == 0) { /** invalid pixels **/
            invalid_pixel_space = invalid_pixel_space + 1;
            float x = 0, y = 0, z = 0;
            continue;
        }
        else { /** normal pixels **/
            /** center of lidar edge distribution **/
            float x = 0, y = 0, z = 0;
            for (int j = 0; j < num_pts; ++j) {
                int idx = tags_map[u][v].pts_indices[j];
                pcl::PointXYZI pt = (*cart_cloud)[idx];
                x = x + pt.x;
                y = y + pt.y;
                z = z + pt.z;
            }
            /** average coordinates -> unbiased estimation of center position **/
            x = x / num_pts;
            y = y / num_pts;
            z = z / num_pts;
            double weight = tags_map[u][v].weight;
            /** store the spatial coordinates into vector **/
            vector<double> coordinates {x, y, z};
            edge_pts.push_back(coordinates);

            /** store the spatial coordinates into vector **/
            pcl::PointXYZI pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            pt.intensity = weight; /** note: I is used to store the point weight **/
            edge_cloud -> points.push_back(pt);
        }
    }

    cout << "number of invalid lookups(lidar): " << invalid_pixel_space << endl;
    this -> edge_pts_vec.push_back(edge_pts);
    this -> edge_cloud_vec.push_back(edge_cloud);

    /** write the coordinates and weights into .txt file **/
    string edgeOrgTxtPath = this -> scenes_files_path_vec[this -> scene_idx].edge_points_coordinates_path;
    ofstream outfile;
    outfile.open(edgeOrgTxtPath, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }

    for (int i = 0; i < edge_cloud ->points.size(); ++i) {
        outfile << edge_cloud ->points[i].x
                << "\t" << edge_cloud ->points[i].y
                << "\t" << edge_cloud ->points[i].z
                << "\t" << edge_cloud ->points[i].intensity << endl;
    }
    outfile.close();
    cout << endl;
}

void LidarProcess::ReadEdge() {
    cout << "----- LiDAR: ReadEdge -----" << endl;
    cout << "Scene Index in LiDAR ReadEdge: " << this -> scene_idx << endl;
    string edge_cloud_txt_path = this -> scenes_files_path_vec[this -> scene_idx].edge_points_coordinates_path;
    EdgePts edge_pts;
    ifstream infile(edge_cloud_txt_path);
    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        string tmp;
        vector<double> v;
        while (getline(ss, tmp, '\t')) {
            /** split string with "\t" **/
            v.push_back(stod(tmp)); /** string->double **/
        }
        if (v.size() == 4) {
            edge_pts.push_back(v);
        }
    }

    ROS_ASSERT_MSG(edge_pts.size() != 0, "LiDAR Read Edge Incorrect! Scene Index: %d", this -> num_scenes);
    cout << "Imported LiDAR points: " << edge_pts.size() << endl;
    /** remove dumplicated points **/
    std::sort(edge_pts.begin(), edge_pts.end());
    edge_pts.erase(unique(edge_pts.begin(), edge_pts.end()), edge_pts.end());
    cout << "LiDAR Edge Points after Dumplicated Removed: " << edge_pts.size() << endl;
    this -> edge_pts_vec.push_back(edge_pts);

    /** construct pcl pointcloud **/
    pcl::PointXYZI pt;
    EdgeCloud edge_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = 0; i < edge_pts.size(); i++) {
        pt.x = edge_pts[i][0];
        pt.y = edge_pts[i][1];
        pt.z = edge_pts[i][2];
        pt.intensity = edge_pts[i][3];
        edge_cloud -> points.push_back(pt);
    }
    cout << "Filtered LiDAR points: " << edge_cloud -> points.size() << endl;
    this -> edge_cloud_vec.push_back(edge_cloud);
    cout << endl;
}

/***** Extrinsic and Inverse Intrinsic Transform for Visualization of LiDAR Points in Flat Image *****/
vector<vector<double>> LidarProcess::EdgeCloudProjectToFisheye(vector<double> _p) {
    cout << "----- LiDAR: EdgeCloudProjectToFisheye -----" << endl;
    cout << "Scene Index in EdgeCloudProjectToFisheye:" << this -> scene_idx << endl;

    Eigen::Matrix<double, 3, 1> eulerAngle(_p[0], _p[1], _p[2]);
    Eigen::Matrix<double, 3, 1> t{_p[3], _p[4], _p[5]};
    Eigen::Matrix<double, 2, 1> uv_0{_p[6], _p[7]};
    Eigen::Matrix<double, 6, 1> a_;
    switch (_p.size() - 3) {
        case 10:
            a_ << _p[8], _p[9], _p[10], _p[11], _p[12], double(0);
            break;
        case 9:
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
    Eigen::AngleAxisd xAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitZ()));
    R = zAngle * yAngle * xAngle;

    Eigen::Matrix<double, 3, 1> p_;
    Eigen::Matrix<double, 3, 1> p_trans;
    Eigen::Matrix<double, 2, 1> S;
    Eigen::Matrix<double, 2, 1> p_uv;

    vector<vector<double>> edge_fisheye_projection(2, vector<double>(this->edge_cloud_vec[this->scene_idx]->points.size()));

    for (int i = 0; i < this -> edge_cloud_vec[this -> scene_idx]->points.size(); i++) {
        p_ << this -> edge_cloud_vec[this->scene_idx]->points[i].x, this -> edge_cloud_vec[this->scene_idx]->points[i].y, this -> edge_cloud_vec[this->scene_idx]->points[i].z;
        p_trans = R * p_ + t;
        // phi = atan2(p_trans(1), p_trans(0)) + M_PI;
        theta = acos(p_trans(2) / sqrt(pow(p_trans(0), 2) + pow(p_trans(1), 2) + pow(p_trans(2), 2)));
        inv_r = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        S = {-inv_r * p_trans(0) / r, -inv_r * p_trans(1) / r};
        p_uv = S + uv_0;
        edge_fisheye_projection[0][i] = p_uv(0);
        edge_fisheye_projection[1][i] = p_uv(1);
    }
    cout << endl;
    return edge_fisheye_projection;
}

int LidarProcess::ReadFileList(const std::string &folder_path, std::vector<std::string> &file_list) {
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(folder_path.c_str())) == NULL) {
        return 0;
    }

    int num = 0;
    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        if (name != "." && name != "..") {
            file_list.push_back(name);
            num++;
        }
    }
    closedir(dp);
    cout << "read file list success" << endl;

    return num;
}

void LidarProcess::BagToPcd(string bag_file) {
    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);
    vector<string> topics;
    topics.push_back(string(this->topic_name));
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    rosbag::View::iterator it = view.begin();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; it != view.end(); it++, i++) {
        auto m = *it;
        sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
        pcl_conversions::toPCL(*input, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *intensityCloud);
        string id_str = to_string(i);
        string pcdsPath = this -> scenes_files_path_vec[this -> scene_idx].pcds_folder_path;
        pcl::io::savePCDFileBinary(pcdsPath + "/" + id_str + ".pcd", *intensityCloud);
    }
}

void LidarProcess::CreateDensePcd() {
    pcl::PCDReader reader; /** used for read PCD files **/
    vector <string> nameList;
    string pcdsPath = this -> scenes_files_path_vec[this -> scene_idx].pcds_folder_path;
    ReadFileList(pcdsPath, nameList);
    sort(nameList.begin(),nameList.end()); /** sort file names by order **/

    int groupSize = this -> numPcds; /** number of pcds to be merged **/
    int groupCount = nameList.size() / groupSize;

    // PCL PointCloud pointer. Remember that the pointer need to be given a new space
    pcl::PointCloud<pcl::PointXYZI>::Ptr input(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);
    int outputId = 0;
    int nameLength = groupSize * groupCount;
    auto nameIter = nameList.begin();
    for(int i = 0; i < groupCount; i++) {
        for(int j = 0; j < groupSize; j++) {
            string fileName = pcdsPath + "/" + *nameIter;
            cout << fileName << endl;
            if(reader.read(fileName, *input) < 0) {      // read PCD files, and save PointCloud in the pointer
                PCL_ERROR("File is not exist!");
                system("pause");
            }
            int pointCount = input -> points.size();
            for(int k = 0; k < pointCount; k++) {
                output -> points.push_back(input -> points[k]);
            }
            nameIter++;
        }
        string outputPath = this -> scenes_files_path_vec[this -> scene_idx].output_folder_path;
        pcl::io::savePCDFileBinary(outputPath + "/lidDense" + to_string(groupSize) + ".pcd", *output);
        cout << "create dense file success" << endl;
    }
}

//void LidarProcess::calculateMaxIncidence()
//{
//    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarDenseCloud(new pcl::PointCloud<pcl::PointXYZI>);
//    string lidDensePcdPath = this -> scenes_files_path_vec[this -> scene_idx].dense_pcd_path;
//    pcl::io::loadPCDFile(lidDensePcdPath, *lidarDenseCloud);
//    float theta;
//    float radius;
//    float x, y, z;
//    int lidarCount = lidarDenseCloud->points.size();
//    vector<float> Theta;
//    for (int i = 0; i < lidarCount; i++)
//    {
//        x = lidarDenseCloud->points[i].x;
//        y = lidarDenseCloud->points[i].y;
//        z = lidarDenseCloud->points[i].z;
//        radius = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
//        theta = asin(z / radius);
//        Theta.push_back(theta);
//    }
//    sort(Theta.begin(), Theta.end());
//    int j = 0;
//    for (auto it = Theta.begin(); it != Theta.end(); it++)
//    {
//        if (*it > 0)
//        {
//            j++;
//            cout << *it << endl;
//        }
//        if (j == 1000)
//        {
//            break;
//        }
//    }
//}

//vector<vector<double>> LidarProcess::EdgeTransform() {
//    const double pix2rad = 4000 / (M_PI * 2);
//    // declaration of point clouds and output vector
//    pcl::PointCloud<pcl::PointXYZ>::Ptr _p_l(new pcl::PointCloud<pcl::PointXYZ>());
//    vector<vector<double>> lidEdgePolar(2, vector<double>(this -> edge_cloud -> points.size()));
//
//    // initialize the transformation matrix
//    Eigen::Affine3d transMat = Eigen::Affine3d::Identity();
//
//    // initialize the euler angle and transform it into rotation matrix
//    Eigen::Vector3d eulerAngle(this->extrinsic.rz, this->extrinsic.ry, this->extrinsic.rx);
//    Eigen::AngleAxisd xAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
//    Eigen::AngleAxisd yAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
//    Eigen::AngleAxisd zAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));
//    Eigen::Matrix3d rot;
//    rot = zAngle * yAngle * xAngle;
//    transMat.rotate(rot);
//
//    // initialize the translation vector
//    transMat.translation() << this->extrinsic.tx, this->extrinsic.ty, this->extrinsic.tz;
//
//    // point cloud rigid transformation
//    pcl::transformPointCloud(*this -> edge_cloud, *_p_l, transMat);
//
//    for (int i = 0; i < this -> edge_cloud -> points.size(); i++) {
//        // assign the polar coordinate (theta, phi) to pcl point cloud
//        lidEdgePolar[0][i] = pix2rad * acos(_p_l->points[i].z / sqrt(pow(_p_l->points[i].x, 2) + pow(_p_l->points[i].y, 2) + pow(_p_l->points[i].z, 2)));
//        lidEdgePolar[1][i] = pix2rad * (M_PI - atan2(_p_l->points[i].y, _p_l->points[i].x));
//    }
//
//    return lidEdgePolar;
//}

// void LidarProcess::lidFlatImageShift(){
//     float pixPerDegree = this -> pixelPerDegree;
//     float camThetaMin = this -> camThetaMin;
//     float camThetaMax = this -> camThetaMax;
//     float lidThetaMin = this -> lidThetaMin;
//     float lidThetaMax = this -> lidThetaMax;
//     this -> imFlatRows = int((camThetaMax - camThetaMin) * pixPerDegree); // defined as the height(pixel) of the flat image // caution: the int division need a type coercion to float, otherwise the result would be zero
//     this -> imFlatCols = int(2 * M_PI * pixPerDegree); // defined as the width(pixel) of the flat image
//     int imFlatRows = this -> imFlatRows;
//     int imFlatCols = this -> imFlatCols;
//
//     cv::Mat lidImage = cv::imread(this -> flatLidarFile, cv::IMREAD_UNCHANGED);
//     try{
//         if (lidImage.rows != this->lidFlatRows || lidImage.cols != this->lidFlatCols){
//             throw "Error: size of flat image (LiDAR) is incorrect!";
//         }
//     }catch (const char* msg){
//         cerr << msg << endl;
//         abort();
//     }
//
//     cv::Mat lidImageShift = cv::Mat::zeros(imFlatRows, imFlatCols, CV_8UC1);
//     int lidMin = lidThetaMin * pixPerDegree;
//     for(int i = 0; i < imFlatRows; i++){
//         for(int j = 0; j < imFlatCols; j++){
//             if(i >= lidMin){
//                 lidImageShift.at<uchar>(i, j) = lidImage.at<uchar>(i - lidMin, j);
//             }
//             else {
//                 lidImageShift.at<uchar>(i, j) = 160;
//             }
//         }
//     }
//     cv::imwrite(this -> flatLidarShiftFile, lidImageShift);
// }

// void LidarProcess::sphereToPlaneKNN(std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, float, float> result){
//     //Input: spherical point cloud with RGB info (radius = 1073)
//     pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar;
//     float thetaMin;
//     float thetaMax;
//     std::tie(lidPolar, thetaMin, thetaMax) = result;

//     float pixPerDegree = this -> imageRadius / (M_PI / 2);
//     int flatHeight = int((thetaMax - thetaMin) * pixPerDegree); // defined as the height(pixel) of the flat image
//     // caution: the int division need a type coercion to float, otherwise the result would be zero
//     int flatWidth = int(2 * M_PI * pixPerDegree); // defined as the width(pixel) of the flat image
//     cv::Mat flatImage = cv::Mat::zeros(flatHeight, flatWidth, CV_32FC1); // define the flat image

//     // construct kdtrees and load the point clouds
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
//     kdtree.setInputCloud(lidPolar);
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree2;
//     kdtree2.setInputCloud(lidPolar);

//     // KNN search nnumber
//     int K = 1;

//     // X, Y are the pixel coordinates in the flat image
//     // use KDTree to search the spherical point cloud
//     for(int X = 0; X < flatWidth ; ++X){
//         float degreePerPix = (M_PI / 2) / this -> imageRadius;
//         // upper bound and lower bound of the current theta unit
//         float phi_lb = X * degreePerPix;
//         float phi_ub = (X+1) * degreePerPix;
//         float phi_center = (phi_ub + phi_lb) / 2;

//         for(int Y = 0; Y < flatHeight ; ++Y){
//             // upper bound and lower bound of the current phi unit
//             float theta_lb = thetaMin + Y * degreePerPix;
//             float theta_ub = thetaMin + (Y+1) * degreePerPix;
//             float theta_center = (theta_ub + theta_lb) / 2;

//             // assign the theta and phi center to the searchPoint
//             pcl::PointXYZI searchPoint;
//             searchPoint.x = phi_center;
//             searchPoint.y = theta_center;
//             searchPoint.z = 0;

//             // define the vector container for storing the info of searched points
//             std::vector<int> pointIdxSearch(K);
//             std::vector<float> pointSquaredDistance(K);

//             // k nearest neighbors search
//             kdtree.nearestKSearch(searchPoint, K, pointIdxSearch, pointSquaredDistance); // number of the radius nearest neighbors
//             flatImage.at<float>(Y, X) = (*lidPolar)[pointIdxSearch[0]].intensity; // intensity
//         }
//     }
//     cv::imwrite("flatLidarImageKNN.bmp", flatImage);
// }

// void LidarProcess::pp_callback(const pcl::visualization::PointPickingEvent& event, void *args)
// {
//     struct callback_args * data = (struct callback_args *)args;//点云数据 & 可视化窗口
//     if (event.getPointIndex() == -1)
//         return;
//     PointT current_point;
//     event.getPoint(current_point.x, current_point.y, current_point.z);
//     data->clicked_points_3d->points.push_back(current_point);
//     //Draw clicked points in red:
//     pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);
//     data->viewerPtr->removePointCloud("clicked_points");
//     data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
//     data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
//     std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;

// }

// bool LidarProcess::pointMark(){
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));

//     if (pcl::io::loadPCDFile(this -> lidarDenseFile, *cloud))
//     {
//         std::cerr << "ERROR: Cannot open file " << this -> lidarDenseFile << "! Aborting..." << std::endl;
//         return false;
//     }

//     std::cout << cloud->points.size() << std::endl;

//     cloud_mutex.lock();    // for not overwriting the point cloud

//     // Display pointcloud:
//     viewer->addPointCloud(cloud, "bunny_source");
//     viewer->setCameraPosition(0, 0, -2, 0, -1, 0, 0);

//     // Add point picking callback to viewer:
//     struct callback_args cb_args;
//     PointCloudT::Ptr clicked_points_3d(new PointCloudT);
//     cb_args.clicked_points_3d = clicked_points_3d;
//     cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);

//     viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);

//     std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

//     // Spin until 'Q' is pressed:
//     viewer->spin();
//     std::cout << "done." << std::endl;

//     cloud_mutex.unlock();

//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce(100);
//         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//     }

//     return true;
// }