/** basic **/
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <numeric>
/** ros **/
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
/** pcl **/
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
/** opencv **/
#include <opencv2/opencv.hpp>

/** headings **/
#include "LidarProcess.h"
/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointCloud<PointT>::Ptr CloudPtr;

LidarProcess::LidarProcess(const string& pkg_path) {
    cout << "----- LiDAR: LidarProcess -----" << endl;
    /** create objects, initialization **/
    PoseFilePath pose_file_path_temp;
    string scenes_path_temp;
    EdgePixels edge_pixels_temp;
    CloudPtr edge_cloud_temp;
    EdgePts edge_pts_temp;
    TagsMap tags_map_temp;
    Eigen::Matrix4f pose_trans_mat_temp;
    for (int i = 0; i < num_spots; ++i) {
        vector<PoseFilePath> pose_file_path_vec_temp;
        vector<string> scenes_path_vec_temp;
        vector<EdgePixels> edge_pixels_vec_temp;
        vector<CloudPtr> edge_cloud_vec_temp;
        vector<EdgePts> edge_pts_vec_temp;
        vector<TagsMap> tags_map_vec_temp;
        vector<Eigen::Matrix4f> pose_trans_mat_vec_temp;
        for (int j = 0; j < num_views; ++j) {
            pose_file_path_vec_temp.push_back(pose_file_path_temp);
            scenes_path_vec_temp.push_back(scenes_path_temp);
            edge_pixels_vec_temp.push_back(edge_pixels_temp);
            edge_cloud_vec_temp.push_back(edge_cloud_temp);
            edge_pts_vec_temp.push_back(edge_pts_temp);
            tags_map_vec_temp.push_back(tags_map_temp);
            pose_trans_mat_vec_temp.push_back(pose_trans_mat_temp);
        }
        this->scenes_files_path_vec.push_back(pose_file_path_vec_temp);
        this->scenes_path_vec.push_back(scenes_path_vec_temp);
        this->edge_pixels_vec.push_back(edge_pixels_vec_temp);
        this->edge_cloud_vec.push_back(edge_cloud_vec_temp);
        this->edge_pts_vec.push_back(edge_pts_vec_temp);
        this->tags_map_vec.push_back(tags_map_vec_temp);
        this->pose_trans_mat_vec.push_back(pose_trans_mat_vec_temp);
    }

    /** degree map **/
    this -> fullview_rec_folder_path = pkg_path + "/data/floor5/fullview_rec";

    for (int i = 0; i < this -> num_spots; ++i) {
        for (int j = 0; j < this -> num_views; ++j) {
            int v_degree = -50 + 50 * j;
            this -> degree_map[j] = v_degree;
            this -> scenes_path_vec[i][j] = pkg_path + "/data/floor5/spot" + to_string(i) + "/" + to_string(v_degree);
        }
    }

    for (int i = 0; i < this -> num_spots; ++i) {
        for (int j = 0; j < this -> num_views; ++j) {
            struct PoseFilePath sc(scenes_path_vec[i][j]);
            this -> scenes_files_path_vec[i][j] = sc;
        }
    }
    cout << endl;
}


void LidarProcess::ICP() {
    const bool kIcpViz = false;
    CloudPtr cloud_target_input(new CloudT);
    CloudPtr cloud_source_input(new CloudT);
    CloudPtr cloud_target_filtered(new CloudT); /** source point cloud **/
    CloudPtr cloud_source_filtered(new CloudT); /** target point cloud **/
    CloudPtr cloud_source_initial_trans(new CloudT); /** souce cloud with initial rigid transformation **/
    CloudPtr cloud_icped(new CloudT); /** apply icp result to source point cloud **/

    std::string pkg_path = ros::package::getPath("calibration");
    std::string src_pcd_path = this -> scenes_files_path_vec[this->spot_idx][this->view_idx].icp_pcd_path;
    std::string tgt_pcd_path = this -> scenes_files_path_vec[this->spot_idx][(this->num_views-1)/2].icp_pcd_path;

    /** file loading check **/
    if (pcl::io::loadPCDFile<PointT>(tgt_pcd_path, *cloud_target_input) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "ICP: Loaded " << cloud_target_input->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointT>(src_pcd_path, *cloud_source_input) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "ICP: Loaded " << cloud_source_input->size() << " data points from source file" << endl;

    /** invalid point filter **/
    std::vector<int> mapping_in;
    std::vector<int> mapping_out;
    pcl::removeNaNFromPointCloud(*cloud_target_input, *cloud_target_input, mapping_in);
    pcl::removeNaNFromPointCloud(*cloud_source_input, *cloud_source_input, mapping_out);

    /** condition filter **/
    pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, -0.4)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<PointT> cond_filter;
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(cloud_source_input);
    cond_filter.filter(*cloud_source_filtered);
    cond_filter.setInputCloud(cloud_target_input);
    cond_filter.filter(*cloud_target_filtered);

    /** radius outlier filter **/
    pcl::RadiusOutlierRemoval <PointT> outlier_filter;
    outlier_filter.setRadiusSearch(0.5);
    outlier_filter.setMinNeighborsInRadius(30);
    outlier_filter.setInputCloud(cloud_target_filtered);
    outlier_filter.filter(*cloud_target_filtered);
    outlier_filter.setInputCloud(cloud_source_filtered);
    outlier_filter.filter(*cloud_source_filtered);

    /** initial rigid transformation **/
    Eigen::Affine3f initial_trans = Eigen::Affine3f::Identity();
    int v_degree = this -> degree_map.at(this->view_idx);
    initial_trans.translation() << 0.0, 0.15 * sin(v_degree/(float)180 * M_PI), 0.15 - 0.15 * cos(v_degree/(float)180 * M_PI);
    float rx = 0.0, ry = v_degree/(float)180, rz = 0.0;

    Eigen::Matrix3f rotation_mat;
    rotation_mat = Eigen::AngleAxisf(rx*M_PI, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(ry*M_PI, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(rz*M_PI, Eigen::Vector3f::UnitZ());
    initial_trans.rotate(rotation_mat);
    cout << initial_trans.matrix() << endl;
    Eigen::Matrix4f initial_trans_mat = initial_trans.matrix();
    pcl::transformPointCloud(*cloud_source_filtered, *cloud_source_initial_trans, initial_trans);

    /** original icp **/
    pcl::IterativeClosestPoint <PointT, PointT> icp;
    icp.setMaximumIterations(500);
    icp.setInputSource(cloud_source_filtered);
    icp.setInputTarget(cloud_target_filtered);
    icp.setMaxCorrespondenceDistance(0.2);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(0.01);
    icp.align(*cloud_icped, initial_trans_mat);
    if (icp.hasConverged()) {
        cout << "ICP Converged, View: " << this->view_idx << endl;
        cout << "ICP Fitness Score: " << icp.getFitnessScore() << endl;
        cout << "ICP Fitness Epsilon: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "ICP Transformation Matrix: \n" << icp.getFinalTransformation() << endl;
        /** write mat to txt file **/
        this -> pose_trans_mat_vec[this->spot_idx][this->view_idx] = icp.getFinalTransformation();
        std::ofstream mat_out;
        mat_out.open(this->scenes_files_path_vec[this->spot_idx][this->view_idx].pose_trans_mat_path);
        mat_out << icp.getFinalTransformation() << endl;
        mat_out.close();
        /** write registered point cloud to pcd file **/
        string registered_cloud_path = this->fullview_rec_folder_path +
                                       "/icp_registered_" + to_string(v_degree) + ".pcd";
        pcl::io::savePCDFileBinary(registered_cloud_path, *cloud_icped);
    }
    else {
        PCL_ERROR("ICP has not converged.\n");
    }

    /** visualization **/
    if (kIcpViz) {
        pcl::visualization::PCLVisualizer viewer("ICP demo");
        int v1(0), v2(1); /** create two view point **/
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        float bckgr_gray_level = 0.0;  /** black **/
        float txt_gray_lvl = 1.0 - bckgr_gray_level;

        /** the color of original target cloud is white **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(cloud_target_filtered, (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl);
        viewer.addPointCloud(cloud_target_filtered, cloud_aim_color_h, "cloud_aim_v1", v1);
        viewer.addPointCloud(cloud_target_filtered, cloud_aim_color_h, "cloud_aim_v2", v2);

        /** the color of original source cloud is green **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_in_color_h(cloud_source_filtered, 20, 180, 20);
        viewer.addPointCloud(cloud_source_initial_trans, cloud_in_color_h, "cloud_in_v1", v1);

        /** the color of transformed source cloud with icp result is red **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_icped_color_h(cloud_icped, 180, 20, 20);
        viewer.addPointCloud(cloud_icped, cloud_icped_color_h, "cloud_icped_v2", v2);
    }
}




std::tuple<CloudPtr, CloudPtr> LidarProcess::LidarToSphere() {
    cout << "----- LiDAR: LidarToSphere -----" << endl;
    /** define the initial projection mode - by intensity or by depth **/
    const bool projByIntensity = this->kProjByIntensity;
    float x, y, z;
    float radius;
    float theta, phi;
    float proj_param;

    string dense_source_pcd_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    string dense_target_pcd_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    /** original cartesian point cloud **/
    CloudPtr org_source_cloud(new CloudT);
    CloudPtr org_target_cloud(new CloudT);
    CloudPtr org_cloud(new CloudT);
    pcl::io::loadPCDFile(dense_source_pcd_path, *org_source_cloud);
    pcl::io::loadPCDFile(dense_target_pcd_path, *org_target_cloud);
    *org_cloud = *org_source_cloud + *org_target_cloud;

    /** check the original point cloud size **/
    int org_cloud_size = org_cloud->points.size();
    cout << "size of original cloud:" << org_cloud_size << endl;

    /** condition filter **/
    CloudPtr cond_filtered_cloud(new CloudT);
    pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, -0.4)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<PointT> cond_filter;
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(org_cloud);
    cond_filter.filter(*cond_filtered_cloud);


    /** check the pass through filtered point cloud size **/
    int cond_filtered_cloud_size = cond_filtered_cloud->points.size();
    cout << "size of cloud after a condition filter:" << cond_filtered_cloud_size << endl;

    /** radius outlier filter **/
    CloudPtr radius_outlier_cloud(new CloudT);
    pcl::RadiusOutlierRemoval<PointT> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(cond_filtered_cloud);
    radius_outlier_filter.setRadiusSearch(0.1);
    radius_outlier_filter.setMinNeighborsInRadius(5);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.filter(*radius_outlier_cloud);

    /** radius outlier filter cloud size check **/
    int radius_outlier_cloud_size = radius_outlier_cloud->points.size();
    cout << "radius outlier filtered cloud size:" << radius_outlier_cloud_size << endl;

    /** initial rigid transformation (generated by ICP algorithm) **/
    Eigen::Matrix4f initial_trans_mat = Eigen::Matrix4f::Identity();
    string initial_trans_mat_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].pose_trans_mat_path;
    std::ifstream mat_file;
    mat_file.open(initial_trans_mat_path);
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            mat_file >> initial_trans_mat(j, k);
        }
    }
    mat_file.close();
    pcl::transformPointCloud(*radius_outlier_cloud, *radius_outlier_cloud, initial_trans_mat);

    /** new cloud **/
    CloudPtr polar_cloud(new CloudT);
    PointT polar_pt;
    for (int i = 0; i < radius_outlier_cloud_size; i++) {
        x = radius_outlier_cloud->points[i].x;
        y = radius_outlier_cloud->points[i].y;
        z = radius_outlier_cloud->points[i].z;
        proj_param = radius_outlier_cloud->points[i].intensity;
        if (!projByIntensity) {
            radius = proj_param;
        }
        else {
            radius = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        }

        /** assign the polar coordinate to pcl point cloud **/
        phi = atan2(y, x);
        theta = acos(z / radius);
        polar_pt.x = theta;
        polar_pt.y = phi;
        polar_pt.z = 0;
        polar_pt.intensity = proj_param;
        polar_cloud->points.push_back(polar_pt);
    }
    cout << "polar cloud size:" << polar_cloud->points.size() << endl;

    /** save to pcd files and create tuple return **/
    string polar_pcd_path = this -> scenes_files_path_vec[this->spot_idx][this->view_idx].polar_pcd_path;
    string cart_pcd_path = this -> scenes_files_path_vec[this->spot_idx][this->view_idx].cart_pcd_path;
    pcl::io::savePCDFileBinary(cart_pcd_path, *radius_outlier_cloud);
    pcl::io::savePCDFileBinary(polar_pcd_path, *polar_cloud);
    tuple<CloudPtr, CloudPtr> result;
    result = make_tuple(polar_cloud, radius_outlier_cloud);
    cout << endl;
    return result;
}

void LidarProcess::SphereToPlane(const CloudPtr& polar_cloud, const CloudPtr& cart_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << endl;
    clock_t start_time = clock();
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(kFlatRows, kFlatCols, CV_32FC1); /** define the flat image **/
    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be setted before the loop **/
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(polar_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const int kScale = 2;
    const float kSearchRadius = kScale * (kRadPerPix / 2);

    /** std range to generate weights **/
    float dis_std_max = 0;
    float dis_std_min = 1;

    for (int u = 0; u < kFlatRows; ++u) {
        /** upper and lower bound of the current theta unit **/
        float theta_lb = u * kRadPerPix;
        float theta_ub = (u + 1) * kRadPerPix;
        float theta_center = (theta_ub + theta_lb) / 2;

        for (int v = 0; v < kFlatCols; ++v) {
            /** upper and lower bound of the current phi unit **/
            float phi_lb = M_PI - v * kRadPerPix;
            float phi_ub = M_PI - (v + 1) * kRadPerPix;
            float phi_center = (phi_ub + phi_lb) / 2;

            /** assign the theta and phi center to the search_center **/
            PointT search_center;
            search_center.x = theta_center;
            search_center.y = phi_center;
            search_center.z = 0;

            /** define the vector container for storing the info of searched points **/
            vector<int> search_pt_idx_vec;
            vector<float> search_pt_squared_dis_vec; /** type of distance vector has to be float **/
            /** use kdtree to search (radius search) the spherical point cloud **/
            int search_num = kdtree.radiusSearch(search_center, kSearchRadius, search_pt_idx_vec, search_pt_squared_dis_vec); // number of the radius nearest neighbors
            if (search_num == 0) {
                flat_img.at<float>(u, v) = 160; /** intensity **/
                invalid_search_num = invalid_search_num + 1;
                /** add tags **/
                tags_map[u][v].label = 0;
                tags_map[u][v].num_pts = 0;
                tags_map[u][v].pts_indices.push_back(0);
                tags_map[u][v].mean = 0;
                tags_map[u][v].sigma = 0;
                tags_map[u][v].weight = 0;
                tags_map[u][v].num_hidden_pts = 0;
            }
            else { /** corresponding points are found in the radius neighborhood **/
                vector<double> intensity_vec;
                vector<double> theta_vec;
                vector<double> phi_vec;
                for (int i = 0; i < search_num; ++i) {
                    if (search_pt_idx_vec[i] > polar_cloud->points.size() - 1) {
                        /** caution: a bug is hidden here, index of the searched point is bigger than size of the whole point cloud **/
                        flat_img.at<float>(u, v) = 160; /** intensity **/
                        invalid_idx_num = invalid_idx_num + 1;
                        continue;
                    }
                    intensity_vec.push_back((*polar_cloud)[search_pt_idx_vec[i]].intensity);
                    theta_vec.push_back((*polar_cloud)[search_pt_idx_vec[i]].x);
                    phi_vec.push_back((*polar_cloud)[search_pt_idx_vec[i]].y);
                    /** add tags **/
                    tags_map[u][v].num_pts = search_num;
                    tags_map[u][v].pts_indices.push_back(search_pt_idx_vec[i]);
                }

                /** hidden points filter **/
                int hidden_pt_num = 0;
                const bool kHiddenPtsFilter = false;
                if (kHiddenPtsFilter) {
                    for (int i = 0; i < search_num; ++i) {
                        float dis_former, dis;
                        if (i == 0) {
                            PointT pt = (*cart_cloud)[tags_map[u][v].pts_indices[i]];
                            dis = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
                        }
                        if (i > 0 && i < (search_num - 1)) {
                            PointT pt_former = (*cart_cloud)[tags_map[u][v].pts_indices[i - 1]];
                            PointT pt = (*cart_cloud)[tags_map[u][v].pts_indices[i]];
                            dis_former = dis;
                            dis = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);

                            if (dis > dis_former) {
                                float x_diff_former = pt.x - pt_former.x;
                                float y_diff_former = pt.y - pt_former.y;
                                float z_diff_former = pt.z - pt_former.z;
                                float dis_diff_former = sqrt(x_diff_former * x_diff_former + y_diff_former * y_diff_former + z_diff_former * z_diff_former);
                                if (dis_diff_former > 0.1 * dis) {
                                    /** Erase the hidden points **/
                                    auto intensity_iter = intensity_vec.begin() + i;
                                    intensity_vec.erase(intensity_iter);
                                    auto theta_iter = theta_vec.begin() + i;
                                    theta_vec.erase(theta_iter);
                                    auto phi_iter = phi_vec.begin() + i;
                                    phi_vec.erase(phi_iter);
                                    auto idx_iter = tags_map[u][v].pts_indices.begin() + i;
                                    tags_map[u][v].pts_indices.erase(idx_iter);
                                    tags_map[u][v].num_pts = tags_map[u][v].num_pts - 1;
                                    hidden_pt_num ++;
                                }
                            }
                        }
                    }
                }

                /** check the size of vectors **/
                ROS_ASSERT_MSG((theta_vec.size() == phi_vec.size()) && (phi_vec.size() == intensity_vec.size()) && (intensity_vec.size() == tags_map[u][v].pts_indices.size()) && (tags_map[u][v].pts_indices.size() == tags_map[u][v].num_pts), "size of the vectors in a pixel region is not the same!");
                if (hidden_pt_num != 0) {
                    cout << "hidden points: " << hidden_pt_num << "/" << theta_vec.size() << endl;
                }
                if (tags_map[u][v].num_pts == 1) {
                    /** only one point in the theta-phi sub-region of a pixel **/
                    tags_map[u][v].label = 1;
                    tags_map[u][v].mean = 0;
                    tags_map[u][v].sigma = 0;
                    tags_map[u][v].weight = 1;
                    tags_map[u][v].num_hidden_pts = hidden_pt_num;
                    double intensity_mean = intensity_vec[0];
                    flat_img.at<float>(u, v) = intensity_mean;
                }
                else if (tags_map[u][v].num_pts == 0) {
                    /** no points in a pixel **/
                    tags_map[u][v].label = 0;
                    tags_map[u][v].mean = 99;
                    tags_map[u][v].sigma = 99;
                    tags_map[u][v].weight = 0;
                    tags_map[u][v].num_hidden_pts = hidden_pt_num;
                    flat_img.at<float>(u, v) = 160;
                }
                else if (tags_map[u][v].num_pts >= 2) {
                    /** Gaussian Distribution Parameters Estimation **/
                    double theta_mean = accumulate(std::begin(theta_vec), std::end(theta_vec), 0.0) / tags_map[u][v].num_pts; /** central position calculation **/
                    double phi_mean = accumulate(std::begin(phi_vec), std::end(phi_vec), 0.0) / tags_map[u][v].num_pts;
                    vector<double> dis_vec(tags_map[u][v].num_pts);
                    double distance = 0.0;
                    for (int i = 0; i < tags_map[u][v].num_pts; i++) {
                        if ((theta_vec[i] > theta_mean && phi_vec[i] >= phi_mean) || (theta_vec[i] < theta_mean && phi_vec[i] <= phi_mean)) {
                            /** consider these two conditions as positive distance **/
                            distance = sqrt(pow((theta_vec[i] - theta_mean), 2) + pow((phi_vec[i] - phi_mean), 2));
                            dis_vec[i] = distance;
                        }
                        else if ((theta_vec[i] >= theta_mean && phi_vec[i] < phi_mean) || (theta_vec[i] <= theta_mean && phi_vec[i] > phi_mean)) {
                            /** consider these two conditions as negative distance **/
                            distance = - sqrt(pow((theta_vec[i] - theta_mean), 2) + pow((phi_vec[i] - phi_mean), 2));
                            dis_vec[i] = distance;
                        }
                        else if (theta_vec[i] == theta_mean && phi_vec[i] == phi_mean) {
                            dis_vec[i] = 0;
                        }
                    }

                    float dis_mean = accumulate(std::begin(dis_vec), std::end(dis_vec), 0.0) / tags_map[u][v].num_pts;
                    float dis_var = 0.0;
                    std::for_each (std::begin(dis_vec), std::end(dis_vec), [&](const double distance) {
                        dis_var += (distance - dis_mean) * (distance - dis_mean);
                    });
                    dis_var = dis_var / tags_map[u][v].num_pts;
                    float dis_std = sqrt(dis_var);
                    if (dis_std > dis_std_max) {
                        dis_std_max = dis_std;
                    }
                    else if (dis_std < dis_std_min) {
                        dis_std_min = dis_std;
                    }

                    tags_map[u][v].label = 1;
                    tags_map[u][v].mean = dis_mean;
                    tags_map[u][v].sigma = dis_std;
                    tags_map[u][v].num_hidden_pts = hidden_pt_num;
                    double intensity_mean = accumulate(std::begin(intensity_vec), std::end(intensity_vec), 0.0) / tags_map[u][v].num_pts;
                    flat_img.at<float>(u, v) = intensity_mean;
                }
            }
        }
    }

    double weight_max = 1;
    double weight_min = 0.7;
    for (int u = 0; u < kFlatRows; ++u) {
        for (int v = 0; v < kFlatCols; ++v) {
            if (tags_map[u][v].num_pts > 1) {
                tags_map[u][v].weight = (dis_std_max - tags_map[u][v].sigma) / (dis_std_max - dis_std_min) * (weight_max - weight_min) + weight_min;
            }
        }
    }

    /** add the tags_map of this specific scene to maps **/
    this->tags_map_vec[this->spot_idx][this->view_idx] = tags_map;
    string tags_map_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].tags_map_path;
    ofstream outfile;
    outfile.open(tags_map_path, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }

    for (int u = 0; u < kFlatRows; ++u) {
        for (int v = 0; v < kFlatCols; ++v) {
            const bool kIdxPrint = false;
            if (kIdxPrint) {
                for (int k = 0; k < tags_map[u][v].pts_indices.size(); ++k) {
                    /** k is the number of lidar points that the [u][v] pixel contains **/
                    if (k == tags_map[u][v].pts_indices.size() - 1) {
                        cout << tags_map[u][v].pts_indices[k] << endl;
                        outfile << tags_map[u][v].pts_indices[k] << "\t" << "*****" << "\t" << tags_map[u][v].pts_indices.size() << endl;
                    }
                    else {
                        cout << tags_map[u][v].pts_indices[k] << endl;
                        outfile << tags_map[u][v].pts_indices[k] << "\t";
                    }
                }
            }
            outfile << "lable: " << tags_map[u][v].label << "\t" << "size: " << tags_map[u][v].num_pts << "\t" << "weight: " << tags_map[u][v].weight << endl;
        }
    }
    outfile.close();

    cout << "number of invalid searches:" << invalid_search_num << endl;
    cout << "number of invalid indices:" << invalid_idx_num << endl;
    string flat_img_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].flat_img_path;
    cout << "LiDAR flat image path: " << flat_img_path << endl;
    cv::imwrite(flat_img_path, flat_img);
    cout << "LiDAR flat image generated successfully in " <<(double)(clock() - start_time) / CLOCKS_PER_SEC 
        << "s.\nSpot Index: " << this->spot_idx << "View Index: " << this->view_idx << endl;
    cout << endl;
}

void LidarProcess::EdgeToPixel() {
    /** generate edge_pixels and push back into edge_pixels_vec **/
    cout << "----- LiDAR: EdgeToPixel -----" << endl;
    string edge_img_path = this -> scenes_files_path_vec[this->spot_idx][this->view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nView Index: %d \nPath: %s", this->view_idx, edge_img_path.data());
    ROS_ASSERT_MSG((edge_img.rows == this->kFlatRows || edge_img.cols == this->kFlatCols), "size of original fisheye image is incorrect! View Index: %d", this->view_idx);

    EdgePixels edge_pixels;
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                vector<int> pixel{u, v};
                edge_pixels.push_back(pixel);
            }
        }
    }
    this->edge_pixels_vec[this->spot_idx][this->view_idx] = edge_pixels;
    cout << endl;
}

void LidarProcess::PixLookUp(const CloudPtr& cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << endl;
    int invalid_pixel_space = 0;
    EdgePixels edge_pixels = this->edge_pixels_vec[this->spot_idx][this->view_idx];
    TagsMap tags_map = this->tags_map_vec[this->spot_idx][this->view_idx];
    EdgePts edge_pts;
    CloudPtr edge_cloud (new CloudT);
    for (auto &edge_pixel : edge_pixels) {
        int u = edge_pixel[0];
        int v = edge_pixel[1];
        int num_pts = tags_map[u][v].num_pts;

        if (tags_map[u][v].label == 0) { /** invalid pixels **/
            invalid_pixel_space = invalid_pixel_space + 1;
            continue;
        }
        else { /** normal pixels **/
            /** center of lidar edge distribution **/
            float x = 0, y = 0, z = 0;
            for (int j = 0; j < num_pts; ++j) {
                int idx = tags_map[u][v].pts_indices[j];
                PointT pt = (*cart_cloud)[idx];
                x = x + pt.x;
                y = y + pt.y;
                z = z + pt.z;
            }
            /** average coordinates->unbiased estimation of center position **/
            x = x / (float)num_pts;
            y = y / (float)num_pts;
            z = z / (float)num_pts;
            float weight = tags_map[u][v].weight;
            /** store the spatial coordinates into vector **/
            vector<double> coordinates {x, y, z};
            edge_pts.push_back(coordinates);

            /** store the spatial coordinates into vector **/
            PointT pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            pt.intensity = weight; /** note: I is used to store the point weight **/
            edge_cloud->points.push_back(pt);
        }
    }
    cout << "number of invalid lookups(lidar): " << invalid_pixel_space << endl;
    this->edge_pts_vec[this->spot_idx][this->view_idx] = edge_pts;
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;

    /** write the coordinates and weights into .txt file **/
    string edge_pts_coordinates_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
    ofstream outfile;
    outfile.open(edge_pts_coordinates_path, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (auto &point : edge_cloud->points) {
        outfile << point.x
                << "\t" << point.y
                << "\t" << point.z
                << "\t" << point.intensity << endl;
    }
    outfile.close();
    cout << endl;
}

void LidarProcess::ReadEdge() {
    cout << "----- LiDAR: ReadEdge -----" << endl;
    cout << "View Index in LiDAR ReadEdge: " << this->view_idx << endl;
    string edge_cloud_txt_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
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

    ROS_ASSERT_MSG(!edge_pts.empty(), "LiDAR Read Edge Incorrect! View Index: %d", this->view_idx);
    cout << "Imported LiDAR points: " << edge_pts.size() << endl;
    /** remove duplicated points **/
    std::sort(edge_pts.begin(), edge_pts.end());
    edge_pts.erase(unique(edge_pts.begin(), edge_pts.end()), edge_pts.end());
    cout << "LiDAR Edge Points after Duplicated Removed: " << edge_pts.size() << endl;
    this->edge_pts_vec[this->spot_idx][this->view_idx] = edge_pts;

    /** construct pcl point cloud **/
    PointT pt;
    CloudPtr edge_cloud (new CloudT);
    for (auto &edge_pt : edge_pts) {
        pt.x = edge_pt[0];
        pt.y = edge_pt[1];
        pt.z = edge_pt[2];
        pt.intensity = edge_pt[3];
        edge_cloud->points.push_back(pt);
    }
    cout << "Filtered LiDAR points: " << edge_cloud -> points.size() << endl;
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;
    cout << endl;
}

/***** Extrinsic and Inverse Intrinsic Transform for Visualization of LiDAR Points in Flat Image *****/
vector<vector<double>> LidarProcess::EdgeCloudProjectToFisheye(vector<double> _p) {
    cout << "----- LiDAR: EdgeCloudProjectToFisheye -----" << endl;
    cout << "View Index in EdgeCloudProjectToFisheye:" << this->view_idx << endl;

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

    vector<vector<double>> edge_fisheye_projection(2, vector<double>(this->edge_cloud_vec[this->spot_idx][this->view_idx]->points.size()));

    for (int i = 0; i < this->edge_cloud_vec[this->spot_idx][this->view_idx]->points.size(); i++) {
        p_ << this->edge_cloud_vec[this->spot_idx][this->view_idx]->points[i].x, this->edge_cloud_vec[this->spot_idx][this->view_idx]->points[i].y, this->edge_cloud_vec[this->spot_idx][this->view_idx]->points[i].z;
        p_trans = R * p_ + t;
        theta = acos(p_trans(2) / sqrt(pow(p_trans(0), 2) + pow(p_trans(1), 2) + pow(p_trans(2), 2)));
        inv_r = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        S = {inv_r * p_trans(0) / r, inv_r * p_trans(1) / r};
        p_uv = S + uv_0;
        edge_fisheye_projection[0][i] = p_uv(0);
        edge_fisheye_projection[1][i] = p_uv(1);
    }
    cout << endl;
    return edge_fisheye_projection;
}

int LidarProcess::ReadFileList(const std::string &folder_path, std::vector<std::string> &file_list) {
    DIR *dp;
    struct dirent *dir_path;
    if ((dp = opendir(folder_path.c_str())) == nullptr) {
        return 0;
    }
    int num = 0;
    while ((dir_path = readdir(dp)) != nullptr) {
        std::string name = std::string(dir_path->d_name);
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
    rosbag::View::iterator iterator = view.begin();
    pcl::PCLPointCloud2 pcl_pc2;
    CloudPtr intensityCloud(new CloudT);
    for (int i = 0; iterator != view.end(); iterator++, i++) {
        auto m = *iterator;
        sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
        pcl_conversions::toPCL(*input, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *intensityCloud);
        string id_str = to_string(i);
        string pcds_folder_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].dense_pcds_folder_path;
        pcl::io::savePCDFileBinary(pcds_folder_path + "/" + id_str + ".pcd", *intensityCloud);
    }
}

void LidarProcess::CreateDensePcd() {
    bool merge_icp = true; /** merge_dense = false **/
    int num_pcds;
    string pcd_path;
    string folder_path;
    if (true) {
        num_pcds = LidarProcess::kNumIcpPcds;
        pcd_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].icp_pcd_path;
        folder_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].icp_pcds_folder_path;
    }
    else {
        num_pcds = LidarProcess::kNumRecPcds;
        pcd_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
        folder_path = this->scenes_files_path_vec[this->spot_idx][this->view_idx].dense_pcds_folder_path;
    }

    pcl::PCDReader reader; /** used for read PCD files **/
    vector<string> file_name_vec;
    ReadFileList(folder_path, file_name_vec);
    sort(file_name_vec.begin(), file_name_vec.end()); /** sort file names by order **/
    const int kPcdsGroupSize = file_name_vec.size() / num_pcds;

    /** PCL PointCloud pointer. Remember that the pointer need to be given a new space **/
    CloudPtr input_cloud(new CloudT);
    CloudPtr output_cloud(new CloudT);
    int output_idx = 0;
    int kFileNameLength = num_pcds * kPcdsGroupSize;
    auto name_iter = file_name_vec.begin();
    for(int i = 0; i < kPcdsGroupSize; i++) {
        for(int j = 0; j < num_pcds; j++) {
            string file_name = folder_path + "/" + *name_iter;
            cout << file_name << endl;
            if(reader.read(file_name, *input_cloud) < 0) {      // read PCD files, and save PointCloud in the pointer
                PCL_ERROR("File is not exist!");
                system("pause");
            }
            int point_num = input_cloud->points.size();
            for(int k = 0; k < point_num; k++) {
                output_cloud->points.push_back(input_cloud->points[k]);
            }
            name_iter++;
        }

        pcl::io::savePCDFileBinary(pcd_path, *output_cloud);
        cout << "Create Dense Point Cloud File Successfully!" << endl;
    }
}

void LidarProcess::CreateFullviewPcd() {
    const bool kDenseFullview = false;
    /** target and fullview cloud path **/
    string fullview_target_cloud_path, fullview_cloud_path;
    if (kDenseFullview) {
        fullview_target_cloud_path = this->scenes_files_path_vec[this->spot_idx][this->full_view_idx].dense_pcd_path;
        fullview_cloud_path = this->fullview_rec_folder_path + "/fullview_dense_cloud.pcd";
    }
    else {
        fullview_target_cloud_path = this->scenes_files_path_vec[this->spot_idx][this->full_view_idx].icp_pcd_path;
        fullview_cloud_path = this->fullview_rec_folder_path + "/fullview_sparse_cloud.pcd";
    }

    /** load full view point cloud **/
    CloudPtr fullview_cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(fullview_target_cloud_path, *fullview_cloud) == -1) {
        PCL_ERROR("Pcd File Not Exist!");
    }
    cout << "Degree 0 Full View Dense Pcd Loaded!" << endl;

    for(int i = 0; i < this->num_views; i++) {
        if (i == this->full_view_idx) {
            continue;
        }
        /** load icp pose transform matrix **/
        string pose_trans_mat_path = this->scenes_files_path_vec[this->spot_idx][i].pose_trans_mat_path;
        std::ifstream mat_in;
        mat_in.open(pose_trans_mat_path);
        Eigen::Matrix4f pose_trans_mat;
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                mat_in >> pose_trans_mat(j, k);
            }
        }
        mat_in.close();
        /** transform point cloud **/
        CloudPtr input_cloud(new CloudT);
        CloudPtr input_cloud_trans(new CloudT);
        string input_cloud_path;
        if (kDenseFullview) {
            input_cloud_path = this->scenes_files_path_vec[this->spot_idx][i].dense_pcd_path;
        }
        else {
            input_cloud_path = this->scenes_files_path_vec[this->spot_idx][i].icp_pcd_path;
        }
        if (pcl::io::loadPCDFile<PointT>(input_cloud_path, *input_cloud) == -1) {
            PCL_ERROR("Pcd File Not Exist!");
        }
        cout << "Degree " << this->degree_map[i] << ": Dense Pcd Loaded!" << endl;
        pcl::transformPointCloud(*input_cloud, *input_cloud_trans, pose_trans_mat);
        /** point cloud addition **/
        *fullview_cloud = *fullview_cloud + *input_cloud_trans;
    }
    pcl::io::savePCDFileBinary(fullview_cloud_path, *fullview_cloud);
    cout << "Create Full View Point Cloud File Successfully!" << endl;
}