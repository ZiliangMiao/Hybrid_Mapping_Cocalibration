/** basic **/
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <numeric>
#include "python3.6/Python.h"
/** ros **/
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
/** pcl **/
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
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
#include <pcl/common/time.h>
/** opencv **/
#include <opencv2/opencv.hpp>

/** headings **/
#include "LidarProcess.h"
#include "utils.h"

/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> CloudT;
typedef pcl::PointCloud<PointT>::Ptr CloudPtr;
typedef pcl::PointXYZRGB RGBPointT;
typedef pcl::PointCloud<RGBPointT> RGBCloudT;
typedef pcl::PointCloud<RGBPointT>::Ptr RGBCloudPtr;

LidarProcess::LidarProcess() {
    cout << "----- LiDAR: LidarProcess -----" << endl;
    /** create objects, initialization **/
    string pose_folder_path_temp;
    PoseFilePath pose_files_path_temp;
    EdgePixels edge_pixels_temp;
    EdgePts edge_pts_temp;
    CloudPtr edge_cloud_temp;
    TagsMap tags_map_temp;
    Eigen::Matrix4f pose_trans_mat_temp;
    for (int i = 0; i < this->num_spots; ++i) {
        vector<string> poses_folder_path_vec_temp;
        vector<PoseFilePath> poses_file_path_vec_temp;
        vector<EdgePixels> edge_pixels_vec_temp;
        vector<EdgePts> edge_pts_vec_temp;
        vector<CloudPtr> edge_cloud_vec_temp;
        vector<TagsMap> tags_map_vec_temp;
        vector<Eigen::Matrix4f> poses_trans_mat_vec_temp;
        for (int j = 0; j < num_views; ++j) {
            poses_folder_path_vec_temp.push_back(pose_folder_path_temp);
            poses_file_path_vec_temp.push_back(pose_files_path_temp);
            edge_pixels_vec_temp.push_back(edge_pixels_temp);
            edge_pts_vec_temp.push_back(edge_pts_temp);
            edge_cloud_vec_temp.push_back(edge_cloud_temp);
            tags_map_vec_temp.push_back(tags_map_temp);
            poses_trans_mat_vec_temp.push_back(pose_trans_mat_temp);
        }
        this->poses_folder_path_vec.push_back(poses_folder_path_vec_temp);
        this->poses_files_path_vec.push_back(poses_file_path_vec_temp);
        this->edge_pixels_vec.push_back(edge_pixels_vec_temp);
        this->edge_pts_vec.push_back(edge_pts_vec_temp);
        this->edge_cloud_vec.push_back(edge_cloud_vec_temp);
        this->tags_map_vec.push_back(tags_map_vec_temp);
        this->pose_trans_mat_vec.push_back(poses_trans_mat_vec_temp);
    }

    for (int i = 0; i < this->num_spots; ++i) {
        for (int j = 0; j < this->num_views; ++j) {
            int v_degree = this->view_angle_init + this->view_angle_step * j;
            this -> degree_map[j] = v_degree;
            this -> poses_folder_path_vec[i][j] = this->kDatasetPath + "/spot" + to_string(i) + "/" + to_string(v_degree);
        }
    }

    for (int i = 0; i < this->num_spots; ++i) {
        string spot_path = this->kDatasetPath + "/spot" + to_string(i);
        for (int j = 0; j < this->num_views; ++j) {
            struct PoseFilePath pose_file_path(spot_path, poses_folder_path_vec[i][j]);
            this->poses_files_path_vec[i][j] = pose_file_path;
        }
    }
}

void LidarProcess::ICP() {
    cout << "----- LiDAR: ICP -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    const bool kIcpViz = true;
    CloudPtr view_cloud_tgt(new CloudT);
    CloudPtr view_cloud_src(new CloudT);
    CloudPtr view_cloud_vg_tgt(new CloudT); /** source point cloud **/
    CloudPtr view_cloud_vg_src(new CloudT); /** target point cloud **/
    CloudPtr view_cloud_init_trans_src(new CloudT); /** souce cloud with initial rigid transformation **/
    CloudPtr view_cloud_registered(new CloudT); /** apply icp result to source point cloud **/

    std::string src_pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    std::string tgt_pcd_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;

    /** file loading check **/
    if (pcl::io::loadPCDFile<PointT>(tgt_pcd_path, *view_cloud_tgt) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "Loaded " << view_cloud_tgt->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointT>(src_pcd_path,*view_cloud_src) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "Loaded " << view_cloud_src->size() << " data points from source file" << endl;

    /** invalid point filter **/
    std::vector<int> mapping_in;
    std::vector<int> mapping_out;
    pcl::removeNaNFromPointCloud(*view_cloud_tgt, *view_cloud_tgt, mapping_in);
    pcl::removeNaNFromPointCloud(*view_cloud_src, *view_cloud_src, mapping_out);

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
    cond_filter.setInputCloud(view_cloud_src);
    cond_filter.filter(*view_cloud_src);
    cond_filter.setInputCloud(view_cloud_tgt);
    cond_filter.filter(*view_cloud_tgt);
    cout << "Size of target view cloud after condition filter: " << view_cloud_tgt->size() << endl;
    cout << "Size of source view cloud after condition filter: " << view_cloud_src->size() << endl;

    /** radius outlier filter **/
    pcl::RadiusOutlierRemoval <PointT> outlier_filter;
    outlier_filter.setRadiusSearch(0.5);
    outlier_filter.setMinNeighborsInRadius(30);
    outlier_filter.setInputCloud(view_cloud_tgt);
    outlier_filter.filter(*view_cloud_tgt);
    outlier_filter.setInputCloud(view_cloud_src);
    outlier_filter.filter(*view_cloud_src);
    cout << "Size of target view cloud after outlier filter: " << view_cloud_tgt->size() << endl;
    cout << "Size of source view cloud after outlier filter: " << view_cloud_src->size() << endl;

    /** initial guess of rigid transformation (from gimbal)**/
    Eigen::Matrix<float, 6, 1> ext_init;
    int v_degree = this -> degree_map.at(this->view_idx);
    ext_init << 0.0f, (float)v_degree/180.0f*M_PI, 0.0f, 
                - 0.15f * sin((float)v_degree/180.0f * M_PI) + 0.15f * sin(0.0f/180.0f * M_PI),
                0.0f,
                - 0.15f * cos((float)v_degree/180.0f * M_PI) + 0.15f * cos(0.0f/180.0f * M_PI);
    Eigen::Matrix4f init_trans_mat = ExtrinsicMat(ext_init);
    pcl::transformPointCloud(*view_cloud_vg_src, *view_cloud_init_trans_src, init_trans_mat);
    Eigen::Matrix4f icp_trans_mat = init_trans_mat;

    /** 3 rounds icp **/
    float leaf_size = 0.08;
    float cor_dis = 0.6;
    for (int i = 0; i < 3; ++i) {
        leaf_size = leaf_size - 0.01;
        cor_dis = cor_dis / 2;
        cout << "ICP round " << i << " " << " leaf size: " << leaf_size << endl;
        /** voxel grid down sampling **/
        pcl::VoxelGrid<PointT> vg_tgt;
        vg_tgt.setLeafSize (leaf_size, leaf_size, leaf_size);
        vg_tgt.setInputCloud (view_cloud_tgt);
        vg_tgt.filter (*view_cloud_vg_tgt);
        PCL_INFO("Size of VG Filtered Target Cloud: %d\n", view_cloud_vg_tgt->size());
        pcl::VoxelGrid<PointT> vg_src;
        vg_src.setLeafSize (leaf_size, leaf_size, leaf_size); /** org: 0.05f **/
        vg_src.setInputCloud (view_cloud_src);
        vg_src.filter (*view_cloud_vg_src);
        PCL_INFO("Size of VG Filtered Source Cloud: %d\n", view_cloud_vg_src->size());

        /** timing **/
        pcl::StopWatch timer;
        timer.reset();

        /** ICP **/
        pcl::IterativeClosestPoint <PointT, PointT> icp;
        icp.setInputSource(view_cloud_vg_src);
        icp.setInputTarget(view_cloud_vg_tgt);
        icp.setMaximumIterations(500);
        icp.setMaxCorrespondenceDistance(cor_dis);
        icp.setTransformationEpsilon(1e-10);
        icp.setEuclideanFitnessEpsilon(0.005);
        icp.align(*view_cloud_registered, icp_trans_mat);
        if (icp.hasConverged()) {
            icp_trans_mat = icp.getFinalTransformation();
            cout << "\nICP has converged, score is: " << icp.getFitnessScore() << endl;
            cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
            cout << "\nICP Trans Mat: \n " << icp_trans_mat << endl;
            cout << "ICP run time: " << timer.getTimeSeconds() << " s" << endl;
        } else {
            PCL_ERROR("\nICP has not converged.\n");
        }
    }

    /** save the view trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[this->spot_idx][this->view_idx].pose_trans_mat_path);
    mat_out << icp_trans_mat << endl;
    mat_out.close();

    /** save the pair registered point cloud **/
    string registered_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_recon_folder_path +
                                   "/icp_registered_" + to_string(v_degree) + ".pcd";
    pcl::io::savePCDFileBinary(registered_cloud_path, *view_cloud_registered + *view_cloud_vg_tgt);

    /** visualization **/
    if (kIcpViz) {
        pcl::visualization::PCLVisualizer viewer("ICP demo");
        int v1(0), v2(1); /** create two view point **/
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        float bckgr_gray_level = 0.0;  /** black **/
        float txt_gray_lvl = 1.0 - bckgr_gray_level;

        /** the color of original target cloud is white **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(view_cloud_vg_tgt, (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl);
        viewer.addPointCloud(view_cloud_vg_tgt, cloud_aim_color_h, "cloud_aim_v1", v1);
        viewer.addPointCloud(view_cloud_vg_tgt, cloud_aim_color_h, "cloud_aim_v2", v2);

        /** the color of original source cloud is green **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_in_color_h(view_cloud_init_trans_src, 20, 180, 20);
        viewer.addPointCloud(view_cloud_init_trans_src, cloud_in_color_h, "cloud_in_v1", v1);

        /** the color of transformed source cloud with icp result is red **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_icped_color_h(view_cloud_registered, 180, 20, 20);
        viewer.addPointCloud(view_cloud_registered, cloud_icped_color_h, "cloud_icped_v2", v2);

        while (!viewer.wasStopped()) {
            viewer.spinOnce();
        }
    }
}

Eigen::Matrix4f LidarProcess::ICP2(int view_idx_tgt) {
    /** ICP2 -> view by view registration **/
    cout << "----- LiDAR: ICP -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    const bool kIcpViz = false;
    CloudPtr cloud_target_input(new CloudT);
    CloudPtr cloud_source_input(new CloudT);
    CloudPtr cloud_target_filtered(new CloudT); /** source point cloud **/
    CloudPtr cloud_source_filtered(new CloudT); /** target point cloud **/
    CloudPtr cloud_source_initial_trans(new CloudT); /** source cloud with initial rigid transformation **/
    CloudPtr cloud_icped(new CloudT); /** apply icp result to source point cloud **/

    std::string src_pcd_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].icp_pcd_path;
    std::string tgt_pcd_path = this -> poses_files_path_vec[this->spot_idx][view_idx_tgt].icp_pcd_path;

    /** file loading check **/
    if (pcl::io::loadPCDFile<PointT>(tgt_pcd_path, *cloud_target_input) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "Loaded " << cloud_target_input->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointT>(src_pcd_path,*cloud_source_input) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "Loaded " << cloud_source_input->size() << " data points from source file" << endl;

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

    Eigen::Matrix<float, 6, 1> ext_init;
    int v_degree = this -> degree_map.at(this->view_idx);
    ext_init << 0.0f, (float)v_degree/180.0f*M_PI, 0.0f, 
                - 0.15f * sin((float)v_degree/180.0f * M_PI) + 0.15f * sin(0.0f/180.0f * M_PI),
                0.0f,
                - 0.15f * cos((float)v_degree/180.0f * M_PI) + 0.15f * cos(0.0f/180.0f * M_PI);
    Eigen::Matrix4f initial_trans_mat = ExtrinsicMat(ext_init);
    pcl::transformPointCloud(*cloud_source_filtered, *cloud_source_initial_trans, initial_trans_mat);

    /** original icp **/
    pcl::IterativeClosestPoint <PointT, PointT> icp;
    icp.setMaximumIterations(500);
    icp.setInputSource(cloud_source_filtered);
    icp.setInputTarget(cloud_target_filtered);
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(0.01);
    icp.align(*cloud_icped, initial_trans_mat);

    Eigen::Matrix4f local_trans = Eigen::Matrix4f::Identity();
    if (icp.hasConverged()) {
        local_trans = icp.getFinalTransformation();
        cout << "ICP Converged" << endl;
        cout << "ICP Fitness Score: " << icp.getFitnessScore() << endl;
        cout << "ICP Fitness Epsilon: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "ICP Transformation Matrix: \n" << icp.getFinalTransformation() << endl;
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
    return local_trans;
}

std::tuple<CloudPtr, CloudPtr> LidarProcess::LidarToSphere() {
    cout << "----- LiDAR: LidarToSphere -----" << " Spot Index: " << this->spot_idx << endl;
    /** define the initial projection mode - by intensity or by depth **/
    const bool projByIntensity = this->kProjByIntensity;
    float x, y, z;
    float radius;
    float theta, phi;
    float theta_min = M_PI, theta_max = -M_PI;
    float proj_param;

    string dense_pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_dense_cloud_path;
    /** original cartesian point cloud **/
    CloudPtr cart_cloud(new CloudT);
    pcl::io::loadPCDFile(dense_pcd_path, *cart_cloud);

    /** Initial Transformation **/
    CloudPtr polar_cloud(new CloudT);
    Eigen::Matrix<float, 6, 1> extrinsic_vec; 
    extrinsic_vec << (float)this->extrinsic.rx, (float)this->extrinsic.ry, (float)this->extrinsic.rz, 
                    // (float)this->extrinsic.tx, (float)this->extrinsic.ty, (float)this->extrinsic.tz;
                    0.0, 0.0, 0.0;
    Eigen::Matrix4f T_mat = ExtrinsicMat(extrinsic_vec);
    pcl::transformPointCloud(*cart_cloud, *polar_cloud, T_mat);

    for (auto &point : polar_cloud->points) {
        // if (!projByIntensity) {
        //     radius = proj_param;
        // }
        // else {
        //     radius = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
        // }
        /** assign the polar coordinate to pcl point cloud **/
        radius = point.getVector3fMap().norm();
        phi = atan2(point.y, point.x);
        theta = acos(point.z / radius);
        point.x = theta;
        point.y = phi;
        point.z = 0;
        if (theta > theta_max) { theta_max = theta; }
        else if (theta < theta_min) { theta_min = theta; }
    }
    cout << "min theta of the fullview cloud: " << theta_min << "\n"
         << " max theta of the fullview cloud: " << theta_max << endl;

    tuple<CloudPtr, CloudPtr> result;
    result = make_tuple(polar_cloud, cart_cloud);
    return result;
}

void LidarProcess::SphereToPlane(const CloudPtr& polar_cloud, const CloudPtr& cart_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << " Spot Index: " << this->spot_idx << endl;
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(kFlatRows, kFlatCols, CV_32FC1); /** define the flat image **/
    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(polar_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kScale = sqrt(2);
    const float kSearchRadius = kScale * (kRadPerPix / 2);

    /** std range to generate weights **/
    float dis_std_max = 0;
    float dis_std_min = 1;
    float theta_center;
    float phi_center;
    int hidden_pt_cnt = 0;

    vector<int> removed_indices;

    for (int u = 0; u < kFlatRows; ++u) {
        /** upper and lower bound of the current theta unit **/
        theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;

        for (int v = 0; v < kFlatCols; ++v) {
            /** upper and lower bound of the current phi unit **/
            phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;

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
                /** Default tag with params = 0 **/
                tags_map[u][v].pts_indices.push_back(0);
            }
            else { /** corresponding points are found in the radius neighborhood **/
                vector<double> intensity_vec, theta_vec, phi_vec;
                int hidden_pt_num = 0;
                float dist = 0, dist_mean = 0;
                CloudPtr local_cloud(new CloudT);
                pcl::copyPointCloud(*polar_cloud, search_pt_idx_vec, *local_cloud);
                for (int i = 0; i < search_num; ++i) {
                    bool skip = false;
                    PointT &polar_pt = (*polar_cloud)[search_pt_idx_vec[i]];
                    /** hidden points filter **/
                    if (this->kHiddenPtsFilter) {
                        PointT &cart_pt = (*cart_cloud)[search_pt_idx_vec[i]];
                        const float sensitivity = 0.02;
                        dist = cart_pt.getVector3fMap().norm();
                        dist_mean = (i * dist_mean + dist) / (i + 1); 
                        if (i > 0) {
                            PointT &cart_pt_former = (*cart_cloud)[search_pt_idx_vec[i - 1]];
                            if ((dist < (1-2*sensitivity) * dist_mean && i == 1) ) {
                                hidden_pt_num++;
                                hidden_pt_cnt++;
                                tags_map[u][v].pts_indices.erase(tags_map[u][v].pts_indices.begin());
                                intensity_vec.erase(intensity_vec.begin());
                                theta_vec.erase(theta_vec.begin());
                                phi_vec.erase(phi_vec.begin());
                                dist_mean = dist;
                            }
                            if ((abs(dist_mean - dist) > dist * sensitivity) || ((dist_mean - dist) > dist * sensitivity && cart_pt.intensity < 20)) {
                            // if ((abs(dist_mean - dist) > dist * sensitivity)) {
                                hidden_pt_num++;
                                hidden_pt_cnt++;
                                skip = true;
                                dist_mean = (dist_mean * (i + 1) - dist) / i;
                            }
                        }
                    }

                    if (!skip) {
                        intensity_vec.push_back(polar_pt.intensity);
                        theta_vec.push_back(polar_pt.x);
                        phi_vec.push_back(polar_pt.y);
                        tags_map[u][v].pts_indices.push_back(search_pt_idx_vec[i]);
                    }
                    else {
                        removed_indices.push_back(search_pt_idx_vec[i]);
                    }
                }

                /** add tags **/
                tags_map[u][v].num_pts = search_num - hidden_pt_num;

                /** check the size of vectors **/
                ROS_ASSERT_MSG((theta_vec.size() == phi_vec.size()) && (phi_vec.size() == intensity_vec.size()) && (intensity_vec.size() == tags_map[u][v].pts_indices.size()) && (tags_map[u][v].pts_indices.size() == tags_map[u][v].num_pts), "size of the vectors in a pixel region is not the same!");

                if (tags_map[u][v].num_pts == 1) {
                    /** only one point in the theta-phi sub-region of a pixel **/
                    tags_map[u][v].label = 1;
                    tags_map[u][v].mean = 0;
                    tags_map[u][v].sigma = 0;
                    tags_map[u][v].weight = 1;
                    flat_img.at<float>(u, v) = intensity_vec[0];
                }
                else if (tags_map[u][v].num_pts == 0) {
                    /** no points in a pixel **/
                    tags_map[u][v].label = 0;
                    tags_map[u][v].mean = 99;
                    tags_map[u][v].sigma = 99;
                    tags_map[u][v].weight = 0;
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
                    double intensity_mean = accumulate(std::begin(intensity_vec), std::end(intensity_vec), 0.0) / intensity_vec.size();
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
    cout << "hidden points: " << hidden_pt_cnt << "/" << polar_cloud->points.size() << endl;

    /** add the tags_map of this specific pose to maps **/
    this->tags_map_vec[this->spot_idx][this->view_idx] = tags_map;
    string tags_map_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].tags_map_path;
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

    string flat_img_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].flat_img_path;
    cout << "LiDAR flat image path: " << flat_img_path << endl;
    cv::imwrite(flat_img_path, flat_img);

    if (kEdgeAnalysis) {
        /** visualization for weight check**/ 
        CloudPtr cart_rgb_cloud(new CloudT);
        pcl::copyPointCloud(*cart_cloud, removed_indices, *cart_rgb_cloud);
        pcl::io::savePCDFileBinary(this->poses_files_path_vec[this->spot_idx][this->view_idx].output_folder_path + "/removed_cart_cloud.pcd", *cart_rgb_cloud);
    }

}

void LidarProcess::EdgeToPixel() {
    /** generate edge_pixels and push back into edge_pixels_vec **/
    cout << "----- LiDAR: EdgeToPixel -----" << " Spot Index: " << this->spot_idx << endl;
    string edge_img_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_img_path;
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
}

void LidarProcess::PixLookUp(const CloudPtr& cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << " Spot Index: " << this->spot_idx << endl;
    int invalid_pixel_space = 0;
    EdgePixels edge_pixels = this->edge_pixels_vec[this->spot_idx][this->view_idx];
    TagsMap tags_map = this->tags_map_vec[this->spot_idx][this->view_idx];
    EdgePts edge_pts;
    CloudPtr edge_cloud(new CloudT);
    /** visualization for weight check**/ 
    CloudPtr weight_rgb_cloud(new CloudT);
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

            CloudPtr pixel_cloud(new CloudT);
            float x_avg = 0.0f, y_avg = 0.0f, z_avg = 0.0f;
            pcl::copyPointCloud(*cart_cloud, tags_map[u][v].pts_indices, *pixel_cloud);
            for (auto &pixel_pt : pixel_cloud->points) {
                x_avg += pixel_pt.x;
                y_avg += pixel_pt.y;
                z_avg += pixel_pt.z;
                // if (kEdgeAnalysis) {
                // /** visualization for weight check**/ 
                //     weight_rgb_cloud->points.push_back(pixel_pt);
                // }
            }
            /** average coordinates->unbiased estimation of center position **/
            x_avg = x_avg / num_pts;
            y_avg = y_avg / num_pts;
            z_avg = z_avg / num_pts;
            float weight = tags_map[u][v].weight;

            /** store the spatial coordinates into vector **/
            vector<double> coordinates {x_avg, y_avg, z_avg};
            edge_pts.push_back(coordinates);

            /** store the spatial coordinates into vector **/
            PointT pt;
            pt.x = x_avg;
            pt.y = y_avg;
            pt.z = z_avg;
            pt.intensity = weight; /** note: I is used to store the point weight **/
            edge_cloud->points.push_back(pt);

            if (kEdgeAnalysis) {
                /** visualization for weight check**/ 
                pt.intensity = 255;
                weight_rgb_cloud->points.push_back(pt);
            }
        }
    }
    cout << "number of invalid lookups(lidar): " << invalid_pixel_space << endl;
    this->edge_pts_vec[this->spot_idx][this->view_idx] = edge_pts;
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;

    /** write the coordinates and weights into .txt file **/
    string edge_pts_coordinates_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
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

    if (kEdgeAnalysis) {
        /** visualization for weight check**/ 
        string edge_cart_pcd_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_cart_pcd_path;
        cout << edge_cart_pcd_path << endl;
        pcl::io::savePCDFileBinary(edge_cart_pcd_path, *weight_rgb_cloud);

        float radius, phi, theta;
        CloudPtr polar_rgb_cloud(new CloudT);
        const float half_pi = (float)M_PI / 2;
        Eigen::Matrix<float, 6, 1> extrinsic_vec; 
        extrinsic_vec << (float)this->extrinsic.rx, (float)this->extrinsic.ry, (float)this->extrinsic.rz, 
                        (float)this->extrinsic.tx, (float)this->extrinsic.ty, (float)this->extrinsic.tz;
        Eigen::Matrix4f T_mat = ExtrinsicMat(extrinsic_vec);
        pcl::transformPointCloud(*weight_rgb_cloud, *polar_rgb_cloud, T_mat);
        for (auto &point : polar_rgb_cloud->points) {
            radius = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
            phi = atan2(point.y, point.x);
            theta = acos(point.z / radius);
            point.x = theta;
            point.y = phi;
            point.z = 0;
            point.intensity = 200;
        }

        string edge_polar_pcd_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_polar_pcd_path;
        cout << edge_polar_pcd_path << endl;
        pcl::io::savePCDFileBinary(edge_polar_pcd_path, *polar_rgb_cloud);
    }

}

void LidarProcess::ReadEdge() {
    cout << "----- LiDAR: ReadEdge -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    string edge_cloud_txt_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
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
    CloudPtr edge_cloud(new CloudT);
    for (auto &edge_pt : edge_pts) {
        pt.x = edge_pt[0];
        pt.y = edge_pt[1];
        pt.z = edge_pt[2];
        pt.intensity = edge_pt[3];
        edge_cloud->points.push_back(pt);
    }
    cout << "Filtered LiDAR points: " << edge_cloud -> points.size() << endl;
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;
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
        string pcds_folder_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcds_folder_path;
        pcl::io::savePCDFileBinary(pcds_folder_path + "/" + id_str + ".pcd", *intensityCloud);
    }
}

void LidarProcess::CreateDensePcd() {
    int num_pcds;
    string folder_path, pcd_path;

    if (this->kDenseCloud) {
        num_pcds = LidarProcess::kNumRecPcds;
        pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
        folder_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcds_folder_path;
    }
    else {
        num_pcds = LidarProcess::kNumIcpPcds;
        pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].icp_pcd_path;
        folder_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].icp_pcds_folder_path;
    }

    pcl::PCDReader reader; /** used for read PCD files **/
    vector<string> file_name_vec;
    ReadFileList(folder_path, file_name_vec);
    sort(file_name_vec.begin(), file_name_vec.end()); /** sort file names by order **/
    const int kPcdsGroupSize = file_name_vec.size() / num_pcds; // always equal to 1?

    /** PCL PointCloud pointer. Remember that the pointer need to be given a new space **/
    CloudPtr input_cloud(new CloudT);
    CloudPtr output_cloud(new CloudT);
    for (int i = 0; i < kPcdsGroupSize; i++) {
        for (auto &name : file_name_vec) {
            string file_name = folder_path + "/" + name;
            if(reader.read(file_name, *input_cloud) < 0) {      // read PCD files, and save PointCloud in the pointer
                PCL_ERROR("File is not exist!");
                system("pause");
            }
            *output_cloud += *input_cloud;
        }

        pcl::io::savePCDFileBinary(pcd_path, *output_cloud);
        cout << "Create Dense Point Cloud File Successfully!" << endl;
    }
}

void LidarProcess::CreateFullviewPcd() {
    cout << "----- LiDAR: CreateFullviewPcd -----" << " Spot Index: " << this->spot_idx << endl;
    /** target and fullview cloud path **/
    string fullview_target_cloud_path, fullview_cloud_path;
    if (this->kDenseCloud) {
        fullview_target_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;
        fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].fullview_dense_cloud_path;
    }
    else {
        fullview_target_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].icp_pcd_path;
        fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_sparse_cloud_path;
    }

    /** load full view point cloud **/
    CloudPtr fullview_cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(fullview_target_cloud_path, *fullview_cloud) == -1) {
        PCL_ERROR("Pcd File Not Exist!");
    }
    cout << "Degree 0 Full View Dense Pcd Loaded!" << endl;

    for(int i = 0; i < this->num_views; i++) {
        if (i == this->fullview_idx) {
            continue;
        }
        /** load icp pose transform matrix **/
        string pose_trans_mat_path = this->poses_files_path_vec[this->spot_idx][i].pose_trans_mat_path;
        Eigen::Matrix4f pose_trans_mat = LoadTransMat(pose_trans_mat_path);
        cout << "Degree " << this->degree_map[i] << " ICP Mat: " << "\n" << pose_trans_mat << endl;
        /** transform point cloud **/
        CloudPtr input_cloud(new CloudT);
        CloudPtr input_cloud_trans(new CloudT);
        string input_cloud_path;
        if (this->kDenseCloud) {
            input_cloud_path = this->poses_files_path_vec[this->spot_idx][i].dense_pcd_path;
        }
        else {
            input_cloud_path = this->poses_files_path_vec[this->spot_idx][i].icp_pcd_path;
        }
        if (pcl::io::loadPCDFile<PointT>(input_cloud_path, *input_cloud) == -1) {
            PCL_ERROR("Pcd File Not Exist!");
        }
        pcl::transformPointCloud(*input_cloud, *input_cloud_trans, pose_trans_mat);
        /** point cloud addition **/
        *fullview_cloud = *fullview_cloud + *input_cloud_trans;
    }

    /** check the original point cloud size **/
    int fullview_cloud_size = fullview_cloud->points.size();
    cout << "size of original cloud:" << fullview_cloud_size << endl;

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
    cond_filter.setInputCloud(fullview_cloud);
    cond_filter.filter(*cond_filtered_cloud);

    /** check the pass through filtered point cloud size **/
    int cond_filtered_cloud_size = cond_filtered_cloud->points.size();
    cout << "size of cloud after a condition filter:" << cond_filtered_cloud_size << endl;

    /** radius outlier filter **/
    CloudPtr radius_outlier_cloud(new CloudT);
    pcl::RadiusOutlierRemoval<PointT> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(cond_filtered_cloud);
    radius_outlier_filter.setRadiusSearch(0.05);
    radius_outlier_filter.setMinNeighborsInRadius(20);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.filter(*radius_outlier_cloud);

    /** radius outlier filter cloud size check **/
    int radius_outlier_cloud_size = radius_outlier_cloud->points.size();
    cout << "radius outlier filtered cloud size:" << radius_outlier_cloud_size << endl;

    pcl::io::savePCDFileBinary(fullview_cloud_path, *radius_outlier_cloud);
    cout << "Create Full View Point Cloud File Successfully!" << endl;

    CloudPtr sparse_cloud(new CloudT);
    pcl::UniformSampling<PointT> us_tgt;
    us_tgt.setInputCloud(radius_outlier_cloud);
    us_tgt.setRadiusSearch(0.05f);
    us_tgt.filter(*sparse_cloud);
    string sparse_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].fullview_sparse_cloud_path;
    pcl::io::savePCDFileBinary(sparse_cloud_path, *sparse_cloud);
    cout << "Create sparse fullview pointcloud successfully!" << endl;
}

void LidarProcess::EdgeExtraction() {
    std::string script_path = this->kPkgPath + "/python_scripts/image_process/EdgeExtraction.py";
    std::string kSpots = to_string(this->spot_idx);
    std::string cmd_str = "python3 " 
        + script_path + " " + this->kDatasetPath + " " + "lidar" + " " + kSpots;
    system(cmd_str.c_str());
}

void LidarProcess::SpotRegistration() {
    /** source index and target index **/
    int tgt_idx;
    int src_idx;
    ros::param::get("tgt_idx", tgt_idx);
    ros::param::get("src_idx", src_idx);
    PCL_INFO("ICP Source Index: %d\n", src_idx);

    /** load lio spot transformation matrix **/
    vector<Eigen::Matrix4f> icp_trans_mat_vec;
    string lio_trans_path = this->poses_files_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Eigen::Matrix4f lio_spot_trans_mat = LoadTransMat(lio_trans_path);

    /** create point cloud container  **/
    CloudPtr spot_cloud_src(new CloudT);
    CloudPtr spot_cloud_tgt(new CloudT);
    CloudPtr spot_cloud_vg_src(new CloudT);
    CloudPtr spot_cloud_vg_tgt(new CloudT);
    CloudPtr spot_cloud_init_trans_src(new CloudT);
    CloudPtr spot_cloud_init_trans_tgt(new CloudT);
    CloudPtr pair_clouds_registered(new CloudT);

    /** load points **/
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[tgt_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_tgt);
    PCL_INFO("Size of Target Cloud: %d\n", spot_cloud_tgt->size());
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_src);
    PCL_INFO("Size of Source Cloud: %d\n", spot_cloud_src->size());

    /** initial transformation, only used for visualization **/
    cout << "Initial Trans Mat by LIO: \n" << lio_spot_trans_mat << endl;
    pcl::transformPointCloud(*spot_cloud_vg_src, *spot_cloud_init_trans_src, lio_spot_trans_mat);
    Eigen::Matrix4f icp_trans_mat = lio_spot_trans_mat;

    /** 3 times icp **/
    float leaf_size = 0.11;
    float cor_dis = 0.6;
    for (int i = 0; i < 3; ++i) {
        leaf_size = leaf_size - 0.03;
        cor_dis = cor_dis / 2;
        cout << "ICP round " << i << " " << " leaf size: " << leaf_size << endl;
        // /** voxel grid down sampling **/
        // pcl::VoxelGrid<PointT> vg_tgt;
        // vg_tgt.setLeafSize(leaf_size, leaf_size, leaf_size);
        // vg_tgt.setInputCloud(spot_cloud_tgt);
        // vg_tgt.setMinimumPointsNumberPerVoxel(10);
        // vg_tgt.filter(*spot_cloud_vg_tgt);
        // PCL_INFO("Size of VG Filtered Target Cloud: %d\n", spot_cloud_vg_tgt->size());
        // pcl::VoxelGrid<PointT> vg_src;
        // vg_src.setLeafSize(leaf_size, leaf_size, leaf_size); /** org: 0.05f **/
        // vg_src.setInputCloud(spot_cloud_src);
        // vg_src.setMinimumPointsNumberPerVoxel(10);
        // vg_src.filter(*spot_cloud_vg_src);
        // PCL_INFO("Size of VG Filtered Source Cloud: %d\n", spot_cloud_vg_src->size());
        /** uniform sampling **/
        pcl::UniformSampling<PointT> us_tgt;
        us_tgt.setInputCloud(spot_cloud_tgt);
        us_tgt.setRadiusSearch(leaf_size);
        us_tgt.filter(*spot_cloud_vg_tgt);
        PCL_INFO("Size of US Filtered Target Cloud: %d\n", spot_cloud_vg_tgt->size());
        pcl::UniformSampling<PointT> us_src;
        us_src.setInputCloud(spot_cloud_src);
        us_src.setRadiusSearch(leaf_size);
        us_src.filter(*spot_cloud_vg_src);
        PCL_INFO("Size of US Filtered Source Cloud: %d\n", spot_cloud_vg_src->size());

        /** timing **/
        pcl::StopWatch timer;
        timer.reset();

        /** ICP **/
        pcl::IterativeClosestPoint <PointT, PointT> icp;
        icp.setInputSource(spot_cloud_vg_src);
        icp.setInputTarget(spot_cloud_vg_tgt);
        icp.setMaximumIterations(500);
        icp.setMaxCorrespondenceDistance(cor_dis);
        icp.setTransformationEpsilon(1e-10);
        icp.setEuclideanFitnessEpsilon(0.005);
        icp.align(*pair_clouds_registered, icp_trans_mat);
        if (icp.hasConverged()) {
            icp_trans_mat = icp.getFinalTransformation();
            cout << "\nICP has converged, score is: " << icp.getFitnessScore() << endl;
            cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
            cout << "\nICP Trans Mat: \n " << icp_trans_mat << endl;
            cout << "ICP run time: " << timer.getTimeSeconds() << " s" << endl;
        } else {
            PCL_ERROR("\nICP has not converged.\n");
        }
    }

    icp_trans_mat_vec.push_back(icp_trans_mat);
    /** save the spot trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path);
    mat_out << icp_trans_mat << endl;
    mat_out.close();

    /** save the pair registered point cloud **/
    string pair_registered_cloud_path = this->poses_files_path_vec[tgt_idx][0].fullview_recon_folder_path +
                                        "/icp_registered_spot_tgt_" + to_string(tgt_idx) + ".pcd";
    pcl::io::savePCDFileBinary(pair_registered_cloud_path, *pair_clouds_registered + *spot_cloud_vg_tgt);

    /** visualization **/
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0), v2(1); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0;  /** black **/
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    /** the color of original target cloud is white **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_color_tgt(spot_cloud_vg_tgt, (int)255 * txt_gray_lvl,
                                                                            (int)255 * txt_gray_lvl,
                                                                            (int)255 * txt_gray_lvl);
    viewer.addPointCloud(spot_cloud_vg_tgt, cloud_color_tgt, "cloud_tgt_v1", v1);
    viewer.addPointCloud(spot_cloud_vg_tgt, cloud_color_tgt, "cloud_tgt_v2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tgt_v1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_tgt_v2");

    pcl::visualization::PointCloudColorHandlerCustom <PointT> init_trans_cloud_color(spot_cloud_init_trans_src, 180, 20, 20);
    viewer.addPointCloud(spot_cloud_init_trans_src, init_trans_cloud_color, "init_trans", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "init_trans");

    pcl::visualization::PointCloudColorHandlerCustom <PointT> icp_trans_cloud_color(pair_clouds_registered, 180, 20, 20);
    viewer.addPointCloud(pair_clouds_registered, icp_trans_cloud_color, "icp_trans", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "icp_trans");

    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);
    viewer.addCoordinateSystem();
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  /** viewer size **/

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void LidarProcess::GlobalColoredRecon() {
    /** global cloud registration **/
    RGBCloudPtr spot_clouds_registered(new RGBCloudT);
    for (int src_idx = 1; src_idx < this->num_spots; ++src_idx) {
        /** source index and target index **/
        int tgt_idx = 0;
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        RGBCloudPtr spot_cloud_src(new RGBCloudT);
        RGBCloudPtr spot_cloud_tgt(new RGBCloudT);
        RGBCloudPtr spot_cloud_vg_src(new RGBCloudT);
        RGBCloudPtr spot_cloud_vg_tgt(new RGBCloudT);

        /** load points **/
        pcl::io::loadPCDFile<RGBPointT>(this->poses_files_path_vec[tgt_idx][0].fullview_rgb_cloud_path,
                                     *spot_cloud_tgt);
        pcl::io::loadPCDFile<RGBPointT>(this->poses_files_path_vec[src_idx][0].fullview_rgb_cloud_path,
                                     *spot_cloud_src);

        /** voxel grid down sampling **/
        pcl::VoxelGrid<RGBPointT> vg_tgt;
        vg_tgt.setLeafSize (0.05f, 0.05f, 0.05f);
        vg_tgt.setInputCloud (spot_cloud_tgt);
        vg_tgt.filter (*spot_cloud_vg_tgt);
        pcl::VoxelGrid<RGBPointT> vg_src;
        vg_src.setLeafSize (0.05f, 0.05f, 0.05f);
        vg_src.setInputCloud (spot_cloud_src);
        vg_src.filter(*spot_cloud_vg_src);

        /** load transformation matrix **/
        Eigen::Matrix4f icp_spot_trans_mat = Eigen::Matrix4f::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = this->poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Eigen::Matrix4f tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
            cout << "Load spot ICP trans mat: \n" << tmp_spot_trans_mat << endl;
        }
        pcl::transformPointCloud(*spot_cloud_vg_src, *spot_cloud_vg_src, icp_spot_trans_mat);
        *spot_clouds_registered += *spot_cloud_vg_src;

        /** save the global registered point cloud **/
        string global_registered_cloud_path = this->poses_files_path_vec[src_idx - 1][0].fullview_recon_folder_path +
                                            "/global_registered_rgb_cloud_at_spot_" + to_string(src_idx - 1) + ".pcd";
        pcl::io::savePCDFileBinary(global_registered_cloud_path, *spot_clouds_registered);
    }

    /** visualization **/
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0), v2(1); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    float bkg_grayscale = 0.0;  /** black **/
    float txt_grayscale = 1.0 - bkg_grayscale;

    /** the color of original target cloud is white **/
    viewer.addPointCloud(spot_clouds_registered, "clouds_color_registered", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "clouds_color_registered");
    viewer.setBackgroundColor(bkg_grayscale, bkg_grayscale, bkg_grayscale, v1);
    viewer.addCoordinateSystem();
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  /** viewer size **/

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}
