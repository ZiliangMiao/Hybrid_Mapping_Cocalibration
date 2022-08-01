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
#include <sensor_msgs/PointCloud2.h>
/** pcl **/
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
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

/** point cloud registration **/
tuple<Eigen::Matrix4f, CloudPtr> LidarProcess::ICP(CloudPtr cloud_tgt, CloudPtr cloud_src, Eigen::Matrix4f init_trans_mat, const bool kIcpViz) {
    /** params **/
    float uniform_radius = 0.05;
    int max_iters = 500;
    float max_corr_dis = 0.2;
    float trans_epsilon = 1e-10;
    float eucidean_epsilon = 0.005;
    float max_fitness_range = 2.0;

    /** get the init trans cloud & init fitness score **/
    CloudPtr cloud_init_trans (new CloudT);
    pcl::transformPointCloud(*cloud_src, *cloud_init_trans, init_trans_mat);

    /** uniform sampling **/
    CloudPtr cloud_us_tgt (new CloudT);
    CloudPtr cloud_us_src (new CloudT);
    CloudPtr cloud_init_trans_us (new CloudT);
    pcl::UniformSampling<PointT> us;
    us.setRadiusSearch(uniform_radius);
    us.setInputCloud(cloud_tgt);
    us.filter(*cloud_us_tgt);
    us.setInputCloud(cloud_src);
    us.filter(*cloud_us_src);
    us.setInputCloud (cloud_init_trans);
    us.filter (*cloud_init_trans_us);
    PCL_INFO("Size of Uniform Sampling Filtered Target Cloud: %d\n", cloud_us_tgt->size());
    PCL_INFO("Size of Uniform Sampling Filtered Source Cloud: %d\n", cloud_us_src->size());

    cout << "Initial Fitness Score: " << GetIcpFitnessScore(cloud_us_tgt, cloud_init_trans_us, max_fitness_range) << endl;
    cout << "Initial Matrix: " << init_trans_mat << endl;

    /** ICP **/
    pcl::StopWatch timer;
    timer.reset(); /** timing **/
    CloudPtr cloud_icp_trans (new CloudT);
    CloudPtr cloud_icp_trans_us (new CloudT);
    Eigen::Matrix4f icp_trans_mat;
    pcl::IterativeClosestPoint <PointT, PointT> icp;
    icp.setInputTarget(cloud_us_tgt);
    icp.setInputSource(cloud_us_src);
    icp.setMaximumIterations(max_iters);
    icp.setMaxCorrespondenceDistance(max_corr_dis);
    icp.setTransformationEpsilon(trans_epsilon);
    icp.setEuclideanFitnessEpsilon(eucidean_epsilon);
    icp.align(*cloud_icp_trans_us, init_trans_mat);

    if (icp.hasConverged()) {
        cout << "ICP run time: " << timer.getTimeSeconds() << " s" << endl;
        icp_trans_mat = icp.getFinalTransformation();
        pcl::transformPointCloud(*cloud_src, *cloud_icp_trans, icp_trans_mat);
        cout << "\nICP has converged, calculated score is: " << GetIcpFitnessScore(cloud_us_tgt, cloud_icp_trans_us, max_fitness_range) << endl;
        cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "\nICP Trans Mat: \n " << icp_trans_mat << endl;

        /** transfer rotation matrix to euler angle **/
        Eigen::Matrix3f icp_rotation_mat = icp_trans_mat.topLeftCorner<3,3>();
        Eigen::Vector3f icp_euler_angle = icp_rotation_mat.eulerAngles(2, 1, 0); /** zyx euler angle **/
        cout << "Euler angle by ICP: \n" << icp_euler_angle << endl;
    }
    else {
        PCL_ERROR("\nICP has not converged.\n");
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
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(cloud_us_tgt, (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl);
        viewer.addPointCloud(cloud_us_tgt, cloud_aim_color_h, "cloud_aim_v1", v1);
        viewer.addPointCloud(cloud_us_tgt, cloud_aim_color_h, "cloud_aim_v2", v2);

        /** the color of original source cloud is green **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_in_color_h(cloud_init_trans_us, 20, 180, 20);
        viewer.addPointCloud(cloud_init_trans_us, cloud_in_color_h, "cloud_in_v1", v1);

        /** the color of transformed source cloud with icp result is red **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_icped_color_h(cloud_icp_trans_us, 180, 20, 20);
        viewer.addPointCloud(cloud_icp_trans_us, cloud_icped_color_h, "cloud_icped_v2", v2);

        viewer.addCoordinateSystem();

        while (!viewer.wasStopped()) {
            viewer.spinOnce();
        }
    }
    tuple<Eigen::Matrix4f, CloudPtr> result;
    result = make_tuple(icp_trans_mat, cloud_icp_trans);
    return result;
}

double LidarProcess::GetIcpFitnessScore(CloudPtr &cloud_tgt, CloudPtr &cloud_src, double max_range) {
    double fitness_score = 0.0;
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    // For each point in the source dataset
    int nr = 0;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    pcl::StopWatch timer;
    timer.reset(); /** timing **/

    #pragma omp parallel for num_threads(16)
    for (auto &point : cloud_src->points) {
        // Find its nearest neighbor in the target
        kdtree.nearestKSearch(point, 1, nn_indices, nn_dists);
        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= max_range) {
            // Add to the fitness score
            fitness_score += nn_dists[0];
            nr++;
        }
    }

    cout << "ICP fitness score evaluation time: " << timer.getTimeSeconds() << endl;
    if (nr > 0)
        return (fitness_score / nr);
    return (std::numeric_limits<double>::max());
}

void LidarProcess::ViewRegistration() {
    cout << "----- LiDAR: ViewRegistration -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    /** load point clouds to be registered **/
    std::string tgt_pcd_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;
    std::string src_pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    CloudPtr view_cloud_tgt(new CloudT);
    CloudPtr view_cloud_src(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(tgt_pcd_path, *view_cloud_tgt) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "Loaded " << view_cloud_tgt->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointT>(src_pcd_path,*view_cloud_src) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "Loaded " << view_cloud_src->size() << " points from source file" << endl;

    /** initial rigid transformation **/
    float v_angle = (float)DEG2RAD(this->degree_map[this->view_idx]);
    float radius = 0.15f;
    Eigen::Matrix<float, 6, 1> trans_params;
    trans_params << 0.0f, v_angle, 0.0f,
                    radius * (sin(v_angle) - 0.0f), 0.0f, radius * (cos(v_angle) - 1.0f);
    Eigen::Matrix4f init_trans_mat = ExtrinsicMat(trans_params);

    /** ICP **/
    std::tuple<Eigen::Matrix4f, CloudPtr> icp_result = ICP(view_cloud_tgt, view_cloud_src, init_trans_mat, false);
    Eigen::Matrix4f icp_trans_mat;
    CloudPtr view_cloud_icp_trans;
    std::tie(icp_trans_mat, view_cloud_icp_trans) = icp_result;

    /** save the view trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[this->spot_idx][this->view_idx].pose_trans_mat_path);
    mat_out << icp_trans_mat << endl;
    mat_out.close();

    /** save the registered point clouds **/
    string registered_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_recon_folder_path +
                                   "/icp_registered_" + to_string(v_angle) + ".pcd";
    pcl::io::savePCDFileBinary(registered_cloud_path, *view_cloud_icp_trans + *view_cloud_tgt);
}

void LidarProcess::SpotRegistration() {
    /** source index and target index **/
    int tgt_idx;
    int src_idx;
    ros::param::get("tgt_idx", tgt_idx);
    ros::param::get("src_idx", src_idx);
    PCL_INFO("ICP Source Index: %d\n", src_idx);

    /** load points **/
    CloudPtr spot_cloud_src(new CloudT);
    CloudPtr spot_cloud_tgt(new CloudT);
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[tgt_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_tgt);
    PCL_INFO("Size of Target Cloud: %d\n", spot_cloud_tgt->size());
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_src);
    PCL_INFO("Size of Source Cloud: %d\n", spot_cloud_src->size());

    /** initial transformation and initial score **/
    vector<Eigen::Matrix4f> icp_trans_mat_vec;
    string lio_trans_path = this->poses_files_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Eigen::Matrix4f lio_spot_trans_mat = LoadTransMat(lio_trans_path);
    Eigen::Matrix3f lio_spot_rotation_mat = lio_spot_trans_mat.topLeftCorner<3,3>();
    Eigen::Vector3f lio_euler_angle = lio_spot_rotation_mat.eulerAngles(2, 1, 0); // zyx euler angle
    cout << "Euler angle by LIO: \n" << lio_euler_angle << endl;
    cout << "Initial Trans Mat by LIO: \n" << lio_spot_trans_mat << endl;

    /** ICP **/
    std::tuple<Eigen::Matrix4f, CloudPtr> icp_result = ICP(spot_cloud_tgt, spot_cloud_tgt, lio_spot_trans_mat, false);
    Eigen::Matrix4f icp_trans_mat;
    CloudPtr spot_cloud_icp_trans;
    std::tie(icp_trans_mat, spot_cloud_icp_trans) = icp_result;
    icp_trans_mat_vec.push_back(icp_trans_mat);

    /** save the spot trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path);
    mat_out << icp_trans_mat << endl;
    mat_out.close();

    /** save the pair registered point cloud **/
    string pair_registered_cloud_path = this->poses_files_path_vec[tgt_idx][0].fullview_recon_folder_path +
                                        "/icp_registered_spot_tgt_" + to_string(tgt_idx) + ".pcd";
    pcl::io::savePCDFileBinary(pair_registered_cloud_path, *spot_cloud_icp_trans + *spot_cloud_tgt);
}

/** get LiDAR data **/
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
    CloudPtr load_pcd_cloud(new CloudT);
    CloudPtr view_raw_cloud(new CloudT);
    for (int i = 0; i < kPcdsGroupSize; i++) {
        for (auto &name : file_name_vec) {
            string file_name = folder_path + "/" + name;
            if(reader.read(file_name, *load_pcd_cloud) < 0) {      // read PCD files, and save PointCloud in the pointer
                PCL_ERROR("File is not exist!");
                system("pause");
            }
            *view_raw_cloud += *load_pcd_cloud;
        }
    }

    // /** invalid point filter **/
    // std::vector<int> null_indices;
    // pcl::removeNaNFromPointCloud(*view_raw_cloud, *view_raw_cloud, null_indices);

    /** condition filter **/
    CloudPtr view_cloud(new CloudT);
    pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, -0.4)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<PointT> cond_filter;
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(view_raw_cloud);
    cond_filter.filter(*view_cloud);

    /** check the pass through filtered point cloud size **/
    int cond_filtered_cloud_size = view_cloud->points.size();
    cout << "size of cloud after a condition filter:" << cond_filtered_cloud_size << endl;

    pcl::io::savePCDFileBinary(pcd_path, *view_cloud);
    cout << "Create Dense Point Cloud File Successfully!" << endl;

}

void LidarProcess::CreateFullviewPcd() {
    cout << "----- LiDAR: CreateFullviewPcd -----" << " Spot Index: " << this->spot_idx << endl;
    /** target and fullview cloud path **/
    string fullview_target_cloud_path, fullview_cloud_path;
    fullview_target_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;
    fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].fullview_dense_cloud_path;

    /** load full view point cloud **/
    CloudPtr fullview_raw_cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(fullview_target_cloud_path, *fullview_raw_cloud) == -1) {
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
        CloudPtr view_cloud(new CloudT);
        string view_cloud_path;
        if (this->kDenseCloud) {
            view_cloud_path = this->poses_files_path_vec[this->spot_idx][i].dense_pcd_path;
        }
        else {
            view_cloud_path = this->poses_files_path_vec[this->spot_idx][i].icp_pcd_path;
        }
        if (pcl::io::loadPCDFile<PointT>(view_cloud_path, *view_cloud) == -1) {
            PCL_ERROR("Pcd File Not Exist!");
        }
        pcl::transformPointCloud(*view_cloud, *view_cloud, pose_trans_mat);

        /** point cloud addition **/
        *fullview_raw_cloud = *fullview_raw_cloud + *view_cloud;
    }

    /** check the original point cloud size **/
    int fullview_cloud_size = fullview_raw_cloud->points.size();
    cout << "size of original cloud:" << fullview_cloud_size << endl;

    /** radius outlier filter **/
    CloudPtr radius_outlier_cloud(new CloudT);
    pcl::RadiusOutlierRemoval<PointT> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(fullview_raw_cloud);
    radius_outlier_filter.setRadiusSearch(0.05);
    radius_outlier_filter.setMinNeighborsInRadius(20);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.filter(*radius_outlier_cloud);

    /** radius outlier filter cloud size check **/
    int radius_outlier_cloud_size = radius_outlier_cloud->points.size();
    cout << "radius outlier filtered cloud size:" << radius_outlier_cloud_size << endl;

    pcl::io::savePCDFileBinary(fullview_cloud_path, *radius_outlier_cloud);
    cout << "Create Full View Point Cloud File Successfully!" << endl;
}

/** data pre-processing **/
void LidarProcess::LidarToSphere(CloudPtr &cart_cloud, CloudPtr &polar_cloud) {
    cout << "----- LiDAR: LidarToSphere -----" << " Spot Index: " << this->spot_idx << endl;
    /** define the initial projection mode - by intensity or by depth **/
    const bool projByIntensity = this->kProjByIntensity;
    float theta_min = M_PI, theta_max = -M_PI;
    float proj_param;

    string fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_dense_cloud_path;
    /** original cartesian point cloud **/
    pcl::io::loadPCDFile(fullview_cloud_path, *cart_cloud);

    /** Initial Transformation **/
    Eigen::Matrix<float, 6, 1> extrinsic_vec; 
    extrinsic_vec << (float)this->extrinsic.rx, (float)this->extrinsic.ry, (float)this->extrinsic.rz, 
                    // (float)this->extrinsic.tx, (float)this->extrinsic.ty, (float)this->extrinsic.tz;
                    0.0, 0.0, 0.0;
    Eigen::Matrix4f T_mat = ExtrinsicMat(extrinsic_vec);
    pcl::transformPointCloud(*cart_cloud, *polar_cloud, T_mat);

    /** Multiprocessing test **/
    #pragma omp parallel for num_threads(16)

    for (auto &point : polar_cloud->points) {
        // if (!projByIntensity) {
        //     radius = proj_param;
        // }
        // else {
        //     radius = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
        // }

        /** assign the polar coordinate to pcl point cloud **/
        float radius = point.getVector3fMap().norm();
        float phi = atan2(point.y, point.x);
        float theta = acos(point.z / radius);
        point.x = theta;
        point.y = phi;
        point.z = 0;
        if (theta > theta_max) { theta_max = theta; }
        else if (theta < theta_min) { theta_min = theta; }
    }
    cout << "min theta of the fullview cloud: " << theta_min << "\n"
         << " max theta of the fullview cloud: " << theta_max << endl;

}

void LidarProcess::SphereToPlane(const CloudPtr& cart_cloud, const CloudPtr& polar_cloud) {
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
                flat_img.at<float>(u, v) = 0; /** intensity **/
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
                        const float sensitivity = 0.02f;
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
                    flat_img.at<float>(u, v) = 0;
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
         pcl::io::savePCDFileBinary(this->poses_files_path_vec[this->spot_idx][this->view_idx].output_folder_path + "/fullview_polar_cloud.pcd", *polar_cloud);
    }

}

void LidarProcess::EdgeExtraction() {
    std::string script_path = this->kPkgPath + "/python_scripts/image_process/EdgeExtraction.py";

    std::string kSpots = to_string(this->spot_idx);
    std::string cmd_str = "python3 "
                          + script_path + " " + this->kDatasetPath + " " + "lidar" + " " + kSpots;

    system(cmd_str.c_str());
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
        
        CloudPtr polar_rgb_cloud(new CloudT);
        Eigen::Matrix<float, 6, 1> extrinsic_vec; 
        extrinsic_vec << (float)this->extrinsic.rx, (float)this->extrinsic.ry, (float)this->extrinsic.rz, 
                        0.0f, 0.0f, 0.0f;
        Eigen::Matrix4f T_mat = ExtrinsicMat(extrinsic_vec);
        pcl::transformPointCloud(*weight_rgb_cloud, *polar_rgb_cloud, T_mat);

        float radius, phi, theta;
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

/** reconstruction **/

void LidarProcess::GlobalColoredRecon() {
    /** global cloud registration **/
    RGBCloudPtr spot_clouds_registered(new RGBCloudT);
    pcl::io::loadPCDFile<RGBPointT>(this->poses_files_path_vec[0][0].fullview_rgb_cloud_path,
                                    *spot_clouds_registered);
    for (int src_idx = 1; src_idx < this->num_spots; ++src_idx) {
        /** source index and target index **/
        int tgt_idx = 0;
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        RGBCloudPtr spot_cloud_src(new RGBCloudT);
        RGBCloudPtr spot_cloud_us_src(new RGBCloudT);

        /** load points **/
        pcl::io::loadPCDFile<RGBPointT>(this->poses_files_path_vec[src_idx][0].fullview_rgb_cloud_path,
                                        *spot_cloud_src);

        /** down sampling **/
        pcl::UniformSampling<RGBPointT> us;
        us.setRadiusSearch(0.01f);
        us.setInputCloud(spot_cloud_src);
        us.filter(*spot_cloud_us_src);

        /** load transformation matrix **/
        Eigen::Matrix4f icp_spot_trans_mat = Eigen::Matrix4f::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = this->poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Eigen::Matrix4f tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
            cout << "Load spot ICP trans mat: \n" << tmp_spot_trans_mat << endl;
        }
        pcl::transformPointCloud(*spot_cloud_us_src, *spot_cloud_us_src, icp_spot_trans_mat);
        *spot_clouds_registered += *spot_cloud_us_src;

        // /** save the global registered point cloud **/
        // string global_registered_cloud_path = this->poses_files_path_vec[src_idx - 1][0].fullview_recon_folder_path +
        //                                     "/global_registered_rgb_cloud_at_spot_" + to_string(src_idx - 1) + ".pcd";
        // pcl::io::savePCDFileBinary(global_registered_cloud_path, *spot_clouds_registered);
    }
    string global_registered_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                        "/global_registered_rgb_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *spot_clouds_registered);
}
