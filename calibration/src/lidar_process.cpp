/** headings **/
#include <lidar_process.h>
#include <common_lib.h>

/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

LidarProcess::LidarProcess() {
    /** parameter server **/
    ros::param::get("essential/kLidarTopic", topic_name);
    ros::param::get("essential/kDatasetName", dataset_name);
    ros::param::get("essential/kNumSpots", num_spots);
    ros::param::get("essential/kNumViews", num_views);
    ros::param::get("essential/kAngleInit", view_angle_init);
    ros::param::get("essential/kAngleStep", view_angle_step);
    kDatasetPath = kPkgPath + "/data/" + dataset_name;
    fullview_idx = (num_views - 1) / 2;

    cout << "----- LiDAR: LidarProcess -----" << endl;
    vector<vector<string>> folder_path_vec_tmp(num_spots, vector<string>(num_views));
    vector<vector<PoseFilePath>> file_path_vec_tmp(num_spots, vector<PoseFilePath>(num_views));
    vector<vector<EdgeCloud>> edge_cloud_vec_tmp(num_spots, vector<EdgeCloud>(num_views));
    vector<vector<TagsMap>> tags_map_vec_tmp(num_spots, vector<TagsMap>(num_views));
    vector<vector<Mat4F>> pose_trans_mat_vec_tmp(num_spots, vector<Mat4F>(num_views));
    this->folder_path_vec = folder_path_vec_tmp;
    this->file_path_vec = file_path_vec_tmp;
    this->edge_cloud_vec = edge_cloud_vec_tmp;
    this->tags_map_vec = tags_map_vec_tmp;
    this->pose_trans_mat_vec = pose_trans_mat_vec_tmp;

    for (int i = 0; i < num_spots; ++i) {
        string spot_path = kDatasetPath + "/spot" + to_string(i);

        for (int j = 0; j < num_views; ++j) {
            int v_degree = view_angle_init + view_angle_step * j;
            degree_map[j] = v_degree;
            folder_path_vec[i][j] = spot_path + "/" + to_string(v_degree);
            struct PoseFilePath pose_file_path(spot_path, folder_path_vec[i][j]);
            file_path_vec[i][j] = pose_file_path;
        }
    }
}

void LidarProcess::BagToPcd(string filepath, CloudI &cloud) {
    rosbag::Bag bag;
    bag.open(filepath, rosbag::bagmode::Read);
    vector<string> topics{topic_name};
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    rosbag::View::iterator iterator = view.begin();
    pcl::PCLPointCloud2 pcl_pc2;

    CloudI::Ptr bag_cloud(new CloudI);
    uint32_t cnt_pcds = 0; 
    while (iterator != view.end()) {
        iterator++;
        cnt_pcds++;
        ROS_ASSERT_MSG((cnt_pcds > 3.6e4), "More than 36000 pcds in a bag, aborted.");
    }

    uint32_t num_pcds = (float)cnt_pcds * ((float)95 / 100);
    uint32_t idx_start = (cnt_pcds - num_pcds) / 2;
    uint32_t idx_end = idx_start + num_pcds;
    iterator = view.begin();

    for (uint32_t i = 0; iterator != view.end(); iterator++, i++) {
        if (i >= idx_start && i < idx_end) {
            auto m = *iterator;
            sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
            pcl_conversions::toPCL(*input, pcl_pc2);
            pcl::fromPCLPointCloud2(pcl_pc2, *bag_cloud);
            cloud += *bag_cloud;
        }
    }
}

/** Data Pre-processing **/
void LidarProcess::LidarToSphere(CloudI::Ptr &cart_cloud, CloudI::Ptr &polar_cloud) {
    cout << "----- LiDAR: LidarToSphere -----" << " Spot Index: " << spot_idx << endl;
    float theta_min = M_PI, theta_max = -M_PI;

    string fullview_cloud_path = file_path_vec[spot_idx][view_idx].spot_cloud_path;
    pcl::io::loadPCDFile(fullview_cloud_path, *cart_cloud);

    /** Initial Transformation **/
    Ext_D extrinsic_vec;
    extrinsic_vec << ext_.head(3), 0, 0, 0;
    Mat4D T_mat = TransformMat(extrinsic_vec);
    pcl::transformPointCloud(*cart_cloud, *polar_cloud, T_mat);

    for (auto &point : polar_cloud->points) {
        float radius = point.getVector3fMap().norm();
        float phi = atan2(point.y, point.x);
        float theta = acos(point.z / radius);
        point.x = theta;
        point.y = phi;
        point.z = radius;
        if (theta > theta_max) { theta_max = theta; }
        else if (theta < theta_min) { theta_min = theta; }
    }

    cout << "LiDAR polar cloud generated. \ntheta_min = " << theta_min << " theta_max = " << theta_max << endl;
}

void LidarProcess::SphereToPlane(CloudI::Ptr& polar_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << " Spot Index: " << spot_idx << endl;
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(kFlatRows, kFlatCols, CV_8U); /** define the flat image **/
    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointI> kdtree;
    CloudI::Ptr polar_flat_cloud(new CloudI);
    pcl::copyPointCloud(*polar_cloud, *polar_flat_cloud);
    for (auto &pt : polar_flat_cloud->points) {pt.z = 0;}
    kdtree.setInputCloud(polar_flat_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num, valid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kSearchRadius = sqrt(2) * (kRadPerPix / 2);
    const float sensitivity = 0.02f;

    #pragma omp parallel for num_threads(THREADS)

    for (int u = 0; u < kFlatRows; ++u) {
        float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;
        for (int v = 0; v < kFlatCols; ++v) {
            float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;

            /** assign the theta and phi center to the search_center **/
            PointI search_center;
            search_center.x = theta_center;
            search_center.y = phi_center;
            search_center.z = 0;

            vector<int> tag;

            /** define the vector container for storing the info of searched points **/
            vector<int> search_pt_idx_vec;
            vector<float> search_pt_squared_dis_vec; /** type of distance vector has to be float **/
            /** use kdtree to search (radius search) the spherical point cloud **/
            int search_num = kdtree.radiusSearch(search_center, kSearchRadius, search_pt_idx_vec, search_pt_squared_dis_vec); // number of the radius nearest neighbors
            if (search_num == 0) {
                flat_img.at<uint8_t>(u, v) = 0; /** intensity **/
                invalid_search_num ++;
            }
            else { /** corresponding points are found in the radius neighborhood **/
                int hidden_pt_num = 0;
                float dist_mean = 0;
                float intensity_mean = 0;
                vector<int> local_vec(search_num, 0);

                for (int i = 0; i < search_num; ++i) {
                    dist_mean += polar_cloud->points[search_pt_idx_vec[i]].z;
                }
                dist_mean = dist_mean / search_num;

                for (int i = 0; i < search_num; ++i) {
                    PointI &local_pt = polar_cloud->points[search_pt_idx_vec[i]];
                    float dist = local_pt.z;
                    if ((abs(dist_mean - dist) > dist * sensitivity) || ((dist_mean - dist) > dist * sensitivity && local_pt.intensity < 20)) {
                        hidden_pt_num++;
                    }
                    else {
                        intensity_mean += local_pt.intensity;
                        local_vec[i] = search_pt_idx_vec[i];
                    }
                }

                /** add tags **/
                local_vec.erase(std::remove(local_vec.begin(), local_vec.end(), 0), local_vec.end());
                tag.insert(tag.begin(), local_vec.data(), local_vec.data()+local_vec.size());

                if (tag.size() > 0) {
                    intensity_mean /= tag.size();
                }                
                flat_img.at<uchar>(u, v) = static_cast<uchar>(intensity_mean);
            }
            
            tags_map[u][v] = tag;
        }
    }

    ROS_ASSERT_MSG(cv::mean(flat_img) < 1, "Warning: blank image generated.")   

    /** add the tags_map of this specific pose to maps **/
    tags_map_vec[spot_idx][view_idx] = tags_map;

    string flat_img_path = this->file_path_vec[spot_idx][view_idx].flat_img_path;
    cv::imwrite(flat_img_path, flat_img);

}

void LidarProcess::EdgeExtraction() {
    string script_path = kPkgPath + "/python_scripts/image_process/edge_extraction.py";
    string kSpots = to_string(spot_idx);
    string cmd_str = "python3 " + script_path + " " + kDatasetPath + " " + "lidar" + " " + kSpots;
    int status = system(cmd_str.c_str());
}

void LidarProcess::GenerateEdgeCloud(CloudI::Ptr& cart_cloud) {
    PoseFilePath &path_vec = this->file_path_vec[spot_idx][view_idx];
    string edge_img_path = this->file_path_vec[spot_idx][view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nView Index: %d \nPath: %s", view_idx, edge_img_path.c_str());
    ROS_ASSERT_MSG((edge_img.rows == kFlatRows || edge_img.cols == kFlatCols), "size of original fisheye image is incorrect! View Index: %d", view_idx);

    TagsMap &tags_map = tags_map_vec[spot_idx][view_idx];
    EdgeCloud::Ptr edge_cloud(new EdgeCloud);
    CloudI::Ptr edge_xyzi (new CloudI);
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                Tags &tag = tags_map[u][v];
                for (int i = 0; i < tag.size(); ++i) { 
                    PointI &pixel_pt = cart_cloud->points[tag[i]];
                    edge_xyzi->points.push_back(pixel_pt);
                }
            }
        }
    }

    /** uniform sampling **/
    pcl::UniformSampling<PointI> us;
    us.setRadiusSearch(0.005);
    us.setInputCloud(edge_xyzi);
    us.filter(*edge_xyzi);

    pcl::copyPointCloud(*edge_xyzi, this->edge_cloud_vec[spot_idx][view_idx]);
    string edge_cloud_path = file_path_vec[spot_idx][view_idx].edge_cloud_path;
    if (kColorMap) {
        pcl::io::savePCDFileBinary(edge_cloud_path, *edge_xyzi);
    }
    else {
        pcl::copyPointCloud(*edge_xyzi, *edge_cloud);
        pcl::io::savePCDFileBinary(edge_cloud_path, *edge_cloud);
    }

}

void LidarProcess::ReadEdge() {
    string edge_cloud_path = this->file_path_vec[spot_idx][view_idx].edge_cloud_path;
    LoadPcd(edge_cloud_path, this->edge_cloud_vec[spot_idx][view_idx], "lidar edge");
}

/** Point Cloud Registration **/
Mat4F LidarProcess::Align(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, Mat4F init_trans_mat, int cloud_type, const bool kIcpViz) {
    /** params **/
    float uniform_radius = 0.05;
    float normal_radius = 0.15;

    int max_iters = 100;
    float max_corr_dis = 0.2;
    float eucidean_epsilon = 1e-12;
    float max_fitness_range = 2.0;

    bool enable_auto_radius = true;
    bool auto_radius_trig = false;
    float target_size = 4e+6;

    pcl::StopWatch timer;

    /** uniform sampling **/
    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);
    CloudI::Ptr cloud_src_init_trans (new CloudI);
    pcl::UniformSampling<PointI> us;
    us.setRadiusSearch(uniform_radius);
    pcl::transformPointCloud(*cloud_src, *cloud_src_init_trans, init_trans_mat);

    while (enable_auto_radius){
        us.setRadiusSearch(uniform_radius);
        us.setInputCloud(cloud_tgt);
        us.filter(*cloud_us_tgt);
        us.setInputCloud(cloud_src_init_trans);
        us.filter(*cloud_us_src);
        uniform_radius *= sqrt((cloud_us_tgt->size() + cloud_us_src->size()) / (2 * target_size));
        if (auto_radius_trig) {break;}
        auto_radius_trig = true;
    }
    normal_radius = uniform_radius * 3;
    ROS_INFO("Uniform sampling for target cloud: %ld -> %ld\n", cloud_tgt->size(), cloud_us_tgt->size());
    ROS_INFO("Uniform sampling for source cloud: %ld -> %ld\n", cloud_src->size(), cloud_us_src->size());
    pcl::transformPointCloud(*cloud_us_src, *cloud_us_src, init_trans_mat.inverse());

    /** invalid point filter **/
    RemoveInvalidPoints(cloud_us_tgt);
    RemoveInvalidPoints(cloud_us_src);

    CloudI::Ptr cloud_us_tgt_effe (new CloudI);
    CloudI::Ptr cloud_us_src_effe (new CloudI);
    std::vector<int> tgt_indices(cloud_us_tgt->size());
    std::vector<int> src_indices(cloud_us_src->size());
    timer.reset();
    
    if (cloud_type == 0) { /** view point cloud **/
        cout << "Range effective filter" << endl;
        const float squared_range_limit = pow(10, 2);
        for (int idx = 0; idx < cloud_us_src->size(); ++idx) {
            auto &pt = cloud_us_src->points[idx];
            src_indices[idx] = (pt.getVector3fMap().squaredNorm() < squared_range_limit && pt.z > -1) ? idx : 0;
        }
        for (int idx = 0; idx < cloud_us_tgt->size(); ++idx) {
            auto &pt = cloud_us_tgt->points[idx];
            tgt_indices[idx] = (pt.getVector3fMap().squaredNorm() < squared_range_limit && pt.z > -1) ? idx : 0;
        }
    }
    else if (cloud_type == 1) { /** spot point cloud **/
        cout << "k-nearest search effective filter" << endl;

        pcl::registration::CorrespondenceEstimation<PointI, PointI> core;
        CloudI::ConstPtr cloud_src_search (cloud_us_src);
        CloudI::ConstPtr cloud_tgt_search (cloud_us_tgt);
        core.setInputSource(cloud_src_search);
        core.setInputTarget(cloud_tgt_search);
        
        boost::shared_ptr<pcl::Correspondences> cor(new pcl::Correspondences);   //共享所有权的智能指针，以kdtree做索引
        core.determineReciprocalCorrespondences(*cor, max_corr_dis);   //点之间的最大距离,cor对应索引

        #pragma omp parallel for num_threads(THREADS)
        for (size_t i = 0; i < cor->size(); i++) {
            int tgt_idx = cor->at(i).index_match;
            int src_idx = cor->at(i).index_query;
            tgt_indices[tgt_idx] = tgt_idx;
            src_indices[src_idx] = src_idx;
        }

        // std::vector<int> nn_indices(1);
        // std::vector<float> nn_dists(1);

        // pcl::KdTreeFLANN<PointI> kdtree_tgt;
        // pcl::KdTreeFLANN<PointI> kdtree_src;
        // kdtree_tgt.setInputCloud (cloud_us_tgt);
        // kdtree_src.setInputCloud (cloud_us_src);
        // #pragma omp parallel for num_threads(THREADS)
        // for (int idx = 0; idx < cloud_us_src->size(); ++idx) {
        //     kdtree_tgt.nearestKSearch (cloud_us_src->points[idx], 1, nn_indices, nn_dists);
        //     if (nn_dists[0] <= max_fitness_range) {
        //         src_indices[idx] = idx;
        //         tgt_indices[nn_indices[0]] = nn_indices[0];
        //     }
        // }
        // #pragma omp parallel for num_threads(THREADS)
        // for (int idx = 0; idx < cloud_us_tgt->size(); ++idx) {
        //     if (tgt_indices[idx] == 0) {
        //         kdtree_src.nearestKSearch (cloud_us_tgt->points[idx], 1, nn_indices, nn_dists);
        //         tgt_indices[idx] = (nn_dists[0] <= max_fitness_range) ? idx : 0;
        //     }
        // }
    }
    tgt_indices.erase(std::remove(tgt_indices.begin(), tgt_indices.end(), 0), tgt_indices.end());
    src_indices.erase(std::remove(src_indices.begin(), src_indices.end(), 0), src_indices.end());
    pcl::copyPointCloud(*cloud_us_tgt, tgt_indices, *cloud_us_tgt_effe);
    pcl::copyPointCloud(*cloud_us_src, src_indices, *cloud_us_src_effe);
    ROS_INFO("Effective filter for target cloud: %ld -> %ld\n", cloud_us_tgt->size(), cloud_us_tgt_effe->size());
    ROS_INFO("Effective filter for source cloud: %ld -> %ld\n", cloud_us_src->size(), cloud_us_src_effe->size());
    ROS_INFO("Run time: %f s\n", timer.getTimeSeconds());

    /** invalid point filter **/
    RemoveInvalidPoints(cloud_us_tgt_effe);
    RemoveInvalidPoints(cloud_us_src_effe);

    /** get the init trans cloud & init fitness score **/
    CloudI::Ptr cloud_init_trans_us (new CloudI);
    pcl::transformPointCloud(*cloud_us_src_effe, *cloud_init_trans_us, init_trans_mat);
    cout << "\nInit Trans Mat: \n " << init_trans_mat << endl;
    pcl::StopWatch timer_fs;
    cout << "Initial Fitness Score: " << GetFitnessScore(cloud_us_tgt_effe, cloud_init_trans_us, max_fitness_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;

    /** Align point clouds **/
    CloudI::Ptr cloud_icp_trans_us (new CloudI);
    CloudIN::Ptr cloud_icp_trans_n (new CloudIN);
        
    timer.reset();
    ROS_INFO("Normal estimation ... \n");

    pcl::NormalEstimationOMP<PointI, pcl::Normal> normal_est;
    pcl::search::KdTree<PointI>::Ptr kdtree(new pcl::search::KdTree<PointI>);
    normal_est.setRadiusSearch(normal_radius);
    normal_est.setSearchMethod(kdtree);

    CloudIN::Ptr cloud_tgt_in(new CloudIN);
    CloudN::Ptr tgt_norms(new CloudN);
    normal_est.setInputCloud(cloud_us_tgt_effe);
    normal_est.compute(*tgt_norms);
    pcl::concatenateFields(*cloud_us_tgt_effe, *tgt_norms, *cloud_tgt_in);

    CloudIN::Ptr cloud_src_in(new CloudIN);
    CloudN::Ptr src_norms(new CloudN);
    normal_est.setInputCloud(cloud_us_src_effe);
    normal_est.compute(*src_norms);
    pcl::concatenateFields(*cloud_us_src_effe, *src_norms, *cloud_src_in);
    ROS_INFO("Run time: %f s\n", timer.getTimeSeconds());
    
    timer.reset();
    ROS_INFO("ICP alignment ... \n");
    pcl::GeneralizedIterativeClosestPoint<PointIN, PointIN> align;
    Mat4F align_trans_mat;

    align.setInputSource(cloud_src_in);
    align.setInputTarget(cloud_tgt_in);
    align.setMaximumIterations(max_iters);
    align.setMaxCorrespondenceDistance(max_corr_dis);
    align.setEuclideanFitnessEpsilon(eucidean_epsilon);
    align.setRotationEpsilon(eucidean_epsilon);
    align.align(*cloud_icp_trans_n, init_trans_mat);
    pcl::copyPointCloud(*cloud_icp_trans_n, *cloud_icp_trans_us);

     if (align.hasConverged()) {
        ROS_INFO("ICP: Converged in %f s.\n", timer.getTimeSeconds());
        ROS_INFO("Fitness score: %f \n", GetFitnessScore(cloud_us_tgt_effe, cloud_icp_trans_us, max_fitness_range));
        ROS_INFO("Epsilon: %f \n", align.getEuclideanFitnessEpsilon());
        align_trans_mat = align.getFinalTransformation();
        cout << align_trans_mat << endl;
        ROS_INFO("Align completed.\n");
    }
    else {
        ROS_INFO("Align: ICP failed to converge. \n");
    }
    return align_trans_mat;
}

void LidarProcess::CalcEdgeDistance(EdgeCloud::Ptr cloud_tgt, EdgeCloud::Ptr cloud_src, float max_range) {
    pcl::StopWatch timer_fs;
    vector<float> dists;
    int valid_cnt = 0;
    float avg_dist = 0;
    float outlier_percentage = 0.1;

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    EdgeCloud::Ptr cloud_tree (new EdgeCloud);
    pcl::copyPointCloud(*cloud_tgt, *cloud_tree);
    kdtree.setInputCloud(cloud_tree);

    for (auto &pt : cloud_src->points) {
        kdtree.nearestKSearch(pt, 1, nn_indices, nn_dists);
        if (nn_dists[0] <= max_range) {
            dists.push_back(nn_dists[0]);
        }
    }

    if (dists.size() * outlier_percentage > 1) {
        sort(dists.data(), dists.data()+dists.size());
        for (size_t i = 0; i < dists.size() * (1-outlier_percentage); i++) {
            avg_dist += dists[i];
            ++valid_cnt;
        }
        if (valid_cnt > 0) {
            avg_dist /= valid_cnt;
            ROS_INFO("Average projection error: %f", avg_dist);
        } 
    }

}

void LidarProcess::CreateDensePcd() {
    cout << "----- LiDAR: CreateDensePcd -----" << " Spot Index: " << spot_idx << " View Index: " << view_idx << endl;

    string pcd_path = file_path_vec[spot_idx][view_idx].view_cloud_path;
    string bag_path = file_path_vec[spot_idx][view_idx].bag_folder_path 
                    + "/" + dataset_name + "_spot" + to_string(spot_idx) 
                    + "_" + to_string(view_angle_init + view_angle_step * view_idx)
                    + ".bag";
    CloudI::Ptr view_cloud(new CloudI);
    BagToPcd(bag_path, *view_cloud);
    cout << "size of loaded point cloud: " << view_cloud->points.size() << endl;

    /** Range filter **/
    const float squared_range_limit = pow(0.5, 2);
    vector<int> indices(view_cloud->size());
    for (int idx = 0; idx < view_cloud->size(); ++idx) {
        PointI &pt = view_cloud->points[idx];
        indices[idx] = (pt.getVector3fMap().squaredNorm() > squared_range_limit) ? idx : 0;
    }
    indices.erase(std::remove(indices.begin(), indices.end(), 0), indices.end());
    pcl::copyPointCloud(*view_cloud, indices, *view_cloud);

    /** invalid point filter **/
    RemoveInvalidPoints(view_cloud);
    cout << "size of cloud:" << view_cloud->points.size() << endl;

    pcl::io::savePCDFileBinary(pcd_path, *view_cloud);
    cout << "view cloud generated." << endl;
}

void LidarProcess::ViewRegistration() {
    // cout << "----- LiDAR: ViewRegistration -----" << " Spot Index: " << spot_idx << " View Index: " << view_idx << endl;
    /** load point clouds to be registered **/
    std::string tgt_pcd_path = file_path_vec[spot_idx][fullview_idx].view_cloud_path;
    std::string src_pcd_path = file_path_vec[spot_idx][view_idx].view_cloud_path;
    CloudI::Ptr view_cloud_tgt(new CloudI);
    CloudI::Ptr view_cloud_src(new CloudI);
    LoadPcd(tgt_pcd_path, *view_cloud_tgt, "target view");
    LoadPcd(src_pcd_path, *view_cloud_src, "source view");

    /** initial rigid transformation **/
    float v_angle = (float)DEG2RAD(degree_map[view_idx]);
    float gimbal_radius = 0.15f;
    Ext_F trans_params;
    trans_params << 0.0f, v_angle, 0.0f,
                    gimbal_radius * (sin(v_angle) - 0.0f), 0.0f, gimbal_radius * (cos(v_angle) - 1.0f); /** LiDAR x-axis: car front; Gimbal positive angle: car front **/
    Mat4F init_trans_mat = TransformMat(trans_params);
    Mat4F align_trans_mat = init_trans_mat;

    /** ICP **/
    align_trans_mat = Align(view_cloud_tgt, view_cloud_src, init_trans_mat, 0, false);
    CloudI::Ptr view_cloud_icp_trans(new CloudI);
    pcl::transformPointCloud(*view_cloud_src, *view_cloud_icp_trans, align_trans_mat);

    /** save the view trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(file_path_vec[spot_idx][view_idx].pose_trans_mat_path);
    mat_out << align_trans_mat << endl;
    mat_out.close();

    if (FULL_OUTPUT) {
        /** save the registered point clouds **/
        string registered_cloud_path = file_path_vec[spot_idx][view_idx].fullview_recon_folder_path +
                                    "/icp_registered_" + to_string(v_angle) + ".pcd";
        pcl::io::savePCDFileBinary(registered_cloud_path, *view_cloud_icp_trans + *view_cloud_tgt);
    }
}

void LidarProcess::FullViewMapping() {
    cout << "----- LiDAR: CreateFullviewPcd -----" << " Spot Index: " << spot_idx << endl;
    /** spot cloud **/
    CloudI::Ptr spot_cloud(new CloudI);
    string spot_cloud_path = file_path_vec[spot_idx][fullview_idx].spot_cloud_path;

    for (int i = 0; i < num_views; i++) {
        CloudI::Ptr view_cloud(new CloudI);
        string view_cloud_path = file_path_vec[spot_idx][i].view_cloud_path;
        LoadPcd(view_cloud_path, *view_cloud, "view");
        if (i != fullview_idx) {
            /** load icp pose transform matrix **/
            string pose_trans_mat_path = file_path_vec[spot_idx][i].pose_trans_mat_path;
            Mat4F pose_trans_mat = LoadTransMat(pose_trans_mat_path);
            cout << "Degree " << degree_map[i] << " ICP Mat: " << "\n" << pose_trans_mat << endl;
            /** transform point cloud **/
            pcl::transformPointCloud(*view_cloud, *view_cloud, pose_trans_mat);
        }
        *spot_cloud = *spot_cloud + *view_cloud;
    }

    /** check the original point cloud size **/
    cout << "size of original cloud:" << spot_cloud->points.size() << endl;

    /** radius outlier filter **/
    pcl::RadiusOutlierRemoval<PointI> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(spot_cloud);
    radius_outlier_filter.setRadiusSearch(0.10);
    radius_outlier_filter.setMinNeighborsInRadius(100);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.setKeepOrganized(false);
    radius_outlier_filter.filter(*spot_cloud);

    pcl::io::savePCDFileBinary(spot_cloud_path, *spot_cloud);
    cout << "Spot cloud generated." << endl;
}

void LidarProcess::SpotRegistration() {
    
    /** source index and target index **/
    int src_idx = spot_idx;
    int tgt_idx = spot_idx - 1;
    ROS_INFO("Registration: spot %d -> spot %d\n", src_idx, tgt_idx);

    /** load points **/
    CloudI::Ptr spot_cloud_tgt(new CloudI);
    CloudI::Ptr spot_cloud_src(new CloudI);
    string spot_cloud_tgt_path = file_path_vec[tgt_idx][0].spot_cloud_path;
    string spot_cloud_src_path = file_path_vec[src_idx][0].spot_cloud_path;
    LoadPcd(spot_cloud_tgt_path, *spot_cloud_tgt, "target spot");
    LoadPcd(spot_cloud_src_path, *spot_cloud_src, "source spot");

    /** initial transformation and initial score **/
    string lio_trans_path = file_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Mat4F lio_spot_trans_mat = LoadTransMat(lio_trans_path);
    
    /** ICP **/
    Mat4F align_spot_trans_mat = Align(spot_cloud_tgt, spot_cloud_src, lio_spot_trans_mat, 1, false);
    CloudI::Ptr spot_cloud_icp_trans(new CloudI);
    pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_icp_trans, align_spot_trans_mat);

    /** save the spot trans matrix by icp **/
    cout << file_path_vec[src_idx][0].icp_spot_trans_mat_path << endl;
    std::ofstream mat_out;
    mat_out.open(file_path_vec[src_idx][0].icp_spot_trans_mat_path);
    mat_out << align_spot_trans_mat << endl;
    mat_out.close();

    if (FULL_OUTPUT) {
        /** save the pair registered point cloud **/
        string pair_registered_cloud_path = file_path_vec[tgt_idx][0].fullview_recon_folder_path +
                                            "/icp_registered_spot_tgt_" + to_string(tgt_idx) + ".pcd";
        cout << pair_registered_cloud_path << endl;
        pcl::io::savePCDFileBinary(pair_registered_cloud_path, *spot_cloud_icp_trans + *spot_cloud_tgt);
    }
}

void LidarProcess::FineToCoarseReg() {

    cout << "----- LiDAR: FineToCoarseReg -----" << " Spot Index: " << spot_idx << endl;
    /** load points **/
    string lio_spot_trans_path = file_path_vec[spot_idx][0].lio_spot_trans_mat_path;
    string lio_static_trans_path = file_path_vec[0][0].fullview_recon_folder_path +
                                    "/lio_static_trans_mat.txt";
    string spot_cloud_path = file_path_vec[spot_idx][0].spot_cloud_path;
    string global_coarse_cloud_path = file_path_vec[0][0].fullview_recon_folder_path +
                                    "/scans.pcd";
    
    CloudI::Ptr spot_cloud(new CloudI);
    CloudI::Ptr global_coarse_cloud(new CloudI);
    Mat4F lio_spot_trans_mat = Mat4F::Identity();
    Mat4F lio_static_trans_mat = Mat4F::Identity();

    LoadPcd(spot_cloud_path, *spot_cloud, "spot");
    LoadPcd(global_coarse_cloud_path, *global_coarse_cloud, "global coarse");

    for (int load_idx = spot_idx; load_idx > 0; --load_idx) {
        string trans_file_path = file_path_vec[load_idx][0].lio_spot_trans_mat_path;
        Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
        lio_spot_trans_mat = tmp_spot_trans_mat * lio_spot_trans_mat;
    }
    cout << "Load spot LIO trans mat: \n" << lio_spot_trans_mat << endl;
    lio_static_trans_mat = LoadTransMat(lio_static_trans_path);
    cout << "Load static LIO trans mat: \n" << lio_static_trans_mat << endl;
    lio_spot_trans_mat = lio_static_trans_mat * lio_spot_trans_mat;
    cout << "Load spot LIO trans mat: \n" << lio_spot_trans_mat << endl;

    pcl::transformPointCloud(*spot_cloud, *spot_cloud, lio_spot_trans_mat);

    // Align(global_coarse_cloud, spot_cloud, lio_spot_trans_mat, 1, false);
}

void LidarProcess::GlobalColoredMapping(bool kGlobalUniformSampling) {
    /** global cloud registration **/
    const float radius = SAMPLING_RADIUS;
    CloudRGB::Ptr global_registered_rgb_cloud(new CloudRGB);
    string init_rgb_cloud_path = file_path_vec[0][0].spot_rgb_cloud_path;
    LoadPcd(init_rgb_cloud_path, *global_registered_rgb_cloud, "fullview rgb");
    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;
    for (int src_idx = 1; src_idx < num_spots; ++src_idx) {
        
        ROS_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudRGB::Ptr spot_cloud_src(new CloudRGB);

        /** load points **/
        string load_rgb_cloud_path = file_path_vec[src_idx][0].spot_rgb_cloud_path;
        LoadPcd(load_rgb_cloud_path, *spot_cloud_src, "fullview rgb");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = file_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
        }
        cout << "Load spot ICP trans mat: \n" << icp_spot_trans_mat << endl;
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);
        *global_registered_rgb_cloud += *spot_cloud_src;
    }

    if (kGlobalUniformSampling) {
        /** down sampling **/
        pcl::UniformSampling<PointRGB> us;
        us.setRadiusSearch(radius);
        us.setInputCloud(global_registered_rgb_cloud);
        us.filter(*global_registered_rgb_cloud);
    }

    string global_registered_cloud_path = file_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_rgb_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_rgb_cloud);
}

void LidarProcess::GlobalMapping(bool kGlobalUniformSampling) {
    /** global cloud registration **/
    const float radius = SAMPLING_RADIUS;

    CloudI::Ptr global_registered_cloud(new CloudI);
    string init_dense_cloud_path = file_path_vec[0][0].spot_cloud_path;
    cout << init_dense_cloud_path << endl;
    LoadPcd(init_dense_cloud_path, *global_registered_cloud, "fullview dense");

    if (kColorMap) {
        for (auto & pt : global_registered_cloud->points) {
            pt.intensity = 40;
        }
    }

    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;

    for (int src_idx = tgt_idx + 1; src_idx < num_spots; ++src_idx) {
        ROS_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudI::Ptr spot_cloud_src(new CloudI);

        /** load points **/
        string load_dense_cloud_path = file_path_vec[src_idx][0].spot_cloud_path;
        LoadPcd(load_dense_cloud_path, *spot_cloud_src, "fullview dense");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = file_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
        }
        cout << "Load spot ICP trans mat: \n" << icp_spot_trans_mat << endl;
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);

        /** for view coloring & viz only **/
        if (kColorMap) {
            for (auto & pt : spot_cloud_src->points) {
                pt.intensity = (src_idx + 1) * 40;
            }
        }
        
        *global_registered_cloud += *spot_cloud_src;
    }

    if (kGlobalUniformSampling) {
        /** down sampling **/
        pcl::UniformSampling<PointI> us;
        us.setRadiusSearch(radius);
        us.setInputCloud(global_registered_cloud);
        us.filter(*global_registered_cloud);
    }

    string global_registered_cloud_path = file_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_cloud);
}

double LidarProcess::GetFitnessScore(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, float max_range) {
    double fitness_score = 0.0;
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    // For each point in the source dataset
    int nr = 0;
    pcl::KdTreeFLANN<PointI> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    #pragma omp parallel for num_threads(THREADS)
    for (auto &pt : cloud_src->points) {
        // Find its nearest neighbor in the target
        kdtree.nearestKSearch(pt, 1, nn_indices, nn_dists);
        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= max_range) {
            // Add to the fitness score
            fitness_score += nn_dists[0];
            nr++;
        }
    }

    if (nr > 0)
        return (fitness_score / nr);
    return (std::numeric_limits<double>::max());
}

void LidarProcess::RemoveInvalidPoints(CloudI::Ptr cloud){
    std::vector<int> null_indices;
    (*cloud).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, null_indices);
}
