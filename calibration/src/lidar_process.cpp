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
    /** create objects, initialization **/
    string pose_folder_path_temp;
    PoseFilePath pose_files_path_temp;
    EdgePixels edge_pixels_temp;
    EdgeCloud::Ptr edge_cloud_temp;
    TagsMap tags_map_temp;
    Mat4F pose_trans_mat_temp;
    for (int i = 0; i < num_spots; ++i) {
        vector<string> poses_folder_path_vec_temp;
        vector<PoseFilePath> poses_file_path_vec_temp;
        vector<EdgePixels> edge_pixels_vec_temp;
        vector<EdgeCloud::Ptr> edge_cloud_vec_temp;
        vector<TagsMap> tags_map_vec_temp;
        vector<Mat4F> poses_trans_mat_vec_temp;
        for (int j = 0; j < num_views; ++j) {
            poses_folder_path_vec_temp.push_back(pose_folder_path_temp);
            poses_file_path_vec_temp.push_back(pose_files_path_temp);
            edge_pixels_vec_temp.push_back(edge_pixels_temp);
            edge_cloud_vec_temp.push_back(edge_cloud_temp);
            tags_map_vec_temp.push_back(tags_map_temp);
            poses_trans_mat_vec_temp.push_back(pose_trans_mat_temp);
        }
        poses_folder_path_vec.push_back(poses_folder_path_vec_temp);
        poses_files_path_vec.push_back(poses_file_path_vec_temp);
        edge_pixels_vec.push_back(edge_pixels_vec_temp);
        edge_cloud_vec.push_back(edge_cloud_vec_temp);
        tags_map_vec.push_back(tags_map_vec_temp);
        pose_trans_mat_vec.push_back(poses_trans_mat_vec_temp);
    }

    // vector<vector<string>> poses_folder_path_vec_tmp(num_spots, vector<string>(num_views));
    // vector<vector<PoseFilePath>> poses_file_path_vec_tmp(num_spots, vector<PoseFilePath>(num_views));
    // vector<vector<EdgePixels>> edge_pixels_vec_tmp(num_spots, vector<EdgePixels>(num_views));
    // vector<vector<EdgeCloud::Ptr>> edge_cloud_vec_tmp(num_spots, vector<EdgeCloud::Ptr>(num_views));
    // vector<vector<TagsMap>> tags_map_vec_tmp(num_spots, vector<TagsMap>(num_views));
    // vector<vector<Mat4F>> poses_trans_mat_vec_tmp(num_spots, vector<Mat4F>(num_views));
    // poses_folder_path_vec = poses_folder_path_vec_tmp;
    // poses_file_path_vec = poses_file_path_vec_tmp;
    // edge_pixels_vec = edge_pixels_vec_tmp;
    // edge_cloud_vec = edge_cloud_vec_tmp;
    // tags_map_vec = tags_map_vec_tmp;
    // poses_trans_mat_vec = poses_trans_mat_vec_tmp;


    for (int i = 0; i < num_spots; ++i) {
        string spot_path = kDatasetPath + "/spot" + to_string(i);

        for (int j = 0; j < num_views; ++j) {
            int v_degree = view_angle_init + view_angle_step * j;
            degree_map[j] = v_degree;
            poses_folder_path_vec[i][j] = spot_path + "/" + to_string(v_degree);
            struct PoseFilePath pose_file_path(spot_path, poses_folder_path_vec[i][j]);
            poses_files_path_vec[i][j] = pose_file_path;
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

    uint32_t num_pcds = (float)cnt_pcds * ((float)5 / 6);
    uint32_t idx_start = (cnt_pcds - num_pcds)/2;
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

    string fullview_cloud_path = poses_files_path_vec[spot_idx][view_idx].spot_cloud_path;
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

    if (kEdgeAnalysis) {
        /** visualization for weight check**/
        pcl::io::savePCDFileBinary(poses_files_path_vec[spot_idx][view_idx].output_folder_path + "/fullview_polar_cloud.pcd", *polar_cloud);
    }

}

void LidarProcess::SphereToPlane(CloudI::Ptr& polar_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << " Spot Index: " << spot_idx << endl;
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(kFlatRows, kFlatCols, CV_32FC1); /** define the flat image **/
    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointI> kdtree;
    CloudI::Ptr polar_flat_cloud(new CloudI);
    pcl::copyPointCloud(*polar_cloud, *polar_flat_cloud);
    for (auto &pt : polar_flat_cloud->points) {pt.z = 0;}
    kdtree.setInputCloud(polar_flat_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kScale = sqrt(2);
    const float kSearchRadius = kScale * (kRadPerPix / 2);
    const float sensitivity = 0.02f;

    /** Multiprocessing test **/
    #pragma omp parallel for num_threads(16)

    for (int u = 0; u < kFlatRows; ++u) {
        float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;
        for (int v = 0; v < kFlatCols; ++v) {
            float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;

            /** assign the theta and phi center to the search_center **/
            PointI search_center;
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
                invalid_search_num ++;
                tags_map[u][v].num_pts = 0;
                tags_map[u][v].pts_indices = {};
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
                tags_map[u][v].num_pts = local_vec.size();
                tags_map[u][v].pts_indices.insert(tags_map[u][v].pts_indices.begin(), local_vec.data(), local_vec.data()+local_vec.size());

                if (tags_map[u][v].num_pts > 0) {
                    flat_img.at<float>(u, v) = intensity_mean / tags_map[u][v].num_pts;
                }
                else {
                    flat_img.at<float>(u, v) = 0;
                }
                
                ROS_ASSERT_MSG((search_num - hidden_pt_num == tags_map[u][v].num_pts), "size of the vectors in a pixel region is not the same!");
            }
        }
    }

    /** add the tags_map of this specific pose to maps **/
    tags_map_vec[spot_idx][view_idx] = tags_map;

    string flat_img_path = poses_files_path_vec[spot_idx][view_idx].flat_img_path;
    cout << "LiDAR flat image path: " << flat_img_path << endl;
    cv::imwrite(flat_img_path, flat_img);

}

void LidarProcess::EdgeExtraction() {
    std::string script_path = kPkgPath + "/python_scripts/image_process/edge_extraction.py";
    std::string kSpots = to_string(spot_idx);
    std::string cmd_str = "python3 " + script_path + " " + kDatasetPath + " " + "lidar" + " " + kSpots;
    system(cmd_str.c_str());
}

void LidarProcess::EdgeToPixel() {
    /** generate edge_pixels and push back into edge_pixels_vec **/
    cout << "----- LiDAR: EdgeToPixel -----" << " Spot Index: " << spot_idx << endl;
    string edge_img_path = this -> poses_files_path_vec[spot_idx][view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nView Index: %d \nPath: %s", view_idx, edge_img_path.c_str());
    ROS_ASSERT_MSG((edge_img.rows == kFlatRows || edge_img.cols == kFlatCols), "size of original fisheye image is incorrect! View Index: %d", view_idx);

    EdgePixels edge_pixels;
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                pcl::PointXYZ edge_pixel;
                edge_pixel.x = u;
                edge_pixel.y = v;
                edge_pixel.z = 0;
                edge_pixels.push_back(edge_pixel);
            }
        }
    }
    edge_pixels_vec[spot_idx][view_idx] = edge_pixels;
}

void LidarProcess::PixLookUp(CloudI::Ptr& cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << " Spot Index: " << spot_idx << endl;
    int num_invalid_pixels = 0;
    
    TagsMap tags_map = tags_map_vec[spot_idx][view_idx];
    EdgePixels edge_pixels = edge_pixels_vec[spot_idx][view_idx];
    EdgeCloud::Ptr edge_cloud(new EdgeCloud);

    for (auto &edge_pixel : edge_pixels) {
        int u = edge_pixel.x;
        int v = edge_pixel.y;
        int num_pts = tags_map[u][v].num_pts;
        if (num_pts == 0) { /** invalid pixels **/
            num_invalid_pixels ++;
        }
        else { /** normal pixels **/
            float x_avg = 0, y_avg = 0, z_avg = 0;
            for (int i = 0; i < num_pts; ++i) {
                if (tags_map[u][v].pts_indices[i] < cart_cloud->size()) {
                    PointI &pixel_pt = cart_cloud->points[tags_map[u][v].pts_indices[i]];
                    x_avg += pixel_pt.x;
                    y_avg += pixel_pt.y;
                    z_avg += pixel_pt.z;
                }
            }

            /** average coordinates->unbiased estimation of center position **/
            x_avg = x_avg / num_pts;
            y_avg = y_avg / num_pts;
            z_avg = z_avg / num_pts;
            
            /** store the spatial coordinates into vector **/
            pcl::PointXYZ pt;
            pt.x = x_avg;
            pt.y = y_avg;
            pt.z = z_avg;
            // pt.intensity = 1; /** note: I is used to store the point weight **/
            edge_cloud->points.push_back(pt);
        }
    }
    cout << "number of invalid lookups(lidar): " << num_invalid_pixels << endl;
    edge_cloud_vec[spot_idx][view_idx] = edge_cloud;

    /** save edge coordinates into .txt file **/
    string edge_pts_coordinates_path = poses_files_path_vec[spot_idx][view_idx].edge_pts_coordinates_path;
    ofstream outfile;
    outfile.open(edge_pts_coordinates_path, ios::out);
    if (!outfile.is_open()) {
        cout << "Open file failure" << endl;
    }
    for (auto &point : edge_cloud->points) {
        outfile << point.x << "\t"
                << point.y << "\t"
                << point.z << endl;
    }
    outfile.close();

    if (kEdgeAnalysis) {
        /** visualization for weight check**/
        string edge_cart_pcd_path = this -> poses_files_path_vec[spot_idx][view_idx].edge_cart_pcd_path;
        cout << edge_cart_pcd_path << endl;
        pcl::io::savePCDFileBinary(edge_cart_pcd_path, *edge_cloud);
    }

}

void LidarProcess::ReadEdge() {
    cout << "----- LiDAR: ReadEdge -----" << " Spot Index: " << spot_idx << " View Index: " << view_idx << endl;
    string edge_cloud_txt_path = poses_files_path_vec[spot_idx][view_idx].edge_pts_coordinates_path;
    vector<vector<double>> edge_pts;
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
        if (v.size() == 3) {
            edge_pts.push_back(v);
        }
    }

    ROS_ASSERT_MSG(!edge_pts.empty(), "LiDAR Read Edge Incorrect! View Index: %d", view_idx);
    cout << "Imported LiDAR points: " << edge_pts.size() << endl;

    /** remove duplicated points **/
    std::sort(edge_pts.begin(), edge_pts.end());
    edge_pts.erase(unique(edge_pts.begin(), edge_pts.end()), edge_pts.end());
    cout << "LiDAR Edge Points after Duplicated Removed: " << edge_pts.size() << endl;

    /** construct pcl point cloud **/
    pcl::PointXYZ pt;
    EdgeCloud::Ptr edge_cloud(new EdgeCloud);
    for (auto &edge_pt : edge_pts) {
        pt.x = edge_pt[0];
        pt.y = edge_pt[1];
        pt.z = edge_pt[2];
        edge_cloud->points.push_back(pt);
    }
    cout << "Filtered LiDAR points: " << edge_cloud->points.size() << endl;
    edge_cloud_vec[spot_idx][view_idx] = edge_cloud;
}

/** Point Cloud Registration **/
Mat4F LidarProcess::Align(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, Mat4F init_trans_mat, int cloud_type, const bool kIcpViz) {
    /** params **/
    float uniform_radius = 0.05;
    float normal_radius = 0.1;

    int max_iters = 200;
    float max_corr_dis = 0.5;
    float trans_epsilon = 1e-12;
    float eucidean_epsilon = 1e-12;
    float max_fitness_range = 2.0;

    bool enable_auto_radius = true;
    bool auto_radius_trig = false;
    float target_size = 2.5e6;

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
    normal_radius = uniform_radius * 5;
    PCL_INFO("Uniform sampling for target cloud: %d -> %d\n", cloud_tgt->size(), cloud_us_tgt->size());
    PCL_INFO("Uniform sampling for source cloud: %d -> %d\n", cloud_src->size(), cloud_us_src->size());
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
        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);

        pcl::KdTreeFLANN<PointI> kdtree_tgt;
        pcl::KdTreeFLANN<PointI> kdtree_src;
        kdtree_tgt.setInputCloud (cloud_us_tgt);
        kdtree_src.setInputCloud (cloud_us_src);
        #pragma omp parallel for num_threads(16)
        for (int idx = 0; idx < cloud_us_src->size(); ++idx) {
            kdtree_tgt.nearestKSearch (cloud_us_src->points[idx], 1, nn_indices, nn_dists);
            if (nn_dists[0] <= max_fitness_range) {
                src_indices[idx] = idx;
                tgt_indices[nn_indices[0]] = nn_indices[0];
            }
        }
        #pragma omp parallel for num_threads(16)
        for (int idx = 0; idx < cloud_us_tgt->size(); ++idx) {
            if (tgt_indices[idx] == 0) {
                kdtree_src.nearestKSearch (cloud_us_tgt->points[idx], 1, nn_indices, nn_dists);
                tgt_indices[idx] = (nn_dists[0] <= max_fitness_range) ? idx : 0;
            }
        }
    }
    tgt_indices.erase(std::remove(tgt_indices.begin(), tgt_indices.end(), 0), tgt_indices.end());
    src_indices.erase(std::remove(src_indices.begin(), src_indices.end(), 0), src_indices.end());
    pcl::copyPointCloud(*cloud_us_tgt, tgt_indices, *cloud_us_tgt_effe);
    pcl::copyPointCloud(*cloud_us_src, src_indices, *cloud_us_src_effe);
    PCL_INFO("Effective filter for target cloud: %d -> %d\n", cloud_us_tgt->size(), cloud_us_tgt_effe->size());
    PCL_INFO("Effective filter for source cloud: %d -> %d\n", cloud_us_src->size(), cloud_us_src_effe->size());
    PCL_INFO("Run time: %f s\n", timer.getTimeSeconds());

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
    PCL_INFO("Normal estimation ... \n");

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
    PCL_INFO("Run time: %f s\n", timer.getTimeSeconds());
    
    timer.reset();
    PCL_INFO("ICP alignment ... \n");
    using PointToPlane = pcl::registration::TransformationEstimationPointToPlaneLLS<PointIN, PointIN>;
    pcl::IterativeClosestPoint<PointIN, PointIN> align;
    PointToPlane::Ptr point_to_plane(new PointToPlane);
    Mat4F align_trans_mat;

    align.setInputSource(cloud_src_in);
    align.setInputTarget(cloud_tgt_in);
    align.setTransformationEstimation(point_to_plane);
    align.setMaximumIterations(max_iters);
    align.setMaxCorrespondenceDistance(max_corr_dis);
    align.setTransformationEpsilon(trans_epsilon);
    align.setEuclideanFitnessEpsilon(eucidean_epsilon);
    align.align(*cloud_icp_trans_n, init_trans_mat);
    pcl::copyPointCloud(*cloud_icp_trans_n, *cloud_icp_trans_us);

     if (align.hasConverged()) {
        PCL_INFO("ICP: Converged in %f s.\n", timer.getTimeSeconds());
        PCL_INFO("Fitness score: %f \n", GetFitnessScore(cloud_us_tgt_effe, cloud_icp_trans_us, max_fitness_range));
        PCL_INFO("Epsilon: %f \n", align.getEuclideanFitnessEpsilon());
        align_trans_mat = align.getFinalTransformation();
        cout << align_trans_mat << endl;
    }
    else {
        PCL_ERROR("ICP: Fail to converge. \n");
    }

    /** visualization **/
    if (kIcpViz) {
        pcl::visualization::PCLVisualizer viewer("ICP demo");
        int v1(0), v2(1); /** create two view point **/
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        float bkg_grayscale = 0.0;  /** black **/
        float txt_grayscale = 1.0 - bkg_grayscale;

        /** the color of original target cloud is white **/
        pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_aim_color_h(cloud_us_tgt_effe, (int)255 * txt_grayscale,
                                                                                    (int)255 * txt_grayscale,
                                                                                    (int)255 * txt_grayscale);
        viewer.addPointCloud(cloud_us_tgt_effe, cloud_aim_color_h, "cloud_aim_v1", v1);
        viewer.addPointCloud(cloud_us_tgt_effe, cloud_aim_color_h, "cloud_aim_v2", v2);

        /** the color of original source cloud is green **/
        pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_in_color_h(cloud_init_trans_us, 20, 180, 20);
        viewer.addPointCloud(cloud_init_trans_us, cloud_in_color_h, "cloud_in_v1", v1);

        /** the color of transformed source cloud with icp result is red **/
        pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_icped_color_h(cloud_icp_trans_us, 180, 20, 20);
        viewer.addPointCloud(cloud_icp_trans_us, cloud_icped_color_h, "cloud_icped_v2", v2);

        viewer.addCoordinateSystem();

        while (!viewer.wasStopped()) {
            viewer.spinOnce();
        }
    }

    PCL_INFO("Align completed.\n");
    return align_trans_mat;
}

void LidarProcess::DistanceAnalysis(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, float uniform_radius, float max_range) {

    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);

    /** uniform sampling **/
    if (uniform_radius > 0) {
        pcl::UniformSampling<PointI> us;
        us.setRadiusSearch(uniform_radius);
        us.setInputCloud(cloud_tgt);
        us.filter(*cloud_us_tgt);
        us.setInputCloud(cloud_src);
        us.filter(*cloud_us_src);
    }
    else {
        pcl::copyPointCloud(*cloud_src, *cloud_us_src);
        pcl::copyPointCloud(*cloud_tgt, *cloud_us_tgt);
    }

    /** invalid point filter **/
    RemoveInvalidPoints(cloud_us_tgt);
    RemoveInvalidPoints(cloud_us_src);

    CloudI::Ptr cloud_us_tgt_effe (new CloudI);
    CloudI::Ptr cloud_us_src_effe (new CloudI);

    // cout << "k search effective filter" << endl;
    pcl::StopWatch timer_effe;
    std::vector<int> src_effe_indices(cloud_us_src->size());
    std::vector<int> tgt_effe_indices(cloud_us_tgt->size());
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    pcl::KdTreeFLANN<PointI> kdtree_tgt;
    kdtree_tgt.setInputCloud (cloud_us_tgt);
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < cloud_us_src->size(); ++i) {
        kdtree_tgt.nearestKSearch (cloud_us_src->points[i], 1, nn_indices, nn_dists);
        src_effe_indices[i] = (nn_dists[0] <= max_range) ? i : 0;
    }
    src_effe_indices.erase(std::remove(src_effe_indices.begin(), src_effe_indices.end(), 0), src_effe_indices.end());
    pcl::copyPointCloud(*cloud_us_src, src_effe_indices, *cloud_us_src_effe);

    pcl::KdTreeFLANN<PointI> kdtree_src;
    kdtree_src.setInputCloud (cloud_us_src);
    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < cloud_us_tgt->size(); ++i) {
        kdtree_src.nearestKSearch (cloud_us_tgt->points[i], 1, nn_indices, nn_dists);
        tgt_effe_indices[i] = (nn_dists[0] <= max_range) ? i : 0;
    }
    tgt_effe_indices.erase(std::remove(tgt_effe_indices.begin(), tgt_effe_indices.end(), 0), tgt_effe_indices.end());
    pcl::copyPointCloud(*cloud_us_tgt, tgt_effe_indices, *cloud_us_tgt_effe);

    pcl::StopWatch timer_fs;
    cout << "Coarse to fine fitness score: " << GetFitnessScore(cloud_us_tgt_effe, cloud_us_src_effe, max_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;
}

void LidarProcess::CreateDensePcd() {
    cout << "----- LiDAR: CreateDensePcd -----" << " Spot Index: " << spot_idx << " View Index: " << view_idx << endl;

    string pcd_path = poses_files_path_vec[spot_idx][view_idx].view_cloud_path;
    string bag_path = poses_files_path_vec[spot_idx][view_idx].bag_folder_path 
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

    /** check the pass through filtered point cloud size **/
    cout << "size of cloud after a condition filter:" << view_cloud->points.size() << endl;

    pcl::io::savePCDFileBinary(pcd_path, *view_cloud);
    cout << "Create Dense Point Cloud File Successfully!" << endl;
}

void LidarProcess::ViewRegistration() {
    cout << "----- LiDAR: ViewRegistration -----" << " Spot Index: " << spot_idx << " View Index: " << view_idx << endl;
    /** load point clouds to be registered **/
    std::string tgt_pcd_path = poses_files_path_vec[spot_idx][fullview_idx].view_cloud_path;
    std::string src_pcd_path = poses_files_path_vec[spot_idx][view_idx].view_cloud_path;
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
    align_trans_mat = Align(view_cloud_tgt, view_cloud_src, init_trans_mat, 1, false);
    CloudI::Ptr view_cloud_icp_trans(new CloudI);
    pcl::transformPointCloud(*view_cloud_src, *view_cloud_icp_trans, align_trans_mat);

    /** save the view trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(poses_files_path_vec[spot_idx][view_idx].pose_trans_mat_path);
    mat_out << align_trans_mat << endl;
    mat_out.close();

    /** save the registered point clouds **/
    string registered_cloud_path = poses_files_path_vec[spot_idx][view_idx].fullview_recon_folder_path +
                                   "/icp_registered_" + to_string(v_angle) + ".pcd";
    pcl::io::savePCDFileBinary(registered_cloud_path, *view_cloud_icp_trans + *view_cloud_tgt);
}

void LidarProcess::FullViewMapping() {
    cout << "----- LiDAR: CreateFullviewPcd -----" << " Spot Index: " << spot_idx << endl;
    /** spot cloud **/
    CloudI::Ptr spot_cloud(new CloudI);
    string spot_cloud_path = poses_files_path_vec[spot_idx][fullview_idx].spot_cloud_path;

    vector<Mat4F> pose_trans_vec (5, Mat4F::Identity());
    vector<float> eval_radius (5, 0);
    float eval_sum = 0;
    for (int i = 0; i < num_views; i++) {
        if (i != fullview_idx) {
            /** load icp pose transform matrix **/
            string pose_trans_mat_path = poses_files_path_vec[spot_idx][i].pose_trans_mat_path;
            Mat4F pose_trans_mat = LoadTransMat(pose_trans_mat_path);
            cout << "Degree " << degree_map[i] << " ICP Mat: " << "\n" << pose_trans_mat << endl;
            pose_trans_vec[i] = pose_trans_mat;

            Mat3F rot_matrix = pose_trans_mat.topLeftCorner(3, 3);
            float radius = (Vec4F(0, 0, 0.15, -1) + pose_trans_mat.col(3)).norm();
            eval_radius[i] = radius;
            eval_sum += radius;
        }
    }

    for (int i = 0; i < num_views; i++) {
        /** matrix check **/
        if (i == 0 || i == num_views - 1) {
            if (abs((eval_radius[i] - 0.15f) > 0.03)) {
                PCL_INFO("radius check failed for degree %i.\n", degree_map[i]);
                Mat4F remap_mat = (i == 0) ? pose_trans_vec[num_views - 1].inverse() : pose_trans_vec[0].inverse();
                for (int j = 1; j < fullview_idx; j++) {
                    if (i == 0) {
                        remap_mat = remap_mat * pose_trans_vec[fullview_idx + j] * pose_trans_vec[fullview_idx - j];
                    }
                    else {
                        remap_mat = remap_mat * pose_trans_vec[fullview_idx - j] * pose_trans_vec[fullview_idx + j];
                    }
                }
                    
                cout << "Re-calculated matrix: \n" << remap_mat << endl;
                pose_trans_vec[i] = remap_mat;
            }
        }

        /** transform point cloud **/
        CloudI::Ptr view_cloud(new CloudI);
        string view_cloud_path = poses_files_path_vec[spot_idx][i].view_cloud_path;
        LoadPcd(view_cloud_path, *view_cloud, "view");
        if (i != fullview_idx) {
            pcl::transformPointCloud(*view_cloud, *view_cloud, pose_trans_vec[i]);
        }
        *spot_cloud = *spot_cloud + *view_cloud;
    }

    /** check the original point cloud size **/
    int fullview_cloud_size = spot_cloud->points.size();
    cout << "size of original cloud:" << fullview_cloud_size << endl;

    /** radius outlier filter **/
    pcl::RadiusOutlierRemoval<PointI> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(spot_cloud);
    radius_outlier_filter.setRadiusSearch(0.1);
    radius_outlier_filter.setMinNeighborsInRadius(100);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.setKeepOrganized(false);
    radius_outlier_filter.filter(*spot_cloud);

    /** radius outlier filter cloud size check **/
    int radius_outlier_cloud_size = spot_cloud->points.size();
    cout << "radius outlier filtered cloud size:" << spot_cloud->points.size() << endl;

    pcl::io::savePCDFileBinary(spot_cloud_path, *spot_cloud);
    cout << "Spot cloud generated." << endl;
}

void LidarProcess::SpotRegistration() {
    
    /** source index and target index **/
    int src_idx = spot_idx;
    int tgt_idx = spot_idx - 1;
    PCL_INFO("Registration: spot %d -> spot %d\n", src_idx, tgt_idx);

    /** load points **/
    CloudI::Ptr spot_cloud_tgt(new CloudI);
    CloudI::Ptr spot_cloud_src(new CloudI);
    string spot_cloud_tgt_path = poses_files_path_vec[tgt_idx][0].spot_cloud_path;
    string spot_cloud_src_path = poses_files_path_vec[src_idx][0].spot_cloud_path;
    LoadPcd(spot_cloud_tgt_path, *spot_cloud_tgt, "target spot");
    LoadPcd(spot_cloud_src_path, *spot_cloud_src, "source spot");

    /** initial transformation and initial score **/
    string lio_trans_path = poses_files_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Mat4F lio_spot_trans_mat = LoadTransMat(lio_trans_path);
    
    /** ICP **/
    Mat4F align_spot_trans_mat = Align(spot_cloud_tgt, spot_cloud_src, lio_spot_trans_mat, 1, false);
    CloudI::Ptr spot_cloud_icp_trans(new CloudI);
    pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_icp_trans, align_spot_trans_mat);

    /** compared the fitness score of lio and icp **/
//    CloudI::Ptr spot_lio_trans (new CloudI);
//    CloudI::Ptr spot_icp_trans (new CloudI);
//    pcl::transformPointCloud(*spot_cloud_src, *spot_lio_trans, lio_spot_trans_mat);
//    pcl::transformPointCloud(*spot_cloud_src, *spot_icp_trans, icp_spot_trans_mat);
//    cout << "Spot Registration Fast-LIO Fitness Score: " << GetFitnessScore(spot_cloud_tgt, spot_lio_trans, 2.0) << endl;
//    cout << "Spot Registration ICP Fitness Score: " << GetIcpFitnessScore(spot_cloud_tgt, spot_icp_trans, 2.0) << endl;

    /** save the spot trans matrix by icp **/
    cout << poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path << endl;
    std::ofstream mat_out;
    mat_out.open(poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path);
    mat_out << align_spot_trans_mat << endl;
    mat_out.close();

    /** save the pair registered point cloud **/
    string pair_registered_cloud_path = poses_files_path_vec[tgt_idx][0].fullview_recon_folder_path +
                                        "/icp_registered_spot_tgt_" + to_string(tgt_idx) + ".pcd";
    cout << pair_registered_cloud_path << endl;
    pcl::io::savePCDFileBinary(pair_registered_cloud_path, *spot_cloud_icp_trans + *spot_cloud_tgt);
}

void LidarProcess::SpotRegAnalysis(int tgt_spot_idx, int src_spot_idx) {
    /** params **/
    float uniform_radius = 0.05;
    bool enable_auto_radius = true;
    bool auto_radius_trig = false;
    float target_size = 2.5e6;

    float max_fitness_range = 2;

    /** load icp transformation matrix **/
    Mat4F icp_spot_trans_mat = Mat4F::Identity();
    for (int load_idx = src_spot_idx; load_idx > 0; --load_idx) {
        string trans_file_path = this->poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
        Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
        icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
    }
    cout << "Load spot ICP trans mat: \n" << icp_spot_trans_mat << endl;

    /** load lio transformation matrix **/
    Mat4F lio_spot_trans_mat = Mat4F::Identity();
    for (int load_idx = src_spot_idx; load_idx > 0; --load_idx) {
        string trans_file_path = this->poses_files_path_vec[load_idx][0].lio_spot_trans_mat_path;
        Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
        lio_spot_trans_mat = tmp_spot_trans_mat * lio_spot_trans_mat;
    }
    cout << "Load spot LIO trans mat: \n" << lio_spot_trans_mat << endl;

    CloudI::Ptr cloud_tgt (new CloudI);
    CloudI::Ptr cloud_src (new CloudI);
    string tgt_spot_path = poses_files_path_vec[tgt_spot_idx][0].spot_cloud_path;
    string src_spot_path = poses_files_path_vec[src_spot_idx][0].spot_cloud_path;
    LoadPcd(tgt_spot_path, *cloud_tgt);
    LoadPcd(src_spot_path, *cloud_src);

    pcl::StopWatch timer;

    /** uniform sampling **/
    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);
    CloudI::Ptr cloud_src_init_trans (new CloudI);
    pcl::UniformSampling<PointI> us;
    pcl::transformPointCloud(*cloud_src, *cloud_src_init_trans, lio_spot_trans_mat);
    while (cloud_us_tgt->size() + cloud_us_src->size() < 2e6 && uniform_radius >= 0.005){
        us.setRadiusSearch(uniform_radius);
        us.setInputCloud(cloud_tgt);
        us.filter(*cloud_us_tgt);
        us.setInputCloud(cloud_src_init_trans);
        us.filter(*cloud_us_src);
        uniform_radius *= sqrt((cloud_us_tgt->size() + cloud_us_src->size()) / (2 * target_size));
        if (auto_radius_trig) {break;}
        auto_radius_trig = true;
    }
    pcl::transformPointCloud(*cloud_us_src, *cloud_us_src, lio_spot_trans_mat.inverse());
    PCL_INFO("Uniform sampling for target cloud: %d -> %d\n", cloud_tgt->size(), cloud_us_tgt->size());
    PCL_INFO("Uniform sampling for source cloud: %d -> %d\n", cloud_src->size(), cloud_us_src->size());

    /** invalid point filter **/
    RemoveInvalidPoints(cloud_us_tgt);
    RemoveInvalidPoints(cloud_us_src);

    CloudI::Ptr cloud_us_tgt_effe (new CloudI);
    CloudI::Ptr cloud_us_src_effe (new CloudI);
    std::vector<int> tgt_indices(cloud_us_tgt->size());
    std::vector<int> src_indices(cloud_us_src->size());
    timer.reset();

    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    pcl::KdTreeFLANN<PointI> kdtree_tgt;
    pcl::KdTreeFLANN<PointI> kdtree_src;
    kdtree_tgt.setInputCloud (cloud_us_tgt);
    kdtree_src.setInputCloud (cloud_us_src);
    #pragma omp parallel for num_threads(16)
    for (int idx = 0; idx < cloud_us_src->size(); ++idx) {
        kdtree_tgt.nearestKSearch (cloud_us_src->points[idx], 1, nn_indices, nn_dists);
        if (nn_dists[0] <= max_fitness_range) {
            src_indices[idx] = idx;
            tgt_indices[nn_indices[0]] = nn_indices[0];
        }
    }
    #pragma omp parallel for num_threads(16)
    for (int idx = 0; idx < cloud_us_tgt->size(); ++idx) {
        if (tgt_indices[idx] == 0) {
            kdtree_src.nearestKSearch (cloud_us_tgt->points[idx], 1, nn_indices, nn_dists);
            tgt_indices[idx] = (nn_dists[0] <= max_fitness_range) ? idx : 0;
        }
    }
    
    tgt_indices.erase(std::remove(tgt_indices.begin(), tgt_indices.end(), 0), tgt_indices.end());
    src_indices.erase(std::remove(src_indices.begin(), src_indices.end(), 0), src_indices.end());
    pcl::copyPointCloud(*cloud_us_tgt, tgt_indices, *cloud_us_tgt_effe);
    pcl::copyPointCloud(*cloud_us_src, src_indices, *cloud_us_src_effe);
    PCL_INFO("Effective filter for target cloud: %d -> %d\n", cloud_us_tgt->size(), cloud_us_tgt_effe->size());
    PCL_INFO("Effective filter for source cloud: %d -> %d\n", cloud_us_src->size(), cloud_us_src_effe->size());
    PCL_INFO("Run time: %f s\n", timer.getTimeSeconds());

    /** invalid point filter **/
    RemoveInvalidPoints(cloud_us_tgt_effe);
    RemoveInvalidPoints(cloud_us_src_effe);

    /** get the init trans cloud & init fitness score **/
    CloudI::Ptr cloud_init_trans_us (new CloudI);
    pcl::transformPointCloud(*cloud_us_src_effe, *cloud_init_trans_us, icp_spot_trans_mat);
    cout << "ICP Fitness Score: " << GetFitnessScore(cloud_us_tgt_effe, cloud_init_trans_us, max_fitness_range) << endl;

    /** get the init trans cloud & init fitness score **/
    pcl::transformPointCloud(*cloud_us_src_effe, *cloud_init_trans_us, lio_spot_trans_mat);
    cout << "LIO Fitness Score: " << GetFitnessScore(cloud_us_tgt_effe, cloud_init_trans_us, max_fitness_range) << endl;


}

void LidarProcess::FineToCoarseReg() {

    cout << "----- LiDAR: FineToCoarseReg -----" << " Spot Index: " << spot_idx << endl;
    /** load points **/
    string lio_spot_trans_path = poses_files_path_vec[spot_idx][0].lio_spot_trans_mat_path;
    string lio_static_trans_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                    "/lio_static_trans_mat.txt";
    string spot_cloud_path = poses_files_path_vec[spot_idx][0].spot_cloud_path;
    string global_coarse_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                    "/scans.pcd";
    
    CloudI::Ptr spot_cloud(new CloudI);
    CloudI::Ptr global_coarse_cloud(new CloudI);
    Mat4F lio_spot_trans_mat = Mat4F::Identity();
    Mat4F lio_static_trans_mat = Mat4F::Identity();

    LoadPcd(spot_cloud_path, *spot_cloud, "spot");
    LoadPcd(global_coarse_cloud_path, *global_coarse_cloud, "global coarse");

    for (int load_idx = spot_idx; load_idx > 0; --load_idx) {
        string trans_file_path = poses_files_path_vec[load_idx][0].lio_spot_trans_mat_path;
        Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
        lio_spot_trans_mat = tmp_spot_trans_mat * lio_spot_trans_mat;
    }
    cout << "Load spot LIO trans mat: \n" << lio_spot_trans_mat << endl;
    lio_static_trans_mat = LoadTransMat(lio_static_trans_path);
    cout << "Load static LIO trans mat: \n" << lio_static_trans_mat << endl;
    lio_spot_trans_mat = lio_static_trans_mat * lio_spot_trans_mat;
    cout << "Load spot LIO trans mat: \n" << lio_spot_trans_mat << endl;

    pcl::transformPointCloud(*spot_cloud, *spot_cloud, lio_spot_trans_mat);

    DistanceAnalysis(global_coarse_cloud, spot_cloud, 0.01, 0.5);

    // Align(global_coarse_cloud, spot_cloud, lio_spot_trans_mat, 1, false);
}

void LidarProcess::GlobalColoredMapping() {
    /** global cloud registration **/
    const float radius = SAMPLING_RADIUS;
    CloudRGB::Ptr global_registered_rgb_cloud(new CloudRGB);
    string init_rgb_cloud_path = poses_files_path_vec[0][0].spot_rgb_cloud_path;
    LoadPcd(init_rgb_cloud_path, *global_registered_rgb_cloud, "fullview rgb");
    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;
    for (int src_idx = 1; src_idx < num_spots; ++src_idx) {
        
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudRGB::Ptr spot_cloud_src(new CloudRGB);

        /** load points **/
        string load_rgb_cloud_path = poses_files_path_vec[src_idx][0].spot_rgb_cloud_path;
        LoadPcd(load_rgb_cloud_path, *spot_cloud_src, "fullview rgb");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
        }
        cout << "Load spot ICP trans mat: \n" << icp_spot_trans_mat << endl;
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);
        *global_registered_rgb_cloud += *spot_cloud_src;
    }

    // /** down sampling **/
    // pcl::UniformSampling<PointRGB> us;
    // us.setRadiusSearch(radius);
    // us.setInputCloud(global_registered_rgb_cloud);
    // us.filter(*global_registered_rgb_cloud);

    string global_registered_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_rgb_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_rgb_cloud);
}

void LidarProcess::GlobalMapping() {
    /** global cloud registration **/
    const float radius = SAMPLING_RADIUS;

    CloudI::Ptr global_registered_cloud(new CloudI);
    string init_dense_cloud_path = poses_files_path_vec[0][0].spot_cloud_path;
    cout << init_dense_cloud_path << endl;
    LoadPcd(init_dense_cloud_path, *global_registered_cloud, "fullview dense");

    for (auto & pt : global_registered_cloud->points) {
        pt.intensity = 40;
    }

    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;

    for (int src_idx = 1; src_idx < num_spots; ++src_idx) {
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudI::Ptr spot_cloud_src(new CloudI);

        /** load points **/
        string load_dense_cloud_path = poses_files_path_vec[src_idx][0].spot_cloud_path;
        LoadPcd(load_dense_cloud_path, *spot_cloud_src, "fullview dense");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            // string trans_file_path = poses_files_path_vec[load_idx][0].lio_spot_trans_mat_path;
            string trans_file_path = poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
        }
        cout << "Load spot ICP trans mat: \n" << icp_spot_trans_mat << endl;
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);

        /** for view coloring & viz only **/
        for (auto & pt : spot_cloud_src->points) {
            pt.intensity = (src_idx + 1) * 40;
            // pt.intensity = 200;
        }
        *global_registered_cloud += *spot_cloud_src;
    }

    // /** down sampling **/
    // pcl::UniformSampling<PointI> us;
    // us.setRadiusSearch(radius);
    // us.setInputCloud(global_registered_cloud);
    // us.filter(*global_registered_cloud);

    string global_registered_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
                                        //   "/global_lio_registered_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_cloud);
}

void LidarProcess::MappingEval(){
    string global_registered_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
    CloudI::Ptr global_registered_cloud(new CloudI);
    LoadPcd(global_registered_cloud_path, *global_registered_cloud, "global registered");

    string global_lio_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                        //   "/roi_scans_xyzi.pcd";
                                        "/global_scans.pcd";
    CloudI::Ptr global_lio_cloud(new CloudI);
    LoadPcd(global_lio_cloud_path, *global_lio_cloud, "global lio");

    /** params **/
    float max_fitness_range = 2.0;

    /** uniform sampling **/
    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);

    /** invalid point filter **/
    pcl::copyPointCloud(*global_lio_cloud, *cloud_us_tgt);
    pcl::copyPointCloud(*global_registered_cloud, *cloud_us_src);
    RemoveInvalidPoints(cloud_us_tgt);
    RemoveInvalidPoints(cloud_us_src);

    Eigen::Matrix<float, 6, 1> transform;
    ros::param::get("transform/rx", transform(0));
    ros::param::get("transform/ry", transform(1));
    ros::param::get("transform/rz", transform(2));
    ros::param::get("transform/tx", transform(3));
    ros::param::get("transform/ty", transform(4));
    ros::param::get("transform/tz", transform(5));
    Mat4F icp_spot_trans_mat = TransformMat(transform);

    cout << icp_spot_trans_mat << endl;

    /** get the init trans cloud & init fitness score **/
    CloudI::Ptr cloud_trans_us (new CloudI);
    pcl::transformPointCloud(*cloud_us_src, *cloud_trans_us, icp_spot_trans_mat);

    pcl::StopWatch timer_fs;
    // cout << "Map Fitness Score: " << GetFitnessScore(cloud_us_tgt, cloud_trans_us, max_fitness_range) << endl;
    // cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;
    Mat4F align_trans_mat = Align(cloud_us_tgt, cloud_us_src, icp_spot_trans_mat, 1, false);

    Vec3F euler_angle_mat = align_trans_mat.topLeftCorner<3, 3>().eulerAngles(2, 1, 0).reverse();
    cout << euler_angle_mat << endl;
    
    string global_tf_cloud_path = poses_files_path_vec[0][0].fullview_recon_folder_path +
                                    "/global_tf_cloud.pcd";
    pcl::io::savePCDFileBinary(global_tf_cloud_path, *cloud_trans_us);

    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    float bkg_grayscale = 0.0;  /** black **/
    float txt_grayscale = 1.0 - bkg_grayscale;

    /** the color of original target cloud is white **/
    pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_aim_color_h(cloud_us_tgt, (int)255 * txt_grayscale,
                                                                                (int)255 * txt_grayscale,
                                                                                (int)255 * txt_grayscale);
    viewer.addPointCloud(cloud_us_tgt, cloud_aim_color_h, "cloud_aim_v1", v1);

    /** the color of original source cloud is green **/
    pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_in_color_h(cloud_trans_us, 20, 180, 20);
    viewer.addPointCloud(cloud_trans_us, cloud_in_color_h, "cloud_in_v1", v1);

    viewer.addCoordinateSystem(5.0);

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

template <typename PointType>
void LidarProcess::LoadPcd(string filepath, pcl::PointCloud<PointType> &cloud, const char* name) {
    if (pcl::io::loadPCDFile<PointType>(filepath, cloud) == -1) {
        PCL_ERROR("Failed to load %s cloud.\n Filepath: %s", name, filepath.c_str());
    }
    else {
        PCL_INFO("Loaded %d points into %s cloud.\n", cloud.points.size(), name);
    }
}

double LidarProcess::GetFitnessScore(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, double max_range) {
    double fitness_score = 0.0;
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    // For each point in the source dataset
    int nr = 0;
    pcl::KdTreeFLANN<PointI> kdtree;
    kdtree.setInputCloud(cloud_tgt);

    #pragma omp parallel for num_threads(16)
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
