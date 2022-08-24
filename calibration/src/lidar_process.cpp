/** headings **/
#include <lidar_process.h>
#include <common_lib.h>

/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

LidarProcess::LidarProcess() {
    /** parameter server **/
    ros::param::get("essential/kLidarTopic", this->topic_name);
    ros::param::get("essential/kDatasetName", this->dataset_name);
    ros::param::get("essential/kNumSpots", this->num_spots);
    ros::param::get("essential/kNumViews", this->num_views);
    ros::param::get("essential/kAngleInit", this->view_angle_init);
    ros::param::get("essential/kAngleStep", this->view_angle_step);
    this->kDatasetPath = this->kPkgPath + "/data/" + this->dataset_name;
    this->fullview_idx = (this->num_views - 1) / 2;

    cout << "----- LiDAR: LidarProcess -----" << endl;
    /** create objects, initialization **/
    string pose_folder_path_temp;
    PoseFilePath pose_files_path_temp;
    EdgePixels edge_pixels_temp;
    EdgeCloud::Ptr edge_cloud_temp;
    TagsMap tags_map_temp;
    Mat4F pose_trans_mat_temp;
    for (int i = 0; i < this->num_spots; ++i) {
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
        this->poses_folder_path_vec.push_back(poses_folder_path_vec_temp);
        this->poses_files_path_vec.push_back(poses_file_path_vec_temp);
        this->edge_pixels_vec.push_back(edge_pixels_vec_temp);
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

/** Point Cloud Loading **/
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

    uint32_t idx_start = (cnt_pcds - this->kNumRecPcds)/2;
    uint32_t idx_end = idx_start + this->kNumRecPcds;
    iterator = view.begin();

    for (int i = 0; iterator != view.end(); iterator++, i++) {
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
    cout << "----- LiDAR: LidarToSphere -----" << " Spot Index: " << this->spot_idx << endl;
    /** define the initial projection mode - by intensity or by depth **/
    // const bool projByIntensity = this->kProjByIntensity;
    float theta_min = M_PI, theta_max = -M_PI;
    float proj_param;

    string fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_dense_cloud_path;
    /** original cartesian point cloud **/
    pcl::io::loadPCDFile(fullview_cloud_path, *cart_cloud);

    /** Initial Transformation **/
    Ext_D extrinsic_vec;
    extrinsic_vec << ext_.head(3), 0, 0, 0;
    Mat4D T_mat = TransformMat(extrinsic_vec);
    pcl::transformPointCloud(*cart_cloud, *polar_cloud, T_mat);

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
        point.z = radius;
        if (theta > theta_max) { theta_max = theta; }
        else if (theta < theta_min) { theta_min = theta; }
    }

    cout << "LiDAR polar cloud generated. \ntheta_min = " << theta_min << " theta_max = " << theta_max << endl;

    if (kEdgeAnalysis) {
        /** visualization for weight check**/
        pcl::io::savePCDFileBinary(this->poses_files_path_vec[this->spot_idx][this->view_idx].output_folder_path + "/fullview_polar_cloud.pcd", *polar_cloud);
    }

}

void LidarProcess::SphereToPlane(CloudI::Ptr& polar_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << " Spot Index: " << this->spot_idx << endl;
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
    int hidden_pt_cnt = 0;
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
                vector<int> zero_vec(search_num, 0);
                tags_map[u][v].pts_indices.insert(tags_map[u][v].pts_indices.begin(), zero_vec.data(), zero_vec.data()+search_num);;

                for (int i = 0; i < search_num; ++i) {
                    dist_mean += polar_cloud->points[search_pt_idx_vec[i]].z;
                }
                dist_mean = dist_mean / search_num;

                for (int i = 0; i < search_num; ++i) {
                    PointI &local_pt = polar_cloud->points[search_pt_idx_vec[i]];
                    float dist = local_pt.z;
                    if ((abs(dist_mean - dist) > dist * sensitivity) || ((dist_mean - dist) > dist * sensitivity && local_pt.intensity < 20)) {
                        hidden_pt_num++;
                        tags_map[u][v].pts_indices[i] = 0;
                    }
                    else {
                        intensity_mean += local_pt.intensity;
                        tags_map[u][v].pts_indices[i] = search_pt_idx_vec[i];
                    }
                }

                /** add tags **/
                tags_map[u][v].num_pts = search_num - hidden_pt_num;
                tags_map[u][v].pts_indices.erase(std::remove(tags_map[u][v].pts_indices.begin(), tags_map[u][v].pts_indices.end(), 0), tags_map[u][v].pts_indices.end());
                
                if (tags_map[u][v].num_pts > 0) {
                    flat_img.at<float>(u, v) = intensity_mean / tags_map[u][v].num_pts;
                }
                else {
                    flat_img.at<float>(u, v) = 0;
                }
                
                // hidden_pt_cnt += hidden_pt_num;
                ROS_ASSERT_MSG((tags_map[u][v].pts_indices.size() == tags_map[u][v].num_pts), "size of the vectors in a pixel region is not the same!");
            }
        }
    }

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
            outfile << "size: " << tags_map[u][v].num_pts << endl;
        }
    }
    outfile.close();

    string flat_img_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].flat_img_path;
    cout << "LiDAR flat image path: " << flat_img_path << endl;
    cv::imwrite(flat_img_path, flat_img);

}

void LidarProcess::EdgeExtraction() {
    std::string script_path = this->kPkgPath + "/python_scripts/image_process/edge_extraction.py";
    std::string kSpots = to_string(this->spot_idx);
    std::string cmd_str = "python3 " + script_path + " " + this->kDatasetPath + " " + "lidar" + " " + kSpots;
    system(cmd_str.c_str());
}

void LidarProcess::EdgeToPixel() {
    /** generate edge_pixels and push back into edge_pixels_vec **/
    cout << "----- LiDAR: EdgeToPixel -----" << " Spot Index: " << this->spot_idx << endl;
    string edge_img_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG((edge_img.rows != 0 && edge_img.cols != 0), "size of original fisheye image is 0, check the path and filename! \nView Index: %d \nPath: %s", this->view_idx, edge_img_path.c_str());
    ROS_ASSERT_MSG((edge_img.rows == this->kFlatRows || edge_img.cols == this->kFlatCols), "size of original fisheye image is incorrect! View Index: %d", this->view_idx);

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
    this->edge_pixels_vec[this->spot_idx][this->view_idx] = edge_pixels;
}

void LidarProcess::PixLookUp(CloudI::Ptr& cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << " Spot Index: " << this->spot_idx << endl;
    int num_invalid_pixels = 0;
    
    TagsMap tags_map = this->tags_map_vec[this->spot_idx][this->view_idx];
    EdgePixels edge_pixels = this->edge_pixels_vec[this->spot_idx][this->view_idx];
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
                PointI &pixel_pt = cart_cloud->points[tags_map[u][v].pts_indices[i]];
                x_avg += pixel_pt.x;
                y_avg += pixel_pt.y;
                z_avg += pixel_pt.z;
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
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;

    /** write the coordinates and weights into .txt file **/
    string edge_pts_coordinates_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
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

    // if (kEdgeAnalysis) {
    //     /** visualization for weight check**/
    //     string edge_cart_pcd_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_cart_pcd_path;
    //     cout << edge_cart_pcd_path << endl;
    //     pcl::io::savePCDFileBinary(edge_cart_pcd_path, *edge_cloud);

    //     CloudI::Ptr polar_rgb_cloud(new CloudI);
    //     Eigen::Matrix<float, 6, 1> extrinsic_vec;
    //     extrinsic_vec << (float)this->extrinsic.rx, (float)this->extrinsic.ry, (float)this->extrinsic.rz,
    //                             0.0f, 0.0f, 0.0f;
    //     Mat4F T_mat = TransformMat(extrinsic_vec);
    //     pcl::transformPointCloud(*edge_cloud, *polar_rgb_cloud, T_mat);

    //     float radius, phi, theta;
    //     for (auto &point : polar_rgb_cloud->points) {
    //         radius = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
    //         phi = atan2(point.y, point.x);
    //         theta = acos(point.z / radius);
    //         point.x = theta;
    //         point.y = phi;
    //         point.z = 0;
    //         point.intensity = 200;
    //     }

    //     string edge_polar_pcd_path = this -> poses_files_path_vec[this->spot_idx][this->view_idx].edge_polar_pcd_path;
    //     cout << edge_polar_pcd_path << endl;
    //     pcl::io::savePCDFileBinary(edge_polar_pcd_path, *polar_rgb_cloud);
    // }

}

void LidarProcess::ReadEdge() {
    cout << "----- LiDAR: ReadEdge -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    string edge_cloud_txt_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].edge_pts_coordinates_path;
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

    ROS_ASSERT_MSG(!edge_pts.empty(), "LiDAR Read Edge Incorrect! View Index: %d", this->view_idx);
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
    cout << "Filtered LiDAR points: " << edge_cloud -> points.size() << endl;
    this->edge_cloud_vec[this->spot_idx][this->view_idx] = edge_cloud;
}

/** Point Cloud Registration **/
tuple<Mat4F, CloudI::Ptr> LidarProcess::ICPRegistration(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, Mat4F init_trans_mat, int cloud_type, const bool kIcpViz) {
    /** params **/
    float uniform_radius = 0.01;
    int max_iters = 200;
    float max_corr_dis = 0.5;
    float trans_epsilon = 1e-10;
    float eucidean_epsilon = 0.01;
    float max_fitness_range = 2.0;

    /** uniform sampling **/
    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);
    pcl::UniformSampling<PointI> us;
    us.setRadiusSearch(uniform_radius);
    us.setInputCloud(cloud_tgt);
    us.filter(*cloud_us_tgt);
    us.setInputCloud(cloud_src);
    us.filter(*cloud_us_src);
    PCL_INFO("Size of Uniform Sampling Filtered Target Cloud: %d\n", cloud_us_tgt->size());
    PCL_INFO("Size of Uniform Sampling Filtered Source Cloud: %d\n", cloud_us_src->size());

    /** invalid point filter **/
    std::vector<int> null_indices_tgt;
    (*cloud_us_tgt).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_tgt, *cloud_us_tgt, null_indices_tgt);
    std::vector<int> null_indices_src;
    (*cloud_us_src).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_src, *cloud_us_src, null_indices_src);

    CloudI::Ptr cloud_us_tgt_effe (new CloudI);
    CloudI::Ptr cloud_us_src_effe (new CloudI);
    if (cloud_type == 0) { /** view point cloud **/
        cout << "box effective filter" << endl;
        /** box down sampling **/
        /** keep the points that satisfied the condition **/
        pcl::ConditionAnd<PointI>::Ptr range_cond(new pcl::ConditionAnd<PointI>());
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("z", pcl::ComparisonOps::GT, -1.0)));
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("z", pcl::ComparisonOps::LT, 15.0)));
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("y", pcl::ComparisonOps::GT, -8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("y", pcl::ComparisonOps::LT, 8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("x", pcl::ComparisonOps::GT, -8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("x", pcl::ComparisonOps::LT, 8.0)));
        pcl::ConditionalRemoval<PointI> cond_filter;
        cond_filter.setKeepOrganized(false); /** default replaced value NaN **/
        cond_filter.setCondition(range_cond);
        cond_filter.setInputCloud(cloud_us_src);
        cond_filter.filter(*cloud_us_src_effe);
        cond_filter.setKeepOrganized(false);
        cond_filter.setInputCloud(cloud_us_tgt);
        cond_filter.filter(*cloud_us_tgt_effe);
        cout << "Size of target cloud after effective point filter: " << cloud_us_tgt_effe->points.size() << endl;
        cout << "Size of source cloud after effective point filter: " << cloud_us_src_effe->points.size() << endl;
    }
    else if (cloud_type == 1) { /** spot point cloud **/
        cout << "k search effective filter" << endl;
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
            if (nn_dists[0] <= max_fitness_range && i < cloud_us_src->size()) {
                src_effe_indices[i] = i;
            }
            else {
                src_effe_indices[i] = 0;
            }
        }
        src_effe_indices.erase(std::remove(src_effe_indices.begin(), src_effe_indices.end(), 0), src_effe_indices.end());
        pcl::copyPointCloud(*cloud_us_src, src_effe_indices, *cloud_us_src_effe);
        cout << "Size of source cloud after effective point filter: " << cloud_us_src_effe->size() << endl;

        pcl::KdTreeFLANN<PointI> kdtree_src;
        kdtree_src.setInputCloud (cloud_us_src);
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < cloud_us_tgt->size(); ++i) {
            kdtree_src.nearestKSearch (cloud_us_tgt->points[i], 1, nn_indices, nn_dists);
            if (nn_dists[0] <= max_fitness_range && i < cloud_us_tgt->size()) {
                tgt_effe_indices[i] = i;
            }
            else {
                tgt_effe_indices[i] = 0;
            }
        }
        tgt_effe_indices.erase(std::remove(tgt_effe_indices.begin(), tgt_effe_indices.end(), 0), tgt_effe_indices.end());
        pcl::copyPointCloud(*cloud_us_tgt, tgt_effe_indices, *cloud_us_tgt_effe);
        cout << "Size of target cloud after effective point filter: " << cloud_us_tgt_effe->size() << endl;
        cout << "Run time of effective point filter: " << timer_effe.getTimeSeconds() << " s" << endl;
    }

    /** invalid point filter **/
    (*cloud_us_tgt_effe).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_tgt_effe, *cloud_us_tgt_effe, null_indices_tgt);
    (*cloud_us_src_effe).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_src_effe, *cloud_us_src_effe, null_indices_src);

    /** get the init trans cloud & init fitness score **/
    CloudI::Ptr cloud_init_trans_us (new CloudI);
    pcl::transformPointCloud(*cloud_us_src_effe, *cloud_init_trans_us, init_trans_mat);
    cout << "\nInit Trans Mat: \n " << init_trans_mat << endl;
    pcl::StopWatch timer_fs;
    cout << "Initial Fitness Score: " << GetIcpFitnessScore(cloud_us_tgt_effe, cloud_init_trans_us, max_fitness_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;

    /** ICP **/
    pcl::StopWatch timer;
    timer.reset(); /** timing **/
    CloudI::Ptr cloud_icp_trans_us (new CloudI);
    Mat4F icp_trans_mat;
    pcl::IterativeClosestPoint <PointI, PointI> icp; /** original icp **/

    icp.setInputTarget(cloud_us_tgt_effe);
    icp.setInputSource(cloud_us_src_effe);
    icp.setMaximumIterations(max_iters);
    icp.setMaxCorrespondenceDistance(max_corr_dis);
    icp.setTransformationEpsilon(trans_epsilon);
    icp.setEuclideanFitnessEpsilon(eucidean_epsilon);
    icp.align(*cloud_icp_trans_us, init_trans_mat);

    /** visualization **/
    if (kIcpViz) {
        pcl::visualization::PCLVisualizer viewer("ICP demo");
        int v1(0), v2(1); /** create two view point **/
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        float bckgr_gray_level = 0.0;  /** black **/
        float txt_gray_lvl = 1.0 - bckgr_gray_level;

        /** the color of original target cloud is white **/
        pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_aim_color_h(cloud_us_tgt_effe, (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl);
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
    tuple<Mat4F, CloudI::Ptr> result;
    result = make_tuple(icp_trans_mat, cloud_icp_trans_us);
    return result;
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
    std::vector<int> null_indices_tgt;
    (*cloud_us_tgt).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_tgt, *cloud_us_tgt, null_indices_tgt);
    std::vector<int> null_indices_src;
    (*cloud_us_src).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_src, *cloud_us_src, null_indices_src);

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
    cout << "Coarse to fine fitness score: " << GetIcpFitnessScore(cloud_us_tgt_effe, cloud_us_src_effe, max_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;
}

double LidarProcess::GetIcpFitnessScore(CloudI::Ptr cloud_tgt, CloudI::Ptr cloud_src, double max_range) {
    double fitness_score = 0.0;
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);
    // For each point in the source dataset
    int nr = 0;
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud_tgt);

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

    if (nr > 0)
        return (fitness_score / nr);
    return (std::numeric_limits<double>::max());
}

void LidarProcess::CreateDensePcd() {
    cout << "----- LiDAR: CreateDensePcd -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;

    num_pcds = LidarProcess::kNumRecPcds;
    pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    folder_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcds_folder_path;

    pcl::PCDReader reader; /** used for read PCD files **/
    vector<string> file_name_vec;
    ReadFileList(folder_path, file_name_vec);
    sort(file_name_vec.begin(), file_name_vec.end()); /** sort file names by order **/
    const int kPcdsGroupSize = file_name_vec.size() / num_pcds; // always equal to 1?

    /** PCL PointCloud pointer. Remember that the pointer need to be given a new space **/
    CloudI::Ptr load_pcd_cloud(new CloudI);
    CloudI::Ptr view_raw_cloud(new CloudI);

    BagToPcd(bag_path, *view_raw_cloud);
    cout << "size of loaded point cloud: " << view_raw_cloud->points.size() << endl;

    /** condition filter **/
    CloudI::Ptr view_cloud(new CloudI);
    pcl::ConditionOr<PointI>::Ptr range_cond(new pcl::ConditionOr<PointI>());
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("z", pcl::ComparisonOps::GT, 0.3))); /** GT: greater than **/
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("z", pcl::ComparisonOps::LT, -0.4))); /** LT: less than **/
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointI>::ConstPtr(new pcl::FieldComparison<PointI> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<PointI> cond_filter;
    cond_filter.setKeepOrganized(false);
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(view_raw_cloud);
    cond_filter.filter(*view_cloud);

    /** invalid point filter **/
    (*view_cloud).is_dense = false;
    std::vector<int> null_indices;
    pcl::removeNaNFromPointCloud(*view_cloud, *view_cloud, null_indices);

    /** check the pass through filtered point cloud size **/
    cout << "size of cloud after a condition filter:" << view_cloud->points.size() << endl;

    pcl::io::savePCDFileBinary(pcd_path, *view_cloud);
    cout << "Create Dense Point Cloud File Successfully!" << endl;
}

void LidarProcess::ViewRegistration() {
    cout << "----- LiDAR: ViewRegistration -----" << " Spot Index: " << this->spot_idx << " View Index: " << this->view_idx << endl;
    /** load point clouds to be registered **/
    std::string tgt_pcd_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;
    std::string src_pcd_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].dense_pcd_path;
    CloudI::Ptr view_cloud_tgt(new CloudI);
    CloudI::Ptr view_cloud_src(new CloudI);
    if (pcl::io::loadPCDFile<PointI>(tgt_pcd_path, *view_cloud_tgt) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "Loaded " << view_cloud_tgt->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointI>(src_pcd_path, *view_cloud_src) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "Loaded " << view_cloud_src->size() << " points from source file" << endl;

    /** initial rigid transformation **/
    float v_angle = (float)DEG2RAD(degree_map[view_idx]);
    float radius = 0.15f;
    Eigen::Matrix<float, 6, 1> trans_params;
    trans_params << 0.0f, v_angle, 0.0f,
                    radius * (sin(v_angle) - 0.0f), 0.0f, radius * (cos(v_angle) - 1.0f); /** LiDAR x-axis: car front; Gimbal positive angle: car front **/
    Mat4F init_trans_mat = TransformMat(trans_params);

    // /** ICP **/
    // std::tuple<Mat4F, CloudI::Ptr> icp_result = ICP(view_cloud_tgt, view_cloud_src, init_trans_mat, 0, false);
    // Mat4F icp_trans_mat;
    // CloudI::Ptr view_cloud_icp_trans;
    // std::tie(icp_trans_mat, view_cloud_icp_trans) = icp_result;

    /** TEASER registration **/
    Mat4F reg_trans_mat;
    CloudI::Ptr view_cloud_icp_trans(new CloudI);
    CloudReg(view_cloud_tgt, view_cloud_src, init_trans_mat, reg_trans_mat);
    pcl::transformPointCloud(*view_cloud_src, *view_cloud_icp_trans, reg_trans_mat);

    /** save the view trans matrix by icp **/
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[this->spot_idx][this->view_idx].pose_trans_mat_path);
    mat_out << reg_trans_mat << endl;
    mat_out.close();

    /** save the registered point clouds **/
    string registered_cloud_path = this->poses_files_path_vec[this->spot_idx][this->view_idx].fullview_recon_folder_path +
                                   "/icp_registered_" + to_string(v_angle) + ".pcd";
    pcl::io::savePCDFileBinary(registered_cloud_path, *view_cloud_icp_trans + *view_cloud_tgt);
}

void LidarProcess::FullViewMapping() {
    cout << "----- LiDAR: CreateFullviewPcd -----" << " Spot Index: " << this->spot_idx << endl;
    /** target and fullview cloud path **/
    string tgt_view_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].dense_pcd_path;
    string fullview_cloud_path = this->poses_files_path_vec[this->spot_idx][this->fullview_idx].fullview_dense_cloud_path;

    /** load full view point cloud **/
    CloudI::Ptr fullview_raw_cloud(new CloudI);
    LoadPcd(tgt_view_cloud_path, *fullview_raw_cloud, "view");
    cout << "Degree 0 Full View Dense Pcd Loaded!" << endl;

    for(int i = 0; i < this->num_views; i++) {
        if (i == this->fullview_idx) {
            continue;
        }
        /** load icp pose transform matrix **/
        string pose_trans_mat_path = this->poses_files_path_vec[this->spot_idx][i].pose_trans_mat_path;
        Mat4F pose_trans_mat = LoadTransMat(pose_trans_mat_path);
        cout << "Degree " << this->degree_map[i] << " ICP Mat: " << "\n" << pose_trans_mat << endl;

        /** transform point cloud **/
        CloudI::Ptr view_cloud(new CloudI);
        string view_cloud_path = this->poses_files_path_vec[this->spot_idx][i].dense_pcd_path;
        LoadPcd(view_cloud_path, *view_cloud, "view");

        pcl::transformPointCloud(*view_cloud, *view_cloud, pose_trans_mat);

        /** point cloud addition **/
        *fullview_raw_cloud = *fullview_raw_cloud + *view_cloud;
    }

    /** check the original point cloud size **/
    int fullview_cloud_size = fullview_raw_cloud->points.size();
    cout << "size of original cloud:" << fullview_cloud_size << endl;

    /** radius outlier filter **/
    CloudI::Ptr radius_outlier_cloud(new CloudI);
    pcl::RadiusOutlierRemoval<PointI> radius_outlier_filter;
    radius_outlier_filter.setInputCloud(fullview_raw_cloud);
    radius_outlier_filter.setRadiusSearch(0.1);
    radius_outlier_filter.setMinNeighborsInRadius(200);
    radius_outlier_filter.setNegative(false);
    radius_outlier_filter.setKeepOrganized(false);
    radius_outlier_filter.filter(*radius_outlier_cloud);

    /** radius outlier filter cloud size check **/
    int radius_outlier_cloud_size = radius_outlier_cloud->points.size();
    cout << "radius outlier filtered cloud size:" << radius_outlier_cloud_size << endl;

    pcl::io::savePCDFileBinary(fullview_cloud_path, *radius_outlier_cloud);
    cout << "Create Full View Point Cloud File Successfully!" << endl;
}

void LidarProcess::SpotRegistration() {
    
    /** source index and target index **/
    int src_idx = this->spot_idx;
    int tgt_idx = this->spot_idx - 1;
    PCL_INFO("ICP Target Spot Index: %d\n", tgt_idx);
    PCL_INFO("ICP Source Spot Index: %d\n", src_idx);

    /** load points **/
    CloudI::Ptr spot_cloud_tgt(new CloudI);
    CloudI::Ptr spot_cloud_src(new CloudI);
    string spot_cloud_tgt_path = this->poses_files_path_vec[tgt_idx][0].fullview_dense_cloud_path;
    string spot_cloud_src_path = this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path;
    LoadPcd(spot_cloud_tgt_path, *spot_cloud_tgt, "target spot");
    LoadPcd(spot_cloud_src_path, *spot_cloud_src, "source spot");

    /** initial transformation and initial score **/
    // vector<Mat4F> icp_trans_mat_vec;
    string lio_trans_path = this->poses_files_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Mat4F lio_spot_trans_mat = LoadTransMat(lio_trans_path);
    Mat3F lio_spot_rotation_mat = lio_spot_trans_mat.topLeftCorner<3, 3>();
    Vec3F lio_euler_angle = lio_spot_rotation_mat.eulerAngles(2, 1, 0); // zyx euler angle
    cout << "Euler angle by LIO: \n" << lio_euler_angle << endl;
    cout << "Initial Trans Mat by LIO: \n" << lio_spot_trans_mat << endl;

    /** ICP **/
    std::tuple<Mat4F, CloudI::Ptr> icp_result = ICPRegistration(spot_cloud_tgt, spot_cloud_src, lio_spot_trans_mat, 1, false);
    Mat4F icp_spot_trans_mat;
    CloudI::Ptr spot_cloud_icp_trans;
    std::tie(icp_spot_trans_mat, spot_cloud_icp_trans) = icp_result;
    // icp_trans_mat_vec.push_back(icp_spot_trans_mat);

    /** compared the fitness score of lio and icp **/
//    CloudI::Ptr spot_lio_trans (new CloudI);
//    CloudI::Ptr spot_icp_trans (new CloudI);
//    pcl::transformPointCloud(*spot_cloud_src, *spot_lio_trans, lio_spot_trans_mat);
//    pcl::transformPointCloud(*spot_cloud_src, *spot_icp_trans, icp_spot_trans_mat);
//    cout << "Spot Registration Fast-LIO Fitness Score: " << GetIcpFitnessScore(spot_cloud_tgt, spot_lio_trans, 2.0) << endl;
//    cout << "Spot Registration ICP Fitness Score: " << GetIcpFitnessScore(spot_cloud_tgt, spot_icp_trans, 2.0) << endl;

    /** save the spot trans matrix by icp **/
    cout << this->poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path << endl;
    std::ofstream mat_out;
    mat_out.open(this->poses_files_path_vec[src_idx][0].icp_spot_trans_mat_path);
    mat_out << icp_spot_trans_mat << endl;
    mat_out.close();

    /** save the pair registered point cloud **/
    string pair_registered_cloud_path = this->poses_files_path_vec[tgt_idx][0].fullview_recon_folder_path +
                                        "/icp_registered_spot_tgt_" + to_string(tgt_idx) + ".pcd";
    cout << pair_registered_cloud_path << endl;
    pcl::io::savePCDFileBinary(pair_registered_cloud_path, *spot_cloud_icp_trans + *spot_cloud_tgt);
}

void LidarProcess::FineToCoarseReg() {

    cout << "----- LiDAR: FineToCoarseReg -----" << " Spot Index: " << this->spot_idx << endl;
    /** load points **/
    string lio_spot_trans_path = this->poses_files_path_vec[this->spot_idx][0].lio_spot_trans_mat_path;
    string lio_static_trans_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                    "/lio_static_trans_mat.txt";
    string spot_cloud_path = this->poses_files_path_vec[this->spot_idx][0].fullview_dense_cloud_path;
    string global_coarse_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                    "/scans.pcd";
    
    CloudI::Ptr spot_cloud(new CloudI);
    CloudI::Ptr global_coarse_cloud(new CloudI);
    Mat4F lio_spot_trans_mat = Mat4F::Identity();
    Mat4F lio_static_trans_mat = Mat4F::Identity();

    LoadPcd(spot_cloud_path, *spot_cloud, "spot");
    LoadPcd(global_coarse_cloud_path, *global_coarse_cloud, "global coarse");

    for (int load_idx = this->spot_idx; load_idx > 0; --load_idx) {
        string trans_file_path = this->poses_files_path_vec[load_idx][0].lio_spot_trans_mat_path;
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

    // std::tuple<Mat4F, CloudI::Ptr> icp_result = ICP(global_coarse_cloud, spot_cloud, lio_spot_trans_mat, 1, triangulatePoints);
    // Mat4F icp_spot_trans_mat;
    // CloudI::Ptr spot_cloud_icp_trans;
    // std::tie(icp_spot_trans_mat, spot_cloud_icp_trans) = icp_result;
}

void LidarProcess::GlobalColoredMapping() {
    /** global cloud registration **/
    const float radius = 0.01f;
    CloudRGB::Ptr global_registered_rgb_cloud(new CloudRGB);
    string init_rgb_cloud_path = this->poses_files_path_vec[0][0].fullview_rgb_cloud_path;
    LoadPcd(init_rgb_cloud_path, *global_registered_rgb_cloud, "fullview rgb");
    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;
    for (int src_idx = 1; src_idx < this->num_spots; ++src_idx) {
        
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudRGB::Ptr spot_cloud_src(new CloudRGB);

        /** load points **/
        string load_rgb_cloud_path = this->poses_files_path_vec[src_idx][0].fullview_rgb_cloud_path;
        LoadPcd(load_rgb_cloud_path, *spot_cloud_src, "fullview rgb");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            string trans_file_path = this->poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
            cout << "Load spot ICP trans mat: \n" << tmp_spot_trans_mat << endl;
        }
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);
        *global_registered_rgb_cloud += *spot_cloud_src;
    }

    /** down sampling **/
    pcl::UniformSampling<PointRGB> us;
    us.setRadiusSearch(radius);
    us.setInputCloud(global_registered_rgb_cloud);
    us.filter(*global_registered_rgb_cloud);

    string global_registered_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_rgb_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_rgb_cloud);
}

void LidarProcess::GlobalMapping() {
    /** global cloud registration **/
    const float radius = 0.01f;

    CloudI::Ptr global_registered_cloud(new CloudI);
    string init_dense_cloud_path = this->poses_files_path_vec[0][0].fullview_dense_cloud_path;
    cout << init_dense_cloud_path << endl;
    LoadPcd(init_dense_cloud_path, *global_registered_cloud, "fullview dense");

    /** source index and target index (align to spot 0) **/
    int tgt_idx = 0;

    for (int src_idx = 1; src_idx < this->num_spots; ++src_idx) {
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudI::Ptr spot_cloud_src(new CloudI);

        /** load points **/
        string load_dense_cloud_path = this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path;
        LoadPcd(load_dense_cloud_path, *spot_cloud_src, "fullview dense");

        /** load transformation matrix **/
        Mat4F icp_spot_trans_mat = Mat4F::Identity();
        for (int load_idx = src_idx; load_idx > 0; --load_idx) {
            // string trans_file_path = this->poses_files_path_vec[load_idx][0].lio_spot_trans_mat_path;
            string trans_file_path = this->poses_files_path_vec[load_idx][0].icp_spot_trans_mat_path;
            Mat4F tmp_spot_trans_mat = LoadTransMat(trans_file_path);
            icp_spot_trans_mat = tmp_spot_trans_mat * icp_spot_trans_mat;
            cout << "Load spot ICP trans mat: \n" << tmp_spot_trans_mat << endl;
        }
        pcl::transformPointCloud(*spot_cloud_src, *spot_cloud_src, icp_spot_trans_mat);
        
        
        /** for view coloring & viz only **/
        for (auto & pt : spot_cloud_src->points) {
            pt.intensity = src_idx * 40;
        }
        *global_registered_cloud += *spot_cloud_src;

    }

    /** down sampling **/
    pcl::UniformSampling<PointI> us;
    us.setRadiusSearch(radius);
    us.setInputCloud(global_registered_cloud);
    us.filter(*global_registered_cloud);

    string global_registered_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *global_registered_cloud);
}

void LidarProcess::MappingEval(){
    string global_registered_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
    CloudI::Ptr global_registered_cloud(new CloudI);
    LoadPcd(global_registered_cloud_path, *global_registered_cloud, "global registered");

    string global_lio_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/scans.pcd";
    CloudI::Ptr global_lio_cloud(new CloudI);
    LoadPcd(global_lio_cloud_path, *global_lio_cloud, "global lio");

    /** params **/
    int max_iters = 200;
    float max_corr_dis = 0.2;
    float trans_epsilon = 1e-10;
    float eucidean_epsilon = 0.01;
    float max_fitness_range = 2.0;

    /** uniform sampling **/
    CloudI::Ptr cloud_us_tgt (new CloudI);
    CloudI::Ptr cloud_us_src (new CloudI);

    /** invalid point filter **/
    std::vector<int> null_indices_tgt;
    (*cloud_us_tgt).is_dense = false;
    pcl::removeNaNFromPointCloud(*global_lio_cloud, *cloud_us_tgt, null_indices_tgt);
    std::vector<int> null_indices_src;
    (*cloud_us_src).is_dense = false;
    pcl::removeNaNFromPointCloud(*global_registered_cloud, *cloud_us_src, null_indices_src);

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
    cout << "Map Fitness Score: " << GetIcpFitnessScore(cloud_us_tgt, cloud_trans_us, max_fitness_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;

    string global_tf_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_tf_cloud.pcd";
    pcl::io::savePCDFileBinary(global_tf_cloud_path, *cloud_trans_us);

    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    float bckgr_gray_level = 0.0;  /** black **/
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    /** the color of original target cloud is white **/
    pcl::visualization::PointCloudColorHandlerCustom <PointI> cloud_aim_color_h(cloud_us_tgt, (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl);
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