/** headings **/
#include "lidar_process.h"
#include "utils.h"
/** namespace **/
using namespace std;
using namespace cv;
using namespace Eigen;

LidarProcess::LidarProcess() {
    /** parameter server **/
    ros::param::get("essential/kLidarTopic", this->dataset_name);
    ros::param::get("essential/kDatasetName", this->dataset_name);
    this->kDatasetPath = this->kPkgPath + "/data/" + this->dataset_name;
    ros::param::get("essential/kNumSpots", this->num_spots);
    ros::param::get("essential/kNumViews", this->num_views);
    ros::param::get("essential/kAngleInit", this->view_angle_init);
    ros::param::get("essential/kAngleStep", this->view_angle_step);
    this->fullview_idx = (this->num_views - 1) / 2;

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

/** Data Pre-processing **/
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

void LidarProcess::SphereToPlane(const CloudPtr& cart_cloud, const CloudPtr& polar_cloud) {
    cout << "----- LiDAR: SphereToPlane -----" << " Spot Index: " << this->spot_idx << endl;
    /** define the data container **/
    cv::Mat flat_img = cv::Mat::zeros(kFlatRows, kFlatCols, CV_32FC1); /** define the flat image **/
    vector<vector<Tags>> tags_map (kFlatRows, vector<Tags>(kFlatCols));

    /** construct kdtrees and load the point clouds **/
    /** caution: the point cloud need to be set before the loop **/
    pcl::KdTreeFLANN<PointT> kdtree;
    CloudPtr polar_flat_cloud(new CloudT);
    pcl::copyPointCloud(*polar_cloud, *polar_flat_cloud);
    for (auto &pt : polar_flat_cloud->points) {pt.z = 0;}
    kdtree.setInputCloud(polar_flat_cloud);

    /** define the invalid search parameters **/
    int invalid_search_num = 0; /** search invalid count **/
    int invalid_idx_num = 0; /** index invalid count **/
    const float kScale = sqrt(2);
    const float kSearchRadius = kScale * (kRadPerPix / 2);

    int hidden_pt_cnt = 0;
    const float sensitivity = 0.02f;

    /** Multiprocessing test **/
    #pragma omp parallel for num_threads(16)

    for (int u = 0; u < kFlatRows; ++u) {

        float theta_center = - kRadPerPix * (2 * u + 1) / 2 + M_PI;

        for (int v = 0; v < kFlatCols; ++v) {

            float phi_center = kRadPerPix * (2 * v + 1) / 2 - M_PI;

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
                invalid_search_num += 1;
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
                    PointT &local_pt = polar_cloud->points[search_pt_idx_vec[i]];
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
                
                hidden_pt_cnt += hidden_pt_num;
                ROS_ASSERT_MSG((tags_map[u][v].pts_indices.size() == tags_map[u][v].num_pts), "size of the vectors in a pixel region is not the same!");
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

void LidarProcess::PixLookUp(CloudPtr cart_cloud) {
    /** generate edge_pts and edge_cloud, push back into vec **/
    cout << "----- LiDAR: PixLookUp -----" << " Spot Index: " << this->spot_idx << endl;
    int num_invalid_pixels = 0;
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
        if (num_pts == 0) { /** invalid pixels **/
            num_invalid_pixels += 1;
        }
        else { /** normal pixels **/
            /** center of lidar edge distribution **/
            float x_avg = 0.0f, y_avg = 0.0f, z_avg = 0.0f;
            for (int i = 0; i < tags_map[u][v].pts_indices.size(); ++i) {
                PointT pixel_pt = cart_cloud->points[tags_map[u][v].pts_indices[i]];
                x_avg += pixel_pt.x;
                y_avg += pixel_pt.y;
                z_avg += pixel_pt.z;
            }
            /** average coordinates->unbiased estimation of center position **/
            x_avg = x_avg / num_pts;
            y_avg = y_avg / num_pts;
            z_avg = z_avg / num_pts;
            
            /** store the spatial coordinates into vector **/
            vector<double> coordinates {x_avg, y_avg, z_avg};
            edge_pts.push_back(coordinates);

            /** store the spatial coordinates into vector **/
            PointT pt;
            pt.x = x_avg;
            pt.y = y_avg;
            pt.z = z_avg;
            pt.intensity = 1; /** note: I is used to store the point weight **/
            edge_cloud->points.push_back(pt);

            if (kEdgeAnalysis) {
                /** visualization for weight check**/
                pt.intensity = 255;
                weight_rgb_cloud->points.push_back(pt);
            }
        }
    }
    cout << "number of invalid lookups(lidar): " << num_invalid_pixels << endl;
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
        outfile << point.x << "\t"
                << point.y << "\t"
                << point.z << "\t"
                << point.intensity << endl;
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

/** Point Cloud Registration **/
tuple<Eigen::Matrix4f, CloudPtr> LidarProcess::ICP(CloudPtr cloud_tgt, CloudPtr cloud_src, Eigen::Matrix4f init_trans_mat, int cloud_type, const bool kIcpViz) {
    /** params **/
    float uniform_radius = 0.02;
    int max_iters = 100;
    float max_corr_dis = 0.2;
    float trans_epsilon = 1e-10;
    float eucidean_epsilon = 0.01;
    float max_fitness_range = 2.0;

    /** uniform sampling **/
    CloudPtr cloud_us_tgt (new CloudT);
    CloudPtr cloud_us_src (new CloudT);
    pcl::UniformSampling<PointT> us;
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

    CloudPtr cloud_us_tgt_effe (new CloudT);
    CloudPtr cloud_us_src_effe (new CloudT);
    if (cloud_type == 0) { /** view point cloud **/
        cout << "box effective filter" << endl;
        /** box down sampling **/
        /** keep the points that satisfied the condition **/
        pcl::ConditionAnd<PointT>::Ptr range_cond(new pcl::ConditionAnd<PointT>());
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, -1.0)));
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, 15.0)));
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, -8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, 8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, -8.0)));
        range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, 8.0)));
        pcl::ConditionalRemoval<PointT> cond_filter;
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

        pcl::KdTreeFLANN<PointT> kdtree_tgt;
        kdtree_tgt.setInputCloud (cloud_us_tgt);
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < cloud_us_src->size(); ++i) {
            kdtree_tgt.nearestKSearch (cloud_us_src->points[i], 1, nn_indices, nn_dists);
            if (nn_dists[0] <= max_fitness_range) {
                src_effe_indices[i] = nn_indices[0];
            }
            else {
                src_effe_indices[i] = 0;
            }
        }
        src_effe_indices.erase(std::remove(src_effe_indices.begin(), src_effe_indices.end(), 0), src_effe_indices.end());
        pcl::IndicesPtr src_effe_indices_ptr = boost::make_shared<std::vector<int>>(src_effe_indices);
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_us_src);
        extract.setIndices(src_effe_indices_ptr);
        extract.setNegative(false);
        extract.setKeepOrganized(false);
        extract.filter(*cloud_us_src_effe);
        cout << "Size of source cloud after effective point filter: " << cloud_us_src_effe->size() << endl;

        pcl::KdTreeFLANN<PointT> kdtree_src;
        kdtree_src.setInputCloud (cloud_us_src);
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < cloud_us_tgt->size(); ++i) {
            kdtree_src.nearestKSearch (cloud_us_tgt->points[i], 1, nn_indices, nn_dists);
            if (nn_dists[0] <= max_fitness_range) {
                tgt_effe_indices[i] = nn_indices[0];
            }
            else {
                tgt_effe_indices[i] = 0;
            }
        }
        tgt_effe_indices.erase(std::remove(tgt_effe_indices.begin(), tgt_effe_indices.end(), 0), tgt_effe_indices.end());
        pcl::IndicesPtr tgt_effe_indices_ptr = boost::make_shared<std::vector<int>>(tgt_effe_indices);
        pcl::ExtractIndices<PointT> extract_tgt;
        extract_tgt.setInputCloud(cloud_us_tgt);
        extract_tgt.setIndices(tgt_effe_indices_ptr);
        extract_tgt.setNegative(false);
        extract_tgt.setKeepOrganized(false);
        extract_tgt.filter(*cloud_us_tgt_effe);
        cout << "Size of target cloud after effective point filter: " << cloud_us_tgt_effe->size() << endl;
        cout << "Run time of effective point filter: " << timer_effe.getTimeSeconds() << " s" << endl;
    }

    /** invalid point filter **/
    (*cloud_us_tgt_effe).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_tgt_effe, *cloud_us_tgt_effe, null_indices_tgt);
    (*cloud_us_src_effe).is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud_us_src_effe, *cloud_us_src_effe, null_indices_src);

    /** get the init trans cloud & init fitness score **/
    CloudPtr cloud_init_trans_us (new CloudT);
    pcl::transformPointCloud(*cloud_us_src_effe, *cloud_init_trans_us, init_trans_mat);
    cout << "\nInit Trans Mat: \n " << init_trans_mat << endl;
    pcl::StopWatch timer_fs;
    cout << "Initial Fitness Score: " << GetIcpFitnessScore(cloud_us_tgt_effe, cloud_init_trans_us, max_fitness_range) << endl;
    cout << "Get fitness score time: " << timer_fs.getTimeSeconds() << " s" << endl;

    /** ICP **/
    pcl::StopWatch timer;
    timer.reset(); /** timing **/
    CloudPtr cloud_icp_trans_us (new CloudT);
    Eigen::Matrix4f icp_trans_mat;
    pcl::IterativeClosestPoint <PointT, PointT> icp; /** original icp **/

//    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> icp; /**  generalized icp **/

    icp.setInputTarget(cloud_us_tgt_effe);
    icp.setInputSource(cloud_us_src_effe);
    icp.setMaximumIterations(max_iters);
    icp.setMaxCorrespondenceDistance(max_corr_dis);
    icp.setTransformationEpsilon(trans_epsilon);
    icp.setEuclideanFitnessEpsilon(eucidean_epsilon);
    icp.align(*cloud_icp_trans_us, init_trans_mat);

//    pcl::NormalDistributionsTransform<PointT, PointT> icp; /** ndt **/
//    icp.setInputTarget(cloud_us_tgt);
//    icp.setInputSource(cloud_us_src);
//    icp.setMaximumIterations(max_iters);
//    icp.setTransformationEpsilon (trans_epsilon);
//    icp.setStepSize (1.0);
//    icp.setResolution (0.5);
//    icp.align(*cloud_icp_trans_us, init_trans_mat);

    if (icp.hasConverged()) {
        cout << "ICP run time: " << timer.getTimeSeconds() << " s" << endl;
        icp_trans_mat = icp.getFinalTransformation();
        cout << "\nICP has converged, calculated score is: " << GetIcpFitnessScore(cloud_us_tgt_effe, cloud_icp_trans_us, max_fitness_range) << endl;
        cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "\nICP Trans Mat: \n " << icp_trans_mat << endl;

        /** transfer rotation matrix to euler angle **/
        Eigen::Matrix3f icp_rotation_mat = icp_trans_mat.topLeftCorner<3, 3>();
        Eigen::Vector3f icp_euler_angle = icp_rotation_mat.eulerAngles(2, 1, 0); /** zyx euler angle **/
        cout << "Euler angle by ICP: \n" << icp_euler_angle << endl;
        cout << "debug1" << endl;
    }
    else {
        PCL_ERROR("\nICP has not converged.\n");
    }

    cout << "debug2" << endl;
    /** visualization **/
    if (kIcpViz) {
        pcl::visualization::PCLVisualizer viewer("ICP demo");
        int v1(0), v2(1); /** create two view point **/
        viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        float bckgr_gray_level = 0.0;  /** black **/
        float txt_gray_lvl = 1.0 - bckgr_gray_level;

        /** the color of original target cloud is white **/
        pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(cloud_us_tgt_effe, (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl,
                                                                                    (int)255 * txt_gray_lvl);
        viewer.addPointCloud(cloud_us_tgt_effe, cloud_aim_color_h, "cloud_aim_v1", v1);
        viewer.addPointCloud(cloud_us_tgt_effe, cloud_aim_color_h, "cloud_aim_v2", v2);

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
    cout << "debug3" << endl;
    tuple<Eigen::Matrix4f, CloudPtr> result;
    cout << "debug4" << endl;
    result = make_tuple(icp_trans_mat, cloud_icp_trans_us);
    cout << "debug5" << endl;
    return result;
}

double LidarProcess::GetIcpFitnessScore(CloudPtr cloud_tgt, CloudPtr cloud_src, double max_range) {
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
    cout << "size of loaded point cloud: " << view_raw_cloud->points.size() << endl;

    /** condition filter **/
    CloudPtr view_cloud(new CloudT);
    pcl::ConditionOr<PointT>::Ptr range_cond(new pcl::ConditionOr<PointT>());
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::GT, 0.3))); /** GT: greater than **/
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("z", pcl::ComparisonOps::LT, -0.4))); /** LT: less than **/
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new pcl::FieldComparison<PointT> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<PointT> cond_filter;
    cond_filter.setKeepOrganized(false);
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(view_raw_cloud);
    cond_filter.filter(*view_cloud);

    /** invalid point filter **/
    (*view_cloud).is_dense = false;
    std::vector<int> null_indices;
    pcl::removeNaNFromPointCloud(*view_cloud, *view_cloud, null_indices);
    (*view_cloud).is_dense = true;

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
    CloudPtr view_cloud_tgt(new CloudT);
    CloudPtr view_cloud_src(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(tgt_pcd_path, *view_cloud_tgt) == -1) {
        PCL_ERROR("Could Not Load Target File!\n");
    }
    cout << "Loaded " << view_cloud_tgt->size() << " points from target file" << endl;
    if (pcl::io::loadPCDFile<PointT>(src_pcd_path, *view_cloud_src) == -1) {
        PCL_ERROR("Could Not Load Source File!\n");
    }
    cout << "Loaded " << view_cloud_src->size() << " points from source file" << endl;

    /** initial rigid transformation **/
    float v_angle = (float)DEG2RAD(this->degree_map[this->view_idx]);
    float radius = 0.15f;
    Eigen::Matrix<float, 6, 1> trans_params;
    trans_params << 0.0f, v_angle, 0.0f,
                    radius * (sin(v_angle) - 0.0f), 0.0f, radius * (cos(v_angle) - 1.0f); /** LiDAR x-axis: car front; Gimbal positive angle: car front **/
    Eigen::Matrix4f init_trans_mat = ExtrinsicMat(trans_params);

    /** ICP **/
    std::tuple<Eigen::Matrix4f, CloudPtr> icp_result = ICP(view_cloud_tgt, view_cloud_src, init_trans_mat, 0, false);
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

void LidarProcess::FullViewMapping() {
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
    radius_outlier_filter.setRadiusSearch(0.1);
    radius_outlier_filter.setMinNeighborsInRadius(20);
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
    CloudPtr spot_cloud_tgt(new CloudT);
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[tgt_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_tgt);
    PCL_INFO("Size of Target Spot Cloud: %d\n", spot_cloud_tgt->size());
    CloudPtr spot_cloud_src(new CloudT);
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path,
                                 *spot_cloud_src);
    PCL_INFO("Size of Source Spot Cloud: %d\n", spot_cloud_src->size());

    /** initial transformation and initial score **/
    vector<Eigen::Matrix4f> icp_trans_mat_vec;
    string lio_trans_path = this->poses_files_path_vec[src_idx][0].lio_spot_trans_mat_path;
    Eigen::Matrix4f lio_spot_trans_mat = LoadTransMat(lio_trans_path);
    Eigen::Matrix3f lio_spot_rotation_mat = lio_spot_trans_mat.topLeftCorner<3, 3>();
    Eigen::Vector3f lio_euler_angle = lio_spot_rotation_mat.eulerAngles(2, 1, 0); // zyx euler angle
    cout << "Euler angle by LIO: \n" << lio_euler_angle << endl;
    cout << "Initial Trans Mat by LIO: \n" << lio_spot_trans_mat << endl;

    /** ICP **/
    std::tuple<Eigen::Matrix4f, CloudPtr> icp_result = ICP(spot_cloud_tgt, spot_cloud_src, lio_spot_trans_mat, 1, false);
    cout << "debug6" << endl;
    Eigen::Matrix4f icp_spot_trans_mat;
    CloudPtr spot_cloud_icp_trans;
    cout << "debug7" << endl;
    std::tie(icp_spot_trans_mat, spot_cloud_icp_trans) = icp_result;
    cout << "debug8" << endl;
    icp_trans_mat_vec.push_back(icp_spot_trans_mat);
    cout << "debug9" << endl;

    /** compared the fitness score of lio and icp **/
//    CloudPtr spot_lio_trans (new CloudT);
//    CloudPtr spot_icp_trans (new CloudT);
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

void LidarProcess::GlobalColoredMapping() {
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

void LidarProcess::GlobalMapping() {
    /** global cloud registration **/
    CloudPtr spot_clouds_registered(new CloudT);
    pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[0][0].fullview_dense_cloud_path,
                                 *spot_clouds_registered);
    for (int src_idx = 1; src_idx < this->num_spots; ++src_idx) {
        /** source index and target index **/
        int tgt_idx = 0;
        PCL_INFO("Spot %d to %d: \n", src_idx, tgt_idx);

        /** create point cloud container  **/
        CloudPtr spot_cloud_src(new CloudT);
        CloudPtr spot_cloud_us_src(new CloudT);

        /** load points **/
        pcl::io::loadPCDFile<PointT>(this->poses_files_path_vec[src_idx][0].fullview_dense_cloud_path,
                                     *spot_cloud_src);

        /** down sampling **/
        pcl::UniformSampling<PointT> us;
        us.setRadiusSearch(0.03f);
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
    }
    string global_registered_cloud_path = this->poses_files_path_vec[0][0].fullview_recon_folder_path +
                                          "/global_registered_cloud.pcd";
    pcl::io::savePCDFileBinary(global_registered_cloud_path, *spot_clouds_registered);
}