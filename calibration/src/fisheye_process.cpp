/** headings **/
#include <fisheye_process.h>
#include <common_lib.h>

/** namespace **/
using namespace std;
using namespace cv;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;
using namespace arma;

FisheyeProcess::FisheyeProcess() {
    /** parameter server **/
    ros::param::get("essential/kDatasetName", this->dataset_name);
    ros::param::get("essential/kNumSpots", this->num_spots);
    ros::param::get("essential/kNumViews", this->num_views);
    ros::param::get("essential/kFisheyeRows", this->kImageSize.first);
    ros::param::get("essential/kFisheyeCols", this->kImageSize.second);
    ros::param::get("essential/kAngleInit", this->view_angle_init);
    ros::param::get("essential/kAngleStep", this->view_angle_step);
    this->kDatasetPath = this->kPkgPath + "/data/" + this->dataset_name;
    this->fullview_idx = (this->num_views - 1) / 2;

    /** filepath and edge cloud**/
    vector<vector<string>> folder_path_tmp(num_spots, vector<string>(num_views));
    vector<vector<PoseFilePath>> file_path_tmp(num_spots, vector<PoseFilePath>(num_views));
    vector<vector<EdgeCloud>> edge_clouds_tmp(num_spots, vector<EdgeCloud>(num_views));
    folder_path_vec = folder_path_tmp;
    file_path_vec = file_path_tmp;
    edge_cloud_vec = edge_clouds_tmp;


    /** degree map **/
    for (int i = 0; i < num_spots; ++i) {
        for (int j = 0; j < num_views; ++j) {
            int v_degree = view_angle_init + view_angle_step * j;
            this->degree_map[j] = v_degree;
            this->folder_path_vec[i][j] = kDatasetPath + "/spot" + to_string(i) + "/" + to_string(v_degree);
            struct PoseFilePath pose_file_path(folder_path_vec[i][j]);
            this->file_path_vec[i][j] = pose_file_path;
        }
    }
}

void FisheyeProcess::ReadEdge() {
    string edge_cloud_path = this->file_path_vec[spot_idx][view_idx].edge_cloud_path;
    LoadPcd(edge_cloud_path, this->edge_cloud_vec[spot_idx][view_idx], "camera edge");
}

cv::Mat FisheyeProcess::LoadImage(bool output) {
    PoseFilePath &path_vec = this->file_path_vec[spot_idx][view_idx];
    string img_path = path_vec.hdr_img_path;
    cv::Mat image = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((image.rows != 0 || image.cols != 0),
                   "Invalid size (%d, %d) from file: %s", image.rows, image.cols, img_path);
    ROS_ASSERT_MSG((!FULL_OUTPUT),
                    "Loaded image from file: %s", img_path)
    if (output) {
        string output_img_path = path_vec.flat_img_path;
        cv::imwrite(output_img_path, image);
    }
    return image;
}


void FisheyeProcess::GenerateEdgeCloud() {
    string edge_img_path = file_path_vec[spot_idx][view_idx].edge_img_path;
    cv::Mat edge_img = cv::imread(edge_img_path, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((image.rows != 0 || image.cols != 0),
                   "Invalid size (%d, %d) from file: %s", image.rows, image.cols, edge_img_path);
    
    EdgeCloud::Ptr edge_cloud(new EdgeCloud);

    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                pcl::PointXYZ edge_pt;
                edge_pt.x = u;
                edge_pt.y = v;
                edge_pt.z = 1;
                edge_cloud->points.push_back(edge_pt);
            }
        }
    }
    this->edge_cloud_vec[spot_idx][view_idx] = *edge_cloud;
    string edge_cloud_path = file_path_vec[spot_idx][view_idx].edge_cloud_path;
    pcl::io::savePCDFileBinary(edge_cloud_path, *edge_cloud);
}

std::vector<double> FisheyeProcess::Kde(double bandwidth, double scale) {
    clock_t start_time = clock();
    const double default_rel_error = 0.05;
    const int n_rows = scale * this->kImageSize.first;
    const int n_cols = scale * this->kImageSize.second;
    arma::mat query;
    // number of rows equal to number of dimensions, query.n_rows == reference.n_rows is required
    EdgeCloud &fisheye_edge = this->edge_cloud_vec[this->spot_idx][this->view_idx];
    const int ref_size = fisheye_edge.size();
    arma::mat reference(2, ref_size);
    for (int i = 0; i < ref_size; ++i) {
        reference(0, i) = fisheye_edge.points[i].x;
        reference(1, i) = fisheye_edge.points[i].y;
    }

    query = arma::mat(2, n_cols * n_rows);
    arma::vec rows = arma::linspace(0, this->kImageSize.first - 1, n_rows);
    arma::vec cols = arma::linspace(0, this->kImageSize.second - 1, n_cols);

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

    if (FULL_OUTPUT) {
        /** kde prediction output **/
        string kde_txt_path = this->file_path_vec[this->spot_idx][this->view_idx].kde_samples_path;
        ofstream outfile;
        outfile.open(kde_txt_path, ios::out);
        if (!outfile.is_open()) {
            cout << "Open file failure" << endl;
        }
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; j++) {
                int index = i * n_cols + j;
                outfile << query.at(0, index) << "\t"
                        << query.at(1, index) << "\t"
                        << kde_estimations(index) << endl;
            }
        }
        outfile.close();
    }

    cout << "New kde image generated with size (" << n_rows << ", " << n_cols << ") in "
         <<(double)(clock() - start_time) / CLOCKS_PER_SEC << "s, bandwidth = " << bandwidth << endl;
    return img;
}

void FisheyeProcess::EdgeExtraction() {
    std::string script_path = this->kPkgPath + "/python_scripts/image_process/edge_extraction.py";
    std::string kSpots = to_string(this->spot_idx);
    std::string cmd_str = "python3 " 
        + script_path + " " + this->kDatasetPath + " " + "fisheye" + " " + kSpots;
    int status = system(cmd_str.c_str());
}
