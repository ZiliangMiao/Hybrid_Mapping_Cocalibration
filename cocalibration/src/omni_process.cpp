/** headings **/
#include <omni_process.h>
#include <common_lib.h>
#include <define.h>

/** namespace **/
using namespace std;
using namespace cv;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;
using namespace arma;

OmniProcess::OmniProcess() {
    /** Param **/
    ros::param::get("essential/kDatasetName", this->DATASET_NAME);
    ros::param::get("essential/kNumSpot", this->NUM_SPOT);
    ros::param::get("essential/kImageRows", this->kImageSize.first);
    ros::param::get("essential/kImageCols", this->kImageSize.second);

    this->ocamEdgeCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    /** Path **/
    this->PKG_PATH = ros::package::getPath("cocalibration");
    this->DATASET_PATH = this->PKG_PATH + "/data/" + this->DATASET_NAME;
    this->COCALIB_PATH = this->DATASET_PATH + "/cocalibration";
    this->EDGE_PATH = this->COCALIB_PATH + "/edges";
    this->RESULT_PATH = this->COCALIB_PATH + "/results";
    this->PYSCRIPT_PATH = this->PKG_PATH + "/python_scripts/image_process/edge_extraction.py";

    this->cocalibImagePath = this->COCALIB_PATH + "/hdr_image.bmp";
    this->cocalibEdgeImagePath = this->EDGE_PATH + "/omni_edge_image.bmp";
    this->cocalibEdgeCloudPath = this->EDGE_PATH + "/edge_cloud.pcd";
    this->cocalibKdePath = RESULT_PATH + "/kde_samples.txt";
}

void OmniProcess::loadCocalibImage() {
    this->cocalibImage = cv::imread(this->cocalibImagePath, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((image.rows != 0 || image.cols != 0),
                   "Invalid size (%d, %d) from file: %s", image.rows, image.cols, cocalibImagePath);
    ROS_ASSERT_MSG((!MESSAGE_EN),
                    "Loaded image from file: %s", cocalibImagePath);
}

void OmniProcess::edgeExtraction() {
    ROS_INFO("Run pythonscripts, Extract ocam edges");
    string mode = "omni";
    string cmd_str = "python3 " + this->PYSCRIPT_PATH + " " + this->DATASET_PATH + " " + mode;
    int status = system(cmd_str.c_str());
}

void OmniProcess::generateEdgeCloud() {
    cv::Mat edge_img = cv::imread(this->cocalibEdgeImagePath, cv::IMREAD_UNCHANGED);
    ROS_ASSERT_MSG((image.rows != 0 || image.cols != 0),
                   "Invalid size (%d, %d) from file: %s", image.rows, image.cols, edge_img_path);
    for (int u = 0; u < edge_img.rows; ++u) {
        for (int v = 0; v < edge_img.cols; ++v) {
            if (edge_img.at<uchar>(u, v) > 127) {
                pcl::PointXYZ edge_pt;
                edge_pt.x = u;
                edge_pt.y = v;
                edge_pt.z = 1;
                this->ocamEdgeCloud->points.push_back(edge_pt);
            }
        }
    }
}

vector<double> OmniProcess::Kde(double bandwidth, double scale) {
    clock_t start_time = clock();
    const double default_rel_error = 0.05;
    const int n_rows = scale * this->kImageSize.first;
    const int n_cols = scale * this->kImageSize.second;
    arma::mat query;
    // number of rows equal to number of dimensions, query.n_rows == reference.n_rows is required
    const int ref_size = this->ocamEdgeCloud->size();
    arma::mat reference(2, ref_size);
    for (int i = 0; i < ref_size; ++i) {
        reference(0, i) = this->ocamEdgeCloud->points[i].x;
        reference(1, i) = this->ocamEdgeCloud->points[i].y;
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

    if (EXTRA_FILE_EN) {
        /** Kde Prediction **/
        ofstream outfile;
        outfile.open(this->cocalibKdePath, ios::out);
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
    if (MESSAGE_EN) {
        ROS_INFO("Kde image generated in %f s.\n bandwidth = %f, size = (%d, %d)", ((float)(clock() - start_time) / CLOCKS_PER_SEC), bandwidth, n_rows, n_cols);
    }
    return img;
}
