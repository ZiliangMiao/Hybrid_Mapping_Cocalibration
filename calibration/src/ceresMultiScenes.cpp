// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
// eigen
#include <Eigen/Core>
// ros
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <ros/package.h>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
// ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
// pcl
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <Eigen/Dense>
// headings
#include "FisheyeProcess.h"
#include "LidarProcess.h"
#include "spline.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace tk;

ofstream outfile;

/** kExtrinsics + kIntrinsics = number of parameters **/
static const int kExtrinsics = 6;
static const int kIntrinsics = 7;

/**
 * @brief Get double from type T (double and ceres::Jet) variables in the ceres optimization process
 *
 * @param x input variable with type T (double and ceres::Jet)
 * @return **
 */
double getDouble(double x) {
    return static_cast<double>(x);
}

template <typename SCALAR, int N>
double getDouble(const ceres::Jet<SCALAR, N> &x) {
    return static_cast<double>(x.a);
}

void customOutput(vector<const char *> name, double *params, vector<double> params_init) {
    std::cout << "Initial ";
    for (unsigned int i = 0; i < name.size(); i++)
    {
        std::cout << name[i] << ": " << params_init[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Final   ";
    for (unsigned int i = 0; i < name.size(); i++)
    {
        std::cout << name[i] << ": " << params[i] << " ";
    }
    std::cout << "\n";
}

tk::spline getPoly(vector<double> params){
    Eigen::Matrix<double, 6, 1> a_;
    switch (params.size()) {
    case 13:
        a_ << params[8], params[9], params[10], params[11], params[12], 0.0;
        break;
    case 12:
        a_ << params[8], params[9], 0.0, params[10], 0.0, params[11];
        break;
    default:
        a_ << 0.0, params[8], 0.0, params[9], 0.0, params[10];
        break;
    }
    int theta_ub = 180;
    std::vector<double> theta_seq(theta_ub);
    std::vector<double> radius_seq(theta_ub);
    for (double theta = 0; theta < theta_ub; ++theta)
    {
        theta_seq[theta] = theta;
        radius_seq[theta] = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
    }

    // default cubic spline (C^2) with natural boundary conditions (f''=0)
    tk::spline spline(radius_seq, theta_seq);			// X needs to be strictly increasing
    return spline;
}

void fusionViz(FisheyeProcess &fisheye, LidarProcess &lidar, vector<double> params, double bandwidth) {
    
    cv::Mat raw_image = fisheye.ReadFisheyeImage();
    cv::Mat lidarRGB = cv::Mat::zeros(raw_image.rows, raw_image.cols, CV_8UC3);
    cv::Mat merge_image = cv::Mat::zeros(raw_image.rows, raw_image.cols, CV_8UC3);

    /** write the edge points projected on fisheye to .txt file **/
    ofstream outfile;
    string edge_proj_txt_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].edge_fisheye_projection_path;
    outfile.open(edge_proj_txt_path, ios::out);

    vector<vector<double>> edge_lid_projection = lidar.EdgeCloudProjectToFisheye(params);
    for (int i = 0; i < edge_lid_projection[0].size(); i++) {
        double theta = edge_lid_projection[0][i];
        double phi = edge_lid_projection[1][i];
        int u = std::clamp(theta, (double)0.0, (double)(lidarRGB.rows - 1));
        int v = std::clamp(phi, (double)0.0, (double)(lidarRGB.cols - 1));
        int b = 0;
        int g = 0;
        int r = 255;
        lidarRGB.at<Vec3b>(u, v)[0] = b;
        lidarRGB.at<Vec3b>(u, v)[1] = g;
        lidarRGB.at<Vec3b>(u, v)[2] = r;
        outfile << u << "," << v << endl;
    }
    outfile.close();

    /** fusion image generation **/
    cv::addWeighted(raw_image, 1, lidarRGB, 1, 0, merge_image);

    tk::spline poly_spline = getPoly(params);

    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult =
        fisheye.FisheyeImageToSphere(merge_image, true, poly_spline);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    fisheye.SphereToPlane(camOrgPolarCloud, bandwidth);
}

void fusionViz3D(FisheyeProcess fisheye, LidarProcess lidar, vector<double> params) {
    Eigen::Vector3d r(params[0], params[1], params[2]);
    Eigen::Vector3d t{params[3], params[4], params[5]};
    Eigen::Vector2d uv_0{params[6], params[7]};
    Eigen::Matrix<double, 6, 1> a_;
    switch (params.size()) {
        case 13:
            a_ << params[8], params[9], params[10], params[11], params[12], 0.0;
            break;
        case 12:
            a_ << params[8], params[9], 0.0, params[10], 0.0, params[11];
            break;
        default:
            a_ << 0.0, params[8], 0.0, params[9], 0.0, params[10];
            break;
    }

    double theta;
    double inv_uv_radius, uv_radius;

    // extrinsic transform
    Eigen::Matrix<double, 3, 3> R;
    Eigen::AngleAxisd Rx(Eigen::AngleAxisd(r(0), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd Ry(Eigen::AngleAxisd(r(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd Rz(Eigen::AngleAxisd(r(2), Eigen::Vector3d::UnitZ()));
    R = Rz * Ry * Rx;

    Eigen::Vector3d lid_point;
    Eigen::Vector3d lid_trans;
    Eigen::Vector2d projection;

    string fullview_cloud_path;
    if (lidar.kDenseCloud) {
        fullview_cloud_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].fullview_dense_cloud_path;
    }
    else {
        fullview_cloud_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].fullview_sparse_cloud_path;
    }

    string fisheye_hdr_img_path = fisheye.poses_files_path_vec[fisheye.spot_idx][fisheye.fullview_idx].fisheye_hdr_img_path;

    CloudPtr fullview_cloud(new CloudT);
    RGBCloudPtr upward_cloud(new RGBCloudT);
    RGBCloudPtr downward_cloud(new RGBCloudT);
    RGBCloudPtr fullview_rgb_cloud(new RGBCloudT);
    pcl::io::loadPCDFile(fullview_cloud_path, *fullview_cloud);
    cv::Mat raw_image = fisheye.ReadFisheyeImage();

    RGBPointT pt;
    int u_max = 0, v_max = 0, u_min = 3000, v_min = 3000;
    cout << "--------------- Generating RGBXYZ Pointcloud ---------------" << endl;
    /** color at view0 **/
    for (auto &point : fullview_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        /** extrinsic trans & invers intrinsic trans **/
        lid_point << point.x, point.y, point.z;
        lid_trans = R * lid_point + t;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        
        if (0 <= u && u < target_view_img.rows && 0 <= v && v < target_view_img.cols) {
            pt.x = point.x;
            pt.y = point.y;
            pt.z = point.z;
            pt.b = target_view_img.at<cv::Vec3b>(u, v)[0];
            pt.g = target_view_img.at<cv::Vec3b>(u, v)[1];
            pt.r = target_view_img.at<cv::Vec3b>(u, v)[2];
            if (u > u_max){ u_max = u;}
            if (v > v_max){ v_max = v;}
            if (u < u_min){ u_min = u;}
            if (v < v_min){ v_min = v;}
            /** push the point back into one of the three point clouds **/
            /** 1200 1026 330 1070 **/
            if (inv_uv_radius < 350 || inv_uv_radius > 1050) {
                upward_cloud->points.push_back(pt);
            }
            else {
                fullview_rgb_cloud->points.push_back(pt);
            }
        }
        else {
            pt.x = point.x;
            pt.y = point.y;
            pt.z = point.z;
            upward_cloud->points.push_back(pt);
        }
    }
    cout << target_view_img.rows << " " << target_view_img.cols << endl;
    cout << u_min << " " << v_min << " " << u_max << " " << v_max << " " << endl;

    /** load upward view icp pose transform matrix **/
    int upward_view_idx = lid.fullview_idx + 1;
    string pose_trans_upward_mat_path = lid.poses_files_path_vec[lid.spot_idx][upward_view_idx].pose_trans_mat_path;
    std::ifstream mat_upward;
    mat_upward.open(pose_trans_upward_mat_path);
    Eigen::Matrix4f pose_trans_upward_mat;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            mat_upward >> pose_trans_upward_mat(j, k);
        }
    }
    mat_upward.close();
    cout << "Upward View: " << " Spot Index: " << lid.spot_idx << " View Index: " << upward_view_idx << "\n"
         << "ICP Trans Mat:" << "\n " << pose_trans_upward_mat << endl;
    Eigen::Matrix4f pose_trans_upward_mat_inv = pose_trans_upward_mat.inverse();

    /** load upward view fisheye image **/
    string upward_fisheye_hdr_img_path = cam.poses_files_path_vec[cam.spot_idx][upward_view_idx].fisheye_hdr_img_path;
    cv::Mat upward_view_img = cv::imread(upward_fisheye_hdr_img_path, cv::IMREAD_UNCHANGED); /** flip **/

    /** load downward view icp pose transform matrix **/
    int downward_view_idx = lid.fullview_idx - 1;
    string pose_trans_downward_mat_path = lid.poses_files_path_vec[lid.spot_idx][downward_view_idx].pose_trans_mat_path;
    std::ifstream mat_downward;
    mat_downward.open(pose_trans_downward_mat_path);
    Eigen::Matrix4f pose_trans_downward_mat;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            mat_downward >> pose_trans_downward_mat(j, k);
        }
    }
    mat_downward.close();
    cout << "Downward View: " << " Spot Index: " << lid.spot_idx << " View Index: " << downward_view_idx << "\n"
         << "ICP Trans Mat:" << "\n " << pose_trans_downward_mat << endl;
    Eigen::Matrix4f pose_trans_downward_mat_inv = pose_trans_downward_mat.inverse();

    /** load downward view fisheye image **/
    string downward_fisheye_hdr_img_path = cam.poses_files_path_vec[cam.spot_idx][downward_view_idx].fisheye_hdr_img_path;
    cv::Mat downward_view_img = cv::imread(downward_fisheye_hdr_img_path, cv::IMREAD_UNCHANGED); /** flip **/

    /** upward and downward cloud recolor **/
    /** inverse transformation to upward view **/
    pcl::transformPointCloud(*upward_cloud, *upward_cloud, pose_trans_upward_mat_inv);
    /** upward cloud recolor **/
    for (auto &point : upward_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        lid_trans << point.x, point.y, point.z;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        if (0 <= u && u < upward_view_img.rows && 0 <= v && v < upward_view_img.cols) {
            /** point cloud recolor at upward view **/
            point.b = upward_view_img.at<cv::Vec3b>(u, v)[0];
            point.g = upward_view_img.at<cv::Vec3b>(u, v)[1];
            point.r = upward_view_img.at<cv::Vec3b>(u, v)[2];
        }
        else {
            pt.x = point.x;
            pt.y = point.y;
            pt.z = point.z;
            downward_cloud->points.push_back(pt);
        }
    }
    /** transformation to target view **/
    pcl::transformPointCloud(*upward_cloud, *upward_cloud, pose_trans_upward_mat);
    pcl::transformPointCloud(*downward_cloud, *downward_cloud, pose_trans_upward_mat); /** the downward cloud here also need a trans **/

    /** inverse transformation to downward view **/
    pcl::transformPointCloud(*downward_cloud, *downward_cloud, pose_trans_downward_mat_inv);
    /** downward cloud recolor **/
    for (auto &point : downward_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        lid_trans << point.x, point.y, point.z;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        if (0 <= u && u < downward_view_img.rows && 0 <= v && v < downward_view_img.cols) {
            /** point cloud recolor at downward view **/
            point.b = downward_view_img.at<cv::Vec3b>(u, v)[0];
            point.g = downward_view_img.at<cv::Vec3b>(u, v)[1];
            point.r = downward_view_img.at<cv::Vec3b>(u, v)[2];
        }
    }
    /** transformation to target pose **/
    pcl::transformPointCloud(*downward_cloud, *downward_cloud, pose_trans_downward_mat);
    cout << fullview_cloud -> points.size() << " " << fullview_rgb_cloud -> points.size() << " " << upward_cloud -> points.size() << " " << downward_cloud -> points.size() << endl;

    /***** Visualization *****/
    pcl::visualization::PCLVisualizer viewer("Reconstruction");
    int v1(0); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    float bckgr_gray_level = 150;  /** black **/

    viewer.addPointCloud(fullview_rgb_cloud, "fullview_rgb_cloud", v1);
    viewer.addPointCloud(upward_cloud, "upward_cloud", v1);
    viewer.addPointCloud(downward_cloud, "downward_cloud", v1);
    viewer.addCoordinateSystem();
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  /** viewer size **/

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

struct Calibration {
    template <typename T>
    bool operator()(const T *const ext_, const T *const int_, T *cost) const {
        // intrinsic parameters
        Eigen::Matrix<T, 3, 1> r(ext_[0], ext_[1], ext_[2]);
        Eigen::Matrix<T, 3, 1> t{ext_[3], ext_[4], ext_[5]};
        Eigen::Matrix<T, 2, 1> uv_0{int_[0], int_[1]};
        Eigen::Matrix<T, 6, 1> a_;
        switch (kIntrinsics + kExtrinsics) {
        case 13:
            a_ << int_[2], int_[3], int_[4], int_[5], int_[6], T(0);
            break;
        case 12:
            a_ << int_[2], int_[3], T(0), int_[4], T(0), int_[5];
            break;
        default:
            a_ << T(0), int_[2], T(0), int_[3], T(0), int_[4];
            break;
        }

        Eigen::Matrix<T, 3, 1> lid_trans;

        T phi, theta, uv_radius;
        T inv_uv_radius;
        T res, val;

        /** extrinsic transform: 3d lidar edge points -> transformed 3d lidar points **/
        Eigen::Matrix<T, 3, 3> R;

        Eigen::AngleAxis<T> Rx(Eigen::AngleAxis<T>(r(0), Eigen::Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> Ry(Eigen::AngleAxis<T>(r(1), Eigen::Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> Rz(Eigen::AngleAxis<T>(r(2), Eigen::Matrix<T, 3, 1>::UnitZ()));
        R = Rz * Ry * Rx;
        lid_trans = R * lid_point_.cast<T>() + t;

        /** (inverse) intrinsic transform: transformed 3d lidar points -> 2d projection on fisheye image **/
        Eigen::Matrix<T, 2, 1> projection;

        theta = acos(lid_trans(2) / sqrt((lid_trans(0) * lid_trans(0)) + (lid_trans(1) * lid_trans(1)) + (lid_trans(2) * lid_trans(2))));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        kde_interpolator_.Evaluate(projection(0) * T(kde_scale_), projection(1) * T(kde_scale_), &val);

        res = T(weight_) * (T(kde_val_) - val);
        cost[0] = res;
        cost[1] = res;
        return true;
    }

    /** DO NOT remove the "&" of the interpolator! **/
    Calibration(const Eigen::Vector3d lid_point,
                const double weight,
                const double ref_val,
                const double scale,
                const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
        : lid_point_(std::move(lid_point)), kde_interpolator_(interpolator), weight_(weight), kde_val_(ref_val), kde_scale_(std::move(scale)) {}

    /**
     * @brief
     * create costfunction for optimization.
     * @param point-xyz coordinate of a 3d lidar edge point;
     * @param weight-weight assigned to the 3d lidar edge point;
     * @param kde_val-default reference value of lidar points;
     * @param kde_scale-scale of the kde image relative to original image;
     * @param interpolator-bicubic interpolator for original fisheye image;
     * @return ** ceres::CostFunction*
     */
    static ceres::CostFunction *Create(const Eigen::Vector3d &lid_point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<Calibration, 2, kExtrinsics, kIntrinsics>(
            new Calibration(lid_point, weight, kde_val, kde_scale, interpolator));
    }

    const Eigen::Vector3d lid_point_;
    const double weight_;
    const double kde_val_;
    const double kde_scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
    const double ref_theta = 99.5 * M_PI / 180;
};

/**
 * @brief
 * custom callback to print something after every iteration (inner iteration is not included)
 */
class OutputCallback : public ceres::IterationCallback
{
public:
    OutputCallback(double *params)
        : params_(params) {}

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary &summary) override {
        for (int i = 0; i < kIntrinsics + kExtrinsics; i++) {
            const double params_out = params_[i];
            outfile << params_out << "\t";
        }
        outfile << "\n";
        return ceres::SOLVER_CONTINUE;
    }

private:
    const double *params_;
};

/**
 * @brief
 * Ceres-solver Optimization
 * @param fisheye FisheyeProcess
 * @param lidar LidarProcess
 * @param bandwidth bandwidth for kde estimation(Gaussian kernel)
 * @param distortion distortion matrix {c, d; e, 1}
 * @param params_init initial parameters
 * @param name name of parameters
 * @param lb lower bounds of the parameters
 * @param ub upper bounds of the parameters
 * @return ** std::vector<double>
 */
std::vector<double> ceresMultiScenes(FisheyeProcess &fisheye,
                                     LidarProcess &lidar,
                                     double bandwidth,
                                     vector<double> params_init,
                                     vector<const char *> name,
                                     vector<double> lb,
                                     vector<double> ub,
                                     int kDisabledBlock) {
    const int kParams = params_init.size();
    const int kViews = fisheye.num_views;
    // const double scale = 1.0;
    const double scale = pow(2, (-(int)floor(log(bandwidth) / log(4))));
    double params[kParams];
    memcpy(params, &params_init[0], params_init.size() * sizeof(double));

    /********* Fisheye KDE -> bicubic interpolators *********/

    // std::vector<ceres::Grid2D<double>> grids;
    // std::vector<double> ref_vals;
    // std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;

    fisheye.SetSpotIdx(0);
    lidar.SetSpotIdx(0);
    fisheye.SetViewIdx((fisheye.num_views - 1) / 2);
    lidar.SetViewIdx((lidar.num_views - 1) / 2);

    /********* Fisheye KDE *********/
    vector<double> p_c = fisheye.Kde(bandwidth, scale);
    double *kde_val = new double[p_c.size()];
    memcpy(kde_val, &p_c[0], p_c.size() * sizeof(double));
    
    double ref_val = *max_element(p_c.begin(), p_c.end());
    const ceres::Grid2D<double> kde_grid(kde_val, 0, fisheye.kFisheyeRows * scale, 0, fisheye.kFisheyeCols * scale);
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(kde_grid);

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;

    problem.AddParameterBlock(params, kExtrinsics);
    problem.AddParameterBlock(params + kExtrinsics, kParams - kExtrinsics);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    Eigen::Vector2d img_size = {fisheye.kFisheyeRows, fisheye.kFisheyeCols};
    
    fisheye.SetSpotIdx(0);
    lidar.SetSpotIdx(0);
    fisheye.SetViewIdx((fisheye.num_views - 1) / 2);
    lidar.SetViewIdx((lidar.num_views - 1) / 2);
    /** a scene weight could be added here **/
    for (int j = 0; j < lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points.size(); ++j) {
        const double weight = lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points[j].intensity;
        Eigen::Vector3d lid_point = {lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points[j].x,
                                        lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points[j].y,
                                        lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points[j].z};
        problem.AddResidualBlock(Calibration::Create(lid_point, weight, ref_val,
                                                     scale, kde_interpolator),
                                    loss_function, params, params + kExtrinsics);
    }
    
    switch (kDisabledBlock) {
    case 1:
        problem.SetParameterBlockConstant(params);
        break;
    case 2:
        problem.SetParameterBlockConstant(params + kExtrinsics);
        break;
    default:
        break;
    }

    for (int i = 0; i < kParams; ++i) {
        if (i < kExtrinsics && kDisabledBlock != 1) {
            problem.SetParameterLowerBound(params, i, lb[i]);
            problem.SetParameterUpperBound(params, i, ub[i]);
        }
        else if (i >= kExtrinsics && kDisabledBlock != 2) {
            problem.SetParameterLowerBound(params + kExtrinsics, i - kExtrinsics, lb[i]);
            problem.SetParameterUpperBound(params + kExtrinsics, i - kExtrinsics, ub[i]);
        }
    }

    /********* Initial Options *********/

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 16;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-6;
    options.use_nonmonotonic_steps = false;

    // lidar.SetViewIdx(1);
    // string paramsOutPath = lidar.scenes_files_path_vec[lidar.spot_idx][lidar.view_idx].output_folder_path +
    //                        "/ParamsRecord_" + to_string(bandwidth) + ".txt";
    // outfile.open(paramsOutPath);
    // OutputCallback callback(params);
    // options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    customOutput(name, params, params_init);
    outfile.close();

    /********* 2D Image Visualization *********/

    std::vector<double> params_res(params, params + sizeof(params) / sizeof(double));
    fisheye.SetSpotIdx(0);
    lidar.SetSpotIdx(0);
    fisheye.SetViewIdx((fisheye.num_views - 1) / 2);
    lidar.SetViewIdx((lidar.num_views - 1) / 2);
    fusionViz(fisheye, lidar, params_res, bandwidth);


    return params_res;
}
