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

using namespace std;
using namespace cv;
using namespace Eigen;

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

// void initQuaternion(double rx, double ry, double rz, vector<double> &init) {
//     Eigen::Vector3d r(rx, ry, rz);
//     Eigen::AngleAxisd Rx(Eigen::AngleAxisd(r(0), Eigen::Vector3d::UnitX()));
//     Eigen::AngleAxisd Ry(Eigen::AngleAxisd(r(1), Eigen::Vector3d::UnitY()));
//     Eigen::AngleAxisd Rz(Eigen::AngleAxisd(r(2), Eigen::Vector3d::UnitZ()));
//     Eigen::Matrix3d R;
//     R = Rz * Ry * Rx;
//     Eigen::Quaterniond q(R);
//     init[0] = q.x();
//     init[1] = q.y();
//     init[2] = q.z();
//     init[3] = q.w();
// }

void fusionViz(FisheyeProcess cam, string edge_proj_txt_path, vector<vector<double>> lid_projection, double bandwidth) {
    cv::Mat raw_image = cam.ReadFisheyeImage();
    int rows = raw_image.rows;
    int cols = raw_image.cols;
    cv::Mat lidarRGB = cv::Mat::zeros(rows, cols, CV_8UC3);

    /** write the edge points projected on fisheye to .txt file **/
    ofstream outfile;
    outfile.open(edge_proj_txt_path, ios::out);
    for (int i = 0; i < lid_projection[0].size(); i++) {
        double theta = lid_projection[0][i];
        double phi = lid_projection[1][i];
        int u = std::clamp(lidarRGB.rows - 1 - theta, (double)0.0, (double)(lidarRGB.rows - 1));
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
    cv::Mat imageShow = cv::Mat::zeros(rows, cols, CV_8UC3);
    cv::addWeighted(raw_image, 1, lidarRGB, 0.8, 0, imageShow);

    /***** need to be modified *****/
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult =
        cam.FisheyeImageToSphere(imageShow);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    cam.SphereToPlane(camOrgPolarCloud, bandwidth);
}

void fusionViz3D(FisheyeProcess cam, LidarProcess lid, vector<double> params) {
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

    string fullview_cloud_path = lid.scenes_path_vec[lid.spot_idx][(lid.num_views - 1) / 2] +
                                 "/full_view/fullview_cloud.pcd";
    string fisheye_hdr_img_path = cam.scenes_files_path_vec[cam.spot_idx][(cam.num_views-1)/2].fisheye_hdr_img_path;

    CloudPtr fullview_cloud(new CloudT);
    RGBCloudPtr upward_cloud(new RGBCloudT);
    RGBCloudPtr downward_cloud(new RGBCloudT);
    RGBCloudPtr fullview_rgb_cloud(new RGBCloudT);

    pcl::io::loadPCDFile(fullview_cloud_path, *fullview_cloud);
    cv::Mat raw_image = cv::imread(fisheye_hdr_img_path, cv::IMREAD_UNCHANGED);

    RGBPointT pt;
    cout << "--------------- Generating RGBXYZ Pointcloud ---------------" << endl;
    for (auto & point : fullview_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        lid_point << point.x, point.y, point.z;
        lid_trans = R * lid_point + t;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), -inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        double fisheye_radius = sqrt(pow((u - uv_0(0)), 2) + pow((v - uv_0(1)), 2));
        if (0 <= u && u < raw_image.rows && 0 <= v && v < raw_image.cols) {
            pt.x = lid_point(0);
            pt.y = lid_point(1);
            pt.z = lid_point(2);
            pt.b = raw_image.at<cv::Vec3b>(u, v)[0];
            pt.g = raw_image.at<cv::Vec3b>(u, v)[1];
            pt.r = raw_image.at<cv::Vec3b>(u, v)[2];
            /** push the point back into one of the three point clouds **/
            /** 1200 1026 330 1070 **/
            if (fisheye_radius < 330) {
                upward_cloud -> points.push_back(pt);
            }
            else if (fisheye_radius > 1070) {
                downward_cloud -> points.push_back(pt);
            }
            else {
                fullview_rgb_cloud -> points.push_back(pt);
            }
        }
        else {
            if (fisheye_radius < 330) {
                upward_cloud -> points.push_back(pt);
            }
            else if (fisheye_radius > 1070) {
                downward_cloud -> points.push_back(pt);
            }
        }
    }

    /** upward and downward cloud recolor **/
    /** load icp pose transform matrix **/
    string pose_trans_upward_mat_path = lid.scenes_files_path_vec[lid.spot_idx][2].pose_trans_mat_path;
    std::ifstream mat_upward;
    mat_upward.open(pose_trans_upward_mat_path);
    Eigen::Matrix4f pose_trans_upward_mat;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            mat_upward >> pose_trans_upward_mat(j, k);
        }
    }
    mat_upward.close();
    Eigen::Matrix4f pose_trans_upward_mat_inv = pose_trans_upward_mat.inverse();

    string pose_trans_downward_mat_path = lid.scenes_files_path_vec[lid.spot_idx][0].pose_trans_mat_path;
    std::ifstream mat_downward;
    mat_downward.open(pose_trans_downward_mat_path);
    Eigen::Matrix4f pose_trans_downward_mat;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            mat_downward >> pose_trans_downward_mat(j, k);
        }
    }
    mat_downward.close();
    Eigen::Matrix4f pose_trans_downward_mat_inv = pose_trans_downward_mat.inverse();

    /** inverse transformation to upward pose **/
    pcl::transformPointCloud(*upward_cloud, *upward_cloud, pose_trans_upward_mat_inv);
    /** upward cloud recolor **/
    for (auto & point : upward_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        lid_point << point.x, point.y, point.z;
        lid_trans = R * lid_point + t;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), -inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        if (0 <= u && u < raw_image.rows && 0 <= v && v < raw_image.cols) {
            /** to be done **/
            point.b = 0;
            point.g = 255;
            point.r = 0;
        }
    }
    /** transformation to target pose **/
    pcl::transformPointCloud(*upward_cloud, *upward_cloud, pose_trans_upward_mat);

    /** inverse transformation to downward pose **/
    pcl::transformPointCloud(*downward_cloud, *downward_cloud, pose_trans_downward_mat_inv);
    /** upward cloud recolor **/
    for (auto & point : downward_cloud->points) {
        if (point.x == 0 && point.y == 0 && point.z == 0) {
            continue;
        }
        lid_point << point.x, point.y, point.z;
        lid_trans = R * lid_point + t;
        theta = acos(lid_trans(2) / sqrt(pow(lid_trans(0), 2) + pow(lid_trans(1), 2) + pow(lid_trans(2), 2)));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), -inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        int u = floor(projection(0));
        int v = floor(projection(1));
        if (0 <= u && u < raw_image.rows && 0 <= v && v < raw_image.cols) {
            /** to be done **/
            point.b = 0;
            point.g = 0;
            point.r = 255;
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

        /** extrinsic transform for original 3d lidar edge points **/
        Eigen::Matrix<T, 3, 3> R;

        Eigen::AngleAxis<T> Rx(Eigen::AngleAxis<T>(r(0), Eigen::Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> Ry(Eigen::AngleAxis<T>(r(1), Eigen::Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> Rz(Eigen::AngleAxis<T>(r(2), Eigen::Matrix<T, 3, 1>::UnitZ()));
        R = Rz * Ry * Rx;
        lid_trans = R * lid_point_.cast<T>() + t;

        /** extrinsic transform for transformed 3d lidar edge points **/
        Eigen::Matrix<T, 2, 1> projection;

        /** r - theta representation: r = f(theta) in polynomial form **/
        theta = acos(lid_trans(2) / sqrt((lid_trans(0) * lid_trans(0)) + (lid_trans(1) * lid_trans(1)) + (lid_trans(2) * lid_trans(2))));
        inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        // phi = atan2(lid_trans(1), lid_trans(0));
        // inv_u = (inv_r * cos(phi) + u0);
        // inv_v = (inv_r * sin(phi) + v0);

        /** compute undistorted uv coordinate of lidar projection point and evaluate the value **/
        uv_radius = sqrt(lid_trans(1) * lid_trans(1) + lid_trans(0) * lid_trans(0));
        projection = {-inv_uv_radius / uv_radius * lid_trans(0) + uv_0(0), -inv_uv_radius / uv_radius * lid_trans(1) + uv_0(1)};
        kde_interpolator_.Evaluate(projection(0) * T(kde_scale_), projection(1) * T(kde_scale_), &val);

        /**
         * residual:
         * 1. diff. of kde image evaluation and lidar point projection(related to constant "ref_val_" and unified radius));
         * 2. polynomial correction: r_max = F(theta_max);
         *  **/
        //        res = T(ref_val_) * (T(1.0) - inv_r * T(0.5 / img_size_[1])) - val + T(1e-8) * abs(T(1071) - a_(0) - a_(1) * T(ref_theta) - a_(2) * T(pow(ref_theta, 2)) - a_(3) * T(pow(ref_theta, 3)) - a_(4) * T(pow(ref_theta, 4) - a_(5) * T(pow(ref_theta, 5))));
        //        res = T(ref_val_) * (T(1.0) - inv_r * T(0.5 / img_size_[1])) - val;
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
     * create multiscenes costfunction for optimization.
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
 * @param cam FisheyeProcess
 * @param lid LidarProcess
 * @param bandwidth bandwidth for kde estimation(Gaussian kernel)
 * @param distortion distortion matrix {c, d; e, 1}
 * @param params_init initial parameters
 * @param name name of parameters
 * @param lb lower bounds of the parameters
 * @param ub upper bounds of the parameters
 * @return ** std::vector<double>
 */
std::vector<double> ceresMultiScenes(FisheyeProcess cam,
                                     LidarProcess lid,
                                     double bandwidth,
                                     vector<double> params_init,
                                     vector<const char *> name,
                                     vector<double> lb,
                                     vector<double> ub,
                                     int kDisabledBlock) {
    const int kParams = params_init.size();
    const int kViews = cam.num_views;
    // const double scale = 1.0;
    const double scale = 2.0 / bandwidth;

    double params[kParams];
    memcpy(params, &params_init[0], params_init.size() * sizeof(double));

    std::vector<ceres::Grid2D<double>> grids;
    std::vector<double> ref_vals;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;

    lid.SetSpotIdx(0);
    cam.SetSpotIdx(0);
    for (int idx = 0; idx < kViews; idx++) {
        lid.SetViewIdx(idx);
        cam.SetViewIdx(idx);
        /********* Fisheye KDE *********/
        vector<double> p_c = cam.Kde(bandwidth, scale, false);
        // Data is a row-major array of kGridRows x kGridCols values of function
        // f(x, y) on the grid, with x in {-kGridColsHalf, ..., +kGridColsHalf},
        // and y in {-kGridRowsHalf, ..., +kGridRowsHalf}
        double *kde_val = new double[p_c.size()];
        memcpy(kde_val, &p_c[0], p_c.size() * sizeof(double));
        const ceres::Grid2D<double> kde_grid(kde_val, 0, cam.kFisheyeRows * scale, 0, cam.kFisheyeCols * scale);
        grids.push_back(kde_grid);
        ref_vals.push_back(*max_element(p_c.begin(), p_c.end()));
    }
    const std::vector<ceres::Grid2D<double>> img_grids(grids);
    lid.SetSpotIdx(0);
    cam.SetSpotIdx(0);
    for (int idx = 0; idx < kViews; idx++) {
        lid.SetViewIdx(idx);
        cam.SetViewIdx(idx);
        const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(img_grids[idx]);
        interpolators.push_back(kde_interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> img_interpolators(interpolators);

    // Ceres Problem
    ceres::Problem problem;

    problem.AddParameterBlock(params, kExtrinsics);
    problem.AddParameterBlock(params + kExtrinsics, kParams - kExtrinsics);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    Eigen::Vector2d img_size = {cam.kFisheyeRows, cam.kFisheyeCols};
    lid.SetSpotIdx(0);
    cam.SetSpotIdx(0);
    for (int idx = 0; idx < kViews; idx++) {
        lid.SetViewIdx(idx);
        cam.SetViewIdx(idx);
        /** a scene weight could be added here **/
        for (int j = 0; j < lid.edge_cloud_vec[lid.spot_idx][idx]->points.size(); ++j) {
            const double weight = lid.edge_cloud_vec[lid.spot_idx][idx]->points[j].intensity;
            Eigen::Vector3d lid_point = {lid.edge_cloud_vec[lid.spot_idx][idx]->points[j].x,
                                         lid.edge_cloud_vec[lid.spot_idx][idx]->points[j].y,
                                         lid.edge_cloud_vec[lid.spot_idx][idx]->points[j].z};
            problem.AddResidualBlock(Calibration::Create(lid_point, weight, ref_vals[idx],
                                                         scale, img_interpolators[idx]),
                                     loss_function, params, params + kExtrinsics);
        }
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

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 16;
    options.function_tolerance = 1e-7;
    options.use_nonmonotonic_steps = false;

    lid.SetViewIdx(1);
    string paramsOutPath = lid.scenes_files_path_vec[lid.spot_idx][lid.view_idx].output_folder_path +
                           "/ParamsRecord_" + to_string(bandwidth) + ".txt";
    outfile.open(paramsOutPath);
    OutputCallback callback(params);
    options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    customOutput(name, params, params_init);
    outfile.close();

    std::vector<double> params_res(params, params + sizeof(params) / sizeof(double));
    lid.SetSpotIdx(0);
    cam.SetSpotIdx(0);
    for (int idx = 0; idx < kViews; idx++) {
        lid.SetViewIdx(idx);
        cam.SetViewIdx(idx);
        vector<vector<double>> edge_fisheye_projection = lid.EdgeCloudProjectToFisheye(params_res);
        string edge_proj_txt_path = lid.scenes_files_path_vec[lid.spot_idx][lid.view_idx].edge_fisheye_projection_path;
        fusionViz(cam, edge_proj_txt_path, edge_fisheye_projection, bandwidth);
    }

    return params_res;
}

// /**
//  * @brief
//  * Ceres-solver Optimization
//  * @param cam camProcess
//  * @param lid LidarProcess
//  * @param bandwidth bandwidth for kde estimation(Gaussian kernel)
//  * @param distortion distortion matrix {c, d; e, 1}
//  * @param params_init initial parameters
//  * @param name name of parameters
//  * @param lb lower bounds of the parameters
//  * @param ub upper bounds of the parameters
//  * @return ** std::vector<double>
//  */
// std::vector<double> ceresAutoDiff(FisheyeProcess cam,
//                                   LidarProcess lid,
//                                   double bandwidth,
//                                   const Eigen::Matrix2d distortion,
//                                   vector<double> params_init,
//                                   vector<const char *> name,
//                                   vector<double> lb,
//                                   vector<double> ub)
// {
//     std::vector<double> p_c = cam.Kde(bandwidth, 1.0, false);
//     const double ref_val = *max_element(p_c.begin(), p_c.end()) / (0.125 * bandwidth);
//     const int num_params = params_init.size();

//     // initQuaternion(0.0, -0.01, M_PI, param_init);

//     double params[num_params];
//     memcpy(params, &params_init[0], params_init.size() * sizeof(double));
//     Eigen::Matrix2d inv_distortion = distortion.inverse();
//     // std::copy(std::begin(params_init), std::end(params_init), std::begin(params));

//     // Data is a row-major array of kGridRows x kGridCols values of function
//     // f(x, y) on the grid, with x in {-kGridColsHalf, ..., +kGridColsHalf},
//     // and y in {-kGridRowsHalf, ..., +kGridRowsHalf}
//     double *kde_data = new double[p_c.size()];
//     memcpy(kde_data, &p_c[0], p_c.size() * sizeof(double));

//     // unable to set coordinate to 2D grid for corresponding interpolator;
//     // use post-processing to scale the grid instead.
//     const ceres::Grid2D<double> kde_grid(kde_data, 0, cam.kdeRows, 0, cam.kdeCols);
//     const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(kde_grid);

//     // Ceres Problem
//     // ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();
//     ceres::Problem problem;

//     // problem.AddParameterBlock(params, 4, q_parameterization);
//     // problem.AddParameterBlock(params + 4, num_params - 4);
//     problem.AddParameterBlock(params, num_q);
//     problem.AddParameterBlock(params + num_q, num_params - num_q);
//     ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

//     Eigen::Vector2d img_size = {cam.kFisheyeRows, cam.kFisheyeCols};
//     for (int i = 0; i < lid.IntensityCloudPtr -> points.size(); ++i)
//     {
//         // Eigen::Vector3d p_l_tmp = p_l.row(i);
//         Eigen::Vector3d p_l_tmp = {lid.IntensityCloudPtr -> points[i].x, lid.IntensityCloudPtr -> points[i].y, lid.IntensityCloudPtr -> points[i].z};
//         problem.AddResidualBlock(Calibration::Create(p_l_tmp, img_size, ref_val, kde_interpolator, inv_distortion),
//                                  loss_function,
//                                  params,
//                                  params + num_q);
//     }

//     for (int i = 0; i < num_params; ++i)
//     {
//         if (i < num_q)
//         {
//             problem.SetParameterLowerBound(params, i, lb[i]);
//             problem.SetParameterUpperBound(params, i, ub[i]);
//         }
//         else
//         {
//             problem.SetParameterLowerBound(params + num_q, i - num_q, lb[i]);
//             problem.SetParameterUpperBound(params + num_q, i - num_q, ub[i]);
//         }
//     }

//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_SCHUR;
//     options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//     options.minimizer_progress_to_stdout = true;
//     options.num_threads = 12;
//     options.function_tolerance = 1e-7;
//     options.use_nonmonotonic_steps = true;

//     // OutputCallback callback(params);
//     // options.callbacks.push_back(&callback);

//     ceres::Solver::Summary summary;

//     ceres::Solve(options, &problem, &summary);

//     std::cout << summary.FullReport() << "\n";
//     customOutput(name, params, params_init);
//     std::vector<double> params_res(params, params + sizeof(params) / sizeof(double));

//     vector<vector<double>> lidProjection = lid.edgeVizTransform(params_res, distortion);
//     string lidEdgeTransTxtPath = lid.scenes_files_path_vec[lid.pose_idx].edge_fisheye_projection_path;
//     fusionViz(cam, lidEdgeTransTxtPath, lidProjection, bandwidth);

//     return params_res;
// }
