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
#include <Eigen/Core>
#include <Eigen/Dense>
// headings
#include "FisheyeProcess.h"
#include "LidarProcess.h"

using namespace std;
using namespace cv;
using namespace Eigen;

ofstream outfile;

/** num_p + num_q = number of parameters **/
static const int num_q = 6;
static const int num_p = 7;

/**
 * @brief Get double from type T (double and ceres::Jet) variables in the ceres optimization process
 *
 * @param x input variable with type T (double and ceres::Jet)
 * @return **
 */
double get_double(double x) {
    return static_cast<double>(x);
}

template <typename SCALAR, int N>
double get_double(const ceres::Jet<SCALAR, N> &x) {
    return static_cast<double>(x.a);
}

void customOutput(vector<const char *> name, double *params, vector<double> params_init) {
    std::cout << "Initial ";
    for (unsigned int i = 0; i < name.size(); i++) {
        std::cout << name[i] << ": " << params_init[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Final   ";
    for (unsigned int i = 0; i < name.size(); i++) {
        std::cout << name[i] << ": " << params[i] << " ";
    }
    std::cout << "\n";
}

void initQuaternion(double rx, double ry, double rz, vector<double> &init) {
    Eigen::Vector3d eulerAngle(rx, ry, rz);
    Eigen::AngleAxisd xAngle(Eigen::AngleAxisd(eulerAngle(0), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d R;
    R = zAngle * yAngle * xAngle;
    Eigen::Quaterniond q(R);
    init[0] = q.x();
    init[1] = q.y();
    init[2] = q.z();
    init[3] = q.w();
}

void fusionViz(FisheyeProcess cam, string edge_proj_txt_path, vector< vector<double> > lidProjection, double bandwidth){
    cv::Mat image = cam.ReadFisheyeImage();
    int rows = image.rows;
    int cols = image.cols;
    cv::Mat lidarRGB = cv::Mat::zeros(rows, cols, CV_8UC3);
//    double pixPerRad = 1000 / (M_PI/2);

    /** write the edge points projected on fisheye to .txt file **/
    ofstream outfile;
    outfile.open(edge_proj_txt_path, ios::out);
    for (int i = 0; i < lidProjection[0].size(); i++){
        double theta = lidProjection[0][i];
        double phi = lidProjection[1][i];
        // int u = (int)pixPerRad * theta;
        // int v = (int)pixPerRad * phi;
        int u = std::clamp(lidarRGB.rows - 1 - theta, (double)0.0, (double)(lidarRGB.rows-1));
        int v = std::clamp(phi, (double)0.0, (double)(lidarRGB.cols-1));;
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
    cv::addWeighted(image, 1, lidarRGB, 0.8, 0, imageShow);

    /***** need to be modified *****/
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> camResult =
            cam.FisheyeImageToSphere(imageShow);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr camOrgPixelCloud;
    std::tie(camOrgPolarCloud, camOrgPixelCloud) = camResult;
    cam.SphereToPlane(camOrgPolarCloud, bandwidth);
}

void fusionViz3D(FisheyeProcess cam, LidarProcess lid, vector<double> _p) {

    Eigen::Matrix<double, 3, 1> eulerAngle(_p[0], _p[1], _p[2]);
    Eigen::Matrix<double, 3, 1> t{_p[3], _p[4], _p[5]};
    Eigen::Matrix<double, 2, 1> uv_0{_p[6], _p[7]};
    Eigen::Matrix<double, 6, 1> a_;
    switch (_p.size())
    {
        case 13:
            a_ << _p[8], _p[9], _p[10], _p[11], _p[12], double(0);
            break;
        case 12:
            a_ << _p[8], _p[9], double(0), _p[10], double(0), _p[11];
            break;
        default:
            a_ << double(0), _p[8], double(0), _p[9], double(0), _p[10];
            break;
    }

    double phi, theta;
    double inv_r, r;
    double res, val;

    // extrinsic transform
    Eigen::Matrix<double, 3, 3> R;
    Eigen::AngleAxisd xAngle(Eigen::AngleAxisd(eulerAngle(0), Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(Eigen::AngleAxisd(eulerAngle(1), Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(Eigen::AngleAxisd(eulerAngle(2), Eigen::Vector3d::UnitZ()));
    R = zAngle * yAngle * xAngle;

    Eigen::Matrix<double, 3, 1> p_;
    Eigen::Matrix<double, 3, 1> p_trans;
    Eigen::Matrix<double, 2, 1> S;
    Eigen::Matrix<double, 2, 1> p_uv;

    string lidDensePcdPath = lid.scenes_files_path_vec[lid.scene_idx].dense_pcd_path;
    string HdrImgPath = cam.scenes_files_path_vec[cam.scene_idx].fisheye_hdr_img_path;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidRaw(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr showCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(lidDensePcdPath, *lidRaw);
    cv::Mat image = cv::imread(HdrImgPath, cv::IMREAD_UNCHANGED);
    int pixelThresh = 10;
    int rows = image.rows;
    int cols = image.cols;

    pcl::PointXYZRGB pt;
    vector<double> ptLoc(3, 0);
    vector<vector<vector<double>>> dict(rows, vector<vector<double>>(cols, ptLoc));

    cout << "---------------Coloring------------" << endl;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(image.at<cv::Vec3b>(i, j)[0] > pixelThresh || image.at<cv::Vec3b>(i, j)[1] > pixelThresh || image.at<cv::Vec3b>(i, j)[2] > pixelThresh){
                if(dict[i][j][0] != 0 && dict[i][j][1] != 0 && dict[i][j][2] != 0){
                    pt.x = dict[i][j][0];
                    pt.y = dict[i][j][1];
                    pt.z = dict[i][j][2];
                    pt.b = image.at<cv::Vec3b>(i, j)[0];
                    pt.g = image.at<cv::Vec3b>(i, j)[1];
                    pt.r = image.at<cv::Vec3b>(i, j)[2];
                    showCloud -> points.push_back(pt);
                }
            }
        }
    }
    pcl::visualization::CloudViewer viewer("Viewer");
    viewer.showCloud(showCloud);

    while(!viewer.wasStopped()){

    }
    cv::waitKey();
}

struct Calibration {
    template <typename T>
    bool operator()(const T *const _q, const T *const _p, T *cost) const {
        // intrinsic parameters
        Eigen::Matrix<T, 3, 1> eulerAngle(_q[0], _q[1], _q[2]);
        Eigen::Matrix<T, 3, 1> t{_q[3], _q[4], _q[5]};
        Eigen::Matrix<T, 2, 1> uv_0{_p[0], _p[1]};
        Eigen::Matrix<T, 6, 1> a_;
        switch (num_p + num_q) {
            case 13:
                a_ << _p[2], _p[3], _p[4], _p[5], _p[6], T(0);
                break;
            case 12:
                a_ << _p[2], _p[3], T(0), _p[4], T(0), _p[5];
                break;
            default:
                a_ << T(0), _p[2], T(0), _p[3], T(0), _p[4];
                break;
        }

        Eigen::Matrix<T, 3, 1> p_ = point_.cast<T>();
        Eigen::Matrix<T, 3, 1> p_trans;

        T phi, theta, r;
        T inv_r;
        T res, val;

        /** extrinsic transform for original 3d lidar edge points **/
        Eigen::Matrix<T, 3, 3> R;

        Eigen::AngleAxis<T> xAngle(Eigen::AngleAxis<T>(eulerAngle(0), Eigen::Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> yAngle(Eigen::AngleAxis<T>(eulerAngle(1), Eigen::Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> zAngle(Eigen::AngleAxis<T>(eulerAngle(2), Eigen::Matrix<T, 3, 1>::UnitZ()));
        R = zAngle * yAngle * xAngle;

        /** Construct rotation matrix using quaternion in Eigen convention (w, x, y, z) **/
        // Eigen::Quaternion<T> q{_q[3], _q[0], _q[1], _q[2]};
        // R = q.toRotationMatrix()

        p_trans = R * p_ + t;

        /** extrinsic transform for transformed 3d lidar edge points **/
        Eigen::Matrix<T, 2, 1> S;
        Eigen::Matrix<T, 2, 1> p_uv;

        /** r - theta representation: r = f(theta) in polynomial form **/
        theta = acos(p_trans(2) / sqrt((p_trans(0) * p_trans(0)) + (p_trans(1) * p_trans(1)) + (p_trans(2) * p_trans(2))));
        inv_r = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        // phi = atan2(p_trans(1), p_trans(0));
        // inv_u = (inv_r * cos(phi) + u0);
        // inv_v = (inv_r * sin(phi) + v0);

        /** compute undistorted uv coordinate of lidar projection point and evaluate the value **/
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        S = {-inv_r * p_trans(0) / r, -inv_r * p_trans(1) / r};
        p_uv = S + uv_0;
        kde_interpolator_.Evaluate(p_uv(0) * T(kde_scale_), p_uv(1) * T(kde_scale_), &val);

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
    Calibration(const Eigen::Vector3d point,
                const double weight,
                const double ref_val,
                const double scale,
                const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
            : point_(std::move(point)), kde_interpolator_(interpolator), weight_(weight), kde_val_(ref_val), kde_scale_(std::move(scale)) {}

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
    static ceres::CostFunction *Create(const Eigen::Vector3d &point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<Calibration, 2, num_q, num_p>(
                new Calibration(point, weight, kde_val, kde_scale, interpolator));
    }

    const Eigen::Vector3d point_;
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
class OutputCallback : public ceres::IterationCallback {
public:
    OutputCallback(double *params)
            : params_(params) {}

    ceres::CallbackReturnType operator()(
            const ceres::IterationSummary& summary) override {
        for (int i = 0; i < num_p + num_q; i++) {
            const double params_out = params_[i];
            outfile << params_out << "\t";
        }
        outfile << "\n";
        return ceres::SOLVER_CONTINUE;
    }

private:
    const double* params_;
};

/**
 * @brief
 * Ceres-solver Optimization
 * @param cam camProcess
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
                                     int setConstant) {
    const int num_params = params_init.size();
    const int num_scenes = cam.num_scenes;
    const double scale = 1.0;

    double params[num_params];
    memcpy(params, &params_init[0], params_init.size() * sizeof(double));
    // std::copy(std::begin(params_init), std::end(params_init), std::begin(params));

    std::vector<ceres::Grid2D<double>> grids;
    std::vector<double> ref_vals;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;

    for (int idx = 0; idx < num_scenes; idx++)
    {
        cam.SetSceneIdx(idx);
        lid.SetSceneIdx(idx);
        /********* Fisheye KDE *********/
        vector<double> p_c = cam.Kde(bandwidth, scale, false);
        // Data is a row-major array of kGridRows x kGridCols values of function
        // f(x, y) on the grid, with x in {-kGridColsHalf, ..., +kGridColsHalf},
        // and y in {-kGridRowsHalf, ..., +kGridRowsHalf}
        double *kde_data = new double[p_c.size()];
        memcpy(kde_data, &p_c[0], p_c.size() * sizeof(double));
//        double *kde_data = p_c.data();
        const ceres::Grid2D<double> kde_grid(kde_data, 0, cam.kFisheyeRows * scale, 0, cam.kFisheyeCols * scale);
        grids.push_back(kde_grid);
        ref_vals.push_back(*max_element(p_c.begin(), p_c.end()));
    }
    const std::vector<ceres::Grid2D<double>> img_grids(grids);
    for (int idx = 0; idx < num_scenes; idx++) {
        cam.SetSceneIdx(idx);
        lid.SetSceneIdx(idx);
        const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(img_grids[idx]);
        interpolators.push_back(kde_interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> img_interpolators(interpolators);

    // Ceres Problem
    // ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem problem;

    // problem.AddParameterBlock(params, 4, q_parameterization);
    // problem.AddParameterBlock(params + 4, num_params - 4);
    problem.AddParameterBlock(params, num_q);
    problem.AddParameterBlock(params + num_q, num_params - num_q);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    Eigen::Vector2d img_size = {cam.kFisheyeRows, cam.kFisheyeCols};
    for (int idx = 0; idx < num_scenes; idx++) {
        cam.SetSceneIdx(idx);
        lid.SetSceneIdx(idx);
        /** a scene weight could be added here **/
        for (int j = 0; j < lid.edge_cloud_vec[idx]->points.size(); ++j) {
            const double weight = lid.edge_cloud_vec[idx]->points[j].intensity;
            Eigen::Vector3d p_l = {lid.edge_cloud_vec[idx]->points[j].x, lid.edge_cloud_vec[idx]->points[j].y, lid.edge_cloud_vec[idx]->points[j].z};
            problem.AddResidualBlock(Calibration::Create(p_l, weight, ref_vals[idx], scale, img_interpolators[idx]),
                                     loss_function,
                                     params,
                                     params + num_q);
        }
    }

    switch (setConstant) {
        case 1:
            problem.SetParameterBlockConstant(params);
            break;
        case 2:
            problem.SetParameterBlockConstant(params + num_q);
            break;
        default:
            break;
    }

    for (int i = 0; i < num_params; ++i)
    {
        if (i < num_q && setConstant != 1)
        {
            problem.SetParameterLowerBound(params, i, lb[i]);
            problem.SetParameterUpperBound(params, i, ub[i]);
        }
        else if (i >= num_q && setConstant != 2)
        {
            problem.SetParameterLowerBound(params + num_q, i - num_q, lb[i]);
            problem.SetParameterUpperBound(params + num_q, i - num_q, ub[i]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 16;
    options.function_tolerance = 1e-7;
    options.use_nonmonotonic_steps = false;

    lid.SetSceneIdx(1);
     string paramsOutPath = lid.scenes_files_path_vec[lid.scene_idx].output_folder_path + "/ParamsRecord_" + to_string(bandwidth) + ".txt";
     outfile.open(paramsOutPath);
     OutputCallback callback(params);
     options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    customOutput(name, params, params_init);
    outfile.close();

    std::vector<double> params_res(params, params + sizeof(params) / sizeof(double));

    for (int idx = 0; idx < num_scenes; idx++) {
        cam.SetSceneIdx(idx);
        lid.SetSceneIdx(idx);
        vector<vector<double>> edge_fisheye_projection = lid.EdgeCloudProjectToFisheye(params_res);
        string edge_proj_txt_path = lid.scenes_files_path_vec[lid.scene_idx].edge_fisheye_projection_path;
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
//     string lidEdgeTransTxtPath = lid.scenes_files_path_vec[lid.scene_idx].edge_fisheye_projection_path;
//     fusionViz(cam, lidEdgeTransTxtPath, lidProjection, bandwidth);

//     return params_res;
// }
