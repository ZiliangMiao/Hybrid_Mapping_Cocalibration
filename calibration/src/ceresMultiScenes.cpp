// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
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
#include <pcl/filters/voxel_grid.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
// headings
#include "FisheyeProcess.h"
#include "LidarProcess.h"
#include "utils.h"

ofstream outfile;

static const int kExtrinsics = 7;
static const int kIntrinsics = 10;

inline double getDouble(double x) {
    return static_cast<double>(x);
}

template <typename SCALAR, int N>
inline double getDouble(const ceres::Jet<SCALAR, N> &x) {
    return static_cast<double>(x.a);
}

struct GradientFunctor {
    template <typename T>
    bool operator()(const T *const p_, T *cost) const {
        Eigen::Matrix<T, 6, 1> extrinsic_;
        Eigen::Matrix<T, kIntrinsics, 1> intrinsic_;
        extrinsic_ << p_[0], p_[1], p_[2], p_[3], p_[4], p_[5];
        intrinsic_ << p_[6], p_[7], p_[8], p_[9], p_[10],
                    p_[11], p_[12], p_[13], p_[14], p_[15]; 
        Eigen::Matrix<T, 4, 4> T_mat = ExtrinsicMat(extrinsic_);

        Eigen::Matrix<T, 4, 1> lidar_point;
        Eigen::Matrix<T, 3, 1> lidar_trans;
        Eigen::Matrix<T, 2, 1> projection;
        T res = T(0), val = T(0);
        
        for (auto &point : lidar_cloud_.points) {
            lidar_point << T(point.x), T(point.y), T(point.z), T(1);
            lidar_trans = (T_mat * lidar_point).head(3);
            Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic_, lidar_trans);
            kde_interpolator_.Evaluate(projection(0) * T(scale_), projection(1) * T(scale_), &val);
            res += T(point.intensity) * T(val);
        }
        cost[0] = -res;
        return true;
    }

    GradientFunctor(const CloudT lidar_cloud,
                    const double scale,
                    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
                    : lidar_cloud_(std::move(lidar_cloud)), kde_interpolator_(interpolator), scale_(std::move(scale)) {}

    static ceres::FirstOrderFunction *Create(const CloudT &lidar_cloud,
                                            const double &scale,
                                            const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffFirstOrderFunction<GradientFunctor, 6 + kIntrinsics>(
                new GradientFunctor(lidar_cloud, scale, interpolator));
    }

    const CloudT lidar_cloud_;
    const double scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
};

struct QuaternionFunctor {
    template <typename T>
    bool operator()(const T *const q_, const T *const t_, const T *const intrinsic_, T *cost) const {
        Eigen::Quaternion<T> q{q_[3], q_[0], q_[1], q_[2]};
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t(t_);
        Eigen::Matrix<T, kIntrinsics, 1> intrinsic(intrinsic_);
        Eigen::Matrix<T, 3, 1> lidar_point = R * lid_point_.cast<T>() + t;
        Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic, lidar_point);
        T res, val;
        kde_interpolator_.Evaluate(projection(0) * T(kde_scale_), projection(1) * T(kde_scale_), &val);
        res = T(weight_) * (T(kde_val_) - val);
        cost[0] = res;
        cost[1] = res;
        cost[2] = res;
        return true;
    }

    QuaternionFunctor(const Eigen::Vector3d lid_point,
                        const double weight,
                        const double ref_val,
                        const double scale,
                        const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
                        : lid_point_(std::move(lid_point)), kde_interpolator_(interpolator), weight_(std::move(weight)), kde_val_(std::move(ref_val)), kde_scale_(std::move(scale)) {}

    static ceres::CostFunction *Create(const Eigen::Vector3d &lid_point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<QuaternionFunctor, 3, 4, 3, kIntrinsics>(
                new QuaternionFunctor(lid_point, weight, kde_val, kde_scale, interpolator));
    }

    const Eigen::Vector3d lid_point_;
    const double weight_;
    const double kde_val_;
    const double kde_scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
};

void Visualization2D(FisheyeProcess &fisheye, LidarProcess &lidar, std::vector<double> &params, double bandwidth) {
    string fisheye_hdr_img_path = fisheye.poses_files_path_vec[fisheye.spot_idx][fisheye.view_idx].fisheye_hdr_img_path;
    cv::Mat raw_image = fisheye.ReadFisheyeImage(fisheye_hdr_img_path);

    /** write the edge points projected on fisheye to .txt file **/
    ofstream outfile;
    string edge_proj_txt_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].edge_fisheye_projection_path;
    outfile.open(edge_proj_txt_path, ios::out);
    
    Eigen::Matrix<double, 6, 1> extrinsic = Eigen::Map<Eigen::Matrix<double, 6 + kIntrinsics, 1>>(params.data()).head(6);
    Eigen::Matrix<double, kIntrinsics, 1> intrinsic = Eigen::Map<Eigen::Matrix<double, 6 + kIntrinsics, 1>>(params.data()).tail(kIntrinsics);
    
    CloudPtr edge_cloud = lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx];
    CloudPtr edge_trans_cloud(new CloudT);
    Eigen::Matrix<double, 4, 4> T_mat = ExtrinsicMat(extrinsic);
    pcl::transformPointCloud(*edge_cloud, *edge_trans_cloud, T_mat);

    Eigen::Vector3d lidar_point;
    Eigen::Vector2d projection;
    RGBPointT rgb_pt;

    for (auto &point : edge_trans_cloud->points) {
        lidar_point << point.x, point.y, point.z;
        projection = IntrinsicTransform(intrinsic, lidar_point);
        int u = std::clamp((int)round(projection(0)), 0, raw_image.rows - 1);
        int v = std::clamp((int)round(projection(1)), 0, raw_image.cols - 1);
        raw_image.at<cv::Vec3b>(u, v)[0] = 0;    // b
        raw_image.at<cv::Vec3b>(u, v)[1] = 0;    // g
        raw_image.at<cv::Vec3b>(u, v)[2] = 255;  // r
        outfile << u << "," << v << endl;
    }
    
    outfile.close();

    // /***** Visualization *****/
    // CloudPtr fullview_cloud(new CloudT);
    // RGBCloudPtr fullview_rgb_cloud(new RGBCloudT);
    // string fullview_cloud_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].fullview_sparse_cloud_path;
    // pcl::io::loadPCDFile(fullview_cloud_path, *fullview_cloud);
    // pcl::transformPointCloud(*fullview_cloud, *fullview_cloud, T_mat);

    // for (auto &point : fullview_cloud->points) {
    //     lidar_point << point.x, point.y, point.z;
    //     projection = IntrinsicTransform(intrinsic, lidar_point);
    //     rgb_pt.x = projection(0);
    //     rgb_pt.y = projection(1);
    //     rgb_pt.z = 0;
    //     int L_2 = 50;
    //     int indicator = point.intensity;
    //     if ((indicator) < L_2) {
    //         rgb_pt.r = 0;
    //         rgb_pt.g = int(255 * ((float)(int(indicator * 0.5) % L_2) / L_2));
    //         rgb_pt.b = int(255 * ((float)1 - ((float)(int(indicator * 0.5) % L_2) / L_2)));
    //     }
    //     else {
    //         rgb_pt.r = int(255 * (float)((int(indicator * 0.5) % L_2) - L_2) / L_2);
    //         rgb_pt.g = int(255 * ((float)1 - (float)((int(indicator * 0.5) % L_2) - L_2) / L_2));
    //         rgb_pt.b = 0;
    //     }
    //     fullview_rgb_cloud->points.push_back(rgb_pt);
    // }

    // pcl::visualization::PCLVisualizer viewer("Reconstruction");
    // int v1(0); /** create two view point **/
    // viewer.createViewPort(0.0, 0.0, 1.0, 1.0, v1);
    // float bkg_grayscale = 150;  /** black **/

    // int point_size = 3;
    // viewer.addPointCloud(fullview_rgb_cloud, "fullview_rgb_cloud", v1);
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, point_size, "fullview_rgb_cloud");
    // viewer.addCoordinateSystem();
    // viewer.setBackgroundColor(bkg_grayscale, bkg_grayscale, bkg_grayscale, v1);
    // viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    // viewer.setSize(1280, 1024);  /** viewer size **/  

    // while (!viewer.wasStopped()) {
    //     viewer.spinOnce();
    // }

    /** generate fusion image **/
    tk::spline poly_spline = InverseSpline(params);

    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fisheyeResult =
        fisheye.FisheyeImageToSphere(raw_image, true, poly_spline);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fisheyeOrgPolarCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fisheyeOrgPixelCloud;
    std::tie(fisheyeOrgPolarCloud, fisheyeOrgPixelCloud) = fisheyeResult;
    fisheye.SphereToPlane(fisheyeOrgPolarCloud, bandwidth);
}

void Visualization3D(FisheyeProcess &fisheye, LidarProcess &lidar, std::vector<double> &params) {
 
    string fullview_cloud_path, pose_mat_path, fisheye_img_path;
    Eigen::Matrix<float, 6, 1> extrinsic;
    Eigen::Matrix<float, kIntrinsics, 1> intrinsic;

    cv::Mat target_view_img;
    CloudPtr fullview_xyz_cloud(new CloudT);
    RGBCloudPtr input_cloud(new RGBCloudT), fullview_rgb_cloud(new RGBCloudT);

    Eigen::Matrix4f T_mat, T_mat_inv;
    Eigen::Matrix4f pose_mat, pose_mat_inv;
    
    Eigen::Vector3f lidar_point;
    Eigen::Vector2f projection;

    fullview_cloud_path = lidar.poses_files_path_vec[lidar.spot_idx][lidar.view_idx].fullview_dense_cloud_path;

    pcl::io::loadPCDFile(fullview_cloud_path, *fullview_xyz_cloud);
    pcl::VoxelGrid<pcl::PointXYZI> voxelgrid;
    voxelgrid.setInputCloud(fullview_xyz_cloud);
    voxelgrid.setLeafSize(0.03f, 0.03f, 0.03f);
    voxelgrid.filter(*fullview_xyz_cloud);
    pcl::copyPointCloud(*fullview_xyz_cloud, *input_cloud);

    /** Loading optimized parameters and initial transform matrix **/
    extrinsic = Eigen::Map<Eigen::Matrix<double, 6 + kIntrinsics, 1>>(params.data()).head(6).cast<float>();
    intrinsic = Eigen::Map<Eigen::Matrix<double, 6 + kIntrinsics, 1>>(params.data()).tail(kIntrinsics).cast<float>();
    T_mat = ExtrinsicMat(extrinsic);
    T_mat_inv = T_mat.inverse();
    pose_mat = Eigen::Matrix4f::Identity();
    pose_mat_inv = Eigen::Matrix4f::Identity();

    pcl::transformPointCloud(*input_cloud, *input_cloud, T_mat);

    for (int i = 0; i < lidar.num_views; i++)
    {
        int fullview_idx = lidar.fullview_idx - (int(0.5 * (i + 1)) * ((2 * (i % 2) - 1)));
        RGBCloudPtr output_cloud(new RGBCloudT);
        std::vector<int> colored_point_idx;
        std::vector<int> remaining_point_idx;

        /** Loading transform matrix between different views **/
        pose_mat_path = lidar.poses_files_path_vec[lidar.spot_idx][fullview_idx].pose_trans_mat_path;
        pose_mat = LoadTransMat(pose_mat_path);
        cout << "View: " << " Spot Index: " << lidar.spot_idx << " View Index: " << fullview_idx << "\n"
            << "ICP Trans Mat:" << "\n " << pose_mat << endl;
        pose_mat_inv = pose_mat.inverse();

        /** Loading transform matrix between different views **/
        fisheye_img_path = fisheye.poses_files_path_vec[fisheye.spot_idx][fullview_idx].fisheye_hdr_img_path;
        target_view_img = fisheye.ReadFisheyeImage(fisheye_img_path);

        /** PointCloud Coloring **/
        pcl::transformPointCloud(*input_cloud, *input_cloud, (T_mat * pose_mat_inv * T_mat_inv)); 

        for (int point_idx = 0; point_idx < input_cloud->points.size(); ++point_idx) {
            RGBPointT &point = input_cloud->points[point_idx];
            if (point.x == 0 && point.y == 0 && point.z == 0) {
                continue;
            }
            lidar_point << point.x, point.y, point.z;
            projection = IntrinsicTransform(intrinsic, lidar_point);
            int u = round(projection(0));
            int v = round(projection(1));

            if (0 <= u && u < target_view_img.rows && 0 <= v && v < target_view_img.cols) {
                double radius = sqrt(pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(0), 2));
                /** Initial guess of center: (1026, 1200) **/
                /** Initial guess of radius limits: 330 and 1070  **/
                /** Modify the limits below to remove black region **/
                if (radius > 500 && radius < 950) {
                    point.b = target_view_img.at<cv::Vec3b>(u, v)[0];
                    point.g = target_view_img.at<cv::Vec3b>(u, v)[1];
                    point.r = target_view_img.at<cv::Vec3b>(u, v)[2];
                    colored_point_idx.push_back(point_idx);
                }
                else {
                    remaining_point_idx.push_back(point_idx);
                }
            }
            else {
                remaining_point_idx.push_back(point_idx);
            }
        }
        pcl::copyPointCloud(*input_cloud, colored_point_idx, *output_cloud);
        pcl::transformPointCloud(*output_cloud, *output_cloud, (T_mat * pose_mat * T_mat_inv));
	    pcl::copyPointCloud(*input_cloud, remaining_point_idx, *input_cloud);
        pcl::transformPointCloud(*input_cloud, *input_cloud, (T_mat * pose_mat * T_mat_inv));
        *fullview_rgb_cloud += *output_cloud;
        cout << input_cloud->points.size() << " " << fullview_rgb_cloud->points.size() << " " << fullview_xyz_cloud->points.size() << endl;
    }

    pcl::transformPointCloud(*fullview_rgb_cloud, *fullview_rgb_cloud, T_mat_inv);

    pcl::io::savePCDFileBinary(lidar.poses_files_path_vec[lidar.spot_idx][lidar.fullview_idx].fullview_rgb_cloud_path, *fullview_rgb_cloud);
}

std::vector<double> GradientCalib(FisheyeProcess &fisheye,
                                LidarProcess &lidar,
                                double bandwidth,
                                std::vector<double> params_init) {
    const int kParams = params_init.size();
    const int kViews = fisheye.num_views;
    const double scale = pow(2, (-(int)floor(log(bandwidth) / log(4))));
    double params[kParams];
    memcpy(params, &params_init[0], params_init.size() * sizeof(double));

    /********* Fisheye KDE *********/
    vector<double> p_c = fisheye.Kde(bandwidth, scale);
    double *kde_val = new double[p_c.size()];
    memcpy(kde_val, &p_c[0], p_c.size() * sizeof(double));
    
    double ref_val = *max_element(p_c.begin(), p_c.end());
    const ceres::Grid2D<double> kde_grid(kde_val, 0, fisheye.kFisheyeRows * scale, 0, fisheye.kFisheyeCols * scale);
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(kde_grid);

    /********* Initialize Ceres Problem *********/
    CloudPtr &edge_cloud = lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx];
    ceres::GradientProblem problem(GradientFunctor::Create((*edge_cloud), scale, kde_interpolator));

    /********* Initial Options *********/
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-6;

    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, params, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    std::vector<double> result_vec(params, params + sizeof(params) / sizeof(double));
    CeresOutput(result_vec, params_init);
    Visualization2D(fisheye, lidar, result_vec, bandwidth);
    return result_vec;
}

std::vector<double> QuaternionCalib(FisheyeProcess &fisheye,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<int> spot_vec,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    int kDisabledBlock) {
    Eigen::Matrix<double, 6 + kIntrinsics, 1> init_params = Eigen::Map<Eigen::Matrix<double, 6 + kIntrinsics, 1>>(init_params_vec.data());
    Eigen::Matrix<double, 6, 1> extrinsic = init_params.head(6);
    Eigen::Matrix<double, kIntrinsics + 3, 1> section = init_params.tail(kIntrinsics + 3);
    Eigen::Matrix<double, kExtrinsics + kIntrinsics, 1> q_vector;
    Eigen::Matrix3d rotation_mat = ExtrinsicMat(extrinsic).topLeftCorner(3, 3);
    cout << rotation_mat << endl;
    Eigen::Quaterniond quaternion(rotation_mat);
    ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    
    const int kParams = q_vector.size();
    const int kViews = fisheye.num_views;
    const double scale = pow(2, (-(int)floor(log(bandwidth) / log(4))));
    q_vector.tail(kIntrinsics + 3) = init_params.tail(kIntrinsics + 3);
    q_vector.head(4) << quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w();
    cout << q_vector.head(4) << endl;
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    for (int i = 0; i < spot_vec.size(); i++) {
        fisheye.SetSpotIdx(spot_vec[i]);
        lidar.SetSpotIdx(spot_vec[i]);
        std::vector<double> fisheye_edge = fisheye.Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_edge.size()];
        memcpy(kde_val, &fisheye_edge[0], fisheye_edge.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, fisheye.kFisheyeRows * scale, 0, fisheye.kFisheyeCols * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_edge.begin(), fisheye_edge.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int i = 0; i < spot_vec.size(); i++) {
        fisheye.SetSpotIdx(spot_vec[i]);
        lidar.SetSpotIdx(spot_vec[i]);
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[i]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    // vector<double> p_c = fisheye.Kde(bandwidth, scale);
    // double *kde_val = new double[p_c.size()];
    // memcpy(kde_val, &p_c[0], p_c.size() * sizeof(double));
    
    // double ref_val = *max_element(p_c.begin(), p_c.end());
    // const ceres::Grid2D<double> kde_grid(kde_val, 0, fisheye.kFisheyeRows * scale, 0, fisheye.kFisheyeCols * scale);
    // const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(kde_grid);

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    problem.AddParameterBlock(params, kExtrinsics-3, q_manifold);
    problem.AddParameterBlock(params+(kExtrinsics-3), 3);
    problem.AddParameterBlock(params+kExtrinsics, kIntrinsics);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /** a scene weight could be added here **/
    for (int idx = 0; idx < spot_vec.size(); idx++) {
        fisheye.SetSpotIdx(spot_vec[idx]);
        lidar.SetSpotIdx(spot_vec[idx]);
        double normalize_weight = 1.0 / sqrt(spot_vec.size());
        for (auto &point : lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points) {
            double weight = point.intensity * normalize_weight;
            Eigen::Vector3d lid_point = {point.x, point.y, point.z};
            problem.AddResidualBlock(QuaternionFunctor::Create(lid_point, weight, ref_vals[idx], scale, kde_interpolators[idx]),
                                    loss_function,
                                    params, params+(kExtrinsics-3), params+kExtrinsics);
            // problem.AddResidualBlock(QuaternionCalibration::Create(lid_point, weight, ref_val, scale, kde_interpolator),
            // loss_function,
            // params, params+(kExtrinsics-3), params+kExtrinsics);
        }
    }
    
    switch (kDisabledBlock) {
    case 1:
        problem.SetParameterBlockConstant(params);
        problem.SetParameterBlockConstant(params+(kExtrinsics-3));
        break;
    case 2:
        problem.SetParameterBlockConstant(params+kExtrinsics);
        break;
    default:
        break;
    }

    for (int i = 0; i < kParams; ++i) {
        if (i < 4 && kDisabledBlock != 1) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-0.15));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+0.15));
        }
        if (i >= 4 && i < kExtrinsics && kDisabledBlock != 1) {
            problem.SetParameterLowerBound(params+(kExtrinsics-3), i-(kExtrinsics-3), lb[i-1]);
            problem.SetParameterUpperBound(params+(kExtrinsics-3), i-(kExtrinsics-3), ub[i-1]);
        }
        else if (i >= kExtrinsics && kDisabledBlock != 2) {
            problem.SetParameterLowerBound(params+kExtrinsics, i-kExtrinsics, lb[i-1]);
            problem.SetParameterUpperBound(params+kExtrinsics, i-kExtrinsics, ub[i-1]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = thread::hardware_concurrency();
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-7;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Eigen::Matrix<double, 6+kIntrinsics, 1> result = Eigen::Map<Eigen::Matrix<double, kExtrinsics + kIntrinsics, 1>>(params).tail(6 + kIntrinsics);
    cout << Eigen::Quaterniond(params[3], params[0], params[1], params[2]).matrix() << endl;
    result.head(3) = Eigen::Quaterniond(params[3], params[0], params[1], params[2]).matrix().eulerAngles(2,1,0).reverse();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());
    CeresOutput(result_vec, init_params_vec);
    extrinsic = result.head(6);
    cout << ExtrinsicMat(extrinsic) << endl;
    for (int &spot_idx : spot_vec) {
        fisheye.SetSpotIdx(spot_idx);
        lidar.SetSpotIdx(spot_idx);
        Visualization2D(fisheye, lidar, result_vec, bandwidth);
    }
    
    return result_vec;
}

void CorrelationAnalysis(FisheyeProcess &fisheye,
                         LidarProcess &lidar,
                         std::vector<int> spot_vec,
                         std::vector<double> params_vec){
    const double bandwidth = 2;
    const int kViews = fisheye.num_views;
    const double scale = pow(2, (-(int)floor(log(bandwidth) / log(4))));

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    for (int i = 0; i < spot_vec.size(); i++) {
        fisheye.SetSpotIdx(spot_vec[i]);
        lidar.SetSpotIdx(spot_vec[i]);
        std::vector<double> fisheye_edge = fisheye.Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_edge.size()];
        memcpy(kde_val, &fisheye_edge[0], fisheye_edge.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, fisheye.kFisheyeRows * scale, 0, fisheye.kFisheyeCols * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_edge.begin(), fisheye_edge.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int i = 0; i < spot_vec.size(); i++) {
        fisheye.SetSpotIdx(spot_vec[i]);
        lidar.SetSpotIdx(spot_vec[i]);
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[i]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    /***** Correlation Analysis *****/
    Eigen::Matrix<double, 6+kIntrinsics, 1> params_mat = Eigen::Map<Eigen::Matrix<double, 6+kIntrinsics, 1>>(params_vec.data());
    Eigen::Matrix<double, 6, 1> extrinsic = params_mat.head(6);
    Eigen::Matrix<double, kIntrinsics, 1> intrinsic = params_mat.tail(kIntrinsics);
    std::vector<double> results;
    std::vector<double> input_x, input_y;
    std::vector<const char*> name = {
            "rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"};
    const int steps[2] = {41, 1};
    const int param_idx[2] = {1, 3};
    const double step_size[2] = {0.01, 0.015};
    const double deg2rad = M_PI / 180;
    double offset[2] = {0, 0};

    /** update evaluate points in 2D grid **/
    for (int i = -int((steps[0]-1)/2); i < int((steps[0]-1)/2)+1; i++) {
        offset[0] = i * step_size[0];
        extrinsic(param_idx[0]) = params_mat(param_idx[0]) + offset[0];

        for (int j = -int((steps[1]-1)/2); j < int((steps[1]-1)/2)+1; j++) {
            offset[1] = j * step_size[1];
            extrinsic(param_idx[1]) = params_mat(param_idx[1]) + offset[1];
            input_x.push_back(offset[0]);
            input_y.push_back(offset[1]);

            double step_res = 0;
            /** Evaluate cost funstion **/
            for (int i = 0; i < spot_vec.size(); i++) {
                lidar.SetSpotIdx(spot_vec[i]);
                double normalize_weight = (double)1 / lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points.size();
                double scene_res = 0;
                for (auto &point : lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx]->points) {
                    double val;
                    double weight = point.intensity * normalize_weight;
                    Eigen::Vector4d lidar_point4 = {point.x, point.y, point.z, 1.0};
                    Eigen::Matrix<double, 4, 4> T_mat = ExtrinsicMat(extrinsic);
                    Eigen::Matrix<double, 3, 1> lidar_point = (T_mat * lidar_point4).head(3);
                    Eigen::Matrix<double, 2, 1> projection = IntrinsicTransform(intrinsic, lidar_point);
                    kde_interpolators[i].Evaluate(projection(0) * scale, projection(1) * scale, &val);
                    scene_res += weight * val;
                }
                step_res += scene_res;
                cout << "spot: " << spot_vec[i] << ", " << name[param_idx[0]]<< ": " << offset[0] << ", " << name[param_idx[1]]<< ": " << offset[1] << endl;
            }
            results.push_back(step_res);
        }
    }

    /** Save & terminal output **/
    string analysis_filepath = lidar.kDatasetPath + "/log/";
    if (steps[0] > 1) {
        analysis_filepath = analysis_filepath + name[param_idx[0]] + "_";
    }
    if (steps[1] > 1) {
        analysis_filepath = analysis_filepath + name[param_idx[1]] + "_";
    }
    outfile.open(analysis_filepath + "result.txt", ios::out);
    for (int i = 0; i < (steps[0] * steps[1]); i++) {
        if (steps[0] > 1) {
            outfile << input_x[i] + params_mat(param_idx[0]) << "\t";
        }
        if (steps[1] > 1) {
            outfile << input_y[i] + params_mat(param_idx[1]) << "\t";
        }
        outfile << results[i] << endl;
    }
    outfile.close();
    
}
