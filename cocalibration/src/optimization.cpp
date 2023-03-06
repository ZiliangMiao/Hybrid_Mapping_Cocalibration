/** headings **/
#include <optimization.h>
#include <common_lib.h>

ofstream outfile;

struct QuaternionFunctor {
    template <typename T>
    bool operator()(const T *const q_, const T *const t_, const T *const intrinsic_, T *cost) const {
        Eigen::Quaternion<T> q{q_[3], q_[0], q_[1], q_[2]};
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> t(t_);
        Eigen::Matrix<T, K_INT, 1> intrinsic(intrinsic_);
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

    QuaternionFunctor(const Vec3D lid_point,
                    const double weight,
                    const double ref_val,
                    const double scale,
                    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator)
                    : lid_point_(std::move(lid_point)), kde_interpolator_(interpolator), weight_(std::move(weight)), kde_val_(std::move(ref_val)), kde_scale_(std::move(scale)) {}

    static ceres::CostFunction *Create(const Vec3D &lid_point,
                                       const double &weight,
                                       const double &kde_val,
                                       const double &kde_scale,
                                       const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &interpolator) {
        return new ceres::AutoDiffCostFunction<QuaternionFunctor, 3, ((6+1)-3), 3, K_INT>(
                new QuaternionFunctor(lid_point, weight, kde_val, kde_scale, interpolator));
    }

    const Vec3D lid_point_;
    const double weight_;
    const double kde_val_;
    const double kde_scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> &kde_interpolator_;
};

void project2Image(OmniProcess &omnicam, LidarProcess &lidar, std::vector<double> &params, double bandwidth) {
    cv::Mat raw_image = omnicam.loadImage();
    ofstream outfile;

    if (MESSAGE_EN) {
        /** save the projected edge points to .txt file **/
        string edge_proj_txt_path = lidar.file_path_vec[lidar.spot_idx][lidar.view_idx].edge_fisheye_projection_path;
        outfile.open(edge_proj_txt_path, ios::out);
    }
    
    Ext_D extrinsic = Eigen::Map<Param_D>(params.data()).head(6);
    Int_D intrinsic = Eigen::Map<Param_D>(params.data()).tail(K_INT);
    
    EdgeCloud::Ptr fisheye_edge_cloud (new EdgeCloud);
    EdgeCloud::Ptr lidar_edge_cloud (new EdgeCloud);
    Mat4D T_mat = transformMat(extrinsic);
    pcl::transformPointCloud(lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx], *lidar_edge_cloud, T_mat);
    pcl::copyPointCloud(omnicam.edge_cloud_vec[lidar.spot_idx][lidar.view_idx], *fisheye_edge_cloud);

    Vec3D lidar_point;
    Vec2D projection;

    for (auto &point : lidar_edge_cloud->points) {
        lidar_point << point.x, point.y, point.z;
        projection = IntrinsicTransform(intrinsic, lidar_point);
        int u = std::clamp((int)round(projection(0)), 0, raw_image.rows - 1);
        int v = std::clamp((int)round(projection(1)), 0, raw_image.cols - 1);
        point.x = projection(0);
        point.y = projection(1);
        point.z = 0;

        float radius = pow(u-intrinsic(0),2) + pow(v-intrinsic(1),2);
        Pair &bounds = omnicam.kEffectiveRadius;
        if (radius > bounds.first * bounds.first && radius < bounds.second * bounds.second) {
            raw_image.at<cv::Vec3b>(u, v)[0] = 0;    // b
            raw_image.at<cv::Vec3b>(u, v)[1] = 255;    // g
            raw_image.at<cv::Vec3b>(u, v)[2] = 0;  // r
            if (MESSAGE_EN) {outfile << u << "," << v << endl; }
        }
    }
    
    if (MESSAGE_EN) {outfile.close(); }

    lidar.getEdgeDistance(fisheye_edge_cloud, lidar_edge_cloud, 30);

    /** generate fusion image **/
    string fusion_img_path = omnicam.file_path_vec[omnicam.spot_idx][omnicam.view_idx].fusion_folder_path 
                            + "/spot_" + to_string(omnicam.spot_idx) + "_fusion_bw_" + to_string(int(bandwidth)) + ".bmp";
    cv::imwrite(fusion_img_path, raw_image); /** fusion image generation **/
}

void SpotColorization(OmniProcess &omnicam, LidarProcess &lidar, std::vector<double> &params) {
 
    string spot_cloud_path, pose_mat_path;
    Ext_F extrinsic;
    Int_F intrinsic;

    cv::Mat target_view_img;
    CloudI::Ptr spot_cloud(new CloudI);
    CloudRGB::Ptr input_cloud(new CloudRGB), spot_rgb_cloud(new CloudRGB);

    Mat4F T_mat, T_mat_inv;
    Mat4F pose_mat, pose_mat_inv;
    
    Vec3F lidar_point;
    Vec2F projection;

    spot_cloud_path = lidar.file_path_vec[lidar.spot_idx][lidar.view_idx].spot_cloud_path;

    pcl::io::loadPCDFile(spot_cloud_path, *spot_cloud);
    pcl::copyPointCloud(*spot_cloud, *input_cloud);

    /** Loading optimized parameters and initial transform matrix **/
    extrinsic = Eigen::Map<Param_D>(params.data()).head(6).cast<float>();
    intrinsic = Eigen::Map<Param_D>(params.data()).tail(K_INT).cast<float>();
    T_mat = transformMat(extrinsic);
    T_mat_inv = T_mat.inverse();
    pose_mat = Mat4F::Identity();
    pose_mat_inv = Mat4F::Identity();

    pcl::transformPointCloud(*input_cloud, *input_cloud, T_mat);

    for (int i = 0; i < lidar.num_views; i++)
    {
        int color_view_idx = lidar.center_view_idx - (int(0.5 * (i + 1)) * ((2 * (i % 2) - 1)));
        CloudRGB::Ptr output_cloud(new CloudRGB);
        std::vector<int> colored_point_idx(spot_cloud->points.size());
        std::vector<int> blank_point_idx(spot_cloud->points.size());

        /** Loading transform matrix between different views **/
        pose_mat_path = lidar.file_path_vec[lidar.spot_idx][color_view_idx].pose_trans_mat_path;
        pose_mat = LoadTransMat(pose_mat_path);
        cout << "View: " << " Spot Index: " << lidar.spot_idx << " View Index: " << color_view_idx << "\n"
            << "ICP Trans Mat:" << "\n " << pose_mat << endl;
        pose_mat_inv = pose_mat.inverse();

        /** Loading transform matrix between different views **/
        omnicam.setView(color_view_idx);
        target_view_img = omnicam.loadImage();

        /** PointCloud Coloring **/
        pcl::transformPointCloud(*input_cloud, *input_cloud, (T_mat * pose_mat_inv * T_mat_inv)); 

        /** Multiprocessing test **/
        #pragma omp parallel for num_threads(THREADS)

        for (int point_idx = 0; point_idx < input_cloud->points.size(); ++point_idx) {
            PointRGB &point = input_cloud->points[point_idx];
            if (point.x == 0 && point.y == 0 && point.z == 0) {
                continue;
            }
            lidar_point << point.x, point.y, point.z;
            projection = IntrinsicTransform(intrinsic, lidar_point);
            int u = round(projection(0));
            int v = round(projection(1));

            if (0 <= u && u < target_view_img.rows && 0 <= v && v < target_view_img.cols) {
                double radius = sqrt(pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(0), 2));
                Pair &bounds = omnicam.kEffectiveRadius;
                int &extra = omnicam.kExcludeRadius;
                if (radius > (bounds.first + extra) && radius < (bounds.second - extra)) {
                    point.b = target_view_img.at<cv::Vec3b>(u, v)[0];
                    point.g = target_view_img.at<cv::Vec3b>(u, v)[1];
                    point.r = target_view_img.at<cv::Vec3b>(u, v)[2];
                    colored_point_idx[point_idx] = point_idx;
                }
                else {
                    blank_point_idx[point_idx] = point_idx;
                }
            }
            else {
                blank_point_idx[point_idx] = point_idx;
            }
        }
        colored_point_idx.erase(std::remove(colored_point_idx.begin(), colored_point_idx.end(), 0), colored_point_idx.end());
        blank_point_idx.erase(std::remove(blank_point_idx.begin(), blank_point_idx.end(), 0), blank_point_idx.end());
        pcl::transformPointCloud(*input_cloud, *input_cloud, (T_mat * pose_mat * T_mat_inv));
        pcl::copyPointCloud(*input_cloud, colored_point_idx, *output_cloud);
	    pcl::copyPointCloud(*input_cloud, blank_point_idx, *input_cloud);
        
        *spot_rgb_cloud += *output_cloud;
        cout << input_cloud->points.size() << " " << spot_rgb_cloud->points.size() << " " << spot_cloud->points.size() << endl;
    }

    pcl::transformPointCloud(*spot_rgb_cloud, *spot_rgb_cloud, T_mat_inv);

    pcl::io::savePCDFileBinary(lidar.file_path_vec[lidar.spot_idx][lidar.center_view_idx].spot_rgb_cloud_path, *spot_rgb_cloud);
}

std::vector<double> QuaternionCalib(OmniProcess &omnicam,
                                    LidarProcess &lidar,
                                    double bandwidth,
                                    std::vector<int> spot_vec,
                                    std::vector<double> init_params_vec,
                                    std::vector<double> lb,
                                    std::vector<double> ub,
                                    bool lock_intrinsic) {
    Param_D init_params = Eigen::Map<Param_D>(init_params_vec.data());
    Ext_D extrinsic = init_params.head(6);
    MatD(K_INT+(6+1), 1) q_vector;
    Mat3D rotation_mat = transformMat(extrinsic).topLeftCorner(3, 3);
    Eigen::Quaterniond quaternion(rotation_mat);
    ceres::EigenQuaternionManifold *q_manifold = new ceres::EigenQuaternionManifold();
    
    const int kParams = q_vector.size();
    const int kViews = omnicam.num_views;
    const double scale = KDE_SCALE;
    q_vector.tail(K_INT + 3) = init_params.tail(K_INT + 3);
    q_vector.head(4) << quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w();
    double params[kParams];
    memcpy(params, &q_vector(0), q_vector.size() * sizeof(double));

    /********* Initialize Ceres Problem *********/
    ceres::Problem problem;
    problem.AddParameterBlock(params, ((6+1)-3), q_manifold);
    problem.AddParameterBlock(params+((6+1)-3), 3);
    problem.AddParameterBlock(params+(6+1), K_INT);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05);

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    omnicam.setView(omnicam.fullview_idx);
    lidar.setView(lidar.center_view_idx);
    for (int idx = 0; idx < spot_vec.size(); idx++) {
        omnicam.setSpot(spot_vec[idx]);
        std::vector<double> fisheye_kde = omnicam.Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_kde.size()];
        memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, omnicam.kImageSize.first * scale, 0, omnicam.kImageSize.second * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int idx = 0; idx < spot_vec.size(); idx++) {
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[idx]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    for (int idx = 0; idx < spot_vec.size(); idx++) {
        lidar.setSpot(spot_vec[idx]);
        EdgeCloud &edge_cloud = lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx];
        double weight = sqrt(50000.0f / edge_cloud.size());
        
        for (auto &point : edge_cloud.points) {
            Vec3D lid_point = {point.x, point.y, point.z};
            problem.AddResidualBlock(QuaternionFunctor::Create(lid_point, weight, ref_vals[idx], scale, kde_interpolators[idx]),
                                loss_function,
                                params, params+((6+1)-3), params+(6+1));
        }
    }

    if (lock_intrinsic) {
        problem.SetParameterBlockConstant(params + (6+1));
    }

    for (int i = 0; i < kParams; ++i) {
        if (i < ((6+1)-3)) {
            problem.SetParameterLowerBound(params, i, (q_vector[i]-Q_LIM));
            problem.SetParameterUpperBound(params, i, (q_vector[i]+Q_LIM));
        }
        if (i >= ((6+1)-3) && i < (6+1)) {
            problem.SetParameterLowerBound(params+((6+1)-3), i-((6+1)-3), lb[i-1]);
            problem.SetParameterUpperBound(params+((6+1)-3), i-((6+1)-3), ub[i-1]);
        }
        else if (i >= (6+1) && !lock_intrinsic) {
            problem.SetParameterLowerBound(params+(6+1), i-(6+1), lb[i-1]);
            problem.SetParameterUpperBound(params+(6+1), i-(6+1), ub[i-1]);
        }
    }

    /********* Initial Options *********/
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = MESSAGE_EN;
    options.num_threads = std::thread::hardware_concurrency();
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-6;
    options.function_tolerance = 1e-12;
    options.use_nonmonotonic_steps = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    /********* 2D Image Visualization *********/
    Param_D result = Eigen::Map<MatD(K_INT+(6+1), 1)>(params).tail(6 + K_INT);
    result.head(3) = Eigen::Quaterniond(params[3], params[0], params[1], params[2]).matrix().eulerAngles(2,1,0).reverse();
    std::vector<double> result_vec(&result[0], result.data()+result.cols()*result.rows());
    string record_path = lidar.file_path_vec[lidar.spot_idx][lidar.view_idx].result_folder_path 
                        + "/result_spot" + to_string(lidar.spot_idx) + ".txt";
    saveResults(record_path, result_vec, bandwidth, summary.initial_cost, summary.final_cost);

    extrinsic = result.head(6);
    for (int &spot_idx : spot_vec) {
        omnicam.setSpot(spot_idx);
        lidar.setSpot(spot_idx);
        project2Image(omnicam, lidar, result_vec, bandwidth);
    }
    
    return result_vec;
}

void costAnalysis(OmniProcess &omnicam,
                  LidarProcess &lidar,
                   std::vector<int> spot_vec,
                  std::vector<double> init_params_vec,
                  std::vector<double> result_vec,
                  double bandwidth) {
    const int kViews = omnicam.num_views;
    const double scale = KDE_SCALE;

    /********* Fisheye KDE *********/
    std::vector<double> ref_vals;
    std::vector<ceres::Grid2D<double>> grids;
    std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> interpolators;
    for (int i = 0; i < spot_vec.size(); i++) {
        omnicam.setSpot(spot_vec[i]);
        lidar.setSpot(spot_vec[i]);
        std::vector<double> fisheye_kde = omnicam.Kde(bandwidth, scale);
        double *kde_val = new double[fisheye_kde.size()];
        memcpy(kde_val, &fisheye_kde[0], fisheye_kde.size() * sizeof(double));
        ceres::Grid2D<double> grid(kde_val, 0, omnicam.kImageSize.first * scale, 0, omnicam.kImageSize.second * scale);
        grids.push_back(grid);
        double ref_val = *max_element(fisheye_kde.begin(), fisheye_kde.end());
        ref_vals.push_back(ref_val);
    }
    const std::vector<ceres::Grid2D<double>> kde_grids(grids);
    for (int i = 0; i < spot_vec.size(); i++) {
        omnicam.setSpot(spot_vec[i]);
        lidar.setSpot(spot_vec[i]);
        ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator(kde_grids[i]);
        interpolators.push_back(interpolator);
    }
    const std::vector<ceres::BiCubicInterpolator<ceres::Grid2D<double>>> kde_interpolators(interpolators);

    /***** Correlation Analysis *****/
    Param_D params_mat = Eigen::Map<Param_D>(result_vec.data());
    Ext_D extrinsic;
    Int_D intrinsic;
    std::vector<double> results;
    std::vector<double> input_x;
    std::vector<double> input_y;
    std::vector<const char*> name = {
            "rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"};
    int steps[3] = {1, 1, 1};
    int param_idx[3] = {0, 3, 6};
    const double step_size[3] = {0.0002, 0.001, 0.01};
    const double deg2rad = M_PI / 180;
    double offset[3] = {0, 0, 0};

    /** update evaluate points in 2D grid **/
    for (int m = 0; m < 6; m++) {
        extrinsic = params_mat.head(6);
        intrinsic = params_mat.tail(K_INT);
        if (m < 3) {
            steps[0] = 201;
            steps[1] = 1;
            steps[2] = 1;
            param_idx[0] = m;
            param_idx[1] = 3;
            param_idx[2] = 6;
        }
        else if (m < 6){
            steps[0] = 1;
            steps[1] = 201;
            steps[2] = 1;
            param_idx[0] = 0;
            param_idx[1] = m;
            param_idx[2] = 6;
        }
        else {
            steps[0] = 1;
            steps[1] = 1;
            steps[2] = 201;
            param_idx[0] = 0;
            param_idx[1] = 3;
            param_idx[2] = m;
        }

        for (int k = 0; k < spot_vec.size(); k++) {
            lidar.setSpot(spot_vec[k]);
            double normalize_weight = sqrt(1.0f / lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx].size());

            /** Save & terminal output **/
            string analysis_filepath = lidar.kDatasetPath + "/log/";
            if (steps[0] > 1) {
                analysis_filepath = analysis_filepath + name[param_idx[0]] + "_";
            }
            if (steps[1] > 1) {
                analysis_filepath = analysis_filepath + name[param_idx[1]] + "_";
            }
            if (steps[2] > 1) {
                analysis_filepath = analysis_filepath + name[param_idx[2]] + "_";
            }
            outfile.open(analysis_filepath + "spot_" + to_string(lidar.spot_idx) + "_bw_" + to_string(int(bandwidth)) + "_result.txt", ios::out);
            if (steps[0] > 1) {
                outfile << init_params_vec[param_idx[0]] << "\t" << result_vec[param_idx[0]] << endl;
            }
            if (steps[1] > 1) {
                outfile << init_params_vec[param_idx[1]] << "\t" << result_vec[param_idx[1]] << endl;
            }
            if (steps[2] > 1) {
                outfile << init_params_vec[param_idx[2]] << "\t" << result_vec[param_idx[2]] << endl;
            }
            
            
            for (int i = -int((steps[0]-1)/2); i < int((steps[0]-1)/2)+1; i++) {
                offset[0] = i * step_size[0];
                extrinsic(param_idx[0]) = params_mat(param_idx[0]) + offset[0];
                
                for (int j = -int((steps[1]-1)/2); j < int((steps[1]-1)/2)+1; j++) {
                    offset[1] = j * step_size[1];
                    extrinsic(param_idx[1]) = params_mat(param_idx[1]) + offset[1];

                    for (int n = -int((steps[2]-1)/2); n < int((steps[2]-1)/2)+1; n++) {
                        offset[2] = n * step_size[2];
                        intrinsic(param_idx[2]-6) = params_mat(param_idx[2]) + offset[2];
                    
                        double step_res = 0;
                        int num_valid = 0;
                        /** Evaluate cost funstion **/
                        for (auto &point : lidar.edge_cloud_vec[lidar.spot_idx][lidar.view_idx].points) {
                            double val;
                            double weight = normalize_weight;
                            Eigen::Vector4d lidar_point4 = {point.x, point.y, point.z, 1.0};
                            Mat4D T_mat = transformMat(extrinsic);
                            Vec3D lidar_point = (T_mat * lidar_point4).head(3);
                            Vec2D projection = IntrinsicTransform(intrinsic, lidar_point);
                            kde_interpolators[k].Evaluate(projection(0) * scale, projection(1) * scale, &val);
                            Pair &bounds = omnicam.kEffectiveRadius;
                            if ((pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(1), 2)) > pow(bounds.first, 2)
                             && (pow(projection(0) - intrinsic(0), 2) + pow(projection(1) - intrinsic(1), 2)) < pow(bounds.second, 2)) {
                                step_res += pow(weight * val, 2);
                            }
                        }
                        if (steps[0] > 1) {
                            outfile << offset[0] + params_mat(param_idx[0]) << "\t";
                        }
                        if (steps[1] > 1) {
                            outfile << offset[1] + params_mat(param_idx[1]) << "\t";
                        }
                        if (steps[2] > 1) {
                            outfile << offset[2] + params_mat(param_idx[2]) << "\t";
                        }
                        outfile << step_res << endl;
                    }
                }
            }
            outfile.close();
        }
    }
}
