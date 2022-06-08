// basic
#include <iostream>
#include <algorithm>
#include <vector>
// eigen
#include <Eigen/Core>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
// ceres
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
// heading
#include "imageProcess.h"
#include "lidarProcess.h"

using namespace std;

// ofstream outfile;

// convert ceres::Jet and double to double in optimization process
double get_double( double x )
{ return static_cast<double>(x); }

template<typename SCALAR, int N>
double get_double( const ceres::Jet<SCALAR, N>& x )
{ return static_cast<double>(x.a); }

void initQuaternion(double rx, double ry, double rz, vector<double> &init){
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


struct Calibration
{
    template <typename T>
    bool operator()(const T* const _q, const T* const _p, T* cost) const
    {
        // intrinsic parameters
        const T rx = _q[0];
        const T ry = _q[1];
        const T rz = _q[2];
        // const T rw = _q[0];
        // const T rx = _q[1];
        // const T ry = _q[2];
        // const T rz = _q[3];
        const T tx = _p[0];
        const T ty = _p[1];
        const T tz = _p[2];
        const T u0 = _p[3];
        const T v0 = _p[4];
        // const T a0 = _p[5];
        // const T a1 = _p[6];
        // const T a2 = _p[7];
        // const T a3 = _p[8];
        // const T a4 = _p[9];
        const T a1 = _p[5];
        const T a3 = _p[6];
        const T a5 = _p[7];

        // rigid transformation
        // Convert to Eigen convention (w, x, y, z)
        // Eigen::Quaternion<T> q{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t{_p[0], _p[1], _p[2]};
        Eigen::Matrix<T, 3, 1> p_;
        Eigen::Matrix<T, 3, 1> p_trans;

        T phi, theta, r;
        T inv_r, inv_u, inv_v;
        T eval_u, eval_v, result, tmp;

        p_ << T(point_(0)), T(point_(1)), T(point_(2));

        // extrinsic transform
        
        Eigen::Matrix<T, 3, 1> eulerAngle(rx, ry, rz);
        Eigen::Matrix<T, 3, 3> R;
        
        Eigen::AngleAxis<T> xAngle(Eigen::AngleAxis<T>(eulerAngle(0), Eigen::Matrix<T, 3, 1>::UnitX()));
        Eigen::AngleAxis<T> yAngle(Eigen::AngleAxis<T>(eulerAngle(1), Eigen::Matrix<T, 3, 1>::UnitY()));
        Eigen::AngleAxis<T> zAngle(Eigen::AngleAxis<T>(eulerAngle(2), Eigen::Matrix<T, 3, 1>::UnitZ()));
        R = zAngle * yAngle * xAngle;
        // R = q.toRotationMatrix()
        
        p_trans = R * p_ + t;

        // intrinsic inverse transform

        // phi = atan2(p_trans(1), p_trans(0));
        theta = acos(p_trans(2) / sqrt((p_trans(0)*p_trans(0)) + (p_trans(1)*p_trans(1)) + (p_trans(2)*p_trans(2))));
        
        //  r-theta representaion: 
        //  1) r = f(theta) using polynomials of 'theta'
        //  2) theta = f(r) using polynomials of 'r' and compute inverse spline mapping 
        // inv_r = a0 + a1 * theta + a2 * pow(theta, 2) + a3 * pow(theta, 3) + a4 * pow(theta, 4);
        inv_r = a1 * theta + a3 * pow(theta, 3) + a5 * pow(theta, 5);
        // inv_u = (inv_r * cos(phi) + u0);
        // inv_v = (inv_r * sin(phi) + v0);
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        inv_u = (inv_r / r * p_trans(0) + u0);
        inv_v = (inv_r / r * p_trans(1) + v0);

        kde_interpolator_.Evaluate(inv_u, inv_v, &tmp);

        // bicubic interpolation evaluation for "polar" version kde
        // kde_interpolator_.Evaluate(phi, inv_r, &tmp);

        // result = T(0.005) * (T(1.0) - inv_r * T(0.5/img_size_[1])) - tmp;

        // char o_[32];
        // sprintf(o_, "%f%s%f%s", get_double(inv_u), ",", get_double(inv_v), "\n");
        // outfile << o_;

        cost[0] = T(scale_) * (T(1.0) - inv_r * T(0.5/img_size_[1])) - tmp;
        cost[1] = T(1e-8) * (T(1071) - a1 * T(ref_theta_) - a3 * T(pow(ref_theta_, 3)) - a5* T(pow(ref_theta_, 5))) ;
        return true;
    }

    Calibration(const Eigen::Vector3d point,
                const Eigen::Vector2d img_size,
                const double scale,
                const ceres::BiCubicInterpolator<ceres::Grid2D<double>>& interpolator)
                : point_(std::move(point)), kde_interpolator_(interpolator), img_size_(img_size), scale_(scale){}

    static ceres::CostFunction* Create(const Eigen::Vector3d& point,
                                        const Eigen::Vector2d& img_size,
                                        const double& scale,
                                        const ceres::BiCubicInterpolator<ceres::Grid2D<double>>& interpolator) {
        return new ceres::AutoDiffCostFunction<Calibration, 2, 3, 8>(
            new Calibration(point, img_size, scale, interpolator));
    }

    const Eigen::Vector3d point_;
    const Eigen::Vector2d img_size_;
    const double scale_;
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>>& kde_interpolator_;
    const double ref_theta_ = 95*M_PI/180;
  
};


class OutputCallback : public ceres::IterationCallback {
    public:
    OutputCallback(string filename, double* params)
        : filename_(filename), params_(params) {}

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary) override {
        // outfile.close();
        // outfile.open(filename_);
        return ceres::SOLVER_CONTINUE;
    }

    private:
    const string filename_;
    const double* params_;
};

// invoke ceres-solver for optimization
std::vector<double> ceresAutoDiff(imageProcess cam, lidarProcess lid, double bandwidth, vector<double> params_init, vector<double> lb, vector<double> ub)
{
    std::vector<double> p_c = cam.kdeBlur(bandwidth, 1.0, false);
    const double scale = *max_element(p_c.begin(), p_c.end()) / (0.125 * bandwidth);
    const int num_params = params_init.size();
    // Quaternion in {x, y, z, w}
    // double param_init[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.14, 0.22548, 596.99269, 42.07474, -54.03895, 20.89775, 1023.0, 1201.0};
    // double dev[] = {1, 1, 1e-3, 1, 1e-3, 1e-3, 5e-4, 5e-0, 5e-0, 5e+1, 5e+0, 5e+0, 5e+0, 5e+0};
    // initQuaternion(0.0, -0.01, M_PI, param_init);
    
    double params[num_params];
    memcpy(params, &params_init[0], params_init.size()*sizeof(double)); 
    // std::copy(std::begin(params_init), std::end(params_init), std::begin(params));

    // Data is a row-major array of kGridRows x kGridCols values of function
    // f(x, y) on the grid, with x in {-kGridColsHalf, ..., +kGridColsHalf},
    // and y in {-kGridRowsHalf, ..., +kGridRowsHalf}
    double *kde_data = new double[p_c.size()];  
    memcpy(kde_data, &p_c[0], p_c.size()*sizeof(double));  

    // unable to set coordinate to 2D grid for corresponding interpolator;
    // use post-processing to scale the grid instead.
    const ceres::Grid2D<double> kde_grid(kde_data, 0, cam.kdeRows, 0, cam.kdeCols);
    const ceres::BiCubicInterpolator<ceres::Grid2D<double>> kde_interpolator(kde_grid);

    // Ceres Problem
    // ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem problem;

    // problem.AddParameterBlock(params, 4, q_parameterization);
    // problem.AddParameterBlock(params + 4, num_params - 4);
    problem.AddParameterBlock(params, 3);
    problem.AddParameterBlock(params + 3, num_params - 3);
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.05); 

    Eigen::Vector2d img_size = {cam.orgRows, cam.orgCols};
    for (int i = 0; i < lid.lidEdgeOrg->points.size(); ++i) {
        // Eigen::Vector3d p_l_tmp = p_l.row(i);
        Eigen::Vector3d p_l_tmp = {lid.lidEdgeOrg->points[i].x, lid.lidEdgeOrg->points[i].y, lid.lidEdgeOrg->points[i].z};
        problem.AddResidualBlock(Calibration::Create(p_l_tmp, img_size, scale, kde_interpolator),
                                nullptr,
                                params,
                                params + 3);
    }

    for (int i = 0; i < num_params; ++i)
    {
        if (i < 3){
            problem.SetParameterLowerBound(params, i, lb[i]);
            problem.SetParameterUpperBound(params, i, ub[i]);
        }
        else{
            problem.SetParameterLowerBound(params + 3, i - 3, lb[i]);
            problem.SetParameterUpperBound(params + 3, i - 3, ub[i]);
        }
        
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 12;
    options.function_tolerance = 1e-7;
    // options.use_nonmonotonic_steps = true;

    OutputCallback callback(lid.lidTransFile, params);
    options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    // std::cout << "Initial rw: " << param_init[0] << " rx: " << param_init[1] << " ry: " << param_init[2] << " rz: " << param_init[3]
    //         << " tx: " << param_init[4] << " ty: " << param_init[5] << " tz: " << param_init[6]
    //         << " a0: " << param_init[7] << " a1: " << param_init[8] << " a2: " << param_init[9] << " a3: " << param_init[10] << " a4: " << param_init[11]
    //         // << " c: " << param_init[11] << " d: " << param_init[12] << " e: " << param_init[13]
    //         << " u0: " << param_init[12] << " v0: " << param_init[13]
    //         << "\n";
    // std::cout << "Final   rw: " << params[0] << " rx: " << params[1] << " ry: " << params[2] << " rz: " << params[3]
    //         << " tx: " << params[4] << " ty: " << params[5] << " tz: " << params[6]
    //         << " a0: " << params[7] << " a1: " << params[8] << " a2: " << params[9] << " a3: " << params[10] << " a4: " << params[11]
    //         // << " c: " << params[11] << " d: " << params[12] << " e: " << params[13]
    //         << " u0: " << params[12] << " v0: " << params[13]
    //         << "\n";
    std::cout << "Initial rx: " << params_init[0] << " ry: " << params_init[1] << " rz: " << params_init[2]
            << " tx: " << params_init[3] << " ty: " << params_init[4] << " tz: " << params_init[5]
            << " u0: " << params_init[6] << " v0: " << params_init[7]
            // << " a0: " << param_init[8] << " a1: " << param_init[9] << " a2: " << param_init[10] << " a3: " << params[11] << " a4: " << params[12] 
            << " a1: " << params_init[8] << " a3: " << params_init[9] << " a5: " << params_init[10]
            // << " c: " << param_init[11] << " d: " << param_init[12] << " e: " << param_init[13]
            << "\n";
    std::cout << "Final   rx: " << params[0] << " ry: " << params[1] << " rz: " << params[2]
            << " tx: " << params[3] << " ty: " << params[4] << " tz: " << params[5]
            << " u0: " << params[6] << " v0: " << params[7]
            // << " a0: " << params[8] << " a1: " << params[9] << " a2: " << params[10] << " a3: " << params[11] << " a4: " << params[12] 
            << " a1: " << params[8] << " a3: " << params[9] << " a5: " << params[10]
            // << " c: " << params[11] << " d: " << params[12] << " e: " << params[13]
            << "\n";
    std::vector<double> round_res(params, params+sizeof(params)/sizeof(double));
    return round_res;
}