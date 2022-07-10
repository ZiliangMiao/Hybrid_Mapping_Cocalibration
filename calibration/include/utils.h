// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

// eigen
#include <Eigen/Core>

// headings
#include "spline.h"

using namespace Eigen;
using namespace tk;

template <typename T>
Eigen::Matrix<T, 4, 4> ExtrinsicMat(Eigen::Matrix<T, 6, 1> &extrinsic, bool degree){
    Eigen::Matrix<T, 6, 1> extrinsic_;
    if (degree) {
        Eigen::Matrix<T, 1, 1> deg2rad;
        deg2rad << (T(M_PI) / T(180));
        extrinsic_ = extrinsic * deg2rad;
    }
    else { extrinsic_ = extrinsic; }
    Eigen::Matrix<T, 4, 4> T_mat = ExtrinsicMat(extrinsic_);
    return T_mat;
}

template <typename T>
Eigen::Matrix<T, 4, 4> ExtrinsicMat(Eigen::Matrix<T, 6, 1> &extrinsic){
    /***** R = Rx * Ry * Rz *****/
    Eigen::Matrix<T, 3, 3> R;
    R = Eigen::AngleAxis<T>(extrinsic(2), Eigen::Matrix<T, 3, 1>::UnitZ())
        * Eigen::AngleAxis<T>(extrinsic(1), Eigen::Matrix<T, 3, 1>::UnitY())
        * Eigen::AngleAxis<T>(extrinsic(0), Eigen::Matrix<T, 3, 1>::UnitX());

    Eigen::Matrix<T, 4, 4> T_matrix;
    T_matrix << R(0,0), R(0,1), R(0,2), extrinsic(3),
        R(1,0), R(1,1), R(1,2), extrinsic(4),
        R(2,0), R(2,1), R(2,2), extrinsic(5),
        T(0.0), T(0.0), T(0.0), T(1.0);
    return T_matrix;
}

template <typename T>
Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, 7, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    
    Eigen::Matrix<T, 2, 1> uv_0{intrinsic(0), intrinsic(1)};
    Eigen::Matrix<T, 5, 1> a_;
    a_ << intrinsic(2), intrinsic(3), intrinsic(4), intrinsic(5), intrinsic(6);

    T theta, uv_radius, inv_uv_radius;
    Eigen::Matrix<T, 2, 1> projection;

    theta = acos(point(2) / sqrt((point(0) * point(0)) + (point(1) * point(1)) + (point(2) * point(2))));
    inv_uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4);
    uv_radius = sqrt(point(1) * point(1) + point(0) * point(0));
    projection = {inv_uv_radius / uv_radius * point(0) + uv_0(0), inv_uv_radius / uv_radius * point(1) + uv_0(1)};
    
    return projection;
}

void CeresOutput(vector<const char *> name, double *params, vector<double> params_init) {
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

tk::spline InverseSpline(vector<double> &params){
    Eigen::Matrix<double, 5, 1> a_;
    a_ << params[8], params[9], params[10], params[11], params[12];
    int theta_ub = 180;
    std::vector<double> theta_seq(theta_ub);
    std::vector<double> radius_seq(theta_ub);

    for (double theta = 0; theta < theta_ub; ++theta)
    {
        theta_seq[theta] = (theta * M_PI / 180);
        radius_seq[theta] = a_(0) + a_(1) * (theta * M_PI / 180) + a_(2) * pow((theta * M_PI / 180), 2) + a_(3) * pow((theta * M_PI / 180), 3) + a_(4) * pow((theta * M_PI / 180), 4);
    }

    sort(theta_seq.begin(),theta_seq.end(),
        [&radius_seq](double a, double b){ return radius_seq[a]>radius_seq[b]; });//此处对数据判断，然后对序号排列

    std::reverse(radius_seq.begin(), radius_seq.end());
    std::reverse(theta_seq.begin(), theta_seq.end());
    // default cubic spline (C^2) with natural boundary conditions (f''=0)
    tk::spline spline(radius_seq, theta_seq);			// X needs to be strictly increasing
    return spline;
}