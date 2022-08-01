// basic
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sys/stat.h>

// eigen
#include <Eigen/Core>

// headings
#include "spline.h"

int CheckFolder(std::string spot_path) {
    int md = 0; /** 0 means the folder is already exist or has been created successfully **/
    if (0 != access(spot_path.c_str(), 0)) {
        /** if this folder not exist, create a new one **/
        md = mkdir(spot_path.c_str(), S_IRWXU);
    }
    return md;
}

Eigen::Matrix4f LoadTransMat(string trans_path){
    std::ifstream load_stream;
    load_stream.open(trans_path);
    Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
            load_stream >> trans_mat(j, k);
        }
    }
    load_stream.close();
    return trans_mat;
}

template <typename T>
Eigen::Matrix<T, 4, 4> ExtrinsicMat(Eigen::Matrix<T, 7, 1> &extrinsic){
    Eigen::Matrix<T, 3, 3> R = Eigen::Quaternion<T>(extrinsic[3], extrinsic[0], extrinsic[1], extrinsic[2]).toRotationMatrix();
    Eigen::Matrix<T, 4, 4> T_mat;
    T_mat << R(0,0), R(0,1), R(0,2), extrinsic(4),
        R(1,0), R(1,1), R(1,2), extrinsic(5),
        R(2,0), R(2,1), R(2,2), extrinsic(6),
        T(0.0), T(0.0), T(0.0), T(1.0);
    return T_mat;
}

template <typename T>
Eigen::Matrix<T, 4, 4> ExtrinsicMat(Eigen::Matrix<T, 6, 1> &extrinsic){
    /***** R = Rx * Ry * Rz *****/
    Eigen::Matrix<T, 3, 3> R;
    R = Eigen::AngleAxis<T>(extrinsic(2), Eigen::Matrix<T, 3, 1>::UnitZ())
        * Eigen::AngleAxis<T>(extrinsic(1), Eigen::Matrix<T, 3, 1>::UnitY())
        * Eigen::AngleAxis<T>(extrinsic(0), Eigen::Matrix<T, 3, 1>::UnitX());

    Eigen::Matrix<T, 4, 4> T_mat;
    T_mat << R(0,0), R(0,1), R(0,2), extrinsic(3),
        R(1,0), R(1,1), R(1,2), extrinsic(4),
        R(2,0), R(2,1), R(2,2), extrinsic(5),
        T(0.0), T(0.0), T(0.0), T(1.0);
    return T_mat;
}

template <typename T>
Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, 7, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    
    Eigen::Matrix<T, 10, 1> intrinsic_;
    intrinsic_.head(7) = intrinsic;
    intrinsic_.tail(3) << T(1), T(0), T(0);
    Eigen::Matrix<T, 2, 1> projection = IntrinsicTransform(intrinsic_, point);

    return projection;
}

template <typename T>
Eigen::Matrix<T, 2, 1> IntrinsicTransform(Eigen::Matrix<T, 10, 1> &intrinsic, Eigen::Matrix<T, 3, 1> &point){
    
    Eigen::Matrix<T, 2, 1> uv_0{intrinsic(0), intrinsic(1)};
    Eigen::Matrix<T, 5, 1> a_;
    Eigen::Matrix<T, 2, 2> affine;
    Eigen::Matrix<T, 2, 2> affine_inv;
    T theta, xy_radius, uv_radius;
    Eigen::Matrix<T, 2, 1> projection;
    Eigen::Matrix<T, 2, 1> undistorted_projection;

    a_ << intrinsic(2), intrinsic(3), intrinsic(4), intrinsic(5), intrinsic(6);
    affine << intrinsic(7), intrinsic(8), intrinsic(9), T(1);

    theta = acos(point(2) / sqrt((point(0) * point(0)) + (point(1) * point(1)) + (point(2) * point(2))));
    uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4);
    xy_radius = sqrt(point(1) * point(1) + point(0) * point(0));
    projection = {uv_radius / xy_radius * point(0) + uv_0(0), uv_radius / xy_radius * point(1) + uv_0(1)};
    affine_inv.row(0) << affine(1, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)), - affine(0, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
    affine_inv.row(1) << - affine(1, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)), affine(0, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
    undistorted_projection = affine_inv * projection;
    return undistorted_projection;
}

void SaveResults(std::string &record_path, std::vector<double> params, double bandwidth, double initial_cost, double final_cost) {
    const std::vector<const char*> name = {
            "rx", "ry", "rz",
            "tx", "ty", "tz",
            "u0", "v0",
            "a0", "a1", "a2", "a3", "a4",
            "c", "d", "e"};

    ofstream write;
    const bool title = (bandwidth <= 0);

    std::string output = title ? ("Parameters:\n") : ("Bandwidth = " + to_string(int(bandwidth)) + ":\n");
    ios_base::openmode mode = title ? (ios::out) : (ios::app);
    
    output += "Result:\n[";
    for (int i = 0; i < name.size(); i++) {
        if (i > 0) {
            output += ", ";
        }
        output += title ? (name[i]) : (to_string(params[i]));
    }
    output += "]\n";
    if (!title) {
        output += "Initial cost: " + to_string(initial_cost) + "\n" +
                "Final cost: " + to_string(final_cost) + "\n";
    }
    
    
    write.open(record_path, mode);
    write << output << endl;
    write.close();
    
    cout << output << endl;
}

static bool cmp(const vector<double>& a, const vector<double>& b) {
    return a.back() < b.back();
}

tk::spline InverseSpline(std::vector<double> &params) {
    int idx = params.size() - 8;
    Eigen::Matrix<double, 5, 1> a_;
    a_ << params[idx], params[idx+1], params[idx+2], params[idx+3], params[idx+4];
    int theta_ub = 180;
    // extend the range to get a stable cubic spline
    int extend = 2;
    std::vector<std::vector<double>> r_theta_pts(theta_ub + extend * 2, std::vector<double>(2));

    for (double theta = 0; theta < theta_ub + extend * 2; ++theta) {
        double theta_rad = (theta - extend) * M_PI / 180;
        r_theta_pts[theta][0] = theta_rad;
        r_theta_pts[theta][1] = a_(0) + a_(1) * theta_rad + a_(2) * pow(theta_rad, 2) + a_(3) * pow(theta_rad, 3) + a_(4) * pow(theta_rad, 4);
    }
    sort(r_theta_pts.begin(), r_theta_pts.end(), cmp);
    std::vector<double> input_radius, input_theta;
    for (int i = 0; i < r_theta_pts.size(); i++) {
        input_theta.push_back(r_theta_pts[i][0]);
        input_radius.push_back(r_theta_pts[i][1]);
    }
    // default cubic spline (C^2) with natural boundary conditions (f''=0)
    tk::spline spline(input_radius, input_theta);			// X needs to be strictly increasing
    return spline;
}