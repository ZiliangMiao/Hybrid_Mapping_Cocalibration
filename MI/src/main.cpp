#include "Calibration.h"
#include <iostream>

std::ofstream outfile;

const bool kOptimization = false;
const bool kCostViz = true;

void DualCost(perls::Calibration calib, Eigen::Vector3d translation, Eigen::Vector3d euler_angle) {
    /***** Correlation Analysis *****/
    std::vector<double> inputs1, inputs2;
    std::vector<const char*> euler_name = {
            "rz", "ry", "rx"};
    std::vector<const char*> translation_name = {
            "tx", "ty", "tz"};

    const int steps = 40;

    const double step_size1 = 0.01;
    const int modified_idx1 = 0; // rz
    double offset1;

    const double step_size2 = 0.015;
    const int modified_idx2 = 0; // tx
    double offset2;

    double cost;
    std::vector<double> results;

    for (int param1 = 0; param1 <= steps; param1++) {
        Eigen::Vector3d euler_angle_tmp = euler_angle;
        offset1 = double(step_size1 * (param1 - (steps/2)));
        euler_angle_tmp(modified_idx1) += offset1;

        for (int param2 = 0; param2 <= steps; param2++) {
            Eigen::Vector3d translation_tmp = translation;
            offset2 = step_size2 * (param2 - (steps/2));
            translation_tmp(modified_idx2) += offset2;

            inputs1.push_back(euler_angle_tmp(modified_idx1));
            inputs2.push_back(translation_tmp(modified_idx2));
            /** Evaluate cost funstion **/
            cost = calib.mi_cost(translation_tmp, euler_angle_tmp);
            results.push_back(cost);
            std::cout << "Step: " << param1 << " " << param2 << " Value of " << euler_name[modified_idx1] << ": " << euler_angle_tmp(modified_idx1) << " Value of " << translation_name[modified_idx2] << ": " << translation_tmp(modified_idx2) << " Cost of MI: " << cost << std::endl;
        }
    }
    outfile.open(calib.cost_path + "/" + euler_name[modified_idx1] + "_" + translation_name[modified_idx2] + "_result.txt", std::ios::out);
    for (int i = 0; i < (steps + 1) * (steps + 1); i++) {
        outfile << inputs1[i] << "\t" << inputs2[i] << "\t" << results[i] << std::endl;
    }
    outfile.close();
}

void SingleTranslationCost(perls::Calibration calib, Eigen::Vector3d translation, Eigen::Vector3d euler_angle, int param_idx) {
    /***** Correlation Analysis *****/
    std::vector<double> inputs;
    std::vector<const char*> translation_name = {"tx", "ty", "tz"};

    const int steps = 200;
    const double step_size = 0.01; /** step in meter **/ /** should be 0.015 **/
    double offset;

    double cost;
    std::vector<double> results;
    for (int param = 0; param <= steps; param++) {
        Eigen::Vector3d translation_tmp = translation;
        offset = step_size * (param - (steps/2));
        translation_tmp(param_idx) = translation_tmp(param_idx) + offset;
        inputs.push_back(translation_tmp(param_idx));
        /** Evaluate cost funstion **/
        cost = calib.mi_cost(translation_tmp, euler_angle);
        results.push_back(cost);
        std::cout << "Step: " << param << " Value of " << translation_name[param_idx] << ": " << translation_tmp(param_idx) << " Cost of MI: " << cost << std::endl;
    }
    outfile.open(calib.cost_path + "/" + translation_name[param_idx] + "_result.txt", std::ios::out);
    for (int i = 0; i < (steps + 1); i++) {
        outfile << inputs[i] << "\t" << results[i] << std::endl;
    }
    outfile.close();
}

void SingleRotationCost(perls::Calibration calib, Eigen::Vector3d translation, Eigen::Vector3d euler_angle, int param_idx) {
    /***** Correlation Analysis *****/
    std::vector<double> inputs;
    std::vector<const char*> euler_name = {"rz", "ry", "rx"};

    const int steps = 200;
    const double step_size = 0.005; /** step in radian **/
    double offset;

    double cost;
    std::vector<double> results;
    for (int param = 0; param <= steps; param++) {
        Eigen::Vector3d euler_angle_tmp = euler_angle;
        offset = double(step_size * (param - (steps/2)));
        euler_angle_tmp(param_idx) += offset;
        inputs.push_back(euler_angle_tmp(param_idx));
        /** Evaluate cost funstion **/
        cost = calib.mi_cost(translation, euler_angle_tmp);
        results.push_back(cost);
        std::cout << "Step: " << param << " Value of " << euler_name[param_idx] << ": " << euler_angle_tmp(param_idx) << " Cost of MI: " << cost << std::endl;
    }
    outfile.open(calib.cost_path + "/" + euler_name[param_idx] + "_result.txt", std::ios::out);
    for (int i = 0; i < (steps + 1); i++) {
        outfile << inputs[i] << "\t" << results[i] << std::endl;
    }
    outfile.close();
}


int main (int argc, char** argv)
{
    perls::Calibration calib;
    /** load intrinsic parameters **/
    double tx = 0.277415;
    double ty = -0.0112217;
    double tz = 0.046939;
    double rx = 0.00326059;
    double ry = 3.13658;
    double rz = 1.56319;
//    double tx = 0.27;
//    double ty = 0.00;
//    double tz = 0.03;
//    double rx = M_PI;
//    double ry = 0.00;
//    double rz = -M_PI/2;

    calib.translation << tx, ty, tz;
    calib.euler_angle << rz, ry, rx; // note that the order of euler angle is zyx

    double cost = calib.mi_cost(calib.translation, calib.euler_angle);

//    calib.mat_rotation = Eigen::AngleAxisd(calib.euler_angle[0], Eigen::Vector3d::UnitZ()) *
//                       Eigen::AngleAxisd(calib.euler_angle[1], Eigen::Vector3d::UnitY()) *
//                       Eigen::AngleAxisd(calib.euler_angle[2], Eigen::Vector3d::UnitX());
//    calib.affine_trans.translation() = calib.translation;
//    calib.affine_trans.rotate(calib.mat_rotation);
//    calib.mat_trans = calib.affine_trans.matrix();
//    std::cout << "Transformation Matrix: \n" << calib.mat_trans << std::endl;

    if (kCostViz) {
        double cost = calib.mi_cost(calib.translation, calib.euler_angle);
        calib.get_histogram(calib.translation, calib.euler_angle);
        for (int idx = 0; idx < 3; ++idx) {
            SingleTranslationCost(calib, calib.translation, calib.euler_angle, idx); /** single translation cost analysis **/
            SingleRotationCost(calib, calib.translation, calib.euler_angle, idx); /** single translation cost analysis **/
        }
//        for (int i = 0; i < 3; ++i) {
//            DualCost(calib, calib.translation, calib.euler_angle); /** dual cost analysis **/
//        }
    }

    /** gradient based optimization **/
    double opt_cost = 0;
    if (kOptimization) {
        /** add a time stamp here toc **/
        printf ("****************************************************************************\n");
        printf ("Cost | x (m) | y (m) | z (m) | roll (degree) | pitch (degree) | yaw (degree)\n");
        printf ("****************************************************************************\n");
        opt_cost = calib.gradient_descent_search (calib.translation, calib.euler_angle);
        printf ("****************************************************************************\n");
        /** add a time stamp here tic **/
        printf ("****************************************************************************\n");
        printf ("Calibration parameters:\n");
        printf ("x       = %lf m\ny       = %lf m\nz       = %lf m\nroll    = %lf rad\npitch   = %lf rad\nheading = %lf rad\n",
                calib.translation[0], calib.translation[1], calib.translation[2], calib.euler_angle[2], calib.euler_angle[1], calib.euler_angle[0]);
        printf ("****************************************************************************\n");

        //Save calibration parameters
        FILE *fptr_out = fopen ("calib_param.txt", "w");
        fprintf (fptr_out, "x       = %lf m\ny        = %lf m\nz        = %lf m\nroll     = %lf rad\npitch   = %lf rad\nheading = %lf rad\n",
                 calib.translation[0], calib.translation[1], calib.translation[2], calib.euler_angle[2], calib.euler_angle[1], calib.euler_angle[0]);
        fflush (fptr_out);
        fclose (fptr_out);

        //Calculate covariance
        Eigen::Matrix4d cov_mat = calib.calculate_covariance_matrix (calib.translation, calib.euler_angle);

        FILE *fptr_cov = fopen ("calib_cov.txt", "w");
        printf ("Variance of parameters:\n");
        for (int i = 0; i < 6; i++)
        {
            for(int j = 0; j < 6; j++)
            {
                fprintf (fptr_cov, "%f ", cov_mat(i,j));
                if(i == j) {
                    if(i < 3)
                        std::cout << "Std: \n" << cov_mat(i,j) << std::endl;
                    else
                        std::cout << "Std: \n" << cov_mat(i,j) << std::endl;
                }
            }
            fprintf( fptr_cov, "\n");
        }
        fflush (fptr_cov);
        fclose (fptr_cov);
    }
    return 0;
}
