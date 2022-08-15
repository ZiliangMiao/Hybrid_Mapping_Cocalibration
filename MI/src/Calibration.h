#ifndef _CALIBRATION_H_
#define _CALIBRATION_H_
/** basic **/
#include <iostream>
#include <fstream>
#include <string.h>
/** opencv **/
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/** pcl **/
#include <pcl/point_cloud.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/common/time.h>
#include <pcl/filters/extract_indices.h>
/** namespace **/
using namespace std;

#define MAX_BINS 256

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define to_radians(x) ( (x) * (M_PI / 180.0 ))
#define to_degrees(x) ( (x) * (180.0 / M_PI ))

#ifndef RTOD
#define RTOD   (180.0 / M_PI)
#endif

#ifndef DTOR
#define DTOR   (M_PI / 180.0)
#endif

#define CAMERA_Z_THRESH 0
#define CAM_PARAM_CONFIG_PATH "../config/master.cfg"
#define RANGE_THRESH 5.0 //in m

namespace perls
{
    class Probability 
    {
        public:
          Probability (){};
          Probability (int n)
          {
              jointProb = cv::Mat::zeros (n, n, CV_32FC1);
              refcProb = cv::Mat::zeros (1, n, CV_32FC1);
              grayProb = cv::Mat::zeros (1, n, CV_32FC1);
              count = 0;
          };
          ~Probability () {};
          //joint Probability
          cv::Mat jointProb;
          //marginal probability reflectivity
          cv::Mat refcProb;
          //marginal probability grayscale
          cv::Mat grayProb;
          int count;
    };

    class Histogram 
    {
        public:
          Histogram (){};
          Histogram (int n)
          {
              jointHist = cv::Mat::zeros (n, n, CV_32FC1);
              refcHist = cv::Mat::zeros (1, n, CV_32FC1);
              grayHist = cv::Mat::zeros (1, n, CV_32FC1);
              count = 0;
          };
          ~Histogram () {};
          //joint Histogram
          cv::Mat jointHist;
          cv::Mat refcHist;
          cv::Mat grayHist; 
          int count;
          int gray_sum;
          int refc_sum;
    };

    class Calibration
    {
        public:
          Calibration ();
          /** parameters **/
          Eigen::VectorXd intrinsic_vec;
          Eigen::Vector3d euler_angle;
          Eigen::Vector3d translation;
          cv::Mat fisheye_img;
          pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud;
          
          string img_path;
          string gray_path;
          string refc_path;
          string refc_hist_img_path;
          string gray_hist_img_path;
          string joint_hist_img_path;
          string refc_prob_img_path;
          string gray_prob_img_path;
          string joint_prob_img_path;
          string point_cloud_org_path;
          string cloud_uv_us_corr_xyz_path;
          string cloud_uv_corr_xyz_path;
          string cloud_uv_us_path;
          string cost_path;

          /**Functions to load the data**/
          int m_estimatorType;
          double m_corrCoeff;
          void   load_point_cloud (std::string cloud_path);
          void   load_image ();
          /*****************************/

          /**Helper functions**/
          void   get_random_numbers (int min, int max, int* index, int num);
          Histogram get_histogram (Eigen::Vector3d translation, Eigen::Vector3d euler);
          Probability get_probability_MLE (Histogram hist);
          Probability get_probability_Bayes (Histogram hist);
          Probability get_probability_JS (Probability probMLE);
          /*****************************/

          /**Cost Functions**/
          float mi_cost (Eigen::Vector3d translation, Eigen::Vector3d euler); 
          float chi_square_cost (Eigen::Vector3d translation, Eigen::Vector3d euler);
          /*****************************/

          /** Covariance Matrix**/
          Eigen::Matrix4d calculate_covariance_matrix (Eigen::Vector3d translation, Eigen::Vector3d euler);
          /*****************************/
          
          /**Optimization Functions**/ 
          float gradient_descent_search (Eigen::Vector3d translation, Eigen::Vector3d euler);
          float exhaustive_grid_search (Eigen::Vector3d translation, Eigen::Vector3d euler);
          /*****************************/
       private:
          int m_NumScans;
          int m_NumCams;
          cv::Mat m_jointTarget;
          cv::Mat m_grayTarget;
          cv::Mat m_refcTarget;
          int m_numBins;
          int m_binFraction; 
    };
}     
#endif //_CALIBRATION_H_
