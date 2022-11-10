#include "Calibration.h"

#define KDE_METHOD
//#define CHI_SQUARE_TEST
//#define _DEBUG_

namespace perls
{
    Calibration::Calibration ()
    {
        /** set intrinsic vectors **/
        this->intrinsic_vec = Eigen::VectorXd(10);

        this->intrinsic_vec <<1022.53, 1198.45, /** u0, v0 **/
                1880.36, -536.721, -12.9298, -18.0154, 5.6414,
                1.00176, -0.00863924, 0.00846056;

        //load images
        load_image ();
        //load scan
        load_point_cloud (this->point_cloud_org_path);

        this->m_jointTarget = cv::Mat::eye (this->m_numBins, this->m_numBins, CV_32FC1)/(this->m_numBins);
        this->m_grayTarget = cv::Mat::ones (1, this->m_numBins, CV_32FC1)/this->m_numBins;
        this->m_refcTarget = cv::Mat::ones (1, this->m_numBins, CV_32FC1)/this->m_numBins;
        return;
    }
   
    /**
     * This function loads the Scan from the file.
     */
    void Calibration::load_point_cloud (std::string cloud_path)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>); /** apply icp result to source point cloud **/
        /** file loading check **/
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(cloud_path, *cloud) == -1) {
            ROS_ERROR("Could Not Load Target File!\n");
        }
        std::cout << "Num of original loaded points = " << cloud->points.size() << std::endl;

        /** initial extrinsic transformation **/
        double tx = 0.27;
        double ty = 0.00;
        double tz = 0.03;
        double rx = M_PI;
        double ry = 0.00;
        double rz = -M_PI/2;
        Eigen::Vector3d translation;
        translation << tx, ty, tz;
        Eigen::Vector3d euler_angle;
        euler_angle << rz, ry, rx; // note that the order of euler angle is zyx
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());

        /** project to fisheye plane **/
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_org = cloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_uv(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_uv_corr_xyz(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_uv_us(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_uv_us_corr_xyz(new pcl::PointCloud<pcl::PointXYZI>);

        /** projection **/
        Eigen::Vector3d point_vec;
        Eigen::Vector2d uv_vec;
        double theta, uv_radius, xy_radius;
        pcl::PointXYZI uv_point;
        for (int i = 0; i < cloud_org->points.size(); i++) {
            uv_point.intensity = cloud_org->points[i].intensity;
            point_vec << cloud_org->points[i].x, cloud_org->points[i].y, cloud_org->points[i].z;
            point_vec = R * point_vec + translation;

            /** intrinsic transformation **/
            Eigen::Vector2d uv_0 = {this->intrinsic_vec(0), this->intrinsic_vec(1)};
            Eigen::VectorXd a_ = Eigen::VectorXd(5);
            Eigen::Matrix2d affine;
            Eigen::Matrix2d affine_inv;

            a_ << this->intrinsic_vec(2), this->intrinsic_vec(3), this->intrinsic_vec(4), this->intrinsic_vec(5), this->intrinsic_vec(6);
            affine << this->intrinsic_vec(7), this->intrinsic_vec(8), this->intrinsic_vec(9), 1.0;

            theta = acos(point_vec(2) / sqrt((point_vec(0) * point_vec(0)) + (point_vec(1) * point_vec(1)) + (point_vec(2) * point_vec(2))));
            uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4);
            xy_radius = sqrt(point_vec(1) * point_vec(1) + point_vec(0) * point_vec(0));
            uv_vec = {uv_radius / xy_radius * point_vec(0) + uv_0(0), uv_radius / xy_radius * point_vec(1) + uv_0(1)};
            affine_inv.row(0) << affine(1, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)),
                    - affine(0, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
            affine_inv.row(1) << - affine(1, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)),
                    affine(0, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
            uv_vec = affine_inv * uv_vec;

            if (0 <= uv_vec[0] && uv_vec[0] < this->fisheye_img.rows && 0 <= uv_vec[1] && uv_vec[1] < this->fisheye_img.cols) {
                if (uv_radius > 400 & uv_radius < 1000) {
                    /** points on uv plane **/
                    uv_point.x = uv_vec[0];
                    uv_point.y = uv_vec[1];
                    uv_point.z = 0;
                    cloud_uv->points.push_back(uv_point);
                    /** corresponding point in 3d spatial space **/
                    cloud_uv_corr_xyz->points.push_back(cloud_org->points[i]);
                }
            }
        }
        std::cout << "Num of point in uv plane: " << cloud_uv->points.size() << std::endl;
        pcl::io::savePCDFileBinary(this->cloud_uv_corr_xyz_path, *cloud_uv_corr_xyz);

        /** uniform sampling in cloud_uv **/
        pcl::UniformSampling<pcl::PointXYZI> us(true);
        us.setRadiusSearch(3.0);
        us.setInputCloud(cloud_uv);
        us.filter(*cloud_uv_us);

        pcl::PointIndices indices;
        const pcl::IndicesConstPtr& us_removed_idx = us.getRemovedIndices();

        pcl::io::savePCDFileBinary(this->cloud_uv_us_path, *cloud_uv_us);
        std::cout << "Num of uniform sampling cloud: " << cloud_uv_us->points.size() << std::endl;

        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(cloud_uv_corr_xyz);
        extract.setIndices(us_removed_idx);
        extract.setNegative(true);
        extract.filter(*cloud_uv_us_corr_xyz);
        std::cout << "Num of cloud in 3d space corresponding to the uniform sampling cloud in uv plane: " << cloud_uv_us_corr_xyz->points.size() << std::endl;

        pcl::io::savePCDFileBinary(this->cloud_uv_us_corr_xyz_path, *cloud_uv_us_corr_xyz);

        /** load the saved point cloud **/
        pcl::io::loadPCDFile(this->cloud_uv_us_path, *cloud_uv_us);
        pcl::io::loadPCDFile(this->cloud_uv_us_corr_xyz_path, *cloud_uv_us_corr_xyz);
        this->point_cloud = cloud_uv_us_corr_xyz;

//        double DIST_THRESH = 10000;
//        for (auto & point : cloud->points) {
//            point.intensity = int(point.intensity / 150 * 255);
//            if (point.intensity >= 255) {
//                point.intensity = 255;
//            }
//            double dist = point.x * point.x + point.y * point.y + point.z * point.z;
//            double range = dist/DIST_THRESH;
//            this->point_range.push_back(range);
//        }

        /** point cloud segmentation by euler space clustering **/
//        // Creating the KdTree object for the search method of the extraction
//        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
//        tree->setInputCloud (cloud_uv_us); //将无法提取平面的点云作为cloud_filtered
//
//        std::vector<pcl::PointIndices> cluster_indices; //保存每一种聚类，每一种聚类下还有具体的聚类的点
//        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec; //实例化一个欧式聚类提取对象
//        ec.setClusterTolerance (5); // 近邻搜索的搜索半径为2cm，重要参数
//        ec.setMinClusterSize (100); //设置一个聚类需要的最少点数目为100
//        ec.setMaxClusterSize (50000); //一个聚类最大点数目为25000
//        ec.setSearchMethod (tree); //设置点云的搜索机制
//        ec.setInputCloud (cloud_uv_us); //设置输入点云
//        ec.extract (cluster_indices); //将聚类结果保存至cluster_indices中
//
//        /** 迭代访问点云索引cluster_indices，直到分割出所有聚类,一个循环提取出一类点云 **/
//        int j = 0; /** cluster index **/
//        int color_index = 0; /** 0->blue, 1->green, 2->red **/
//        int num_clusters = cluster_indices.size();
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZRGB>);
//        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
//            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_single_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
//            //创建新的点云数据集cloud_cluster，直到分割出所有聚类
//            if (color_index == 0) {
//                pcl::PointXYZRGB point;
//                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {
//                    point.x = cloud_uv_us->points[*pit].x;
//                    point.y = cloud_uv_us->points[*pit].y;
//                    point.z = cloud_uv_us->points[*pit].z;
//                    point.b = 255;
//                    point.g = 0;
//                    point.r = 0;
//                    cloud_single_cluster->points.push_back (point);
//                }
//                color_index = 1;
//            }
//            else if (color_index == 1) {
//                pcl::PointXYZRGB point;
//                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {
//                    point.x = cloud_uv_us->points[*pit].x;
//                    point.y = cloud_uv_us->points[*pit].y;
//                    point.z = cloud_uv_us->points[*pit].z;
//                    point.b = 0;
//                    point.g = 255;
//                    point.r = 0;
//                    cloud_single_cluster->points.push_back (point);
//                }
//                color_index = 2;
//            }
//            else if (color_index == 2) {
//                pcl::PointXYZRGB point;
//                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) {
//                    point.x = cloud_uv_us->points[*pit].x;
//                    point.y = cloud_uv_us->points[*pit].y;
//                    point.z = cloud_uv_us->points[*pit].z;
//                    point.b = 0;
//                    point.g = 0;
//                    point.r = 255;
//                    cloud_single_cluster->points.push_back (point);
//                }
//                color_index = 0;
//            }
//            std::cout << "PointCloud representing the Cluster: " << cloud_single_cluster->points.size () << " data points." << std::endl;
//            j++;
//            *cloud_clusters += *cloud_single_cluster;
//        }
//        pcl::io::savePCDFileBinary(this->cloud_clusters_path, *cloud_clusters);
    }

    /**
     * This function loads the images
     */
    void Calibration::load_image () {
        cv::Mat fisheye_hdr_image = cv::imread(this->img_path, cv::IMREAD_UNCHANGED);
        cv::Mat fisheye_greyscale_img;
        cv::cvtColor(fisheye_hdr_image, fisheye_greyscale_img, cv::COLOR_BGR2GRAY);

//        cv::GaussianBlur (imageMat, outMat, cv::Size (3, 3), 0.75);
        this->fisheye_img = fisheye_greyscale_img;
    }

    /**
     * This function computes the smoothed distribution at a given transformation x
     */
    Histogram Calibration::get_histogram (Eigen::Vector3d translation, Eigen::Vector3d euler) {
        cv::Mat img = this->fisheye_img;
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitZ())
                       * Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY())
                       * Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitX());
//        std::cout << "Translation Vector: \n" << translation << std::endl;
//        std::cout << "Rotation Matrix: \n" << R << std::endl;

        //count the number of points that project onto the valid image region
        float gray_sum = 0;
        float refc_sum = 0;

        Histogram hist (this->m_numBins);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = this->point_cloud;

        //Loop over all points
        Eigen::Vector3d point_vec;
        Eigen::Vector3d point_trans_vec;
        Eigen::Vector2d uv_vec;
        double theta, uv_radius, xy_radius;
        pcl::PointXYZI point, uv_point;

        /** check of projected points on uv plane **/
        pcl::PointCloud<pcl::PointXYZI>::Ptr uv_cloud(new pcl::PointCloud<pcl::PointXYZI>);

        std::ofstream gray_outfile;
        gray_outfile.open(this->gray_path, std::ios::out);
        std::ofstream refc_outfile;
        refc_outfile.open(this->refc_path, std::ios::out);

        for (int i = 0; i < cloud->points.size(); i++) {
            point = cloud->points[i];
            uv_point.intensity = point.intensity;

            point_vec << point.x, point.y, point.z;
            point_trans_vec = R * point_vec + translation;
//                std::cout << "Transformed Point Position: \n" << point_trans_vec << std::endl;

            // calculate projection on image
            /** intrinsic transformation **/
            Eigen::Vector2d uv_0 = {this->intrinsic_vec(0), this->intrinsic_vec(1)};
            Eigen::VectorXd a_ = Eigen::VectorXd(5);
            Eigen::Matrix2d affine;
            Eigen::Matrix2d affine_inv;

            a_ << this->intrinsic_vec(2), this->intrinsic_vec(3), this->intrinsic_vec(4), this->intrinsic_vec(5), this->intrinsic_vec(6);
            affine << this->intrinsic_vec(7), this->intrinsic_vec(8), this->intrinsic_vec(9), 1.0;

            theta = acos(point_trans_vec(2) / sqrt((point_trans_vec(0) * point_trans_vec(0)) + (point_trans_vec(1) * point_trans_vec(1)) + (point_trans_vec(2) * point_trans_vec(2))));
            uv_radius = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4);
            xy_radius = sqrt(point_trans_vec(1) * point_trans_vec(1) + point_trans_vec(0) * point_trans_vec(0));
            uv_vec = {uv_radius / xy_radius * point_trans_vec(0) + uv_0(0), uv_radius / xy_radius * point_trans_vec(1) + uv_0(1)};
            affine_inv.row(0) << affine(1, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)),
                                 - affine(0, 1) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
            affine_inv.row(1) << - affine(1, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1)),
                                 affine(0, 0) / (affine(0, 0) * affine(1, 1) - affine(1, 0) * affine(0, 1));
            uv_vec = affine_inv * uv_vec;

            //if image_point is within the frame
            int gray, refc;
            if (0 <= uv_vec[0] && uv_vec[0] < img.rows && 0 <= uv_vec[1] && uv_vec[1] < img.cols) {
                if (uv_radius > 400 & uv_radius < 1000) {
                    /** points on uv plane **/
                    uv_point.x = uv_vec[0];
                    uv_point.y = uv_vec[1];
                    uv_point.z = 0;
                    uv_cloud->points.push_back(uv_point);

                    /** get the grayscale if point within frame **/
//                        int b = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[0];
//                        int g = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[1];
//                        int r = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[2];
//                        gray = (0.114 * b + 0.587 * g + 0.299 * r) / this->m_binFraction;
                    gray = img.at<cv::Vec3b>(uv_vec[0], uv_vec[1])[0] / this->m_binFraction;
                    refc = point.intensity / this->m_binFraction;

//                        gray_outfile << gray << "\t" << std::endl;
//                        refc_outfile << refc << "\t" << std::endl;

                    hist.grayHist.at<float>(gray) = hist.grayHist.at<float>(gray) + 1;
                    hist.refcHist.at<float>(refc) = hist.refcHist.at<float>(refc) + 1;
                    hist.jointHist.at<float>(gray, refc) = hist.jointHist.at<float>(gray, refc) + 1;
                    hist.count++;
                    gray_sum = gray_sum + gray;
                    refc_sum = refc_sum + refc;
                }
            }
        }
        gray_outfile.close();
        refc_outfile.close();

        /** save the hist of grayscale and reflectivity **/


        int gray_size = hist.grayHist.cols;
        int refc_size = hist.refcHist.cols;
        int joint_r_size = hist.jointHist.rows;
        int joint_c_size = hist.jointHist.cols;

//        std::ofstream outfile;
//        outfile.open(this->gray_hist_path, std::ios::out);
//        for (int i = 0; i < gray_size; i++) {
//            outfile << hist.grayHist.at<float>(i) << "\t";
//        }
//        outfile.close();
//        outfile.open(this->refc_hist_path, std::ios::out);
//        for (int i = 0; i < refc_size; i++) {
//            outfile << hist.refcHist.at<float>(i) << "\t";
//        }
//        outfile.close();
//        outfile.open(this->joint_hist_path, std::ios::out);
//        for (int r = 0; r < joint_r_size; ++r) {
//            for (int c = 0; c < joint_c_size; ++c) {
//                outfile << hist.refcHist.at<float>(r, c) << "\t";
//            }
//            outfile << "\n";
//        }
//        outfile.close();

        /** save the joint hist image **/
        cv::imwrite(this->refc_hist_img_path, hist.refcHist);
        cv::imwrite(this->gray_hist_img_path, hist.grayHist);
        cv::imwrite(this->joint_hist_img_path, hist.jointHist);

//        /** save the uv point to check **/
//        pcl::io::savePCDFileBinary(this->check_uv_cloud_path, *uv_cloud);

        hist.gray_sum = gray_sum;
        hist.refc_sum = refc_sum;
        return hist;
    }

    Probability
    Calibration::get_probability_MLE (Histogram hist)
    {
        //Calculate sample covariance matrix
        float mu_gray = hist.gray_sum/hist.count;
        float mu_refc = hist.refc_sum/hist.count;
        //Covariances
        double sigma_gray = 0;
        double sigma_refc = 0;
        //Cross correlation
        double sigma_gr = 0;
        
        Probability probMLE (this->m_numBins);
        
        for (int i = 0; i < this->m_numBins; i++)
        {
           for (int j = 0; j < this->m_numBins; j++)
           {
             //Cross Correlation term;
             sigma_gr = sigma_gr + hist.jointHist.at<float>(i, j)*(i - mu_refc)*(j - mu_gray);
             //Normalize the histogram so that the value is between (0,1)
             probMLE.jointProb.at<float>(i, j) = hist.jointHist.at<float>(i, j)/(hist.count);
           }
    
           //calculate sample covariance 
           sigma_gray = sigma_gray + (hist.grayHist.at<float>(i)*(i - mu_gray)*(i - mu_gray));
           sigma_refc = sigma_refc + (hist.refcHist.at<float>(i)*(i - mu_refc)*(i - mu_refc));
           
           probMLE.grayProb.at<float>(i) = hist.grayHist.at<float>(i)/hist.count;
           probMLE.refcProb.at<float>(i) = hist.refcHist.at<float>(i)/hist.count;
        }
    
        sigma_gray = sigma_gray/hist.count;
        sigma_refc = sigma_refc/hist.count;
        sigma_gr = sigma_gr/hist.count;
        double corr_coeff = ((sigma_gr)/(sigma_gray*sigma_refc));
        corr_coeff = sqrt (corr_coeff*corr_coeff);
        this->m_corrCoeff = corr_coeff;
        
        //Compute the optimal bandwidth (Silverman's rule of thumb)
        sigma_gray = 1.06*sqrt (sigma_gray)/pow (hist.count, 0.2);
        sigma_refc = 1.06*sqrt (sigma_refc)/pow (hist.count, 0.2); 

        cv::GaussianBlur (probMLE.grayProb, probMLE.grayProb, cv::Size(0, 0), sigma_gray);
        cv::GaussianBlur (probMLE.refcProb, probMLE.refcProb, cv::Size(0, 0), sigma_refc);
        cv::GaussianBlur (probMLE.jointProb, probMLE.jointProb, cv::Size(0, 0), sigma_gray, sigma_refc);
        probMLE.count = hist.count; 
        return probMLE; 
    }

    /**
     * This function calculates the JS estimate from the MLE.
     */
    Probability 
    Calibration::get_probability_JS (Probability probMLE)
    {
        //Calculate JS estimate
        //Estimate lamda from the data
        /** 
        //Reference:[1] Entropy inference and the James Stein estimator. Hausser and Strimmer.
        //Sample Variance of MLE
        //Using unbiased estimator of variance as given in [1]:
        //Var(\theta_k) = \frac{\theta_k(1 - \theta_k)}{n-1}
        //=> \lambda = \frac{1 - \sum_{k=0}^{K} (\theta_k)^2}{\sum_{k=0}^{K} (t_k - \theta_k)
        //Here t_k      = target distribution (here m_jointTarget)
        //     \theta_k = MLE estimate (here probMLE.jointProb)  
        **/
        float squareSumMLE = cv::norm (probMLE.jointProb);
        squareSumMLE = (squareSumMLE*squareSumMLE); 
        //Difference of MLE from the target 
        float squareDiffMLETarget = cv::norm (this->m_jointTarget, probMLE.jointProb);
        squareDiffMLETarget =  (squareDiffMLETarget*squareDiffMLETarget);
         
        float lambda = (1.0-squareSumMLE)/squareDiffMLETarget;
        lambda = (lambda/(probMLE.count-1));
        //lambda = squareDiffMLETarget/squareSumMLE;
        //lambda = sqrt(this->m_corrCoeff); 
//        std::cout << lambda << " " << squareSumMLE << " " << squareDiffMLETarget << std::endl;
        //lambda = 1 - sqrt(squareSumMLE); 
        //lambda = 0.0; 
        if (lambda > 1)
            lambda = 1;
        if (lambda < 0)
            lambda = 0; 
        
        //Scale the target distribution by lambda
        //Scale the MLE or the histograms by (1-lambda)  
        //Get the JS estimate as a weighted combination of target and the MLE
        Probability probJS (this->m_numBins);

        probJS.jointProb = this->m_jointTarget*lambda + probMLE.jointProb*(1.0-lambda);
        //cv::GaussianBlur (jointJSEstimate, jointJSEstimate, cv::Size(5, 5), 1.2, 1.2);
        probJS.grayProb = this->m_grayTarget*lambda + probMLE.grayProb*(1.0-lambda);
        //cv::GaussianBlur (margJSEstimate1, margJSEstimate1, cv::Size(1, 5), 0, 1.2);
        probJS.refcProb = this->m_refcTarget*lambda + probMLE.refcProb*(1.0-lambda);
        //cv::GaussianBlur (margJSEstimate2, margJSEstimate2, cv::Size(1, 5), 0, 1.2);

        /** save the image of probability **/
        cv::imwrite(this->joint_prob_img_path, probJS.jointProb * 255);
        
        return probJS;
    }
    
    /**
     * This calculates the Bayes estimate of distribution
     */ 
    Probability 
    Calibration::get_probability_Bayes (Histogram hist)
    {
        float a = 1; //0.5 , 1/this->m_numBins, sqrt (count)/this->m_numBins etc
        float A_joint = this->m_numBins*this->m_numBins;
        float A_marg = this->m_numBins;
        Probability probBayes (this->m_numBins);
        //Calculate sample covariance matrix
        float mu_gray = hist.gray_sum/hist.count;
        float mu_refc = hist.refc_sum/hist.count;
        //Covariances
        double sigma_gray = 0;
        double sigma_refc = 0;
        //Cross correlation
        double sigma_gr = 0;
        
        for (int i = 0; i < this->m_numBins; i++)
        {
           for (int j = 0; j < this->m_numBins; j++)
           {
             //Cross Correlation term;
             sigma_gr = sigma_gr + hist.jointHist.at<float>(i, j)*(i - mu_refc)*(j - mu_gray);
             //Normalize the histogram so that the value is between (0,1)
             probBayes.jointProb.at<float>(i, j) = (hist.jointHist.at<float>(i, j)+a)/(hist.count + A_joint);
           }
    
           //calculate sample covariance 
           sigma_gray = sigma_gray + (hist.grayHist.at<float>(i)*(i - mu_gray)*(i - mu_gray));
           sigma_refc = sigma_refc + (hist.refcHist.at<float>(i)*(i - mu_refc)*(i - mu_refc));
           
           probBayes.grayProb.at<float>(i) = (hist.grayHist.at<float>(i) + a)/(hist.count + A_marg);
           probBayes.refcProb.at<float>(i) = (hist.refcHist.at<float>(i) + a)/(hist.count + A_marg);
        }
    
        sigma_gray = sigma_gray/hist.count;
        sigma_refc = sigma_refc/hist.count;
        sigma_gr = sigma_gr/hist.count;
        double corr_coeff = ((sigma_gr)/(sigma_gray*sigma_refc));
        corr_coeff = sqrt (corr_coeff*corr_coeff);
        this->m_corrCoeff = corr_coeff;
        
        //Compute the optimal bandwidth (Silverman's rule of thumb)
        sigma_gray = 1.06*sqrt (sigma_gray)/pow (hist.count, 0.2);
        sigma_refc = 1.06*sqrt (sigma_refc)/pow (hist.count, 0.2); 
        
        cv::GaussianBlur (probBayes.grayProb, probBayes.grayProb, cv::Size(0, 0), sigma_gray);
        cv::GaussianBlur (probBayes.refcProb, probBayes.refcProb, cv::Size(0, 0), sigma_refc);
        cv::GaussianBlur (probBayes.jointProb, probBayes.jointProb, cv::Size(0, 0), sigma_gray, sigma_refc);
        return probBayes; 
    }


    /**
     * This function calculates the cost based on mutual information with multiple scans
     */
    float
    Calibration::mi_cost (Eigen::Vector3d translation, Eigen::Vector3d euler)
    {
        //Get MLE of probability distribution
        Histogram hist = get_histogram (translation, euler);
        Probability prob;
        Probability probMLE; 
        switch (this->m_estimatorType)
        {
            case 1: //MLE
                prob = get_probability_MLE (hist);
                break;
            case 2: //James-Stein type
                probMLE = get_probability_MLE (hist);
                prob = get_probability_JS (probMLE); 
                break;
            case 3: //Bayes estimator
                prob = get_probability_Bayes (hist);
                break;
        }

        //Calculate log of JS estimate
        cv::Mat jointLog = cv::Mat::zeros(this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat grayLog = cv::Mat::zeros(1, this->m_numBins, CV_32FC1); 
        cv::Mat refcLog = cv::Mat::zeros(1, this->m_numBins, CV_32FC1);
 
        cv::log (prob.jointProb, jointLog);
        cv::log (prob.grayProb, grayLog);
        cv::log (prob.refcProb, refcLog);
        
        cv::Mat jointEntropyMat, grayEntropyMat, refcEntropyMat;
        //jointEntropyMat = jointJSEstimate*jointJSLog;
        cv::multiply (prob.jointProb, jointLog, jointEntropyMat);
        //margEntropyMat1 = margJSEstimate1*margJSLog1;
        cv::multiply (prob.grayProb, grayLog, grayEntropyMat); 
        //margEntropyMat2 = margJSEstimate2*margJSLog2;
        cv::multiply (prob.refcProb, refcLog, refcEntropyMat); 
        
        //Sum all the elements
        float Hx  = cv::norm (grayEntropyMat, cv::NORM_L1);
        float Hy  = cv::norm (refcEntropyMat, cv::NORM_L1);
        float Hxy = cv::norm (jointEntropyMat, cv::NORM_L1);
        
        float cost = Hx + Hy - Hxy;
        //float cost = Hxy;
        return cost;
    }

    /**
     * This function calculates the chi square cost
     */
    float 
    Calibration::chi_square_cost (Eigen::Vector3d translation, Eigen::Vector3d euler)
    {
        //Get the joint and marginal probabilities
        Histogram hist = get_histogram (translation, euler);
        Probability P = get_probability_MLE (hist);
        
        //Calculate the chi square cost
        float cost = 0;
        for (int i = 0; i < this->m_numBins; i++)
        {
          for (int j = 0; j < this->m_numBins; j++)
          {
            if (P.refcProb.at<float>(i) > 0 && P.grayProb.at<float>(j) > 0)
              cost = cost + (P.jointProb.at<float>(i, j) - P.refcProb.at<float>(i)*P.grayProb.at<float>(j))*
                   (P.jointProb.at<float>(i, j) - P.refcProb.at<float>(i)*P.grayProb.at<float>(j))/(P.refcProb.at<float>(i)*P.grayProb.at<float>(j));
          }
        }
        //printf ("chi_square_cost = %lf\n", cost);
        return cost;
    }

    /**
     * This function performs the exhaustive grid based search for the transformation 
     * parameters
     */
    float
    Calibration::exhaustive_grid_search (Eigen::Vector3d translation, Eigen::Vector3d euler)
    {
        //create the 1st level grid around x0_hl
        //1st level grid : 
        //[x, y, z] = +-0.20m and step = 0.05m
        //[r, p, h] = +-3 degrees and step = +-1 degree

        Eigen::Vector3d translation_0;
        Eigen::Vector3d euler_0;
        Eigen::Vector3d translation_max_l1 = translation;
        Eigen::Vector3d euler_max_l1 = euler;
        Eigen::Vector3d translation_max_l2;
        Eigen::Vector3d euler_max_l2;
     
        double gridsize_trans = 0.2;
        double step_trans = 0.05;
        double gridsize_rot = 3*DTOR;
        double step_rot = 1*DTOR;
        double max_cost_l1 = 0; 

        float curr_cost = 0;
        for (double x = translation[0] - gridsize_trans; x <= translation[0] + gridsize_trans; x = x + step_trans) {
          for (double y = translation[1] - gridsize_trans; y <= translation[1] + gridsize_trans; y = y + step_trans) {
            for (double z = translation[2] - gridsize_trans; z <= translation[2] + gridsize_trans; z = z + step_trans) {
              for (double r = euler[2] - gridsize_rot; r <= euler[2] + gridsize_rot; r = r + step_rot) {
                for (double p = euler[1] - gridsize_rot; p <= euler[1] + gridsize_rot; p = p + step_rot) {
                  for (double h = euler[0] - gridsize_rot; h <= euler[0] + gridsize_rot; h = h + step_rot) {
                      translation_0[0] = x; translation_0[1] = y; translation_0[2] = z;
                      euler_0[2] = r; euler_0[1] = p; euler_0[0] = h;
                      curr_cost = this->mi_cost (translation_0, euler_0);

                      if (curr_cost > max_cost_l1) {
                          max_cost_l1 = curr_cost;
                          translation_max_l1 = translation_0;
                          euler_max_l1 = euler_0;
                      }
                  }
                }
              }
            }
          }
        }
    
        printf ("Level 1 Grid search done\n"); 
        printf ("%lf %lf %lf %lf %lf %lf %lf\n", max_cost_l1, translation_max_l1[0], translation_max_l1[1], translation_max_l1[2], euler_max_l1[2]*RTOD, euler_max_l1[1]*RTOD, euler_max_l1[0]*RTOD);

        //create the second level grid around the point with max cost in level 1 search
        //2nd level grid :
        //[x, y, z] = +-0.05m and step = 0.01m
        //[r, p, h] = +-1 degree and step = 0.1 degree

        //max cost point
        translation = translation_max_l1;
        euler = euler_max_l1;
        step_trans = 0.01;
        gridsize_trans = 0.04;
        gridsize_rot = 0.5*DTOR;
        step_rot = 0.1*DTOR;
        float max_cost_l2 = this->mi_cost (translation_max_l1, euler_max_l1);
        translation_max_l2 = translation_max_l1;
        euler_max_l2 = euler_max_l1;

        curr_cost = 0;
        for (double x = translation[0] - gridsize_trans; x <= translation[0] + gridsize_trans; x = x + step_trans) {
            for (double y = translation[1] - gridsize_trans; y <= translation[1] + gridsize_trans; y = y + step_trans) {
                for (double z = translation[2] - gridsize_trans; z <= translation[2] + gridsize_trans; z = z + step_trans) {
                    for (double r = euler[2] - gridsize_rot; r <= euler[2] + gridsize_rot; r = r + step_rot) {
                        for (double p = euler[1] - gridsize_rot; p <= euler[1] + gridsize_rot; p = p + step_rot) {
                            for (double h = euler[0] - gridsize_rot; h <= euler[0] + gridsize_rot; h = h + step_rot) {
                                translation_0[0] = x; translation_0[1] = y; translation_0[2] = z;
                                euler_0[2] = r; euler_0[1] = p; euler_0[0] = h;
                                curr_cost = this->mi_cost (translation_0, euler_0);

                                if (curr_cost > max_cost_l2) {
                                    max_cost_l2 = curr_cost;
                                    translation_max_l2 = translation_0;
                                    euler_max_l2 = euler_0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        //Set x0_hl to the maxima.
        translation = translation_max_l2;
        euler = euler_max_l2;
        return max_cost_l2;
    }

    /**
     * This function performs the gradient descent search for the transformation 
     * parameters
     */
    float
    Calibration::gradient_descent_search (Eigen::Vector3d translation, Eigen::Vector3d euler)
    {
        //step parameter
        double gama_trans = 0.01;
        double gama_rot = 0.001;
        double gama_trans_u = 0.1;
        double gama_trans_l = 0.001;
        double gama_rot_u = 0.05;
        double gama_rot_l = 0.0005;
    
        double deltax = 0.01, deltay = 0.01, deltaz = 0.01;
        double deltar = 0.1*DTOR, deltap = 0.1*DTOR, deltah = 0.1*DTOR;

        Eigen::Vector3d euler_k = this->euler_angle; // zyx
        Eigen::Vector3d translation_k = this->translation; // xyz

        Eigen::Vector3d euler_k_minus_1 = this->euler_angle; // zyx
        Eigen::Vector3d translation_k_minus_1 = this->translation; // xyz

        int index = 0;
        int MAX_ITER = 300;
        double delF_delX_ = 0, delF_delY_ = 0, delF_delZ_ = 0;
        double delF_delR_ = 0, delF_delP_ = 0, delF_delH_ = 0;
        double f_max = 0;
        while (index < MAX_ITER)
        {
            std::cout << "The current search iteration: " << index << std::endl;
            //Evaluate function value
            double f_prev;
            #ifdef CHI_SQUARE_TEST
              f_prev = chi_square_cost (translation_k, euler_k);
            #else
              f_prev = mi_cost (translation_k, euler_k);
            #endif
            if (f_prev > f_max)
                f_max = f_prev;
    
            double _f = 0; 
    
            double x, y, z, r, p, h;
            x = translation_k[0]; y = translation_k[1]; z = translation_k[2];
            r = euler_k[2]; p = euler_k[1]; h = euler_k[0];

            Eigen::Vector3d euler_delta = {h, p, r}; // zyx
            Eigen::Vector3d translation_delta = {x + deltax, y, z}; // xyz

            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
    
            double delF_delX = (_f - f_prev)/deltax;

            translation_delta = {x, y + deltay, z}; // xyz
            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
            double delF_delY = (_f - f_prev)/deltay;

            translation_delta = {x, y, z + deltaz}; // xyz
            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
            double delF_delZ = (_f - f_prev)/deltaz;

            euler_delta = {h, p, r + deltar}; // zyx
            translation_delta = {x, y, z}; // xyz
            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
            double delF_delR = (_f - f_prev)/deltar;

            euler_delta = {h, p + deltap, r}; // zyx
            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
            double delF_delP = (_f - f_prev)/deltap;

            euler_delta = {h + deltah, p, r}; // zyx
            #ifdef CHI_SQUARE_TEST
              _f = chi_square_cost (translation_delta, euler_delta);
            #else 
              _f = mi_cost (translation_delta, euler_delta);
            #endif
            double delF_delH = (_f - f_prev)/deltah;
    
            double norm_delF_del_trans = sqrt(delF_delX*delF_delX + delF_delY*delF_delY + delF_delZ*delF_delZ); 
            double norm_delF_del_rot   = sqrt(delF_delR*delF_delR + delF_delP*delF_delP + delF_delH*delF_delH);
    
            delF_delX = delF_delX/norm_delF_del_trans;
            delF_delY = delF_delY/norm_delF_del_trans;
            delF_delZ = delF_delZ/norm_delF_del_trans;
            delF_delR = delF_delR/norm_delF_del_rot;
            delF_delP = delF_delP/norm_delF_del_rot;
            delF_delH = delF_delH/norm_delF_del_rot;
   
            double delta_trans = ((translation_k[0] - translation_k_minus_1[0])*(translation_k[0] - translation_k_minus_1[0])
                               + (translation_k[1] - translation_k_minus_1[1])*(translation_k[1] - translation_k_minus_1[1])
                               + (translation_k[2] - translation_k_minus_1[2])*(translation_k[2] - translation_k_minus_1[2]));
            double delta_rot = ((euler_k[2] - euler_k_minus_1[2])*(euler_k[2] - euler_k_minus_1[2])
                             + (euler_k[1] - euler_k_minus_1[1])*(euler_k[1] - euler_k_minus_1[1])
                             + (euler_k[0] - euler_k_minus_1[0])*(euler_k[0] - euler_k_minus_1[0]));
            
            //get the scaling factor
            if (delta_trans > 0)
            {
               double temp_deno_trans = ((translation_k[0] - translation_k_minus_1[0])*(delF_delX - delF_delX_)
                                      + (translation_k[1] - translation_k_minus_1[1])*(delF_delY - delF_delY_)
                                      + (translation_k[2] - translation_k_minus_1[2])*(delF_delZ - delF_delZ_));
               temp_deno_trans = sqrt (temp_deno_trans*temp_deno_trans);
               gama_trans = delta_trans/temp_deno_trans; 
            }
            else
               gama_trans = gama_trans_u;
    
            if (delta_rot > 0)
            {
               double temp_deno_rot = ((euler_k[2] - euler_k_minus_1[2])*(delF_delR - delF_delR_)
                                    + (euler_k[1] - euler_k_minus_1[1])*(delF_delP - delF_delP_)
                                    + (euler_k[0] - euler_k_minus_1[0])*(delF_delH - delF_delH_));
               temp_deno_rot = sqrt (temp_deno_rot*temp_deno_rot);
               gama_rot = delta_rot/temp_deno_rot;  
            }
            else
               gama_rot = gama_rot_u;
            
            //printf ("Before: gama_trans = %f, gama_rot = %f\n", gama_trans, gama_rot);
            //Since we are looking at maxima.
            if (gama_trans > gama_trans_u)
                gama_trans = gama_trans_u;
            if (gama_trans < gama_trans_l)
                gama_trans = gama_trans_l;
            
            if (gama_rot > gama_rot_u)
                gama_rot = gama_rot_u;
            if (gama_rot < gama_rot_l)
                gama_rot = gama_rot_l;
    
            //printf ("After: gama_trans = %f, gama_rot = %f\n", gama_trans, gama_rot);
            translation_k_minus_1[0] = translation_k[0];
            translation_k_minus_1[1] = translation_k[1];
            translation_k_minus_1[2] = translation_k[2];
            euler_k_minus_1[2] = euler_k[2];
            euler_k_minus_1[1] = euler_k[1];
            euler_k_minus_1[0] = euler_k[0];

            translation_k[0] = translation_k[0] + gama_trans*delF_delX;
            translation_k[1] = translation_k[1] + gama_trans*delF_delY;
            translation_k[2] = translation_k[2] + gama_trans*delF_delZ;
            euler_k[2] = euler_k[2] + gama_rot*delF_delR;
            euler_k[1] = euler_k[1] + gama_rot*delF_delP;
            euler_k[0] = euler_k[0] + gama_rot*delF_delH;
    
            double f_curr;
            #ifdef CHI_SQUARE_TEST
              f_curr = chi_square_cost (translation_k, euler_k);
            #else 
              f_curr = mi_cost (translation_k, euler_k);
            #endif
    
            if (f_curr < f_prev)
            {
              translation_k[0] = translation_k[0] - gama_trans*delF_delX;
              translation_k[1] = translation_k[1] - gama_trans*delF_delY;
              translation_k[2] = translation_k[2] - gama_trans*delF_delZ;
              euler_k[2] = euler_k[2] - gama_rot*delF_delR;
              euler_k[1] = euler_k[1] - gama_rot*delF_delP;
              euler_k[0] = euler_k[0] - gama_rot*delF_delH;
              gama_rot_u = gama_rot_u/1.2;
              gama_rot_l = gama_rot_l/1.2;
              gama_trans_u = gama_trans_u/1.2;
              gama_trans_l = gama_trans_l/1.2;
              
              deltax = deltax/1.1; deltay = deltay/1.1; deltaz = deltaz/1.1;
              deltar = deltar/1.1; deltap = deltap/1.1; deltah = deltah/1.1;
              //printf ("f_curr = %lf,  f_prev = %lf, deltax = %lf\n", f_curr, f_prev, deltax);
              index++;
              if (deltax < 0.001)
                 break;
              else
                 continue; 
            }
            index = index + 1;

            delF_delX_ = delF_delX;
            delF_delY_ = delF_delY;
            delF_delZ_ = delF_delZ;
            delF_delR_ = delF_delR;
            delF_delP_ = delF_delP;
            delF_delH_ = delF_delH;

            printf ("%lf %lf %lf %lf %lf %lf %lf\n", f_curr, translation_k[0], translation_k[1], translation_k[2], euler_k[2]*RTOD, euler_k[1]*RTOD, euler_k[0]*RTOD);
        }

        translation = translation_k;
        euler = euler_k;
        return index;
    }

    /**
     * This function calculates the Cramer Rao lower bound of the covariance matrix
     * using the Fisher Information matrix.
     */
    Eigen::Matrix4d Calibration::calculate_covariance_matrix (Eigen::Vector3d translation, Eigen::Vector3d euler)
    {
        //Calculate the Fisher Information matrix
        Eigen::Matrix4d fisher_info_mat;
        fisher_info_mat.setZero();

        //get P at x
        Histogram hist = get_histogram (translation, euler);
        Probability P = get_probability_MLE (hist);

        Eigen::Vector3d translation_tmp;
        Eigen::Vector3d euler_tmp;

        //Get dLnP_dtx
        double h_t = 0.0001;
        translation_tmp[0] = translation[0] + h_t;
        translation_tmp[1] = translation[1];
        translation_tmp[2] = translation[2];
        euler_tmp[2] = euler[2];
        euler_tmp[1] = euler[1];
        euler_tmp[0] = euler[0];
        Histogram hist_plus_dtx = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_dtx = get_probability_MLE (hist_plus_dtx);

//        translation_tmp[0] = translation[0] - h_t;
//        translation_tmp[1] = translation[1];
//        translation_tmp[2] = translation[2];
//        euler_tmp[2] = euler[2];
//        euler_tmp[1] = euler[1];
//        euler_tmp[0] = euler[0];
//        Probability P_minus_dtx = get_probability_MLE (translation_tmp, euler_tmp);

        //Get dLnP_dty
        translation_tmp[0] = translation[0];
        translation_tmp[1] = translation[1] + h_t;
        translation_tmp[2] = translation[2];
        Histogram hist_plus_dty = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_dty = get_probability_MLE (hist_plus_dty);
        
        //Get dLnP_dtz
        translation_tmp[0] = translation[0];
        translation_tmp[1] = translation[1];
        translation_tmp[2] = translation[2] + h_t;
        Histogram hist_plus_dtz = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_dtz = get_probability_MLE (hist_plus_dtz);
        
        //Get dLnP_drx
        double h_r = 0.001*DTOR;
        translation_tmp[0] = translation[0];
        translation_tmp[1] = translation[1];
        translation_tmp[2] = translation[2];
        euler_tmp[2] = euler[2] + h_r;
        euler_tmp[1] = euler[1];
        euler_tmp[0] = euler[0];
        Histogram hist_plus_dtr = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_drx = get_probability_MLE (hist_plus_dtr);
        
        //Get dLnP_dry
        euler_tmp[2] = euler[2];
        euler_tmp[1] = euler[1] + h_r;
        euler_tmp[0] = euler[0];
        Histogram hist_plus_dtp = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_dry = get_probability_MLE (hist_plus_dtp);
        
        //Get dLnP_drz
        euler_tmp[2] = euler[2];
        euler_tmp[1] = euler[1];
        euler_tmp[0] = euler[0] + h_r;
        Histogram hist_plus_dth = get_histogram (translation_tmp, euler_tmp);
        Probability P_plus_drz = get_probability_MLE (hist_plus_dth);

        //Calculate derivative of log of probability distribution
        cv::Mat dLnP_dtx = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat dLnP_dty = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat dLnP_dtz = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat dLnP_drx = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat dLnP_dry = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);
        cv::Mat dLnP_drz = cv::Mat::zeros (this->m_numBins, this->m_numBins, CV_32FC1);

        for (int i = 0; i < this->m_numBins; i++)
        {
            for (int j = 0; j < this->m_numBins; j++)
            {
                if (P_plus_dtx.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_dtx.at<float>(i, j) = (log (P_plus_dtx.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_t);

                if (P_plus_dty.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_dty.at<float>(i, j) = (log (P_plus_dty.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_t);

                if (P_plus_dtz.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_dtz.at<float>(i, j) = (log (P_plus_dtz.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_t);

                if (P_plus_drx.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_drx.at<float>(i, j) = (log (P_plus_drx.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_r);

                if (P_plus_dry.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_dry.at<float>(i, j) = (log (P_plus_dry.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_r);

                if (P_plus_drz.jointProb.at<float>(i, j) > 0 && P.jointProb.at<float>(i, j) > 0)
                    dLnP_drz.at<float>(i, j) = (log (P_plus_drz.jointProb.at<float>(i, j)) - log (P.jointProb.at<float>(i, j)))/(h_r);
            }
        } 
        
        //Calculate Fisher Information Matrix
        for(int k = 0; k < this->m_numBins; k++) {
          for(int l = 0; l < this->m_numBins; l++) {
              for (int i = 0; i < 6; ++i) {
                  for (int j = i; j < 6; ++j) {
                      fisher_info_mat(i,j) = fisher_info_mat(i,j) + dLnP_dtx.at<float>(k, l)*dLnP_dtx.at<float>(k, l)*P.jointProb.at<float>(k, l)*hist.jointHist.at<float>(k, l);
                  }
              }
          }
        }
        fisher_info_mat = fisher_info_mat.transpose();
        return fisher_info_mat.inverse();
    }
}
