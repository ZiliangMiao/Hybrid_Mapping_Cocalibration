#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ia_ransac.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main(int argc, char** argv) {
    ros::init(argc, argv, "fullview_icp");
    ros::NodeHandle nh;

    PointCloudT::Ptr view0_cloud(new PointCloudT);
    PointCloudT::Ptr view25_cloud(new PointCloudT);
    PointCloudT::Ptr view50_cloud(new PointCloudT);

    PointCloudT::Ptr view25_cloud_init_trans(new PointCloudT);
    PointCloudT::Ptr view50_cloud_init_trans(new PointCloudT);

    PointCloudT::Ptr view25_cloud_icp_trans(new PointCloudT);
    PointCloudT::Ptr view50_cloud_icp_trans(new PointCloudT);

    PointCloudT::Ptr global_trans_result(new PointCloudT); /** apply icp result to source point cloud **/

    pcl::StopWatch timer;
    std::string pkg_path = ros::package::getPath("calibration");

    std::string pose_0_pcd_path = pkg_path + "/data/lh3_global/spot0/0/outputs/lidar_outputs/icp_cloud.pcd";
    std::string pose_25_pcd_path = pkg_path + "/data/lh3_global/spot0/25/outputs/lidar_outputs/icp_cloud.pcd";
    std::string pose_50_pcd_path = pkg_path + "/data/lh3_global/spot0/50/outputs/lidar_outputs/icp_cloud.pcd";

    Eigen::Matrix4f icp_trans_mat_50_25;
    Eigen::Matrix4f icp_trans_mat_25_0;

    /** file loading check **/
    pcl::io::loadPCDFile<PointT>(pose_0_pcd_path, *view0_cloud);
    std::cout << "Loaded " << view0_cloud->size() << " data points from view 0 cloud" << std::endl;
    pcl::io::loadPCDFile<PointT>(pose_25_pcd_path, *view25_cloud);
    std::cout << "Loaded " << view25_cloud->size() << " data points from view 25 cloud" << std::endl;
    pcl::io::loadPCDFile<PointT>(pose_50_pcd_path, *view50_cloud);
    std::cout << "Loaded " << view50_cloud->size() << " data points from view 50 cloud" << std::endl;

    /** invalid point filter **/
    std::vector<int> mapping_0;
    std::vector<int> mapping_25;
    std::vector<int> mapping_50;
    pcl::removeNaNFromPointCloud(*view25_cloud, *view25_cloud, mapping_0);
    pcl::removeNaNFromPointCloud(*view25_cloud, *view25_cloud, mapping_25);
    pcl::removeNaNFromPointCloud(*view50_cloud, *view50_cloud, mapping_50);

    /** condition filter **/
    pcl::ConditionOr<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionOr<pcl::PointXYZ>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, -0.4)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("y", pcl::ComparisonOps::LT, -0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::GT, 0.3)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ> ("x", pcl::ComparisonOps::LT, -0.3)));
    pcl::ConditionalRemoval<pcl::PointXYZ> cond_filter;
    cond_filter.setCondition(range_cond);
    cond_filter.setInputCloud(view0_cloud);
    cond_filter.filter(*view0_cloud);
    cond_filter.setInputCloud(view25_cloud);
    cond_filter.filter(*view25_cloud);
    cond_filter.setInputCloud(view50_cloud);
    cond_filter.filter(*view50_cloud);

    /** radius outlier filter **/
    pcl::RadiusOutlierRemoval <PointT> outlier_filter;
    outlier_filter.setRadiusSearch(0.5);
    outlier_filter.setMinNeighborsInRadius(30);
    outlier_filter.setInputCloud(view0_cloud);
    outlier_filter.filter(*view0_cloud);
    outlier_filter.setInputCloud(view25_cloud);
    outlier_filter.filter(*view25_cloud);
    outlier_filter.setInputCloud(view50_cloud);
    outlier_filter.filter(*view50_cloud);

    /** initial rigid transformation from 50 to 25 **/
    Eigen::Affine3f initial_trans_50_25 = Eigen::Affine3f::Identity();
    initial_trans_50_25.translation() << 0.15 * sin(50/(float)180 * M_PI) - 0.15 * sin(25/(float)180 * M_PI),
                                         0.0,
                                         0.15 * cos(50/(float)180 * M_PI) - 0.15 * cos(25/(float)180 * M_PI);
    float rx = 0.0, ry = 25/(float)180, rz = 0.0;
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(rx*M_PI, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(ry*M_PI,  Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(rz*M_PI, Eigen::Vector3f::UnitZ());
    initial_trans_50_25.rotate(R);
    cout << initial_trans_50_25.matrix() << endl;
    Eigen::Matrix4f initial_trans_mat_50_25 = initial_trans_50_25.matrix();

    /** initial rigid transformation from 25 to 0 **/
    Eigen::Affine3f initial_trans_25_0 = Eigen::Affine3f::Identity();
    initial_trans_25_0.translation() << 0.15 * sin(25/(float)180 * M_PI) - 0.15 * sin(0/(float)180 * M_PI),
                                        0.0,
                                        0.15 * cos(25/(float)180 * M_PI) - 0.15 * cos(0/(float)180 * M_PI);
    initial_trans_25_0.rotate(R);
    cout << initial_trans_25_0.matrix() << endl;
    Eigen::Matrix4f initial_trans_mat_25_0 = initial_trans_25_0.matrix();

    /** initial transformation from 50 -> 25 degree **/
    pcl::transformPointCloud(*view50_cloud, *view50_cloud_init_trans, initial_trans_mat_50_25);
    /** initial transformation from 25 -> 0 degree **/
    pcl::transformPointCloud(*view25_cloud, *view25_cloud_init_trans, initial_trans_mat_25_0);



//    /** visualization **/
//    pcl::visualization::PCLVisualizer viewer("ICP demo");
//    int v1(0); /** create two view point **/
//    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
//    float bckgr_gray_level = 0.0;  /** black **/
//    float txt_gray_lvl = 1.0 - bckgr_gray_level;
//
//    /** the color of original target cloud is white **/
//    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(view0_cloud, (int)255 * txt_gray_lvl,
//                                                                                (int)255 * txt_gray_lvl,
//                                                                                (int)255 * txt_gray_lvl);
//    viewer.addPointCloud(view0_cloud, cloud_aim_color_h, "cloud_aim_v1", v1);
//
//    /** the color of original source cloud is green **/
//    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_in_color_h(view25_cloud_init_trans, 20, 180, 20);
//    viewer.addPointCloud(view25_cloud_init_trans, cloud_in_color_h, "cloud_in_v1", v1);
//
//    /** the color of original source cloud is red **/
//    pcl::transformPointCloud(*view50_cloud_init_trans, *view50_cloud_init_trans, initial_trans_mat_25_0);
//    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_dual_color_h(view50_cloud_init_trans, 180, 20, 20);
//    viewer.addPointCloud(view50_cloud_init_trans, cloud_dual_color_h, "cloud_dual_v1", v1);
//
//    /** add text **/
//    //在指定视口viewport=v1添加字符串, 其中"icp_info_1"是添加字符串的ID标志，（10，15）为坐标16为字符大小 后面分别是RGB值
//    viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl,
//                   txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
//
//    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
//
//    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
//    viewer.setSize(1280, 1024);  /** viewer size **/
//    viewer.addCoordinateSystem();
//
//    while (!viewer.wasStopped()) {
//        viewer.spinOnce();
//    }




    /** voxel down sampling filter **/
//    pcl::VoxelGrid <PointT> vg;
//    vg.setInputCloud(cloud_source_filtered);
//    vg.setLeafSize(0.01f, 0.01f, 0.01f);
//    vg.filter(*cloud_source_filtered);

    timer.reset();

    /** generalized icp **/
//    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
//    gicp.setMaximumIterations(500);    //设置最大迭代次数iterations=true
//    gicp.setMaxCorrespondenceDistance(0.03);
//    gicp.setTransformationEpsilon(1e-10);
//    gicp.setEuclideanFitnessEpsilon(0.1);
//    gicp.setInputCloud(cloud_target_filtered); //设置输入点云
//    gicp.setInputTarget(cloud_source_filtered); //设置目标点云（输入点云进行仿射变换，得到目标点云）
//    gicp.align(*local_trans_result, initial_trans_mat); //匹配后源点云
//    if (gicp.hasConverged()) {
//        cout << "\nGICP Iterations: " << gicp.getMaximumOptimizerIterations() << endl;
//        cout << "\nGICP has converged, score is: " << gicp.getFitnessScore() << std::endl;
//        cout << "\nGICP has converged, Epsilon is: " << gicp.getEuclideanFitnessEpsilon() << std::endl;
//        cout << "\nGICP transformation is \n " << gicp.getFinalTransformation() << std::endl;
//    } else {
//        PCL_ERROR("\nGICP has not converged.\n");
//        return (-1);
//    }
//    std::cout << "GICP run time: " << timer.getTimeSeconds() << " s" << std::endl;

    /** original icp **/
    pcl::IterativeClosestPoint <PointT, PointT> icp; //创建ICP对象，用于ICP配准
    icp.setMaximumIterations(500);
    icp.setInputCloud(view50_cloud); //设置输入点云
    icp.setInputTarget(view25_cloud); //设置目标点云（输入点云进行仿射变换，得到目标点云）
    icp.setMaxCorrespondenceDistance(0.1);
    icp.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
    icp.setEuclideanFitnessEpsilon(0.005); // 均方误差
    icp.align(*view50_cloud_icp_trans, initial_trans_mat_50_25); //匹配后源点云
    if (icp.hasConverged()) {
        icp_trans_mat_50_25 = icp.getFinalTransformation();
        cout << "\nICP has converged, score is: " << icp.getFitnessScore() << endl;
        cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "\nICP transformation is \n " << icp_trans_mat_50_25 << endl;
    } else {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }

    pcl::IterativeClosestPoint <PointT, PointT> icp2; //创建ICP对象，用于ICP配准
    icp2.setMaximumIterations(500);
    icp2.setInputCloud(view25_cloud); //设置输入点云
    icp2.setInputTarget(view0_cloud); //设置目标点云（输入点云进行仿射变换，得到目标点云）
    icp2.setMaxCorrespondenceDistance(0.1);
    icp2.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
    icp2.setEuclideanFitnessEpsilon(0.005); // 均方误差
    icp2.align(*view25_cloud_icp_trans, initial_trans_mat_25_0); //匹配后源点云
    if (icp2.hasConverged()) {
        icp_trans_mat_25_0 = icp2.getFinalTransformation();
        cout << "\nICP has converged, score is: " << icp2.getFitnessScore() << endl;
        cout << "\nICP has converged, Epsilon is: " << icp2.getEuclideanFitnessEpsilon() << endl;
        cout << "\nICP transformation is \n " << icp_trans_mat_25_0 << endl;
    } else {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }

    cout << "Initial Trans Mat from 50 to 0: \n" << initial_trans_mat_25_0 * initial_trans_mat_50_25 << endl;
    cout << "ICP Trans Mat from 50 to 0: \n" << icp_trans_mat_25_0 * icp_trans_mat_50_25 << endl;

    std::cout << "ICP run time: " << timer.getTimeSeconds() << " s" << std::endl;

    /** visualization **/
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0), v2(1); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0;  /** black **/
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    /** the color of original target cloud is white **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(view0_cloud, (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl);
    viewer.addPointCloud(view0_cloud, cloud_aim_color_h, "cloud_aim_v1", v1);
    viewer.addPointCloud(view0_cloud, cloud_aim_color_h, "cloud_aim_v2", v2);

    /** the color of initial transformation cloud is green **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> init_trans_color_25(view25_cloud_init_trans, 180, 20, 20);
    viewer.addPointCloud(view25_cloud_init_trans, init_trans_color_25, "init_trans_25", v1);

    pcl::transformPointCloud(*view50_cloud_init_trans, *view50_cloud_init_trans, initial_trans_mat_25_0);
    pcl::visualization::PointCloudColorHandlerCustom <PointT> init_trans_color_50(view50_cloud_init_trans, 20, 180, 20);
    viewer.addPointCloud(view50_cloud_init_trans, init_trans_color_50, "init_trans_50", v1);

    /** the color of icp transformation cloud is red **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> icp_trans_color_25(view25_cloud_icp_trans, 180, 20, 20);
    viewer.addPointCloud(view25_cloud_icp_trans, icp_trans_color_25, "icp_trans_25", v2);

    pcl::transformPointCloud(*view50_cloud_icp_trans, *view50_cloud_icp_trans, icp_trans_mat_25_0);
    pcl::visualization::PointCloudColorHandlerCustom <PointT> icp_trans_color_50(view50_cloud_icp_trans, 20, 180, 20);
    viewer.addPointCloud(view50_cloud_icp_trans, icp_trans_color_50, "icp_trans_50", v2);

    /** add text **/
    //在指定视口viewport=v1添加字符串, 其中"icp_info_1"是添加字符串的ID标志，（10，15）为坐标16为字符大小 后面分别是RGB值
    viewer.addText("White: Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_lvl,
                   txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
    viewer.addText("White: Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl,
                   txt_gray_lvl, "icp_info_2", v2);

    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  /** viewer size **/

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
    return 0;
}