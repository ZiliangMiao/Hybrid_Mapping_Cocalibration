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
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main(int argc, char** argv) {
    ros::init(argc, argv, "icp");
    ros::NodeHandle nh;

    PointCloudT::Ptr cloud_target_input(new PointCloudT);
    PointCloudT::Ptr cloud_source_input(new PointCloudT);
    PointCloudT::Ptr cloud_target_filtered(new PointCloudT); /** source point cloud **/
    PointCloudT::Ptr cloud_source_filtered(new PointCloudT); /** target point cloud **/
    PointCloudT::Ptr cloud_source_initial_trans(new PointCloudT); /** souce cloud with initial rigid transformation **/
    PointCloudT::Ptr cloud_icped(new PointCloudT); /** apply icp result to source point cloud **/

    pcl::StopWatch timeer;
    std::string pkg_path = ros::package::getPath("calibration");
    std::string pose_1_pcd_path = pkg_path + "/data/conferenceF2-P1/outputs/lidDense1.pcd";
    std::string pose_2_pcd_path = pkg_path + "/data/conferenceF2-P2/outputs/lidDense1.pcd";

    /** file loading check **/
    if (pcl::io::loadPCDFile<PointT>(pose_1_pcd_path, *cloud_source_input) == -1) {
        PCL_ERROR("Couldn't read file1 \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud_source_input->size() << " data points from file1" << std::endl;
    if (pcl::io::loadPCDFile<PointT>(pose_2_pcd_path, *cloud_target_input) == -1) {
        PCL_ERROR("Couldn't read file2 \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud_target_input->size() << " data points from file2" << std::endl;

    /** invalid point filter **/
    std::vector<int> mapping_in;
    std::vector<int> mapping_out;
    pcl::removeNaNFromPointCloud(*cloud_target_input, *cloud_target_input, mapping_in);
    pcl::removeNaNFromPointCloud(*cloud_source_input, *cloud_source_input, mapping_out);

    /** pass through filter **/
    pcl::PassThrough<PointT> z_passthrough_filter;
    z_passthrough_filter.setFilterFieldName("z");
    z_passthrough_filter.setFilterLimits(2.3, 5);
    z_passthrough_filter.setNegative(true);
    z_passthrough_filter.setInputCloud(cloud_source_input);
    z_passthrough_filter.filter(*cloud_source_filtered);
    z_passthrough_filter.setInputCloud(cloud_target_input);
    z_passthrough_filter.filter(*cloud_target_filtered);

    /** radius outlier filter **/
//    pcl::RadiusOutlierRemoval <PointT> outrem;
//    outrem.setInputCloud(cloud_target_input);
//    outrem.setRadiusSearch(0.04);
//    outrem.secatltMinNeighborsInRadius(1);
//    outrem.filter(*cloud_target_filtered);
//    outrem.filter(*cloud_source_filtered);

    /** initial rigid transformation **/
    Eigen::Affine3f initial_trans = Eigen::Affine3f::Identity();
    initial_trans.translation() << 0.6, 5.1, 0.0;
    float rx = 0.0, ry = 0.0, rz = +1.45/(float)180;
    Eigen::Matrix3f R;
    R = Eigen::AngleAxisf(rx*M_PI, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(ry*M_PI,  Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(rz*M_PI, Eigen::Vector3f::UnitZ());
    initial_trans.rotate(R);
    cout << "Initial Transformation Matrix: " << endl;
    cout << initial_trans.matrix() << endl;
    pcl::transformPointCloud(*cloud_source_filtered, *cloud_source_initial_trans, initial_trans);

    Eigen::Matrix4f initial_trans_mat = initial_trans.matrix();

//    /** voxel down sampling filter **/
//    pcl::VoxelGrid <PointT> vg;
//    vg.setInputCloud(cloud_source_filtered);
//    vg.setLeafSize(0.01f, 0.01f, 0.01f);
//    vg.filter(*cloud_source_filtered);
//
//
//    /** Normal Calculation **/
//    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
//    ne_src.setInputCloud(cloud_source_filtered);
//    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
//    ne_src.setSearchMethod(tree_src);
//    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
//    ne_src.setRadiusSearch(0.02);
//    ne_src.compute(*cloud_src_normals);
//    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_tgt;
//    ne_tgt.setInputCloud(cloud_target_filtered);
//    pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
//    ne_tgt.setSearchMethod(tree_tgt);
//    pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
//    ne_tgt.setRadiusSearch(0.02);
//    ne_tgt.compute(*cloud_tgt_normals);
//
//
//    /** FPFH Calculation **/
//    pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_src;
//    fpfh_src.setInputCloud(cloud_source_filtered);
//    fpfh_src.setInputNormals(cloud_src_normals);
//    pcl::search::KdTree<PointT>::Ptr tree_src_fpfh (new pcl::search::KdTree<PointT>);
//    fpfh_src.setSearchMethod(tree_src_fpfh);
//    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
//    fpfh_src.setRadiusSearch(0.05);
//    fpfh_src.compute(*fpfhs_src);
//    std::cout << "compute *cloud_src fpfh" << endl;
//
//    pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_tgt;
//    fpfh_tgt.setInputCloud(cloud_target_filtered);
//    fpfh_tgt.setInputNormals(cloud_tgt_normals);
//    pcl::search::KdTree<PointT>::Ptr tree_tgt_fpfh (new pcl::search::KdTree<PointT>);
//    fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
//    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
//    fpfh_tgt.setRadiusSearch(0.05);
//    fpfh_tgt.compute(*fpfhs_tgt);
//    std::cout<<"compute *cloud_tgt fpfh"<<endl;
//
//    /** SAC **/
//    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
//    scia.setInputSource(cloud_source_filtered);
//    scia.setInputTarget(cloud_target_filtered);
//    scia.setSourceFeatures(fpfhs_src);
//    scia.setTargetFeatures(fpfhs_tgt);
//    //scia.setMinSampleDistance(1);
//    //scia.setNumberOfSamples(2);
//    //scia.setCorrespondenceRandomness(20);
//    PointCloudT::Ptr sac_result (new PointCloudT);
//    scia.align(*sac_result, initial_trans_mat);
//    std::cout  <<"sac has converged:"<<scia.hasConverged()<<"  score: "<<scia.getFitnessScore()<<endl;
//    Eigen::Matrix4f sac_trans_mat = scia.getFinalTransformation();
//    cout << "SAC Transformation Matrix: " << endl;
//    cout << sac_trans_mat << endl;

    timeer.reset();

    /** generalized icp **/
//    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
//    gicp.setMaximumIterations(500);    //设置最大迭代次数iterations=true
//    gicp.setMaxCorrespondenceDistance(0.03);
//    gicp.setTransformationEpsilon(1e-10);
//    gicp.setEuclideanFitnessEpsilon(0.1);
//    gicp.setInputCloud(cloud_target_filtered); //设置输入点云
//    gicp.setInputTarget(cloud_source_filtered); //设置目标点云（输入点云进行仿射变换，得到目标点云）
//    gicp.align(*cloud_icped, initial_trans_mat); //匹配后源点云
//    if (gicp.hasConverged()) {
//        cout << "\nGICP Iterations: " << gicp.getMaximumOptimizerIterations() << endl;
//        cout << "\nGICP has converged, score is: " << gicp.getFitnessScore() << std::endl;
//        cout << "\nGICP has converged, Epsilon is: " << gicp.getEuclideanFitnessEpsilon() << std::endl;
//        cout << "\nGICP transformation is \n " << gicp.getFinalTransformation() << std::endl;
//    } else {
//        PCL_ERROR("\nGICP has not converged.\n");
//        return (-1);
//    }
//    std::cout << "GICP run time: " << timeer.getTimeSeconds() << " s" << std::endl;

    /** original icp **/
    pcl::IterativeClosestPoint <PointT, PointT> icp; //创建ICP对象，用于ICP配准
    icp.setMaximumIterations(500);
    icp.setInputSource(cloud_source_filtered); //设置输入点云
    icp.setInputTarget(cloud_target_filtered); //设置目标点云（输入点云进行仿射变换，得到目标点云）
    icp.setMaxCorrespondenceDistance(0.01);
    icp.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
    icp.setEuclideanFitnessEpsilon(1e-15); // 均方误差
    icp.align(*cloud_icped, initial_trans_mat); //匹配后源点云
    if (icp.hasConverged()) {
        cout << "\nICP has converged, score is: " << icp.getFitnessScore() << endl;
        cout << "\nICP has converged, Epsilon is: " << icp.getEuclideanFitnessEpsilon() << endl;
        cout << "\nICP transformation is \n " << icp.getFinalTransformation() << endl;
    } else {
        PCL_ERROR("\nICP has not converged.\n");
        return (-1);
    }
    std::cout << "ICP run time: " << timeer.getTimeSeconds() << " s" << std::endl;

    /** visualization **/
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    int v1(0), v2(1); /** create two view point **/
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    float bckgr_gray_level = 0.0;  /** black **/
    float txt_gray_lvl = 1.0 - bckgr_gray_level;

    /** the color of original target cloud is white **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_aim_color_h(cloud_target_filtered, (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl,
                                                                                (int)255 * txt_gray_lvl);
    viewer.addPointCloud(cloud_target_filtered, cloud_aim_color_h, "cloud_aim_v1", v1);
    viewer.addPointCloud(cloud_target_filtered, cloud_aim_color_h, "cloud_aim_v2", v2);

    /** the color of original source cloud is green **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_in_color_h(cloud_source_filtered, 20, 180, 20);
    viewer.addPointCloud(cloud_source_initial_trans, cloud_in_color_h, "cloud_in_v1", v1);

    /** the color of transformed source cloud with icp result is red **/
    pcl::visualization::PointCloudColorHandlerCustom <PointT> cloud_icped_color_h(cloud_icped, 180, 20, 20);
    viewer.addPointCloud(cloud_icped, cloud_icped_color_h, "cloud_icped_v2", v2);

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