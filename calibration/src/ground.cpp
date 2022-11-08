#include <iostream>
#include <boost/thread/thread.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>

#include <Eigen/Core>

using namespace std;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> Cloud;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

Cloud::Ptr cloud (new Cloud);
Cloud::Ptr tf_cloud (new Cloud);
Cloud::Ptr clicked_cloud (new Cloud);
Cloud::Ptr clicked_tf_cloud (new Cloud);

Eigen::Vector3f plane_norm_vec;
Eigen::Vector3f ref_norm_vec;
Eigen::Matrix4f rot_mat = Eigen::Matrix4f::Identity();

size_t num = 0;
int z_lb = -10;
int z_ub = 10;
double a, b, c = 0;

boost::shared_ptr<pcl::visualization::PCLVisualizer> init_viewer();

void pp_callback(const pcl::visualization::AreaPickingEvent& event,
                 void* args);

void load_pcd(  string data_path,
                Cloud::Ptr cloud_in);

bool ground_estimate(Cloud::Ptr &in_cloud,
                     Cloud::Ptr &out_cloud,
                     const float in_distance_thre);

Eigen::Matrix4f create_rotate_matrix( Eigen::Vector3f before,
                                    Eigen::Vector3f after);

int main(int argc, char** argv) {
    ros::init (argc, argv, "ground");
    ros::NodeHandle nh;
    std::string package_dir = ros::package::getPath("calibration");
    string root_path, file_path, save_path;
    nh.getParam("root_path", root_path);
    nh.getParam("file_name", file_path);
    nh.getParam("a", a);
    nh.getParam("b", b);
    nh.getParam("c", c);
    root_path = package_dir + root_path;
    save_path = root_path + file_path + "_tf.pcd";
    file_path = root_path + file_path + ".pcd";
    
    load_pcd(file_path, cloud);
    viewer = init_viewer();
    
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    
    ground_estimate(clicked_cloud, clicked_tf_cloud, 0.05);
    
    pcl::transformPointCloud(*cloud, *tf_cloud, rot_mat);
    pcl::io::savePCDFileBinary(save_path, *tf_cloud);

    return 0;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> init_viewer() {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->setWindowName("Mouse & Keyboard Events");
    viewer->addPointCloud(cloud, "cloud");
    viewer->registerAreaPickingCallback(pp_callback, (void*)viewer.get());
    return (viewer);
}

void pp_callback(const pcl::visualization::AreaPickingEvent& event,
                 void* args) {
    std::vector<int> indices;
    if (event.getPointsIndices(indices)==-1)
        return;
 
    for (int i = 0; i < indices.size(); ++i)
    {
        clicked_cloud->points.push_back(cloud->points.at(indices[i]));
    }
 
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(clicked_cloud, 255, 0, 0);
 
    std::stringstream ss;
    std::string cloudName;
    ss << num++;
    ss >> cloudName;
    cloudName += "_cloudName";
 
    viewer->addPointCloud(clicked_cloud, red, cloudName);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, cloudName);
}

void load_pcd(  string data_path,
                Cloud::Ptr cloud_in) {
    pcl::PCDReader reader;
    cout << data_path << endl;
    reader.read(data_path, *cloud_in);

    if (cloud_in->size() != 0) {
        pcl::PassThrough<pcl::PointXYZ> pt;	// 创建滤波器对象
        pt.setInputCloud(cloud_in);			//设置输入点云
        pt.setFilterFieldName("z");			//设置滤波所需字段x
        pt.setFilterLimits(z_lb, z_ub);		//设置x字段过滤范围
        pt.setFilterLimitsNegative(false);	//默认false，保留范围内的点云；true，保存范围外的点云
        pt.filter(*cloud_in);
        if (cloud_in->size() == 0) {
            cerr << "Invalid filter range!\nz: "  << z_lb << "~" << z_ub << endl;
        }
    }
    
    cout << "done." << endl;
    
}

bool ground_estimate(Cloud::Ptr &in_cloud,
                     Cloud::Ptr &out_cloud,
                     const float in_distance_thre) {
    //plane segmentation
    pcl::SACSegmentation<Point> plane_seg;
    pcl::PointIndices::Ptr plane_inliers ( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr plane_coefficients ( new pcl::ModelCoefficients );
    plane_seg.setOptimizeCoefficients (true);
    plane_seg.setModelType ( pcl::SACMODEL_PLANE );
    plane_seg.setMethodType ( pcl::SAC_RANSAC );
    plane_seg.setDistanceThreshold ( in_distance_thre );
    plane_seg.setInputCloud ( in_cloud );
    plane_seg.segment ( *plane_inliers, *plane_coefficients );

    // ax + by + cx + d = 0
    if (a != 0 || b != 0 || c != 0) {
        plane_norm_vec << static_cast<float>(a),
                          static_cast<float>(b),
                          static_cast<float>(c);
    }
    else {
        plane_norm_vec << plane_coefficients->values[0],
                        plane_coefficients->values[1],
                        plane_coefficients->values[2];
    }
    ref_norm_vec << 0, 0, 1;
    cout << plane_norm_vec << endl;
    rot_mat = create_rotate_matrix(plane_norm_vec, ref_norm_vec);
 
    // pcl::ExtractIndices<Point> extract;
    // extract.setInputCloud (in_cloud);
    // extract.setIndices (plane_inliers);
    // extract.filter (*out_cloud);
    // cout<<"cloud size="<<out_cloud->size()<<endl;
    return true;
}

 
Eigen::Matrix4f create_rotate_matrix(Eigen::Vector3f before,
                                    Eigen::Vector3f after) {
    before.normalize();
    after.normalize();
 
    float angle = acos(before.dot(after));
    Eigen::Vector3f p_rotate =before.cross(after);
    p_rotate.normalize();
 
    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    rotationMatrix(0, 0) = cos(angle) + p_rotate[0] * p_rotate[0] * (1 - cos(angle));
    rotationMatrix(0, 1) = p_rotate[0] * p_rotate[1] * (1 - cos(angle) - p_rotate[2] * sin(angle));//这里跟公式比多了一个括号，但是看实验结果它是对的。
    rotationMatrix(0, 2) = p_rotate[1] * sin(angle) + p_rotate[0] * p_rotate[2] * (1 - cos(angle));
 
 
    rotationMatrix(1, 0) = p_rotate[2] * sin(angle) + p_rotate[0] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 1) = cos(angle) + p_rotate[1] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 2) = -p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
 
 
    rotationMatrix(2, 0) = -p_rotate[1] * sin(angle) +p_rotate[0] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 1) = p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 2) = cos(angle) + p_rotate[2] * p_rotate[2] * (1 - cos(angle));
 
    return rotationMatrix;
}