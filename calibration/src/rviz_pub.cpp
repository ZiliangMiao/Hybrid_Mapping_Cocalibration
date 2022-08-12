// basic
#include <string>
#include <sstream>
// ros 
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>
// eigen
#include <Eigen/Core>
// pcl library
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

using namespace std;

template <typename T>
Eigen::Matrix<T, 4, 4> TransformMat(Eigen::Matrix<T, 6, 1> &ext_){
    /***** R = Rx * Ry * Rz *****/
    Eigen::Matrix<T, 3, 3> R;
    R = Eigen::AngleAxis<T>(ext_(2), Eigen::Matrix<T, 3, 1>::UnitZ())
        * Eigen::AngleAxis<T>(ext_(1), Eigen::Matrix<T, 3, 1>::UnitY())
        * Eigen::AngleAxis<T>(ext_(0), Eigen::Matrix<T, 3, 1>::UnitX());

    Eigen::Matrix<T, 4, 4> T_mat;
    T_mat << R(0,0), R(0,1), R(0,2), ext_(3),
        R(1,0), R(1,1), R(1,2), ext_(4),
        R(2,0), R(2,1), R(2,2), ext_(5),
        T(0.0), T(0.0), T(0.0), T(1.0);
    return T_mat;
}

void getMessage(sensor_msgs::PointCloud2 &msg,
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                size_t msg_size,
                size_t partition) {
    size_t cloud_size = cloud->points.size();
    pcl::PointCloud<pcl::PointXYZI>::Ptr msg_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = (partition * msg_size); i < ((partition + 1) * msg_size) && i < cloud_size; ++i) {
        msg_cloud->points.push_back(cloud->points[i]);
    }
    pcl::toROSMsg(*msg_cloud, msg);
    msg.header.frame_id = "livox_frame"; //this has been done in order to be able to visualize our PointCloud2 message on the RViz visualizer
    msg.header.stamp = ros::Time::now();
    msg.header.seq = partition;
}

int main (int argc, char **argv) {
    ros::init (argc, argv, "rviz_pub");
    ros::NodeHandle nh;
    double rx = 0, ry = 0, rz = 0, tx = 0, ty = 0, tz = 0; 
    bool mono_color;

    std::string currPkgDir = ros::package::getPath("calibration");
    std::string data_path;

    nh.getParam("data_path", data_path);
    nh.getParam("rx", rx);
    nh.getParam("ry", ry);
    nh.getParam("rz", rz);
    nh.getParam("tx", tx);
    nh.getParam("ty", ty);
    nh.getParam("tz", tz);
    nh.getParam("mono_color", mono_color);
    data_path = currPkgDir + data_path;

    ros::Publisher orgPub = nh.advertise<sensor_msgs::PointCloud2> ("/livox/lidar", 1e5);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::io::loadPCDFile(data_path, *cloud);

    Eigen::Matrix<float, 6, 1> transform;
    transform << (float)rx, (float)ry, (float)rz, (float)tx, (float)ty, (float)tz;
    Eigen::Matrix4f tf_mat = TransformMat(transform);
    cout << tf_mat << endl;
    pcl::transformPointCloud(*cloud, *cloud, tf_mat); 

    if (mono_color) {
        for (auto &pt : cloud->points) {
        pt.intensity = (cloud->points.size() > 2e7) ? 200 : 40;
        }
    }

    sensor_msgs::PointCloud2 msg;

    ros::Rate loop_rate(50);
    size_t msg_size = 1e5;
    size_t limit = int(cloud->points.size() / msg_size) + 1;
    size_t cnt = 0;
    while (ros::ok()) {
        // fltPub.publish(fltMsg);
        if (cnt < limit) {
            getMessage(msg, cloud, msg_size, cnt);
            orgPub.publish(msg);
            cnt++;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}