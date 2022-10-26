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

#include <common_lib.h>

using namespace std;

template <typename PointT>
void getMessage(sensor_msgs::PointCloud2 &msg,
                pcl::PointCloud<PointT> &cloud,
                int msg_size,
                int partition) {
    int cloud_size = cloud.size();
    pcl::PointCloud<PointT> msg_cloud;
    for (int i = (partition * msg_size); i < ((partition + 1) * msg_size) && i < cloud_size; ++i) {
        msg_cloud.points.push_back(cloud.points[i]);
    }
    pcl::toROSMsg(msg_cloud, msg);
    msg.header.frame_id = "livox_frame"; //this has been done in order to be able to visualize our PointCloud2 message on the RViz visualizer
    msg.header.stamp = ros::Time::now();
    msg.header.seq = partition;
}

double uniform_rand(double lowerBndr, double upperBndr) {
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

double gauss_rand(double mean, double sigma) {
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

template <typename PointT>
void process(pcl::PointCloud<PointT> &cloud) {
    float sigma = 0.01;
    for (auto &pt : cloud) {
        pt.x = gauss_rand(pt.x, sigma);
        pt.y = gauss_rand(pt.y, sigma);
        pt.z = gauss_rand(pt.z, sigma);
    }
} 

template <typename PointT>
void broadcast( pcl::PointCloud<PointT> &cloud,
                ros::NodeHandle &nh) {

    double rx = 0, ry = 0, rz = 0, tx = 0, ty = 0, tz = 0; 
    bool save_pcd_en;
    int msg_size, point_lim;
    string save_path;

    nh.getParam("rx", rx);
    nh.getParam("ry", ry);
    nh.getParam("rz", rz);
    nh.getParam("tx", tx);
    nh.getParam("ty", ty);
    nh.getParam("tz", tz);
    nh.getParam("save_pcd_en", save_pcd_en);
    nh.getParam("save_path", save_path);
    nh.getParam("msg_size", msg_size);
    nh.getParam("point_limit", point_lim);

    ros::Publisher orgPub = nh.advertise<sensor_msgs::PointCloud2> ("/livox/lidar", 1e5);

    Eigen::Matrix<float, 6, 1> transform;
    transform << (float)rx, (float)ry, (float)rz, (float)tx, (float)ty, (float)tz;
    Eigen::Matrix4f tf_mat = TransformMat(transform);
    cout << tf_mat << endl;

    sensor_msgs::PointCloud2 msg_cloud;

    ros::Rate loop_rate(50);

    pcl::PointCloud<PointT> cloud_tf;
    pcl::transformPointCloud(cloud, cloud_tf, tf_mat); 
    process(cloud_tf);
    cloud += cloud_tf;
    cout << cloud.size() << endl;
    int size_limit = int(cloud.size() / msg_size) + 1;
    int point_limit = int(point_lim / msg_size) + 1;
    int cnt = 0;
    while (ros::ok()) {
        if (cnt < size_limit && cnt < point_limit) {
            getMessage(msg_cloud, cloud, msg_size, cnt);
            orgPub.publish(msg_cloud);
            cnt++;
        }
        else {
            cout << "done." << endl;
            break;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    if (save_pcd_en) {
        pcl::io::savePCDFileBinary(ros::package::getPath("calibration") + save_path, cloud);
    }
}

int main (int argc, char **argv) {
    ros::init (argc, argv, "rviz_pub");
    ros::NodeHandle nh;

    string data_path, cloud_type;
    nh.getParam("data_path", data_path);
    nh.getParam("type", cloud_type);
    data_path = ros::package::getPath("calibration") + data_path;
    cout << "type: " << cloud_type << endl;

    if (cloud_type == "xyzrgb") {
        typedef pcl::PointXYZRGB PointType;
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path, *cloud);
        broadcast(*cloud, nh);
    }
    else if (cloud_type == "xyzi") {
        typedef pcl::PointXYZI PointType;
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path, *cloud);
        broadcast(*cloud, nh);
    }
    else if (cloud_type == "xyz") {
        typedef pcl::PointXYZ PointType;
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path, *cloud);
        broadcast(*cloud, nh);
    }
    return 0;
}