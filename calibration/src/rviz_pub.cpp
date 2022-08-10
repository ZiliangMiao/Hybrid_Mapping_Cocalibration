// basic
#include <string>
#include <sstream>
// ros 
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>
// pcl librarya
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
using namespace std;

void getMessage(sensor_msgs::PointCloud2 &msg,
                pcl::PointCloud<pcl::PointXYZI> &cloud,
                size_t msg_size,
                size_t partition) {
    size_t cloud_size = cloud.points.size();
    pcl::PointCloud<pcl::PointXYZI>::Ptr msg_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (size_t i = (partition * msg_size); i < ((partition + 1) * msg_size) && i < cloud_size; ++i) {
        if (cloud.points[i].z > -0.5 && cloud.points[i].z < 3) {
            msg_cloud->points.push_back(cloud.points[i]);
        }
    }
    pcl::toROSMsg(*msg_cloud, msg);
    msg.header.frame_id = "livox_frame"; //this has been done in order to be able to visualize our PointCloud2 message on the RViz visualizer
    msg.header.stamp = ros::Time::now();
    msg.header.seq = partition;
}

int main (int argc, char **argv) {
    ros::init (argc, argv, "rviz_pub");
    ros::NodeHandle nh;

    std::string currPkgDir = ros::package::getPath("calibration");
    std::string data_path;

    nh.getParam("data_path", data_path);
    data_path = currPkgDir + data_path;

    ros::Publisher orgPub = nh.advertise<sensor_msgs::PointCloud2> ("/livox/lidar", 1e5);
    // ros::Publisher fltPub = nh.advertise<sensor_msgs::PointCloud2> ("rvizFltTopic", 1);
    pcl::PointCloud<pcl::PointXYZI> orgCloud;
    // pcl::PointCloud<pcl::PointXYZI> fltCloud;

    pcl::io::loadPCDFile (data_path, orgCloud);

    sensor_msgs::PointCloud2 orgMsg;
    // sensor_msgs::PointCloud2 fltMsg;
    // pcl::toROSMsg(orgCloud, orgMsg);
    // pcl::toROSMsg(fltCloud, fltMsg);

    // fltMsg.header.frame_id = "flt";

    ros::Rate loop_rate(50);
    size_t msg_size = 1e5;
    size_t limit = int(orgCloud.points.size() / msg_size) + 1;
    size_t cnt = 0;
    while (ros::ok()) {
        // fltPub.publish(fltMsg);
        if (cnt < limit) {
            getMessage(orgMsg, orgCloud, msg_size, cnt);
            orgPub.publish(orgMsg);
            cnt++;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}