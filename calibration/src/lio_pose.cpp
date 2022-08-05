#include <iostream>
/** pcl **/
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>
#include <Eigen/Dense>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Path.h>

using namespace std;

nav_msgs::Path path;

double time_stamp;
double pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w;
void path_callback(const nav_msgs::Path &path_msg) {
    time_stamp = ros::Time::now().toSec();
    ROS_INFO("Time stamp when receiving nav path: %.2f", time_stamp);
    int poses_size = path_msg.poses.size();
    cout << "Size of nav::Path: " << poses_size << endl;
    /** value assignment **/
    pos_x = path_msg.poses[path_msg.poses.size() - 1].pose.position.x;
    pos_y = path_msg.poses[path_msg.poses.size() - 1].pose.position.y;
    pos_z = path_msg.poses[path_msg.poses.size() - 1].pose.position.z;
    ori_x = path_msg.poses[path_msg.poses.size() - 1].pose.orientation.x;
    ori_y = path_msg.poses[path_msg.poses.size() - 1].pose.orientation.y;
    ori_z = path_msg.poses[path_msg.poses.size() - 1].pose.orientation.z;
    ori_w = path_msg.poses[path_msg.poses.size() - 1].pose.orientation.w;

    cout << pos_x << " " << pos_y << " " << pos_z << " " << endl;
    cout << ori_x << " " << ori_y << " " << ori_z << " " << ori_w << endl;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "lio_pose");
    ros::NodeHandle nh;
    time_stamp = ros::Time::now().toSec();
    ros::Subscriber sub = nh.subscribe("/path", 1, path_callback);

    while (ros::ok()) {
        double time_stamp_diff = ros::Time::now().toSec() - time_stamp;
        cout << time_stamp_diff << endl;
        ros::spinOnce();
        if (time_stamp_diff > 5) {
            ros::shutdown();
        }
    }

    if (true) {
        /** save the lio spot transformation matrix **/
        Eigen::Affine3d lio_spot_trans = Eigen::Affine3d::Identity();
        Eigen::Matrix4d lio_spot_trans_mat = Eigen::Matrix4d::Identity();
        lio_spot_trans.translation() << pos_x, pos_y, pos_z;
        Eigen::Quaternion<double> q{ori_w, ori_x, ori_y, ori_z}; /** note: the input of Eigen::Quaternion is w, x, y, z!!! **/
        lio_spot_trans.rotate(q.toRotationMatrix());
        lio_spot_trans_mat = lio_spot_trans.matrix();

        /** save the spot trans matrix by lio **/
        std::ofstream lio_mat_out;
        lio_mat_out.open("/home/godm/Desktop/lio_spot_trans_mat.txt");
        lio_mat_out << lio_spot_trans_mat << endl;
        lio_mat_out.close();
    }

    return 0;
}
