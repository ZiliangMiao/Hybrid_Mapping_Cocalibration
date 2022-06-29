#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <boost/asio.hpp>
#include <geometry_msgs/Twist.h>
#include "gimbal.h"

/***** callback function of the cmd control instructions *****/
//void CmdModeCallback(const geometry_msgs::Twist& msg) {
//    if((msg.linear.x!=0) || (msg.angular.z!=0)) {
//        rotation_mode = msg.linear.x ;
//    }
//}

int main(int argc, char **argv) {
    Gimbal gimbal;
    gimbal.GimbalInitialization();
    ros::init(argc, argv, "gimbal_control");
    ros::NodeHandle nh("~");
//    ros::Subscriber sub = nh.subscribe("cmd_instruction", 100, CmdModeCallback);
    int rotation_mode = 0;
    nh.getParam("rotation_mode", rotation_mode);
    std::cout << "Gimbal Control Mode: " << rotation_mode << std::endl;
    gimbal.SetRotationMode(rotation_mode);

    /** loop **/
//    ros::Rate loop_rate(100); //10ms
//	while (ros::ok()) {
//		ros::spinOnce();
//        gimbal.SetRotationMode(rotation_mode);
//        usleep(10);
//        loop_rate.sleep();
//	}
	return 0;
}







    
