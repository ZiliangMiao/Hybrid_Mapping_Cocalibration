#!/usr/bin/env python
# -*- coding: utf-8 -*-

from locale import atoi
import time
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

control_mode_msg = [
    "",
    "",
    "",
    "",
    "vertical -50 degrees",
    "vertical -25 degrees",
    "vertical 0 degree",
    "vertical 25 degrees",
    "vertical 50 degrees",
    ""
]

def GetKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('gimbal_publisher')
    pub = rospy.Publisher('/cmd_instruction', Twist, queue_size=5)
    control_mode = atoi(sys.argv[1])
    rospy.loginfo('Get rotation_mode = %d', control_mode)
    # control_mode = int(sys.argv[1])
    time_interval = 60
    print("%s, time_interval = %d"  % (control_mode_msg[control_mode], time_interval))
    start_time = time.time()
    while(1):
        exit_key = GetKey()
        # create and publish "twist" msg, which contains the control mode
        twist = Twist()
        twist.linear.x = control_mode
        pub.publish(twist)
        if (exit_key == '\x03' or time.time() - start_time > time_interval):
            print("Process Terminates!")
            break

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)