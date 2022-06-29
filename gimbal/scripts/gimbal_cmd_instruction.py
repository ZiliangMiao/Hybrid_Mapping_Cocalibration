#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

msg = """

--------------------------help--------------------------
Welcome to ISEE RECONSTUCTION TORCH!                   |
=========================                              |                                                  |
Moving around:                                         |
   q    w    e                                         |
   a    s    d        k                                |
   z    x    c                                         |
                                                       |
a : horizontal 0 degree                                |
d : horizontal 90 degrees                              |
q : vertical 60 degrees                                |
w : vertical 40 degrees                                |
e : vertical 20 degrees                                |
s : vertical 0 degree                                  |
z : vertical -20 degrees                               |
x : vertical -40 degrees                               |
c : vertical -60 degrees                               |
k : stop                                               |
                                                       |
CTRL-C to quit                                         |
--------------------------help--------------------------

"""

control_mode_dictionary = {
        'a':1,
        'd':2,
        'q':3,
        'w':4,
        'e':5,
        's':6,
        'z':7,
        'x':8,
        'c':9,
        'k':0,
        }

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
    rospy.init_node('gimbal_cmd_instruction')
    pub = rospy.Publisher('/cmd_instruction', Twist, queue_size=5)
    control_mode = 0

    print(msg)
    while(1):
        key = GetKey()
        if key in control_mode_dictionary.keys():
            control_mode = control_mode_dictionary[key]
            print("control_mode = %d"  % control_mode)
            # create and publish "twist" msg, which contains the control mode
            twist = Twist()
            twist.linear.x = control_mode
            pub.publish(twist)
        if (key == '\x03'):
            print("Process Terminates!")
            break

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
