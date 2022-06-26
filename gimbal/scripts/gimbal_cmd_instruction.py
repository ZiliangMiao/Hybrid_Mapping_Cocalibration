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
        w                                              |
   a    s    d        k                                |
        x                                              |
a : horizontal 0 degree                                |
d : horizontal 90 degrees                              |
w : vertical 60 degrees                                |
s : vertical 0 degree                                  |
x : vertical -60 degrees                               |
k : stop                                               |
                                                       |
CTRL-C to quit                                         |
--------------------------help--------------------------

"""

control_mode_dictionary = {
        'a':1,
        'd':2,
        'w':3,
        's':4,
        'x':5,
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
    try:
        print(msg)
        while(1):
            key = GetKey()
            if key in control_mode_dictionary.keys():
                control_mode = control_mode_dictionary[key]
                print("control_mode = %d"  % control_mode)
            # stop key
            elif key == ' ' or key == 'k' :
                control_mode = 0
            elif (key == '\x03'):
                break
            # create and publish "twist" msg, which contains the control mode
            twist = Twist()
            twist.linear.x = control_mode
            pub.publish(twist)
    except:
        print(msg)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
