#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

msg = """

-------------------------help--------------------------
Welcome to ISEE RECONSTUCTION TORCH!                   |
=========================                              |
                                                       |
Moving around:                                         |
   u    i    o    p                                    |
   j    k    l                                         |
   m    ,    .                                         |
q - mode1: horizontal rotation angle
w - mode2: vertical rotation angle
u : Rotate left forward                                |
i : Rotate forward                                     |
o : Rotate right forward                               |
j : Rotate left                                        |
k : stop                                               |
l : Rotate right                                       |
m : Rotate left backward                               |
, : Rotate backward                                    |
. : Rotate right backward                              |
p : turn to deault position                            |
                                                       |
CTRL-C to quit                                         |
--------------------------help--------------------------

"""
# Rotation parameters
rotation_mode = 0
h_angle = 0  # horizontal target rotation angle
v_angle = 0  # vertical target rotation angle
h_speed = 0.5  # horizontal rotation speed
v_speed = 0.5  # vertical rotation speed
h_scale = 1  # horizontal rotation speed scale
v_scale = 1  # vertical rotation speed scale
# Input keys dictionaries
modeBindings = {
        'q': (1),
        'w': (2),
        'i': (8),
        'o': (9),
        'j': (4),
        'l': (6),
        'u': (7),
        'k': (5),
        '.': (3),
        'p': (10),
        ';': (12),
        '/': (13),
        '[': (14),
        ']': (15),
        '\'': (16),
}

speedBindings = {
        '1': (0.9, 0.9),  # horizontal, vertical --
        '2': (0.9, 1.0),  # horizontal, vertical -|
        '3': (1.0, 0.9),  # horizontal, vertical |-
        '4': (1.1, 1.0),  # horizontal, vertical +|
        '5': (1.0, 1.1),  # horizontal, vertical |+
        '6': (1.1, 1.1),  # horizontal, vertical ++
}

def getKey():
    # sys.stdin表示标准化输入
    # termios.tcgetattr(fd)返回一个包含文件描述符fd的tty属性的列表
    property_list = termios.tcgetattr(sys.stdin)
    # tty.setraw(fd, when=termios.TCSAFLUSH)将文件描述符fd的模式更改为raw。如果when被省略，则默认为termios.TCSAFLUSH，并传递给termios.tcsetattr()
    tty.setraw(sys.stdin.fileno())
    # 第一个参数是需要监听可读的套接字, 第二个是需要监听可写的套接字, 第三个是需要监听异常的套接字, 第四个是时间限制设置
    # 如果监听的套接字满足了可读可写条件, 那么所返回的can_read 或 can_write就会有值, 然后就可以利用这些返回值进行后续操作
    can_read, _, _ = select.select([sys.stdin], [], [], 0.1)
    if can_read:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, property_list)
    return key

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('pan_keyboard_op_node')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    try:
        print(msg)
        while 1:SS
            key = getKey()
            print("press key: %s\n" % key)
            print("rotation mode: %.2f\n" % modeBindings[key][0])
            if key.strip() != '':
                print("press key: %s, rotation mode: %d\n" % (key, modeBindings[key][0]))
            if (modeBindings[key][0] == 1):
                print("Horizontal Rotation Angle Input:\n")
                h_angle = int(sys.stdin.readline())
            if (modeBindings[key][0] == 2):
                print("Vertical Rotation Angle Input:\n")
                v_angle = int(sys.stdin.readline())
            # change the rotation mode
            if key in modeBindings.keys():
                rotation_mode = moduleBindings[key][0]
                print("rotation mode: %d" % rotation_mode)
            # change the rotation speed
            elif key in speedBindings.keys():
                h_scale = speedBindings[key][0]
                v_scale = speedBindings[key][1]
                print("rotation speed modified:\thorizontal\t%.1f, vertical\t%.1f" % (h_scale, v_scale))
                v_speed = v_speed * v_scale
                h_speed = h_speed * h_scale
            # rotation stop
            elif key == ' ' or key == 'k':
                control_speed = 0
            # program stop
            elif key == '\x03':  # ctrl+c
                break

            # 创建并发布twist消息
            twist = Twist()
            twist.linear.x = rotation_mode
            twist.linear.y = v_angle
            twist.linear.z = h_angle
            twist.angular.x = 0  # empty
            twist.angular.y = v_speed
            twist.angular.z = h_speed

            pub.publish(twist)

    except:
        print("Invalid Input!")

    finally:
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        pub.publish(twist)

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
