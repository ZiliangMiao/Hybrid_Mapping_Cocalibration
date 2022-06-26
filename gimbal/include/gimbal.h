#ifndef HOLDER_H
#define HOLDER_H

#include <string.h>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MY_IP "192.168.8.156" /** mask: 255.255.255.0 **/
#define MY_PORT 6666
#define DEST_IP "192.168.8.200"
#define DEST_PORT 6666

/*****
     Pelco-D Format
     data[7] = {0xff, 0x00, 0x00, 0x00, 0X00, 0x00, 0x00};

     data[0] = 0xff;														  //起始位
     data[1] = 0x01;														  //云台地址
     data[2] = 0x00;														  //Pelco-D 协议中。字节二必须为0
     data[3] = 0x00;														  //指令类型（类似于寄存器地址）   0 ： 停止云台转动
     data[4] = 0x00;														  //具体指令内容1  高8位
     data[5] = 0x00;														  //具体指令内容2  低8位
     data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff;    //校验码 = （地址 + Data1 + Data2 + Data3 + Data4) & 0xff
*****/

class Gimbal {
public:
    /***** functions *****/
    Gimbal();   //构造函数
    ~Gimbal();  //析构函数
    void SetResetPosition(); //设置复位位置
    void SetRotationMode(int rotate_angle); //设置旋转数据                                                               //设置旋转数据
    void ZeroPosition();
    void SendData(unsigned char *pdata, int data_len);                                                              //发送一次数据
    int SocketInitialization();                                                                                                 //初始化网络节点
    void GimbalInitialization();                                                                                                //初始化云台
};

#endif
