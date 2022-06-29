﻿#include "gimbal.h"
using namespace std;

typedef struct GimbalSocket {
	int fd;
	struct sockaddr_in addr_my;
	unsigned int addrlen_my;
};

GimbalSocket gimbal_socket;
GimbalSocket m_destsock;

/***** empty construction function *****/
Gimbal::Gimbal() {}
Gimbal::~Gimbal() {}

void Gimbal::ZeroPosition() {
    /** data[3]: horizontal - 0x4b, vertical - 0x4d **/
    unsigned char data[7] = {0xff, 0x01, 0x00, 0x00, 0X00, 0x00, 0x00};
    data[3] = 0x4b;
    data[4] = ((short int)(0* 100)) >> 8;
    data[5] = ((short int)(0* 100)) & 0x00ff;
    data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff;
    SendData(data, 7);
    data[3] = 0x4d;
    data[4] = ((short int)(0* 100)) >> 8;
    data[5] = ((short int)(0* 100)) & 0x00ff;
    data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff;
    SendData(data, 7);
}

void Gimbal::SetResetPosition() {
    ZeroPosition();
}

void Gimbal::GimbalInitialization() {
    SocketInitialization();
    SetResetPosition();
}

int Gimbal::SocketInitialization() {
    gimbal_socket.fd = socket(AF_INET, SOCK_DGRAM, 0);
    m_destsock.fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (gimbal_socket.fd < 0 || m_destsock.fd < 0) {
        perror("create socket error!\n");
        return -1;
    }

    gimbal_socket.addr_my.sin_family = AF_INET;
    gimbal_socket.addr_my.sin_addr.s_addr = inet_addr(MY_IP);
    gimbal_socket.addr_my.sin_port = htons(MY_PORT);
    gimbal_socket.addrlen_my = sizeof(struct sockaddr_in);

    m_destsock.addr_my.sin_family = AF_INET;
    m_destsock.addr_my.sin_addr.s_addr = inet_addr(DEST_IP);
    m_destsock.addr_my.sin_port = htons(DEST_PORT);
    m_destsock.addrlen_my = sizeof(struct sockaddr_in);

    int ret = bind(gimbal_socket.fd, (struct sockaddr *)&gimbal_socket.addr_my, gimbal_socket.addrlen_my);
    if (ret == -1) {
        return -1;
    }
    return 0;
}

void Gimbal::SetRotationMode(int rotation_mode) {
    unsigned char data[7] = {0xff, 0x01, 0x00, 0x00, 0X00, 0x00, 0x00};
    /***** Rotation Speed Control Mode *****/
    /**
     * data[0] - 0xff - static start bit
     * data[1] - 0x01 - default address
     * data[2] - 0x00 - default
     * data[3] - ???? - rotation direction
     * data[4] -  ??  - z-axis (horizontal) rotation speed
     * data[5] -  ??  - x-axis (vertical) rotation speed
     * data[6] - cycle redundancy check (crc) code
    **/

    /***** Rotation Angle Control Mode *****/
    /**
     * data[0] - 0xff - static start bit
     * data[1] - 0x01 - default address
     * data[2] - 0x00 - default
     * data[3] - 0x4b/0x4d - horizontal angle/vertical angle
     * data[4] -  ??  - higher eight bits of the angle (100 times + integer)
     * data[5] -  ??  - lower eight bits of the angle (100 times + integer)
     * data[6] - cycle redundancy check (crc) code
    **/

    switch (rotation_mode) {
        case 1: {
            /** 'a' - horizontal 0 **/
            data[3] = 0x4b;
            data[4] = ((short int)(0 * 100)) >> 8;					  //low 8 bit
            data[5] = ((short int)(0 * 100)) & 0x00ff;				  //high 8 bit
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff; //Check code: crc
            SendData(data, 7);
            break;
        }
        case 2: {
            /** 'd' - horizontal 90 **/
            data[3] = 0x4b;
            data[4] = ((short int)(90 * 100)) >> 8;					  //low 8 bit
            data[5] = ((short int)(90 * 100)) & 0x00ff;				  //high 8 bit
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff; //Check code: crc
            SendData(data, 7);
            break;
        }
        case 3: {
            /** 'w' - vertical 60 **/
            data[3] = 0x4d;
            data[4] = ((short int)(60 * 100)) >> 8;					  //low 8 bit
            data[5] = ((short int)(60 * 100)) & 0x00ff;				  //high 8 bit
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff; //Check code: crc
            SendData(data, 7);
            break;
        }
        case 4: {
            /** 's' - vertical 0 **/
            data[3] = 0x4d;
            data[4] = ((short int)(0 * 100)) >> 8;					  //low 8 bit
            data[5] = ((short int)(0 * 100)) & 0x00ff;				  //high 8 bit
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff; //Check code: crc
            SendData(data, 7);
            break;
        }
        case 5: {
            /** 'x' - vertical -60 **/
            data[3] = 0x4d;
            data[4] = ((short int)(-60 * 100)) >> 8;					  //low 8 bit
            data[5] = ((short int)(-60 * 100)) & 0x00ff;				  //high 8 bit
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff; //Check code: crc
            SendData(data, 7);
            break;
        }
        case 0: {
            /** velocity -> 0 **/
            data[3] = 0x00;
            data[4] = 0x00;
            data[5] = 0;
            data[6] = (data[1] + data[2] + data[3] + data[4] + data[5]) & 0x00ff;
            SendData(data, 7);
            break;
        }
        default:
            return;
    }
}

void Gimbal::SendData(unsigned char *pdata, int data_len) {
    GimbalSocket *destsock = (GimbalSocket *)& m_destsock;
    sendto(destsock->fd, pdata, data_len, 0,
           (struct sockaddr *)&destsock->addr_my, destsock->addrlen_my);
}



	
