#!/usr/bin/python3
# -*- coding: utf-8 -*-

# python3 -m pip install pyserial

from threading import Timer
from datetime import datetime
import serial
import rospy

ttl_ser   = serial.Serial('/dev/ttyUSB0')
# rs232_ser = serial.Serial(port="COM14", baudrate=9600)
encoding = "ascii"

def GPRMC_Simulator():
    # reference:
    # https://blog.csdn.net/xuw_xy/article/details/120365050
    # https://blog.csdn.net/Quietly_water/article/details/49001725
    # 
    # Example:
    # https://gitlab.com/nmeasim/nmeasim/-/blob/master/nmeasim/models.py
    # >>> from datetime import datetime, timedelta, timezone
    # >>> from nmeasim.models import GpsReceiver
    # >>> gps = GpsReceiver(
    # ...     date_time=datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc),
    # ...     output=('RMC',)
    # ... )
    # >>> for i in range(3):
    # ...     gps.date_time += timedelta(seconds=1)
    # ...     gps.get_output()
    # ... 
    # ['$GPRMC,123457.000,A,0000.000,N,00000.000,E,0.0,0.0,010120,,,A*6A']
    # ['$GPRMC,123458.000,A,0000.000,N,00000.000,E,0.0,0.0,010120,,,A*65']
    # ['$GPRMC,123459.000,A,0000.000,N,00000.000,E,0.0,0.0,010120,,,A*64']

    __utc = datetime.now().astimezone()
    result = __utc.strftime("%H%M%S")
    fractional = __utc.strftime("%f")[:3]
    __nmea_time = result if not fractional else ".".join([result, fractional])
    __fmt_time = __utc.strftime("%d%m%y") if __utc is not None else ""
    parts = [
            "GPRMC",
            __nmea_time,
            "A,0000.000,N,00000.000,E,0.0,0.0",
            __fmt_time,
            "",
            "",
            "A",
        ]
    sentence = (",".join(parts))
    sentence_b = sentence.encode(encoding=encoding)
    crc = sentence_b[0]
    for ch in sentence_b[1:]:
        crc = crc ^ ch
    suffix = str(hex(crc))[2:]
    return "$"+ sentence + "*" +suffix

def PPS_SetHigh():
    ttl_ser.setRTS(False)

def PPS_SetLow():
    ttl_ser.setRTS(True)

def PPS_Init():
    PPS_SetLow()

def GPRMC_Output():
    output = GPRMC_Simulator().encode(encoding) + b'\r\n'
    # print(output)
    ttl_ser.write(output)	# <GPRMC message> + '\r\n'
    # ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    # print('gprmc_out :', ts, ts[-3:])

def Null_Function():
    pass

def PPS_Output():
    PPS_SetHigh()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]	# ms
    Timer(1 - int(ts[-3:])/1000, Null_Function).start()	# correction
    Timer(0.02, PPS_Init).start()   # PPS high for 20ms
    Timer(0.10, GPRMC_Output).start()  # send GPRMC after 100ms
    Timer(2 - int(ts[-3:])/1000, PPS_Output).start()  # correction

if __name__ == "__main__":
    rospy.init_node('sync')
    PPS_Init()
    PPS_Output()
    
