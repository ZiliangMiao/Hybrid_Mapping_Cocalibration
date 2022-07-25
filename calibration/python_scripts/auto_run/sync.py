#!/usr/bin/python3

# python3 -m pip install pyserial

from threading import Timer
from datetime import datetime
import serial

ttl_ser   = serial.Serial('/dev/ttyUSB0')	# 右边板子的串口号, 用到RTS或者DTR引脚, 模拟PPS
# rs232_ser = serial.Serial(port="COM14", baudrate=9600)	# 左边板子的串口号, 用到TXD, 9600波特率, 模拟GPRMC

hCamera = 0
pFrameBuffer = 0
image_output_path = "/home/isee/software/fisheyeSDK/demo/python_demo/data/auto_capture_test"
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
    ttl_ser.setRTS(False)	# -9V, 3.3V, PPS输出高, 如果用DTR引脚就setDTR

def PPS_SetLow():
    ttl_ser.setRTS(True)	# 9V, 0V, PPS输出低

def PPS_Init():
    PPS_SetLow()	# 初始低电平

def gprmc_out():
    output = GPRMC_Simulator().encode(encoding) + b'\r\n'
    print(output)
    ttl_ser.write(output)	# 发出GPRMC信息, 结尾加上\r\n
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print('gprmc_out :', ts, ts[-3:])
    print()

def pps_out():
    PPS_SetHigh()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]	# 精确到ms
    Timer(1 - int(ts[-3:])/1000, print, "").start()	# 每次都进行补偿
    Timer(0.02, PPS_Init).start()   # PPS高电平持续时间20ms, 示波器实测约32ms
    Timer(0.10, gprmc_out).start()  # PPS上升沿后100ms, 发出GPRMC
    Timer(2 - int(ts[-3:])/1000, pps_out).start()  # 每次都进行补偿
    print(ts)

if __name__ == "__main__":
    PPS_Init()
    pps_out()
    
