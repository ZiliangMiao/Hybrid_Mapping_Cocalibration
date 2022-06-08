import cv2
import numpy as np

img=cv2.imread('/home/xwy/WS/fisheye_ws/src/Fisheye-LiDAR-Fusion/data_process/data/lycheeHill/imagePressed/grab12.bmp')

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 2, (255, 255, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    0.8, (255,255,255), thickness = 1)
        cv2.imshow("image", img)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img)
    if cv2.waitKey(0)&0xFF==27:
        break
cv2.destroyAllWindows()