##### 1.Dependencies required

1.1 OpenCV

1.2 PCL

##### 2.ROS Workspace

2.1 Create a new ros workspace, you can refer to the following link.

[Create a new ros workspace](https://www.cnblogs.com/huangjianxin/p/6347416.html)

2.2 Change your directory to catkin_ws/src, and download the code.

```
cd src/
```

```
git clone git@github.com:SylviaFSky/data_process.git
```

if you don't have git, then

```
conda install git
```

or

```
pip3 install git 
```

2.3 Change the directory to catkin_ws, and Compile the workspace

```
cd ..
```

```
catkin_make
```

```
source devel/setup.bash
```

##### 3.Data

Download the data from baiduYunpan.

链接: https://pan.baidu.com/s/1r4pwjDo8rzHeC4aBVAKR5g?pwd=ct5a 提取码: ct5a 复制这段内容后打开百度网盘手机App，操作更方便哦

Download the data and put them in catkin_ws/src/data_process.

##### 4.Run the projection node

4.1 Setup a terminal and launch roscore

```
roscore
```

4.2 Run the lidar projection node

```
rosrun data_process readLidar lidar.bag 0
```

Before run the node, the path in the readLidar.cpp file may need to be changed.

4.3 Run the image projection node

```
rosrun data_process readImage Image/grab5.bmp 0 1 0
```

Before run the node, the path in the readImage.cpp file may need to be changed.
