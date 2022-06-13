cmake_minimum_required(VERSION 3.0.2)
project(data_process)

set(MAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")


set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")

find_package(mlpack REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
  rosbag
  pcl_ros
  std_msgs
)

set(PCL_INCLUDE_DIRS /usr/include/pcl-1.8)
message(${PCL_LIBRARIES})

catkin_package(
  CATKIN_DEPENDS roscpp rosmsg rospy
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${MLPACK_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(imageProcess
  include/imageProcess.h
  src/imageProcess.cpp
)
add_library(lidarProcess
  include/lidarProcess.h
  src/lidarProcess.cpp
)
add_library(MeanShift
  include/MeanShift.h
  src/MeanShift.cpp
)
add_library(ceresMultiScenes
  src/ceresMultiScenes.cpp
)
add_library(calibOpt
  src/calibOpt.cpp
)
add_library(visualization
  src/visualization.cpp
)

## Add executable cpp files
add_executable(mainNode src/main.cpp)
add_executable(rvizPub src/rvizPub.cpp)

## Add dependencies
add_dependencies(mainNode ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(imageProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(lidarProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(MeanShift ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rvizPub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ceresMultiScenes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(calibOpt ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(visualization ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link libraries
target_link_libraries(MeanShift ${catkin_LIBRARIES})
target_link_libraries(lidarProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(imageProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${MLPACK_LIBRARIES})
target_link_libraries(rvizPub ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

target_link_libraries(mainNode
  imageProcess
  lidarProcess
  ceresMultiScenes
  calibOpt
  visualization
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${MLPACK_LIBRARIES}
)

target_link_libraries(ceresMultiScenes
  imageProcess
  lidarProcess
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

target_link_libraries(calibOpt
  imageProcess
  lidarProcess
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)
