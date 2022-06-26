cmake_minimum_required(VERSION 3.0.2)
project(calibration)

set(MAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

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
add_library(FisheyeProcess
        include/FisheyeProcess.h
        src/FisheyeProcess.cpp
)
add_library(LidarProcess
        include/LidarProcess.h
        src/LidarProcess.cpp
)
add_library(ceresMultiScenes
  src/ceresMultiScenes.cpp
)

## Add executable cpp files
add_executable(calibration src/calibration.cpp)
add_executable(rvizPub src/rvizPub.cpp)

## Add dependencies
add_dependencies(calibration ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(FisheyeProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(LidarProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rvizPub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ceresMultiScenes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link libraries
target_link_libraries(LidarProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(FisheyeProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${MLPACK_LIBRARIES})
target_link_libraries(rvizPub ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

target_link_libraries(calibration
  FisheyeProcess
  LidarProcess
  ceresMultiScenes
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${MLPACK_LIBRARIES}
)

target_link_libraries(ceresMultiScenes
  FisheyeProcess
  LidarProcess
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)

