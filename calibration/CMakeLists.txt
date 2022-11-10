## Headings
cmake_minimum_required(VERSION 3.0.2)
project(calibration)

## Cmake Settings
set(MAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

## Find Package
set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")
find_package(mlpack REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
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

## OpenMP Package
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Set Include Directories
# set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.12)
set(PCL_INCLUDE_DIRS /usr/include/pcl-1.8)
message(${PCL_LIBRARIES})
catkin_package(
  CATKIN_DEPENDS roscpp rosmsg rospy
)
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${MLPACK_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

## Add C++ Libraries
add_library(lidar_process
        include/lidar_process.h
        src/lidar_process.cpp
)
add_library(fisheye_process
        include/fisheye_process.h
        src/fisheye_process.cpp
)
add_library(optimization
        include/optimization.h
        src/optimization.cpp
)

## Add Executable Files
add_executable(main src/main.cpp)
add_executable(rviz_pub src/rviz_pub.cpp)
add_executable(lio_pose src/lio_pose.cpp)
add_executable(segment src/segment.cpp)
add_executable(ground src/ground.cpp)

## Add Dependencies
add_dependencies(lidar_process ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(fisheye_process ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(optimization ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(main ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rviz_pub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(lio_pose ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(segment ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ground ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


## Link Libraries
target_link_libraries(lidar_process ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(fisheye_process ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${MLPACK_LIBRARIES})
target_link_libraries(optimization
  fisheye_process
  lidar_process
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
)
target_link_libraries(main
  fisheye_process
  lidar_process
  optimization
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${MLPACK_LIBRARIES}
)
target_link_libraries(rviz_pub ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(lio_pose ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(segment ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(ground ${catkin_LIBRARIES} ${PCL_LIBRARIES})