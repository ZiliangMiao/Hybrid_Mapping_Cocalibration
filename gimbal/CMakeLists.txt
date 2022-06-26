cmake_minimum_required (VERSION 3.5.1)                       #version
project (gimbal)                                        #project name

SET(CMAKE_BUILD_TYPE "Release")
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  tf
)

catkin_package(
   INCLUDE_DIRS include
)

add_compile_options (-std=c++11 -Wall)                   #compile dependency
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)       #output path of executable file
aux_source_directory (src SRC_LIST)                          #include .cpp file with SRC_LIST
include_directories (include ${catkin_INCLUDE_DIRS})         #include .h file

add_executable (gimbal ${SRC_LIST})                     #generate executable file
target_link_libraries(gimbal -lpthread -luuid ${catkin_LIBRARIES})
