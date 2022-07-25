cmake_minimum_required(VERSION 3.0.2)
project(calibration)

set(MAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")
# set(Open3D_DIR "/usr/local/lib/cmake/Open3D")

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

if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(PCL_INCLUDE_DIRS /usr/include/pcl-1.8)
# set(Open3D_INCLUDE_DIRS /usr/local/include/open3d)
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


# Open3D Related
# find_package(Open3D  REQUIRED)
# include_directories(${Open3D_INCLUDE_DIRS})
# link_directories(${Open3D_LIBRARY_DIRS})
# add_executable(spot_reg src/spot_registration.cpp)
# target_link_libraries(spot_reg ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Open3D_LIBRARIES})
# target_include_directories(spot_reg PUBLIC ${Open3D_INCLUDE_DIRS})


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

add_executable(icp src/icp.cpp)
add_executable(fullview_icp src/fullview_icp.cpp)
add_executable(icpnl src/icpnl.cpp)
add_executable(pairwise src/pairwise_incremental_registration.cpp)

## Add dependencies
add_dependencies(calibration ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(FisheyeProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(LidarProcess ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(rvizPub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ceresMultiScenes ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_dependencies(icp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(fullview_icp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(icpnl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(pairwise ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link libraries
target_link_libraries(LidarProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(FisheyeProcess ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${MLPACK_LIBRARIES})
target_link_libraries(rvizPub ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

target_link_libraries(icp ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(fullview_icp ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(icpnl ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(pairwise ${catkin_LIBRARIES} ${PCL_LIBRARIES})

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




