## Headings
cmake_minimum_required(VERSION 3.0.2)
project(MI)

## Cmake Settings
set(MAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

## Find Package
set(PCL_DIR "/usr/local/share/pcl-1.12")
find_package(OpenMP REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(OpenCV REQUIRED)

## OpenMP Package
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Add C++ Libraries
add_library(Calibration
        src/Calibration.h
        src/Calibration.cpp
)

## Add Executable Files
set(PCL_INCLUDE_DIRS "/usr/local/include/pcl-1.12")
set (SRC
  src/main.cpp 
  src/Calibration.cpp  src/Calibration.h
)
add_executable (MI ${SRC})
target_link_libraries (MI
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
include_directories (
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${SRC_DIR}
)
