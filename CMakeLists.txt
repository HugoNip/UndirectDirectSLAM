cmake_minimum_required(VERSION 2.8)
project(UndirectDirectSLAM)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

# OpenCV
# modify the opencv version according to your PC, for instance,
# version 3.0.0, 3.1.0, ... -> 3.0
# version 4.0.0, 4.1.0, ... -> 4.0
find_package(OpenCV 4.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(ORB_KP_Stereo_opencv ORB_KP_Stereo_opencv.cpp)
target_link_libraries(ORB_KP_Stereo_opencv ${OpenCV_LIBS})

add_executable(ORB_KP_opencv ORB_KP_opencv.cpp)
target_link_libraries(ORB_KP_opencv ${OpenCV_LIBS})

add_executable(ORB_KP_function ORB_KP_function.cpp)
target_link_libraries(ORB_KP_function ${OpenCV_LIBS})

add_executable(LK_KP LK_KP.cpp)
target_link_libraries(LK_KP ${OpenCV_LIBS})

add_executable(directSLAM directSLAM.cpp)
target_link_libraries(directSLAM ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
