cmake_minimum_required(VERSION 2.8)
project(UndirectDirectSLAM)

set(CMAKE_CXX_STANDARD 11)

# modify the opencv version according to your PC, for instance,
# version 3.0.0, 3.1.0, ... -> 3.0
# version 4.0.0, 4.1.0, ... -> 4.0
find_package(OpenCV 4.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ORB_KP_Stereo_opencv ORB_KP_Stereo_opencv.cpp)
target_link_libraries(ORB_KP_Stereo_opencv ${OpenCV_LIBS})

add_executable(ORB_KP_opencv ORB_KP_opencv.cpp)
target_link_libraries(ORB_KP_opencv ${OpenCV_LIBS})
