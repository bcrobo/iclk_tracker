cmake_minimum_required(VERSION 3.19)
project(Iclk)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED core imgproc)
find_package(yaml-cpp REQUIRED)

# Library
add_library(iclk
  SHARED
    src/image_pyramid.cpp
    src/timer.cpp
    src/parameters.cpp
)

target_link_libraries(iclk
  PUBLIC
    Eigen3::Eigen
    opencv_core
  PRIVATE
    opencv_imgproc
    ${YAML_CPP_LIBRARIES}
)

target_include_directories(iclk
  PUBLIC
    include
)

target_compile_features(iclk PRIVATE cxx_std_17)

# Main
add_executable(iclkTracker
  src/main.cpp
)
target_include_directories(iclkTracker
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(iclkTracker
  iclk
  opencv_core
  opencv_imgproc
  opencv_highgui
  opencv_videoio
)

target_compile_features(iclkTracker PRIVATE cxx_std_17)

add_subdirectory(test)

