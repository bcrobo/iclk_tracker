find_package(Catch2 REQUIRED)

add_executable(testImagePyramid
  test_image_pyramid.cpp
)
target_include_directories(testImagePyramid
  PRIVATE
    iclk
)
target_link_libraries(testImagePyramid
  PRIVATE
    iclk
    Catch2::Catch2
)

target_compile_features(testImagePyramid PRIVATE cxx_std_17)
