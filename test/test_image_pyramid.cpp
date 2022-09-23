#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iclk/image_pyramid.hpp>

#include <opencv2/core.hpp>

ImagePyramidParams makeImagePyramidParams(int levels, float scale, int border) {
  return {levels, scale, border, true};
}

TEST_CASE("Image pyramid") {

  cv::Mat image(480, 640, CV_8UC1);
  cv::randu(image, 0, 256);

  SECTION("One level image pyramid") {
    const auto p = makeImagePyramidParams(1, 1, 0);
    auto pyramid = ImagePyramid{p};
    pyramid.buildPyramid(image);

    const auto &layers = pyramid.getLevelInfo();
    REQUIRE(std::size(layers) == 1);
    REQUIRE(layers.at(0).inv_scale_factor == 1);
    REQUIRE(layers.at(0).image_roi == cv::Rect{0, 0, image.cols, image.rows});

    const auto &pyr = pyramid.getPyramid();
    REQUIRE(std::size(pyr) == 1);
    const auto diff = pyr.at(0) != image;
    REQUIRE(cv::countNonZero(diff) == 0);
  }

  SECTION("Downsampling pyramid with two levels") {
    const auto p = makeImagePyramidParams(2, 2, 2);
    auto pyramid = ImagePyramid{p};
    pyramid.buildPyramid(image);

    const auto &layers = pyramid.getLevelInfo();
    REQUIRE(std::size(layers) == 2);
    REQUIRE(layers.at(0).inv_scale_factor == 1);
    REQUIRE(layers.at(0).image_roi == cv::Rect{2, 2, image.cols, image.rows});
    REQUIRE(layers.at(1).inv_scale_factor == 0.5);
    REQUIRE(layers.at(1).image_roi ==
            cv::Rect{2, 2, image.cols / 2, image.rows / 2});

    const auto &pyr = pyramid.getPyramid();
    REQUIRE(std::size(pyr) == 2);
    REQUIRE(pyr.at(0).rows == image.rows);
    REQUIRE(pyr.at(0).cols == image.cols);
    REQUIRE(pyr.at(1).rows == image.rows / 2);
    REQUIRE(pyr.at(1).cols == image.cols / 2);
  }

  SECTION("Upsampling pyramid with two levels") {
    const auto p = makeImagePyramidParams(2, 0.5, 2);
    auto pyramid = ImagePyramid{p};
    pyramid.buildPyramid(image);

    const auto &layers = pyramid.getLevelInfo();
    REQUIRE(std::size(layers) == 2);
    REQUIRE(layers.at(0).inv_scale_factor == 1);
    REQUIRE(layers.at(0).image_roi == cv::Rect{2, 2, image.cols, image.rows});
    REQUIRE(layers.at(1).inv_scale_factor == 2);
    REQUIRE(layers.at(1).image_roi ==
            cv::Rect{2, 2, image.cols * 2, image.rows * 2});

    const auto &pyr = pyramid.getPyramid();
    REQUIRE(std::size(pyr) == 2);
    REQUIRE(pyr.at(0).rows == image.rows);
    REQUIRE(pyr.at(0).cols == image.cols);
    REQUIRE(pyr.at(1).rows == image.rows * 2);
    REQUIRE(pyr.at(1).cols == image.cols * 2);
  }
}
