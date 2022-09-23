#include <iclk/parameters.hpp>
#include <iclk/tracker.hpp>
#include <iclk/warp_models.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>     // std::steady_clock::time_point
#include <filesystem> // fs::path
#include <iostream>   // std::cerr
#include <memory>     // std::unique_ptr
#include <optional>
#include <string>

namespace fs = std::filesystem;

constexpr int ENTER = 13;
constexpr int Q = 113;
constexpr int ESC = 27;

fs::path configFileFromCmdLine(int argc, char **argv) {
  if (argc != 2) {
    throw std::runtime_error{
        "Number of argument invalid. 1 argument is expected (config file)."};
  }

  const auto cfg_file = fs::path{argv[1]};
  if (not fs::is_regular_file(cfg_file)) {
    throw std::runtime_error{"Provided path is not a regular file."};
  }
  return cfg_file;
}

template <typename M>
void process(cv::VideoCapture &capture, const Parameters &params) {
  cv::Mat frame, gray;
  const std::string frame_window{"frame"};
  const std::string roi_select_window{"select roi"};
  cv::namedWindow(frame_window);
  cv::namedWindow(roi_select_window);

  auto tracker = IclkTracker<M>{params};
  auto key = int{};
  auto template_roi = cv::Rect{};

  auto last_frame_timestamp =
      std::optional<std::chrono::steady_clock::time_point>{};
  while (key = static_cast<int>(cv::waitKey(10))) {
    capture >> frame;
    if (frame.empty())
      continue;

    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    if (tracker.hasTemplate()) {
      tracker.trackTemplate(gray);
    }
    if (key == ESC or key == Q)
      break;
    if (key == ENTER) {
      template_roi = cv::selectROI(roi_select_window, gray);
      if (template_roi.area() > 0) {
        cv::destroyWindow(roi_select_window);
        tracker.setTemplate(gray, template_roi);
      }
    }
    cv::imshow(frame_window, gray);
  }

  capture.release();
  cv::destroyAllWindows();
}

void trackingLoop(cv::VideoCapture &capture, const Parameters &p) {
  if (p.model == Model::affine)
    process<Affine2D>(capture, p);
  else if (p.model == Model::translation)
    process<Translation2D>(capture, p);
  else if (p.model == Model::sl3)
    process<SL3>(capture, p);
  else if (p.model == Model::sim2)
    process<Sim2>(capture, p);
  else
    throw std::runtime_error{"Model not supported"};
}

auto main(int argc, char **argv) -> int {
  // Read config YAML file from argument list
  const auto params = makeParameters(configFileFromCmdLine(argc, argv));
  auto capture = cv::VideoCapture(-1);
  if (not capture.isOpened()) {
    std::cerr << "- Could not find any video device\n";
    return -1;
  }
  trackingLoop(capture, params);
}
