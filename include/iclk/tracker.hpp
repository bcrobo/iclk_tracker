#pragma once

#include <iclk/image_pyramid.hpp>
#include <iclk/parameters.hpp>
#include <iclk/warp_models.hpp>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp> // cv::eigen2cv
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> // cv::Sobel

#include <array>
#include <iostream>
#include <type_traits> // std::is_same_v
#include <vector>

template <typename WarpModel> class IclkTracker {
public:
  using Scalar = typename WarpModel::Scalar;
  using Jacobian = typename WarpModel::Jacobian;
  using Hessian = typename WarpModel::Hessian;
  using ParameterVector = typename WarpModel::ParameterVector;
  using WarpMatrix = typename WarpModel::MatrixType;
  using ImageGradient = Eigen::Matrix<Scalar, 1, 2>;

  static_assert(std::is_same_v<Scalar, double>, "Require double warp model");

  IclkTracker(const Parameters &params)
      : m_initial_warp(WarpModel::Identity()),
        m_warp(WarpModel::Identity()),
	m_pyramid_params{params.pyramid_levels, params.pyramid_scale, 0, true},
        m_params(params),
	m_template_image_pyramid{m_pyramid_params},
        m_current_image_pyramid{m_pyramid_params}
  {}

  void setTemplate(const cv::Mat &image, const cv::Rect &roi) {
    // Convert integer rect to float rect
    m_template_roi = static_cast<Roi>(roi);
    m_warp.setIdentity();
    m_initial_warp.setIdentity();

    // Normalizes intensity between 0 and 1
    cv::Mat normalized_image;
    image.convertTo(normalized_image, CV_32FC1, 1.0 / 255.0);
    m_template_image_pyramid.buildPyramid(normalized_image);

    // Build image pyramid
    const auto &pyramid = m_template_image_pyramid.getPyramid();
    for (auto i = 0u; i < m_pyramid_params.nb_levels; ++i) {
      // Add a potentially shrinked roi for the given level of the pyramid
      const auto &rect = shrinkedRoiAtLevel(i, pyramid.at(i));
      m_template_rois.emplace_back(rect);

      // Show each pyramid level image
      std::stringstream ss;
      ss << i;
      cv::imshow(ss.str(), pyramid.at(i)(rect));
    }

    // Set initial warp
    const auto init_warp_coord =
        m_template_rois.at(m_pyramid_params.nb_levels - 1);
    // clang-format off
    m_initial_warp.translation() =
        (Eigen::Vector2d() << init_warp_coord.x, init_warp_coord.y) .finished();
    // clang-format on

    allocate();
    precalc();
  }

  void trackTemplate(const cv::Mat &image) {

    cv::Mat warpped_image;
    cv::Mat normalized_image;
    // Normalize input image between 0 and 1
    image.convertTo(normalized_image, CV_32FC1, 1.0 / 255.0);

    // Build input image pyramid
    m_current_image_pyramid.buildPyramid(normalized_image);
    const auto &image_pyramid = m_current_image_pyramid.getPyramid();
    const auto &template_pyramid = m_template_image_pyramid.getPyramid();

    for (auto i = 0; i < m_pyramid_params.nb_levels; ++i) {
      const auto pyramid_index = m_pyramid_params.nb_levels - i - 1;
      const auto &current_image = image_pyramid.at(pyramid_index);
      const auto &template_roi = m_template_rois.at(pyramid_index);
      const auto area = template_roi.area();
      const auto &template_image =
          template_pyramid.at(pyramid_index)(template_roi);
      const auto &template_jacobian_vectors =
          m_steepest_descent_images.at(pyramid_index);
      const auto &hessians = m_H.at(pyramid_index);

      // The warpped image using the current transformation
      cv::Mat warpped_image;
      cv::Mat residual_image(template_roi.height, template_roi.width, CV_32F);
      auto iterations{0u};
      while (iterations < m_params.iterations.at(pyramid_index)) {

        warpped_image.setTo(0);
        residual_image.setTo(0);

        auto residuals{0.f};
        ParameterVector p = ParameterVector::Zero();

        const WarpModel warp = m_initial_warp * m_warp;

        cv::Mat W;
        cv::eigen2cv(warp.matrix(), W);
        cv::warpPerspective(current_image, warpped_image, W,
                            cv::Size(template_roi.width, template_roi.height),
                            cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

        residual_image = warpped_image - template_image;
        for (auto v = 0u; v < template_roi.height; ++v) {
          for (auto u = 0u; u < template_roi.width; ++u) {
            const int pixel_idx = v * template_roi.width + u;
            const auto residual = residual_image.at<float>(v, u);
            p += template_jacobian_vectors.at(pixel_idx) *
                 static_cast<double>(residual);
            residuals += std::abs(residual);
          }
        }
        cv::imshow("warpped", warpped_image);
        cv::imshow("template", template_image);
        // while(static_cast<int>(cv::waitKey(5)) != 'q');
        residuals /= area;

        // Warp update
        const ParameterVector dp = m_H.at(pyramid_index).ldlt().solve(p);
        const auto inverse_warp = WarpModel::fromParameters(dp).inverse();
        m_warp = m_warp * inverse_warp;

        if (dp.norm() < 1e-4) {
          break;
        }
        ++iterations;
      }

      // If we are not at the lowest level of the pyramid (pyramid_index != 0)
      // and the pyramid scale factor is different of 1 we scale the warp for
      // the bottom pyramid level
      if (pyramid_index and m_pyramid_params.scale_factor != 1) {
        m_warp.scaleWarp(m_pyramid_params.scale_factor);
      }
    }

    displayTrackedWarp(image, m_initial_warp * m_warp, m_template_rois.front());
  }

  const ImagePyramid &getImagePyramid() const { return m_current_image_pyramid; }
  ImagePyramid &getImagePyramid() { return m_current_image_pyramid; }

  const WarpModel &getWarp() const { return m_warp; }

  bool hasTemplate() const { return m_template_roi.area() != 0; }

  std::optional<float> interpolateBilinear(const cv::Mat &current,
                                           const WarpModel &W, int u,
                                           int v) const {
    using ImageVec = Eigen::Matrix<float, 3, 1>;

    // 1. Apply warp transformation
    const auto uv1 = (ImageVec() << u, v, 1).finished();
    ImageVec uvw = W.matrix().template cast<float>() * uv1;
    uvw /= uvw(2);

    // 2. Check image boundaries
    const auto p =
        cv::Point{static_cast<int>(uvw(0)), static_cast<int>(uvw(1))};
    const auto data_points =
        std::array<cv::Point, 4>{cv::Point{p.x - 1, p.y - 1},  // top left
                                 cv::Point{p.x + 1, p.y - 1},  // top right
                                 cv::Point{p.x + 1, p.y + 1},  // bottom right
                                 cv::Point{p.x - 1, p.y + 1}}; // bottom left

    auto neighbors = std::vector<cv::Point>{};
    neighbors.reserve(std::size(data_points));
    const auto rect = cv::Rect{0, 0, current.cols, current.rows};
    for (const auto &data_point : data_points) {
      if (rect.contains(data_point)) {
        neighbors.emplace_back(data_point);
      }
    }

    if (std::empty(neighbors)) {
      return {};
    }

    // Average intensities with existing data points
    if (std::size(neighbors) != 4) {
      float intensity{0};
      for (const auto &p : neighbors)
        intensity += current.at<float>(p.y, p.x);
      return intensity /= std::size(neighbors);
    }

    // 3. Perform bilinear interpolation
    const auto &tl = data_points.at(0);
    const auto &tr = data_points.at(1);
    const auto &br = data_points.at(2);
    const auto &bl = data_points.at(3);
    const auto f = 1. / ((br.x - bl.x) * (bl.y - tl.y));
    const auto x =
        (Eigen::Matrix<float, 1, 2>() << br.x - uvw(0), uvw(0) - bl.x)
            .finished();
    const auto y =
        (Eigen::Vector2f() << uvw(1) - tl.y, bl.y - uvw(1)).finished();
    const auto Q =
        (Eigen::Matrix2f() << current.at<float>(bl.y, bl.x),
         current.at<float>(tl.y, tl.x), current.at<float>(br.y, br.x),
         current.at<float>(tr.y, tr.x))
            .finished();
    return f * x * Q * y;
  }

private:
  using Roi = cv::Rect_<float>;

  /** @brief Initial guess of the warp at maximum level of the pyramid */
  WarpModel m_initial_warp;
  /** @brief Curretn estimate of the warp*/
  WarpModel m_warp;
  /** @brief Algorithm parameters */
  Parameters m_params;

  ImagePyramidParams m_pyramid_params;
  ImagePyramid m_template_image_pyramid;
  ImagePyramid m_current_image_pyramid;

  /** @brief Template roi in the source image  */
  Roi m_template_roi;
  /** @bief Template roi for each pyramid level */
  std::vector<cv::Rect> m_template_rois;

  /** @brief Image derivatives of each pixel of the template at a given level of
   * the pyramid */
  std::vector<cv::Mat> m_dxdy;
  /** @brief Hessian matrix for each level of the pyramid */
  std::vector<Hessian> m_H;
  /** @brief Jacobian for each pixel [row_i * width + col_i] of the template
   * image for each pyramid level */
  std::vector<std::vector<ParameterVector>> m_steepest_descent_images;
  /** @brief Weights for weighted L2 optimization */
  std::vector<cv::Mat> m_weights;

  void allocate() {
    m_dxdy.clear();
    m_weights.clear();
    m_H.clear();
    m_steepest_descent_images.clear();

    m_dxdy.resize(m_pyramid_params.nb_levels);
    m_weights.resize(m_pyramid_params.nb_levels);
    m_steepest_descent_images.resize(m_pyramid_params.nb_levels);
    m_H.resize(m_pyramid_params.nb_levels);

    const auto &template_pyramid = m_template_image_pyramid.getPyramid();
    for (auto i = 0u; i < m_pyramid_params.nb_levels; ++i) {
      const auto &template_roi = m_template_rois.at(i);
      const auto &image = template_pyramid.at(i)(template_roi);

      // Allocate gradient
      auto &dxdy = m_dxdy.at(i);
      dxdy.create(image.rows * image.cols /* rows */, 2 /* cols */, CV_32F);
      // Allocate weights
      auto &weights = m_weights.at(i);
      weights.create(image.rows, image.cols, CV_32F);
      weights.setTo(1);

      // Allocate steepest descent images
      m_steepest_descent_images.reserve(image.rows * image.cols);
    }
  }
  void precalc() {

    const auto &template_pyramid = m_template_image_pyramid.getPyramid();
    for (auto i = 0u; i < m_pyramid_params.nb_levels; ++i) {
      const auto &template_roi = m_template_rois.at(i);
      const auto &image = template_pyramid.at(i)(template_roi);

      calcImageGradient(image, m_dxdy.at(i));
      if (m_params.weighted)
        calcPixelWeights(image, m_dxdy.at(i), m_weights.at(i));
      calcSteepestDescent(m_dxdy.at(i), m_weights.at(i), template_roi.width,
                          template_roi.height, m_steepest_descent_images.at(i),
                          m_H.at(i));
    }
  }

  void calcImageGradient(const cv::Mat &image, cv::Mat &dxdy) const {
    cv::Mat dx, dy;
    cv::Scharr(image, dx, CV_32F, 1, 0);
    cv::Scharr(image, dy, CV_32F, 0, 1);
    dx.reshape(1, dxdy.rows).copyTo(dxdy.col(0));
    dy.reshape(1, dxdy.rows).copyTo(dxdy.col(1));
  }

  void calcPixelWeights(const cv::Mat &image, const cv::Mat &dxdy,
                        cv::Mat &weights) const {
    float sum{0};
    const auto N = image.rows * image.cols;
    for (auto v = 0u; v < image.rows; ++v) {
      for (auto u = 0u; u < image.cols; ++u) {
        const auto idx = v * image.cols + u;
        // Set higher confidence to pixels with higher gradient magnitude
        weights.at<float>(v, u) = cv::norm(dxdy.row(idx));
        sum += weights.at<float>(v, u);
      }
    }
    weights.convertTo(weights, CV_32F, N / sum);
  }

  void calcSteepestDescent(const cv::Mat &dxdy, const cv::Mat &weights,
                           int width, int height,
                           std::vector<ParameterVector> &jacobian_vectors,
                           Hessian &H) const {
    H.setZero();
    for (auto v = 0; v < height; ++v) {
      for (auto u = 0; u < width; ++u) {
        const auto idx = v * width + u;
        const auto weight = static_cast<double>(weights.at<float>(v, u));
        const auto dx = static_cast<double>(dxdy.at<float>(idx, 0));
        const auto dy = static_cast<double>(dxdy.at<float>(idx, 1));
        const auto grad = (ImageGradient() << dx, dy).finished();
        const auto uv = (Eigen::Vector2d() << u, v).finished();
        const auto dW_dp = WarpModel::jacobian(m_warp, uv);
        const auto &J =
            jacobian_vectors.emplace_back(weight * (grad * dW_dp).transpose());
        // The jacobian is stored transposed it avoids to do it tracking time
        H += weight * J * J.transpose();
      }
    }

    H.diagonal().array() += 1e-7;
  }

  /** @brief Returns the corresponding roi of the template at a given level of
   * the pyramid */
  Roi templateRoiAtLevel(int pyr_level) const {
    const auto level_info =
        m_template_image_pyramid.getLevelInfo().at(pyr_level);
    return {level_info.inv_scale_factor * m_template_roi.x,
            level_info.inv_scale_factor * m_template_roi.y,
            level_info.inv_scale_factor * m_template_roi.width,
            level_info.inv_scale_factor * m_template_roi.height};
  }

  Roi shrinkedRoiAtLevel(int pyr_level, const cv::Mat &level_image) const {
    using S = typename Roi::value_type;
    constexpr auto zero = S{0};
    constexpr auto div_by_2 = S{0.5};
    const auto roi_at_level = templateRoiAtLevel(pyr_level);
    const auto level_image_cols = static_cast<S>(level_image.cols);
    const auto level_image_rows = static_cast<S>(level_image.rows);
    const auto half_h = m_template_roi.height * div_by_2;
    const auto half_w = m_template_roi.width * div_by_2;

    auto roi_center =
        cv::Point_<S>{roi_at_level.x + roi_at_level.width * div_by_2,
                      roi_at_level.y + roi_at_level.height * div_by_2};
    roi_center.x = std::min(roi_center.x, level_image_cols);
    roi_center.y = std::min(roi_center.y, level_image_rows);

    auto rect = Roi{};
    rect.x = std::max(zero, roi_center.x - half_w);
    rect.y = std::max(zero, roi_center.y - half_h);

    auto max_w = (roi_center.x + half_w) - rect.x;
    if (rect.x + max_w >= level_image_cols) {
      max_w = level_image_cols - rect.x;
    }
    auto max_h = (roi_center.y + half_h) - rect.y;
    if (rect.y + max_h >= level_image_rows) {
      max_h = level_image_rows - rect.y;
    }

    rect.width = max_w;
    rect.height = max_h;
    std::cout << "Template roi at level " << pyr_level << " " << roi_at_level
              << " ";
    std::cout << "roi center " << roi_center << " ";
    std::cout << "add shrinked roi " << rect << "\n";
    return rect;
  }

  void displayTrackedWarp(const cv::Mat &image, const WarpModel &model,
                          const Roi &template_roi) {
    cv::Mat W_cv;
    const WarpMatrix &W = model.matrix();
    cv::eigen2cv(W, W_cv);
    const std::array<cv::Mat, 4> corners = {
        // top left
        (cv::Mat_<double>(3, 1) << 0, 0, 1),
        // top right
        (cv::Mat_<double>(3, 1) << template_roi.width, 0, 1),
        // bottom right
        (cv::Mat_<double>(3, 1) << template_roi.width, template_roi.height, 1),
        // bottom left
        (cv::Mat_<double>(3, 1) << 0, template_roi.height, 1),
    };

    std::array<cv::Point2f, 4> warpped_corners;
    for (auto i = 0u; i < std::size(corners); ++i) {
      const cv::Mat c = W_cv * corners[i];
      const auto x = c.at<double>(0);
      const auto y = c.at<double>(1);
      const auto w = c.at<double>(2);
      const auto u = static_cast<float>(x / w);
      const auto v = static_cast<float>(y / w);
      warpped_corners[i].x = u;
      warpped_corners[i].y = v;
    }

    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2RGB);
    const std::array corner_indexes = {0, 1, 2, 3, 0};
    for (auto i = 0; i < std::size(corners); ++i) {
      cv::line(color_image, warpped_corners[corner_indexes[i]],
               warpped_corners[corner_indexes[i + 1]], cv::Scalar(0, 0, 255),
               2);
    }

    // Center point
    const auto half_width = static_cast<double>(template_roi.width) * 0.5;
    const auto half_height = static_cast<double>(template_roi.height) * 0.5;
    const cv::Mat center =
        W_cv * (cv::Mat_<double>(3, 1) << half_width, half_height, 1);
    const cv::Point center_pt{
        static_cast<int>(center.at<double>(0) / center.at<double>(2)),
        static_cast<int>(center.at<double>(1) / center.at<double>(2))};
    cv::circle(color_image, center_pt, 1, cv::Scalar(0, 0, 255), -1);
    cv::imshow("Tracked", color_image);
    cv::waitKey(5);
  }
};
