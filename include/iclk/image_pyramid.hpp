#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct ImagePyramidParams
{
  /** @brief Number of levels in the pyramid, 0 being the full resolution image */
  int nb_levels;
  /** @brief Scale factor by which the image at L is scaled to produce L-1 */
  float scale_factor;
  /** @brief Optional border added to each image of the pyramid */
  int edge_threshold;
  /** @brief Apply blur at each level of the pyramid */
  bool gaussian_blur;
};

class ImagePyramid
{
  using Pyramid = std::vector<cv::Mat>;
  struct LevelInfo
  {
    /** @brief ROI containing the image without borders */
    cv::Rect image_roi;
    /** @brief 1/scale_factor for each pyramid level */
    float inv_scale_factor;
  };

public:
  explicit ImagePyramid(const ImagePyramidParams& params);
  /** @brief Build the image a pyramid where pyramid[0] is the full image resolution (with extra-border) */
  void buildPyramid(const cv::Mat& image);
  const Pyramid& getPyramid() const;
  const std::vector<LevelInfo>& getLevelInfo() const;
  void setGaussianSigma(double sigma);

private:
  ImagePyramidParams m_params;
  double m_gaussian_sigma;

  /** @brief Effective image at level L */
  std::vector<LevelInfo> m_level_info;
  /** @brief Zero based index (full resolution) up to max index (lowest resolution) */
  Pyramid m_pyramid;

  /** @brief Initializes the inverse scale factor for each pyramid level */
  void calcInvScaleFactors();
  /** @brief Initializes the sigma for gausian smoothing */
  double calcSigmaFromNbLevels() const;
};

