#include <iclk/image_pyramid.hpp>

#include <opencv2/imgproc.hpp> // cv::resize
#include <opencv2/highgui.hpp> // cv::nameWindow, cv::imshow, cv::destroyWindow, cv::waitKey

#include <sstream> // std::stringstream

ImagePyramid::ImagePyramid(const ImagePyramidParams& params)
  :m_params(params)
  ,m_gaussian_sigma(calcSigmaFromNbLevels())
{
  if(m_params.nb_levels < 1) {
    std::stringstream ss;
    ss << "Impossible to create an image pyramid with " << m_params.nb_levels;
    throw std::runtime_error(ss.str());
  }
  if(m_params.scale_factor <= 0)
    throw std::runtime_error("Cannot create an image pyramid with scale factor less or equal to zero");

  calcInvScaleFactors();
}

double ImagePyramid::calcSigmaFromNbLevels() const
{
  return 0.3 * ((m_params.nb_levels / 2) - 1) + 0.8;
}

void ImagePyramid::setGaussianSigma(double sigma)
{
  m_gaussian_sigma = sigma;
}

void ImagePyramid::calcInvScaleFactors()
{
  std::vector<float> scale_factors(m_params.nb_levels);
  m_level_info.resize(m_params.nb_levels);

  scale_factors[0] = 1;
  m_level_info[0].inv_scale_factor = 1;

  for(auto i=1; i < m_params.nb_levels; ++i) {
    scale_factors[i] = scale_factors[i - 1] * m_params.scale_factor;
    m_level_info[i].inv_scale_factor = 1 / scale_factors[i];
  }
}

void ImagePyramid::buildPyramid(const cv::Mat& image)
{
  m_pyramid.resize(m_params.nb_levels);
  const auto gauss_kernel = cv::Size{3, 3};

  for (auto level=0; level < m_params.nb_levels; ++level)
  {
      const auto inv_scale = m_level_info[level].inv_scale_factor;
      const cv::Size sz(cvRound(static_cast<float>(image.cols) * inv_scale), cvRound(static_cast<float>(image.rows) * inv_scale));
      // Take border into account
      const cv::Size whole_size(sz.width + m_params.edge_threshold*2, sz.height + m_params.edge_threshold*2);
      cv::Mat image_with_borders(whole_size, image.type());
      const auto effective_image_roi = cv::Rect(m_params.edge_threshold, m_params.edge_threshold, sz.width, sz.height);
      m_pyramid[level] = image_with_borders(effective_image_roi);
      m_level_info[level].image_roi = effective_image_roi;

      // Compute the resized image
      if( level != 0 )
      {
        cv::resize(m_pyramid[level-1], m_pyramid[level], sz, 0, 0, cv::INTER_LINEAR);
	if(m_params.gaussian_blur) {
          cv::GaussianBlur(m_pyramid[level], m_pyramid[level], gauss_kernel, m_gaussian_sigma);
	}
        cv::copyMakeBorder(m_pyramid[level], image_with_borders, m_params.edge_threshold, m_params.edge_threshold, m_params.edge_threshold, m_params.edge_threshold, cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
      }
      else
      {
	if(m_params.gaussian_blur) {
          cv::GaussianBlur(m_pyramid[level], m_pyramid[level], gauss_kernel, m_gaussian_sigma);
	}
        // Add image border for the first level	 
        cv::copyMakeBorder(image, image_with_borders, m_params.edge_threshold, m_params.edge_threshold, m_params.edge_threshold, m_params.edge_threshold,
                         cv::BORDER_REFLECT_101);            
      }
  }

}

const ImagePyramid::Pyramid& ImagePyramid::getPyramid() const
{
  return m_pyramid;
}

const std::vector<ImagePyramid::LevelInfo>& ImagePyramid::getLevelInfo() const
{
  return m_level_info;
}

