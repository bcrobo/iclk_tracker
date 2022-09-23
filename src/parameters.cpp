#include <iclk/parameters.hpp>

#include <yaml-cpp/yaml.h>

#include <iostream>

const std::unordered_map<std::string, Model> &modelStrToModelEnum() {
  static const std::unordered_map<std::string, Model> model_str_to_model_enum{
      {"none", Model::none},
      {"affine", Model::affine},
      {"translation", Model::translation},
      {"sl3", Model::sl3},
      {"sim2", Model::sim2}};
  return model_str_to_model_enum;
}

Parameters makeParameters(const fs::path &cfg_file) {
  auto p = Parameters{};
  auto config = YAML::LoadFile(cfg_file);
  const auto &model_str_to_model_enum = modelStrToModelEnum();
  p.model = model_str_to_model_enum.at(config["model"].as<std::string>());
  p.pyramid_levels = config["pyramid_levels"].as<int>();
  p.pyramid_scale = config["pyramid_scale"].as<float>();
  p.weighted = config["weighted"].as<bool>();
  p.iterations = config["iterations"].as<std::vector<int>>();

  if (std::size(p.iterations) != p.pyramid_levels) {
    throw std::runtime_error{
        "Number of iteration should be equal to the number of pyramid layer"};
  }
  return p;
}
