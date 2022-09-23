#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

enum class Model { none, affine, translation, sl3, sim2 };

// Map string model to model enum
const std::unordered_map<std::string, Model> &modelStrToModelEnum();

struct Parameters {
  Model model{Model::none};
  int pyramid_levels{4};
  float pyramid_scale{1.2};
  bool weighted{true};
  std::vector<int> iterations{40, 60, 90, 180};
};

Parameters makeParameters(const fs::path &cfg_file);
