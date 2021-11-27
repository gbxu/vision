#pragma once

#include <torch/nn.h>

namespace vision {
namespace models {
namespace modelsimpl {

// TODO here torch::relu_ and torch::adaptive_avg_pool2d wrapped in
// torch::nn::Fuctional don't work. so keeping these for now

inline torch::Tensor& relu_(const torch::Tensor& x) {
  return x.relu_();
}

inline torch::Tensor& relu6_(const torch::Tensor& x) {
  return x.clamp_(0, 6);
}

inline torch::Tensor adaptive_avg_pool2d(
    const torch::Tensor& x,
    torch::ExpandingArray<2> output_size) {
  return torch::adaptive_avg_pool2d(x, output_size);
}

inline torch::Tensor avg_pool2d(
    const torch::Tensor& x,
    torch::ExpandingArray<2> kernel_size,
    torch::ExpandingArray<2> stride) {
  return torch::avg_pool2d(x, kernel_size, stride);
}

inline bool double_compare(double a, double b) {
  return double(std::abs(a - b)) < std::numeric_limits<double>::epsilon();
};

inline void deprecation_warning() {
  TORCH_WARN_ONCE(
      "The vision::models namespace is not actively maintained, use at "
      "your own discretion. We recommend using Torch Script instead: "
      "https://pytorch.org/tutorials/advanced/cpp_export.html");
}

} // namespace modelsimpl
} // namespace models
} // namespace vision
