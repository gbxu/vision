#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor _interpolate_bilinear2d_aa(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners = false);

VISION_API at::Tensor _interpolate_bicubic2d_aa(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners = false);

namespace detail {

VISION_API at::Tensor _interpolate_bilinear2d_aa_backward(
    const at::Tensor& grad,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners = false);

VISION_API at::Tensor _interpolate_bicubic2d_aa_backward(
    const at::Tensor& grad,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners = false);

} // namespace detail

} // namespace ops
} // namespace vision
