#pragma once

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tinytensor {

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

Shape broadcast_shape(std::span<const size_t> a, std::span<const size_t> b);
Strides broadcast_strides(std::span<const size_t> shape,
                          std::span<const size_t> strides,
                          std::span<const size_t> target_shape);
size_t compute_offset(std::span<const size_t> indices,
                      std::span<const size_t> strides);
std::vector<size_t> linear_to_indices(size_t linear,
                                      std::span<const size_t> shape);

} // namespace tinytensor