#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <tinytensor/shape.hpp>
#include <tinytensor/tensor.hpp>

namespace tinytensor {

Shape broadcast_shape(std::span<const size_t> a, std::span<const size_t> b) {
  if (b.size() > a.size()) {
    swap(a, b);
  }
  Shape result(a.size());
  for (size_t i = 0; i < a.size() - b.size(); i++) {
    result[i] = a[i];
  }
  size_t diff = a.size() - b.size();
  for (size_t i = diff; i < a.size(); i++) {
    if (a[i] == b[i - diff]) {
      result[i] = a[i];
    } else if (a[i] == 1 || b[i - diff] == 1) {
      result[i] = std::max(a[i], b[i - diff]);
    } else {
      throw std::invalid_argument(" Tensors can't be broadcasted. ");
    }
  }
  return result;
}

Strides broadcast_strides(std::span<const size_t> shape,
                          std::span<const size_t> strides,
                          std::span<const size_t> target_shape) {
  Strides new_strides(target_shape.size());
  size_t diff = target_shape.size() - shape.size();
  for (size_t i = 0; i < diff; i++) {
    new_strides[i] = 0;
  }
  for (size_t i = diff; i < new_strides.size(); i++) {
    if (shape[i - diff] == target_shape[i]) {
      new_strides[i] = strides[i - diff];
    } else {
      new_strides[i] = 0;
    }
  }
  return new_strides;
}

size_t compute_offset(std::span<const size_t> indices,
                      std::span<const size_t> strides) {

  if (indices.size() != strides.size()) {
    throw std::invalid_argument(
        " Indices size should be equal to strides size. ");
  }

  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

std::vector<size_t> linear_to_indices(size_t linear,
                                      std::span<const size_t> shape) {
  std::vector<size_t> indices(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    indices[i] = linear % shape[i];
    linear /= shape[i];
  }
  return indices;
}

} // namespace tinytensor