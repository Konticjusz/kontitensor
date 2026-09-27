#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <tinytensor/optimizer.hpp>
#include <tinytensor/shape.hpp>
#include <tinytensor/tensor.hpp>

namespace tinytensor {

SGD::SGD(float lr, std::vector<Tensor *> params)
    : lr(lr), parameters(std::move(params)) {}

void SGD::step() {
  for (auto param : parameters) {
    param->add_(param->grad(), -lr);
  }
}

void SGD::zero_grad() {
  for (auto param : parameters) {
    param->zero_grad();
  }
}
} // namespace tinytensor
