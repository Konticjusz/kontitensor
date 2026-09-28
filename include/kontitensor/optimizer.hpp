#pragma once


#include <vector>

namespace kontitensor {

class Tensor;


class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;
};

class SGD : public Optimizer {
public:
  explicit SGD(float lr, std::vector<Tensor*> parameters);
  void step() override;
  void zero_grad() override;

private:
  float lr;
  std::vector<Tensor *> parameters;
};

} // namespace kontitensor