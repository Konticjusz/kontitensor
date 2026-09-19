#pragma once

#include <stdexcept>
#include <vector>
#include <memory>

namespace tinytensor {

    class GradFn;

    class TensorImpl {
    public:

        std::vector<float> data;
        std::vector<size_t> shape;

        bool requires_grad = false;
        std::unique_ptr<TensorImpl> grad;
        std::shared_ptr<GradFn> grad_fn;
        
        explicit TensorImpl(std::vector<size_t> shape);
        void accumulate_grad(const TensorImpl& gradient);


    };


    class GradFn {
        public:
            virtual void backward(const TensorImpl& grad) = 0;
            virtual ~GradFn() = default;

    };

    class Tensor {

        public:
            explicit Tensor(std::vector<size_t> shape);
            Tensor operator+(const Tensor& other) const;


        private:
            std::shared_ptr<TensorImpl> impl;

    };



}