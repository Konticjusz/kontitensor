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
        TensorImpl add(const TensorImpl& other) const;
        TensorImpl relu() const;
        void accumulate_grad(const TensorImpl& gradient);


    };


    class GradFn {
        public:
            virtual void backward(const TensorImpl& grad) = 0;
            virtual ~GradFn() = default;
            virtual std::vector<std::shared_ptr<TensorImpl>> parents() const = 0;

    };

    class Tensor {

        public:
            explicit Tensor(std::shared_ptr<TensorImpl> impl);
            explicit Tensor(std::vector<size_t> shape);
            Tensor operator+(const Tensor& other) const;
            Tensor relu() const;


        private:
            std::shared_ptr<TensorImpl> impl;

    };



}