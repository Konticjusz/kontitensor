#pragma once

#include <stdexcept>
#include <vector>
#include <memory>
#include <utility> 


namespace tinytensor {

    class TensorImpl;
    class GradFn;

    using GradResult = 
        std::pair<std::shared_ptr<TensorImpl>, TensorImpl>;

    class TensorImpl {
    public:

        std::vector<float> data;
        std::vector<size_t> shape;

        bool requires_grad = false;
        std::unique_ptr<TensorImpl> grad;
        std::shared_ptr<GradFn> grad_fn;
        

        explicit TensorImpl(std::vector<size_t> shape);
        TensorImpl(const TensorImpl& other);
        
        TensorImpl add(const TensorImpl& other) const;
        TensorImpl relu() const;
        void accumulate_grad(const TensorImpl& gradient);


    };



    class GradFn {
        public:
            virtual std::vector<GradResult> backward(const TensorImpl& grad) = 0;
            virtual ~GradFn() = default;

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