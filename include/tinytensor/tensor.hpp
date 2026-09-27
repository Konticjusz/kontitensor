#pragma once

#include <stdexcept>
#include <vector>
#include <memory>
#include <utility>
#include <span>


namespace tinytensor
{

    class TensorImpl;
    class GradFn;

    using GradResult =
        std::pair<std::shared_ptr<TensorImpl>, TensorImpl>;

    class TensorImpl
    {
    public:
        std::vector<float> data;
        std::vector<size_t> shape;
        std::vector<size_t> strides;

        bool requires_grad = false;
        std::unique_ptr<TensorImpl> grad;
        std::shared_ptr<GradFn> grad_fn;

        std::vector<std::shared_ptr<TensorImpl>> build_topo() const;

        explicit TensorImpl(std::vector<size_t> shape);
        TensorImpl(const TensorImpl &other);
        TensorImpl multiply(float scalar) const;
        TensorImpl multiply(const TensorImpl &other) const;
        TensorImpl matmul(const TensorImpl &other) const;
        TensorImpl add(const TensorImpl &other) const;
        TensorImpl transpose() const;
        TensorImpl broadcast_to(std::span<const size_t> target_shape) const;
        TensorImpl sum() const;
        void add_inplace(const TensorImpl &other);
        size_t compute_offset(std::span<const size_t> indices) const;
        TensorImpl relu() const;
        void accumulate_grad(const TensorImpl &gradient);
        TensorImpl sum_to_shape(std::span<const size_t> target_shape) const;
    };

    class GradFn
    {
    public:
        virtual std::vector<GradResult> backward(const TensorImpl &grad) = 0;
        virtual ~GradFn() = default;
        virtual std::vector<std::shared_ptr<TensorImpl>> parents() const = 0;
    };

    class Tensor
    {

    public:
        explicit Tensor(std::shared_ptr<TensorImpl> impl);
        explicit Tensor(std::vector<size_t> shape, bool requires_grad = false);
        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor operator*(float scalar) const;
        Tensor matmul(const Tensor& other) const;
        Tensor relu() const;
        Tensor mean() const;
        Tensor sum() const;
        void backward();

    private:
        std::shared_ptr<TensorImpl> impl;
    };

}