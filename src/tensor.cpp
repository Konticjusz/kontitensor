#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <random>

#include <tinytensor/tensor.hpp>
#include <tinytensor/shape.hpp>


namespace tinytensor
{

    class AddBackward : public GradFn
    {
    public:
        AddBackward(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b)
            : a(std::move(a)), b(std::move(b)) {}

        std::vector<GradResult> backward(const TensorImpl &grad) override
        {

            return {{a, grad.sum_to_shape(a->shape)}, {b, grad.sum_to_shape(b->shape)}};
        }

        std::vector<std::shared_ptr<TensorImpl>> parents() const override
        {
            return {a, b};
        }

    private:
        std::shared_ptr<TensorImpl> a;
        std::shared_ptr<TensorImpl> b;
    };

    class ReluBackward : public GradFn
    {
    public:
        ReluBackward(std::shared_ptr<TensorImpl> a)
            : a(std::move(a)) {}

        std::vector<GradResult> backward(const TensorImpl &grad) override
        {

            if (!a->requires_grad)
                return {};
            TensorImpl output_grad(grad.shape);

            for (size_t i = 0; i < grad.data.size(); i++)
            {
                output_grad.data[i] = (a->data[i] > 0.f) ? grad.data[i] : 0.f;
            }

            return {{a, output_grad}};
        }

        std::vector<std::shared_ptr<TensorImpl>> parents() const override
        {
            return {a};
        }

    private:
        std::shared_ptr<TensorImpl> a;
    };

    class ScalarMulBackward : public GradFn{
    public:
        ScalarMulBackward(std::shared_ptr<TensorImpl> parent, float scalar)
        : parent(std::move(parent)), scalar(scalar) {}

        std::vector<GradResult> backward(const TensorImpl &grad) override {

            return {{parent, grad.multiply(scalar)}};

        }
        std::vector<std::shared_ptr<TensorImpl>> parents() const override
        {
            return {parent};
        }


    private:
        std::shared_ptr<TensorImpl> parent;
        float scalar;
    };


    class MulBackward : public GradFn{
    public:
        MulBackward(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b)
        : a(std::move(a)), b(std::move(b)) {}

        std::vector<GradResult> backward(const TensorImpl &grad) override {
            // TODO: returns grad for parents that may not need it.
            return {{a, grad.multiply(*b).sum_to_shape(a->shape)}, {b, grad.multiply(*a).sum_to_shape(b->shape)}};

        }
        std::vector<std::shared_ptr<TensorImpl>> parents() const override
        {
            return {a, b};
        }


    private:
        std::shared_ptr<TensorImpl> a, b;
    };

    class MatMulBackward : public GradFn{
    public:
        MatMulBackward(std::shared_ptr<TensorImpl> lhs, std::shared_ptr<TensorImpl> rhs)
        : lhs(std::move(lhs)), rhs(std::move(rhs)) {}

        std::vector<GradResult> backward(const TensorImpl &grad) override {
            // TODO: returns grad for parents that may not need it.
            return {{lhs, grad.matmul(rhs->transpose())}, {rhs, lhs->transpose().matmul(grad)}};

        }
        std::vector<std::shared_ptr<TensorImpl>> parents() const override
        {
            return {lhs, rhs};
        }


    private:
        std::shared_ptr<TensorImpl> lhs, rhs;
    };


    class SumBackward : public GradFn{
        public:
            SumBackward(std::shared_ptr<TensorImpl> parent)
            : parent(std::move(parent)) {}

            std::vector<GradResult> backward(const TensorImpl &grad) override {
                return {{parent, grad.broadcast_to(parent->shape)}};
            }

            std::vector<std::shared_ptr<TensorImpl>> parents() const override{
                return {parent};
            }

        
        private:
            std::shared_ptr<TensorImpl> parent;

    };

    TensorImpl::TensorImpl(const TensorImpl &other)
        : data(other.data),
          shape(other.shape),
          strides(other.strides),
          requires_grad(false),
          grad(nullptr),
          grad_fn(nullptr) {}

    TensorImpl::TensorImpl(std::vector<size_t> shape)
        : shape(std::move(shape))
    {

        size_t n = 1;

        for (size_t dim : this->shape)
        {
            n *= dim;
        }
        data.resize(n);

        strides.resize(this->shape.size());
        if (strides.size()){
            strides.back() = 1;
            for (size_t i = this->shape.size() - 1; i > 0; i--){
                strides[i-1] = strides[i] * this->shape[i];
            }   
        }
    }

    size_t TensorImpl::compute_offset(std::span<const size_t> indices) const {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++){
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    void TensorImpl::accumulate_grad(const TensorImpl &gradient)
    {

        if (gradient.shape != shape)
        {
            throw std::invalid_argument("Gradient shape mismatch");
        }
        if (grad == nullptr)
        {
            grad = std::make_shared<TensorImpl>(gradient.shape);
        }
        for (size_t i = 0; i < gradient.data.size(); i++)
        {
            grad->data[i] += gradient.data[i];
        }
    }

    TensorImpl TensorImpl::sum_to_shape(std::span<const size_t> target_shape) const {
                TensorImpl res(std::vector<size_t>(target_shape.begin(), target_shape.end()));
                std::vector<size_t> new_indices(target_shape.size());
                size_t diff = shape.size() - target_shape.size();
                for (size_t linear = 0; linear < data.size(); linear++){
                    std::vector<size_t> indices = linear_to_indices(linear, shape);
                    for (size_t i = diff; i < shape.size(); i++){
                        if (target_shape[i - diff] != shape[i]){
                            new_indices[i - diff] = 0;
                        }
                        else{
                            new_indices[i - diff] = indices[i];
                        }
                    }
                    res.data[res.compute_offset(new_indices)] += data[compute_offset(indices)];
                }
                return res;
            };

    
    TensorImpl TensorImpl::broadcast_to(std::span<const size_t> target_shape) const{
                TensorImpl result(std::vector<size_t>{target_shape.begin(), target_shape.end()});
                std::vector<size_t> curr_indices(shape.size()); // indices from which we will access data from current tensor.
                size_t diff = target_shape.size() - shape.size();
                for (size_t linear_idx = 0; linear_idx < result.data.size(); linear_idx++){
                    std::vector<size_t> indices = linear_to_indices(linear_idx, target_shape);
                    for (size_t i = diff; i < target_shape.size(); i++){
                        if (target_shape[i] == shape[i - diff]){
                            curr_indices[i - diff] = indices[i];
                        }
                        else if (shape[i-diff] == 1){
                            curr_indices[i - diff] = 0;
                        }
                        else{
                            throw std::invalid_argument(" Shapes are not broadcastable. ");
                        }
                    }
                    result.data[result.compute_offset(indices)] = data[compute_offset(curr_indices)];
                }
                return result;
            }





                    

    TensorImpl TensorImpl::add(const TensorImpl &other) const
    {   
        // TODO fastpath for contigous 
        auto result_shape = broadcast_shape(shape, other.shape);
        TensorImpl result(result_shape);
        auto a_strides = broadcast_strides(shape, strides, result_shape);
        auto b_strides = broadcast_strides(other.shape, other.strides, result_shape);
        
        for (size_t i = 0; i < result.data.size(); i++){
            // TODO change that vector indices isn't allocated each time.
            std::vector<size_t> indices = linear_to_indices(i, result.shape);
            result.data[result.compute_offset(indices)] = data[tinytensor::compute_offset(indices, a_strides)] 
            + other.data[tinytensor::compute_offset(indices, b_strides)];
        }

        return result;

    };

    void TensorImpl::add_inplace(const TensorImpl &other){
        if (shape != other.shape)
        {
            throw std::invalid_argument("Tensors have different shapes");
        }
        for (size_t i = 0; i < data.size(); i++){
            data[i] += other.data[i];
        }
    }

    TensorImpl TensorImpl::relu() const
    {
        TensorImpl result(shape);

        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = std::max(data[i], 0.f);
        }


        return result;
    }

    TensorImpl TensorImpl::multiply(float scalar) const {
        
        TensorImpl result(shape);

        for (size_t i = 0; i < data.size(); i++)
        {
            result.data[i] = data[i] * scalar;
        }

        return result;

    }

    TensorImpl TensorImpl::multiply(const TensorImpl &other) const
    {   
        // TODO fastpath for contigous 
        auto result_shape = broadcast_shape(shape, other.shape);
        TensorImpl result(result_shape);
        auto a_strides = broadcast_strides(shape, strides, result_shape);
        auto b_strides = broadcast_strides(other.shape, other.strides, result_shape);
        
        for (size_t i = 0; i < result.data.size(); i++){
            // TODO change that vector indices isn't allocated each time.
            std::vector<size_t> indices = linear_to_indices(i, result.shape);
            result.data[result.compute_offset(indices)] = data[tinytensor::compute_offset(indices, a_strides)] 
            * other.data[tinytensor::compute_offset(indices, b_strides)];
        }

        return result;

    };

    TensorImpl TensorImpl::matmul(const TensorImpl &other) const
    {
        if (shape.size() != 2) // TODO: Support higher dimensions.
        {
            throw std::invalid_argument("matmul currently supports only 2D tensors");
        }

        if (shape[1] != other.shape[0]){
            throw std::invalid_argument("tensors' shapes mismatch");
        }

        TensorImpl result({shape[0], other.shape[1]});
        for (size_t i = 0; i < shape[0]; i++)
        {
            for (size_t j = 0; j < other.shape[1]; j++){
                for (size_t k = 0; k < shape[1]; k++){
                    result.data[result.compute_offset(std::array{i,j})] += 
                    (data[this->compute_offset(std::array{i,k})] * 
                    other.data[other.compute_offset(std::array{k,j})]);
                }
            }
        }
        return result;
    };

    TensorImpl TensorImpl::transpose () const {
        if (shape.size() != 2){
            throw std::invalid_argument("Transpose currently supports only 2D tensors");
        }
        TensorImpl result({shape[1], shape[0]});

        for (size_t i = 0; i < shape[1]; i++){
            for (size_t j = 0; j < shape[0]; j++){
                result.data[i * shape[0] + j] = data[j*shape[1] + i];
            }
        }
        return result;
    }

    TensorImpl TensorImpl::sum () const{
        TensorImpl result({1});
        float sum = 0.0f;
        for (float elem: data){
            sum += elem;
        }
        result.data[0] = sum;
        return result;
    }



    Tensor::Tensor(std::vector<size_t> shape, bool requires_grad)
        : impl(std::make_shared<TensorImpl>(std::move(shape))) {
            impl->requires_grad = requires_grad;
        };

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl)
        : impl(std::move(impl)) {};

    Tensor::Tensor(std::vector<float> data, std::vector<size_t> shape, bool requires_grad)
        : impl(std::make_shared<TensorImpl>(std::move(shape))) {
            if (data.size() != impl->data.size()) {
                throw std::invalid_argument("Data size does not match tensor shape");
            }
            impl->requires_grad = requires_grad;
            impl->data = std::move(data);
        }

    Tensor Tensor::operator+(const Tensor &other) const
    {
        auto tmp = std::make_shared<TensorImpl>(impl->add(*other.impl));
        Tensor result = Tensor(tmp);

        result.impl->requires_grad = impl->requires_grad || other.impl->requires_grad;
        result.impl->grad_fn = std::make_shared<AddBackward>(impl, other.impl);
        return result;
    }

    Tensor Tensor::operator*(float scalar) const{
        Tensor result(std::make_shared<TensorImpl>(impl->multiply(scalar)));

        result.impl->requires_grad = impl->requires_grad;

        if (result.impl->requires_grad){
            result.impl->grad_fn = std::make_shared<ScalarMulBackward>(impl, scalar);
        }

        return result;
    }

    Tensor Tensor::operator-(const Tensor &other) const
    {
        auto result = (*this) + (other * (-1.0f));

        
        return result;
    }





    Tensor Tensor::operator*(const Tensor &other) const{
        Tensor result(std::make_shared<TensorImpl>(impl->multiply(*other.impl)));

        result.impl->requires_grad = impl->requires_grad || other.impl->requires_grad;

        if (result.impl->requires_grad){
            result.impl->grad_fn = std::make_shared<MulBackward>(impl, other.impl);
        }

        return result;
    }

    Tensor Tensor::matmul (const Tensor &other) const {

        Tensor result(std::make_shared<TensorImpl>(impl->matmul(*other.impl)));

        result.impl->requires_grad = impl->requires_grad || other.impl->requires_grad;

        if (result.impl->requires_grad){
            result.impl->grad_fn = std::make_shared<MatMulBackward>(impl, other.impl);
        }

        return result;

    }


    Tensor Tensor::relu() const
    {
        auto tmp = std::make_shared<TensorImpl>(impl->relu());
        Tensor result = Tensor(tmp);

        result.impl->requires_grad = impl->requires_grad;
        result.impl->grad_fn = std::make_shared<ReluBackward>(impl);
        return result;
    }


    Tensor Tensor::sum() const{
        auto tmp = std::make_shared<TensorImpl>(impl->sum());
        Tensor result(tmp);
        if (impl->requires_grad){
            result.impl->requires_grad = true;
            result.impl->grad_fn = std::make_shared<SumBackward>(impl);
        }
        return result;
        
    }


    Tensor Tensor::mean() const{
        if (impl->data.size() == 0){
            throw std::invalid_argument( "Tensor has no elements. ");
        }
        auto result = sum() * (1.0f / (float) impl->data.size());
        return result;
    }


    float Tensor::item() const{
        if (impl->data.size() != 1){
            throw std::runtime_error("item() requires a tensor with one element");
        }
        return impl->data[0];
    }

    namespace
    {
        void build_topo(
            const std::shared_ptr<TensorImpl> &node,
            std::vector<std::shared_ptr<TensorImpl>> &topo,
            std::unordered_set<TensorImpl *> &visited)
        {
            visited.insert(node.get());

            if (node->grad_fn != nullptr)
            {
                for (const auto &parent : node->grad_fn->parents())
                {
                    if (parent->requires_grad && !visited.count(parent.get()))
                    {
                        build_topo(parent, topo, visited);
                    }
                }
            }

            topo.push_back(node);
        }
    }

    void Tensor::backward()
    {

        if (impl->data.size() != 1)
        {
            throw std::runtime_error("Backward should be applied on a scalar tensor.");
        }

        std::vector<std::shared_ptr<TensorImpl>> topo;
        std::unordered_set<TensorImpl *> visited;
        build_topo(impl, topo, visited);

        TensorImpl initial_grad({1});
        initial_grad.data[0] = 1.0f;


        std::unordered_map<TensorImpl*, TensorImpl> current_grads;
        current_grads.emplace(impl.get(), std::move(initial_grad));

        for (auto it = topo.rbegin(); it != topo.rend(); it++){
            auto& node = *it;

            auto grad_it = current_grads.find(node.get());
            if (grad_it== current_grads.end()){
                continue;
            }
            auto& grad = (*grad_it).second;
            node->accumulate_grad(grad);
            
            if (node->grad_fn == nullptr){
                continue;
            }

            for (const auto& [parent, parent_grad] : node->grad_fn->backward(grad)){
                auto par_grad_it = current_grads.find(parent.get());
                if (par_grad_it == current_grads.end()){
                    current_grads.emplace(parent.get(), parent_grad);
                }
                else{
                     par_grad_it->second.add_inplace(parent_grad);
                }
            }
        }

    }

    Tensor Tensor::grad() const{
        return Tensor(impl->grad);
    }

    void Tensor::add_(const Tensor& other, float alpha){
        impl->add_inplace(other.impl->multiply(alpha));
    }

    void Tensor::zero_grad(){
        if (!impl->grad){
            return;
        }
        for (auto& elem: impl->grad->data){
            elem = 0.0f;
        }
    }

    Tensor Tensor::kaiming_normal(std::vector<size_t> shape, bool requires_grad){
        if (shape.size() < 2){
            throw std::invalid_argument("Kaiming initialization requires at least 2D tensor");
        }

        size_t fan_in = shape[0];

        float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));

        size_t numel = 1;
        for (size_t dim: shape){
            numel *= dim;
        }

        std::vector<float> data(numel);

        static std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, stddev);

        for (auto& elem: data){
            elem = dist(rng);

        }
        
        return Tensor(std::move(data), std::move(shape), requires_grad);
    }




}
