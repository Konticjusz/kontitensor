#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm> 


#include <tinytensor/tensor.hpp>

namespace tinytensor {




    class AddBackward : public GradFn {
    public:
        
        AddBackward(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b)
            : a(std::move(a)), b(std::move(b)) {}
        
        std::vector<GradResult> backward(const TensorImpl& grad) override{

            return {{a, grad}, {b, grad}};
            
        }


    private:

        std::shared_ptr<TensorImpl> a;
        std::shared_ptr<TensorImpl> b;

    
    };

    class ReluBackward : public GradFn {
    public:
        
        ReluBackward(std::shared_ptr<TensorImpl> a)
            : a(std::move(a)) {}
        
        std::vector<GradResult> backward(const TensorImpl& grad) override{
            
            if (!a->requires_grad) return;
            TensorImpl output_grad(grad.shape);

            for (size_t i = 0; i < grad.data.size(); i++){
                output_grad.data[i] = (a->data[i] > 0.f) ? grad.data[i] : 0.f;
            }

            return {{a, output_grad}};
            
        }


    private:

        std::shared_ptr<TensorImpl> a;

    
    };



    TensorImpl::TensorImpl(const TensorImpl& other)
        : data(other.data),
        shape(other.shape),
        requires_grad(false),
        grad(nullptr),
        grad_fn(nullptr) {}

    TensorImpl::TensorImpl(std::vector<size_t> shape) 
        : shape(std::move(shape)) {
                
        size_t n = 1;

        for (size_t dim : this->shape) {
            n *= dim;
        }

        data.resize(n);

    }

    void TensorImpl::accumulate_grad(const TensorImpl& gradient){

        if (gradient.shape != shape){
            throw std::invalid_argument("Gradient shape mismatch");
        }
        if (grad == nullptr){
            grad = std::make_unique<TensorImpl>(gradient.shape);
        }
        for (size_t i = 0; i < gradient.data.size(); i++){
            grad->data[i] += gradient.data[i];
        }

    }


    TensorImpl TensorImpl::add (const TensorImpl& other) const {
        if (shape != other.shape){
            throw std::invalid_argument ("Tensors have different shapes");
        }
        TensorImpl result(shape);
        for (size_t i = 0; i < data.size(); i++){
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    };


    TensorImpl TensorImpl::relu() const{
        TensorImpl result(shape);

        for (size_t i = 0; i < data.size(); i++){
            result.data[i] = std::max(data[i], 0.f);
        }

        return result;

    }





    Tensor::Tensor(std::vector<size_t> shape) 
        : impl(std::make_shared<TensorImpl>(std::move(shape))) {};

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl) 
        : impl(std::move(impl)) {};


    
    Tensor Tensor::operator+(const Tensor& other) const {
        auto tmp = std::make_shared<TensorImpl>(impl->add(*other.impl));
        Tensor result = Tensor(tmp);

        result.impl->requires_grad = impl->requires_grad || other.impl->requires_grad;
        result.impl->grad_fn = std::make_shared<AddBackward>(impl, other.impl);
        return result;
    }

    Tensor Tensor::relu() const {
        auto tmp = std::make_shared<TensorImpl>(impl->relu());
        Tensor result = Tensor(tmp);

        result.impl->requires_grad = impl->requires_grad;
        result.impl->grad_fn = std::make_shared<ReluBackward>(impl);
        return result;

    }


}

    
