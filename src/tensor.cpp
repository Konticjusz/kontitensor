#include <stdexcept>
#include <vector>
#include <memory>

#include <tinytensor/tensor.hpp>

namespace tinytensor {


    class AddBackward : GradFn {
    public:
        
        AddBackward(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b)
            : a(std::move(a)), b(std::move(b)) {}
        
        void backward(const TensorImpl& grad) override{

            if (a->requires_grad){
                a->accumulate_grad(grad);
            }
            if (b->requires_grad){
                b->accumulate_grad(grad);
            }
            

        }

    private:

        std::shared_ptr<TensorImpl> a;
        std::shared_ptr<TensorImpl> b;

    
    };



    TensorImpl::TensorImpl(std::vector<size_t> shape) 
        : shape(std::move(shape)) {
                
        size_t n = 1;

        for (size_t dim : this->shape) {
            n *= dim;
        }

        data.resize(n);

    }





    Tensor::Tensor(std::vector<size_t> shape) 
        : impl(std::make_shared<TensorImpl>(std::move(shape))) {};

    
    Tensor Tensor::operator+(const Tensor& other) const {
        if (impl->shape != other.impl->shape){
            throw std::invalid_argument ("Tensors have different shapes");
        }

        Tensor result(impl->shape);

        result.impl->requires_grad = impl->requires_grad || other.impl->requires_grad;

        for (size_t i = 0; i < impl->data.size(); i++){
            result.impl->data[i] = impl->data[i] + other.impl->data[i];
        }

        return result;

    }


}

    
