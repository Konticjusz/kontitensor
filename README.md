# KontiTensor

KontiTensor is a small C++ tensor and autograd library built from scratch for learning purposes.

## Features

- N-dimensional tensors
- Automatic differentiation
- Broadcasting
- Element-wise operations:
  - Addition
  - Subtraction
  - Multiplication
  - Scalar multiplication
- Matrix multiplication
- ReLU
- `sum()` and `mean()`
- SGD optimizer
- Kaiming initialization
- MNIST training example

## MNIST

The repository contains an example of training a small MLP on MNIST.

Download the dataset:

```bash
./scripts/download_mnist.sh
```

Build the project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the example:

```bash
./build/mnist
```

## Roadmap

Planned features include:

- Tensor views
- SIMD kernels
- Multithreading
- Quantization
- Additional neural-network operations

## License

MIT