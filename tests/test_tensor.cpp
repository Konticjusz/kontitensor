#include <cassert>
#include <cmath>
#include <iostream>
#include <tinytensor/tensor.hpp>

using namespace tinytensor;

int main() {
    TensorImpl a({2, 3});
    a.data = {
        1, 2, 3,
        4, 5, 6
    };

    TensorImpl b({3, 2});
    b.data = {
        7,  8,
        9, 10,
        11, 12
    };

    TensorImpl c = a.matmul(b);

    assert(c.shape == std::vector<size_t>({2, 2}));

    assert(std::abs(c.data[0] - 58.0f) < 1e-5);
    assert(std::abs(c.data[1] - 64.0f) < 1e-5);
    assert(std::abs(c.data[2] - 139.0f) < 1e-5);
    assert(std::abs(c.data[3] - 154.0f) < 1e-5);

    std::cout << "matmul test passed\n";
}