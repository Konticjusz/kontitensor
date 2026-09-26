#include <cassert>
#include <cmath>
#include <iostream>
#include <tinytensor/tensor.hpp>
#include <tinytensor/shape.hpp>

using namespace tinytensor;

void test_matmul() {
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

void test_broadcast_add()
{
    TensorImpl a({2, 3});
    a.data = {
        1, 2, 3,
        4, 5, 6
    };

    TensorImpl b({1, 3});
    b.data = {
        10, 20, 30
    };

    TensorImpl result = a.add(b);

    assert((result.shape == Shape{2, 3}));

    std::vector<float> expected = {
        11, 22, 33,
        14, 25, 36
    };

    assert(result.data == expected);
}


void test_broadcast_add_column()
{
    TensorImpl a({2, 3});
    a.data = {
        1, 2, 3,
        4, 5, 6
    };

    TensorImpl b({2, 1});
    b.data = {
        10,
        20
    };

    TensorImpl result = a.add(b);

    assert((result.shape == Shape{2, 3}));

    std::vector<float> expected = {
        11, 12, 13,
        24, 25, 26
    };

    assert(result.data == expected);
}

int main(){
    test_matmul();
    test_broadcast_add();
    test_broadcast_add_column();
}