#include <gtest/gtest.h>

#include <brezel/autograd/autograd.hpp>
#include <brezel/tensor/tensor.hpp>
#include <cmath>

using namespace brezel::autograd;
using namespace brezel::autograd::functions;
using namespace brezel::tensor;

// Helper function for floating-point comparison
bool is_close(float a, float b, float rtol = 1e-5, float atol = 1e-8) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// Helper function for tensor comparison
bool tensors_close(const Tensor<float>& a, const Tensor<float>& b,
                   float rtol = 1e-5, float atol = 1e-8) {
    if (a.shape() != b.shape())
        return false;

    for (size_t i = 0; i < a.numel(); ++i) {
        if (!is_close(a.data()[i], b.data()[i], rtol, atol))
            return false;
    }

    return true;
}

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code - can initialize shared resources here
    }

    void TearDown() override {
        // Cleanup code
    }
};

TEST_F(AutogradTest, BasicAdd) {
    auto a = tensor(Tensor<float>({2, 2}, 1.0f), true);
    auto b = tensor(Tensor<float>({2, 2}, 2.0f), true);
    auto c = add(a, b);

    ASSERT_EQ(c->data().shape(), Shape({2, 2}));
    ASSERT_EQ(c->data().at({0, 0}), 3.0f);
    ASSERT_TRUE(c->requires_grad());

    c->backward(Tensor<float>({2, 2}, 1.0f));
    ASSERT_EQ(a->grad().at({0, 0}), 1.0f);
    ASSERT_EQ(b->grad().at({0, 0}), 1.0f);
}