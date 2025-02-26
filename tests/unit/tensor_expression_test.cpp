#include <gtest/gtest.h>

#include <brezel/tensor/tensor.hpp>
#include <brezel/tensor/tensor_expressions.hpp>
#include <vector>

using namespace brezel::tensor;

class TensorExpressionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test tensors
        a = Tensor<float>({2, 2}, 2.0f);
        b = Tensor<float>({2, 2}, 3.0f);
        scalar = 5.0f;
    }

    Tensor<float> a;
    Tensor<float> b;
    float scalar;
};

TEST_F(TensorExpressionsTest, BasicExpression) {
    EXPECT_TRUE((std::is_base_of<TensorExpression<Tensor<float>, float>,
                                 Tensor<float>>::value));
}
