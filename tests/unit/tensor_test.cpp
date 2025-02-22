#include <gtest/gtest.h>

#include <brezel/core/error/error.hpp>
#include <brezel/tensor/tensor.hpp>

using namespace brezel::tensor;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, DefaultCreation) {
    Tensor<float> t1;

    EXPECT_EQ(t1.shape().size(), 0);
    EXPECT_EQ(t1.numel(), 1);
}

TEST_F(TensorTest, Creation) {
    Tensor<float> t1({2, 3}, 1.0f);

    EXPECT_EQ(t1.shape().size(), 2);
    EXPECT_EQ(t1.shape()[0], 2);
    EXPECT_EQ(t1.shape()[1], 3);
    EXPECT_EQ(t1.numel(), 6);
}

TEST_F(TensorTest, ZeroCreation) {
    auto t2 = Tensor<float>::zeros({2, 3});

    EXPECT_TRUE(t2.all([](float x) { return x == 0.0f; }));
}

TEST_F(TensorTest, OnesCreation) {
    auto t3 = Tensor<float>::ones({2, 3});

    EXPECT_TRUE(t3.all([](float x) { return x == 1.0f; }));
}

TEST_F(TensorTest, EyeCreation) {
    auto t4 = Tensor<float>::eye(3);

    EXPECT_EQ(t4.shape()[0], 3);
    EXPECT_EQ(t4.shape()[1], 3);
    EXPECT_EQ(t4.at({0, 0}), 1.0f);
    EXPECT_EQ(t4.at({1, 1}), 1.0f);
    EXPECT_EQ(t4.at({2, 2}), 1.0f);
    EXPECT_EQ(t4.at({0, 1}), 0.0f);
}

TEST_F(TensorTest, RandomCreation) {
    auto t5 = Tensor<float>::random_uniform({2, 3}, 0.0f, 1.0f);

    EXPECT_EQ(t5.shape().size(), 2);
    EXPECT_TRUE(t5.all([](float x) { return x >= 0.0f && x <= 1.0f; }));
}

TEST_F(TensorTest, ElementAccess) {
    auto tensor = Tensor<float>({2, 2}, 1.0f);

    EXPECT_EQ(tensor.at({0, 0}), 1.0f);
    EXPECT_EQ(tensor.at({0, 1}), 1.0f);
    EXPECT_EQ(tensor.at({1, 0}), 1.0f);
    EXPECT_EQ(tensor.at({1, 1}), 1.0f);

    EXPECT_THROW(tensor.at({3, 3}), brezel::core::error::LogicError);
}

TEST_F(TensorTest, ElementWiseOperations) {
    auto t1 = Tensor<float>({2, 2}, 2.0f);
    auto t2 = Tensor<float>({2, 2}, 3.0f);

    auto add_result = t1 + t2;
    EXPECT_TRUE(add_result.all([](float x) { return x == 5.0f; }));

    auto sub_result = t2 - t1;
    EXPECT_TRUE(sub_result.all([](float x) { return x == 1.0f; }));

    auto mul_result = t1 * t2;
    EXPECT_TRUE(mul_result.all([](float x) { return x == 6.0f; }));

    auto div_result = t2 / t1;
    EXPECT_TRUE(div_result.all([](float x) { return x == 1.5f; }));
}

TEST_F(TensorTest, Broadcasting) {
    // Test broadcasting with compatible shapes
    auto t1 = Tensor<float>({2, 1}, 2.0f);
    auto t2 = Tensor<float>({1, 3}, 3.0f);

    EXPECT_TRUE(t1.shape().is_broadcastable_with(t2.shape()));

    // Test broadcasting scalar with tensor
    auto scalar = Tensor<float>({1}, 2.0f);
    auto tensor = Tensor<float>({2, 3}, 3.0f);
    auto scalar_result = scalar + tensor;
    EXPECT_EQ(scalar_result.shape()[0], 2);
    EXPECT_EQ(scalar_result.shape()[1], 3);
    EXPECT_TRUE(scalar_result.all([](float x) { return x == 5.0f; }));

    // Test broadcasting with same shape
    auto t3 = Tensor<float>({2, 3}, 2.0f);
    auto t4 = Tensor<float>({2, 3}, 3.0f);
    auto same_shape_result = t3 + t4;
    EXPECT_EQ(same_shape_result.shape()[0], 2);
    EXPECT_EQ(same_shape_result.shape()[1], 3);
    EXPECT_TRUE(same_shape_result.all([](float x) { return x == 5.0f; }));

    // Test incompatible shapes
    auto t5 = Tensor<float>({2, 3}, 2.0f);
    auto t6 = Tensor<float>({3, 2}, 3.0f);
    EXPECT_FALSE(t5.shape().is_broadcastable_with(t6.shape()));
    EXPECT_THROW(t5 + t6, brezel::core::error::LogicError);
}

TEST_F(TensorTest, Reduction) {
    auto tensor = Tensor<float>({2, 2}, 2.0f);

    auto sum_result = tensor.sum();
    EXPECT_EQ(sum_result.at({0}), 8.0f);

    auto mean_result = tensor.mean();
    EXPECT_EQ(mean_result.at({0}), 2.0f);

    auto max_result = tensor.max();
    EXPECT_EQ(max_result.at({0}), 2.0f);

    auto min_result = tensor.min();
    EXPECT_EQ(min_result.at({0}), 2.0f);
}

TEST_F(TensorTest, Reshape) {
    auto tensor = Tensor<float>({2, 3}, 1.0f);
    auto reshaped = tensor.reshape({3, 2});

    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 2);
    EXPECT_EQ(reshaped.numel(), 6);
    EXPECT_TRUE(reshaped.all([](float x) { return x == 1.0f; }));

    EXPECT_THROW(tensor.reshape({2, 4}), brezel::core::error::LogicError);
}

TEST_F(TensorTest, Transpose) {
    auto tensor = Tensor<float>::eye(2);
    auto transposed = tensor.transpose();

    EXPECT_EQ(transposed.at({0, 0}), 1.0f);
    EXPECT_EQ(transposed.at({1, 1}), 1.0f);
    EXPECT_EQ(transposed.at({0, 1}), 0.0f);
    EXPECT_EQ(transposed.at({1, 0}), 0.0f);
}

TEST_F(TensorTest, MatrixMultiplication) {
    auto t1 = Tensor<float>({2, 3}, 2.0f);
    auto t2 = Tensor<float>({3, 2}, 3.0f);

    auto result = t1.matmul(t2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);
    EXPECT_TRUE(result.all([](float x) { return x == 18.0f; }));

    EXPECT_THROW(t1.matmul(t1), brezel::core::error::LogicError);
}

TEST_F(TensorTest, Comparison) {
    auto t1 = Tensor<float>({2, 2}, 1.0f);
    auto t2 = Tensor<float>({2, 2}, 1.0f);
    auto t3 = Tensor<float>({2, 2}, 2.0f);

    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 == t3);
    EXPECT_TRUE(t1.allclose(t2));
    EXPECT_FALSE(t1.allclose(t3));
}

TEST_F(TensorTest, TypeConversion) {
    auto t1 = Tensor<float>({2, 2}, 1.5f);
    auto t2 = t1.to<int>();

    EXPECT_TRUE(t2.all([](int x) { return x == 1; }));
}

TEST_F(TensorTest, ViewOperations) {
    auto tensor = Tensor<float>({4, 4}, 1.0f);
    auto view = tensor.narrow(0, 1, 2);

    EXPECT_EQ(view.shape()[0], 2);
    EXPECT_EQ(view.shape()[1], 4);
    EXPECT_TRUE(view.all([](float x) { return x == 1.0f; }));
}

TEST_F(TensorTest, StatisticalOperations) {
    auto tensor = Tensor<float>({3, 3});
    for (size_t i = 0; i < 9; ++i) {
        tensor.data()[i] = static_cast<float>(i);
    }

    auto std_result = tensor.std();
    auto var_result = tensor.var();

    EXPECT_NEAR(std_result.at({0}), 2.582f, 0.001f);
    EXPECT_NEAR(var_result.at({0}), 6.667f, 0.001f);
}

TEST_F(TensorTest, LinearAlgebra) {
    // Test with a known 2x2 matrix
    auto tensor = Tensor<float>({2, 2});
    tensor.data()[0] = 1.0f;  // [1 2]
    tensor.data()[1] = 2.0f;  // [3 4]
    tensor.data()[2] = 3.0f;
    tensor.data()[3] = 4.0f;

    // Compute determinant
    auto det = tensor.det();
    EXPECT_NEAR(det, -2.0f, 1e-5f);

    // Compute QR decomposition
    auto [q, r] = tensor.qr();

    // Check Q is orthogonal (Q^T * Q ≈ I)
    auto qT = q.transpose();
    auto qTq = qT.matmul(q);
    auto I = Tensor<float>::eye(2);
    if (!qTq.allclose(I, 1e-4f, 1e-4f)) {
        std::cout << "\nQ^T * Q - I max difference:\n";
        auto diff = qTq.subtract(I);
        float max_diff = 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            max_diff = std::max(max_diff, std::abs(diff.data()[i]));
        }
        std::cout << "Max difference: " << max_diff << "\n";
        FAIL() << "Q is not orthogonal";
    }

    // Check R is upper triangular
    EXPECT_NEAR(r.at({1, 0}), 0.0f, 1e-4f);

    // Check reconstruction
    auto reconstructed = q.matmul(r);
    if (!reconstructed.allclose(tensor, 1e-4f, 1e-4f)) {
        std::cout << "\nReconstruction error:\n";
        std::cout << "Original:\n";
        tensor.print();
        std::cout << "Reconstructed:\n";
        reconstructed.print();
        std::cout << "Difference:\n";
        auto diff = reconstructed.subtract(tensor);
        diff.print();
        FAIL() << "Reconstruction failed";
    }
}