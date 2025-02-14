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

TEST_F(TensorTest, ElementAccess) {
    Tensor<float> t({2, 3}, 1.0f);

    EXPECT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_EQ(t.at({1, 2}), 1.0f);

    EXPECT_THROW(t.at({2, 0}), brezel::core::error::LogicError);
    EXPECT_THROW(t.at({0, 3}), brezel::core::error::LogicError);
}

TEST_F(TensorTest, ElementAssignment) {
    Tensor<float> t({2, 3}, 1.0f);

    t.at({0, 0}) = 2.0f;
    t.at({1, 2}) = 3.0f;

    EXPECT_EQ(t.at({0, 0}), 2.0f);
    EXPECT_EQ(t.at({1, 2}), 3.0f);

    EXPECT_THROW(t.at({2, 0}) = 4.0f, brezel::core::error::LogicError);
    EXPECT_THROW(t.at({0, 3}) = 5.0f, brezel::core::error::LogicError);
}

TEST_F(TensorTest, ArithmeticOperations) {
    Tensor<float> t1({2, 2}, 2.0f);
    Tensor<float> t2({2, 2}, 3.0f);

    auto t3 = t1 + t2;
    EXPECT_TRUE(t3.all([](float x) { return x == 5.0f; }));

    auto t4 = t1 - t2;
    EXPECT_TRUE(t4.all([](float x) { return x == -1.0f; }));

    auto t5 = t1 * t2;
    EXPECT_TRUE(t5.all([](float x) { return x == 6.0f; }));

    auto t6 = t1 / t2;
    EXPECT_TRUE(t6.all([](float x) { return x == 2.0f / 3.0f; }));
}

TEST_F(TensorTest, Reduction) {
    Tensor<float> t({2, 3});
    float val = 0.0f;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            t.at({i, j}) = val++;

    auto sum = t.sum();
    EXPECT_EQ(sum.numel(), 1);

    auto mean = t.mean();
    EXPECT_EQ(mean.numel(), 1);

    auto max = t.max();
    EXPECT_EQ(max.numel(), 1);

    auto min = t.min();
    EXPECT_EQ(min.numel(), 1);

    EXPECT_EQ(sum.at({0}), 15.0f);
    EXPECT_EQ(mean.at({0}), 2.5f);
    EXPECT_EQ(max.at({0}), 5.0f);
    EXPECT_EQ(min.at({0}), 0.0f);
}

TEST_F(TensorTest, Reshape) {
    Tensor<float> t({6}, 1.0f);
    auto t1 = t.reshape({2, 3});

    EXPECT_EQ(t1.shape()[0], 2);
    EXPECT_EQ(t1.shape()[1], 3);
    EXPECT_EQ(t1.numel(), 6);

    EXPECT_THROW(t.reshape({2, 4}), brezel::core::error::LogicError);
}

TEST_F(TensorTest, Broadcasting) {
    Tensor<float> t1({2, 1}, 2.0f);
    Tensor<float> t2({1, 3}, 3.0f);
    auto t3 = t1 + t2;

    EXPECT_EQ(t3.shape()[0], 2);
    EXPECT_EQ(t3.shape()[1], 3);
    EXPECT_TRUE(t3.all([](float x) { return x == 5.0f; }));
}

TEST_F(TensorTest, MatrixOperations) {
    Tensor<float> t1({2, 2});
    t1.at({0, 0}) = 1.0f;
    t1.at({0, 1}) = 2.0f;
    t1.at({1, 0}) = 3.0f;
    t1.at({1, 1}) = 4.0f;

    Tensor<float> t2({2, 2});
    t2.at({0, 0}) = 5.0f;
    t2.at({0, 1}) = 6.0f;
    t2.at({1, 0}) = 7.0f;
    t2.at({1, 1}) = 8.0f;

    auto t3 = t1.matmul(t2);
    EXPECT_EQ(t3.at({0, 0}), 19.0f);
    EXPECT_EQ(t3.at({0, 1}), 22.0f);
    EXPECT_EQ(t3.at({1, 0}), 43.0f);
    EXPECT_EQ(t3.at({1, 1}), 50.0f);

    auto t4 = t1.transpose();
    EXPECT_EQ(t4.at({0, 0}), 1.0f);
    EXPECT_EQ(t4.at({0, 1}), 3.0f);
    EXPECT_EQ(t4.at({1, 0}), 2.0f);
    EXPECT_EQ(t4.at({1, 1}), 4.0f);
}

TEST_F(TensorTest, TypeConversion) {
    Tensor<float> t1({2, 2}, 1.5f);

    auto t2 = t1.to<int>();
    EXPECT_TRUE(t2.all([](int x) { return x == 1; }));

    auto t3 = t1.to<double>();
    EXPECT_TRUE(t3.all([](double x) { return x == 1.5; }));
}

TEST_F(TensorTest, ComparisonOperations) {
    Tensor<float> t1({2, 2}, 1.0f);
    Tensor<float> t2({2, 2}, 1.0f);
    Tensor<float> t3({2, 2}, 2.0f);

    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 == t3);
    EXPECT_TRUE(t1.allclose(t2));
    EXPECT_FALSE(t1.allclose(t3));
}

TEST_F(TensorTest, ElementwiseFunctions) {
    Tensor<float> t({2, 2});
    t.at({0, 0}) = 0.0f;
    t.at({0, 1}) = 1.0f;
    t.at({1, 0}) = 2.0f;
    t.at({1, 1}) = 3.0f;

    auto t1 = t.exp();
    EXPECT_FLOAT_EQ(t1.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t1.at({0, 1}), std::exp(1.0f));

    auto t2 = t1.log();
    EXPECT_NEAR(t2.at({0, 1}), 1.0f, 1e-5);

    auto t3 = t.sqrt();
    EXPECT_FLOAT_EQ(t3.at({1, 1}), std::sqrt(3.0f));
}

TEST_F(TensorTest, StatisticalFunctions) {
    Tensor<float> t({3, 3});
    float val = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            t.at({i, j}) = val++;
        }
    }

    auto var = t.var();
    EXPECT_NEAR(var.at({0}), 7.5f, 1e-5);

    auto std = t.std();
    EXPECT_NEAR(std.at({0}), std::sqrt(7.5f), 1e-5);
}