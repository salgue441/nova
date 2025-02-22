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
