#include <gtest/gtest.h>

#include <brezel/tensor/tensor.hpp>
#include <brezel/tensor/tensor_expressions.hpp>
#include <vector>

using namespace brezel::tensor;

template <typename T>
class MockTensor : public TensorExpression<MockTensor<T>, T> {
public:
    MockTensor() : m_shape(Shape()) {}

    explicit MockTensor(std::vector<T> data, Shape shape)
        : m_data(std::move(data)), m_shape(std::move(shape)) {}

    BREZEL_NODISCARD size_t numel() const { return m_data.size(); }
    BREZEL_NODISCARD const Shape& shape() const { return m_shape; }
    BREZEL_NODISCARD bool is_contiguous() const { return true; }
    BREZEL_NODISCARD T operator[](size_t i) const { return m_data[i]; }
    BREZEL_NODISCARD T at(std::span<const int64_t> indices) const {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            size_t stride = 1;
            for (size_t j = i + 1; j < indices.size(); ++j) {
                stride *= m_shape[j];
            }

            index += indices[i] * stride;
        }

        return m_data[index];
    }

private:
    std::vector<T> m_data;
    Shape m_shape;
};

class TensorExpressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock tensors for testing
        a_data = {1.0f, 2.0f, 3.0f, 4.0f};
        b_data = {5.0f, 6.0f, 7.0f, 8.0f};
        a_shape = Shape({2, 2});
        b_shape = Shape({2, 2});

        a = MockTensor<float>(a_data, a_shape);
        b = MockTensor<float>(b_data, b_shape);

        // Create broadcasting test tensors
        c_data = {1.0f, 2.0f};
        c_shape = Shape({2, 1});
        c = MockTensor<float>(c_data, c_shape);

        d_data = {3.0f, 4.0f};
        d_shape = Shape({1, 2});
        d = MockTensor<float>(d_data, d_shape);
    }

    std::vector<float> a_data, b_data, c_data, d_data;
    Shape a_shape, b_shape, c_shape, d_shape;
    MockTensor<float> a, b, c, d;
};

TEST_F(TensorExpressionTest, BinaryOperations) {
    auto add_expr = a + b;
    auto add_result = add_expr.eval();

    ASSERT_EQ(add_result.numel(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(add_result.data()[i], a_data[i] + b_data[i]);
    }
}