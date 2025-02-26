#include <gtest/gtest.h>

#include <brezel/tensor/tensor_concept.hpp>
#include <cmath>
#include <string>
#include <vector>

using namespace brezel::tensor;

// Test structure that doesn't satisfy TensorScalar
struct NotScalar {
    int x, y;
};

// Test structure that doesn't satisfy TensorNumeric
enum class EnumType { A, B, C };
struct CustomFloatOp {
    template <typename T>
    T operator()(T a, T b) const {
        return std::pow(a, b);
    }
};

class TensorConceptTest : public ::testing::Test {};

TEST_F(TensorConceptTest, ScalarTypeConstruction) {
    // Test valid scalar types
    EXPECT_TRUE(TensorScalar<int>);
    EXPECT_TRUE(TensorScalar<float>);
    EXPECT_TRUE(TensorScalar<double>);
    EXPECT_TRUE(TensorScalar<bool>);
    EXPECT_TRUE(TensorScalar<char>);
    EXPECT_TRUE(TensorScalar<unsigned int>);
    EXPECT_TRUE(TensorScalar<int64_t>);

    // Test invalid scalar types
    EXPECT_FALSE(TensorScalar<std::string>);
    EXPECT_FALSE(TensorScalar<std::vector<int>>);
    EXPECT_FALSE(TensorScalar<NotScalar>);
    EXPECT_FALSE(TensorScalar<void>);
    EXPECT_FALSE(TensorScalar<EnumType>);
}

TEST_F(TensorConceptTest, NumericTypeConstraints) {
    // Test valid numeric types
    EXPECT_TRUE(TensorNumeric<int>);
    EXPECT_TRUE(TensorNumeric<float>);
    EXPECT_TRUE(TensorNumeric<double>);
    EXPECT_TRUE(TensorNumeric<char>);
    EXPECT_TRUE(TensorNumeric<unsigned int>);
    EXPECT_TRUE(TensorNumeric<int64_t>);

    // Boolean is not a numeric type
    EXPECT_FALSE(TensorNumeric<bool>);

    // Test invalid numeric types
    EXPECT_FALSE(TensorNumeric<std::string>);
    EXPECT_FALSE(TensorNumeric<std::vector<int>>);
    EXPECT_FALSE(TensorNumeric<NotScalar>);
    EXPECT_FALSE(TensorNumeric<void>);
    EXPECT_FALSE(TensorNumeric<EnumType>);
}

TEST_F(TensorConceptTest, FloatTypeConstraints) {
    // Test valid floating-point types
    EXPECT_TRUE(TensorFloat<float>);
    EXPECT_TRUE(TensorFloat<double>);
    EXPECT_TRUE(TensorFloat<long double>);

    // Test invalid floating-point types
    EXPECT_FALSE(TensorFloat<int>);
    EXPECT_FALSE(TensorFloat<bool>);
    EXPECT_FALSE(TensorFloat<char>);
    EXPECT_FALSE(TensorFloat<unsigned int>);
    EXPECT_FALSE(TensorFloat<std::string>);
    EXPECT_FALSE(TensorFloat<std::vector<float>>);
    EXPECT_FALSE(TensorFloat<NotScalar>);
}

TEST_F(TensorConceptTest, TypeCheckUtility) {
    // Test scalar type checks
    EXPECT_TRUE((TypeCheck<int>::is_scalar));
    EXPECT_TRUE((TypeCheck<float>::is_scalar));
    EXPECT_TRUE((TypeCheck<double>::is_scalar));
    EXPECT_TRUE((TypeCheck<bool>::is_scalar));

    // Test numeric type checks
    EXPECT_TRUE((TypeCheck<int>::is_numeric));
    EXPECT_TRUE((TypeCheck<float>::is_numeric));
    EXPECT_TRUE((TypeCheck<double>::is_numeric));
    EXPECT_FALSE((TypeCheck<bool>::is_numeric));

    // Test float type checks
    EXPECT_FALSE((TypeCheck<int>::is_float));
    EXPECT_TRUE((TypeCheck<float>::is_float));
    EXPECT_TRUE((TypeCheck<double>::is_float));
    EXPECT_FALSE((TypeCheck<bool>::is_float));
}

TEST_F(TensorConceptTest, SupportedOperations) {
    // Check operations supported by different types
    EXPECT_TRUE((TypeCheck<float>::supports_operation<std::plus<float>>()));
    EXPECT_TRUE((TypeCheck<float>::supports_operation<std::minus<float>>()));
    EXPECT_TRUE(
        (TypeCheck<float>::supports_operation<std::multiplies<float>>()));
    EXPECT_TRUE((TypeCheck<float>::supports_operation<std::divides<float>>()));

    EXPECT_TRUE((TypeCheck<int>::supports_operation<std::plus<int>>()));
    EXPECT_TRUE((TypeCheck<int>::supports_operation<std::minus<int>>()));
    EXPECT_TRUE((TypeCheck<int>::supports_operation<std::multiplies<int>>()));
    EXPECT_TRUE((TypeCheck<int>::supports_operation<std::divides<int>>()));

    // Boolean supports basic arithmetic but not division
    EXPECT_TRUE((TypeCheck<bool>::supports_operation<std::plus<bool>>()));
    EXPECT_TRUE((TypeCheck<bool>::supports_operation<std::minus<bool>>()));
    EXPECT_TRUE((TypeCheck<bool>::supports_operation<std::multiplies<bool>>()));
    EXPECT_FALSE((TypeCheck<bool>::supports_operation<std::divides<bool>>()));

    EXPECT_TRUE((TypeCheck<float>::supports_operation<CustomFloatOp>()));
    EXPECT_TRUE((TypeCheck<double>::supports_operation<CustomFloatOp>()));
    EXPECT_FALSE((TypeCheck<int>::supports_operation<CustomFloatOp>()));
    EXPECT_FALSE((TypeCheck<bool>::supports_operation<CustomFloatOp>()));
}