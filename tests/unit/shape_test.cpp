#include <gtest/gtest.h>

#include <brezel/core/error/error.hpp>
#include <brezel/tensor/shape.hpp>
#include <vector>

using namespace brezel::tensor;
using namespace brezel::core::error;

class ShapeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ShapeTest, DefaultConstructor) {
    Shape shape;

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.size(), 0);
    EXPECT_EQ(shape.numel(), 1);  // Scalar has one element
}

TEST_F(ShapeTest, InitializerListConstructor) {
    Shape shape{2, 3, 4};

    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
}

TEST_F(ShapeTest, RangeConstructor) {
    std::vector<int64_t> dims{2, 3, 4};
    Shape shape(dims);

    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
}

TEST_F(ShapeTest, NegativeDimensionsTest) {
    try {
        Shape({2, -3, 4});
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(), "Negative dimension -3 is invalid");
    }
}

TEST_F(ShapeTest, NegativeDimensionsRange) {
    std::vector<int64_t> dims{2, -3, 4};

    try {
        Shape shape(dims);
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(), "Negative dimension -3 is invalid");
    }
}

TEST_F(ShapeTest, NegativeDimensionsPushBack) {
    Shape shape{2, 3, 4};
    try {
        shape.push_back(-1);
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(), "Negative dimension -1 is invalid");
    }
}

TEST_F(ShapeTest, AccessOperations) {
    Shape shape{2, 3, 4};

    // Test operator[]
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);

    // Test at() with valid indices
    EXPECT_EQ(shape.at(0), 2);
    EXPECT_EQ(shape.at(1), 3);
    EXPECT_EQ(shape.at(2), 4);

    // Test at() with invalid indices
    try {
        [[maybe_unused]] auto& val = shape.at(3);
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(),
                     "Index 3 out of range for shape with 3 dimensions");
    }
}

TEST_F(ShapeTest, ModifierOperations) {
    Shape shape{2, 3};

    // Test push_back
    shape.push_back(4);
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[2], 4);

    // Test pop_back
    shape.pop_back();
    EXPECT_EQ(shape.size(), 2);

    // Test clear
    shape.clear();
    EXPECT_TRUE(shape.empty());

    // Test pop_back on empty shape
    try {
        shape.pop_back();
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(), "Cannot pop from empty shape");
    }
}

TEST_F(ShapeTest, NumelCalculation) {
    // Test empty shape (scalar)
    Shape scalar;
    EXPECT_EQ(scalar.numel(), 1);

    // Test 1D shape
    Shape shape1D{5};
    EXPECT_EQ(shape1D.numel(), 5);

    // Test 2D shape
    Shape shape2D{2, 3};
    EXPECT_EQ(shape2D.numel(), 6);

    // Test 3D shape
    Shape shape3D{2, 3, 4};
    EXPECT_EQ(shape3D.numel(), 24);

    // Test shape with zero dimension
    Shape shapeZero{2, 0, 4};
    EXPECT_EQ(shapeZero.numel(), 0);
}