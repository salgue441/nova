#include <gtest/gtest.h>

#include <brezel/core/error/error.hpp>
#include <brezel/tensor/stride.hpp>
#include <vector>

using namespace brezel::tensor;
using namespace brezel::core::error;

class StridesTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(StridesTest, DefaultConstructor) {
    Strides strides;
    EXPECT_TRUE(strides.empty());
    EXPECT_EQ(strides.size(), 0);
}

TEST_F(StridesTest, ConstructFromShape) {
    // 1D shape
    {
        Shape shape{4};
        Strides strides(shape);
        EXPECT_EQ(strides.size(), 1);
        EXPECT_EQ(strides[0], 1);
    }

    // 2D shape (2x3)
    {
        Shape shape{2, 3};
        Strides strides(shape);
        EXPECT_EQ(strides.size(), 2);
        EXPECT_EQ(strides[0], 3);  // stride for first dimension
        EXPECT_EQ(strides[1], 1);  // stride for second dimension
    }

    // 3D shape (2x3x4)
    {
        Shape shape{2, 3, 4};
        Strides strides(shape);
        EXPECT_EQ(strides.size(), 3);
        EXPECT_EQ(strides[0], 12);  // 3 * 4
        EXPECT_EQ(strides[1], 4);   // 4
        EXPECT_EQ(strides[2], 1);   // 1
    }
}

TEST_F(StridesTest, InitializerListConstructor) {
    Strides strides{12, 4, 1};
    EXPECT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
}

TEST_F(StridesTest, AccessOperations) {
    Strides strides{12, 4, 1};

    // Test operator[]
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);

    // Test at() with valid indices
    EXPECT_EQ(strides.at(0), 12);
    EXPECT_EQ(strides.at(1), 4);
    EXPECT_EQ(strides.at(2), 1);

    // Test at() with invalid index
    try {
        [[maybe_unused]] auto val = strides.at(3);
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(),
                     "Index 3 out of range for strides with 3 dimensions");
    }
}

TEST_F(StridesTest, LinearIndexCalculation) {
    Shape shape{2, 3, 4};
    Strides strides(shape);

    // Test various index combinations
    std::vector<int64_t> indices{0, 0, 0};
    EXPECT_EQ(strides.get_linear_index(indices), 0);

    indices = {0, 0, 1};
    EXPECT_EQ(strides.get_linear_index(indices), 1);

    indices = {0, 1, 0};
    EXPECT_EQ(strides.get_linear_index(indices), 4);

    indices = {1, 0, 0};
    EXPECT_EQ(strides.get_linear_index(indices), 12);

    indices = {1, 2, 3};
    EXPECT_EQ(strides.get_linear_index(indices), 23);

    // Test error case with wrong number of indices
    std::vector<int64_t> wrong_indices{0, 0};
    try {
        [[maybe_unused]] auto idx = strides.get_linear_index(wrong_indices);
        FAIL() << "Expected LogicError";
    } catch (const LogicError& e) {
        EXPECT_STREQ(e.what(), "Expected 3 indices but got 2");
    }
}

TEST_F(StridesTest, ContiguityCheck) {
    // Test contiguous cases
    {
        Shape shape{2, 3, 4};
        Strides strides(shape);
        EXPECT_TRUE(strides.is_contiguous(shape));
    }

    {
        Shape shape{5};
        Strides strides(shape);
        EXPECT_TRUE(strides.is_contiguous(shape));
    }

    // Test non-contiguous cases
    {
        Shape shape{2, 3, 4};
        Strides strides{8, 3, 1};  // Non-standard strides
        EXPECT_FALSE(strides.is_contiguous(shape));
    }

    {
        Shape shape{2, 3};
        Strides strides{3, 2};  // Wrong strides for shape
        EXPECT_FALSE(strides.is_contiguous(shape));
    }

    // Test empty case
    {
        Shape shape;
        Strides strides;
        EXPECT_TRUE(strides.is_contiguous(shape));
    }

    // Test mismatched dimensions
    {
        Shape shape{2, 3};
        Strides strides{12, 4, 1};  // Different number of dimensions
        EXPECT_FALSE(strides.is_contiguous(shape));
    }
}

TEST_F(StridesTest, StringRepresentation) {
    // Empty strides
    {
        Strides strides;
        EXPECT_EQ(strides.to_string(), "()");
    }

    // 1D strides
    {
        Strides strides{1};
        EXPECT_EQ(strides.to_string(), "(1)");
    }

    // 2D strides
    {
        Strides strides{3, 1};
        EXPECT_EQ(strides.to_string(), "(3, 1)");
    }

    // 3D strides
    {
        Strides strides{12, 4, 1};
        EXPECT_EQ(strides.to_string(), "(12, 4, 1)");
    }
}

TEST_F(StridesTest, ComparisonOperators) {
    Strides strides1{12, 4, 1};
    Strides strides2{12, 4, 1};
    Strides strides3{12, 4, 2};
    Strides strides4{12, 4};

    EXPECT_EQ(strides1, strides2);
    EXPECT_NE(strides1, strides3);
    EXPECT_NE(strides1, strides4);
}

TEST_F(StridesTest, IteratorSupport) {
    Strides strides{12, 4, 1};

    // Test range-based for loop
    std::vector<int64_t> expected{12, 4, 1};
    size_t idx = 0;
    for (const auto& stride : strides) {
        EXPECT_EQ(stride, expected[idx++]);
    }

    // Test iterator operations
    auto it = strides.begin();
    EXPECT_EQ(*it, 12);
    ++it;
    EXPECT_EQ(*it, 4);
    ++it;
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(it, strides.end());
}