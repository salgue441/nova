#include <gtest/gtest.h>

#include <array>
#include <brezel/tensor/layout.hpp>
#include <vector>

using namespace brezel::tensor;

class LayoutTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(LayoutTest, DefaultConstructor) {
    Shape empty_shape;
    LayoutDescriptor layout(empty_shape);

    EXPECT_EQ(layout.ndim(), 0);
    EXPECT_EQ(layout.numel(), 1);
    EXPECT_TRUE(layout.is_contiguous());
    EXPECT_EQ(layout.device(), DeviceType::CPU);
    EXPECT_EQ(layout.format(), MemoryFormat::Contiguous);
    EXPECT_EQ(layout.layout(), MemoryLayout::RowMajor);
}

TEST_F(LayoutTest, ShapeConstructor) {
    // 1D shape
    {
        Shape shape({5});
        LayoutDescriptor layout(shape);

        EXPECT_EQ(layout.ndim(), 1);
        EXPECT_EQ(layout.strides()[0], 1);
        EXPECT_TRUE(layout.is_contiguous());
    }

    // 2D shape with row-major layout
    {
        Shape shape({3, 4});
        LayoutDescriptor layout(shape, MemoryLayout::RowMajor);

        EXPECT_EQ(layout.ndim(), 2);
        EXPECT_EQ(layout.strides()[0], 4);
        EXPECT_EQ(layout.strides()[1], 1);
        EXPECT_TRUE(layout.is_contiguous());
    }

    // 2D shape with column-major layout
    {
        Shape shape({3, 4});
        LayoutDescriptor layout(shape, MemoryLayout::ColumnMajor);

        EXPECT_EQ(layout.ndim(), 2);
        EXPECT_EQ(layout.strides()[0], 1);
        EXPECT_EQ(layout.strides()[1], 3);
        EXPECT_TRUE(layout.is_contiguous());
    }

    // 3D shape
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape);

        EXPECT_EQ(layout.ndim(), 3);
        EXPECT_EQ(layout.strides()[0], 12);
        EXPECT_EQ(layout.strides()[1], 4);
        EXPECT_EQ(layout.strides()[2], 1);
        EXPECT_TRUE(layout.is_contiguous());
    }
}

TEST_F(LayoutTest, StridesConstructor) {
    Shape shape({2, 3, 4});
    std::vector<int64_t> strides = {24, 8, 1};

    LayoutDescriptor layout(shape, strides);

    EXPECT_EQ(layout.ndim(), 3);
    EXPECT_EQ(layout.strides()[0], 24);
    EXPECT_EQ(layout.strides()[1], 8);
    EXPECT_EQ(layout.strides()[2], 1);
    EXPECT_EQ(layout.layout(), MemoryLayout::Strided);
    EXPECT_FALSE(layout.is_contiguous());
}

TEST_F(LayoutTest, FormatConstructor) {
    // Test NCHW (ChannelsFirst) layout
    {
        Shape shape({2, 3, 4, 5});  // N, C, H, W
        LayoutDescriptor layout(shape, MemoryFormat::ChannelsFirst);

        EXPECT_EQ(layout.ndim(), 4);
        EXPECT_EQ(layout.format(), MemoryFormat::ChannelsFirst);
        EXPECT_EQ(layout.layout(), MemoryLayout::RowMajor);

        // Check strides for NCHW (row-major)
        EXPECT_EQ(layout.strides()[0], 3 * 4 * 5);
        EXPECT_EQ(layout.strides()[1], 4 * 5);
        EXPECT_EQ(layout.strides()[2], 5);
        EXPECT_EQ(layout.strides()[3], 1);
    }

    // Test NHWC (ChannelsLast) layout
    {
        Shape shape({2, 3, 4, 5});  // N, H, W, C
        LayoutDescriptor layout(shape, MemoryFormat::ChannelsLast);

        EXPECT_EQ(layout.ndim(), 4);
        EXPECT_EQ(layout.format(), MemoryFormat::ChannelsLast);
        EXPECT_EQ(layout.layout(), MemoryLayout::Strided);

        // Check strides for NHWC
        EXPECT_EQ(layout.strides()[0], 3 * 4 * 5);
        EXPECT_EQ(layout.strides()[1], 4 * 5);
        EXPECT_EQ(layout.strides()[2], 5);
        EXPECT_EQ(layout.strides()[3], 1);
    }

    // Test invalid shape for format
    {
        Shape shape({2, 3});  // Not 4D
        EXPECT_THROW(LayoutDescriptor(shape, MemoryFormat::ChannelsLast),
                     brezel::core::error::InvalidArgument);
    }
}

TEST_F(LayoutTest, LinearIndexing) {
    // Test row-major indexing
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape);

        // Test various index combinations
        std::array<int64_t, 3> indices = {0, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 0);

        indices = {0, 0, 1};
        EXPECT_EQ(layout.get_linear_index(indices), 1);

        indices = {0, 1, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 4);

        indices = {1, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 12);

        indices = {1, 2, 3};
        EXPECT_EQ(layout.get_linear_index(indices), 23);
    }

    // Test column-major indexing
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape, MemoryLayout::ColumnMajor);

        // Test various index combinations
        std::array<int64_t, 3> indices = {0, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 0);

        indices = {1, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 1);

        indices = {0, 1, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 2);

        indices = {0, 0, 1};
        EXPECT_EQ(layout.get_linear_index(indices), 6);
    }

    // Test custom stride indexing
    {
        Shape shape({2, 3, 4});
        std::vector<int64_t> strides = {16, 4, 1};  // Non-standard strides
        LayoutDescriptor layout(shape, strides);

        std::array<int64_t, 3> indices = {0, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 0);

        indices = {0, 0, 1};
        EXPECT_EQ(layout.get_linear_index(indices), 1);

        indices = {0, 1, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 4);

        indices = {1, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 16);
    }

    // Test with offset
    {
        Shape shape({2, 3});
        LayoutDescriptor layout(shape);
        layout.set_offset(100);

        std::array<int64_t, 2> indices = {0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 100);

        indices = {1, 2};
        EXPECT_EQ(layout.get_linear_index(indices), 105);
    }
}

TEST_F(LayoutTest, IndexToCoordinates) {
    // Test row-major
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape);

        std::array<int64_t, 3> coords;

        layout.get_indices(0, coords);
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
        EXPECT_EQ(coords[2], 0);

        layout.get_indices(1, coords);
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
        EXPECT_EQ(coords[2], 1);

        layout.get_indices(4, coords);
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 1);
        EXPECT_EQ(coords[2], 0);

        layout.get_indices(23, coords);
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 2);
        EXPECT_EQ(coords[2], 3);
    }

    // Test column-major indexing
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape, MemoryLayout::ColumnMajor);

        // Test various index combinations
        std::array<int64_t, 3> indices = {0, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 0);

        indices = {1, 0, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 1);

        indices = {0, 1, 0};
        EXPECT_EQ(layout.get_linear_index(indices), 2);

        indices = {0, 0, 1};
        EXPECT_EQ(layout.get_linear_index(indices), 6);
    }

    // Test with offset
    {
        Shape shape({2, 3});
        LayoutDescriptor layout(shape);
        layout.set_offset(100);

        std::array<int64_t, 2> coords;

        layout.get_indices(100, coords);
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);

        layout.get_indices(105, coords);
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 2);
    }
}

TEST_F(LayoutTest, IsContiguous) {
    // Test row-major contiguous
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape);
        EXPECT_TRUE(layout.is_contiguous());
        EXPECT_TRUE(layout.is_row_contiguous());
        EXPECT_FALSE(layout.is_column_contiguous());
    }

    // Test column-major contiguous
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape, MemoryLayout::ColumnMajor);
        EXPECT_TRUE(layout.is_contiguous());
        EXPECT_FALSE(layout.is_row_contiguous());
        EXPECT_TRUE(layout.is_column_contiguous());
    }

    // Test non-contiguous
    {
        Shape shape({2, 3, 4});
        std::vector<int64_t> strides = {16, 4, 1};  // Non-standard strides
        LayoutDescriptor layout(shape, strides);
        EXPECT_FALSE(layout.is_contiguous());
    }

    // Test with offset
    {
        Shape shape({2, 3, 4});
        LayoutDescriptor layout(shape);
        layout.set_offset(10);
        EXPECT_FALSE(layout.is_contiguous());
    }
}

TEST_F(LayoutTest, Transpose) {
    Shape shape({2, 3, 4});
    LayoutDescriptor layout(shape);

    auto transposed = layout.transpose(0, 2);

    EXPECT_EQ(transposed.strides()[0], 1);
    EXPECT_EQ(transposed.strides()[1], 4);
    EXPECT_EQ(transposed.strides()[2], 12);
}

TEST_F(LayoutTest, Permute) {
    Shape shape({2, 3, 4, 5});
    LayoutDescriptor layout(shape);

    // Permute dimensions to [3, 0, 2, 1]
    std::vector<size_t> perm = {3, 0, 2, 1};
    auto permuted = layout.permute(perm);

    EXPECT_EQ(permuted.shape()[0], 5);
    EXPECT_EQ(permuted.shape()[1], 2);
    EXPECT_EQ(permuted.shape()[2], 4);
    EXPECT_EQ(permuted.shape()[3], 3);

    EXPECT_EQ(permuted.strides()[0], 1);
    EXPECT_EQ(permuted.strides()[1], 60);
    EXPECT_EQ(permuted.strides()[2], 5);
    EXPECT_EQ(permuted.strides()[3], 20);

    // Invalid permutation (duplicate dimensions)
    std::vector<size_t> invalid_perm = {0, 0, 2, 3};
    EXPECT_THROW(layout.permute(invalid_perm),
                 brezel::core::error::InvalidArgument);

    // Invalid permutation (out of bounds)
    std::vector<size_t> oob_perm = {0, 1, 2, 4};
    EXPECT_THROW(layout.permute(oob_perm),
                 brezel::core::error::InvalidArgument);
}