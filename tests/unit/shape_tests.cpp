#include <gtest/gtest.h>
#include <nova/core/shape.hpp>
#include <nova/core/error.hpp>

using namespace nova::core;

namespace testing
{
  class ShapeTest : public ::testing::Test
  {
  protected:
    void SetUp() override {}
    void TearDown() override {}
  };

  TEST_F(ShapeTest, DefaultConstruction)
  {
    Shape<> shape;

    EXPECT_EQ(shape.ndim(), 0);
    EXPECT_EQ(shape.numel(), 0);

    EXPECT_EQ(shape.dims().size(), 0);
    EXPECT_EQ(shape.strides().size(), 0);

    EXPECT_EQ(shape.dims().empty(), true);
    EXPECT_EQ(shape.strides().empty(), true);
  }

  TEST_F(ShapeTest, VectorConstruction)
  {
    Shape<> shape({2, 3, 4});

    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
  }

  TEST_F(ShapeTest, VariadicConstruction)
  {
    Shape<> shape(2, 3, 4);

    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
  }

  // Stride tests
  TEST_F(ShapeTest, StrideComputation)
  {
    Shape<> shape({2, 3, 4});
    auto reshaped = shape.reshape(std::vector<int64_t>{6, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped[0], 6);
    EXPECT_EQ(reshaped[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
  }

  TEST_F(ShapeTest, StrideComputationInferred)
  {
    Shape<> shape({2, 3, 4});
    auto reshaped = shape.reshape(std::vector<int64_t>{-1, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped[0], 6);
    EXPECT_EQ(reshaped[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
  }

  TEST_F(ShapeTest, InvalidReshape)
  {
    Shape<> shape({2, 3, 4});

    EXPECT_THROW(shape.reshape(std::vector<int64_t>{-1, -1}),
                 Error);
  }

  // Broadcasting
  TEST_F(ShapeTest, BroadcastCompatibility)
  {
    Shape<> shape1({2, 1, 4});
    Shape<> shape2({2, 3, 4});
    EXPECT_TRUE(shape1.can_broadcast_with(shape2));

    Shape<> shape3({2, 3, 5});
    EXPECT_FALSE(shape1.can_broadcast_with(shape3));
  }

  TEST_F(ShapeTest, BroadcastShapes)
  {
    Shape<> shape1({2, 1, 4});
    Shape<> shape2({2, 3, 4});

    auto result = Shape<>::broadcast_shapes(shape1, shape2);
    EXPECT_EQ(result.ndim(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
  }

  // Mixed types tests
  TEST_F(ShapeTest, MixedTypeComparison)
  {
    Shape<int32_t> shape32(2, 3, 4);
    Shape<int64_t> shape64(2, 3, 4);
    EXPECT_EQ(shape32, shape64);

    Shape<int64_t> different(2, 3, 5);
    EXPECT_NE(shape32, different);
  }

  TEST_F(ShapeTest, MixedTypeBroadcasting)
  {
    Shape<int32_t> shape32(2, 1, 4);
    Shape<int64_t> shape64(2, 3, 4);
    EXPECT_TRUE(shape32.can_broadcast_with(shape64));

    auto result = Shape<int32_t>::broadcast_shapes(shape32, shape64);
    EXPECT_EQ(result.ndim(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
  }

  // Error cases
  TEST_F(ShapeTest, DimensionOutOfRange)
  {
    Shape<> shape({2, 3, 4});
    EXPECT_THROW(shape.dim(3), Error);
  }

  TEST_F(ShapeTest, InvalidTranspose)
  {
    Shape<> shape({2, 3, 4});
    EXPECT_THROW(shape.transpose(0, 3), Error);
  }

  // Performance critical operations
  TEST_F(ShapeTest, NumelComputation)
  {
    Shape<> shape({100, 100, 100});
    EXPECT_EQ(shape.numel(), 1000000);
  }

  // Custom allocator test
  TEST_F(ShapeTest, CustomAllocator)
  {
    Shape<int64_t, std::allocator<int64_t>> shape({2, 3, 4});
    EXPECT_EQ(shape.numel(), 24);
  }

  // Negative dimension tests
  TEST_F(ShapeTest, NegativeDimensions)
  {
    std::vector<int64_t> dims{2, -3, 4};
    EXPECT_THROW(Shape<>(dims), Error);
  }

  TEST_F(ShapeTest, ExplicitTypes)
  {
    Shape32 shape32({2, 3, 4});
    EXPECT_EQ(shape32.ndim(), 3);

    Shape64 shape64({2, 3, 4});
    EXPECT_EQ(shape64.ndim(), 3);
  }
} // namespace testing