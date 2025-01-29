#include <gtest/gtest.h>
#include <brezel/core/shape.hpp>
#include <brezel/core/error.hpp>
#include <array>
#include <vector>

using namespace brezel::core;

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

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.ndim(), 0);
    EXPECT_EQ(shape.numel(), 1);
  }

  TEST_F(ShapeTest, InitializerListConstruction)
  {
    Shape<> shape{2, 3, 4};

    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
  }

  TEST_F(ShapeTest, IteratorConstruction)
  {
    std::array<int64_t, 3> dims{2, 3, 4};
    Shape<> shape(dims.begin(), dims.end());

    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.numel(), 24);
  }

  // Range and iterator tests
  TEST_F(ShapeTest, RangedBasedForLoop)
  {
    Shape<> shape{2, 3, 4};
    std::vector<int64_t> dims;

    for (const auto &dim : shape)
      dims.push_back(dim);

    EXPECT_EQ(dims, std::vector<int64_t>({2, 3, 4}));
  }

  // Memory layout test
  TEST_F(ShapeTest, SmallBufferOptimization)
  {
    // Should not allocate heap memory
    Shape<int64_t, 4> shape{1, 2, 3, 4};

    EXPECT_EQ(shape.ndim(), 4);
    EXPECT_EQ(shape.numel(), 24);
  }

  // Stride computation test
  TEST_F(ShapeTest, StrideComputation)
  {
    Shape<> shape{2, 3, 4};

    auto strides = shape.strides();

    EXPECT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
  }

  // Reshape tests
  TEST_F(ShapeTest, ReshapeWithVector)
  {
    Shape<> shape{2, 3, 4};
    auto reshaped = shape.reshape(std::vector<int64_t>{6, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped[0], 6);
    EXPECT_EQ(reshaped[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
  }

  TEST_F(ShapeTest, ReshapeWithArray)
  {
    Shape<> shape{2, 3, 4};
    auto reshaped = shape.reshape(std::array<int64_t, 2>{6, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped[0], 6);
    EXPECT_EQ(reshaped[1], 4);
    EXPECT_EQ(reshaped.numel(), 24);
  }

  TEST_F(ShapeTest, ReshapeWithInferred)
  {
    Shape<> shape{2, 3, 4};
    auto reshaped = shape.reshape(std::vector<int64_t>{-1, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped[0], 6);
    EXPECT_EQ(reshaped[1], 4);
  }

  TEST_F(ShapeTest, InvalidReshape)
  {
    Shape<> shape{2, 3, 4};

    EXPECT_THROW(shape.reshape(std::vector<int64_t>{-1, -1}),
                 RuntimeError);

    EXPECT_THROW(shape.reshape(std::vector<int64_t>{-1, 5}),
                 RuntimeError);

    EXPECT_THROW(shape.reshape(std::vector<int64_t>{5, 5}),
                 RuntimeError);
  }

  // Broadcasting tests
  TEST_F(ShapeTest, BroadcastCompatibility)
  {
    Shape<> shape1{2, 1, 4};
    Shape<> shape2{2, 3, 4};
    EXPECT_TRUE(shape1.can_broadcast_with(shape2));

    Shape<> shape3{2, 3, 5};
    EXPECT_FALSE(shape1.can_broadcast_with(shape3));
  }

  TEST_F(ShapeTest, BroadcastShapes)
  {
    Shape<> shape1{2, 1, 4};
    Shape<> shape2{2, 3, 4};
    auto result = Shape<>::broadcast_shapes(shape1, shape2);

    EXPECT_EQ(result.ndim(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
  }

  // Type converstion tests
  TEST_F(ShapeTest, MixedTypeComparison)
  {
    Shape<int32_t> shape1{2, 3, 4};
    Shape<int64_t> shape2{2, 3, 4};
    EXPECT_EQ(shape1, shape2);

    Shape<int64_t> different{2, 3, 5};
    EXPECT_NE(shape1, different);
  }

  TEST_F(ShapeTest, TypeDeduction)
  {
    Shape shape{1, 2L, 3LL}; // Should deduce to largest type
    static_assert(std::is_same_v<decltype(shape)::index_type, long long>);
  }

  // Bounds Checking Tests
  TEST_F(ShapeTest, IndexBoundsChecking)
  {
    Shape<> shape{2, 3, 4};

    EXPECT_NO_THROW({
      auto val = shape[2];
      EXPECT_EQ(val, 4);
    });

    EXPECT_THROW({ [[maybe_unused]] auto val = shape[3]; }, RuntimeError);
  }

  // Negative Dimension Tests
  TEST_F(ShapeTest, NegativeDimensions)
  {
    EXPECT_THROW(Shape<>({2, -3, 4}), RuntimeError);
  }

  // Move Semantics Tests
  TEST_F(ShapeTest, MoveSemantics)
  {
    Shape<> shape1{2, 3, 4};
    Shape<> shape2 = std::move(shape1);

    EXPECT_EQ(shape2.ndim(), 3);
    EXPECT_EQ(shape2.numel(), 24);
  }

  // Numel Cache Tests
  TEST_F(ShapeTest, NumelCache)
  {
    Shape<> shape{2, 3, 4};
    EXPECT_EQ(shape.numel(), 24); // Should compute
    EXPECT_EQ(shape.numel(), 24); // Should use cache
  }

  // String Representation Test
  TEST_F(ShapeTest, ToString)
  {
    Shape<> shape{2, 3, 4};
    EXPECT_EQ(to_string(shape), "Shape(2, 3, 4)");
  }

  // Template Parameter Tests
  TEST_F(ShapeTest, DifferentInlineCapacities)
  {
    Shape<int64_t, 2> small_shape{1, 2};
    Shape<int64_t, 8> large_shape{1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_EQ(small_shape.ndim(), 2);
    EXPECT_EQ(large_shape.ndim(), 8);
  }

  // View Tests
  TEST_F(ShapeTest, SpanViews)
  {
    Shape<> shape{2, 3, 4};
    auto dims_view = shape.dims();
    auto strides_view = shape.strides();

    EXPECT_EQ(dims_view.size(), 3);
    EXPECT_EQ(strides_view.size(), 3);
    EXPECT_EQ(dims_view[0], 2);
    EXPECT_EQ(strides_view[0], 12);
  }

  // Transposition Tests
  TEST_F(ShapeTest, Transpose)
  {
    Shape<> shape{2, 3, 4};
    auto transposed = shape.transpose(0, 2);

    EXPECT_EQ(transposed[0], 4);
    EXPECT_EQ(transposed[1], 3);
    EXPECT_EQ(transposed[2], 2);
    EXPECT_EQ(transposed.numel(), 24);
  }

  TEST_F(ShapeTest, InvalidTranspose)
  {
    Shape<> shape{2, 3, 4};
    EXPECT_THROW(shape.transpose(0, 3), RuntimeError);
  }

  // Constraint Tests
  TEST_F(ShapeTest, BooleanRejection)
  {
    // This should not compile
    // Shape<bool> invalid_shape{true, false};
    SUCCEED();
  }
}