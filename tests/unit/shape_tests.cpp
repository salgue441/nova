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
}