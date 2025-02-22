#include <gtest/gtest.h>

#include <brezel/core/error/error_category.hpp>
#include <future>
#include <thread>
#include <vector>

using namespace brezel::core::error;

class ErrorCategoryTest : public ::testing::Test {};

TEST_F(ErrorCategoryTest, SystemCategoryBasics) {
    const auto& category = SystemCategory::instance();
    EXPECT_EQ(category.name(), "System");

    auto message = category.message(ENOENT);
    EXPECT_FALSE(message.empty());
}

TEST_F(ErrorCategoryTest, RuntimeCategoryBasics) {
    const auto& category = RuntimeCategory::instance();
    EXPECT_EQ(category.name(), "Runtime");

    auto msg1 = category.message(
        static_cast<int>(RuntimeCategory::Code::InvalidOperation));
    EXPECT_EQ(msg1, "Invalid operation");

    auto msg2 = category.message(9999);
    EXPECT_EQ(msg2, "Unknown error code: 9999");
}

TEST_F(ErrorCategoryTest, LogicCategoryBasics) {
    const auto& category = LogicCategory::instance();
    EXPECT_EQ(category.name(), "Logic");

    auto msg1 = category.message(
        static_cast<int>(LogicCategory::Code::InvalidArgument));
    EXPECT_EQ(msg1, "Invalid argument");

    auto msg2 =
        category.message(static_cast<int>(LogicCategory::Code::OutOfRange));
    EXPECT_EQ(msg2, "Out of range");
}

TEST_F(ErrorCategoryTest, MessageCachingConcurrent) {
    const auto& category = RuntimeCategory::instance();
    constexpr int num_threads = 8;
    std::vector<std::future<std::string>> futures;

    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, [&category]() {
            return category.message(
                static_cast<int>(RuntimeCategory::Code::InvalidOperation));
        }));
    }

    std::string first_message = futures[0].get();
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(futures[i].get(), first_message);
    }
}

TEST_F(ErrorCategoryTest, CategorySingletonThreadSafety) {
    constexpr int num_threads = 8;
    std::vector<std::future<const RuntimeCategory*>> futures;

    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(
            std::launch::async, []() { return &RuntimeCategory::instance(); }));
    }

    const RuntimeCategory* first_instance = futures[0].get();
    for (int i = 1; i < num_threads; ++i) {
        EXPECT_EQ(futures[i].get(), first_instance);
    }
}

TEST_F(ErrorCategoryTest, ImmovableUncopyable) {
    EXPECT_FALSE(std::is_copy_constructible_v<RuntimeCategory>);
    EXPECT_FALSE(std::is_copy_assignable_v<RuntimeCategory>);
    EXPECT_FALSE(std::is_move_constructible_v<RuntimeCategory>);
    EXPECT_FALSE(std::is_move_assignable_v<RuntimeCategory>);
}
