#include <gtest/gtest.h>

#include <brezel/core/error/error_code.hpp>
#include <future>
#include <thread>

using namespace brezel::core::error;

class ErrorCodeTest : public ::testing::Test {};

TEST_F(ErrorCodeTest, DefaultConstruction) {
    ErrorCode code;
    EXPECT_FALSE(code);
    EXPECT_EQ(code.value(), 0);
    EXPECT_EQ(code.category().name(), LogicCategory::instance().name());
}

TEST_F(ErrorCodeTest, CategoryConstructionAndAccess) {
    auto runtime = make_error_code(RuntimeCategory::Code::InvalidOperation);
    EXPECT_TRUE(runtime);
    EXPECT_EQ(runtime.category().name(), RuntimeCategory::instance().name());

    auto logic = make_error_code(LogicCategory::Code::InvalidArgument);
    EXPECT_TRUE(logic);
    EXPECT_EQ(logic.category().name(), LogicCategory::instance().name());

    auto system = make_system_error(ENOENT);
    EXPECT_TRUE(system);
    EXPECT_EQ(system.category().name(), SystemCategory::instance().name());
}

TEST_F(ErrorCodeTest, MessageCaching) {
    auto code = make_error_code(RuntimeCategory::Code::InvalidOperation);

    const std::string& msg1 = code.message();
    const std::string& msg2 = code.message();

    EXPECT_FALSE(msg1.empty());
    EXPECT_EQ(std::addressof(msg1), std::addressof(msg2));
}

TEST_F(ErrorCodeTest, ThreadSafeMessageCaching) {
    constexpr int kThreads = 8;
    std::vector<std::future<std::string>> futures;

    auto code = make_error_code(RuntimeCategory::Code::InvalidOperation);
    for (int i = 0; i < kThreads; ++i) {
        futures.push_back(
            std::async(std::launch::async, [&code] { return code.message(); }));
    }

    const std::string& baseline = futures[0].get();
    for (int i = 1; i < kThreads; ++i) {
        EXPECT_EQ(futures[i].get(), baseline);
    }
}

TEST_F(ErrorCodeTest, ContextManagement) {
    auto code = make_error_code(RuntimeCategory::Code::InvalidOperation);
    EXPECT_EQ(code.context(), nullptr);

    code.add_context("Test note");
    ASSERT_NE(code.context(), nullptr);
    EXPECT_EQ(code.context()->notes.size(), 1);
    EXPECT_EQ(code.context()->notes[0], "Test note");

    auto location = std::source_location::current();
    code.add_context("Second note", location);
    EXPECT_EQ(code.context()->notes.size(), 2);
    EXPECT_EQ(code.context()->location.line(), location.line());
}

TEST_F(ErrorCodeTest, CopyAndMove) {
    auto original = make_error_code(RuntimeCategory::Code::InvalidOperation);
    original.add_context("Note");

    auto copy = original;
    EXPECT_EQ(copy, original);
    EXPECT_NE(copy.context(), original.context());
    EXPECT_EQ(copy.context()->notes[0], original.context()->notes[0]);

    auto moved = std::move(original);
    EXPECT_EQ(moved, copy);
    EXPECT_NE(moved.context(), copy.context());
}

TEST_F(ErrorCodeTest, Comparison) {
    auto code1 = make_error_code(RuntimeCategory::Code::InvalidOperation);
    auto code2 = make_error_code(RuntimeCategory::Code::InvalidOperation);
    auto code3 = make_error_code(RuntimeCategory::Code::OutOfMemory);
    auto code4 = make_error_code(LogicCategory::Code::InvalidArgument);

    EXPECT_EQ(code1, code2);
    EXPECT_NE(code1, code3);
    EXPECT_NE(code1, code4);
    EXPECT_TRUE(code1 <=> code2 == 0);
    EXPECT_TRUE(code1 <=> code3 != 0);
}