#include <gtest/gtest.h>

#include <barrier>
#include <brezel/core/error/error.hpp>
#include <future>
#include <vector>

using namespace brezel::core::error;

class ErrorTest : public ::testing::Test {};

TEST_F(ErrorTest, RuntimeErrorConstruction) {
    try {
        throw RuntimeError("Test error {}", 42);
        FAIL() << "Expected RuntimeError";
    } catch (const RuntimeError& e) {
        EXPECT_STREQ(e.what(), "Test error 42");
        ASSERT_NE(e.context(), nullptr);
        EXPECT_FALSE(e.context()->stacktrace.empty());
        EXPECT_TRUE(e.notes().empty());
    }
}

TEST_F(ErrorTest, ErrorWithCode) {
    try {
        auto code = make_error_code(RuntimeCategory::Code::InvalidOperation);
        throw RuntimeError(code, "Error with code");
    } catch (const RuntimeError& e) {
        ASSERT_NE(e.code(), nullptr);
        EXPECT_EQ(e.code()->category().name(), "Runtime");
        EXPECT_EQ(e.code()->value(),
                  static_cast<int>(RuntimeCategory::Code::InvalidOperation));
    }
}

TEST_F(ErrorTest, ErrorNotes) {
    try {
        RuntimeError error("Base error");
        error.add_note("Note 1");
        error.add_note("Note 2");
        throw error;
    } catch (const RuntimeError& e) {
        auto notes =
            std::vector<std::string>{e.notes().begin(), e.notes().end()};
        ASSERT_EQ(notes.size(), 2);
        EXPECT_EQ(notes[0], "Note 1");
        EXPECT_EQ(notes[1], "Note 2");
    }
}

TEST_F(ErrorTest, ErrorHistoryTracking) {
    auto initial_count = std::ranges::distance(Error::error_history());

    try {
        throw RuntimeError("Test error");
    } catch (const RuntimeError&) {
        auto final_count = std::ranges::distance(Error::error_history());
        EXPECT_EQ(final_count, initial_count + 1);
    }
}

TEST_F(ErrorTest, ThreadSafeErrorTracking) {
    constexpr int kThreads = 8;
    std::vector<std::future<void>> futures;

    auto initial_count = std::ranges::distance(Error::error_history());

    // Create barrier to synchronize threads
    std::barrier sync_point(kThreads + 1);

    for (int i = 0; i < kThreads; ++i) {
        futures.push_back(std::async(std::launch::async, [&sync_point]() {
            try {
                throw RuntimeError("Thread error");
            } catch (const RuntimeError&) {
                sync_point.arrive_and_wait();
            }
        }));
    }

    sync_point.arrive_and_wait();

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }

    auto final_count = std::ranges::distance(Error::error_history());
    EXPECT_EQ(final_count, initial_count + kThreads);
}

TEST_F(ErrorTest, ErrorMacros) {
    EXPECT_THROW(BREZEL_CHECK(false, "Check failed"), RuntimeError);
    EXPECT_THROW(BREZEL_ENSURE(false, "Ensure failed"), LogicError);
    EXPECT_THROW(BREZEL_THROW_IF(true, InvalidArgument, "Invalid argument"),
                 InvalidArgument);
    EXPECT_THROW(BREZEL_NOT_IMPLEMENTED(), NotImplemented);
    EXPECT_THROW(BREZEL_UNREACHABLE(), LogicError);
}

TEST_F(ErrorTest, MovedError) {
    RuntimeError error("Test");
    error.add_note("Note");

    RuntimeError moved(std::move(error));
    EXPECT_STREQ(moved.what(), "Test");

    auto notes =
        std::vector<std::string>{moved.notes().begin(), moved.notes().end()};
    ASSERT_EQ(notes.size(), 1);
    EXPECT_EQ(notes[0], "Note");
}

TEST_F(ErrorTest, ResultHandling) {
    auto success = Result<int>{42};
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(*success, 42);

    // Create error directly instead of using make_shared
    auto error = std::shared_ptr<RuntimeError>(new RuntimeError("Failed"));
    auto failure = Result<int>{tl::unexpected(error)};
    EXPECT_FALSE(failure.has_value());
    EXPECT_STREQ(failure.error()->what(), "Failed");
}