#pragma once

#include <tbb/concurrent_vector.h>

#include <boost/circular_buffer.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <brezel/core/macros.hpp>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <source_location>
#include <stacktrace>

namespace brezel::core::error {
// Forward declarations
class ErrorCode;
class ErrorCategory;
class Error;
class RuntimeError;
class LogicError;

namespace detail {
BREZEL_NORETURN void assert_failure(const char* condition, const char* message,
                                    const char* file, unsigned line);
}  // namespace detail

/**
 * @brief Rich error context containing diagnostic information
 *
 * @details Stores detailed error context including source location,
 * stacktrace, and optional additional diagnostics data using efficient
 * containers.
 */
struct ErrorContext {
    std::source_location location;
    std::stacktrace stacktrace;
    boost::container::small_vector<std::string, 4> notes;
    std::optional<ErrorCode> code;

    /// @brief Thread-safe error history for debugging
    static inline tbb::concurrent_vector<std::weak_ptr<const ErrorContext>>
        error_history;
};

/**
 * @brief Base exception class for all brezel errors
 *
 * @details Exception base class with rich context information,
 * efficient storage, and thread-safe error tracking.
 */
class BREZEL_API Error : public std::exception {
    /**
     * @brief Constructs an error with formatted message
     *
     * @tparam Args Format argument types
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    BREZEL_NODISCARD explicit Error(std::format_string<Args...> fmt,
                                    Args&&... args)
        : m_context(std::make_shared<ErrorContext>()),
          m_message(std::format(fmt, std::forward<Args>(args)...)) {
        init_context();
    }

    /**
     * @brief Constructs an error with error code and formatted message
     *
     * @tparam Args Format argument types
     * @param code Error code
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    BREZEL_NODISCARD Error(const ErrorCode& code,
                           std::format_string<Args...> fmt, Args&&... args)
        : m_context(std::make_shared<ErrorContext>()),
          m_message(std::format(fmt, std::forward<Args>(args)...)) {
        m_context->code = code;

        init_context();
    }

    // Access methods
    /**
     * @brief Gets the error message
     * @return Error message
     */
    BREZEL_NODISCARD const char* what() const noexcept override {
        return m_message.c_str();
    }

    /**
     * @brief Gets the error context
     * @return Shared pointer to error context
     */
    BREZEL_NODISCARD std::shared_ptr<const ErrorContext> context()
        const noexcept {
        return m_context;
    }

    /**
     * @brief Gets the error code if present
     * @return Optional error code
     */
    BREZEL_NODISCARD const ErrorCode* code() const noexcept {
        return m_context->code.has_value() ? &m_context->code.value() : nullptr;
    }

    /**
     * @brief Adds a note to the error context
     *
     * @param note Note to add
     */
    void add_note(std::string note) {
        m_context->notes.emplace_back(std::move(note));
    }

    /**
     * @brief Gets the error notes
     * @return Range of error notes
     */
    BREZEL_NODISCARD auto notes() const noexcept {
        return std::ranges::views::all(m_context->notes);
    }

    /**
     * @brief Gets the error history
     * @return Range of recent errors
     */
    BREZEL_NODISCARD static auto error_history() {
        return std::ranges::views::all(ErrorContext::error_history) |
               std::ranges::views::filter(
                   [](const auto& wp) { return !wp.expired(); });
    }

protected:
    std::shared_ptr<ErrorContext> m_context;
    std::string m_message;

private:
    /**
     * @brief Initializes the error context information
     */
    void init_context() {
        m_context->location = std::source_location::current();
        m_context->stacktrace = std::stacktrace::current();

        ErrorContext::error_history.push_back(m_context);
    }
};

/**
 * @brief Runtime error class for errors that can only be detected at runtime
 */
class BREZEL_API RuntimeError : public Error {
public:
    template <typename... Args>
    explicit RuntimeError(std::format_string<Args...> fmt, Args&&... args)
        : Error(std::forward<std::format_string<Args...>>(fmt),
                std::forward<Args>(args)...) {}

    template <typename... Args>
    RuntimeError(const ErrorCode& code, std::format_string<Args...> fmt,
                 Args&&... args)
        : Error(code, std::forward<std::format_string<Args...>>(fmt),
                std::forward<Args>(args)...) {}
};

/**
 * @brief Logic error class for programming errors that could be detected at
 * compile time
 */
class BREZEL_API LogicError : public Error {
public:
    using Error::Error;
};

// Specialized error classes with boost::static_vector for fixed-size storage
class BREZEL_API InvalidArgument : public LogicError {
    using LogicError::LogicError;

private:
    boost::container::static_vector<std::string, 4> m_arg_info;
};

class BREZEL_API OutOfRange : public LogicError {
    using LogicError::LogicError;

private:
    boost::container::static_vector<size_t, 2> m_range_info;
};

class BREZEL_API InvalidOperation : public RuntimeError {
    using RuntimeError::RuntimeError;

private:
    boost::circular_buffer<std::string> m_operation_history{16};
};

class BREZEL_API NotImplemented : public RuntimeError {
    using RuntimeError::RuntimeError;
};

class BREZEL_API InvalidOperation : public RuntimeError {
    using RuntimeError::RuntimeError;
};

// Error monad type for error handling without exceptions
template <typename T, typename E = Error>
using Result = std::expected<T, E>;

// Helper functions
namespace datail {
/**
 * @brief Handles assertion failures
 *
 * @param condition Failed condition string
 * @param message Error message
 * @param file Source file
 * @param line Line number
 */
BREZEL_NORETURN void assert_failure(const char* condition, const char* message,
                                    const char* file, unsigned line) {
    throw RuntimeError("Assertion failed: {} at {}:{} - {}", condition, file,
                       line, message);
}
}  // namespace datail
}  // namespace brezel::core::error

// Error checking macros
#define BREZEL_CHECK(condition, message, ...)                  \
    do {                                                       \
        if (BREZEL_PREDICT_FALSE(!(condition))) {              \
            throw ::BREZEL::RuntimeError(message __VA_ARGS__); \
        }                                                      \
    } while (0)

#define BREZEL_ENSURE(condition, message, ...)               \
    do {                                                     \
        if (BREZEL_PREDICT_FALSE(!(condition))) {            \
            throw ::BREZEL::LogicError(message __VA_ARGS__); \
        }                                                    \
    } while (0)

#define BREZEL_THROW_IF(condition, exception_type, message, ...) \
    do {                                                         \
        if (BREZEL_PREDICT_FALSE(condition)) {                   \
            throw exception_type(message __VA_ARGS__);           \
        }                                                        \
    } while (0)

#define BREZEL_NOT_IMPLEMENTED()                 \
    throw ::brezel::core::error::NotImplemented( \
        "Function not implemented: {}", __FUNCTION__)

#define BREZEL_UNREACHABLE()                         \
    throw ::brezel::core::error::brezel::LogicError( \
        "Unreachable code reached at {}:{}", __FILE__, __LINE__)

#define BREZEL_INVALID_OPERATION()                                       \
    throw ::brezel::core::error::InvalidOperation("Operation failed {}", \
                                                  __FILE__)