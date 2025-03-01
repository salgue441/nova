#pragma once

#include <fmt/format.h>
#include <tbb/concurrent_vector.h>

#include <brezel/core/error/error_code.hpp>
#include <brezel/core/macros.hpp>
#include <memory>
#include <ranges>
#include <string>
#include <tl/expected.hpp>
#include <vector>

namespace brezel::core::error {
/**
 * @brief Base error class with rich context and history tracking
 *
 */
class BREZEL_API Error : public std::runtime_error {
public:
    static inline tbb::concurrent_vector<std::weak_ptr<const ErrorContext>>
        error_history;

    template <typename... Args>
    explicit Error(fmt::format_string<Args...> fmt, Args&&... args)
        : std::runtime_error(fmt::format(fmt, std::forward<Args>(args)...)),
          m_context(std::make_shared<ErrorContext>()) {
        error_history.push_back(m_context);
    }

    template <typename... Args>
    explicit Error(const ErrorCode& code, fmt::format_string<Args...> fmt,
                   Args&&... args)
        : std::runtime_error(fmt::format(fmt, std::forward<Args>(args)...)),
          m_context(std::make_shared<ErrorContext>()),
          m_code(code) {
        error_history.push_back(m_context);
    }

    // Context accessors
    const ErrorContext* context() const noexcept { return m_context.get(); }
    const ErrorCode* code() const noexcept {
        return m_code ? &*m_code : nullptr;
    }

    void add_note(std::string note) {
        m_context->notes.push_back(std::move(note));
    }

    const std::vector<std::string>& notes() const noexcept {
        return m_context->notes;
    }

    // Static error history access
    static auto error_history_view() {
        return error_history |
               std::views::filter([](const auto& wp) { return !wp.expired(); });
    }

protected:
    std::shared_ptr<ErrorContext> m_context;
    std::optional<ErrorCode> m_code;
};

// Derived error classes
class BREZEL_API RuntimeError : public Error {
public:
    using Error::Error;
};

class BREZEL_API LogicError : public Error {
public:
    using Error::Error;
};

class BREZEL_API InvalidArgument : public LogicError {
public:
    using LogicError::LogicError;
};

class BREZEL_API NotImplemented : public RuntimeError {
public:
    using RuntimeError::RuntimeError;
};

// Result type for error handling
template <typename T>
using Result = tl::expected<T, std::shared_ptr<Error>>;
}  // namespace brezel::core::error

#define BREZEL_ENSURE(condition, ...)                             \
    do {                                                          \
        if (!(condition)) {                                       \
            throw ::brezel::core::error::LogicError(__VA_ARGS__); \
        }                                                         \
    } while (0)

#define BREZEL_CHECK(condition, ...)                                \
    do {                                                            \
        if (!(condition)) {                                         \
            throw ::brezel::core::error::RuntimeError(__VA_ARGS__); \
        }                                                           \
    } while (0)

#define BREZEL_THROW_IF(condition, exception_type, ...)               \
    do {                                                              \
        if (condition) {                                              \
            throw ::brezel::core::error::exception_type(__VA_ARGS__); \
        }                                                             \
    } while (0)

#define BREZEL_NOT_IMPLEMENTED()                 \
    throw ::brezel::core::error::NotImplemented( \
        "Function not implemented: {}", __FUNCTION__)

#define BREZEL_UNREACHABLE()                                             \
    throw ::brezel::core::error::LogicError("Unreachable code at {}:{}", \
                                            __FILE__, __LINE__)
