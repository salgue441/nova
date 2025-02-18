#pragma once

#include <fmt/format.h>
#include <tbb/concurrent_vector.h>

#include <boost/circular_buffer.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/stacktrace.hpp>
#include <brezel/core/error/error_code.hpp>
#include <brezel/core/macros.hpp>
#include <memory>
#include <ranges>
#include <source_location>
#include <tl/expected.hpp>

namespace brezel::core::error {
namespace detail {
BREZEL_NORETURN void assert_failure(const char* condition, const char* message,
                                    const char* file, unsigned line);
}  // namespace detail

/// @brief Error context containing diagnostic information
struct BREZEL_ALIGN_CACHE ErrorContext {
    std::source_location location;
    boost::stacktrace::stacktrace stacktrace;
    boost::container::small_vector<std::string, 4> notes;
    std::optional<ErrorCode> code;

    static inline tbb::concurrent_vector<std::weak_ptr<const ErrorContext>>
        error_history;
};

/**
 * @brief Base exception class with rich context and diagnostic information
 *
 */
class BREZEL_API Error : public std::exception {
public:
    /**
     * @brief Constructs an Error with a formatted string
     *
     * @tparam Args Format argument types
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    BREZEL_NODISCARD explicit Error(fmt::format_string<Args...> fmt,
                                    Args&&... args)
        : m_context(std::make_shared<ErrorContext>()),
          m_message(fmt::format(fmt, std::forward<Args>(args)...)) {
        init_context();
    }

    /**
     * @brief Constructs an Error with an ErrorCode and a formatted string
     *
     * @tparam Args Format argument types
     * @param code ErrorCode to be assigned
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    BREZEL_NODISCARD Error(const ErrorCode& code,
                           fmt::format_string<Args...> fmt, Args&&... args)
        : m_context(std::make_shared<ErrorContext>()),
          m_message(fmt::format(fmt, std::forward<Args>(args)...)) {
        m_context->code = code;

        init_context();
    }

    BREZEL_UNCOPYABLE(Error);
    Error(Error&&) noexcept = default;
    Error& operator=(Error&&) = default;
    ~Error() override = default;

    /**
     * @brief Gets the message in c-string format
     *
     * @return char* Pointer to the message
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const char* what()
        const noexcept override {
        return m_message.c_str();
    }

    /**
     * @brief Gets the context of the error
     *
     * @return std::shared_ptr<const ErrorCode> ErrorCode reference
     */
    BREZEL_NODISCARD BREZEL_PURE std::shared_ptr<const ErrorContext> context()
        const noexcept {
        return m_context;
    }

    /**
     * @brief Gets the error code
     *
     * @return ErrorCode value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const ErrorCode* code()
        const noexcept {
        return m_context->code.has_value() ? &m_context->code.value() : nullptr;
    }

    /**
     * @brief Adds a new note to the context
     *
     * @param note Note string to add
     */
    BREZEL_HOT void add_note(std::string note) {
        m_context->notes.emplace_back(std::move(note));
    }

    /**
     * @brief Returns the views for the notes object
     *
     * @return std::ranges::views
     */
    BREZEL_NODISCARD auto notes() const noexcept {
        return std::ranges::views::all(m_context->notes);
    }

    /**
     * @brief Gets the views from error history
     *
     * @return std::ranges::views
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
     * @brief Initializes the context of an Error
     *
     */
    BREZEL_FORCE_INLINE void init_context() {
        m_context->location = std::source_location::current();
        m_context->stacktrace = boost::stacktrace::stacktrace();

        ErrorContext::error_history.push_back(m_context);
    }
};

/**
 * @brief Runtime error class for errors that can only be detected at runtime
 */
class BREZEL_API RuntimeError : public Error {
public:
    using Error::Error;
};

/**
 * @brief Logic error class for programming errors that could be detected at
 * compile time
 */
class BREZEL_API LogicError : public Error {
public:
    using Error::Error;
};

/**
 * @brief Invalid argument error with argument information storage
 */
class BREZEL_API InvalidArgument : public LogicError {
public:
    using LogicError::LogicError;

private:
    boost::container::static_vector<std::string, 4> m_arg_info;
};

/**
 * @brief Out of range error with range boundary storage
 */
class BREZEL_API OutOfRange : public LogicError {
public:
    using LogicError::LogicError;

private:
    boost::container::static_vector<size_t, 2> m_range_info;
};

/**
 * @brief Invalid operation error with operation history
 */
class BREZEL_API InvalidOperation : public RuntimeError {
public:
    using RuntimeError::RuntimeError;

private:
    boost::circular_buffer<std::string> m_operation_history{16};
};

/**
 * @brief Not implemented error for unimplemented functionality
 */
class BREZEL_API NotImplemented : public RuntimeError {
public:
    using RuntimeError::RuntimeError;
};

template <typename T>
using Result = tl::expected<T, std::shared_ptr<Error>>;
}  // namespace brezel::core::error

#define BREZEL_CHECK(condition, message, ...)                               \
    do {                                                                    \
        if (BREZEL_PREDICT_FALSE(!(condition))) {                           \
            throw ::brezel::core::error::RuntimeError(message __VA_ARGS__); \
        }                                                                   \
    } while (0)

#define BREZEL_ENSURE(condition, message, ...)                            \
    do {                                                                  \
        if (BREZEL_PREDICT_FALSE(!(condition))) {                         \
            throw ::brezel::core::error::LogicError(message __VA_ARGS__); \
        }                                                                 \
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

#define BREZEL_UNREACHABLE()                 \
    throw ::brezel::core::error::LogicError( \
        "Unreachable code reached at {}:{}", __FILE__, __LINE__)