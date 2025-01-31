#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>
#include <brezel/core/macros.hpp>
#include <compare>
#include <memory>
#include <source_location>
#include <string>

#include "error_category.hpp"

namespace brezel::core::error {
/**
 * @brief Error code class combining error value and category
 *
 * @details Thread-safe error code class with efficient storage
 * and caching of error messages. Supports comparison, hashing,
 * and provides rich context information
 */
class BREZEL_API ErrorCode {
public:
    /// @brief Additional context for error code
    struct Context {
        boost::container::small_vector<std::string, 2> notes;
        std::source_location location;
    };

    /**
     * @brief Default constructor creates a success error code
     */
    BREZEL_NODISCARD constexpr ErrorCode() noexcept
        : m_code(0), m_category(&LogicCategory::instance()) {}

    /**
     * @brief Copy constructor
     *
     */
    BREZEL_NODISCARD ErrorCode(const ErrorCode& other) noexcept
        : m_code(other.m_code), m_category(other.m_category) {
        if (other.m_context)
            m_context = std::make_unique<Context>(*other.m_context);
    }

    /**
     * @brief Move constructor
     */
    BREZEL_NODISCARD ErrorCode(ErrorCode&& othher) noexcept = default;

    /**
     * @brief Construct from error code and category
     *
     * @param code Numeric error code
     * @param category Error category
     */
    BREZEL_NODISCARD constexpr ErrorCode(int code,
                                         const ErrorCategory& category) noexcept
        : m_code(code), m_category(&category) {}

    /**
     * @brief Construct from runtime category code
     *
     * @param code Runtime error code
     */
    BREZEL_NODISCARD constexpr ErrorCode(RuntimeCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&RuntimeCategory::instance()) {}

    /**
     * @brief Construct from logic category code
     *
     * @param code Logic error code
     */
    BREZEL_NODISCARD constexpr ErrorCode(LogicCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&LogicCategory::instance()) {}

    /**
     * @brief Construct from optional error code
     *
     * @param opt Optional error code
     */
    BREZEL_NODISCARD explicit ErrorCode(
        const std::optional<ErrorCode>& opt) noexcept
        : ErrorCode(opt.value_or(ErrorCode{})) {}

    // Access methods
    /**
     * @brief Get the error value
     *
     * @return Numeric error value
     */
    BREZEL_NODISCARD constexpr int value() const noexcept { return m_code; }

    /**
     * @brief Get the error category
     *
     * @return Reference to the error category
     */
    BREZEL_NODISCARD constexpr const ErrorCategory& category() const noexcept {
        return *m_category;
    }

    // Methods
    /**
     * @brief Get the error message with optional caching
     *
     * @return Error message string
     */
    BREZEL_NODISCARD std::string message() const {
        auto it = s_messages_cache.find(m_code);
        if (it != s_messages_cache.end())
            return it->second;

        auto msg = category().message(value());
        s_messages_cache.insert({m_code, msg});

        return msg;
    }

    /**
     * @brief Check if the error code represent success
     *
     * @return true if the error code is non-zero
     */
    BREZEL_NODISCARD constexpr explicit operator bool() const noexcept {
        return value() != 0;
    }

    /**
     * @brief Add context information to the error code
     *
     * @param note Additional note about the error
     * @param location Source location (defaults to current)
     */
    void add_context(std::string note, const std::source_location& location =
                                           std::source_location::current()) {
        if (!m_context)
            m_context = std::make_unique<Context>();

        m_context->notes.push_back(std::move(note));
        m_context->location = location;
    }

    /**
     * @brief Get the current context if available
     *
     * @return Optional reference to the error context
     */
    BREZEL_NODISCARD const Context* context() const noexcept {
        return m_context.get();
    }

    // Operators
    /**
     * @brief Three-way comparison operator
     *
     * @param other Error code to compare with
     * @return Comparison result
     */
    BREZEL_NODISCARD constexpr auto operator<=>(
        const ErrorCode& other) const noexcept {
        if (auto cmp = &category() <=> &other.category(); cmp != 0)
            return cmp;

        return value() <=> other.value();
    }

    /**
     * @brief Equality comparison operator
     *
     * @param other Error code to compare with
     * @return true if error codes are equal
     */
    BREZEL_NODISCARD constexpr bool operator==(
        const ErrorCode& other) const noexcept {
        return *this <=> other == 0;
    }

    /**
     * @brief Copy assignment operator
     */
    BREZEL_NODISCARD ErrorCode& operator=(const ErrorCode& other) noexcept {
        if (this != &other) {
            m_code = other.m_code;
            m_category = other.m_category;

            if (other.m_context)
                m_context = std::make_unique<Context>(*other.m_context);

            else
                m_context.reset();
        }

        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    ErrorCode& operator=(ErrorCode&& other) noexcept = default;

private:
    int m_code;
    const ErrorCategory* m_category;
    std::unique_ptr<Context> m_context;

    static inline tbb::concurrent_unordered_map<int, std::string>
        s_messages_cache;
};

// Helper functions to create error code
/**
 * @brief Create a runtime error code
 *
 * @param code Runtime error code
 * @return Error code instance
 */
BREZEL_NODISCARD inline ErrorCode make_error_code(
    RuntimeCategory::Code code) noexcept {
    return ErrorCode{code};
}

/**
 * @brief Create a logic error code
 *
 * @param code Logic error code
 * @return Error code instance
 */
BREZEL_NODISCARD inline ErrorCode make_error_code(
    LogicCategory::Code code) noexcept {
    return ErrorCode{code};
}

/**
 * @brief Creates an error code with custom category
 *
 * @param code Numeric error code
 * @param category Error category
 * @return Error code instance
 */
BREZEL_NODISCARD inline ErrorCode make_error_code(
    int code, const ErrorCategory& category) noexcept {
    return ErrorCode{code, category};
}

/**
 * @brief Creates a system error code
 *
 * @param error_code System error code
 * @return Error code instance
 */
BREZEL_NODISCARD inline ErrorCode make_system_error(int error_code) noexcept {
    return ErrorCode{error_code, SystemCategory::instance()};
}

// Helper macros
/**
 * @brief Try an operation and return its error code on failure
 */
#define BREZEL_TRY(expr)                                                 \
    do {                                                                 \
        if (auto&& result = (expr); !result) {                           \
            return brezel::core::error::make_error_code(result.error()); \
        }                                                                \
    } while (0)

/**
 * @brief Try an operation and assign its result, returning error code on
 * failure
 */
#define BREZEL_TRY_ASSIGN(var, expr)                              \
    auto&& var = (expr);                                          \
    if (!var) {                                                   \
        return brezel::core::error::make_error_code(var.error()); \
    }
}  // namespace brezel::core::error

// Hash support for ErrorCode
using ErrorCode = brezel::core::error::ErrorCode;

template <>
struct std::hash<ErrorCode> {
    /**
     * @brief Hash an error code
     *
     * @param code Error code to hash
     * @return Hash value
     */
    BREZEL_NODISCARD size_t operator()(const ErrorCode& code) const noexcept {
        return std::hash<int>()(code.value()) ^
               std::hash<const brezel::core::error::ErrorCategory*>()(
                   &code.category());
    }
};