#pragma once

#include <boost/stacktrace.hpp>
#include <brezel/core/error/error_category.hpp>
#include <brezel/core/macros.hpp>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <vector>

namespace brezel::core::error {
/**
 * @brief Rich error context for detailed error information
 *
 */
struct ErrorContext {
    std::source_location location{std::source_location::current()};
    std::vector<std::string> notes;
    boost::stacktrace::stacktrace stacktrace;
};

/**
 * @brief Represents error codes with associated category and context
 *
 */
class BREZEL_API ErrorCode {
public:
    // Constructors
    ErrorCode() noexcept : m_category(&LogicCategory::instance()) {}
    ErrorCode(int code, const ErrorCategory& category) noexcept
        : m_code(code), m_category(&category) {}

    ErrorCode(RuntimeCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&RuntimeCategory::instance()) {}

    ErrorCode(LogicCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&LogicCategory::instance()) {}

    // Copy constructor & assignment
    ErrorCode(const ErrorCode& other) noexcept
        : m_code(other.m_code), m_category(other.m_category) {
        if (other.m_context)
            m_context = std::make_unique<ErrorContext>(*other.m_context);
    }

    ErrorCode& operator=(const ErrorCode& other) noexcept {
        if (this != &other) {
            m_code = other.m_code;
            m_category = other.m_category;

            if (other.m_context)
                m_context = std::make_unique<ErrorContext>(*other.m_context);

            else
                m_context.reset();
        }

        return *this;
    }

    // Move operations
    ErrorCode(ErrorCode&&) noexcept = default;
    ErrorCode& operator=(ErrorCode&&) noexcept = default;

    // Context management
    void add_context(std::string note, const std::source_location& location =
                                           std::source_location::current()) {
        if (!m_context)
            m_context = std::make_unique<ErrorContext>();

        m_context->notes.push_back(std::move(note));
        m_context->location = location;
    }

    BREZEL_NODISCARD constexpr int value() const noexcept { return m_code; }
    BREZEL_NODISCARD const ErrorCategory& category() const noexcept {
        return *m_category;
    }

    BREZEL_NODISCARD std::string message() const {
        return category().message(value());
    }

    BREZEL_NODISCARD const ErrorContext* context() const noexcept {
        return m_context.get();
    }

    // Comparison operators
    constexpr auto operator<=>(const ErrorCode& other) const noexcept {
        if (auto cmp = std::compare_three_way{}(&category(), &other.category());
            cmp != 0) {
            return cmp;
        }

        return value() <=> other.value();
    }

    constexpr bool operator==(const ErrorCode& other) const noexcept {
        return value() == other.value() && &category() == &other.category();
    }

    // Boolean conversion (explicit)
    BREZEL_NODISCARD constexpr explicit operator bool() const noexcept {
        return value() != 0;
    }

private:
    int m_code{0};
    const ErrorCategory* m_category;
    std::unique_ptr<ErrorContext> m_context;
};

// Factory functions
inline ErrorCode make_error_code(RuntimeCategory::Code code) noexcept {
    return ErrorCode{code};
}

inline ErrorCode make_error_code(LogicCategory::Code code) noexcept {
    return ErrorCode{code};
}

inline ErrorCode make_system_error(int error_code) noexcept {
    return ErrorCode{error_code, SystemCategory::instance()};
}
}  // namespace brezel::core::error

namespace std {
template <>
struct hash<brezel::core::error::ErrorCode> {
    BREZEL_NODISCARD size_t
    operator()(const brezel::core::error::ErrorCode& code) const noexcept {
        return std::hash<int>()(code.value()) ^
               std::hash<const brezel::core::error::ErrorCategory*>()(
                   &code.category());
    }
};
}  // namespace std