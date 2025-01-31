#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <boost/container/small_vector.hpp>
#include <boost/smart_ptr/atomic_shared_ptr.hpp>
#include <brezel/core/error/error_category.hpp>
#include <brezel/core/macros.hpp>
#include <compare>
#include <memory>
#include <source_location>

namespace brezel::core::error {
/**
 * @brief Represents error codes with associated context and category
 * information
 *
 * @details Provides efficient, thread-safe error handling with message
 * caching, contextual information storage, and comparison operations.
 */
class BREZEL_API ErrorCode {
public:
    /// @brief Context information for an error code
    struct Context {
        boost::container::small_vector<std::string, 2> notes;
        std::source_location location;
    };

    /// @brief Default constructor creates a success (0) error code
    BREZEL_NODISCARD constexpr ErrorCode() noexcept = default;
    constexpr ErrorCode(const ErrorCode& other) noexcept
        : m_code(other.m_code), m_category(other.m_category) {
        if (other.m_context)
            m_context = std::make_unique<Context>(*other.m_context);
    }

    ErrorCode& operator=(const ErrorCode& other) noexcept = default;
    ErrorCode(ErrorCode&& other) noexcept = default;
    ErrorCode& operator=(ErrorCode&& other) noexcept = default;

    /**
     * @brief Constructs error code with specified value and category
     *
     * @param code Numeric error code
     * @param category Error category
     */
    BREZEL_NODISCARD constexpr ErrorCode(int code,
                                         const ErrorCategory& category) noexcept
        : m_code(code), m_category(&category) {}

    /**
     * @brief Constructs from runtime category code
     *
     * @param code Runtime error code
     */
    BREZEL_NODISCARD constexpr ErrorCode(RuntimeCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&RuntimeCategory::instance()) {}

    /**
     * @brief Constructs from logic category code
     *
     * @param code Logic error code
     */
    BREZEL_NODISCARD constexpr ErrorCode(LogicCategory::Code code) noexcept
        : m_code(static_cast<int>(code)),
          m_category(&LogicCategory::instance()) {}

    // Access methods
    /**
     * @brief Gets the numeric error value
     *
     * @return Error code value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE constexpr int value() const noexcept {
        return m_code;
    }

    /**
     * @brief Gets the error category
     *
     * @return Reference to error category
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE constexpr const ErrorCategory&
    category() const noexcept {
        return *m_category;
    }

    /**
     * @brief Gets error context if available
     *
     * @return Pointer to context or nullptr
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Context* context()
        const noexcept {
        return m_context.get();
    }

    // Methods
    /**
     * @brief Gets cached error message
     *
     * @return Reference to error message
     */
    BREZEL_NODISCARD const std::string& message() const {
        auto it = s_messages_cache.find(m_code);
        if (BREZEL_PREDICT_TRUE(it != s_messages_cache.end()))
            return it->second;

        auto [it_new, _] =
            s_messages_cache.insert({m_code, category().message(value())});

        return it_new->second;
    }

    /**
     * @brief Adds contextual information to error
     *
     * @param note Additional note
     * @param location Source location
     */
    BREZEL_HOT void add_context(std::string note,
                                const std::source_location& location =
                                    std::source_location::current()) {
        if (!m_context)
            m_context = std::make_unique<Context>();

        m_context->notes.push_back(std::move(note));
        m_context->location = location;
    }

    // Operators
    /**
     * @brief Checks if error code represents an error
     *
     * @return true if error code is non-zero
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE constexpr explicit operator bool()
        const noexcept {
        return value() != 0;
    }

    /**
     * @brief Three-way comparison operator
     */
    BREZEL_NODISCARD constexpr auto operator<=>(
        const ErrorCode& other) const noexcept {
        if (auto cmp = &category() <=> &other.category(); cmp != 0)
            return cmp;

        return value() <=> other.value();
    }

    /**
     * @brief Equality operator
     *
     * @param other Other ErrorCode to check
     */
    BREZEL_NODISCARD constexpr bool operator==(
        const ErrorCode& other) const noexcept {
        return *this <=> other == 0;
    }

private:
    BREZEL_ALIGN_CACHE int m_code{0};
    BREZEL_ALIGN_CACHE const ErrorCategory* m_category{
        &LogicCategory::instance()};
    std::unique_ptr<Context> m_context;

    static inline tbb::concurrent_unordered_map<int, std::string>
        s_messages_cache;
};

/**
 * @brief Creates runtime error code
 *
 * @param code Runtime error code
 * @return ErrorCode instance
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE ErrorCode
make_error_code(RuntimeCategory::Code code) noexcept {
    return ErrorCode{code};
}

/**
 * @brief Creates logic error code
 *
 * @param code Logic error code
 * @return ErrorCode instance
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE ErrorCode
make_error_code(LogicCategory::Code code) noexcept {
    return ErrorCode{code};
}

/**
 * @brief Creates error code with custom category
 *
 * @param code Error code value
 * @param category Error category
 * @return ErrorCode instance
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE ErrorCode
make_error_code(int code, const ErrorCategory& category) noexcept {
    return ErrorCode{code, category};
}

/**
 * @brief Creates system error code
 *
 * @param error_code System error code
 * @return ErrorCode instance
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE ErrorCode
make_system_error(int error_code) noexcept {
    return ErrorCode{error_code, SystemCategory::instance()};
}
}  // namespace brezel::core::error

namespace std {
template <>
struct hash<brezel::core::error::ErrorCode> {
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    operator()(const brezel::core::error::ErrorCode& code) const noexcept {
        return std::hash<int>()(code.value()) ^
               std::hash<const brezel::core::error::ErrorCategory*>()(
                   &code.category());
    }
};
}  // namespace std