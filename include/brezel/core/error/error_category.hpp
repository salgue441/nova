#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <boost/container/flat_map.hpp>
#include <brezel/core/macros.hpp>
#include <format>
#include <memory>
#include <shared_mutex>
#include <string_view>

namespace brezel::core::error {
/**
 * @brief Abstract base class for error categories using type erasure pattern
 *
 * @details Thread-safe error category base class with efficient message
 * caching and concurrent access support.
 */
class BREZEL_API ErrorCategory {
public:
    /// @brief Virtual destructor
    virtual ~ErrorCategory() = default;

    /**
     * @brief Gets the category name
     * @return Category name view
     */
    BREZEL_NODISCARD virtual std::string_view name() const noexcept = 0;

    /**
     * @brief Gets the error message for a code
     *
     * @param code Error code
     * @return Error message
     */
    BREZEL_NODISCARD virtual const std::string& message(int code) const = 0;

protected:
    ErrorCategory() = default;
    BREZEL_IMMOVABLE(ErrorCategory);
    BREZEL_UNCOPYABLE(ErrorCategory);

    /// @brief Thread-safe message cache
    mutable tbb::concurrent_unordered_map<int, std::string> m_message_cache;
};

/**
 * @brief Template base class for implementing error categories
 * @tparam Derived CRTP derived class type
 */
template <typename Derived>
class BREZEL_API ErrorCategoryImpl : public ErrorCategory {
public:
    /**
     * @brief Gets the category name
     * @return Category name view
     */
    BREZEL_NODISCARD std::string_view name() const noexcept override {
        return Derived::category_name;
    }

    /**
     * @brief Gets the error message with caching
     *
     * @param code Error code
     * @return Error message
     */
    BREZEL_NODISCARD const std::string& message(int code) const override {
        auto it = m_message_cache.find(code);
        if (it != m_message_cache.end()) {
            return it->second;
        }

        return m_message_cache
            .insert({code, static_cast<const Derived*>(this)->do_message(code)})
            .first->second;
    }

protected:
    ErrorCategoryImpl() = default;
};

/**
 * @brief System error category for OS-related errors
 */
class BREZEL_API SystemCategory final
    : public ErrorCategoryImpl<SystemCategory> {
public:
    static constexpr std::string_view category_name = "System";

    /**
     * @brief Gets the message for a system error code
     *
     * @param code System error code
     * @return Error message
     */
    BREZEL_NODISCARD std::string do_message(int code) const {
        return std::system_category().message(code);
    }

    /**
     * @brief Gets the singleton instance
     * @return Category instance
     */
    BREZEL_NODISCARD static const SystemCategory& instance() noexcept {
        static const SystemCategory category;
        return category;
    }

private:
    SystemCategory() = default;
};

/**
 * @brief Runtime error category for general runtime errors
 */
class BREZEL_API RuntimeCategory final
    : public ErrorCategoryImpl<RuntimeCategory> {
public:
    /// @brief Runtime error codes
    enum class Code {
        Success = 0,
        Unknown,
        InvalidOperation,
        OutOfMemory,
        InvalidState,
        Timeout,
        IoError
    };

    static constexpr std::string_view category_name = "Runtime";

    /**
     * @brief Gets the message for a runtime error code
     *
     * @param code Runtime error code
     * @return Error message
     */
    BREZEL_NODISCARD std::string do_message(int code) const {
        static const boost::container::flat_map<int, std::string_view> messages{
            {static_cast<int>(Code::Success), "Success"},
            {static_cast<int>(Code::Unknown), "Unknown runtime error"},
            {static_cast<int>(Code::InvalidOperation), "Invalid operation"},
            {static_cast<int>(Code::OutOfMemory), "Out of memory"},
            {static_cast<int>(Code::InvalidState), "Invalid state"},
            {static_cast<int>(Code::Timeout), "Operation timed out"},
            {static_cast<int>(Code::IoError), "I/O error"}};

        if (auto it = messages.find(code); it != messages.end())
            return std::string{it->second};

        return std::format("Unknown error code: {}", code);
    }

    /**
     * @brief Gets the singleton instance
     * @return Category instance
     */
    BREZEL_NODISCARD static const RuntimeCategory& instance() noexcept {
        static const RuntimeCategory category;
        return category;
    }

private:
    RuntimeCategory() = default;
};

/**
 * @brief Logic error category for programming errors
 */
class BREZEL_API LogicCategory final : public ErrorCategoryImpl<LogicCategory> {
public:
    /// @brief Logic error codes
    enum class Code {
        Success = 0,
        Unknown,
        InvalidArgument,
        OutOfRange,
        InvalidCast,
        DivideByZero,
        NullPointer,
        InvalidState,
    };

    static constexpr std::string_view category_name = "Logic";

    /**
     * @brief Gets the message for a logic error code
     * @param code Logic error code
     * @return Error message
     */
    BREZEL_NODISCARD std::string do_message(int code) const {
        static const boost::container::flat_map<int, std::string_view> messages{
            {static_cast<int>(Code::Success), "Success"},
            {static_cast<int>(Code::Unknown), "Unknown logic error"},
            {static_cast<int>(Code::InvalidArgument), "Invalid argument"},
            {static_cast<int>(Code::OutOfRange), "Out of range"},
            {static_cast<int>(Code::InvalidCast), "Invalid type cast"},
            {static_cast<int>(Code::DivideByZero), "Division by zero"},
            {static_cast<int>(Code::NullPointer), "Null pointer dereference"},
            {static_cast<int>(Code::InvalidState), "Invalid state"}};

        if (auto it = messages.find(code); it != messages.end())
            return std::string{it->second};

        return std::format("Unknown error code: {}", code);
    }

    /**
     * @brief Gets the singleton instance
     * @return Category instance
     */
    BREZEL_NODISCARD static const LogicCategory& instance() noexcept {
        static const LogicCategory category;
        return category;
    }

private:
    LogicCategory() = default;
};
}  // namespace brezel::core::error