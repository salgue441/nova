#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <brezel/core/error/error_fwd.hpp>
#include <brezel/core/macros.hpp>
#include <string>
#include <string_view>
#include <system_error>

namespace brezel::core::error {
/**
 * @brief Base class for error categories
 *
 */
class BREZEL_API ErrorCategory {
public:
    virtual ~ErrorCategory() = default;

    BREZEL_NODISCARD virtual std::string_view name() const noexcept = 0;
    BREZEL_NODISCARD virtual std::string message(int code) const = 0;

protected:
    ErrorCategory() = default;
    ErrorCategory(const ErrorCategory&) = delete;
    ErrorCategory& operator=(const ErrorCategory&) = delete;
    ErrorCategory(ErrorCategory&&) = delete;
    ErrorCategory& operator=(ErrorCategory&&) = delete;

    mutable tbb::concurrent_unordered_map<int, std::string> m_message_cache;
};

/**
 * @brief System error category for OS-related errors
 *
 */
class BREZEL_API SystemCategory final : public ErrorCategory {
public:
    static const SystemCategory& instance() noexcept {
        static const SystemCategory category;
        return category;
    }

    BREZEL_NODISCARD std::string_view name() const noexcept override {
        return "System";
    }

    BREZEL_NODISCARD std::string message(int code) const override {
        auto it = m_message_cache.find(code);
        if (it != m_message_cache.end())
            return it->second;

        auto msg = std::system_category().message(code);
        return m_message_cache.insert({code, std::move(msg)}).first->second;
    }

private:
    SystemCategory() = default;
};

/**
 * @brief Runtime error category for runtime errors
 */
class BREZEL_API RuntimeCategory final : public ErrorCategory {
public:
    enum class Code {
        Success = 0,
        Unknown,
        InvalidOperation,
        OutOfMemory,
        InvalidState,
        Timeout,
        IoError
    };

    static const RuntimeCategory& instance() noexcept {
        static const RuntimeCategory category;
        return category;
    }

    BREZEL_NODISCARD std::string_view name() const noexcept override {
        return "Runtime";
    }

    BREZEL_NODISCARD std::string message(int code) const override {
        auto it = m_message_cache.find(code);
        if (it != m_message_cache.end()) 
            return it->second;
        

        std::string msg;
        switch (static_cast<Code>(code)) {
            case Code::Success:
                msg = "Success";
                break;

            case Code::Unknown:
                msg = "Unknown runtime error";
                break;

            case Code::InvalidOperation:
                msg = "Invalid operation";
                break;

            case Code::OutOfMemory:
                msg = "Out of memory";
                break;

            case Code::InvalidState:
                msg = "Invalid state";
                break;

            case Code::Timeout:
                msg = "Operation timed out";
                break;

            case Code::IoError:
                msg = "I/O error";
                break;

            default:
                msg = "Unknown error code: " + std::to_string(code);
        }

        return m_message_cache.insert({code, std::move(msg)}).first->second;
    }

private:
    RuntimeCategory() = default;
};

/**
 * @brief Logic error category for logical/programming errors
 */
class BREZEL_API LogicCategory final : public ErrorCategory {
public:
    enum class Code {
        Success = 0,
        Unknown,
        InvalidArgument,
        OutOfRange,
        InvalidCast,
        NullPointer,
        InvalidState,
        InvalidOperation
    };

    static const LogicCategory& instance() noexcept {
        static const LogicCategory category;
        return category;
    }

    BREZEL_NODISCARD std::string_view name() const noexcept override {
        return "Logic";
    }

    BREZEL_NODISCARD std::string message(int code) const override {
        auto it = m_message_cache.find(code);
        if (it != m_message_cache.end()) 
            return it->second;
        

        std::string msg;
        switch (static_cast<Code>(code)) {
            case Code::Success:
                msg = "Success";
                break;

            case Code::Unknown:
                msg = "Unknown logic error";
                break;

            case Code::InvalidArgument:
                msg = "Invalid argument";
                break;

            case Code::OutOfRange:
                msg = "Out of range";
                break;

            case Code::InvalidCast:
                msg = "Invalid type cast";
                break;

            case Code::NullPointer:
                msg = "Null pointer dereference";
                break;

            case Code::InvalidState:
                msg = "Invalid state";
                break;

            case Code::InvalidOperation:
                msg = "Invalid operation";
                break;

            default:
                msg = "Unknown error code: " + std::to_string(code);
        }

        return m_message_cache.insert({code, std::move(msg)}).first->second;
    }

private:
    LogicCategory() = default;
};
}  // namespace brezel::core::error