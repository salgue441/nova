#pragma once

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error_code.hpp>
#include <brezel/core/macros.hpp>
#include <concepts>
#include <expected>
#include <optional>
#include <string_view>

namespace brezel::core::status {
/// @brief Represents operation status cods
enum class StatusCode : uint8_t {
    Ok = 0,
    Cancelled,
    InvalidArgument,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated
};

/**
 * @brief Lightweight status class for representing operation results
 * @details Optimized for minimal overhead and quick status check
 */
class BREZEL_API Status {
public:
    /**
     * @brief Creates a success status
     */
    BREZEL_NODISCARD static constexpr Status Ok() noexcept { return Status{}; }

    /**
     * @brief Creates a status with the given code
     */
    BREZEL_NODISCARD static constexpr Status FromCode(
        StatusCode code) noexcept {
        return Status{code};
    }

    /**
     * @brief Creates a status with code and message
     *
     * @tparam Args Format argument types
     * @param code Error code
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    BREZEL_NODISCARD static Status FromMessage(StatusCode code,
                                               std::format_string<Args...> fmt,
                                               Args&&... args) {
        return Status{code, std::format(fmt, std::forward<Args>(args)...)};
    }

    /**
     * @brief Default constructor creates OK status
     */
    constexpr Status() noexcept = default;

    /**
     * @brief Creates a status with only a code
     *
     * @param code Error code
     */
    constexpr explicit Status(StatusCode code) noexcept : m_code(code) {}

    /**
     * @brief Creates a status with a code and message
     *
     * @param code Error code
     * @param message Error message
     */
    Status(StatusCode code, std::string message) noexcept
        : m_code(code), m_message(std::move(message)) {}

    /**
     * @brief Get the status code
     */
    BREZEL_NODISCARD constexpr StatusCode code() const noexcept {
        return m_code;
    }

    /**
     * @brief Get the status message if any
     */
    BREZEL_NODISCARD const std::string& message() const& noexcept {
        return m_message;
    }

    /**
     * @brief Convert to error code if error
     */
    BREZEL_NODISCARD std::optional<error::ErrorCode> to_error() const noexcept {
        if (ok())
            return std::nullopt;

        using EC = error::RuntimeCategory::Code;
        switch (m_code) {
            case StatusCode::InvalidArgument:
                return error::make_error_code(EC::InvalidOperation);

            case StatusCode::ResourceExhausted:
                return error::make_error_code(EC::OutOfMemory);

            case StatusCode::Internal:
                return error::make_error_code(EC::Unknown);

            case StatusCode::Unavailable:
                return error::make_error_code(EC::InvalidState);

            default:
                return error::make_error_code(EC::Unknown);
        }
    }

    /**
     * @brief Check if status is OK
     */
    BREZEL_NODISCARD constexpr bool ok() const noexcept {
        return m_code == StatusCode::Ok;
    }

    /**
     * @brief Boolean conversion operator
     */
    BREZEL_NODISCARD constexpr explicit operator bool() const noexcept {
        return ok();
    }

    /**
     * @brief Comparison operators
     */
    BREZEL_NODISCARD friend constexpr bool operator==(const Status&,
                                                      const Status&) = default;

private:
    StatusCode m_code{StatusCode::Ok};
    std::string m_message;
};

/**
 * @brief Represents a status with an attached value
 * @tparam T Value types
 */
template <typename T>
class BREZEL_API StatusOr {
public:
    /**
     * @brief Constructs with a value
     */
    template <typename U = T>
        requires std::constructible_from<T, U>
    constexpr StatusOr(U&& value) noexcept(
        std::is_nothrow_constructible_v<T, U>)
        : m_value(std::forward<U>(value)) {}

    /**
     * @brief Constructs with a status
     */
    constexpr StatusOr(Status status) noexcept : m_status(std::move(status)) {
        BREZEL_ASSERT(!m_status.ok(), "Status must not be OK");
    }

    /**
     * @brief Get status
     */
    BREZEL_NODISCARD const Status& status() const& noexcept { return m_status; }

    /**
     * @brief Check if contains value
     */
    BREZEL_NODISCARD constexpr bool has_value() const noexcept {
        return m_status.ok();
    }

    /**
     * @brief Boolean conversion operator
     */
    BREZEL_NODISCARD constexpr explicit operator bool() const noexcept {
        return has_value();
    }

    /**
     * @brief Get value reference
     */
    BREZEL_NODISCARD constexpr const T& value() const& {
        if (!has_value()) {
            throw error::InvalidOperation("Accessing value of non-OK StatusOr");
        }
        return m_value;
    }

    /**
     * @brief Get value reference
     */
    BREZEL_NODISCARD constexpr T& value() & {
        if (!has_value()) {
            throw error::InvalidOperation("Accessing value of non-OK StatusOr");
        }
        return m_value;
    }

    /**
     * @brief Get value rvalue reference
     */
    BREZEL_NODISCARD constexpr T&& value() && {
        if (!has_value()) {
            throw error::InvalidOperation("Accessing value of non-OK StatusOr");
        }
        return std::move(m_value);
    }

    /**
     * @brief Access value or return alternative
     */
    template <typename U>
    BREZEL_NODISCARD constexpr T value_or(U&& default_value) const& {
        return has_value() ? m_value
                           : static_cast<T>(std::forward<U>(default_value));
    }

    /**
     * @brief Access value or return alternative
     */
    template <typename U>
    BREZEL_NODISCARD constexpr T value_or(U&& default_value) && {
        return has_value() ? std::move(m_value)
                           : static_cast<T>(std::forward<U>(default_value));
    }

private:
    Status m_status{};
    std::conditional_t<std::is_void_v<T>, std::monostate, T> m_value{};
};

// Helper macros for status checking
#define BREZEL_RETURN_IF_ERROR(expr)              \
    do {                                          \
        if (auto status = (expr); !status.ok()) { \
            return status;                        \
        }                                         \
    } while (0)

#define BREZEL_ASSIGN_OR_RETURN(lhs, expr) \
    auto&& status_or = (expr);             \
    if (!status_or.ok()) {                 \
        return status_or.status();         \
    }                                      \
    lhs = std::move(status_or).value()

}  // namespace brezel::core::status