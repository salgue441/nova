#pragma once

#include <tbb/concurrent_queue.h>

#include <boost/circular_buffer.hpp>
#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error_code.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/core/status/status.hpp>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <ranges>
#include <source_location>
#include <thread>

namespace brezel::log {
/// @brief Log severity levels
enum class Severity : uint8_t { Trace, Debug, Info, Warning, Error, Fatal };

/// @brief Log entry containing message and metadata
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    Severity severity;
    std::string message;
    std::source_location location;
    std::thread::id thread_id;
    std::optional<core::status::Status> status;
    std::optional<core::error::ErrorCode> error;
};

/// @brief Interface for log sinks
class BREZEL_API LogSink {
public:
    virtual ~LogSink() = default;
    virtual void write(const LogEntry& entry) = 0;
    virtual void flush() = 0;

protected:
    LogSink() = default;
    BREZEL_IMMOVABLE(LogSink);
};

/**
 * @brief File-based log sink with rotation
 */
class BREZEL_API FileLogSink : public LogSink {
public:
    struct Config {
        std::filesystem::path path;
        size_t max_size = 10 * 1024 * 1024;  // 10MB
        size_t max_files = 5;
        bool auto_flush = true;
    };

    explicit FileLogSink(Config config);
    void write(const LogEntry& entry) override;
    void flush() override;

private:
    void rotate_if_needed();

    Config m_config;
    std::ofstream m_file;
    size_t m_current_size{0};
    std::mutex m_mutex;
};

/**
 * @brief Memory-based circular buffer sink
 */
class BREZEL_API CircularBufferSink : public LogSink {
public:
    explicit CircularBufferSink(size_t capacity);
    void write(const LogEntry& entry) override;
    void flush() override {}

    auto entries() const {
        std::shared_lock lock(m_mutex);
        return m_buffer | std::views::all;
    }

private:
    boost::circular_buffer<LogEntry> m_buffer;
    mutable std::shared_mutex m_mutex;
};

/**
 * @brief Async logger with multiple sinks
 */
class BREZEL_API Logger {
public:
    struct Config {
        Severity min_severity = Severity::Info;
        size_t queue_capacity = 8192;
        size_t batch_size = 32;
        bool async = true;
    };

    explicit Logger(Config config = Config{});
    ~Logger();

    void add_sink(std::shared_ptr<LogSink> sink);

    /**
     * @brief Logs a new entry
     *
     * @tparam Args Format argument types
     * @param severity Severity level
     * @param location Location where the message occured
     * @param fmt Format string
     * @param args Format argument
     */
    template <typename... Args>
    void log(Severity severity, const std::source_location& location,
             std::format_string<Args...> fmt, Args&&... args) {
        if (severity < m_config.min_severity)
            return;

        LogEntry entry{
            .timestamp = std::chrono::system_clock::now(),
            .severity = severity,
            .message = std::format(fmt, std::forward<Args>(args)...),
            .location = location.thread_id = std::this_thread::get_id()};

        if (m_config.async)
            m_queue.push(std::move(entry));

        else:
          write_entry(entry);
    }

    /**
     * @brief Logs a new entry with error information
     *
     * @tparam Args Format argument types
     * @param severity Severity level
     * @param error Error that occurred
     * @param location Location where the message occured
     * @param fmt Format string
     * @param args Format argument
     */
    template <typename... Args>
    void log_with_error(Severity severity, const core::error::ErrorCode& error,
                        const std::source_location& location,
                        std::format_string<Args...> fmt, Args&&... args) {
        if (severity < m_config.min_severity)
            return;

        LogEntry entry{.timestamp = std::chrono::system_clock::now(),
                       .severity = severity,
                       .message = std::format(fmt, std::forward<Args>(args)...),
                       .location = location,
                       .thread_id = std::this_thread::get_id(),
                       .error = error};

        if (m_config.async)
            m_queue.push(std::move(entry));

        else
            write_entry(entry);
    }

    void flush();

private:
    void process_queue();
    void write_entry(const LogEntry& entry);

    Config m_config;
    std::atomic<bool> m_running{true};
    tbb::concurrent_queue<LogEntry> m_queue;
    boost::container::small_vector<std::shared_ptr<LogSink>, 4> m_sinks;
    std::jthread m_worker;
    std::mutex m_mutex;
};

// Global logger instance
BREZEL_API Logger& global_logger();

// Convenience macros
#define BREZEL_LOG_TRACE(...)                                           \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Trace,  \
                                       std::source_location::current(), \
                                       __VA_ARGS__)

#define BREZEL_LOG_DEBUG(...)                                           \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Debug,  \
                                       std::source_location::current(), \
                                       __VA_ARGS__)

#define BREZEL_LOG_INFO(...)                                            \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Info,   \
                                       std::source_location::current(), \
                                       __VA_ARGS__)

#define BREZEL_LOG_WARNING(...)                                          \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Warning, \
                                       std::source_location::current(),  \
                                       __VA_ARGS__)

#define BREZEL_LOG_ERROR(...)                                           \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Error,  \
                                       std::source_location::current(), \
                                       __VA_ARGS__)

#define BREZEL_LOG_FATAL(...)                                           \
    ::brezel::log::global_logger().log(::brezel::log::Severity::Fatal,  \
                                       std::source_location::current(), \
                                       __VA_ARGS__)

// Status and error integration macros
#define BREZEL_LOG_STATUS(severity, status, ...)    \
    ::brezel::log::global_logger().log_with_status( \
        severity, status, std::source_location::current(), __VA_ARGS__)

#define BREZEL_LOG_ERROR_CODE(severity, error, ...) \
    ::brezel::log::global_logger().log_with_error(  \
        severity, error, std::source_location::current(), __VA_ARGS__)

}  // namespace brezel::log