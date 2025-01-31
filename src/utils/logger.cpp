#include <fmt/chrono.h>
#include <fmt/format.h>

#include <algorithm>
#include <brezel/utils/logger/logger.hpp>
#include <chrono>
#include <ranges>

namespace brezel::log {
namespace {
constexpr std::string_view severity_strings[] = {"TRACE",   "DEBUG", "INFO",
                                                 "WARNING", "ERROR", "FATAL"};
}

FileLogSink::FileLogSink(Config config) : m_config(std::move(config)) {
    m_file.open(m_config.path, std::ios::app);
    if (!m_file) {
        // TODO: Reference invalid operation
        throw "";
    }
}

void FileLogSink::write(const LogEntry& entry) {
    std::lock_guard lock(m_mutex);
    std::array<char, 1024> buffer;

      auto time = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto len = std::snprintf(
        buffer.data(), buffer.size(), "[%s] [%s] [%lu] %s\n",
        std::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(time)).c_str(),
        severity_strings[static_cast<size_t>(entry.severity)].data(),
        std::hash<std::thread::id>{}(entry.thread_id), entry.message.c_str());

    if (len > 0) {
        m_file.write(buffer.data(), len);
        m_current_size += len;

        if (m_config.auto_flush) {
            m_file.flush();
        }
    }

    rotate_if_needed();
}

void FileLogSink::flush() {
    std::lock_guard lock(m_mutex);
    m_file.flush();
}

void FileLogSink::rotate_if_needed() {
    if (m_current_size < m_config.max_size)
        return;

    m_file.close();

    auto base = m_config.path.string();
    for (int i = m_config.max_files - 1; i >= 0; --i) {
        auto old_name = std::format("{}.{}", base, i);
        auto new_name = std::format("{}.{}", base, i + 1);
        std::filesystem::rename(old_name, new_name);
    }

    std::filesystem::rename(base, std::format("{}.0", base));
    m_file.open(m_config.path, std::ios::app);
    m_current_size = 0;
}

CircularBufferSink::CircularBufferSink(size_t capacity) : m_buffer(capacity) {}

void CircularBufferSink::write(const LogEntry& entry) {
    std::lock_guard lock(m_mutex);
    m_buffer.push_back(entry);
}

Logger::Logger(Config config) : m_config(std::move(config)) {
    if (m_config.async) {
        m_worker = std::jthread([this] { process_queue(); });
    }
}

Logger::~Logger() {
    m_running = false;
    flush();
}

void Logger::add_sink(std::shared_ptr<LogSink> sink) {
    std::lock_guard lock(m_mutex);
    m_sinks.push_back(std::move(sink));
}

void Logger::flush() {
    LogEntry entry;
    while (m_queue.try_pop(entry)) {
        write_entry(entry);
    }

    std::lock_guard lock(m_mutex);
    for (auto& sink : m_sinks) {
        sink->flush();
    }
}

void Logger::process_queue() {
    std::vector<LogEntry> batch;
    batch.reserve(m_config.batch_size);

    while (m_running) {
        batch.clear();

        // Try to fill batch
        for (size_t i = 0; i < m_config.batch_size; ++i) {
            LogEntry entry;
            if (!m_queue.try_pop(entry))
                break;
            batch.push_back(std::move(entry));
        }

        // Process batch
        for (const auto& entry : batch) {
            write_entry(entry);
        }

        // Small sleep if queue is empty
        if (batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Logger::write_entry(const LogEntry& entry) {
    std::lock_guard lock(m_mutex);
    for (auto& sink : m_sinks) {
        sink->write(entry);
    }
}

Logger& global_logger() {
    static Logger logger;
    return logger;
}

}  // namespace brezel::log