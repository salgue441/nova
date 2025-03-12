#pragma once

#include <tbb/task_group.h>

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/parallel/thread_pool.hpp>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brezel::tensor::parallel {
/**
 * @brief Task execution status
 */
enum class TaskStatus {
    /// @brief Task has not been started
    NotStarted,

    /// @brief Task is currently executing
    Running,

    /// @brief Task has been completed
    Completed,

    /// @brief Task failed with an error
    Failed,

    /// @brief Task has been cancelled
    Cancelled
};

/**
 * @brief Convert TaskStatus to string
 *
 * @param status Task status
 * @return std::string Task status as string
 */
BREZEL_NODISCARD inline std::string task_status_to_string(TaskStatus status) {
    switch (status) {
        case TaskStatus::NotStarted:
            return "NotStarted";

        case TaskStatus::Running:
            return "Running";

        case TaskStatus::Completed:
            return "Completed";

        case TaskStatus::Failed:
            return "Failed";

        case TaskStatus::Cancelled:
            return "Canceled";
        default:
            return "Unknown";
    }
}

/**
 * @brief Task Information
 */
struct BREZEL_API TaskInfo {
    /// Task identifier
    std::string id;

    /// Task name (for debugging)
    std::string name;

    /// Task execution status
    TaskStatus status = TaskStatus::NotStarted;

    /// Error message (if failed)
    std::string error_message;

    /// Task priority (higher = more important)
    int priority = 0;

    /// Whether the task is cancellable
    bool cancellable = true;

    /// Whether to propagate exceptions to the caller
    bool propagate_exceptions = true;

    /// Task dependencies (IDs of tasks that must complete first)
    std::vector<std::string> dependencies;
};

/**
 * @brief Abstract task interface
 *
 * @details Base class for all tasks that can be executed by the task system.
 * Tasks encapsulate a unit of work that can be executed asynchronously.
 */
class BREZEL_API Task {
public:
public:
    /**
     * @brief Constructor
     *
     * @param info Task information
     */
    explicit Task(TaskInfo info) : m_info(std::move(info)) {}

    /**
     * @brief Virtual destructor
     */
    virtual ~Task() = default;

    /**
     * @brief Execute the task
     *
     * @details This method is called by the task system to execute the task.
     * Override this method to implement task-specific logic.
     */
    virtual void execute() = 0;

    /**
     * @brief Cancel the task
     *
     * @details This method is called by the task system to cancel the task.
     * Override this method to implement task-specific cancellation logic.
     *
     * @return bool True if the task was successfully canceled
     */
    virtual bool cancel() {
        if (!m_info.cancellable) {
            return false;
        }

        m_cancelled.store(true, std::memory_order_relaxed);
        return true;
    }

    /**
     * @brief Check if the task is canceled
     *
     * @return bool True if the task is canceled
     */
    bool is_canceled() const {
        return m_cancelled.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get the task information
     *
     * @return const TaskInfo& Task information
     */
    const TaskInfo& info() const { return m_info; }

    /**
     * @brief Get the task status
     *
     * @return TaskStatus Task status
     */
    TaskStatus status() const { return m_info.status; }

    /**
     * @brief Set the task status
     *
     * @param status New status
     */
    void set_status(TaskStatus status) { m_info.status = status; }

    /**
     * @brief Set the error message
     *
     * @param message Error message
     */
    void set_error(std::string message) {
        m_info.error_message = std::move(message);
    }

private:
    TaskInfo m_info;
    std::atomic<bool> m_cancelled{false};
};

/**
 * @brief Shared pointer to a task
 */
using TaskPtr = std::shared_ptr<Task>;

/**
 * @brief Function task implementation
 *
 * @details Task implementation that wraps a function
 */
class BREZEL_API FunctionTask : public Task {
public:
    /**
     * @brief Constructor
     *
     * @param info Task information
     * @param func Function to execute
     */
    FunctionTask(TaskInfo info, std::function<void()> func)
        : Task(std::move(info)), m_func(std::move(func)) {}

    /**
     * @brief Execute the function
     */
    void execute() override {
        if (m_func) {
            m_func();
        }
    }

private:
    std::function<void()> m_func;
};

/**
 * @brief Task system for managing and executing tasks
 *
 * @details The task system provides a high-level interface for submitting and
 * managing tasks. It handles dependencies between tasks and ensures that tasks
 * are executed in the correct order. It also provides monitoring and
 * cancellation capabilities.
 */
class BREZEL_API TaskSystem {
public:
    /**
     * @brief Get the task system singleton instance
     * @return TaskSystem& Task system instance
     */
    static TaskSystem& instance() {
        static TaskSystem instance;
        return instance;
    }

    /**
     * @brief Initialize the task system
     *
     * @param num_threads Number of threads to use for task execution
     */
    void initialize(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) {
            return;
        }

        ThreadPool::get_instance().initialize(num_threads);
        m_initialized = true;
    }

    /**
     * @brief Submit a task to the task system
     *
     * @param task Task to submit
     * @return std::string Task ID
     */
    std::string submit(TaskPtr task) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) {
            initialize();
        }

        const auto& task_id = task->info().id;
        if (m_tasks.find(task_id) != m_tasks.end()) {
            throw core::error::InvalidArgument("Task ID '{}' is already in use",
                                               task_id);
        }

        m_tasks[task_id] = task;
        bool can_execute = true;

        for (const auto& dep_id : task->info().dependencies) {
            auto it = m_tasks.find(dep_id);
            if (it == m_tasks.end()) {
                throw core::error::InvalidArgument(
                    "Task '{}' depends on unkown task '{}'", task_id, dep_id);
            }

            if (it->second->status() != TaskStatus::Completed) {
                can_execute = false;
                m_dependents[dep_id].push_back(task_id);
            }
        }

        if (can_execute) {
            execute_task(task);
        }

        return task_id;
    }

    /**
     * @brief Submit a function as a task
     *
     * @param info Task information
     * @param func Function to execute
     * @return std::string Task ID
     */
    std::string submit(TaskInfo info, std::function<void()> func) {
        if (info.id.empty()) {
            info.id = generate_task_id();
        }

        auto task =
            std::make_shared<FunctionTask>(std::move(info), std::move(func));

        return submit(task);
    }

    /**
     * @brief Cancel a task
     *
     * @param task_id Task ID
     * @return bool True if the task was successfully canceled
     */
    bool cancel(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_tasks.find(task_id);

        if (it == m_tasks.end()) {
            return false;
        }

        auto& task = it->second;
        if (task->status() == TaskStatus::Completed ||
            task->status() == TaskStatus::Cancelled) {
            return false;
        }

        if (!task->cancel()) {
            return false;
        }

        task->set_status(TaskStatus::Cancelled);

        auto dep_it = m_dependents.find(task_id);
        if (dep_it != m_dependents.end()) {
            for (const auto& dep_task_id : dep_it->second) {
                cancel(dep_task_id);
            }
        }

        return true;
    }

    /**
     * @brief Wait for a task to complete
     *
     * @param task_id Task ID
     * @param timeout_ms Timeout in milliseconds (0 = wait indefinitely)
     * @return bool True if the task completed, false if timed out
     */
    bool wait_for(const std::string& task_id, long timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(m_mutex);

        auto it = m_tasks.find(task_id);
        if (it == m_tasks.end()) {
            throw core::error::InvalidArgument("Unknown task ID '{}'", task_id);
        }

        auto& task = it->second;
        if (task->status() == TaskStatus::Completed ||
            task->status() == TaskStatus::Failed ||
            task->status() == TaskStatus::Cancelled) {
            return true;
        }

        if (timeout_ms > 0) {
            auto predicate = [this, &task_id]() {
                auto it = m_tasks.find(task_id);
                if (it == m_tasks.end()) {
                    return true;
                }

                return it->second->status() == TaskStatus::Completed ||
                       it->second->status() == TaskStatus::Failed ||
                       it->second->status() == TaskStatus::Cancelled;
            };

            return m_condition.wait_for(
                lock, std::chrono::milliseconds(timeout_ms), predicate);
        } else {
            m_condition.wait(lock, [this, &task_id]() {
                auto it = m_tasks.find(task_id);
                if (it == m_tasks.end()) {
                    return true;
                }

                return it->second->status() == TaskStatus::Completed ||
                       it->second->status() == TaskStatus::Failed ||
                       it->second->status() == TaskStatus::Cancelled;
            });

            return true;
        }
    }

    /**
     * @brief Wait for all submitted tasks to complete
     *
     * @param timeout_ms Timeout in milliseconds (0 = wait indefinitely)
     * @return bool True if all tasks completed, false if timed out
     */
    bool wait_all(long timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(m_mutex);

        bool all_done =
            std::all_of(m_tasks.begin(), m_tasks.end(), [](const auto& pair) {
                return pair.second->status() == TaskStatus::Completed ||
                       pair.second->status() == TaskStatus::Failed ||
                       pair.second->status() == TaskStatus::Cancelled;
            });

        if (all_done) {
            return true;
        }

        if (timeout_ms > 0) {
            return m_condition.wait_for(
                lock, std::chrono::milliseconds(timeout_ms), [this]() {
                    return std::all_of(m_tasks.begin(), m_tasks.end(),
                                       [](const auto& pair) {
                                           return pair.second->status() ==
                                                      TaskStatus::Completed ||
                                                  pair.second->status() ==
                                                      TaskStatus::Failed ||
                                                  pair.second->status() ==
                                                      TaskStatus::Cancelled;
                                       });
                });
        } else {
            m_condition.wait(lock, [this]() {
                return std::all_of(
                    m_tasks.begin(), m_tasks.end(), [](const auto& pair) {
                        return pair.second->status() == TaskStatus::Completed ||
                               pair.second->status() == TaskStatus::Failed ||
                               pair.second->status() == TaskStatus::Cancelled;
                    });
            });

            return true;
        }
    }

    /**
     * @brief Get the status of a task
     *
     * @param task_id Task ID
     * @return TaskStatus Task status
     */
    TaskStatus status(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(m_mutex);

        auto it = m_tasks.find(task_id);
        if (it == m_tasks.end()) {
            throw core::error::InvalidArgument("Unknown task ID '{}'", task_id);
        }

        return it->second->status();
    }

    /**
     * @brief Get the error message for a failed task
     *
     * @param task_id Task ID
     * @return std::string Error message (empty if task did not fail)
     */
    std::string error_message(const std::string& task_id) {
        std::lock_guard<std::mutex> lock(m_mutex);

        auto it = m_tasks.find(task_id);
        if (it == m_tasks.end()) {
            throw core::error::InvalidArgument("Unknown task ID '{}'", task_id);
        }

        return it->second->info().error_message;
    }

    /**
     * @brief Clear completed and failed tasks
     *
     * @param keep_failed Whether to keep failed tasks
     */
    void clear_completed(bool keep_failed = false) {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<std::string> to_remove;

        for (const auto& [task_id, task] : m_tasks) {
            if (task->status() == TaskStatus::Completed ||
                (task->status() == TaskStatus::Failed && !keep_failed) ||
                task->status() == TaskStatus::Cancelled) {
                to_remove.push_back(task_id);
            }
        }

        for (const auto& task_id : to_remove) {
            m_tasks.erase(task_id);
            m_dependents.erase(task_id);
        }
    }

    /**
     * @brief Shutdown the task system
     *
     * @param cancel_active Whether to cancel active tasks
     */
    void shutdown(bool cancel_active = true) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (!m_initialized) {
            return;
        }

        if (cancel_active) {
            for (auto& [task_id, task] : m_tasks) {
                if (task->status() == TaskStatus::Running) {
                    task->cancel();
                }
            }
        }

        ThreadPool::get_instance().wait_all();

        m_tasks.clear();
        m_dependents.clear();
        m_initialized = false;
    }

    /**
     * @brief Destructor - shuts down the task system
     */
    ~TaskSystem() { shutdown(); }

private:
    // Private constructor
    TaskSystem() : m_initialized(false), m_next_task_id(0) {}

    /**
     * @brief Execute a task
     *
     * @param task Task to execute
     */
    void execute_task(TaskPtr task) {
        task->set_status(TaskStatus::Running);
        ThreadPool::get_instance().submit([this, task]() {
            try {
                task->execute();

                std::lock_guard<std::mutex> lock(m_mutex);
                task->set_status(TaskStatus::Completed);

                process_dependents(task->info().id);
                m_condition.notify_all();
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(m_mutex);

                task->set_status(TaskStatus::Failed);
                task->set_error(e.what());

                m_condition.notify_all();
                if (task->info().propagate_exceptions) {
                    throw;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(m_mutex);

                task->set_status(TaskStatus::Failed);
                task->set_error("Unknown exception");

                m_condition.notify_all();
                if (task->info().propagate_exceptions) {
                    throw;
                }
            }
        });
    }

    /**
     * @brief Process dependent tasks after a task completes
     *
     * @param task_id ID of the completed task
     */
    void process_dependents(const std::string& task_id) {
        auto it = m_dependents.find(task_id);
        if (it == m_dependents.end()) {
            return;
        }

        for (const auto& dep_task_id : it->second) {
            auto task_it = m_tasks.find(dep_task_id);
            if (task_it == m_tasks.end()) {
                continue;
            }

            auto& task = task_it->second;
            bool can_execute = true;

            for (const auto& dep_id : task->info().dependencies) {
                auto dep_it = m_tasks.find(dep_id);
                if (dep_it == m_tasks.end()) {
                    can_execute = false;
                    break;
                }

                if (dep_it->second->status() != TaskStatus::Completed) {
                    can_execute = false;
                    break;
                }
            }

            if (can_execute) {
                execute_task(task);
            }
        }

        m_dependents.erase(task_id);
    }

    /**
     * @brief Generate a unique task ID
     *
     * @return std::string Unique task ID
     */
    std::string generate_task_id() {
        return "task_" + std::to_string(m_next_task_id++);
    }

    // Task system state
    bool m_initialized;
    std::atomic<size_t> m_next_task_id;

    // Task storage
    std::unordered_map<std::string, TaskPtr> m_tasks;

    // Task dependencies (task_id -> list of dependent task IDs)
    std::unordered_map<std::string, std::vector<std::string>> m_dependents;

    // Synchronization
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
};

}  // namespace brezel::tensor::parallel