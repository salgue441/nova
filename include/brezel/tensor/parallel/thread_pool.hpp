#pragma once

#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace brezel::tensor::parallel {
/**
 * @brief Thread pool implementation for parallel execution tasks
 *
 * @details Provides a configurable thread pool that can be used to execute
 * tasks in parallel. The pool manages a fixed number of worker threads and
 * distributes tasks among them. This implementation uses a work-stealing
 * approach for better load balancing and avoids creating unnecessary threads.
 */
class BREZEL_API ThreadPool {
public:
    /**
     * @brief Get a reference to the singleton thread pool instance
     * @return ThreadPool& reference to the singleton instance
     */
    static ThreadPool& get_instance() {
        static ThreadPool instance;
        return instance;
    }

    /**
     * @brief Initializes the thead pool with a specific number of threadsa
     *
     * @param num_threads Number of threads to use (0 means auto-detect)
     */
    void initialize(size_t num_threads = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_initialized) {
            return;
        }

        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();

            if (num_threads == 0) {
                num_threads = 4;
            }
        }

        m_global_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, num_threads);

        m_arena =
            std::make_unique<tbb::task_arena>(static_cast<int>(num_threads));

        m_workers.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back(&ThreadPool::worker_thread, this);
        }

        m_num_threads = num_threads;
        m_initialized = true;
    }

    /**
     * @brief Shutdown the thread pool and wait for all tasks to complete
     *
     * @param wait_for_tasks If true, the function waits for all tasks to
     * complete before shutting down the thread pool
     */
    void shutdown(bool wait_for_tasks = true) {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (!m_initialized) {
            return;
        }

        if (wait_for_tasks) {
            m_condition.wait(lock, [this] {
                return m_tasks.empty() && m_active_tasks == 0;
            });
        }

        m_stop = true;
        lock.unlock();
        m_condition.notify_all();

        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        m_workers.clear();
        m_arena.reset();
        m_global_control.reset();

        m_initialized = false;
        m_stop = false;
    }

    /**
     * @brief Submit a task to the thread pool for execution
     *
     * @tparam F Type of the task function
     * @tparam Args Types of the task function arguments
     * @param f Task function to execute
     * @param args Arguments to pass to the task function
     * @return std::future<typename std::invoke_result<F, Args...>::type> Future
     * object representing the result of the task
     */
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        if (!m_initialized) {
            initialize();
        }

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> future = task->get_future();

        {
            std::unique_lock<std::mutex> lock(m_task_mutex);
            if (m_stop) {
                throw brezel::core::error::RuntimeError(
                    "Cannot submit task to stopped thread pool");
            }

            m_tasks.emplace([task]() { (*task)(); });
        }

        m_condition.notify_one();
        return future;
    }

    /**
     * @brief Wait for all submitted tasks to complete
     */
    void wait_all() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(
            lock, [this] { return m_tasks.empty() && m_active_tasks == 0; });
    }

    /**
     * @brief Get the number of threads in the pool
     *
     * @return size_t Number of threads
     */
    size_t num_threads() const { return m_num_threads; }

    /**
     * @brief Get the number of currently active tasks
     *
     * @return size_t Number of active tasks
     */
    size_t active_tasks() const { return m_active_tasks; }

    /**
     * @brief Get the number of tasks in the queue
     *
     * @return size_t Number of queued tasks
     */
    size_t queued_tasks() const {
        std::lock_guard<std::mutex> lock(m_task_mutex);
        return m_tasks.size();
    }

    /**
     * @brief Execute a function in the TBB task arena
     *
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return typename std::invoke_result<F, Args...>::type Function result
     */
    template <class F, class... Args>
    auto execute_in_arena(F&& f, Args&&... args) ->
        typename std::invoke_result<F, Args...>::type {
        if (!m_initialized) {
            initialize();
        }

        return m_arena->execute(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    }

    /**
     * @brief Destructor that shuts down the thread pool
     */
    ~ThreadPool() { shutdown(); }

private:
    ThreadPool()
        : m_stop(false),
          m_initialized(false),
          m_active_tasks(0),
          m_num_threads(0) {}

    /**
     * @brief Function executed by each worker thread in the thread pool.
     *
     * This function runs in an infinite loop, waiting for tasks to be added to
     * the task queue. It acquires a lock on the task queue, waits for a
     * condition variable to be notified, and then processes tasks from the
     * queue. If the thread pool is stopped and there are no remaining tasks,
     * the thread exits the loop and terminates.
     *
     * The function ensures that the number of active tasks is correctly tracked
     * and notifies other threads when tasks are completed.
     */
    void worker_thread() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(m_task_mutex);
                m_condition.wait(lock,
                                 [this] { return m_stop || !m_tasks.empty(); });

                if (m_stop && m_tasks.empty()) {
                    return;
                }

                task = std::move(m_tasks.front());
                m_tasks.pop();

                ++m_active_tasks;
            }

            task();
            --m_active_tasks;
            m_condition.notify_all();
        }
    }

    // Pool state
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::unique_ptr<tbb::task_arena> m_arena;
    std::unique_ptr<tbb::global_control> m_global_control;

    // Synchronization
    mutable std::mutex m_mutex;
    mutable std::mutex m_task_mutex;
    std::condition_variable m_condition;

    // State flags
    std::atomic<bool> m_stop;
    bool m_initialized;
    std::atomic<size_t> m_active_tasks;
    size_t m_num_threads;
};
}  // namespace brezel::tensor::parallel