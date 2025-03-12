#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/parallel/thread_pool.hpp>
#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace brezel::tensor::parallel {
/**
 * @brief Parallelization configuration options
 */
struct BREZEL_API ParallelConfig {
    /// @brief  Minimum number of threads to use (0 = auto)
    size_t num_threads = 0;

    /// @brief Minimum workload size per thread
    size_t grain_size = 1024;

    /// @brief Whether to use dynamic scheduling
    bool dynamic_schedule = true;

    /// @brief Priority of the task (higher more important)
    int priority = 0;
};

/**
 * @brief Get the default parallel configuration
 * @return ParallelConfig Default parallel configuration
 */
BREZEL_NODISCARD inline ParallelConfig default_parallel_config() {
    return ParallelConfig{};
}

/**
 * @brief Parallel for loop execution
 *
 * @details Executes a function over a range of indices in parallel using TBB.
 * This is a wrapper around tbb::parallel_for that provides a simpler interface
 * and integrates with the ThreadPool class.
 *
 * @tparam Index Index type (must be an integral type)
 * @tparam Func Function type
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param func Function to execute for each index
 * @param config Parallelization configuration options
 */
template <typename Index, typename Func>
void parallel_for(Index start, Index end, Func&& func,
                  const ParallelConfig& config = default_parallel_config()) {
    static_assert(std::is_integral_v<Index>,
                  "Index type must be an intergral type");

    if (end <= start) {
        return;
    }

    const auto range_size = end - start;
    if (range_size <= static_cast<Index>(config.grain_size)) {
        for (Index i = start; i < end; ++i) {
            func(i);
        }

        return;
    }

    auto& pool = ThreadPool::get_instance();
    if (!pool.num_threads()) {
        pool.initialize(config.num_threads);
    }

    size_t effective_grain_size = config.grain_size;
    if (range_size > 0) {
        if (static_cast<size_t>(range_size) > 1'000'000) {
            effective_grain_size =
                std::max(effective_grain_size, static_cast<size_t>(range_size) /
                                                   (4 * pool.num_threads()));
        }
    }

    // Execute parallel for using TBB
    pool.execute_in_arena([&]() {
        tbb::parallel_for(
            tbb::blocked_range<Index>(start, end,
                                      static_cast<Index>(effective_grain_size)),
            [&](const tbb::blocked_range<Index>& range) {
                for (Index i = range.begin(); i < range.end(); ++i) {
                    func(i);
                }
            },
            tbb::simple_partitioner{});
    });
}

/**
 * @brief 2D parallel for loop execution
 *
 * @details Executes a function over a 2D range of indices in parallel using
 * TBB.
 * @details For 2D ranges, we linearize the iterations and parallelize along the
 * outer dimension.
 *
 * @tparam Index Index type (must be an integer type)
 * @tparam Func Function type
 * @param start_i First dimension start index (inclusive)
 * @param end_i First dimension end index (exclusive)
 * @param start_j Second dimension start index (inclusive)
 * @param end_j Second dimension end index (exclusive)
 * @param func Function to execute for each (i,j) pair
 * @param config Parallel configuration options
 */
template <typename Index, typename Func>
void parallel_for_2d(Index start_i, Index end_i, Index start_j, Index end_j,
                     Func&& func,
                     const ParallelConfig& config = default_parallel_config()) {
    static_assert(std::is_integral_v<Index>, "Index type must be an integer");

    if (end_i <= start_i || end_j <= start_j) {
        return;
    }

    const auto range_i_size = end_i - start_i;
    const auto range_j_size = end_j - start_j;
    const auto total_size = range_i_size * range_j_size;

    if (total_size <= static_cast<Index>(config.grain_size)) {
        for (Index i = start_i; i < end_i; ++i) {
            for (Index j = start_j; j < end_j; ++j) {
                func(i, j);
            }
        }

        return;
    }

    auto& pool = ThreadPool::get_instance();
    if (!pool.num_threads()) {
        pool.initialize(config.num_threads);
    }

    parallel_for(
        start_i, end_i,
        [&](Index i) {
            for (Index j = start_j; j < end_j; ++j) {
                func(i, j);
            }
        },
        config);
}

/**
 * @brief Parallel reduction over a range of indices
 *
 * @details Performs a parallel reduction over a range of indices using TBB.
 *
 * @tparam Index Index type (must be an integer type)
 * @tparam T Result type
 * @tparam MapFunc Mapping function type
 * @tparam ReduceFunc Reduction function type
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param identity Identity value for the reduction
 * @param map_func Function to map each index to a value
 * @param reduce_func Function to combine two values
 * @param config Parallel configuration options
 * @return T Result of the reduction
 */
template <typename Index, typename T, typename MapFunc, typename ReduceFunc>
BREZEL_NODISCARD T
parallel_reduce(Index start, Index end, const T& identity, MapFunc&& map_fun,
                MapFunc&& reduce_func,
                const ParallelConfig& config = default_parallel_config()) {
    static_assert(std::is_integral_v<Index>, "Index type must be an interger");

    if (end <= start) {
        return identity;
    }

    const auto range_size = end - start;
    if (range_size <= static_cast<Index>(config.grain_size)) {
        T result = identity;
        for (Index i = start; i < end; ++i) {
            result = reduce_func(result, map_fun(i));
        }

        return result;
    }

    auto& pool = ThreadPool::get_instance();
    if (!pool.num_threads()) {
        pool.initialize(config.num_threads);
    }

    size_t effective_grain_size = config.grain_size;
    if (range_size > 0) {
        if (static_cast<size_t>(range_size) > 1'000'000) {
            effective_grain_size =
                std::max(effective_grain_size, static_cast<size_t>(range_size) /
                                                   (4 * pool.num_threads()));
        }
    }

    return pool.execute_in_arena([&]() {
        return tbb::parallel_reduce(
            tbb::blocked_range<Index>(start, end,
                                      static_cast<Index>(effective_grain_size)),
            identity,
            [&](const tbb::blocked_range<Index>& range, T init) {
                T result = init;
                for (Index i = range.begin(); i < range.end(); ++i) {
                    result = reduce_func(result, map_func(i));
                }

                return result;
            },
            reduce_func);
    });
}

/**
 * @brief Parallel execution of a collection of tasks
 *
 * @details Executes a collection of tasks in parallel using the thread pool.
 *
 * @tparam T Task type
 * @param tasks Collection of tasks to execute
 * @param config Parallel configuration options
 */
template <typename T>
void parallel_invoke(const std::vector<T>& tasks,
                     const ParallelConfig& config = default_parallel_config()) {
    auto& pool = ThreadPool::get_instance();
    if (!pool.num_threads()) {
        pool.initialize(config.num_threads);
    }

    std::vector<std::function<void>> futures;
    futures.reserve(tasks.size());

    for (const auto& task : tasks) {
        futures.push_back(pool.submit(task));
    }

    for (auto& future : futures) {
        future.wait();
    }
}

/**
 * @brief Create a parallel task
 *
 * @details Creates a task that can be submitted to the thread pool.
 *
 * @tparam F Function type
 * @tparam Args Argument types
 * @param f Function to execute
 * @param args Arguments to pass to the function
 * @return std::function<void()> Task function
 */
template <typename F, typename... Args>
std::function<void()> make_task(F&& f, Args&&... args) {
    return std::bind(std::forward<F>(f), std::forward<Args>(args)...);
}
}  // namespace brezel::tensor::parallel