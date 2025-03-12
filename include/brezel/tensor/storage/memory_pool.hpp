#pragma once

#include <tbb/concurrent_queue.h>

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/storage/allocator.hpp>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace brezel::tensor::storage {

/**
 * @brief Memory pool configuration options
 */
struct BREZEL_API MemoryPoolConfig {
    /// Maximum pool size in bytes
    size_t max_pool_size = 1024 * 1024 * 1024;  // 1GB default

    /// Maximum number of blocks per size class
    size_t max_blocks_per_size = 32;

    /// Minimum block size that can be pooled
    size_t min_pooled_size = 64;

    /// Maximum block size that can be pooled
    size_t max_pooled_size = 1024 * 1024 * 16;  // 16MB default

    /// Whether to cache blocks for reuse
    bool enable_caching = true;

    /// Allocator type to use for memory allocation
    AllocatorFactory::AllocatorType allocator_type =
        AllocatorFactory::AllocatorType::Default;
};

/**
 * @brief Memory pool for efficient allocation and reuse
 *
 * @details Implements a size-based memory pooling strategy to reduce allocation
 * overhead and memory fragmentation. Unused memory blocks are cached for future
 * reuse, improving performance for tensor operations that frequently allocate
 * temporary buffers.
 */
class BREZEL_API MemoryPool {
public:
    /**
     * @brief Get the singleton instance of the memory pool
     *
     * @return MemoryPool& Reference to singleton instance
     */
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    /**
     * @brief Configure the memory pool
     *
     * @param config Configuration options
     */
    void configure(const MemoryPoolConfig& config) {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_config = config;
        if (!m_allocator) {
            m_allocator = AllocatorFactory::create(m_config.allocator_type);
        }

        if (!m_config.enable_caching) {
            for (auto& [size, queue] : m_block_pools) {
                void* block = nullptr;
                while (queue.try_pop(block)) {
                    if (block) {
                        m_allocator->deallocate(block, size);
                        m_current_pool_size -= size;
                    }
                }
            }

            m_block_pools.clear();
        }
    }

    /**
     * @brief Allocate memory from the pool
     *
     * @param size Size in bytes to allocate
     * @return void* Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    void* allocate(size_t size) {
        if (size == 0) {
            return nullptr;
        }

        if (!is_poolable(size)) {
            void* ptr = m_allocator->allocate(size);
            if (!ptr) {
                throw std::bad_alloc();
            }

            return ptr;
        }

        const size_t size_class = get_size_class(size);
        void* block = nullptr;

        if (m_config.enable_caching) {
            auto& pool = m_block_pools[size_class];

            if (pool.try_pop(block)) {
                m_current_pool_size -= size_class;
                return block;
            }
        }

        block = m_allocator->allocate(size_class);
        if (!block) {
            throw std::bad_alloc();
        }

        return block;
    }

    /**
     * @brief Deallocate memory, potentially returning it to the pool
     *
     * @param ptr Pointer to memory
     * @param size Size of the allocation
     */
    void deallocate(void* ptr, size_t size) noexcept {
        if (!ptr) {
            return;
        }

        if (!is_poolable(size)) {
            m_allocator->deallocate(ptr, size);
            return;
        }

        const size_t size_class = get_size_class(size);
        if (m_config.enable_caching) {
            auto& pool = m_block_pools[size_class];

            if (pool.unsafe_size() < m_config.max_blocks_per_size &&
                m_current_pool_size + size_class <= m_config.max_pool_size) {
                pool.push(ptr);

                m_current_pool_size += size_class;
                return;
            }
        }

        m_allocator->deallocate(ptr, size_class);
    }

    /**
     * @brief Allocate typed memory from the pool
     *
     * @tparam T Element type
     * @param n Number of elements
     * @return T* Pointer to allocated memory
     */
    template <typename T>
    T* allocate_typed(size_t n) {
        if (n == 0) {
            return nullptr;
        }

        void* ptr = allocate(n * sizeof(T));
        return static_cast<T*>(ptr);
    }

    /**
     * @brief Deallocate typed memory
     *
     * @tparam T Element type
     * @param ptr Pointer to memory
     * @param n Number of elements
     */
    template <typename T>
    void deallocate_typed(T* ptr, size_t n) noexcept {
        if (!ptr) {
            return;
        }

        deallocate(static_cast<void*>(ptr), n * sizeof(T));
    }

    /**
     * @brief Release all cached memory
     */
    void release_all() {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (auto& [size, queue] : m_block_pools) {
            void* block = nullptr;

            while (queue.try_pop(block)) {
                if (block) {
                    m_allocator->deallocate(block, size);
                }
            }
        }

        m_block_pools.clear();
        m_current_pool_size = 0;
    }

    /**
     * @brief Shrink the pool to a target size
     *
     * @param target_size Target size in bytes
     */
    void shrink_to(size_t target_size) {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_current_pool_size <= target_size) {
            return;
        }

        size_t to_release = m_current_pool_size - target_size;
        std::vector<size_t> sizes;
        for (const auto& [size, _] : m_block_pools) {
            sizes.push_back(size);
        }

        std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());

        for (size_t size : sizes) {
            auto& queue = m_block_pools[size];

            void* block = nullptr;
            while (to_release >= size && queue.try_pop(block)) {
                if (block) {
                    m_allocator->deallocate(block, size);
                    m_current_pool_size -= size;
                    to_release -= size;
                }
            }

            if (to_release == 0) {
                break;
            }
        }
    }

    /**
     * @brief Get current pool size in bytes
     *
     * @return size_t Current size
     */
    size_t pool_size() const noexcept { return m_current_pool_size; }

    /**
     * @brief Set the allocator
     *
     * @param allocator Allocator implementation
     */
    void set_allocator(std::unique_ptr<AllocatorInterface> allocator) {
        std::lock_guard<std::mutex> lock(m_mutex);

        release_all();
        m_allocator = std::move(allocator);
    }

    /**
     * @brief Get the alignment of the memory pool
     *
     * @return size_t Alignment in bytes
     */
    size_t alignment() const noexcept {
        return m_allocator ? m_allocator->alignment()
                           : AlignmentConstants::kDefaultAlignment;
    }

private:
    /**
     * @brief Private constructor for singleton
     */
    MemoryPool() : m_current_pool_size(0) {
        m_config = MemoryPoolConfig{};
        m_allocator = AllocatorFactory::create(m_config.allocator_type);
    }

    /**
     * @brief Check if a size is suitable for pooling
     *
     * @param size Size in bytes
     * @return bool True if poolable
     */
    bool is_poolable(size_t size) const noexcept {
        return m_config.enable_caching && size >= m_config.min_pooled_size &&
               size <= m_config.max_pooled_size;
    }

    /**
     * @brief Get the size class for a size
     *
     * @details Rounds up to the nearest power of 2 or multiple of 1024 for
     * larger sizes. This helps reduce fragmentation by standardizing allocation
     * sizes.
     *
     * @param size Size in bytes
     * @return size_t Size class
     */
    size_t get_size_class(size_t size) const noexcept {
        if (size <= 4096) {
            size_t power = 1;
            while (power < size) {
                power *= 2;
            }

            return power;
        }

        const size_t block_size = 4096;
        return ((size + block_size - 1) / block_size) * block_size;
    }

    MemoryPoolConfig m_config;
    std::unique_ptr<AllocatorInterface> m_allocator;
    std::unordered_map<size_t, tbb::concurrent_queue<void*>> m_block_pools;
    std::atomic<size_t> m_current_pool_size;
    mutable std::mutex m_mutex;
};

/**
 * @brief STL-compatible typed allocator that uses the memory pool
 *
 * @tparam T Element type
 */
template <typename T>
class BREZEL_API PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };

    /**
     * @brief Default constructor
     */
    PoolAllocator() = default;

    /**
     * @brief Copy constructor
     *
     * @param other Other allocator
     */
    PoolAllocator(const PoolAllocator& other) = default;

    /**
     * @brief Move constructor
     *
     * @param other Other allocator
     */
    PoolAllocator(PoolAllocator&& other) noexcept = default;

    /**
     * @brief Copy assignment
     *
     * @param other Other allocator
     * @return PoolAllocator& Reference to this
     */
    PoolAllocator& operator=(const PoolAllocator& other) = default;

    /**
     * @brief Move assignment
     *
     * @param other Other allocator
     * @return PoolAllocator& Reference to this
     */
    PoolAllocator& operator=(PoolAllocator&& other) noexcept = default;

    /**
     * @brief Rebinding constructor
     *
     * @tparam U Other element type
     * @param other Other allocator
     */
    template <typename U>
    PoolAllocator(const PoolAllocator<U>& other) {}

    /**
     * @brief Destructor
     */
    ~PoolAllocator() = default;

    /**
     * @brief Allocate memory for n elements
     *
     * @param n Number of elements
     * @return pointer Pointer to allocated memory
     */
    pointer allocate(size_type n) {
        return MemoryPool::instance().allocate_typed<T>(n);
    }

    /**
     * @brief Deallocate memory
     *
     * @param p Pointer to memory
     * @param n Number of elements
     */
    void deallocate(pointer p, size_type n) noexcept {
        MemoryPool::instance().deallocate_typed(p, n);
    }

    /**
     * @brief Equality comparison
     *
     * @param other Other allocator
     * @return bool True if equal
     */
    bool operator==(const PoolAllocator& other) const noexcept { return true; }

    /**
     * @brief Inequality comparison
     *
     * @param other Other allocator
     * @return bool True if not equal
     */
    bool operator!=(const PoolAllocator& other) const noexcept { return false; }
};

}  // namespace brezel::tensor::storage