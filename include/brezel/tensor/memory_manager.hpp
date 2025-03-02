#pragma once

#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/scalable_allocator.h>

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

#if defined(BREZEL_PLATFORM_WINDOWS)
#include <Windows.h>
#endif

namespace brezel::tensor::memory {
/**
 * @brief Enum for memory alignment options
 */
enum class Alignment : size_t {
    Default = alignof(std::max_align_t),
    L1Cache = BREZEL_CACHE_LINE_SIZE,
    AVX = 32,
    AVX512 = 64
};

/**
 * @brief Memory allocation flags for fine-tuning allocator behavior
 */
enum class AllocFlags : uint32_t {
    None = 0,
    ZeroMemory = 1 << 0,       // Initialize memory with zeros
    GrowUpward = 1 << 1,       // Prefer growing memory upward (stacks)
    GrowDownward = 1 << 2,     // Prefer growing memory downard
    AllowLargePages = 1 << 3,  // Allow large page allocations
    DoNotTrack = 1 << 4,       // Do no track this allocation in debugging
    AllowSlowInitialization = 1 << 5  // Allow slow init
};

inline AllocFlags operator|(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<uint32_t>(a) |
                                   static_cast<uint32_t>(b));
}

inline AllocFlags operator&(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<uint32_t>(a) &
                                   static_cast<uint32_t>(b));
}

/**
 * @brief Base allocator interface
 * @details Defines the contract for memory allocators in the framework
 */
class BREZEL_API IAllocator {
public:
    virtual ~IAllocator() = default;

    /**
     * @brief Allocate memory block
     *
     * @param size Size of the memory block in bytes
     * @param alignment Memory alignment requirement
     * @param flags Allocation flags for behavioral modifications
     * @return Pointer to the allocated memory
     */
    BREZEL_NODISCARD virtual void* allocate(
        size_t size, Alignment alignment = Alignment::Default,
        AllocFlags flags = AllocFlags::None) = 0;

    /**
     * @brief Deallocate memory block
     *
     * @param ptr Pointer to memory block to deallocate
     * @param size Size of the memory block (optional, may help some allocators)
     */
    virtual void deallocate(void* ptr, size_t size = 0) noexcept = 0;

    /**
     * @brief Check if allocator supports resize operations
     * @return True if resize operations are supported
     */
    BREZEL_NODISCARD virtual bool supports_resize() const noexcept {
        return false;
    }

    /**
     * @brief Attempt to resize an existing memory block
     *
     * @param ptr Pointer to memory block to resize
     * @param old_size Current size of the memory block
     * @param new_size Desired new size of the memory block
     * @param flags Allocation flags for behavioral modifications
     * @return Pointer to the resized memory (may be different from input)
     */
    BREZEL_NODISCARD virtual void* resize(void* ptr, size_t old_size,
                                          size_t new_size,
                                          AllocFlags flags = AllocFlags::None) {
        // Default implementation falls back to allocate/deallocate
        void* new_ptr = allocate(new_size, Alignment::Default, flags);
        if (ptr && old_size > 0) {
            std::memcpy(new_ptr, ptr, std::min(old_size, new_size));
            deallocate(ptr, old_size);
        }

        return new_ptr;
    }

    /**
     * @brief Get memory allocation statistics
     * @return Current memory usage and other stats
     */
    BREZEL_NODISCARD virtual size_t get_allocated_bytes() const noexcept {
        return 0;
    }
};

/**
 * @brief Default allocator using TBB's cache-aligned allocator
 * @details Provides efficient thread-safe memory allocation with proper cache
 * alignment
 */
class BREZEL_API DefaultAllocator : public IAllocator {
public:
    // Constructor
    DefaultAllocator() = default;

    // Destructor to ensure proper cleanup
    ~DefaultAllocator() {
        // Clean up any remaining allocations
        for (auto& ptr : m_aligned_allocations) {
#if defined(BREZEL_PLATFORM_WINDOWS)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
    }

    /**
     * @brief Allocate memory block with cache alignment
     *
     * @param size Size of the memory block in bytes
     * @param alignment Memory alignment requirement
     * @param flags Allocation flags for behavior modifications
     * @return Pointer to the allocated memory
     */
    BREZEL_NODISCARD void* allocate(
        size_t size, Alignment alignment = Alignment::Default,
        AllocFlags flags = AllocFlags::None) override {
        if (size == 0)
            return nullptr;

        void* ptr = nullptr;
        const bool zero_memory =
            (flags & AllocFlags::ZeroMemory) == AllocFlags::ZeroMemory;

        if (alignment == Alignment::Default) {
            ptr = tbb::scalable_allocator<std::byte>().allocate(size);
        } else {
            const size_t align_val = static_cast<size_t>(alignment);

#if defined(BREZEL_PLATFORM_WINDOWS)
            ptr = _aligned_malloc(size, align_val);
#else
            if (posix_memalign(&ptr, align_val, size) != 0) {
                throw brezel::core::error::RuntimeError(
                    "Failed to allocate {} bytes with alignment {}", size,
                    align_val);
            }
#endif

            // Track aligned allocations
            if (ptr) {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_aligned_allocations.insert(ptr);
            }
        }

        if (!ptr) {
            throw brezel::core::error::RuntimeError(
                "Memory allocation failed for {} bytes", size);
        }

        if (zero_memory) {
            std::memset(ptr, 0, size);
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_allocated_bytes += size;
        }

        return ptr;
    }

    /**
     * @brief Deallocate memory block
     *
     * @param ptr Pointer to memory block to deallocate
     * @param size Size of the memory block (used for stats)
     */
    void deallocate(void* ptr, size_t size = 0) noexcept override {
        if (!ptr)
            return;

        if (size > 0) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_allocated_bytes -= std::min(m_allocated_bytes, size);
        }

        // Check if this was an aligned allocation
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_aligned_allocations.find(ptr);
            if (it != m_aligned_allocations.end()) {
#if defined(BREZEL_PLATFORM_WINDOWS)
                _aligned_free(ptr);
#else
                free(ptr);
#endif
                m_aligned_allocations.erase(it);
                return;
            }
        }

        // Use TBB deallocator for regular allocations
        tbb::scalable_allocator<std::byte>().deallocate(
            static_cast<std::byte*>(ptr), size);
    }

    /**
     * @brief Get memory allocation statistics
     * @return Current memory usage in bytes
     */
    BREZEL_NODISCARD size_t get_allocated_bytes() const noexcept override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_allocated_bytes;
    }

private:
    mutable std::mutex m_mutex;
    size_t m_allocated_bytes = 0;
    std::unordered_set<void*> m_aligned_allocations;
};

/**
 * @brief Memory pool allocator for efficient small-block allocation
 * @details Uses a pool of fixed-size memory blocks for efficient allocation
 *          of many small objects of the same size
 */
class BREZEL_API PoolAllocator : public IAllocator {
public:
    /**
     * @brief Construct a pool allocator
     *
     * @param block_size Size of each block in the pool
     * @param initial_capacity Initial number of blocks to allocate
     * @param alignment Memory alignment requirement
     */
    explicit PoolAllocator(size_t block_size, size_t initial_capacity = 64,
                           Alignment alignment = Alignment::Default)
        : m_block_size(block_size), m_alignment(alignment) {
        grow_pool(initial_capacity);
    }

    ~PoolAllocator() {
        for (auto& block : m_memory_blocks) {
#if defined(BREZEL_PLATFORM_WINDOWS)
            _aligned_free(block);
#else
            free(block);
#endif
        }
    }

    /**
     * @brief Allocate memory block from the pool
     *
     * @param size Size of the memory block in bytes (must match block_size)
     * @param alignment Memory alignment requirement (ignored, uses pool
     * alignment)
     * @param flags Allocation flags for behavior modifications
     * @return Pointer to the allocated memory
     */
    BREZEL_NODISCARD void* allocate(
        size_t size, Alignment alignment = Alignment::Default,
        AllocFlags flags = AllocFlags::None) override {
        if (size > m_block_size) {
            throw brezel::core::error::RuntimeError(
                "Requested allocation size {} exceeds pool block size {}", size,
                m_block_size);
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_free_list.empty()) {
            grow_pool(m_memory_blocks.size() * 2);
        }

        void* ptr = m_free_list.back();
        m_free_list.pop_back();

        if ((flags & AllocFlags::ZeroMemory) == AllocFlags::ZeroMemory) {
            std::memset(ptr, 0, m_block_size);
        }

        m_allocated_blocks++;
        return ptr;
    }

    /**
     * @brief Return memory block to the pool
     *
     * @param ptr Pointer to memory block to deallocate
     * @param size Size of the memory block (ignored for pools)
     */
    void deallocate(void* ptr, size_t size = 0) noexcept override {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(m_mutex);
        bool found = false;

        for (auto& block : m_memory_blocks) {
            uintptr_t block_addr = reinterpret_cast<uintptr_t>(block);
            uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);

            if (ptr_addr >= block_addr &&
                ptr_addr < block_addr + m_blocks_per_chunk * m_block_size) {
                found = true;
                break;
            }
        }

        if (!found)
            return;

        m_free_list.push_back(ptr);
        m_allocated_blocks--;
    }

    /**
     * @brief Check if allocator supports resize operations
     * @return Always false for pool allocators
     */
    BREZEL_NODISCARD bool supports_resize() const noexcept override {
        return false;
    }

    /**
     * @brief Get memory allocation statistics
     * @return Total allocated bytes
     */
    BREZEL_NODISCARD size_t get_allocated_bytes() const noexcept override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_allocated_blocks * m_block_size;
    }

private:
    /**
     * @brief Grow the memory pool by adding more blocks
     *
     * @param additional_blocks Number of blocks to add
     */
    void grow_pool(size_t additional_blocks) {
        const size_t min_blocks = 16;
        additional_blocks = std::max(additional_blocks, min_blocks);

        const size_t align_val = static_cast<size_t>(m_alignment);
        const size_t chunk_size = additional_blocks * m_block_size;
        void* chunk = nullptr;

#if defined(BREZEL_PLATFORM_WINDOWS)
        chunk = _aligned_malloc(chunk_size, align_val);
#else
        if (posix_memalign(&chunk, align_val, chunk_size) != 0) {
            throw brezel::core::error::RuntimeError(
                "Failed to allocate chunk of {} bytes with alignment {}",
                chunk_size, align_val);
        }
#endif

        if (!chunk) {
            throw brezel::core::error::RuntimeError(
                "Failed to grow memory pool by {} blocks", additional_blocks);
        }

        m_memory_blocks.push_back(chunk);
        auto* current = static_cast<std::byte*>(chunk);

        for (size_t i = 0; i < additional_blocks; ++i) {
            m_free_list.push_back(current);
            current += m_block_size;
        }

        m_blocks_per_chunk = additional_blocks;
    }

    size_t m_block_size;
    Alignment m_alignment;
    std::vector<void*> m_memory_blocks;
    std::vector<void*> m_free_list;
    size_t m_blocks_per_chunk = 0;
    size_t m_allocated_blocks = 0;
    mutable std::mutex m_mutex;
};

/**
 * @brief Memory manager singleton
 * @details Central registry for creating and managing allocators
 */
class BREZEL_API MemoryManager {
public:
    /**
     * @brief Gets the singleton instance
     * @return Reference to the memory manager
     */
    static MemoryManager& instance() {
        static MemoryManager instance;
        return instance;
    }

    /**
     * @brief Get the default allocator
     * @return Reference to the default allocator
     */
    BREZEL_NODISCARD IAllocator& default_allocator() {
        return m_default_allocator;
    }

    /**
     * @brief Create or get a pool allocator for a specific block size
     *
     * @param block_size Size of blocks in the pool
     * @return Reference to the pool allocator
     */
    BREZEL_NODISCARD IAllocator& get_pool_allocator(size_t block_size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_pool_allocators.find(block_size);

        if (it == m_pool_allocators.end()) {
            auto [new_it, inserted] = m_pool_allocators.emplace(
                block_size, std::make_unique<PoolAllocator>(block_size));

            return *new_it->second;
        }

        return *it->second;
    }

    /**
     * @brief Get memory allocation statistics
     * @return Total allocated bytes across all managed allocators
     */
    BREZEL_NODISCARD size_t get_total_allocated_bytes() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t total = m_default_allocator.get_allocated_bytes();

        for (const auto& [size, allocator] : m_pool_allocators) {
            total += allocator->get_allocated_bytes();
        }

        return total;
    }

private:
    MemoryManager() = default;
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;

    DefaultAllocator m_default_allocator;
    std::unordered_map<size_t, std::unique_ptr<IAllocator>> m_pool_allocators;
    mutable std::mutex m_mutex;
};

/**
 * @brief Smart pointer for memory managed by custom allocators
 * @details Provides RAII semantics for memory allocated through the memory
 * system
 *
 * @tparam T Type of the managed object
 */
template <typename T>
class BREZEL_API AllocatedPtr {
public:
    using element_type = T;
    using pointer = T*;

    /**
     * @brief Default constructor (null pointer)
     */
    AllocatedPtr() noexcept : m_ptr(nullptr), m_allocator(nullptr), m_size(0) {}

    /**
     * @brief Construct from raw pointer and allocator
     * @param ptr Raw pointer to manage
     * @param allocator Allocator that created the memory
     * @param size Size of the allocation in bytes
     */
    explicit AllocatedPtr(pointer ptr, IAllocator* allocator = nullptr,
                          size_t size = 0) noexcept
        : m_ptr(ptr), m_allocator(allocator), m_size(size) {}

    /**
     * @brief Destructor, automatically deallocates memory
     */
    ~AllocatedPtr() { reset(); }

    // Move semantics
    AllocatedPtr(AllocatedPtr&& other) noexcept
        : m_ptr(other.m_ptr),
          m_allocator(other.m_allocator),
          m_size(other.m_size) {
        other.m_ptr = nullptr;
        other.m_allocator = nullptr;
        other.m_size = 0;
    }

    AllocatedPtr& operator=(AllocatedPtr&& other) noexcept {
        if (this != &other) {
            reset();

            m_ptr = other.m_ptr;
            m_allocator = other.m_allocator;
            m_size = other.m_size;

            other.m_ptr = nullptr;
            other.m_allocator = nullptr;
            other.m_size = 0;
        }

        return *this;
    }

    // Delete copy operations
    AllocatedPtr(const AllocatedPtr&) = delete;
    AllocatedPtr& operator=(const AllocatedPtr&) = delete;

    /**
     * @brief Resets the pointer (deallocate if non-null)
     *
     * @param ptr New pointer to manage (optional)
     * @param allocator New allocator (optional)
     * @param size New size in bytes (optional)
     */
    void reset(pointer ptr = nullptr, IAllocator* allocator = nullptr,
               size_t size = 0) noexcept {
        if (m_ptr && m_allocator) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                m_ptr->~T();
            }

            m_allocator->deallocate(m_ptr, m_size);
        }

        m_ptr = ptr;
        m_allocator = allocator;
        m_size = size;
    }

    /**
     * @brief Get the raw pointer
     * @return Raw pointer
     */
    BREZEL_NODISCARD pointer get() const noexcept { return m_ptr; }

    /**
     * @brief Dereference operator
     * @return Reference to managed object
     */
    BREZEL_NODISCARD T& operator*() const { return *m_ptr; }

    /**
     * @brief Member access operator
     * @return Pointer to managed object
     */
    BREZEL_NODISCARD pointer operator->() const noexcept { return m_ptr; }

    /**
     * @brief Boolean conversion operator
     * @return True if pointer is non-null
     */
    BREZEL_NODISCARD explicit operator bool() const noexcept {
        return m_ptr != nullptr;
    }

    /**
     * @brief Get the size of the allocation
     * @return Size in bytes
     */
    BREZEL_NODISCARD size_t size() const noexcept { return m_size; }

    /**
     * @brief Get the allocator
     * @return Pointer to the allocator
     */
    BREZEL_NODISCARD IAllocator* allocator() const noexcept {
        return m_allocator;
    }

private:
    pointer m_ptr;
    IAllocator* m_allocator;
    size_t m_size;
};

/**
 * @brief Create an allocated pointer for a single object
 *
 * @tparam T Type of object to create
 * @tparam Args Constructor argument types
 * @param allocator Allocator to use
 * @param alignment Memory alignment
 * @param args Constructor arguments
 * @return Smart pointer to the allocated object
 */
template <typename T, typename... Args>
BREZEL_NODISCARD AllocatedPtr<T> make_allocated(
    IAllocator& allocator = MemoryManager::instance().default_allocator(),
    Alignment alignment = Alignment::Default, Args&&... args) {
    void* memory = allocator.allocate(sizeof(T), alignment);
    T* object = new (memory) T(std::forward<Args>(args)...);

    return AllocatedPtr<T>(object, &allocator, sizeof(T));
}

/**
 * @brief Creates an allocated array
 *
 * @tparam T Type of elements
 * @param count Number of elements
 * @param allocator Allocator to use
 * @param alignment Memory alignment
 * @param init_val Initial value for elements (optional)
 * @return Smart pointer to the allocated array
 */
template <typename T>
BREZEL_NODISCARD AllocatedPtr<T> make_allocated_array(
    size_t count,
    IAllocator& allocator = MemoryManager::instance().default_allocator(),
    Alignment alignment = Alignment::Default, const T& init_val = T()) {
    if (count == 0)
        return AllocatedPtr<T>();

    const size_t total_size = sizeof(T) * count;
    void* memory = allocator.allocate(total_size, alignment);
    T* array = static_cast<T*>(memory);

    if constexpr (std::is_trivially_constructible_v<T>) {
        T default_val = T();
        if (std::memcmp(&init_val, &default_val, sizeof(T)) == 0) {
            std::memset(array, 0, total_size);
        } else {
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(&array[i], &init_val, sizeof(T));
            }
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            new (&array[i]) T(init_val);
        }
    }

    return AllocatedPtr<T>(array, &allocator, total_size);
}

}  // namespace brezel::tensor::memory