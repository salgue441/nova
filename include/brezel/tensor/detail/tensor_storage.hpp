#pragma once

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/memory_manager.hpp>
#include <memory>

namespace brezel::tensor::detail {
/**
 * @brief A reference-counted storage class for tensor data
 *
 * @tparam T The scalar type stored in the tensor, must satisfy TensorScalar
 * concept
 *
 * This class manages the memory storage for tensor data with these key
 * features:
 * - Reference counting for shared ownership
 * - Custom memory allocation through allocator interface
 * - Cache-aligned memory allocation for optimal performance
 * - Proper handling of trivial and non-trivial types
 * - Thread-safe reference counting operations
 *
 * The storage provides:
 * - Automatic memory management
 * - Zero-initialization for trivial types
 * - Proper construction/destruction for non-trivial types
 * - Direct data access through raw pointers
 * - Memory alignment to L1 cache boundaries
 *
 * @note This class is not copyable or movable to ensure proper reference
 * counting
 * @note Memory is aligned to L1 cache line boundaries for optimal performance
 *
 * @throws std::bad_alloc If memory allocation fails during construction
 */
template <TensorScalar T>
class TensorStorage {
public:
    using allocator_type = memory::IAllocator&;

    /**
     * @brief Defaul constructor, creates empty storage
     */
    TensorStorage() noexcept
        : m_data(nullptr),
          m_size(0),
          m_allocator(&memory::MemoryManager::instance().default_allocator()),
          m_ref_count(1) {}

    /**
     * @brief Constructs a new TensorStorage object with specified size and
     * allocator
     *
     * This constructor allocates memory for storing tensor elements and
     * initializes them. For trivially default-constructible types, it
     * zero-initializes the memory. For non-trivial types, it performs in-place
     * construction of each element.
     *
     * @param size Number of elements to allocate space for
     * @param allocator Memory allocator to use for storage allocation (defaults
     * to global memory manager's default allocator)
     *
     * @throws std::bad_alloc If memory allocation fails
     *
     * @note The storage is aligned to L1 cache line boundaries for optimal
     * performance
     */
    explicit TensorStorage(
        size_t size, allocator_type allocator =
                         memory::MemoryManager::instance().default_allocator())
        : m_size(size), m_allocator(&allocator), m_ref_count(1) {
        if (size > 0) {
            m_data = static_cast<T*>(m_allocator->allocate(
                size * sizeof(T), memory::Alignment::L1Cache));

            if constexpr (std::is_trivially_default_constructible_v<T>) {
                std::memset(m_data, 0, size * sizeof(T));
            } else {
                for (size_t i = 0; i < size; ++i) {
                    new (m_data + i) T();
                }
            }
        }
    }

    /**
     * @brief Constructs a tensor storage with given size and initial value
     *
     * This constructor creates a tensor storage object that allocates memory
     * for the specified number of elements and initializes them with the
     * provided value. The allocation is done using the provided allocator or
     * the default allocator from the MemoryManager if none is specified.
     *
     * @tparam T The type of elements stored in the tensor
     * @param size The number of elements to allocate
     * @param value The value to initialize all elements with
     * @param allocator The allocator to use for memory management (defaults to
     * MemoryManager's default allocator)
     *
     * @note For trivial types, direct assignment is used for initialization.
     *       For non-trivial types, placement new is used to properly construct
     *       objects.
     * @throws May throw if memory allocation fails or if T's copy constructor
     * throws
     */
    TensorStorage(size_t size, const T& value,
                  allocator_type allocator =
                      memory::MemoryManager::instance().default_allocator())
        : m_size(size), m_allocator(&allocator), m_ref_count(1) {
        if (size > 0) {
            m_data = static_cast<T*>(m_allocator->allocate(
                size * sizeof(T), memory::Alignment::L1Cache));

            if constexpr (std::is_trivially_copyable_v<T> &&
                          std::is_trivial_v<T>) {
                for (size_t i = 0; i < size; ++i) {
                    m_data[i] = value;
                }
            } else {
                for (size_t i = 0; i < size; ++i) {
                    new (m_data + i) T(value);
                }
            }
        }
    }

    /**
     * @brief Destructor for TensorStorage class
     *
     * Properly cleans up allocated memory by:
     * 1. Calling destructors for non-trivially destructible types
     * 2. Deallocating the memory using the allocator
     *
     * The destructor performs cleanup only if m_data pointer is not null.
     * For trivially destructible types, individual destructor calls are skipped
     * for better performance.
     */
    ~TensorStorage() {
        if (m_data) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = 0; i < size; ++i) {
                    m_data[i].~T();
                }
            }

            m_allocator->deallocate(m_data, m_size * sizeof(T));
        }
    }

    // Reference counting
    /**
     * @brief Increments the reference count of the tensor storage in a
     * thread-safe manner
     *
     * This method atomically increases the internal reference counter by 1
     * using relaxed memory ordering. The operation is marked as force inline
     * for performance optimization.
     *
     * @note This operation is lock-free and thread-safe
     */
    BREZEL_FORCE_INLINE void increment_ref() noexcept {
        m_ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Decrements the reference count of the tensor storage in a
     * thread-safe manner
     * @return true if it decrements, false otherwise
     * @note This method is thread-safe as it uses atomic operations.
     */
    BREZEL_FORCE_INLINE bool decrement_ref() noexcept {
        return m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

    /**
     * @brief Returns the current reference count of the tensor storage.
     * @return The number of references to this storage as a 64-bit integer.
     * @note This method is thread-safe as it uses atomic operations.
     */
    BREZEL_FORCE_INLINE int64_t ref_count() const noexcept {
        return m_ref_count.load(std::memory_order_relaxed);
    }

    // Data access
    /**
     * @brief Returns a pointer to the raw data storage of the tensor.
     * @return T* Non-null pointer to the underlying data array.
     */
    BREZEL_NODISCARD inline T* data() noexcept { return m_data; }

    /**
     * @brief Returns a pointer to the raw data storage of the tensor.
     * @return T* Non-null pointer to the underlying data array.
     */
    BREZEL_NODISCARD inline const T* data() const noexcept { return m_data; }

    /**
     * @brief Returns the size of the tensor storage
     * @return The size of the storage
     */
    BREZEL_NODISCARD inline size_t size() const noexcept { return m_size; }

private:
    T* m_data;
    size_t m_size;
    allocator_type m_allocator;
    std::atomic<int64_t> m_ref_count;

    // Prevent copying and moving
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;
    TensorStorage(TensorStorage&&) = delete;
    TensorStorage& operator=(TensorStorage&&) = delete;
};
}  // namespace brezel::tensor::detail