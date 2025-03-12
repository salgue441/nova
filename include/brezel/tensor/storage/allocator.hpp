#pragma once

#include <tbb/cache_aligned_allocator.h>

#include <boost/align/aligned_allocator.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>

namespace brezel::tensor::storage {

/**
 * @brief Memory alignment constants
 */
struct BREZEL_API AlignmentConstants {
    /// Default alignment for tensor data (64 bytes for most cache lines)
    static constexpr size_t kDefaultAlignment = BREZEL_CACHE_LINE_SIZE;

/// Alignment for SIMD operations (typically 16, 32, or 64 bytes)
#if defined(BREZEL_SIMD_AVX512F)
    static constexpr size_t kSimdAlignment = 64;
#elif defined(BREZEL_SIMD_AVX2) || defined(BREZEL_SIMD_AVX)
    static constexpr size_t kSimdAlignment = 32;
#elif defined(BREZEL_SIMD_SSE4_2) || defined(BREZEL_SIMD_SSE4_1) || \
    defined(BREZEL_SIMD_SSE3) || defined(BREZEL_SIMD_SSE2)
    static constexpr size_t kSimdAlignment = 16;
#else
    static constexpr size_t kSimdAlignment = kDefaultAlignment;
#endif

    /// Maximum alignment supported
    static constexpr size_t kMaxAlignment = 128;
};

/**
 * @brief Base allocator interface for tensor data
 *
 * @details Defines the interface for memory allocation and deallocation
 * used by tensor storage. This class provides an abstraction over different
 * allocation strategies.
 */
class BREZEL_API AllocatorInterface {
public:
    virtual ~AllocatorInterface() = default;

    /**
     * @brief Allocate memory
     *
     * @param size Size in bytes to allocate
     * @return void* Pointer to allocated memory
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocate memory
     *
     * @param ptr Pointer to memory
     * @param size Size of the allocation
     */
    virtual void deallocate(void* ptr, size_t size) noexcept = 0;

    /**
     * @brief Get the alignment of the allocator
     *
     * @return size_t Alignment in bytes
     */
    virtual size_t alignment() const noexcept = 0;

    /**
     * @brief Clone the allocator
     *
     * @return std::unique_ptr<AllocatorInterface> New allocator instance
     */
    virtual std::unique_ptr<AllocatorInterface> clone() const = 0;

protected:
    AllocatorInterface() = default;
    AllocatorInterface(const AllocatorInterface&) = default;
    AllocatorInterface& operator=(const AllocatorInterface&) = default;
    AllocatorInterface(AllocatorInterface&&) noexcept = default;
    AllocatorInterface& operator=(AllocatorInterface&&) noexcept = default;
};

/**
 * @brief Standard aligned allocator
 *
 * @details Uses aligned memory allocation for tensor data with a specified
 * alignment boundary. This allocator is designed for general-purpose use.
 */
class BREZEL_API StandardAllocator : public AllocatorInterface {
public:
    /**
     * @brief Construct with a specific alignment
     *
     * @param alignment Alignment boundary in bytes
     */
    explicit StandardAllocator(
        size_t alignment = AlignmentConstants::kDefaultAlignment)
        : m_alignment(alignment) {}

    /**
     * @brief Allocate aligned memory
     *
     * @param size Size in bytes to allocate
     * @return void* Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    void* allocate(size_t size) override {
        if (size == 0) {
            return nullptr;
        }

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(size, m_alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        int result = posix_memalign(&ptr, m_alignment, size);
        if (result != 0) {
            throw std::bad_alloc();
        }
#endif

        return ptr;
    }

    /**
     * @brief Deallocate aligned memory
     *
     * @param ptr Pointer to memory
     * @param size Size of the allocation (unused)
     */
    void deallocate(void* ptr, size_t size) noexcept override {
        if (!ptr) {
            return;
        }

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    /**
     * @brief Get the alignment
     *
     * @return size_t Alignment in bytes
     */
    size_t alignment() const noexcept override { return m_alignment; }

    /**
     * @brief Clone the allocator
     *
     * @return std::unique_ptr<AllocatorInterface> New allocator instance
     */
    std::unique_ptr<AllocatorInterface> clone() const override {
        return std::make_unique<StandardAllocator>(m_alignment);
    }

private:
    size_t m_alignment;
};

/**
 * @brief Cache-aligned allocator using TBB
 *
 * @details Wraps TBB's cache_aligned_allocator, which is optimized for
 * thread-safe allocations aligned to cache lines for better performance
 * in parallel processing.
 */
class BREZEL_API CacheAlignedAllocator : public AllocatorInterface {
public:
    /**
     * @brief Default constructor
     */
    CacheAlignedAllocator() = default;

    /**
     * @brief Allocate cache-aligned memory
     *
     * @param size Size in bytes to allocate
     * @return void* Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    void* allocate(size_t size) override {
        if (size == 0) {
            return nullptr;
        }

        // TBB allocator allocates in terms of bytes
        char* ptr = m_allocator.allocate(size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<void*>(ptr);
    }

    /**
     * @brief Deallocate cache-aligned memory
     *
     * @param ptr Pointer to memory
     * @param size Size of the allocation
     */
    void deallocate(void* ptr, size_t size) noexcept override {
        if (!ptr) {
            return;
        }

        m_allocator.deallocate(static_cast<char*>(ptr), size);
    }

    /**
     * @brief Get the alignment
     *
     * @return size_t Alignment in bytes
     */
    size_t alignment() const noexcept override {
        // TBB cache line size is typically 128 bytes
        return BREZEL_CACHE_LINE_SIZE;
    }

    /**
     * @brief Clone the allocator
     *
     * @return std::unique_ptr<AllocatorInterface> New allocator instance
     */
    std::unique_ptr<AllocatorInterface> clone() const override {
        return std::make_unique<CacheAlignedAllocator>();
    }

private:
    tbb::cache_aligned_allocator<char> m_allocator;
};

/**
 * @brief SIMD-optimized allocator
 *
 * @details Allocator that guarantees alignment suitable for SIMD operations
 * on the target architecture. This is particularly important for vectorized
 * operations on tensor data.
 */
class BREZEL_API SimdAllocator : public AllocatorInterface {
public:
    /**
     * @brief Default constructor
     */
    SimdAllocator() : m_alignment(AlignmentConstants::kSimdAlignment) {}

    /**
     * @brief Allocate SIMD-aligned memory
     *
     * @param size Size in bytes to allocate
     * @return void* Pointer to allocated memory
     */
    void* allocate(size_t size) override {
        if (size == 0) {
            return nullptr;
        }

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(size, m_alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        int result = posix_memalign(&ptr, m_alignment, size);
        if (result != 0) {
            throw std::bad_alloc();
        }
#endif

        return ptr;
    }

    /**
     * @brief Deallocate SIMD-aligned memory
     *
     * @param ptr Pointer to memory
     * @param size Size of the allocation (unused)
     */
    void deallocate(void* ptr, size_t size) noexcept override {
        if (!ptr) {
            return;
        }

#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    /**
     * @brief Get the alignment
     *
     * @return size_t Alignment in bytes
     */
    size_t alignment() const noexcept override { return m_alignment; }

    /**
     * @brief Clone the allocator
     *
     * @return std::unique_ptr<AllocatorInterface> New allocator instance
     */
    std::unique_ptr<AllocatorInterface> clone() const override {
        return std::make_unique<SimdAllocator>();
    }

private:
    size_t m_alignment;
};

/**
 * @brief Factory for creating allocators
 */
class BREZEL_API AllocatorFactory {
public:
    /**
     * @brief Allocator types
     */
    enum class AllocatorType {
        /// Standard aligned allocator
        Standard,

        /// Cache-aligned allocator (TBB)
        CacheAligned,

        /// SIMD-optimized allocator
        Simd,

        /// Default allocator (currently CacheAligned)
        Default = CacheAligned
    };

    /**
     * @brief Create an allocator
     *
     * @param type Allocator type
     * @param alignment Alignment for standard allocator (ignored for other
     * types)
     * @return std::unique_ptr<AllocatorInterface> Allocator instance
     */
    static std::unique_ptr<AllocatorInterface> create(
        AllocatorType type = AllocatorType::Default,
        size_t alignment = AlignmentConstants::kDefaultAlignment) {
        switch (type) {
            case AllocatorType::Standard:
                return std::make_unique<StandardAllocator>(alignment);

            case AllocatorType::CacheAligned:
                return std::make_unique<CacheAlignedAllocator>();

            case AllocatorType::Simd:
                return std::make_unique<SimdAllocator>();

            default:
                return std::make_unique<CacheAlignedAllocator>();
        }
    }
};

/**
 * @brief STL-compatible typed allocator that wraps AllocatorInterface
 *
 * @tparam T Element type
 */
template <typename T>
class BREZEL_API Allocator {
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
        using other = Allocator<U>;
    };

    /**
     * @brief Default constructor
     */
    Allocator() : m_impl(AllocatorFactory::create()) {}

    /**
     * @brief Construct from allocator type
     *
     * @param type Allocator type
     * @param alignment Alignment for standard allocator
     */
    explicit Allocator(AllocatorFactory::AllocatorType type,
                       size_t alignment = AlignmentConstants::kDefaultAlignment)
        : m_impl(AllocatorFactory::create(type, alignment)) {}

    /**
     * @brief Construct from implementation
     *
     * @param impl Allocator implementation
     */
    explicit Allocator(std::unique_ptr<AllocatorInterface> impl)
        : m_impl(std::move(impl)) {}

    /**
     * @brief Copy constructor
     *
     * @param other Other allocator
     */
    Allocator(const Allocator& other)
        : m_impl(other.m_impl ? other.m_impl->clone() : nullptr) {}

    /**
     * @brief Move constructor
     *
     * @param other Other allocator
     */
    Allocator(Allocator&& other) noexcept = default;

    /**
     * @brief Copy assignment
     *
     * @param other Other allocator
     * @return Allocator& Reference to this
     */
    Allocator& operator=(const Allocator& other) {
        if (this != &other) {
            m_impl = other.m_impl ? other.m_impl->clone() : nullptr;
        }
        return *this;
    }

    /**
     * @brief Move assignment
     *
     * @param other Other allocator
     * @return Allocator& Reference to this
     */
    Allocator& operator=(Allocator&& other) noexcept = default;

    /**
     * @brief Rebinding constructor
     *
     * @tparam U Other element type
     * @param other Other allocator
     */
    template <typename U>
    Allocator(const Allocator<U>& other)
        : m_impl(other.get_impl() ? other.get_impl()->clone() : nullptr) {}

    /**
     * @brief Destructor
     */
    ~Allocator() = default;

    /**
     * @brief Allocate memory for n elements
     *
     * @param n Number of elements
     * @return pointer Pointer to allocated memory
     */
    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        if (!m_impl) {
            throw core::error::RuntimeError("Allocator implementation is null");
        }

        return static_cast<pointer>(m_impl->allocate(n * sizeof(T)));
    }

    /**
     * @brief Deallocate memory
     *
     * @param p Pointer to memory
     * @param n Number of elements
     */
    void deallocate(pointer p, size_type n) noexcept {
        if (!p || !m_impl) {
            return;
        }

        m_impl->deallocate(p, n * sizeof(T));
    }

    /**
     * @brief Get the underlying implementation
     *
     * @return const AllocatorInterface* Pointer to implementation
     */
    const AllocatorInterface* get_impl() const noexcept { return m_impl.get(); }

    /**
     * @brief Get the alignment
     *
     * @return size_t Alignment in bytes
     */
    size_t alignment() const noexcept {
        return m_impl ? m_impl->alignment()
                      : AlignmentConstants::kDefaultAlignment;
    }

    /**
     * @brief Equality comparison
     *
     * @param other Other allocator
     * @return bool True if equal
     */
    bool operator==(const Allocator& other) const noexcept {
        if (!m_impl && !other.m_impl) {
            return true;
        }

        if (!m_impl || !other.m_impl) {
            return false;
        }

        return m_impl->alignment() == other.m_impl->alignment();
    }

    /**
     * @brief Inequality comparison
     *
     * @param other Other allocator
     * @return bool True if not equal
     */
    bool operator!=(const Allocator& other) const noexcept {
        return !(*this == other);
    }

private:
    std::unique_ptr<AllocatorInterface> m_impl;

    template <typename U>
    friend class Allocator;
};

}  // namespace brezel::tensor::storage