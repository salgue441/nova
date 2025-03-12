#pragma once

#include <algorithm>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/storage/allocator.hpp>
#include <brezel/tensor/storage/memory_pool.hpp>
#include <brezel/tensor/utils/type_traits.hpp>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace brezel::tensor::storage {
/**
 * @brief Storage flags for customizing storage behavior
 */
enum class StorageFlags : uint32_t {
    /// No special flags
    None = 0,

    /// Storage is read-only and cannot be modified
    ReadOnly = 1 << 0,

    /// Storage is pinned in memory (cannot be swapped out)
    Pinned = 1 << 1,

    /// Storage is managed by the memory pool
    Pooled = 1 << 2,

    /// Storage is aligned for SIMD operations
    SimdAligned = 1 << 3,

    /// Storage is uninitialized (elements are not default constructed)
    Uninitialized = 1 << 4,

    /// Default flags (pooled, SIMD-aligned)
    Default = Pooled | SimdAligned
};

/**
 * @brief Combine storage flags
 *
 * @param lhs First flags
 * @param rhs Second flags
 * @return StorageFlags Combined flags
 */
inline StorageFlags operator|(StorageFlags lhs, StorageFlags rhs) {
    return static_cast<StorageFlags>(static_cast<uint32_t>(lhs) |
                                     static_cast<uint32_t>(rhs));
}

/**
 * @brief Check if flags contain a specific flag
 *
 * @param flags Combined flags
 * @param flag Flag to check
 * @return bool True if flag is set
 */
inline bool has_flag(StorageFlags flags, StorageFlags flag) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
}

/**
 * @brief Base storage class for tensor data
 *
 * @details This class manages memory allocation and access for tensor data. It
 * provides a type-erased interface and can be specialized for different data
 * types
 */
class BREZEL_API StorageBase {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~StorageBase() = default;

    /**
     * @brief Get the size in bytes
     *
     * @return size_t Size in bytes
     */
    virtual size_t size_bytes() const noexcept = 0;

    /**
     * @brief Get the number of elements
     *
     * @return size_t Number of elements
     */
    virtual size_t numel() const noexcept = 0;

    /**
     * @brief Get the element size in bytes
     *
     * @return size_t Element size
     */
    virtual size_t element_size() const noexcept = 0;

    /**
     * @brief Get raw pointer to data
     *
     * @return void* Pointer to data
     */
    virtual void* data() noexcept = 0;

    /**
     * @brief Get raw pointer to data (const)
     *
     * @return const void* Pointer to data
     */
    virtual const void* data() const noexcept = 0;

    /**
     * @brief Check if storage is empty
     *
     * @return bool True if empty
     */
    virtual bool empty() const noexcept = 0;

    /**
     * @brief Get storage flags
     *
     * @return StorageFlags Flags
     */
    virtual StorageFlags flags() const noexcept = 0;

    /**
     * @brief Check if storage is contiguous
     *
     * @return bool True if contiguous
     */
    virtual bool is_contiguous() const noexcept = 0;

    /**
     * @brief Get a copy of the storage
     *
     * @return std::unique_ptr<StorageBase> Copy
     */
    virtual std::unique_ptr<StorageBase> clone() const = 0;

    /**
     * @brief Resize the storage
     *
     * @param new_size New size in elements
     */
    virtual void resize(size_t new_size) = 0;

    /**
     * @brief Fill the storage with a value
     *
     * @param value Byte pattern to fill with
     */
    virtual void fill(uint8_t value) = 0;

protected:
    StorageBase() = default;
    StorageBase(const StorageBase&) = default;
    StorageBase& operator=(const StorageBase&) = default;
    StorageBase(StorageBase&&) noexcept = default;
    StorageBase& operator=(StorageBase&&) noexcept = default;
};

/**
 * @brief Typed storage implementation for tensor data
 *
 * @tparam T Element type
 */
template <typename T>
class BREZEL_API Storage : public StorageBase {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    /**
     * @brief Create an empty storage
     *
     * @param flags Storage flags
     */
    explicit Storage(StorageFlags flags = StorageFlags::Default)
        : m_size(0), m_data(nullptr), m_flags(flags), m_owns_data(false) {}

    /**
     * @brief Create storage with a specific size
     *
     * @param size Number of elements
     * @param flags Storage flags
     */
    explicit Storage(size_t size, StorageFlags flags = StorageFlags::Default)
        : m_size(size), m_flags(flags), m_owns_data(true) {
        if (size == 0) {
            m_data = nullptr;
            return;
        }

        if (has_flag(flags, StorageFlags::Pooled)) {
            m_data = MemoryPool::instance().allocate_typed<T>(size);
        } else {
            AllocatorFactory::AllocatorType alloc_type =
                has_flag(flags, StorageFlags::SimdAligned)
                    ? AllocatorFactory::AllocatorType::Simd
                    : AllocatorFactory::AllocatorType::CacheAligned;

            Allocator<T> allocator(alloc_type);
            m_data = allocator.allocate(size);
        }

        if (!has_flag(flags, StorageFlags::Uninitialized)) {
            std::uninitialized_default_construct_n(m_data, size);
        }
    }

    /**
     * @brief Create storage with a specific size and initialize with a value
     *
     * @param size Number of elements
     * @param value Value to initialize with
     * @param flags Storage flags
     */
    Storage(size_t size, const T& value,
            StorageFlags flags = StorageFlags::Default)
        : Storage(size, flags) {
        if (size > 0) {
            std::uninitialized_fill_n(m_data, size, value);
        }
    }

    /**
     * @brief Create storage with a range of values
     *
     * @tparam InputIt Input iterator type
     * @param first Iterator to first element
     * @param last Iterator to one past last element
     * @param flags Storage flags
     */
    template <typename InputIt>
    Storage(InputIt first, InputIt last,
            StorageFlags flags = StorageFlags::Default)
        : Storage(std::distance(first, last), flags) {
        if (m_size > 0) {
            std::uninitialized_copy(first, last, m_data);
        }
    }

    /**
     * @brief Create storage from initializer list
     *
     * @param init Initializer list
     * @param flags Storage flags
     */
    Storage(std::initializer_list<T> init,
            StorageFlags flags = StorageFlags::Default)
        : Storage(init.begin(), init.end(), flags) {}

    /**
     * @brief Create storage that wraps external data (does not take ownership)
     *
     * @param data Pointer to external data
     * @param size Number of elements
     * @param flags Storage flags
     */
    Storage(pointer data, size_t size,
            StorageFlags flags = StorageFlags::Default)
        : m_size(size), m_data(data), m_flags(flags), m_owns_data(false) {}

    /**
     * @brief Copy constructor
     *
     * @param other Other storage
     */
    Storage(const Storage& other)
        : m_size(other.m_size), m_flags(other.m_flags), m_owns_data(true) {
        if (other.m_size == 0) {
            m_data = nullptr;
            return;
        }

        if (has_flag(m_flags, StorageFlags::Pooled)) {
            m_data = MemoryPool::instance().allocate_typed<T>(m_size);
        } else {
            AllocatorFactory::AllocatorType alloc_type =
                has_flag(m_flags, StorageFlags::SimdAligned)
                    ? AllocatorFactory::AllocatorType::Simd
                    : AllocatorFactory::AllocatorType::CacheAligned;

            Allocator<T> allocator(alloc_type);
            m_data = allocator.allocate(m_size);
        }

        if (m_size > 0) {
            std::uninitialized_copy_n(other.m_data, m_size, m_data);
        }
    }

    /**
     * @brief Move constructor
     *
     * @param other Other storage
     */
    Storage(Storage&& other) noexcept
        : m_size(other.m_size),
          m_data(other.m_data),
          m_flags(other.m_flags),
          m_owns_data(other.m_owns_data) {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_owns_data = false;
    }

    /**
     * @brief Copy assignment
     *
     * @param other Other storage
     * @return Storage& Reference to this
     */
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            if (m_owns_data && m_data) {
                destroy_and_deallocate();
            }

            m_size = other.m_size;
            m_flags = other.m_flags;
            m_owns_data = true;

            if (other.m_size == 0) {
                m_data = nullptr;
                return *this;
            }

            if (has_flag(m_flags, StorageFlags::Pooled)) {
                m_data = MemoryPool::instance().allocate_typed<T>(m_size);
            } else {
                AllocatorFactory::AllocatorType alloc_type =
                    has_flag(m_flags, StorageFlags::SimdAligned)
                        ? AllocatorFactory::AllocatorType::Simd
                        : AllocatorFactory::AllocatorType::CacheAligned;

                Allocator<T> allocator(alloc_type);
                m_data = allocator.allocate(m_size);
            }

            if (m_size > 0) {
                std::uninitialized_copy_n(other.m_data, m_size, m_data);
            }
        }

        return *this;
    }

    /**
     * @brief Move assignment
     *
     * @param other Other storage
     * @return Storage& Reference to this
     */
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (m_owns_data && m_data) {
                destroy_and_deallocate();
            }

            m_size = other.m_size;
            m_data = other.m_data;
            m_flags = other.m_flags;
            m_owns_data = other.m_owns_data;

            other.m_data = nullptr;
            other.m_size = 0;
            other.m_owns_data = false;
        }

        return *this;
    }

    /**
     * @brief Destructor
     */
    ~Storage() override {
        if (m_owns_data && m_data) {
            destroy_and_deallocate();
        }
    }

    // Iterator access
    /**
     * @brief Get iterator to beginning
     *
     * @return iterator
     */
    iterator begin() noexcept { return m_data; }

    /**
     * @brief Get iterator to end
     *
     * @return iterator
     */
    iterator end() noexcept { return m_data + m_size; }

    /**
     * @brief Get const iterator to beginning
     *
     * @return const_iterator
     */
    const_iterator begin() const noexcept { return m_data; }

    /**
     * @brief Get const iterator to end
     *
     * @return const_iterator
     */
    const_iterator end() const noexcept { return m_data + m_size; }

    /**
     * @brief Get const iterator to beginning
     *
     * @return const_iterator
     */
    const_iterator cbegin() const noexcept { return m_data; }

    /**
     * @brief Get const iterator to end
     *
     * @return const_iterator
     */
    const_iterator cend() const noexcept { return m_data + m_size; }

    // Element access
    /**
     * @brief Access element
     *
     * @param idx Index
     * @return reference Reference to element
     */
    reference operator[](size_t idx) noexcept { return m_data[idx]; }

    /**
     * @brief Access element (const)
     *
     * @param idx Index
     * @return const_reference Reference to element
     */
    const_reference operator[](size_t idx) const noexcept {
        return m_data[idx];
    }

    /**
     * @brief Access element with bounds checking
     *
     * @param idx Index
     * @return reference Reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    reference at(size_t idx) {
        if (idx >= m_size) {
            throw core::error::LogicError(
                "Index {} out of bounds for storage with {} elements", idx,
                m_size);
        }

        return m_data[idx];
    }

    /**
     * @brief Access element with bounds checking (const)
     *
     * @param idx Index
     * @return const_reference Reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    const_reference at(size_t idx) const {
        if (idx >= m_size) {
            throw core::error::LogicError(
                "Index {} out of bounds for storage with {} elements", idx,
                m_size);
        }

        return m_data[idx];
    }

    /**
     * @brief Get first element
     *
     * @return reference Reference to first element
     */
    reference front() noexcept { return m_data[0]; }

    /**
     * @brief Get first element (const)
     *
     * @return const_reference Reference to first element
     */
    const_reference front() const noexcept { return m_data[0]; }

    /**
     * @brief Get last element
     *
     * @return reference Reference to last element
     */
    reference back() noexcept { return m_data[m_size - 1]; }

    /**
     * @brief Get last element (const)
     *
     * @return const_reference Reference to last element
     */
    const_reference back() const noexcept { return m_data[m_size - 1]; }

    // StorageBase implementation
    /**
     * @brief Get the size in bytes
     *
     * @return size_t Size in bytes
     */
    size_t size_bytes() const noexcept override { return m_size * sizeof(T); }

    /**
     * @brief Get the number of elements
     *
     * @return size_t Number of elements
     */
    size_t numel() const noexcept override { return m_size; }

    /**
     * @brief Get the element size in bytes
     *
     * @return size_t Element size
     */
    size_t element_size() const noexcept override { return sizeof(T); }

    /**
     * @brief Get raw pointer to data
     *
     * @return void* Pointer to data
     */
    void* data() noexcept override { return static_cast<void*>(m_data); }

    /**
     * @brief Get raw pointer to data (const)
     *
     * @return const void* Pointer to data
     */
    const void* data() const noexcept override {
        return static_cast<const void*>(m_data);
    }

    /**
     * @brief Check if storage is empty
     *
     * @return bool True if empty
     */
    bool empty() const noexcept override {
        return m_size == 0 || m_data == nullptr;
    }

    /**
     * @brief Get storage flags
     *
     * @return StorageFlags Flags
     */
    StorageFlags flags() const noexcept override { return m_flags; }

    /**
     * @brief Check if storage is contiguous
     *
     * @return bool True if contiguous
     */
    bool is_contiguous() const noexcept override { return true; }

    /**
     * @brief Get a copy of the storage
     *
     * @return std::unique_ptr<StorageBase> Copy
     */
    std::unique_ptr<StorageBase> clone() const override {
        return std::make_unique<Storage>(*this);
    }

    /**
     * @brief Resize the storage
     *
     * @param new_size New size in elements
     */
    void resize(size_t new_size) override {
        if (new_size == m_size) {
            return;
        }

        if (has_flag(m_flags, StorageFlags::ReadOnly)) {
            throw core::error::LogicError("Cannot resize read-only storage");
        }

        if (m_size == 0) {
            *this = Storage(new_size, m_flags);
            return;
        }

        if (new_size == 0) {
            if (m_owns_data) {
                destroy_and_deallocate();
            }

            m_data = nullptr;
            m_size = 0;
            return;
        }

        pointer new_data = nullptr;
        if (has_flag(m_flags, StorageFlags::Pooled)) {
            new_data = MemoryPool::instance().allocate_typed<T>(new_size);
        } else {
            AllocatorFactory::AllocatorType alloc_type =
                has_flag(m_flags, StorageFlags::SimdAligned)
                    ? AllocatorFactory::AllocatorType::Simd
                    : AllocatorFactory::AllocatorType::CacheAligned;

            Allocator<T> allocator(alloc_type);
            new_data = allocator.allocate(new_size);
        }

        const size_t copy_size = std::min(m_size, new_size);
        if (copy_size > 0) {
            std::uninitialized_copy_n(m_data, copy_size, new_data);
        }

        if (new_size > m_size &&
            !has_flag(m_flags, StorageFlags::Uninitialized)) {
            std::uninitialized_default_construct_n(new_data + m_size,
                                                   new_size - m_size);
        }
        if (m_owns_data) {
            destroy_and_deallocate();
        }

        m_data = new_data;
        m_size = new_size;
        m_owns_data = true;
    }

    /**
     * @brief Fill the storage with a value
     *
     * @param value Byte pattern to fill with
     */
    void fill(uint8_t value) override {
        if (has_flag(m_flags, StorageFlags::ReadOnly)) {
            throw core::error::LogicError("Cannot fill read-only storage");
        }

        if (m_size > 0 && m_data != nullptr) {
            std::memset(m_data, value, m_size * sizeof(T));
        }
    }

    /**
     * @brief Fill with a specific value
     *
     * @param value Value to fill with
     */
    void fill(const T& value) {
        if (has_flag(m_flags, StorageFlags::ReadOnly)) {
            throw core::error::LogicError("Cannot fill read-only storage");
        }

        if (m_size > 0 && m_data != nullptr) {
            std::fill_n(m_data, m_size, value);
        }
    }

    /**
     * @brief Get raw pointer to data
     *
     * @return pointer
     */
    pointer data_ptr() noexcept { return m_data; }

    /**
     * @brief Get raw pointer to data (const)
     *
     * @return const_pointer
     */
    const_pointer data_ptr() const noexcept { return m_data; }

    /**
     * @brief Get the size
     *
     * @return size_t
     */
    size_t size() const noexcept { return m_size; }

    /**
     * @brief Set storage to read-only
     */
    void set_read_only() noexcept {
        m_flags = m_flags | StorageFlags::ReadOnly;
    }

    /**
     * @brief Check if storage is read-only
     *
     * @return bool
     */
    bool is_read_only() const noexcept {
        return has_flag(m_flags, StorageFlags::ReadOnly);
    }

    /**
     * @brief Check if storage owns its data
     *
     * @return bool
     */
    bool owns_data() const noexcept { return m_owns_data; }

private:
    /**
     * @brief Destroy elements and deallocate memory
     */
    void destroy_and_deallocate() noexcept {
        if (!m_data) {
            return;
        }

        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_n(m_data, m_size);
        }

        if (has_flag(m_flags, StorageFlags::Pooled)) {
            MemoryPool::instance().deallocate_typed(m_data, m_size);
        } else {
            AllocatorFactory::AllocatorType alloc_type =
                has_flag(m_flags, StorageFlags::SimdAligned)
                    ? AllocatorFactory::AllocatorType::Simd
                    : AllocatorFactory::AllocatorType::CacheAligned;

            Allocator<T> allocator(alloc_type);
            allocator.deallocate(m_data, m_size);
        }

        m_data = nullptr;
    }

    size_t m_size;
    pointer m_data;
    StorageFlags m_flags;
    bool m_owns_data;
};

/**
 * @brief Create a new storage object
 *
 * @tparam T Element type
 * @param size Number of elements
 * @param flags Storage flags
 * @return Storage<T> New storage
 */
template <typename T>
BREZEL_NODISCARD inline Storage<T> make_storage(
    size_t size, const T& value, StorageFlags flags = StorageFlags::Default) {
    return Storage<T>(size, value, flags);
}

/**
 * @brief Create a new storage object from a range of values
 *
 * @tparam T Element type
 * @tparam InputIt Input iterator type
 * @param first Iterator to first element
 * @param last Iterator to one past last element
 * @param flags Storage flags
 * @return Storage<T> New storage
 */
template <typename T, typename InputIt>
BREZEL_NODISCARD inline Storage<T> make_storage(
    InputIt first, InputIt last, StorageFlags flags = StorageFlags::Default) {
    return Storage<T>(first, last, flags);
}

/**
 * @brief Create a new storage object from initializer list
 *
 * @tparam T Element type
 * @param init Initializer list
 * @param flags Storage flags
 * @return Storage<T> New storage
 */
template <typename T>
BREZEL_NODISCARD inline Storage<T> make_storage(
    std::initializer_list<T> init, StorageFlags flags = StorageFlags::Default) {
    return Storage<T>(init, flags);
}

/**
 * @brief Create a storage object that wraps external data (does not take
 * ownership)
 *
 * @tparam T Element type
 * @param data Pointer to external data
 * @param size Number of elements
 * @param flags Storage flags
 * @return Storage<T> Storage wrapper
 */
template <typename T>
BREZEL_NODISCARD inline Storage<T> make_storage_external(
    T* data, size_t size, StorageFlags flags = StorageFlags::Default) {
    return Storage<T>(data, size, flags);
}
}  // namespace brezel::tensor::storage