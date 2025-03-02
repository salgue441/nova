
#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/layout.hpp>
#include <brezel/tensor/memory_manager.hpp>
#include <memory>
#include <type_traits>
#include <utility>

namespace brezel::tensor {
/**
 * @brief Concept for valid storage data types
 *
 * @tparam T The type being checked against the concept
 */
template <typename T>
concept StorageScalar = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

template <StorageScalar>
class Storage;

/**
 * @brief Type-erased storage base class
 * @details Provides a common interface for all storage types without exposing
 * the template
 */
class BREZEL_API StorageBase {
public:
    virtual ~StorageBase() = default;

    /**
     * @brief Gets the number of elements in the storage
     * @return Number of elements
     */
    BREZEL_NODISCARD virtual size_t size() const noexcept = 0;

    /**
     * @brief Gets the size of each element in bytes
     * @return Element size in bytes
     */
    BREZEL_NODISCARD virtual size_t element_size() const noexcept = 0;

    /**
     * @brief Gets the total size of the storage in bytes
     * @return Total storage size in bytes
     */
    BREZEL_NODISCARD virtual size_t nbytes() const noexcept = 0;

    /**
     * @brief Gets the raw data pointer
     * @return Void pointer to the raw data
     */
    BREZEL_NODISCARD virtual void* data() noexcept = 0;

    /**
     * @brief Gets the raw data pointer (const version)
     * @return Const void pointer to the raw data
     */
    BREZEL_NODISCARD virtual const void* data() const noexcept = 0;

    /**
     * @brief Checks if the storage is contiguous in memory
     * @return True if the storage is contiguous
     */
    BREZEL_NODISCARD virtual bool is_contiguous() const noexcept = 0;

    /**
     * @brief Returns the device type where data is stored
     * @return The device type
     */
    BREZEL_NODISCARD virtual DeviceType device() const noexcept = 0;

    /**
     * @brief Creates a clone of the storage
     * @return Unique pointer to the cloned storage
     */
    BREZEL_NODISCARD virtual std::unique_ptr<StorageBase> clone() const = 0;

    /**
     * @brief Resize the storage to the new size
     * @param new_size New number of elements
     * @param preserve_data Whether to preserve existing data (when possible)
     */
    virtual void resize(size_t new_size, bool preserve_data = true) = 0;

protected:
    StorageBase() = default;
    StorageBase(const StorageBase&) = default;
    StorageBase& operator=(const StorageBase&) = default;
    StorageBase(StorageBase&&) noexcept = default;
    StorageBase& operator=(StorageBase&&) noexcept = default;
};

/**
 * @brief Storage implementation for tensor data
 * @details Manages memory for tensor data with reference counting and
 * allocator-aware memory management.
 *
 * @tparam T Data type for storage elements
 */
template <StorageScalar T>
class BREZEL_API Storage : public StorageBase {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    /**
     * @brief Creates an empty storage
     */
    BREZEL_NODISCARD Storage() : m_size(0), m_device(DeviceType::CPU) {}

    /**
     * @brief Creates a storage with the specified size
     *
     * @param size Number of elements
     * @param allocator Allocator to use (defaults to memory manager's default)
     * @param device Device where the data is stored
     * @throws brezel::core::error::RuntimeError if allocation fails
     */
    BREZEL_NODISCARD explicit Storage(
        size_t size,
        memory::IAllocator& allocator =
            memory::MemoryManager::instance().default_allocator(),
        DeviceType device = DeviceType::CPU)
        : m_size(size), m_device(device), m_allocator(&allocator) {
        if (size == 0)
            return;

        size_t byte_size = size * sizeof(T);
        m_data =
            memory::AllocatedPtr<T>(static_cast<T*>(allocator.allocate(
                                        byte_size, memory::Alignment::L1Cache)),
                                    &allocator, byte_size);

        if (!m_data) {
            throw core::error::RuntimeError(
                "Failed to allocate storage of size {}", size);
        }
    }

    /**
     * @brief Creates a storage with specified size and fills it with a value
     *
     * @param size Number of elements
     * @param value Value to fill the storage with
     * @param allocator Allocator to use (defaults to memory manager's default)
     * @param device Device where the data is stored
     * @throws brezel::core::error::RuntimeError if allocation fails
     */
    BREZEL_NODISCARD Storage(
        size_t size, T value,
        memory::IAllocator& allocator =
            memory::MemoryManager::instance().default_allocator(),
        DeviceType device = DeviceType::CPU)
        : Storage(size, allocator, device) {
        if (size == 0) {
            return;
        }

        // Fill the storage with the value
        std::fill_n(m_data.get(), size, value);
    }

    /**
     * @brief Creates a storage from an existing data pointer
     *
     * @param data Pointer to existing data
     * @param size Number of elements
     * @param copy_data Whether to copy the data or take ownership
     * @param allocator Allocator to use (defaults to memory manager's default)
     * @param device Device where the data is stored
     * @throws brezel::core::error::RuntimeError if allocation fails and
     * copy_data is true
     */
    BREZEL_NODISCARD Storage(
        pointer data, size_t size, bool copy_data = true,
        memory::IAllocator& allocator =
            memory::MemoryManager::instance().default_allocator(),
        DeviceType device = DeviceType::CPU)
        : m_size(size), m_device(device), m_allocator(&allocator) {
        if (data == nullptr || size == 0) {
            m_size = 0;
            return;
        }

        if (copy_data) {
            size_t byte_size = size * sizeof(T);
            m_data = memory::AllocatedPtr<T>(
                static_cast<T*>(
                    allocator.allocate(byte_size, memory::Alignment::L1Cache)),
                &allocator, byte_size);

            if (!m_data) {
                throw core::error::RuntimeError(
                    "Failed to allocate storage of size {}", size);
            }

            std::memcpy(m_data.get(), data, byte_size);
        } else {
            // Take ownership without copying
            // NOTE: This is dangerous as we don't know how the memory was
            // allocated This should only be used when the memory was allocated
            // using the same allocator
            m_data =
                memory::AllocatedPtr<T>(data, &allocator, size * sizeof(T));
        }
    }

    /**
     * @brief Copy constructor
     */
    BREZEL_NODISCARD Storage(const Storage& other)
        : m_size(other.m_size),
          m_device(other.m_device),
          m_allocator(other.m_allocator) {
        if (m_size == 0) {
            return;
        }

        // Allocate memory and copy data
        size_t byte_size = m_size * sizeof(T);
        m_data =
            memory::AllocatedPtr<T>(static_cast<T*>(m_allocator->allocate(
                                        byte_size, memory::Alignment::L1Cache)),
                                    m_allocator, byte_size);

        if (!m_data) {
            throw core::error::RuntimeError(
                "Failed to allocate storage of size {}", m_size);
        }

        std::memcpy(m_data.get(), other.m_data.get(), byte_size);
    }

    /**
     * @brief Move constructor
     */
    BREZEL_NODISCARD Storage(Storage&& other) noexcept
        : m_size(other.m_size),
          m_device(other.m_device),
          m_allocator(other.m_allocator),
          m_data(std::move(other.m_data)) {
        other.m_size = 0;
    }

    /**
     * @brief Copy assignment operator
     */
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            Storage temp(other);
            *this = std::move(temp);
        }
        return *this;
    }

    /**
     * @brief Move assignment operator
     */
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            m_size = other.m_size;
            m_device = other.m_device;
            m_allocator = other.m_allocator;
            m_data = std::move(other.m_data);
            other.m_size = 0;
        }
        return *this;
    }

    /**
     * @brief Gets the number of elements in the storage
     * @return Number of elements
     */
    BREZEL_NODISCARD size_t size() const noexcept override { return m_size; }

    /**
     * @brief Gets the size of each element in bytes
     * @return Element size in bytes
     */
    BREZEL_NODISCARD size_t element_size() const noexcept override {
        return sizeof(T);
    }

    /**
     * @brief Gets the total size of the storage in bytes
     * @return Total storage size in bytes
     */
    BREZEL_NODISCARD size_t nbytes() const noexcept override {
        return m_size * sizeof(T);
    }

    /**
     * @brief Gets the raw data pointer
     * @return Void pointer to the raw data
     */
    BREZEL_NODISCARD void* data() noexcept override { return m_data.get(); }

    /**
     * @brief Gets the raw data pointer (const version)
     * @return Const void pointer to the raw data
     */
    BREZEL_NODISCARD const void* data() const noexcept override {
        return m_data.get();
    }

    /**
     * @brief Gets the typed data pointer
     * @return Typed pointer to the data
     */
    BREZEL_NODISCARD pointer data_ptr() noexcept { return m_data.get(); }

    /**
     * @brief Gets the typed data pointer (const version)
     * @return Const typed pointer to the data
     */
    BREZEL_NODISCARD const_pointer data_ptr() const noexcept {
        return m_data.get();
    }

    /**
     * @brief Checks if the storage is contiguous in memory
     * @return Always true for Storage<T>
     */
    BREZEL_NODISCARD bool is_contiguous() const noexcept override {
        return true;
    }

    /**
     * @brief Returns the device type where data is stored
     * @return The device type
     */
    BREZEL_NODISCARD DeviceType device() const noexcept override {
        return m_device;
    }

    /**
     * @brief Creates a clone of the storage
     * @return Unique pointer to the cloned storage
     */
    BREZEL_NODISCARD std::unique_ptr<StorageBase> clone() const override {
        return std::make_unique<Storage<T>>(*this);
    }

    /**
     * @brief Resizes the storage to the new size.
     *
     * This function resizes the storage to the specified new size. If the new
     * size is the same as the current size, the function returns immediately.
     * If the new size is zero, the storage is cleared. If the storage is
     * currently empty, a new storage is allocated with the specified size.
     *
     * If the allocator supports resizing, the storage is resized in place.
     * Otherwise, a new storage is allocated, and the existing data is copied to
     * the new storage if `preserve_data` is true.
     *
     * @param new_size New number of elements.
     * @param preserve_data Whether to preserve existing data (when possible).
     * Defaults to true.
     * @throws brezel::core::error::RuntimeError if allocation fails during
     * resizing.
     */
    void resize(size_t new_size, bool preserve_data = true) override {
        if (new_size == m_size) {
            return;
        }

        if (new_size == 0) {
            m_data.reset();
            m_size = 0;

            return;
        }

        if (m_size == 0 || !m_data) {
            *this = Storage<T>(new_size, *m_allocator, m_device);
            return;
        }

        // Check if allocator supports resize
        size_t old_bytes = m_size * sizeof(T);
        size_t new_bytes = new_size * sizeof(T);

        if (m_allocator->supports_resize()) {
            void* new_memory = m_allocator->resize(
                m_data.get(), old_bytes, new_bytes,
                preserve_data ? memory::AllocFlags::None
                              : memory::AllocFlags::ZeroMemory);

            if (new_memory) {
                m_data.reset(static_cast<T*>(new_memory), m_allocator,
                             new_bytes);
                m_size = new_size;

                return;
            }
        }

        memory::AllocatedPtr<T> new_data(
            static_cast<T*>(m_allocator->allocate(
                new_bytes, memory::Alignment::L1Cache,
                preserve_data ? memory::AllocFlags::None
                              : memory::AllocFlags::ZeroMemory)),
            m_allocator, new_bytes);

        if (!new_data) {
            throw core::error::RuntimeError("Failed to resize storage to {}",
                                            new_size);
        }

        if (preserve_data) {
            std::memcpy(new_data.get(), m_data.get(),
                        std::min(old_bytes, new_bytes));
        }

        m_data = std::move(new_data);
        m_size = new_size;
    }

    /**
     * @brief Element access operator
     *
     * @param idx Index of the element
     * @return Reference to the element
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE reference
    operator[](size_t idx) noexcept {
        return m_data.get()[idx];
    }

    /**
     * @brief Element access operator (const version)
     *
     * @param idx Index of the element
     * @return Const reference to the element
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const_reference
    operator[](size_t idx) const noexcept {
        return m_data.get()[idx];
    }

    /**
     * @brief Bounds-checked element access
     *
     * @param idx Index of the element
     * @return Reference to the element
     * @throws brezel::core::error::LogicError if index is out of bounds
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE reference at(size_t idx) {
        if (idx >= m_size) {
            throw brezel::core::error::LogicError(
                "Index {} out of bounds for storage size {}", idx, m_size);
        }

        return m_data.get()[idx];
    }

    /**
     * @brief Bounds-checked element access (const version)
     *
     * @param idx Index of the element
     * @return Reference to the element
     * @throws brezel::core::error::LogicError if index is out of bounds
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const_reference at(size_t idx) const {
        if (idx >= m_size) {
            throw brezel::core::error::LogicError(
                "Index {} out of bounds for storage size {}", idx, m_size);
        }

        return m_data.get()[idx];
    }

    /**
     * @brief Checks if storage is empty
     *
     * @return True if storage is empty
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool empty() const noexcept {
        return m_size == 0 || !m_data;
    }

    /**
     * @brief Fills the storage with a value
     *
     * @param value Value to fill the storage with
     */
    BREZEL_FORCE_INLINE void fill(T value) {
        if (empty()) {
            return;
        }

        std::fill_n(m_data.get(), m_size, value);
    }

    /**
     * @brief Get the allocator used by this storage
     *
     * @return Pointer to the allocator
     */
    BREZEL_NODISCARD memory::IAllocator* allocator() const noexcept {
        return m_allocator;
    }

private:
    size_t m_size = 0;
    DeviceType m_device = DeviceType::CPU;
    memory::IAllocator* m_allocator =
        &memory::MemoryManager::instance().default_allocator();
    memory::AllocatedPtr<T> m_data;
};

/**
 * @brief Shared storage class with reference counting
 * @details Provides shared ownership semantics for tensor storage
 */
class BREZEL_API SharedStorage {
public:
    /**
     * @brief Creates an empty shared storage
     */
    BREZEL_NODISCARD SharedStorage()
        : m_storage(nullptr), m_ref_count(nullptr) {}

    /**
     * @brief Creates a shared storage from a unique storage pointer
     *
     * @param storage Unique pointer to storage
     */
    BREZEL_NODISCARD explicit SharedStorage(
        std::unique_ptr<StorageBase>&& storage)
        : m_storage(storage.release()), m_ref_count(new std::atomic<int>(1)) {}

    /**
     * @brief Copy constructor
     */
    BREZEL_NODISCARD SharedStorage(const SharedStorage& other)
        : m_storage(other.m_storage), m_ref_count(other.m_ref_count) {
        if (m_ref_count) {
            m_ref_count->fetch_add(1, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Move constructor
     */
    BREZEL_NODISCARD SharedStorage(SharedStorage&& other) noexcept
        : m_storage(other.m_storage), m_ref_count(other.m_ref_count) {
        other.m_storage = nullptr;
        other.m_ref_count = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~SharedStorage() {
        if (m_ref_count &&
            m_ref_count->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete m_storage;
            delete m_ref_count;
        }
    }

    /**
     * @brief Copy assignment operator
     */
    SharedStorage& operator=(const SharedStorage& other) {
        if (this != &other) {
            if (other.m_ref_count) {
                other.m_ref_count->fetch_add(1, std::memory_order_relaxed);
            }

            if (m_ref_count &&
                m_ref_count->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete m_storage;
                delete m_ref_count;
            }

            m_storage = other.m_storage;
            m_ref_count = other.m_ref_count;
        }

        return *this;
    }

    /**
     * @brief Assignment operator
     */
    SharedStorage& operator=(SharedStorage&& other) noexcept {
        if (this != &other) {
            if (m_ref_count &&
                m_ref_count->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete m_storage;
                delete m_ref_count;
            }

            m_storage = other.m_storage;
            m_ref_count = other.m_ref_count;
            other.m_storage = nullptr;
            other.m_ref_count = nullptr;
        }

        return *this;
    }

    /**
     * @brief Gets the underlying storage pointer
     *
     * @return Pointer to the storage
     */
    BREZEL_NODISCARD StorageBase* get() const noexcept { return m_storage; }

    /**
     * @brief Gets the underlying storage pointer for the specified type
     *
     * @tparam T Storage element type
     * @return Typed pointer to the storage
     * @throws brezel::core::error::LogicError if storage types doesn't match
     */
    template <typename T>
    BREZEL_NODISCARD Storage<T>* get() const {
        if (!m_storage) {
            return nullptr;
        }

        if (m_storage->element_size() != sizeof(T)) {
            throw core::error::LogicError(
                "Storage type mismatch: requested element size {} but storage "
                "has element size {}",
                sizeof(T), m_storage->element_size());
        }

        return static_cast<Storage<T>*>(m_storage);
    }

    /**
     * @brief Checks if the storage is empty
     *
     * @return True if the storage is empty or null
     */
    BREZEL_NODISCARD bool empty() const noexcept {
        return !m_storage || m_storage->size() == 0;
    }

    /**
     * @brief Gets the number of elements in the storage
     *
     * @return Number of elements
     */
    BREZEL_NODISCARD size_t size() const noexcept {
        return m_storage ? m_storage->size() : 0;
    }

    /**
     * @brief Gets the total size of the storage in bytes
     *
     * @return Total storage size in bytes
     */
    BREZEL_NODISCARD size_t nbytes() const noexcept {
        return m_storage ? m_storage->nbytes() : 0;
    }

    /**
     * @brief Gets the size of each element in bytes
     *
     * @return Element size in bytes
     */
    BREZEL_NODISCARD size_t element_size() const noexcept {
        return m_storage ? m_storage->element_size() : 0;
    }

    /**
     * @brief Gets the raw data pointer
     *
     * @return Void pointer to the raw data
     */
    BREZEL_NODISCARD void* data() noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Gets the raw data pointer (const version)
     *
     * @return Const void pointer to the raw data
     */
    BREZEL_NODISCARD const void* data() const noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Checks if the storage is contiguous in memory
     *
     * @return True if the storage is contiguous
     */
    BREZEL_NODISCARD bool is_contiguous() const noexcept {
        return m_storage ? m_storage->is_contiguous() : true;
    }

    /**
     * @brief Returns the device type where data is stored
     *
     * @return The device type
     */
    BREZEL_NODISCARD DeviceType device() const noexcept {
        return m_storage ? m_storage->device() : DeviceType::CPU;
    }

    /**
     * @brief Creates a copy of the storage
     *
     * @return New SharedStorage with a copy of the data
     */
    BREZEL_NODISCARD SharedStorage clone() const {
        if (!m_storage) {
            return SharedStorage();
        }

        return SharedStorage(m_storage->clone());
    }

    /**
     * @brief Gets the current reference count
     *
     * @return Current reference count
     */
    BREZEL_NODISCARD int use_count() const noexcept {
        return m_ref_count ? m_ref_count->load(std::memory_order_relaxed) : 0;
    }

    /**
     * @brief Checks if storage is unique (reference count is 1)
     *
     * @return True if storage is unique
     */
    BREZEL_NODISCARD bool unique() const noexcept { return use_count() == 1; }

    /**
     * @brief Boolean conversion operator
     *
     * @return True if storage is non-empty
     */
    BREZEL_NODISCARD explicit operator bool() const noexcept {
        return m_storage != nullptr;
    }

    /**
     * @brief Resize the storage to the new size
     *
     * @param new_size New number of elements
     * @param preserve_data Whether to preserve existing data (when possible)
     */
    void resize(size_t new_size, bool preserve_data = true) {
        if (!m_storage) {
            return;
        }

        m_storage->resize(new_size, preserve_data);
    }

private:
    StorageBase* m_storage = nullptr;
    std::atomic<int>* m_ref_count = nullptr;
};

/**
 * @brief Creates a typed shared storage
 *
 * @tparam T Storage element type
 * @param size Number of elements
 * @param allocator Allocator to use
 * @param device Device where the data is stored
 * @return SharedStorage instance
 */
template <StorageScalar T>
BREZEL_NODISCARD SharedStorage
make_storage(size_t size,
             memory::IAllocator& allocator =
                 memory::MemoryManager::instance().default_allocator(),
             DeviceType device = DeviceType::CPU) {
    return SharedStorage(std::make_unique<Storage<T>>(size, allocator, device));
}

/**
 * @brief Creates a typed shared storage filled with a value
 *
 * @tparam T Storage element type
 * @param size Number of elements
 * @param value Value to fill the storage with
 * @param allocator Allocator to use
 * @param device Device where the data is stored
 * @return SharedStorage instance
 */
template <StorageScalar T>
BREZEL_NODISCARD SharedStorage
make_storage(size_t size, T value,
             memory::IAllocator& allocator =
                 memory::MemoryManager::instance().default_allocator(),
             DeviceType device = DeviceType::CPU) {
    return SharedStorage(
        std::make_unique<Storage<T>>(size, value, allocator, device));
}

/**
 * @brief Creates a typed shared storage from existing data
 *
 * @tparam T Storage element type
 * @param data Pointer to existing data
 * @param size Number of elements
 * @param copy_data Whether to copy the data or take ownership
 * @param allocator Allocator to use
 * @param device Device where the data is stored
 * @return SharedStorage instance
 */
template <StorageScalar T>
BREZEL_NODISCARD SharedStorage
make_storage(T* data, size_t size, bool copy_data = true,
             memory::IAllocator& allocator =
                 memory::MemoryManager::instance().default_allocator(),
             DeviceType device = DeviceType::CPU) {
    return SharedStorage(
        std::make_unique<Storage<T>>(data, size, copy_data, allocator, device));
}

/**
 * @brief Creates a typed shared storage with zeros
 *
 * @tparam T Storage element type
 * @param size Number of elements
 * @param allocator Allocator to use
 * @param device Device where the data is stored
 * @return SharedStorage instance
 */
template <StorageScalar T>
BREZEL_NODISCARD SharedStorage
make_zeros_storage(size_t size,
                   memory::IAllocator& allocator =
                       memory::MemoryManager::instance().default_allocator(),
                   DeviceType device = DeviceType::CPU) {
    return SharedStorage(
        std::make_unique<Storage<T>>(size, T(0), allocator, device));
}

/**
 * @brief Creates a typed shared storage with ones
 *
 * @tparam T Storage element type
 * @param size Number of elements
 * @param allocator Allocator to use
 * @param device Device where the data is stored
 * @return SharedStorage instance
 */
template <StorageScalar T>
BREZEL_NODISCARD SharedStorage
make_ones_storage(size_t size,
                  memory::IAllocator& allocator =
                      memory::MemoryManager::instance().default_allocator(),
                  DeviceType device = DeviceType::CPU) {
    return SharedStorage(
        std::make_unique<Storage<T>>(size, T(1), allocator, device));
}
}  // namespace brezel::tensor