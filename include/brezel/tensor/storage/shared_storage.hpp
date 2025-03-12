#pragma once

#include <atomic>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/storage/storage.hpp>
#include <brezel/tensor/utils/type_traits.hpp>
#include <memory>
#include <type_traits>
#include <typeindex>

namespace brezel::tensor::storage {

/**
 * @brief Reference-counted shared storage wrapper
 *
 * @details Provides shared ownership semantics for tensor storage with
 * automatic memory management and copy-on-write optimization. Multiple tensors
 * can share the same storage without unnecessary copies, and modifications are
 * automatically isolated when needed.
 */
class BREZEL_API SharedStorageBase {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~SharedStorageBase() = default;

    /**
     * @brief Check if storage is shared with others
     *
     * @return bool True if shared (reference count > 1)
     */
    virtual bool is_shared() const noexcept = 0;

    /**
     * @brief Get reference count
     *
     * @return long Number of references
     */
    virtual long use_count() const noexcept = 0;

    /**
     * @brief Make a unique (non-shared) copy if needed
     *
     * @return SharedStorageBase& Reference to potentially new storage
     */
    virtual SharedStorageBase& unshare() = 0;

    /**
     * @brief Get raw pointer to underlying storage
     *
     * @return StorageBase* Pointer to storage
     */
    virtual StorageBase* get() noexcept = 0;

    /**
     * @brief Get raw pointer to underlying storage (const)
     *
     * @return const StorageBase* Pointer to storage
     */
    virtual const StorageBase* get() const noexcept = 0;

    /**
     * @brief Clone the shared storage
     *
     * @return std::unique_ptr<SharedStorageBase> Clone
     */
    virtual std::unique_ptr<SharedStorageBase> clone() const = 0;

    /**
     * @brief Create a type-erased copy of this storage
     *
     * @return std::unique_ptr<SharedStorageBase> Type-erased copy
     */
    virtual std::unique_ptr<SharedStorageBase> type_erased_copy() const = 0;

    /**
     * @brief Get the element type information
     *
     * @return std::type_index Type index
     */
    virtual std::type_index type_index() const noexcept = 0;

    /**
     * @brief Get the element size in bytes
     *
     * @return size_t Element size
     */
    virtual size_t element_size() const noexcept = 0;

protected:
    SharedStorageBase() = default;
    SharedStorageBase(const SharedStorageBase&) = default;
    SharedStorageBase& operator=(const SharedStorageBase&) = default;
    SharedStorageBase(SharedStorageBase&&) noexcept = default;
    SharedStorageBase& operator=(SharedStorageBase&&) noexcept = default;
};

/**
 * @brief Typed shared storage implementation
 *
 * @tparam T Element type
 */
template <typename T>
class BREZEL_API SharedStorage : public SharedStorageBase {
public:
    using StorageType = Storage<T>;
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    /**
     * @brief Create an empty shared storage
     *
     * @param flags Storage flags
     */
    explicit SharedStorage(StorageFlags flags = StorageFlags::Default)
        : m_storage(std::make_shared<StorageType>(flags)) {}

    /**
     * @brief Create shared storage with a specific size
     *
     * @param size Number of elements
     * @param flags Storage flags
     */
    explicit SharedStorage(size_t size,
                           StorageFlags flags = StorageFlags::Default)
        : m_storage(std::make_shared<StorageType>(size, flags)) {}

    /**
     * @brief Create shared storage with a specific size and initialize with a
     * value
     *
     * @param size Number of elements
     * @param value Value to initialize with
     * @param flags Storage flags
     */
    SharedStorage(size_t size, const T& value,
                  StorageFlags flags = StorageFlags::Default)
        : m_storage(std::make_shared<StorageType>(size, value, flags)) {}

    /**
     * @brief Create shared storage from existing storage
     *
     * @param storage Existing storage
     */
    explicit SharedStorage(const StorageType& storage)
        : m_storage(std::make_shared<StorageType>(storage)) {}

    /**
     * @brief Create shared storage from existing storage (move)
     *
     * @param storage Existing storage
     */
    explicit SharedStorage(StorageType&& storage)
        : m_storage(std::make_shared<StorageType>(std::move(storage))) {}

    /**
     * @brief Create shared storage from existing storage pointer
     *
     * @param storage_ptr Existing storage pointer
     */
    explicit SharedStorage(std::shared_ptr<StorageType> storage_ptr)
        : m_storage(std::move(storage_ptr)) {
        if (!m_storage) {
            throw core::error::LogicError("Null storage pointer");
        }
    }

    /**
     * @brief Create shared storage that wraps external data (does not take
     * ownership)
     *
     * @param data Pointer to external data
     * @param size Number of elements
     * @param flags Storage flags
     */
    SharedStorage(pointer data, size_t size,
                  StorageFlags flags = StorageFlags::Default)
        : m_storage(std::make_shared<StorageType>(data, size, flags)) {}

    /**
     * @brief Check if storage is shared with others
     *
     * @return bool True if shared (reference count > 1)
     */
    bool is_shared() const noexcept override {
        return m_storage.use_count() > 1;
    }

    /**
     * @brief Get reference count
     *
     * @return long Number of references
     */
    long use_count() const noexcept override { return m_storage.use_count(); }

    /**
     * @brief Make a unique (non-shared) copy if needed
     *
     * @return SharedStorage& Reference to potentially new storage
     */
    SharedStorageBase& unshare() override {
        if (is_shared()) {
            // Create a new copy
            m_storage = std::make_shared<StorageType>(*m_storage);
        }

        return *this;
    }

    /**
     * @brief Get raw pointer to underlying storage
     *
     * @return StorageBase* Pointer to storage
     */
    StorageBase* get() noexcept override { return m_storage.get(); }

    /**
     * @brief Get raw pointer to underlying storage (const)
     *
     * @return const StorageBase* Pointer to storage
     */
    const StorageBase* get() const noexcept override { return m_storage.get(); }

    /**
     * @brief Clone the shared storage
     *
     * @return std::unique_ptr<SharedStorageBase> Clone
     */
    std::unique_ptr<SharedStorageBase> clone() const override {
        return std::make_unique<SharedStorage>(*this);
    }

    /**
     * @brief Create a type-erased copy of this storage
     *
     * @return std::unique_ptr<SharedStorageBase> Type-erased copy
     */
    std::unique_ptr<SharedStorageBase> type_erased_copy() const override {
        return std::make_unique<SharedStorage>(*this);
    }

    /**
     * @brief Get the element type information
     *
     * @return std::type_index Type index
     */
    std::type_index type_index() const noexcept override {
        return std::type_index(typeid(T));
    }

    /**
     * @brief Get the element size in bytes
     *
     * @return size_t Element size
     */
    size_t element_size() const noexcept override { return sizeof(T); }

    /**
     * @brief Get typed pointer to storage
     *
     * @return std::shared_ptr<StorageType> Storage pointer
     */
    std::shared_ptr<StorageType> storage_ptr() const noexcept {
        return m_storage;
    }

    /**
     * @brief Get reference to storage
     *
     * @return StorageType& Storage reference
     */
    StorageType& storage() noexcept { return *m_storage; }

    /**
     * @brief Get reference to storage (const)
     *
     * @return const StorageType& Storage reference
     */
    const StorageType& storage() const noexcept { return *m_storage; }

    /**
     * @brief Get raw pointer to data
     *
     * @return pointer Data pointer
     */
    pointer data() noexcept { return m_storage->data_ptr(); }

    /**
     * @brief Get raw pointer to data (const)
     *
     * @return const_pointer Data pointer
     */
    const_pointer data() const noexcept { return m_storage->data_ptr(); }

    /**
     * @brief Get the number of elements
     *
     * @return size_t Number of elements
     */
    size_t size() const noexcept { return m_storage->size(); }

    /**
     * @brief Check if storage is empty
     *
     * @return bool True if empty
     */
    bool empty() const noexcept { return m_storage->empty(); }

    /**
     * @brief Get storage flags
     *
     * @return StorageFlags Flags
     */
    StorageFlags flags() const noexcept { return m_storage->flags(); }

    /**
     * @brief Resize the storage (creates a unique copy if shared)
     *
     * @param new_size New size in elements
     */
    void resize(size_t new_size) {
        unshare();
        m_storage->resize(new_size);
    }

    /**
     * @brief Fill with a value (creates a unique copy if shared)
     *
     * @param value Value to fill with
     */
    void fill(const T& value) {
        unshare();
        m_storage->fill(value);
    }

    /**
     * @brief Set storage to read-only
     */
    void set_read_only() noexcept { m_storage->set_read_only(); }

    /**
     * @brief Check if storage is read-only
     *
     * @return bool True if read-only
     */
    bool is_read_only() const noexcept { return m_storage->is_read_only(); }

    /**
     * @brief Element access (creates a unique copy if shared and writing)
     *
     * @param idx Index
     * @return reference Reference to element
     */
    reference operator[](size_t idx) {
        unshare();
        return (*m_storage)[idx];
    }

    /**
     * @brief Element access (const)
     *
     * @param idx Index
     * @return const_reference Reference to element
     */
    const_reference operator[](size_t idx) const noexcept {
        return (*m_storage)[idx];
    }

    /**
     * @brief Element access with bounds checking (creates a unique copy if
     * shared and writing)
     *
     * @param idx Index
     * @return reference Reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    reference at(size_t idx) {
        unshare();
        return m_storage->at(idx);
    }

    /**
     * @brief Element access with bounds checking (const)
     *
     * @param idx Index
     * @return const_reference Reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    const_reference at(size_t idx) const { return m_storage->at(idx); }

private:
    std::shared_ptr<StorageType> m_storage;
};

/**
 * @brief Type-erased shared storage for heterogeneous operations
 *
 * @details Provides a type-erased view of a shared storage with any element
 * type. This is useful for implementing operations that need to work with
 * tensors of different types without template specialization.
 */
class BREZEL_API TypeErasedStorage {
public:
    /**
     * @brief Default constructor
     */
    TypeErasedStorage() = default;

    /**
     * @brief Construct from shared storage
     *
     * @tparam T Element type
     * @param storage Shared storage
     */
    template <typename T>
    explicit TypeErasedStorage(const SharedStorage<T>& storage)
        : m_storage(storage.clone()),
          m_type_index(std::type_index(typeid(T))),
          m_element_size(sizeof(T)) {}

    /**
     * @brief Construct from shared storage base
     *
     * @param storage Shared storage base
     */
    explicit TypeErasedStorage(std::unique_ptr<SharedStorageBase> storage)
        : m_storage(std::move(storage)) {
        if (m_storage) {
            m_type_index = m_storage->type_index();
            m_element_size = m_storage->element_size();
        }
    }

    /**
     * @brief Convert back to typed shared storage
     *
     * @tparam T Desired element type
     * @return SharedStorage<T> Typed shared storage
     * @throws core::error::LogicError if types are incompatible
     */
    template <typename T>
    SharedStorage<T> to_typed() const {
        if (std::type_index(typeid(T)) != m_type_index) {
            throw core::error::LogicError(
                "Type mismatch: cannot convert from {} to {}",
                m_type_index.name(), typeid(T).name());
        }

        if (!m_storage) {
            throw core::error::LogicError("Storage is null");
        }

        // Downcast to the specific type
        auto* storage_ptr = dynamic_cast<SharedStorage<T>*>(m_storage.get());
        if (!storage_ptr) {
            throw core::error::LogicError("Failed to cast storage");
        }

        return *storage_ptr;
    }

    /**
     * @brief Check if storage is null
     *
     * @return bool True if null
     */
    bool is_null() const noexcept { return m_storage == nullptr; }

    /**
     * @brief Get type index
     *
     * @return std::type_index Type index
     */
    std::type_index type_index() const noexcept { return m_type_index; }

    /**
     * @brief Get element size in bytes
     *
     * @return size_t Element size
     */
    size_t element_size() const noexcept { return m_element_size; }

    /**
     * @brief Get the raw storage pointer
     *
     * @return SharedStorageBase* Storage pointer
     */
    SharedStorageBase* get() noexcept { return m_storage.get(); }

    /**
     * @brief Get the raw storage pointer (const)
     *
     * @return const SharedStorageBase* Storage pointer
     */
    const SharedStorageBase* get() const noexcept { return m_storage.get(); }

    /**
     * @brief Check if storage is of a specific type
     *
     * @tparam T Type to check
     * @return bool True if types match
     */
    template <typename T>
    bool is_type() const noexcept {
        return std::type_index(typeid(T)) == m_type_index;
    }

private:
    std::unique_ptr<SharedStorageBase> m_storage;  ///< Storage pointer
    std::type_index m_type_index{typeid(void)};    ///< Type index
    size_t m_element_size = 0;                     ///< Element size in bytes
};

/**
 * @brief Create a new shared storage
 *
 * @tparam T Element type
 * @param size Number of elements
 * @param flags Storage flags
 * @return SharedStorage<T> New shared storage
 */
template <typename T>
BREZEL_NODISCARD inline SharedStorage<T> make_shared_storage(
    size_t size, StorageFlags flags = StorageFlags::Default) {
    return SharedStorage<T>(size, flags);
}

/**
 * @brief Create a new shared storage with a specific value
 *
 * @tparam T Element type
 * @param size Number of elements
 * @param value Value to initialize with
 * @param flags Storage flags
 * @return SharedStorage<T> New shared storage
 */
template <typename T>
BREZEL_NODISCARD inline SharedStorage<T> make_shared_storage(
    size_t size, const T& value, StorageFlags flags = StorageFlags::Default) {
    return SharedStorage<T>(size, value, flags);
}

/**
 * @brief Create a shared storage that wraps external data (does not take
 * ownership)
 *
 * @tparam T Element type
 * @param data Pointer to external data
 * @param size Number of elements
 * @param flags Storage flags
 * @return SharedStorage<T> Shared storage wrapper
 */
template <typename T>
BREZEL_NODISCARD inline SharedStorage<T> make_shared_storage_external(
    T* data, size_t size, StorageFlags flags = StorageFlags::Default) {
    return SharedStorage<T>(data, size, flags);
}

}  // namespace brezel::tensor::storage