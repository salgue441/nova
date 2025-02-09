#pragma once

#include <tbb/concurrent_vector.h>

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace brezel::tensor {
/**
 * @brief Supported data types for tensor operations
 *
 */
enum class DataType { Float32, Float64, Int32, Int64, Bool };

/**
 * @brief Memory layout for tensor storage
 *
 * @details Contiguous has elements stored contiguosuly in memory
 * @details Strided Elements are stored with strides
 * @details Sparse Only non-zero elements are stored
 */
enum class Layout { Contiguous, Strided, Sparse };

/**
 * @brief Runtime device specification
 *
 * @details CUDA support is not yet implemented
 */
enum class Device { CPU, CUDA };

namespace details {
/**
 * @brief Helper class for type-based operations
 * @tparam T Data type
 */
template <typename T>
struct TypeInfo {
    static constexpr DataType dtype =
        std::is_same_v<T, float>     ? DataType::Float32
        : std::is_same_v<T, double>  ? DataType::Float64
        : std::is_same_v<T, int32_t> ? DataType::Int32
        : std::is_same_v<T, int64_t> ? DataType::Int64
        : std::is_same_v<T, bool>
            ? DataType::Bool
            : throw core::error::LogicError("Unsupported data type");
};

/**
 * @brief Base storage class for tensor data
 *
 */
class BREZEL_API StorageBase {
public:
    virtual ~StorageBase() = default;

    BREZEL_NODISCARD virtual size_t size() const noexcept = 0;
    BREZEL_NODISCARD virtual size_t itemsize() const noexcept = 0;
    BREZEL_NODISCARD virtual DataType dtype() const noexcept = 0;
    BREZEL_NODISCARD virtual void* data() noexcept = 0;
    BREZEL_NODISCARD virtual const void* data() const noexcept = 0;
};

/**
 * @brief Type-specific storage implementation
 * @tparam T Data Type
 */
template <typename T>
class Storage : public StorageBase {
public:
    explicit Storage(size_t size) : m_data(size) {}

    BREZEL_NODISCARD size_t size() const noexcept override {
        return m_data.size();
    }

    BREZEL_NODISCARD size_t itemsize() const noexcept override {
        return sizeof(T);
    }

    BREZEL_NODISCARD DataType dtype() const noexcept override {
        return TypeInfo<T>::dtype;
    }

    BREZEL_NODISCARD void* data() noexcept override { return m_data.data(); }

    BREZEL_NODISCARD const void* data() const noexcept override {
        return m_data.data();
    }

private:
    std::vector<T> m_data;
};
}  // namespace details

/**
 * @brief Main tensor class
 * @note PyTorch-like functionality
 * @details Implements a multi-dimensional array with dynamic shape
 */
class BREZEL_API Tensor {
private:
    Shape m_shape;
    Strides m_strides;
    std::shared_ptr<details::StorageBase> m_storage;
};
}  // namespace brezel::tensor