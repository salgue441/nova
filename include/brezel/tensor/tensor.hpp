#pragma once

#include <brezel/core/macros.hpp>
#include <brezel/tensor/storage.hpp>

namespace brezel::tensor {
/**
 * @brief Concept for valid tensor data types
 * @tparam T The begin checked against the concept
 */
template <typename T>
concept TensorScalar = StorageScalar<T>;

// Forward declarations
template <TensorScalar T>
class Tensor;

// Common tensor types
using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;
using IntTensor = Tensor<int32_t>;
using LongTensor = Tensor<int64_t>;
using BoolTensor = Tensor<bool>;

/**
 * @brief Main tensor class for n-dimensional array operations
 * @tparam T Data type for tensor elements
 */
template <TensorScalar T>
class BREZEL_API Tensor {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;

    // Constructors
private:
    LayoutDescriptor m_layout;
    SharedStorage m_storage;

    // Helper methods
    /**
     * @brief Applies a binary operation to corresponding elements of two
     * tensors
     *
     * @tparam BinaryOp Type of the binary operation
     * @param other The second tensor
     * @param op The binary operation to apply
     * @return Result of the operation
     */
    template <typename BinaryOp>
    BREZEL_NODISCARD Tensor binary_op(const Tensor& other, BinaryOp op) const {}
};
}  // namespace brezel::tensor