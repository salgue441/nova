#pragma once

#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/layout.hpp>
#include <brezel/tensor/shape.hpp>
#include <brezel/tensor/stride.hpp>
#include <brezel/tensor/tensor_impl.hpp>
#include <brezel/tensor/view.hpp>

namespace brezel::tensor {
// Template aliases for common tensor types
using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;
using Int8Tensor = Tensor<int8_t>;
using Int16Tensor = Tensor<int16_t>;
using Int32Tensor = Tensor<int32_t>;
using Int64Tensor = Tensor<int64_t>;
using UInt8Tensor = Tensor<uint8_t>;
using UInt16Tensor = Tensor<uint16_t>;
using UInt32Tensor = Tensor<uint32_t>;
using UInt64Tensor = Tensor<uint64_t>;
using BoolTensor = Tensor<bool>;

// Convenience factory functions
/**
 * @brief Creates a tensor filled with zeros of the specified shape
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param shape The shape of the tensor to be created
 * @return Tensor<T> A new tensor with all elements initialized to zero
 */
template <TensorScalar T>
inline Tensor<T> zeros(const Shape& shape) {
    return Tensor<T>::zeros(shape);
}

/**
 * @brief Creates a new tensor filled with ones with the specified shape.
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param shape The desired shape of the tensor
 * @return Tensor<T> A new tensor of the specified shape filled with ones
 */
template <TensorScalar T>
inline Tensor<T> ones(const Shape& shape) {
    return Tensor<T>::ones(shape);
}

/**
 * @brief Creates an identity matrix tensor of size n x n
 *
 * Creates a square 2-dimensional tensor where elements on the main diagonal are
 * 1 and all other elements are 0.
 *
 * @tparam T Scalar type of the tensor elements
 * @param n Number of rows and columns in the identity matrix
 * @return Tensor<T> An n x n identity matrix tensor
 *
 * @note The returned tensor will have shape (n, n)
 */
template <TensorScalar T>
inline Tensor<T> eye(int64_t n) {
    return Tensor<T>::eye(n);
}

/**
 * @brief Creates a tensor filled with a specified value.
 *
 * This function creates a new tensor with the given shape where all elements
 * are initialized to the specified value.
 *
 * @tparam T The scalar type of the tensor elements
 * @param shape The shape of the tensor to create
 * @param value The value to fill the tensor with
 * @return Tensor<T> A new tensor filled with the specified value
 */
template <TensorScalar T>
inline Tensor<T> full(const Shape& shape, const T& value) {
    return Tensor<T>::full(shape, value);
}

/**
 * @brief Creates a 1-dimensional tensor with values ranging from start to end
 * with a specified step
 *
 * @tparam T Scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param start The first value in the sequence
 * @param end The upper limit of the sequence (exclusive)
 * @param step The difference between consecutive values (defaults to 1)
 * @return Tensor<T> A 1-dimensional tensor containing the generated sequence
 *
 * @note The sequence includes start but excludes end, similar to Python's range
 * function
 * @note The resulting tensor will have shape [ceil((end - start) / step)]
 */
template <TensorScalar T>
inline Tensor<T> arange(T start, T end, T step = T(1)) {
    return Tensor<T>::arange(start, end, step);
}

/**
 * @brief Creates a tensor with evenly spaced values between start and end.
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param start The starting value of the sequence
 * @param end The ending value of the sequence
 * @param steps The number of points in the sequence
 * @return Tensor<T> A new tensor containing evenly spaced values from start to
 * end
 *
 * @note The values are computed as a linear progression from start to end,
 *       including both endpoints.
 */
template <TensorScalar T>
inline Tensor<T> linspace(T start, T end, int64_t steps) {
    return Tensor<T>::linspace(start, end, steps);
}

/**
 * @brief Creates a tensor filled with random values within a specified range
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param shape The shape of the tensor to create
 * @param min The minimum value for random numbers (default: 0)
 * @param max The maximum value for random numbers (default: 1)
 * @return Tensor<T> A new tensor with the specified shape filled with random
 * values
 *
 * This function generates a tensor with random values uniformly distributed
 * between min and max (inclusive). The shape parameter determines the
 * dimensions of the resulting tensor.
 */
template <TensorScalar T>
inline Tensor<T> random(const Shape& shape, T min = T(0), T max = T(1)) {
    return Tensor<T>::random(shape, min, max);
}

/**
 * @brief Creates a new tensor with random values from a normal (Gaussian)
 * distribution
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param shape The shape of the tensor to create
 * @param mean The mean (μ) of the normal distribution (default: 0)
 * @param std_dev The standard deviation (σ) of the normal distribution
 * (default: 1)
 * @return Tensor<T> A new tensor filled with random values from N(mean,
 * std_dev²)
 *
 * @note The values are generated using a normal distribution where each element
 * x follows the probability density function: P(x) = 1/(std_dev·√(2π)) ·
 * e^(-(x-mean)²/(2·std_dev²))
 */
template <TensorScalar T>
inline Tensor<T> randn(const Shape& shape, T mean = T(0), T std_dev = T(1)) {
    return Tensor<T>::randn(shape, mean, std_dev);
}

/**
 * @brief Creates a tensor from a range of values
 *
 * @tparam T The scalar type of the tensor elements
 * @tparam R The range type containing the data
 * @param shape The shape of the tensor to be created
 * @param data The range containing the data to initialize the tensor
 * @return Tensor<T> A new tensor with the specified shape and data
 *
 * @note The range value type must be convertible to the tensor scalar type T
 * @note The range size must match the product of the shape dimensions
 */
template <TensorScalar T, std::ranges::range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, T>
inline Tensor<T> from_range(const Shape& shape, R&& data) {
    return Tensor<T>(shape, std::forward<R>(data));
}

// Tensor view operations (avoids circular dependency)
template <TensorScalar T>
TensorView<T> Tensor<T>::reshape(const Shape& new_shape) const {
    if (new_shape.numel() != numel()) {
        throw core::error::LogicError(
            "Cannot reshape tensor of size {} to size {}", numel(),
            new_shape.numel());
    }

    if (is_contiguous()) {
        auto new_layout = m_layout.reshape(new_shape);
        return TensorView<T>(*this, new_layout);
    }

    return contiguous().reshape(new_shape);
}

template <TensorScalar T>
TensorView<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const {
    const size_t dims = ndim();

    if (dim0 < 0)
        dim0 += dims;

    if (dim1 < 0)
        dim1 += dims;

    if (dim0 < 0 || dim0 >= static_cast<int64_t>(dims) || dim1 < 0 ||
        dim1 >= static_cast<int64_t>(dims)) {
        throw core::error::LogicError(
            "Dimension out of range for transpose. Got dimensions {} and {} "
            "for tensor with {} dimensions",
            dim0, dim1, dims);
    }

    if (dim0 == dim1) {
        return TensorView<T>(*this, m_layout);
    }

    auto new_layout = m_layout.transpose(dim0, dim1);
    return TensorView<T>(*this, new_layout);
}

template <TensorScalar T>
TensorView<T> Tensor<T>::slice(
    const std::vector<std::pair<int64_t, int64_t>>& indices) const {
    const auto& current_shape = shape();
    const size_t dims = ndim();

    if (indices.size() != dims) {
        throw core::error::LogicError(
            "Expected {} indices for slice operation, but got {}", dims,
            indices.size());
    }

    Shape new_shape;
    std::vector<int64_t> new_strides;
    std::vector<int64_t> offsets;

    for (size_t i = 0; i < dims; ++i) {
        const auto& [start, end] = indices[i];
        int64_t dim_size = current_shape[i];
        int64_t real_start = start < 0 ? start + dim_size : start;
        int64_t real_end = end < 0 ? end + dim_size : end;

        if (real_start < 0 || real_start >= dim_size) {
            throw core::error::LogicError(
                "Slice start index {} out of bounds for dimension {} with size "
                "{}",
                start, i, dim_size);
        }

        if (real_end <= real_start || real_end > dim_size) {
            throw core::error::LogicError(
                "Slice end index {} out of bounds for dimension {} with size "
                "{}",
                end, i, dim_size);
        }

        new_shape.push_back(real_end - real_start);
        new_strides.push_back(strides()[i]);
        offsets.push_back(real_start * strides()[i]);
    }

    size_t total_offset =
        std::accumulate(offsets.begin(), offsets.end(), static_cast<size_t>(0));

    LayoutDescriptor new_layout(new_shape, new_strides);
    new_layout.set_offset(m_layout.offset() + total_offset);
    new_layout.set_device(m_layout.device());
    new_layout.set_format(m_layout.format());

    return TensorView<T>(*this, new_layout);
}

template <TensorScalar T>
TensorView<T> Tensor<T>::permute(const std::vector<size_t>& dims) const {
    auto new_layout = m_layout.permute(dims);
    return TensorView<T>(*this, new_layout);
}

template <TensorScalar T>
TensorView<T> Tensor<T>::squeeze(int64_t dim) const {
    const auto& current_shape = shape();
    const size_t dims = ndim();

    if (dim >= 0) {
        if (dim >= static_cast<int64_t>(dims)) {
            throw core::error::LogicError(
                "Dimension {} out of range for tensor with {} dimensions", dim,
                dims);
        }

        if (current_shape[dim] != 1) {
            throw core::error::LogicError(
                "Cannot squeeze dimension {} with size {}", dim,
                current_shape[dim]);
        }

        std::vector<int64_t> new_shape_dims;
        new_shape_dims.reserve(dims - 1);

        for (size_t i = 0; i < dims; ++i) {
            if (i != static_cast<size_t>(dim)) {
                new_shape_dims.push_back(current_shape[i]);
            }
        }

        return reshape(Shape(new_shape_dims));
    } else {
        std::vector<int64_t> new_shape_dims;
        new_shape_dims.reserve(dims);

        for (size_t i = 0; i < dims; ++i) {
            if (current_shape[i] != 1) {
                new_shape_dims.push_back(current_shape[i]);
            }
        }

        if (new_shape_dims.empty()) {
            return reshape(Shape{});
        }

        return reshape(Shape(new_shape_dims));
    }
}

template <TensorScalar T>
TensorView<T> Tensor<T>::unsqueeze(int64_t dim) const {
    const auto& current_shape = shape();
    const size_t dims = ndim();

    if (dim < 0) {
        dim += dims + 1;
    }

    if (dim < 0 || dim > static_cast<int64_t>(dims)) {
        throw core::error::LogicError(
            "Dimension {} out of range for unsqueeze on tensor with {} "
            "dimensions",
            dim, dims);
    }

    std::vector<int64_t> new_shape_dims;
    new_shape_dims.reserve(dims + 1);

    for (size_t i = 0; i < dims + 1; ++i) {
        if (i == static_cast<size_t>(dim)) {
            new_shape_dims.push_back(1);
        }

        if (i < dims) {
            new_shape_dims.push_back(current_shape[i]);
        }
    }

    return reshape(Shape(new_shape_dims));
}
}  // namespace brezel::tensor