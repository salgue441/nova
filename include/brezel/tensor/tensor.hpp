#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/container/small_vector.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <brezel/tensor/strides.hpp>
#include <brezel/tensor/tensor_concept.hpp>
#include <brezel/tensor/tensor_expressions.hpp>
#include <memory>
#include <random>

namespace brezel::tensor {
// Forward declaration for factory class
template <TensorScalar T>
class TensorFactory;

/**
 * @brief Main tensor class providing N-dimensional array functionality with
 * SIMD acceleration, lazy evaluation, and PyTorch-like API.
 *
 * @details This class represents a multi-dimensional array of elements of type
 * T, with automatic memory management, broadcasting, and vectorized operations.
 * It integrates with the expression template system for lazy evaluation of
 * complex tensor operations without unnecessary temporaries.
 *
 * @tparam T Data type of tensor elements (must satisfy TensorScalar concept)
 * @see TensorScalar
 */
template <TensorScalar T>
class BREZEL_API Tensor : public TensorExpression<Tensor<T>, T> {
    static constexpr size_t kDefaultBlockSize = 1024;
    static constexpr size_t kCacheLineSize = BREZEL_CACHE_LINE_SIZE;

    friend class TensorFactory<T>;

public:
    // Type aliases for STL compatibility
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using allocator_type = tbb::cache_aligned_allocator<T>;

    // Container types
    using storage_vector = boost::container::vector<T, allocator_type>;
    using indices_vector = boost::container::small_vector<int64_t, 8>;
    using dims_vector = boost::container::small_vector<int64_t, 8>;

    // Iterators
    using iterator = typename storage_vector::iterator;
    using const_iterator = typename storage_vector::const_iterator;
    using reverse_iterator = typename storage_vector::reverse_iterator;
    using const_reverse_iterator =
        typename storage_vector::const_reverse_iterator;

    // Result types
    template <typename U>
    using BinaryOpResult =
        Tensor<decltype(std::declval<T>() + std::declval<U>())>;

    // Function types
    using unary_function_type = std::function<T(T)>;
    using binary_function_type = std::function<T(T, T)>;
    using reducer_function_type = std::function<T(T, T)>;
    using elementwise_function_type = std::function<void(T&)>;

    enum class DeviceType { CPU, CUDA, MKL, OPENCL };
    enum class LayoutType { Strided, Contiguous, Sparse };
    static constexpr DeviceType kDefaultDevice = DeviceType::CPU;
    static constexpr LayoutType kDefaultLayout = LayoutType::Contiguous;

    // Constructors
    /**
     * @brief Creates an empty scalar tensor
     */
    BREZEL_NODISCARD Tensor() = default;

    /**
     * @brief Creates a tensor with the specified shape, uninitialized values
     *
     * @param shape Shape of the tensor
     */
    BREZEL_NODISCARD explicit Tensor(const Shape& shape)
        : m_shape(shape),
          m_strides(shape),
          m_storage(std::make_shared<Storage>(shape.numel())) {}

    /**
     * @brief Creates a tensor with the specified shape from a list of
     * dimensions
     *
     * @param dims Dimensions of the tensor
     */
    BREZEL_NODISCARD explicit Tensor(std::initializer_list<int64_t> dims)
        : Tensor(Shape(dims)) {}

    /**
     * @brief Creates a tensor with the specified shape and initial value
     *
     * @param shape Shape of the tensor
     * @param value Initial value for all elements
     */
    BREZEL_NODISCARD Tensor(const Shape& shape, T value)
        : m_shape(shape),
          m_strides(shape),
          m_storage(std::make_shared<Storage>(shape.numel(), value)) {}

    /**
     * @brief Creates a tensor with the specified shape and initial value
     *
     * @param dims Dimensions of the tensor
     * @param value Initial value for all elements
     */
    BREZEL_NODISCARD Tensor(std::initializer_list<int64_t> dims, T value)
        : Tensor(Shape(dims), value) {}

    /**
     * @brief Creates a 1D tensor from an initializer list
     *
     * @param data Initializer list of values
     */
    BREZEL_NODISCARD Tensor(std::initializer_list<T> data)
        : m_shape(Shape({static_cast<int64_t>(data.size())})),
          m_strides(m_shape),
          m_storage(std::make_shared<Storage>(data.size())) {
        std::copy(data.begin(), data.end(), m_storage->data());
    }

    /**
     * @brief Creates a 2D tensor from an initializer list
     *
     * @param data Initializer list of values
     */
    BREZEL_NODISCARD Tensor(
        std::initializer_list<std::initializer_list<T>> data)
        : m_shape(Shape({static_cast<int64_t>(data.size()),
                         static_cast<int64_t>(data.begin()->size())})),
          m_strides(m_shape) {
        BREZEL_ENSURE(!data.empty(),
                      "Cannot create tensor from empty initializer list");
        BREZEL_ENSURE(
            std::all_of(data.begin(), data.end(),
                        [size = data.begin()->size()](const auto& row) {
                            return row.size() == size;
                        }),
            "All rows must have the same size");

        const size_t rows = data.size();
        const size_t cols = data.begin()->size();
        m_storage = std::make_shared<Storage>(rows * cols);

        size_t i = 0;
        for (const auto& row : data) {
            for (const auto& val : row) {
                m_storage->data()[i++] = val;
            }
        }
    }

    /**
     * @brief Creates a tensor from raw data
     *
     * @param data Raw data pointer
     * @param shape Shape of the tensor
     * @param copy Whether to copy the data or take ownership
     */
    BREZEL_NODISCARD Tensor(const T* data, const Shape& shape, bool copy = true)
        : m_shape(shape), m_strides(shape) {
        if (copy) {
            m_storage = std::make_shared<Storage>(shape.numel());
            std::copy(data, data + shape.numel(), m_storage->data());
        } else {
            m_storage = std::make_shared<Storage>(shape.numel());
            std::memcpy(m_storage->data(), data, shape.numel() * sizeof(T));
        }
    }

    /**
     * @brief Creates a tensor from a vector
     *
     * @param data Vector of values
     * @param shape Shape of the tensor
     */
    template <typename VecT>
    BREZEL_NODISCARD Tensor(const std::vector<VecT>& data, const Shape& shape)
        : m_shape(shape),
          m_strides(shape),
          m_storage(std::make_shared<Storage>(shape.numel())) {
        BREZEL_ENSURE(data.size() == shape.numel(),
                      "Data size must match shape's total number of elements");

        if constexpr (std::is_same_v<VecT, T>) {
            std::copy(data.begin(), data.end(), m_storage->data());
        } else {
            std::transform(data.begin(), data.end(), m_storage->data(),
                           [](const VecT& val) { return static_cast<T>(val); });
        }
    }

    /**
     * @brief Creates a tensor from a TensorExpression (lazy evaluation)
     *
     * @param expr Expression to evaluate
     */
    template <typename E>
    BREZEL_NODISCARD Tensor(const TensorExpression<E, T>& expr)
        : m_shape(expr.shape()),
          m_strides(expr.shape()),
          m_storage(std::make_shared<Storage>(expr.numel())) {
        const size_t n = expr.numel();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end();
                                   ++i) {
                                  m_storage->data()[i] = expr[i];
                              }
                          });
    }

    // Enable both copy and move
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Data access methods
    /**
     * @brief Gets the raw data pointer
     * @return Raw data pointer
     */
    BREZEL_NODISCARD pointer data() noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Gets the raw data pointer
     * @return Raw data pointer
     */
    BREZEL_NODISCARD const_pointer data() const noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Gets the shape of the tensor
     * @return Reference to shape
     */
    BREZEL_NODISCARD const Shape& shape() const noexcept { return m_shape; }

    /**
     * @brief Gets the total number of elements in the tensor
     * @return Number of elements
     */
    BREZEL_NODISCARD size_t numel() const noexcept { return m_shape.numel(); }

    /**
     * @brief Gets the size of the tensor along a specific dimension
     * @param dim Dimension
     * @return Size along the dimension
     */
    BREZEL_NODISCARD int64_t size(int64_t dim) const {
        dim = handle_negative_dim(dim);
        BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                      "Dimension out of range");

        return m_shape[dim];
    }

    /**
     * @brief Gets the number of dimensions of the tensor
     * @return Number of dimensions
     */
    BREZEL_NODISCARD size_t ndim() const noexcept { return m_shape.size(); }

    /**
     * @brief Checks if the tensor is contiguous in memory
     * @return True if contiguous
     */
    BREZEL_NODISCARD bool is_contiguous() const noexcept {
        return m_strides.is_contiguous(m_shape);
    }

    /**
     * @brief Access element at the specified location
     *
     * @param indices List of indices
     * @return Reference to element
     */
    BREZEL_NODISCARD reference at(std::span<const int64_t> indices) {
        validate_indices(indices);
        return data()[m_strides.get_linear_index(indices)];
    }

    /**
     * @brief Access element at the specified location
     *
     * @param indices List of indices
     * @return Const reference to element
     */
    BREZEL_NODISCARD const_reference
    at(std::span<const int64_t> indices) const {
        validate_indices(indices);
        return data()[m_strides.get_linear_index(indices)];
    }

private:
    /**
     * @brief Storage class for tensor data with atomic reference counting
     * Provides thread-safe memory management for tensor data
     */
    class Storage {
    public:
        /**
         * @brief Creates unitialized storage for specified number of elements
         *
         * @param size Number of elements
         */
        explicit Storage(size_t size) : m_data(size) {}

        /**
         * @brief Creates storage filled with specified values
         *
         * @param size Number of elements
         * @param value Value to fill storage with
         */
        Storage(size_t size, T value) : m_data(size) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, size, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::fill(m_data.begin() + range.begin(),
                              m_data.begin() + range.end(), value);
                });
        }

        /**
         * @brief Access to raw data pointer
         * @return Pointer to data
         */
        BREZEL_NODISCARD T* data() noexcept { return m_data.data(); }

        /**
         * @brief Access to raw data pointer (const)
         * @return Pointer to data
         */
        BREZEL_NODISCARD const T* data() const noexcept {
            return m_data.data();
        }

        /**
         * @brief Get number of elements in storage
         * @return Number of elements
         */
        BREZEL_NODISCARD size_t size() const noexcept { return m_data.size(); }

    private:
        storage_vector m_data;
    };

    // Core tensor properties
    Shape m_shape;
    Strides m_strides;
    std::shared_ptr<Storage> m_storage;

    /**
     * @brief Validates indices against tensor shape
     *
     * @param indices Indices to validate
     * @throws LogicError if indices are invalid
     */
    void validate_indices(std::span<const int64_t> indices) const {
        using brezel::core::error::LogicError;

        if (indices.size() != ndim()) {
            throw LogicError("Expected {} indices but got {}", ndim(),
                             indices.size());
        }

        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= m_shape[i]) {
                throw LogicError("Index {} out of bounds for dimension {}",
                                 indices[i], i);
            }
        }
    }

    /**
     * @brief Handle negative dimension index (Python-like indexing)
     *
     * @param dim Dimension index (negative values wrap around)
     * @return Normalized dimension index
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE int64_t
    handle_negative_dim(int64_t dim) const {
        const int64_t n_dims = static_cast<int64_t>(ndim());

        if (dim < 0) {
            dim += n_dims;
        }

        return dim;
    }

    /**
     * @brief Convert linear index to multi-dimensional indices
     *
     * @param linear_idx Linear index
     * @return Array of indices
     */
    BREZEL_NODISCARD std::vector<int64_t> linear_index_to_indices(
        size_t linear_idx) const {
        std::vector<int64_t> indices(ndim());
        size_t remaining = linear_idx;

        for (int64_t i = ndim() - 1; i >= 0; --i) {
            indices[i] = remaining % m_shape[i];
            remaining /= m_shape[i];
        }

        return indices;
    }

    /**
     * @brief Apply binary operation to tensors in parallel with SIMD
     * optimization where possible
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor
     * @param op Binary operation
     * @return Result tensor
     */
    template <typename BinaryOp>
    BREZEL_NODISCARD Tensor binary_op(const Tensor& other, BinaryOp op) const {
        using brezel::core::error::LogicError;

        if (m_shape != other.m_shape) {
            if (!m_shape.is_broadcastable_with(other.m_shape)) {
                LogicError("Cannot broadcast shapes {} and {}",
                           m_shape.to_string(), other.m_shape.to_string());
            }

            auto expr = make_binary_op(*this, other, op);
            return Tensor(expr);
        }

        Tensor result(m_shape);
        const size_t n = numel();

        if (is_contiguous() && other.is_contiguous()) {
            detail::apply_simd_op(result.data(), data(), other.data(), n, op,
                                  kDefaultBlockSize);
        } else {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto indices = linear_index_to_indices(i);
                        result.data()[i] = op(at(indices), other.at(indices));
                    }
                });
        }

        return result;
    }

    /**
     * @brief Apply in-place binary operation with SIMD optimization where
     * possible
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor
     * @param op Binary operation
     * @return Reference to this tensor
     */
    template <typename BinaryOp>
    Tensor& binary_op_inplace(const Tensor& other, BinaryOp op) {
        using brezel::core::error::LogicError;

        if (m_shape != other.m_shape) {
            throw LogicError(
                "Cannot perform in-place operation with different shapes {} "
                "and {}",
                m_shape.to_string(), other.m_shape.to_string());
        }

        const size_t n = numel();

        if (is_contiguous() && other.is_contiguous()) {
            detail::apply_simd_op(data(), data(), other.data(), n, op,
                                  kDefaultBlockSize);
        } else {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        auto indices = linear_index_to_indices(i);
                        at(indices) = op(at(indices), other.at(indices));
                    }
                });
        }

        return *this;
    }

    /**
     * @brief Apply unary operation to tensor
     *
     * @tparam UnaryOp Unary operation type
     * @param op Unary operation
     * @return Result tensor
     */
    template <typename UnaryOp>
    BREZEL_NODISCARD Tensor unary_op(UnaryOp op) const {
        Tensor result(m_shape);
        const size_t n = numel();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end();
                                   ++i) {
                                  if (is_contiguous()) {
                                      result.data()[i] = op(data()[i]);
                                  } else {
                                      auto indices = linear_index_to_indices(i);
                                      result.data()[i] = op(at(indices));
                                  }
                              }
                          });

        return result;
    }

    /**
     * @brief Apply reduction operation along specified dimension
     *
     * @tparam ReduceOp Reduction operation type
     * @param dim Dimension to reduce
     * @param op Reduction operation
     * @param init Initial value for reduction
     * @return Result tensor
     */
    template <typename ReduceOp>
    BREZEL_NODISCARD Tensor reduce(int64_t dim, ReduceOp op, T init) const {
        using brezel::core::error::LogicError;

        if (dim == -1) {
            T result = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize), init,
                [&](const tbb::blocked_range<size_t>& range, T local_init) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        if (is_contiguous()) {
                            local_init = op(local_init, data()[i]);
                        } else {
                            auto indices = linear_index_to_indices(i);
                            local_init = op(local_init, at(indices));
                        }
                    }
                    return local_init;
                },
                [&](T a, T b) { return op(a, b); });

            return Tensor({1}, result);
        }

        dim = handle_negative_dim(dim);
        if (dim < 0 || dim >= static_cast<int64_t>(ndim())) {
            throw LogicError("Dimension out of bounds: {}", dim);
        }

        std::vector<int64_t> new_shape;
        for (size_t i = 0; i < ndim(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                new_shape.push_back(m_shape[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
        }

        Tensor result(Shape(new_shape), init);
        const size_t n_slices = m_shape[dim];
        const size_t slice_sizes = numel() / n_slices;
        const size_t dim_stride = m_strides[dim];

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t out_idx = range.begin(); out_idx < range.end();
                     ++out_idx) {
                    auto out_indices = result.linear_index_to_indices(out_idx);
                    T accum = init;

                    for (size_t d = 0; d < n_slices; ++d) {
                        std::vector<int64_t> in_indices;
                        size_t out_pos = 0;

                        for (size_t i = 0; i < ndim(); ++i) {
                            if (i != static_cast<size_t>(dim)) {
                                in_indices.push_back(out_indices[out_pos++]);
                            } else {
                                in_indices.push_back(d);
                            }
                        }

                        accum = op(accum, at(in_indices));
                    }

                    result.data()[out_idx] = accum;
                }
            });

        return result;
    }

    /**
     * @brief Get the type name as string
     *
     * @tparam U Type to get name for
     * @return String representation of type
     */
    template <typename U>
    BREZEL_NODISCARD static std::string type_name() {
        if constexpr (std::is_same_v<U, float>) {
            return "float32";
        } else if constexpr (std::is_same_v<U, double>) {
            return "float64";
        } else if constexpr (std::is_same_v<U, int32_t>) {
            return "int32";
        } else if constexpr (std::is_same_v<U, int64_t>) {
            return "int64";
        } else if constexpr (std::is_same_v<U, bool>) {
            return "bool";
        } else {
            return typeid(U).name();
        }
    }
};
}  // namespace brezel::tensor

#include <brezel/tensor/tensor_factory.hpp>
