#pragma once

#include <tbb/blocked_range.h>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <boost/align/aligned_allocator.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/vector.hpp>
#include <boost/smart_ptr/atomic_shared_ptr.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <brezel/tensor/strides.hpp>
#include <concepts>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>

#ifdef BREZEL_SIMD_AVX512
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX2)
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX)
#include <immintrin.h>
#endif

namespace brezel::tensor {
/**
 * @brief Concept for valid tensor data types
 *
 */
template <typename T>
concept TensorScalar = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

/**
 * @brief Main tensor class providing n-dimensional array functionality with
 * SIMD and parallel processing
 * @tparam T Data type of tensor elements
 *
 */
template <TensorScalar T>
class BREZEL_API Tensor {
    static constexpr size_t kCacheLineSize = BREZEL_CACHE_LINE_SIZE;
    static constexpr size_t kDefaultBlockSize = 1024;

public:
    using value_type = T;
    using pointer = *T;
    using const_pointer = const* T;
    using reference = &T;
    using const_reference = const& T;
    using allocator_type = tbb::cache_aligned_allocator<T>;
    using storage_vector = boost::container::vector<T, allocator_type>;

    /**
     * @brief Creates an empty tensor
     *
     */
    BREZEL_NODISCARD Tensor() = default;

    /**
     * @brief Creates a tesnor with a given shape
     * @param shape Shape of the tensor
     */
    BREZEL_NODISCARD explicit Tensor(const Shape& shape)
        : m_shape(shape),
          m_strides(shape),
          m_storage(std::make_shared<Storage>(shape.numel())) {}

    /**
     * @brief Creates a tensor with given shape and initial value
     *
     * @param shape Shape of the tensor
     * @param value Initial value for all elements
     */
    BREZEL_NODISCARD Tensor(const Shape& shape, T value)
        : m_shape(shape),
          m_strides(shape),
          m_storage(std::make_shared<Storage>(shape.numel(), value)) {}

    // Move semantics
    BREZEL_MOVEABLE(Tensor);
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    /**
     * @brief Element access with bounds checking
     *
     * @param indices Array of indices
     * @return Reference to element
     * @throws LogicError if indices are out of bounds
     */
    BREZEL_NODISCARD reference at(std::span<const int64_t> indices) {
        validate_indices(indices);
        return data()[m_strides.get_linear_index(indices)];
    }

    BREZEL_NODISCARD const_reference
    at(std::span<const int64_t> indices) const {
        validate_indices(indices);
        return data()[m_strides.get_linear_index(indices)];
    }

    // Basic operations
    /**
     * @brief Adds another tensor element-wise
     *
     * @param other Tensor to add
     * @return New tensor containing the result
     */
    BREZEL_NODISCARD Tensor add(const Tensor& other) const {
        BREZEL_ENSURE(m_shape == other.m_shape, "Tensor shapes must match");
        Tensor result(m_shape);

        if (is_contiguous() && other.is_contiguous()) {
            parallel_simd_op(other, result, std::plus<T>());
        } else {
            const size_t n = numel();
            for (size_t i = 0; i < n; ++i)
                result.data()[i] = data()[i] + other.data()[i];
        }

        return result;
    }

    /**
     * @brief Multiplies with another tensor element-wise
     *
     * @param other Tensor to multiply with
     * @return New tensor containing the result
     */
    BREZEL_NODISCARD Tensor multiply(const Tensor& other) const {
        BREZEL_ENSURE(m_shape == other.m_shape, "Tensor shapes must match");
        Tensor result(m_shape);

        if (is_contiguous() && other.is_contiguous()) {
            parallel_simd_op(other, result, std::multiplies<T>());
        } else {
            const size_t n = numel();
            for (size_t i = 0; i < n; ++i)
                result.data()[i] = data()[i] * other.data()[i];
        }

        return result;
    }

    /**
     * @brief Reduces tensor along specified dimension
     *
     * @param dim Dimension to reduce
     * @param op Reduction operation
     * @return Reduced tensor
     */
    template <typename BinaryOp>
    BREZEL_NODISCARD Tensor reduce(size_t dim, BinaryOp op) const {
        BREZEL_ENSURE(dim < ndim(), "Invalid reduction dimension");

        Shape new_shape = m_shape;
        new_shape[dim] = 1;

        Tensor result(new_shape);
        const size_t n = numel();
        const size_t stride = m_strides[dim];
        const size_t dim_size = m_shape[dim];

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n / dim_size),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end();
                                   ++i) {
                                  T acc = data()[i * dim_size];
                                  for (size_t j = 1; j < dim_size; ++j)
                                      acc = op(acc, data()[i * dim_size + j]);

                                  result.data()[i] = acc;
                              }
                          });

        return result;
    }

    // Accessors
    BREZEL_NODISCARD BREZEL_FORCE_INLINE pointer data() noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE const_pointer data() const noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const noexcept {
        return m_shape;
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Strides& strides()
        const noexcept {
        return m_strides;
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t ndim() const noexcept {
        return m_shape.size();
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const noexcept {
        return m_shape.numel();
    }

    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous() const noexcept {
        return m_strides.is_contiguous(m_shape);
    }

    /**
     * @brief Creates a contiguous copy of the tensor
     *
     */
    BREZEL_NODISCARD Tensor contiguous() const {
        if (is_contiguous())
            return *this;

        Tensor result(m_shape);
        const size_t n = numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % m_shape[j];
                        temp /= m_shape[j];
                    }

                    result.data()[i] = this->at(indices);
                }
            });

        return result;
    }

private:
    /**
     * @brief Storage class for tensor data with atomic reference counting
     *
     */
    class Storage {
    public:
        explicit Storage(size_t size) : m_data(size) {}

        Storage(size_t size, T value) : m_data(size) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, size, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::fill_n(m_data.begin() + range.begin(),
                                range.end() - range.begin(), value);
                });
        }

        BREZEL_NODISCARD T* data() noexcept { return m_data.data(); }
        BREZEL_NODISCARD const T* data() const noexcept {
            return m_data.data();
        }

    private:
        storage_vector m_data;
    };

    /**
     * @brief SIMD-optimized parallel operation
     *
     */
    template <typename BinaryOp>
    void parallel_simd_op(const Tensor& other, Tensor& result,
                          BinaryOp op) const {
        const size_t n = numel();

#if defined(BREZEL_SIMD_AVX512) && std::is_same_v<T, float>
        constexpr size_t simd_width = 16;
        const size_t simd_size = n - (n % simd_width);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, simd_size, kDefaultBlockSize),
            [&](const std::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end();
                     i += simd_width) {
                    __m512 a = _mm512_load_ps(&data()[i]);
                    __m512 b = _mm512_load_ps(&other.data()[i]);
                    __m512 c;

                    if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                        c = _mm512_add_ps(a, b);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::multiplies<T>>) {
                        c = _mm512_mul_ps(a, b);
                    }
                    
                    _mm512_store_ps(&result.data()[i], c);
                }
            });
#elif defined(BREZEL_SIMD_AVX2) && std::is_same_v<T, float>
#endif

        for (size_t i = simd_size; i < n; ++i)
            result.data()[i] = op(data()[i], other.data()[i]);
    }

    /**
     * @brief Validates the given indices
     *
     * @param indices
     */
    void validate_indices(std::span<const int64_t> indices) const {
        BREZEL_ENSURE(indices.size() == ndim(),
                      "Expected {} indices but got {}", ndim(), indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            BREZEL_ENSURE(indices[i] >= 0 && indices[i] < m_shape[i],
                          "Index {} out of bounds for dimension {}", indices[i],
                          i);
        }
    }

    Shape m_shape;
    Strides m_strides;
    boost::atomic_shared_ptr<Storage> m_storage;
    size_t m_offset{0};  /// For view and slices
};
}  // namespace brezel::tensor