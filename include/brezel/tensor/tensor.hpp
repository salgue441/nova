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
#include <cmath>
#include <concepts>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <type_traits>
#include <vector>

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
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using allocator_type = tbb::cache_aligned_allocator<T>;
    using storage_vector = boost::container::vector<T, allocator_type>;

    // Static factory methods
    /**
     * @brief Creates an empty tensor with no elements
     *
     * @return Empty tensor
     */
    BREZEL_NODISCARD static Tensor create_empty() { return Tensor(); }

    /**
     * @brief Creates a tensor filled with zeros
     *
     * @param shape Desired shape
     * @return Tensor Resulting tensor
     */
    BREZEL_NODISCARD static Tensor zeros(const Shape& shape) {
        return Tensor(shape, T(0));
    }

    /**
     * @brief Creates a default uninitialized tensor with a specific shape
     *
     * @param shape Shape of the tensor
     * @return Uninitialized tensor with the specified shape
     */
    BREZEL_NODISCARD static Tensor default_tensor(
        const Shape& shape = Shape()) {
        return Tensor(shape);
    }

    /**
     * @brief Creates a tensor filled with ones
     *
     * @param shape Desired shape
     * @return Tensor Resulting tensor
     */
    BREZEL_NODISCARD static Tensor ones(const Shape& shape) {
        return Tensor(shape, T(1));
    }

    /**
     * @brief Creates a tensor filled with random values from uniform
     * distribution
     *
     * @param shape Desired shape
     * @param min Minimum value
     * @param max Maximum value
     * @return Tensor Resulting tensor
     */
    BREZEL_NODISCARD static Tensor random_uniform(const Shape& shape,
                                                  T min = T(0), T max = T(1)) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);

        Tensor result(shape);
        const size_t n = result.numel();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end();
                                   ++i) {
                                  result.data()[i] = dist(gen);
                              }
                          });

        return result;
    }

    /**
     * @brief Creates a tensor with random values from normal distribution
     *
     * @param shape Desired shape
     * @param mean Mean of the distribution
     * @param std Standard deviation of the distribution
     */
    BREZEL_NODISCARD static Tensor random_normal(const Shape& shape,
                                                 T mean = T(0), T std = T(1)) {
        Tensor result(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, std);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                std::mt19937 local_gen(rd());  // Thread-local generator

                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = dist(local_gen);
            });

        return result;
    }

    /**
     * @brief Creates an identity matrix of size n
     *
     * @param n Size of the identity matrix
     * @return Tensor Resulting identity matrix
     */
    BREZEL_NODISCARD static Tensor eye(int64_t n) {
        Tensor result(Shape({n, n}), T(0));

        tbb::parallel_for(tbb::blocked_range<int64_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<int64_t>& range) {
                              for (int64_t i = range.begin(); i < range.end();
                                   ++i)
                                  result.at({i, i}) = T(1);
                          });

        return result;
    }

    /**
     * @brief Range of values from start to end
     *
     * @param start Start value
     * @param end End value
     * @param step Step size between values (default 1)
     * @return Tensor Resulting tensor
     */
    BREZEL_NODISCARD static Tensor arange(T start, T end, T step = T(1)) {
        BREZEL_ENSURE(step != T(0), "Step cannot be zero");

        Shape shape({static_cast<int64_t>((end - start) / step)});
        Tensor result(shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = start + i * step;
            });

        return result;
    }

    /**
     * @brief Linear space of values from start to end
     *
     * @param start Start value
     * @param end End value
     * @param steps Number of steps (default 100)
     */
    BREZEL_NODISCARD static Tensor linspace(T start, T end, size_t steps) {
        Shape shape({static_cast<int64_t>(steps)});
        Tensor result(shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = start + i * (end - start) / (steps - 1);
            });

        return result;
    }

    /**
     * @brief Unitialized tensor with given shape
     *
     * @param shape Shape of the tensor
     * @return Tensor Uninitialized tensor
     */
    BREZEL_NODISCARD static Tensor uninitialized(const Shape& shape) {
        return Tensor(shape);
    }

    /**
     * @brief Returns a tensor with given values
     * @note Like ones but with custom values
     *
     * @param shape Shape of the tensor
     * @param value Value to fill the tensor with
     */
    BREZEL_NODISCARD static Tensor full(const Shape& shape, T value) {
        return Tensor(shape, value);
    }

    /**
     * @brief Creates a diagonal matrix
     *
     * @param diag Diagonal values
     * @return Tensor Resulting diagonal matrix
     */
    BREZEL_NODISCARD static Tensor diag(const Tensor& diag) {
        Shape shape({diag.numel(), diag.numel()});
        Tensor result(shape, T(0));

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, diag.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.at({i, i}) = diag.data()[i];
            });

        return result;
    }

    /**
     * @brief Random tensor with same shape
     *
     * @param tensor Tensor to get shape from
     * @return Tensor Random tensor with same shape
     */
    BREZEL_NODISCARD static Tensor random_like(const Tensor& tensor) {
        return random_uniform(tensor.shape());
    }

    //  Constructors
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

    // Enable both move and copy
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Element access
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

    /**
     * @brief Element access with bounds checking
     *
     * @param indices Indices of the element
     * @return const_reference to element
     */
    BREZEL_NODISCARD const_reference
    at(std::span<const int64_t> indices) const {
        validate_indices(indices);
        return data()[m_strides.get_linear_index(indices)];
    }

    /**
     * @brief Element access with bounds checking
     *
     * @param indices Array of indices
     * @return Reference to element
     * @throws LogicError if indices are out of bounds
     */
    BREZEL_NODISCARD reference at(std::initializer_list<int64_t> indices) {
        validate_indices(indices);
        return at(std::span(indices));
    }

    /**
     * @brief Element access with bounds checking
     *
     * @param indices Indices of the element
     * @return const_reference to element
     */
    BREZEL_NODISCARD const_reference
    at(std::initializer_list<int64_t> indices) const {
        validate_indices(indices);
        return at(std::span(indices));
    }

    // Empty
    /**
     * @brief Check if the tensor is empty (has no elements)
     *
     * @return true if the tensor is empty, false otherwise
     */
    BREZEL_NODISCARD bool empty() const noexcept {
        return m_storage == nullptr || numel() == 0;
    }

    // Element-wise operations
    /**
     * @brief Adds another tensor element-wise to the current tensor
     *
     * @param other Other tensor to add
     * @return Tensor Result of the addition
     */
    BREZEL_NODISCARD Tensor add(const Tensor& other) const {
        return binary_op(other, std::plus<T>());
    }

    /**
     * @brief Subtracts another tensor element-wise from the current tensor
     *
     * @param other Other tensor to subtract
     * @return Tensor Result of the subtraction
     */
    BREZEL_NODISCARD Tensor subtract(const Tensor& other) const {
        return binary_op(other, std::minus<T>());
    }

    /**
     * @brief Multiplies another tensor element-wise with the current tensor
     *
     * @param other Other tensor to multiply
     * @return Tensor Result of the multiplication
     */
    BREZEL_NODISCARD Tensor multiply(const Tensor& other) const {
        return binary_op(other, std::multiplies<T>());
    }

    /**
     * @brief Divides another tensor element-wise with the current tensor
     *
     * @param other Other tensor to divide
     * @return Tensor Result of the division
     */
    BREZEL_NODISCARD Tensor divide(const Tensor& other) const {
        return binary_op(other, std::divides<T>());
    }

    /**
     * @brief Adds another tensor element-wise to the current tensor
     * in-place
     *
     * @param other Other tensor to add
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor pow(const Tensor& other) const {
        return binary_op(other, [](T a, T b) { return std::pow(a, b); });
    }

    /**
     * @brief Exponentiates the tensor element-wise
     *
     * @return Tensor Result of the exponentiation
     */
    BREZEL_NODISCARD Tensor exp() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::exp(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the natural logarithm of the tensor element-wise
     *
     * @return Tensor Result of the logarithm
     */
    BREZEL_NODISCARD Tensor log() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::log(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the square root of the tensor element-wise
     *
     * @return Tensor Result of the square root
     */
    BREZEL_NODISCARD Tensor sqrt() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::sqrt(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the absolute value of the tensor element-wise
     *
     * @return Tensor Result of the absolute value
     */
    BREZEL_NODISCARD Tensor abs() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::abs(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the sign of the tensor element-wise
     *
     * @return Tensor Result of the sign
     */
    BREZEL_NODISCARD Tensor sign() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = data()[i] < T(0) ? T(-1) : T(1);
            });

        return result;
    }

    /**
     * @brief Computes the floor of the tensor element-wise
     *
     * @return Tensor Result of the floor
     */
    BREZEL_NODISCARD Tensor floor() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::floor(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the ceil of the tensor element-wise
     *
     * @return Tensor Result of the ceil
     */
    BREZEL_NODISCARD Tensor ceil() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::ceil(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the round of the tensor element-wise
     *
     * @return Tensor Result of the round
     */
    BREZEL_NODISCARD Tensor round() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::round(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the clip of the tensor element-wise
     *
     * @param min Minimum value
     * @param max Maximum value
     * @return Tensor Result of the clip
     */
    BREZEL_NODISCARD Tensor clip(T min, T max) const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    if (data()[i] < min)
                        result.data()[i] = min;

                    else if (data()[i] > max)
                        result.data()[i] = max;

                    else
                        result.data()[i] = data()[i];
                }
            });

        return result;
    }

    /**
     * @brief Computes the sine of the tensor element-wise
     *
     * @return Tensor Result of the sine
     */
    BREZEL_NODISCARD Tensor sin() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::sin(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the cosine of the tensor element-wise
     *
     * @return Tensor Result of the cosine
     */
    BREZEL_NODISCARD Tensor cos() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::cos(data()[i]);
            });

        return result;
    }

    /**
     * @brief Computes the tangent of the tensor element-wise
     *
     * @return Tensor Result of the tangent
     */
    BREZEL_NODISCARD Tensor tan() const {
        Tensor result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = std::tan(data()[i]);
            });

        return result;
    }

    // In-place operations
    /**
     * @brief Adds another tensor element-wise to the current tensor
     * in-place
     *
     * @param other Other tensor to add
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor& add_(const Tensor& other) {
        return binary_op_inplace(other, std::plus<T>());
    }

    /**
     * @brief Subtracts another tensor element-wise from the current tensor
     * in-place
     *
     * @param other Other tensor to subtract
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor& subtract_(const Tensor& other) {
        return binary_op_inplace(other, std::minus<T>());
    }

    /**
     * @brief Multiplies another tensor element-wise with the current tensor
     * in-place
     *
     * @param other Other tensor to multiply
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor& multiply_(const Tensor& other) {
        return binary_op_inplace(other, std::multiplies<T>());
    }

    /**
     * @brief Divides another tensor element-wise with the current tensor
     * in-place
     *
     * @param other Other tensor to divide
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor& divide_(const Tensor& other) {
        return binary_op_inplace(other, std::divides<T>());
    }

    /**
     * @brief Raises the current tensor to the power of another tensor
     * element-wise in-place
     *
     * @param other Other tensor to raise to
     * @return Tensor& Reference to the current tensor
     */
    BREZEL_NODISCARD Tensor& pow_(const Tensor& other) {
        return binary_op_inplace(other,
                                 [](T a, T b) { return std::pow(a, b); });
    }

    /**
     * @brief Fills the tensor with a given value
     *
     * @param value Value to fill the tensor with
     */
    BREZEL_NODISCARD Tensor& fill_(T value) {
        const size_t n = numel();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<size_t>& range) {
                              for (size_t i = range.begin(); i < range.end();
                                   ++i)
                                  data()[i] = value;
                          });

        return *this;
    }

    // Tensor slicing
    BREZEL_NODISCARD Tensor operator[](std::span<const int64_t> indices) const {
        return at(indices);
    }

    BREZEL_NODISCARD Tensor
    operator[](std::initializer_list<int64_t> indices) const {
        return at(indices);
    }

    BREZEL_NODISCARD Tensor narrow(int64_t dim, int64_t start,
                                   int64_t length) const {
        BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                      "Dimension out of bounds");

        BREZEL_ENSURE(start >= 0 && start < m_shape[dim],
                      "Start index out of bounds");

        BREZEL_ENSURE(length > 0 && start + length <= m_shape[dim],
                      "Length out of bounds");

        Shape new_shape = m_shape;
        new_shape[dim] = length;

        Tensor result(new_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % new_shape[j];
                        temp /= new_shape[j];
                    }

                    indices[dim] += start;
                    result.data()[i] = at(indices);
                }
            });

        return result;
    }

    BREZEL_NODISCARD Tensor unfold(int64_t dim, int64_t size,
                                   int64_t step) const {
        BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                      "Dimension out of bounds");

        BREZEL_ENSURE(size > 0, "Size must be greater than zero");
        BREZEL_ENSURE(step > 0, "Step must be greater than zero");

        Shape new_shape = m_shape;
        new_shape[dim] = size;
        new_shape.push_back((m_shape[dim] - size) / step + 1);

        Tensor result(new_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, result.numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % new_shape[j];
                        temp /= new_shape[j];
                    }

                    indices[dim] = indices[dim] * step + indices[ndim()];
                    result.data()[i] = at(indices);
                }
            });

        return result;
    }

    // Reduction operations
    /**
     * @brief Sums all elements of the tensor
     *
     * @param dim Dimension to sum along
     * @return Tensor Result of the sum
     */
    BREZEL_NODISCARD Tensor sum(int64_t dim = -1) const {
        return reduce(dim, std::plus<T>(), T(0));
    }

    /**
     * @brief Computes the mean of all elements of the tensor
     *
     * @param dim Dimension to compute the mean along
     * @return Tensor Result of the mean
     */
    BREZEL_NODISCARD Tensor mean(int64_t dim = -1) const {
        auto sum_result = sum(dim);
        if (dim == -1)
            return sum_result.divide(Tensor({1}, T(numel())));

        return sum_result.divide(Tensor({1}, T(m_shape[dim])));
    }

    /**
     * @brief Computes the maximum element of the tensor
     *
     * @param dim Dimension to compute the maximum along
     * @return Tensor Result of the maximum
     */
    BREZEL_NODISCARD Tensor max(int64_t dim = -1) const {
        return reduce(
            dim, [](T a, T b) { return std::max(a, b); },
            std::numeric_limits<T>::lowest());
    }

    /**
     * @brief Computes the minimum element of the tensor
     *
     * @param dim Dimension to compute the minimum along
     * @return Tensor Result of the minimum
     */
    BREZEL_NODISCARD Tensor min(int64_t dim = -1) const {
        return reduce(
            dim, [](T a, T b) { return std::min(a, b); },
            std::numeric_limits<T>::max());
    }

    // Shape operations
    /**
     * @brief Reshapes the tensor to a new shape
     *
     * @param shape New shape
     * @return Tensor Result of the reshape
     * @throw LogicError if the new shape has a different number of elements
     */
    BREZEL_NODISCARD Tensor reshape(const Shape& shape) const {
        BREZEL_ENSURE(shape.numel() == numel(),
                      "New shape must have the same number of elements");

        if (is_contiguous()) {
            Tensor result = *this;

            result.m_shape = shape;
            result.m_strides = Strides(shape);

            return result;
        }

        return contiguous().reshape(shape);
    }

    /**
     * @brief Squeezes the tensor by removing dimensions of size 1
     *
     * @param dim Dimension to squeeze
     * @return Tensor Result of the squeeze
     * @throw LogicError if the dimension is out of bounds
     * @throw LogicError if the dimension is not 1
     */
    BREZEL_NODISCARD Tensor squeeze(int64_t dim = -1) const {
        boost::container::small_vector<int64_t, 4> new_dim;

        if (dim == -1) {
            for (size_t i = 0; i < ndim(); ++i) {
                if (m_shape[i] != 1)
                    new_dim.push_back(m_shape[i]);
            }
        } else {
            BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                          "Dimension out of bounds");

            BREZEL_ENSURE(m_shape[dim] == 1, "Can only squeeze dimension of 1");

            for (size_t i = 0; i < ndim(); ++i) {
                if (i != static_cast<size_t>(dim))
                    new_dim.push_back(m_shape[i]);
            }
        }

        return reshape(Shape(new_dim));
    }

    /**
     * @brief Unsqueezes the tensor by adding dimensions of size 1
     *
     * @param dim Dimension to unsqueeze
     * @return Tensor Result of the unsqueeze
     * @throw LogicError if the dimension is out of bounds
     */
    BREZEL_NODISCARD Tensor unsqueeze(int64_t dim) const {
        BREZEL_ENSURE(dim >= 0 && dim <= static_cast<int64_t>(ndim()),
                      "Invalid dimensions for unsqueeze");

        boost::container::small_vector<int64_t, 4> new_dim;
        new_dim.reserve(ndim() + 1);

        for (size_t i = 0; i < ndim(); ++i) {
            if (i == static_cast<size_t>(dim))
                new_dim.push_back(1);

            if (i < ndim())
                new_dim.push_back(m_shape[i]);
        }

        return reshape(Shape(new_dim));
    }

    /**
     * @brief Broadcasts the tensor to a new shape
     *
     * @param target_shape Target shape to broadcast to
     * @return Tensor Result of the broadcast
     * @throw LogicError if the number of elements is different
     * @throw LogicError if the dimensions are not broadcastable
     */
    BREZEL_NODISCARD Tensor broadcast_to(const Shape& target_shape) const {
        if (!m_shape.is_broadcastable_with(target_shape)) {
            throw core::error::LogicError(
                "Shapes {} and {} cannot be broadcast together",
                m_shape.to_string(), target_shape.to_string());
        }

        if (m_shape == target_shape) {
            return *this;
        }

        Tensor result(target_shape);
        const size_t n = result.numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> target_indices(
                        target_shape.size());
                    size_t temp = i;
                    for (int64_t j = target_shape.size() - 1; j >= 0; --j) {
                        target_indices[j] = temp % target_shape[j];
                        temp /= target_shape[j];
                    }

                    boost::container::small_vector<int64_t, 4> source_indices(
                        m_shape.size());
                    for (size_t j = 0; j < m_shape.size(); ++j) {
                        source_indices[j] =
                            m_shape[j] == 1 ? 0 : target_indices[j];
                    }

                    result.data()[i] = at(source_indices);
                }
            });

        return result;
    }

    /**
     * @brief Expands the tensor by adding dimensions of size 1
     *
     * @param dim Dimension to expand
     * @return Tensor Result of the expand
     * @throw LogicError if the dimension is out of bounds
     */
    BREZEL_NODISCARD Tensor expand_dims(int64_t dim) const {
        BREZEL_ENSURE(dim >= 0 && dim <= static_cast<int64_t>(ndim()),
                      "Invalid dimensions for expand_dims");

        boost::container::small_vector<int64_t, 4> new_dim;
        new_dim.reserve(ndim() + 1);

        for (size_t i = 0; i < ndim(); ++i) {
            if (i == static_cast<size_t>(dim))
                new_dim.push_back(1);

            new_dim.push_back(m_shape[i]);
        }

        return reshape(Shape(new_dim));
    }

    /**
     * @brief Slices the tensor along the given indices
     *
     * @param indices Array of index pairs for each dimension
     * @return Tensor Result of the slice
     * @throw LogicError if the number of indices is different from the
     * number of dimensions
     * @throw LogicError if the indices are out of bounds
     */
    BREZEL_NODISCARD Tensor
    slice(std::span<const std::pair<int64_t, int64_t>> indices) const {
        BREZEL_ENSURE(indices.size() == ndim(),
                      "Expected {} indices but got {}", ndim(), indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            BREZEL_ENSURE(
                indices[i].first >= 0 && indices[i].second < m_shape[i],
                "Index {} out of bounds for dimension {}", indices[i], i);
        }

        Shape new_shape = m_shape;
        for (size_t i = 0; i < indices.size(); ++i)
            new_shape[i] = indices[i].second - indices[i].first;

        Tensor result(new_shape);
        const size_t n = result.numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> new_indices(
                        ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        new_indices[j] = temp % new_shape[j];
                        temp /= new_shape[j];
                    }

                    boost::container::small_vector<int64_t, 4> old_indices(
                        ndim());
                    for (size_t j = 0; j < ndim(); ++j)
                        old_indices[j] = new_indices[j] + indices[j].first;

                    result.data()[i] = at(old_indices);
                }
            });

        return result;
    }

    // Data type conversion
    /**
     * @brief Converts the tensor to a different data type
     *
     * @tparam U New data type
     * @return Tensor<U> Resulting tensor
     */
    template <TensorScalar U>
    BREZEL_NODISCARD Tensor<U> to() const {
        Tensor<U> result(m_shape);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    result.data()[i] = static_cast<U>(data()[i]);
            });

        return result;
    }

    // Linear-algebra operations
    /**
     * @brief Transposes the tensor along the given dimensions
     *
     * @param dim0 First dimension to transpose
     * @param dim1 Second dimension to transpose
     * @return Tensor Result of the transpose
     * @throw LogicError if the dimensions are out of bounds
     */
    BREZEL_NODISCARD Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const {
        BREZEL_ENSURE(dim0 < ndim() && dim1 < ndim(),
                      "Transpose dimensions out of range");

        Shape new_shape = m_shape;
        std::swap(new_shape[dim0], new_shape[dim1]);

        Tensor result(new_shape);
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

                    std::swap(indices[dim0], indices[dim1]);
                    result.data()[result.m_strides.get_linear_index(indices)] =
                        data()[i];
                }
            });

        return result;
    }

    /**
     * @brief Matrix multiplication with another tensor
     *
     * @param other Other tensor to multiply
     * @return Tensor Result of the matrix multiplication
     * @throw LogicError if the tensors are not 2D
     * @throw LogicError if the matrix dimensions do not match
     */
    BREZEL_NODISCARD Tensor matmul(const Tensor& other) const {
        BREZEL_ENSURE(ndim() >= 2 && other.ndim() >= 2,
                      "Matrix multiplication requires at least 2D tensors");
        BREZEL_ENSURE(m_shape[ndim() - 1] == other.m_shape[other.ndim() - 2],
                      "Matrix dimensions do not match for multiplication");

        const size_t M = m_shape[ndim() - 2];
        const size_t N = other.m_shape[other.ndim() - 1];
        const size_t K = m_shape[ndim() - 1];

        Shape result_shape = m_shape;
        result_shape[ndim() - 1] = N;
        Tensor result(result_shape, T(0));

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, M),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        T sum = T(0);

                        for (size_t k = 0; k < K; ++k)
                            sum += data()[i * K + k] * other.data()[k * N + j];

                        result.data()[i * N + j] = sum;
                    }
                }
            });

        return result;
    }

    /**
     * @brief Computes the dot product with another 1D tensor
     *
     * @param other Other tensor to compute the dot product with
     * @return Tensor Result of the dot product
     * @throw LogicError if the tensors are not 1D
     * @throw LogicError if the tensors are not the same size
     */
    BREZEL_NODISCARD Tensor dot(const Tensor& other) const {
        BREZEL_ENSURE(ndim() == 1 && other.ndim() == 1,
                      "Dot product requires 1D tensors");

        BREZEL_ENSURE(numel() == other.numel(),
                      "Dot product requires tensors of the same size");

        const size_t n = numel();
        T result = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize), T(0),
            [&](const tbb::blocked_range<size_t>& range, T init) {
                for (size_t i = range.begin(); i < range.end(); ++i)
                    init += data()[i] * other.data()[i];

                return init;
            },
            std::plus<T>());

        return Tensor({1}, result);
    }

    /**
     * @brief Computes the inverse of a square matrix
     *
     * @return Tensor Result of the matrix inversion
     * @throw LogicError if the tensor is not 2D or square
     */
    BREZEL_NODISCARD Tensor inverse() const {
        BREZEL_ENSURE(ndim() == 2, "Matrix inverse requires a 2D tensor");
        BREZEL_ENSURE(m_shape[0] == m_shape[1], "Matrix must be square");

        const size_t n = m_shape[0];
        Tensor result(m_shape);
        Tensor lu = *this;
        std::vector<int64_t> pivots(n);

        for (size_t k = 0; k < n; k++) {
            pivots[k] = k;

            T max_val = std::abs(lu.data()[k * n + k]);
            size_t max_idx = k;

            for (size_t i = k + 1; i < n; i++) {
                T val = std::abs(lu.data()[i * n + k]);

                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            if (max_idx != k) {
                std::swap(pivots[k], pivots[max_idx]);

                for (size_t j = 0; j < n; j++) {
                    std::swap(lu.data()[k * n + j], lu.data()[max_idx * n + j]);
                }
            }

            BREZEL_ENSURE(std::abs(lu.data()[k * n + k]) > T(1e-10),
                          "Matrix is singular");

            for (size_t i = k + 1; i < n; i++) {
                lu.data()[i * n + k] /= lu.data()[k * n + k];

                for (size_t j = k + 1; j < n; j++) {
                    lu.data()[i * n + j] -=
                        lu.data()[i * n + k] * lu.data()[k * n + j];
                }
            }
        }

        for (size_t j = 0; j < n; j++) {
            boost::container::small_vector<T, 4> b(n, T(0));
            b[j] = T(1);

            for (size_t i = 0; i < n; i++) {
                T sum = b[pivots[i]];

                for (size_t k = 0; k < i; k++) {
                    sum -= lu.data()[i * n + k] * b[k];
                }

                b[i] = sum;
            }

            for (int64_t i = n - 1; i >= 0; i--) {
                T sum = b[i];

                for (size_t k = i + 1; k < n; k++) {
                    sum -= lu.data()[i * n + k] * result.data()[k * n + j];
                }

                result.data()[i * n + j] = sum / lu.data()[i * n + i];
            }
        }

        return result;
    }

    /**
     * @brief Computes the QR decomposition using Modified Gram-Schmidt process
     *
     * @details Decomposes matrix A into A = QR where:
     *          Q is orthogonal (Q^T Q = I)
     *          R is upper triangular
     *
     * @return std::pair<Tensor, Tensor> {Q, R} where Q is orthogonal and R is
     * upper triangular
     * @throw LogicError if tensor is not 2D
     */
    /**
     * @brief Computes the QR decomposition of a matrix using Classical
     * Gram-Schmidt
     * @return std::pair<Tensor, Tensor> {Q, R} where Q is orthogonal and R is
     * upper triangular
     */
    BREZEL_NODISCARD std::pair<Tensor, Tensor> qr() const {
        BREZEL_ENSURE(ndim() == 2, "QR decomposition requires a 2D tensor");

        const auto m = static_cast<int64_t>(m_shape[0]);  // rows
        const auto n = static_cast<int64_t>(m_shape[1]);  // cols

        // Initialize Q and R
        auto q = Tensor<T>(Shape({m, n}));  // Will hold orthonormal vectors
        auto r = Tensor<T>(Shape({n, n}), T(0));  // Upper triangular matrix

        // For tracking the current column vector being processed
        std::vector<T> v(m);

        // Process each column
        for (int64_t j = 0; j < n; ++j) {
            // Get current column from input matrix
            for (int64_t i = 0; i < m; ++i) {
                v[i] = data()[i * n + j];
            }

            // First compute R entries by projecting onto previous vectors
            for (int64_t k = 0; k < j; ++k) {
                // Compute r_kj = q_k^T * a_j
                T rkj = T(0);
                for (int64_t i = 0; i < m; ++i) {
                    rkj += q.data()[i * n + k] * v[i];
                }
                r.data()[k * n + j] = rkj;

                // Subtract projection from v
                for (int64_t i = 0; i < m; ++i) {
                    v[i] -= rkj * q.data()[i * n + k];
                }
            }

            // Compute the norm of the remaining vector
            T norm = T(0);
            for (int64_t i = 0; i < m; ++i) {
                norm += v[i] * v[i];
            }
            norm = std::sqrt(norm);

            // Check for linear dependence
            if (norm < T(1e-10)) {
                throw core::error::LogicError(
                    "Matrix is numerically rank deficient");
            }

            // Set diagonal entry in R
            r.data()[j * n + j] = norm;

            // Normalize the vector and store in Q
            T inv_norm = T(1) / norm;
            for (int64_t i = 0; i < m; ++i) {
                q.data()[i * n + j] = v[i] * inv_norm;
            }
        }

// Verify orthogonality of Q in debug mode
#ifdef BREZEL_DEBUG
        // Check Q^T * Q ≈ I
        auto qT = q.transpose();
        auto qTq = qT.matmul(q);
        auto I = Tensor<T>::eye(n);

        // Print Q matrix for debugging
        std::cout << "\nQ matrix:\n";
        q.print();
        std::cout << "\nQ^T * Q:\n";
        qTq.print();
        std::cout << "\nExpected I:\n";
        I.print();

        bool is_orthogonal = qTq.allclose(I, T(1e-5), T(1e-5));
        if (!is_orthogonal) {
            T max_deviation = T(0);
            for (int64_t i = 0; i < n * n; ++i) {
                T expected = (i / n == i % n) ? T(1) : T(0);
                T actual = qTq.data()[i];
                max_deviation =
                    std::max(max_deviation, std::abs(actual - expected));
            }
            std::cout << "\nMax deviation from identity: " << max_deviation
                      << "\n";
            throw core::error::LogicError(
                "Q is not orthogonal in QR decomposition");
        }

        // Verify Q * R = A
        auto qr_product = q.matmul(r);
        if (!qr_product.allclose(*this, T(1e-5), T(1e-5))) {
            throw core::error::LogicError(
                "QR decomposition failed numerical validation");
        }
#endif

        return std::make_pair(std::move(q), std::move(r));
    }

    /**
     * @brief Computes the matrix / vector norm
     *
     * @param ord Order of the norm (1, 2, or inf)
     * @return Tensor Result of the norm
     */
    BREZEL_NODISCARD Tensor norm(int64_t ord = 2) const {
        BREZEL_ENSURE(ndim() <= 2,
                      "Norm only supported for vectors and matrices");

        if (ndim() == 1) {
            if (ord == 1)
                return abs().sum();

            else if (ord == 2)
                return sqrt((multiply(*this)).sum());

            else if (ord == std::numeric_limits<int64_t>::max())
                return abs().max();
        }

        if (ord == 1)
            return abs().sum(0).max();

        else if (ord == 2) {
            auto [u, s, v] = svd();
            return s[0];
        } else if (ord == std::numeric_limits<int64_t>::max())
            return abs().max(0).max();

        BREZEL_ENSURE(false, "Unsupported norm order");
        return Tensor();
    }

    /**
     * @brief Computes the Singular Value Decomposition (SVD) of a matrix
     * @details Decomposes matrix A into U * Σ * V^T where:
     *          - U is orthogonal matrix of left singular vectors
     *          - Σ is diagonal matrix of singular values
     *          - V^T is transpose of orthogonal matrix of right singular
     * vectors
     *
     * @return std::tuple<Tensor,Tensor,Tensor> {U, Σ, V}
     * @throw LogicError if tensor is not 2D
     */
    BREZEL_NODISCARD std::tuple<Tensor, Tensor, Tensor> svd() const {
        BREZEL_ENSURE(ndim() == 2, "SVD requires 2D tensor");

        const size_t m = m_shape[0];
        const size_t n = m_shape[1];
        const size_t k = std::min(m, n);

        Tensor u = *this;
        Tensor s({k}, T(0));
        Tensor v({n, n}, T(0));
        boost::container::small_vector<T, 4> d(n);
        boost::container::small_vector<T, 4> e(n);

        for (size_t i = 0; i < n; i++) {
            T scale = 0;
            T h = 0;

            for (size_t j = i; j < m; j++) {
                scale = std::max(scale, std::abs(u.data()[j * n + i]));
            }

            if (scale > 0) {
                for (size_t j = i; j < m; j++) {
                    u.data()[j * n + i] /= scale;
                    h += u.data()[j * n + i] * u.data()[j * n + i];
                }

                T f = u.data()[i * n + i];
                T g = std::sqrt(h);
                if (f > 0)
                    g = -g;

                d[i] = g * scale;
                h = h - f * g;
                u.data()[i * n + i] = f - g;

                for (size_t j = i + 1; j < n; j++) {
                    T f = 0;
                    for (size_t k = i; k < m; k++) {
                        f += u.data()[k * n + i] * u.data()[k * n + j];
                    }
                    f = f / h;

                    for (size_t k = i; k < m; k++) {
                        u.data()[k * n + j] -= f * u.data()[k * n + i];
                    }
                }

                for (size_t j = i; j < m; j++) {
                    u.data()[j * n + i] *= scale;
                }
            } else {
                d[i] = u.data()[i * n + i];
            }

            if (i < n - 1) {
                scale = 0;
                h = 0;

                for (size_t j = i + 1; j < n; j++) {
                    scale = std::max(scale, std::abs(u.data()[i * n + j]));
                }

                if (scale > 0) {
                    for (size_t j = i + 1; j < n; j++) {
                        u.data()[i * n + j] /= scale;
                        h += u.data()[i * n + j] * u.data()[i * n + j];
                    }

                    T f = u.data()[i * n + i + 1];
                    T g = std::sqrt(h);
                    if (f > 0)
                        g = -g;

                    e[i] = g * scale;
                    h = h - f * g;
                    u.data()[i * n + i + 1] = f - g;

                    for (size_t j = i + 1; j < n; j++) {
                        e[j] = u.data()[i * n + j];
                    }

                    for (size_t j = i + 1; j < m; j++) {
                        T f = 0;
                        for (size_t k = i + 1; k < n; k++) {
                            f += u.data()[j * n + k] * u.data()[i * n + k];
                        }

                        for (size_t k = i + 1; k < n; k++) {
                            u.data()[j * n + k] -= f * e[k] / h;
                        }
                    }

                    for (size_t j = i + 1; j < n; j++) {
                        u.data()[i * n + j] *= scale;
                    }
                } else {
                    e[i] = u.data()[i * n + i + 1];
                }
            }
        }

        for (size_t i = n - 1; i > 0; i--) {
            if (e[i - 1] != 0) {
                for (size_t j = i; j < n; j++) {
                    T f = 0;
                    for (size_t k = i; k < n; k++) {
                        f += v.data()[k * n + j] * u.data()[(i - 1) * n + k];
                    }
                    f = f / (u.data()[(i - 1) * n + i] * e[i - 1]);

                    for (size_t k = i - 1; k < n; k++) {
                        v.data()[k * n + j] += f * u.data()[(i - 1) * n + k];
                    }
                }
            }

            for (size_t j = i; j < n; j++) {
                v.data()[(i - 1) * n + j] = 0;
                v.data()[j * n + i - 1] = 0;
            }

            v.data()[(i - 1) * n + i - 1] = 1;
            if (d[i] != 0) {
                for (size_t j = i; j < n; j++) {
                    T f = 0;
                    for (size_t k = i; k < n; k++) {
                        f += v.data()[j * n + k] * v.data()[i * n + k];
                    }

                    for (size_t k = i; k < n; k++) {
                        v.data()[j * n + k] -= f * v.data()[i * n + k];
                    }
                }
            }

            for (size_t j = i; j < n; j++) {
                v.data()[i * n + j] = 0;
            }
            v.data()[i * n + i] = 1;
        }

        const size_t max_iter = 1000;
        const T eps = std::numeric_limits<T>::epsilon();

        for (size_t k = n - 1; k > 0; k--) {
            size_t iter = 0;

            while (iter < max_iter) {
                size_t l;
                for (l = k; l > 0; l--) {
                    if (std::abs(e[l - 1]) <=
                        eps * (std::abs(d[l - 1]) + std::abs(d[l]))) {
                        e[l - 1] = 0;
                        break;
                    }
                }

                if (l == k) {
                    break;
                }

                T c = 0;
                T s = 1;

                for (size_t i = l; i <= k; i++) {
                    T f = s * e[i - 1];
                    e[i - 1] = c * e[i - 1];

                    if (std::abs(f) <=
                        eps * (std::abs(d[i - 1]) + std::abs(d[i]))) {
                        break;
                    }

                    T g = d[i];
                    T h = std::sqrt(f * f + g * g);
                    d[i] = h;
                    c = g / h;
                    s = -f / h;

                    for (size_t j = 0; j < m; j++) {
                        T y = u.data()[j * n + l - 1];
                        T z = u.data()[j * n + i];
                        u.data()[j * n + l - 1] = y * c + z * s;
                        u.data()[j * n + i] = -y * s + z * c;
                    }
                }

                iter++;
            }

            BREZEL_ENSURE(iter < max_iter, "SVD did not converge");
        }

        for (size_t i = 0; i < k; i++) {
            s.data()[i] = std::abs(d[i]);
        }

        for (size_t i = 0; i < k - 1; i++) {
            size_t max_idx = i;
            T max_val = s.data()[i];

            for (size_t j = i + 1; j < k; j++) {
                if (s.data()[j] > max_val) {
                    max_val = s.data()[j];
                    max_idx = j;
                }
            }

            if (max_idx != i) {
                std::swap(s.data()[i], s.data()[max_idx]);

                for (size_t j = 0; j < m; j++) {
                    std::swap(u.data()[j * n + i], u.data()[j * n + max_idx]);
                }

                for (size_t j = 0; j < n; j++) {
                    std::swap(v.data()[j * n + i], v.data()[j * n + max_idx]);
                }
            }
        }

        return {u, s, v};
    }

    // Statistics
    /**
     * @brief Computes the standard deviation of the tensor
     *
     * @param dim Dimension to compute the standard deviation along
     * @return Tensor Result of the standard deviation
     */
    BREZEL_NODISCARD Tensor std(int64_t dim = -1) const {
        return var(dim).sqrt();
    }

    /**
     * @brief Computes the variance of the tensor
     *
     * @param dim Dimension to compute the variance along
     * @return Tensor Result of the variance
     */
    BREZEL_NODISCARD Tensor var(int64_t dim = -1) const {
        auto mean_result = mean(dim);
        auto diff = subtract(mean_result);
        auto squared_diff = diff.pow(Tensor({1}, T(2)));

        return squared_diff.mean(dim);
    }

    /**
     * @brief Computes the median of the tensor
     *
     * @param dim Dimension to compute the median along
     */
    BREZEL_NODISCARD Tensor median(int64_t dim = -1) const {
        Shape new_shape = m_shape;
        if (dim == -1) {
            new_shape = {1};
        } else {
            new_shape[dim] = 1;
        }

        Tensor result(new_shape);
        const size_t n = numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n / m_shape[dim]),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % m_shape[j];
                        temp /= m_shape[j];
                    }

                    std::sort(indices.begin(), indices.end());
                    result.data()[i] = at(indices);
                }
            });

        return result;
    }

    // View operations
    /**
     * @brief Creates a view of the tensor with the given indices
     *
     * @param indices Array of index pairs for each dimension
     * @return Tensor Result of the view
     * @throw LogicError if the number of indices is different from the
     * number of dimensions
     * @throw LogicError if the indices are out of bounds
     */
    BREZEL_NODISCARD Tensor view(const Shape& shape) const {
        BREZEL_ENSURE(shape.numel() == numel(),
                      "New shape must have the same number of elements");

        Tensor result(shape);
        const size_t n = result.numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % shape[j];
                        temp /= shape[j];
                    }

                    result.data()[i] = at(indices);
                }
            });

        return result;
    }

    /**
     * @brief Permutes the dimensions of the tensor
     *
     * @param dims New dimensions
     * @return Tensor Result of the permutation
     */
    BREZEL_NODISCARD Tensor permute(const Shape& dims) const {
        BREZEL_ENSURE(dims.size() == ndim(),
                      "Permute dimensions must match tensor dimensions");

        Shape new_shape = m_shape;
        Strides new_strides = m_strides;

        for (size_t i = 0; i < dims.size(); ++i) {
            new_shape[i] = m_shape[dims[i]];
            new_strides[i] = m_strides[dims[i]];
        }

        Tensor result(new_shape);
        const size_t n = numel();

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    boost::container::small_vector<int64_t, 4> indices(ndim());
                    size_t temp = i;

                    for (size_t j = 0; j < ndim(); ++j) {
                        indices[j] = temp % new_shape[j];
                        temp /= new_shape[j];
                    }

                    result.data()[i] = at(indices);
                }
            });

        return result;
    }

    // Utility functions
    /**
     * @brief Checks if two tensors are approximately equal within tolerance
     *
     * @param other Tensor to compare with
     * @param rtol Relative tolerance
     * @param atol Absolute tolerance
     * @return bool true if tensors are approximately equal
     */
    BREZEL_NODISCARD bool allclose(const Tensor& other, T rtol = T(1e-5),
                                   T atol = T(1e-8)) const {
        if (m_shape != other.m_shape)
            return false;

        if (empty() && other.empty())
            return true;

        if (empty() || other.empty())
            return false;

        const size_t n = numel();
        bool all_close = true;
        T max_diff = T(0);

        for (size_t i = 0; i < n; ++i) {
            const T a = data()[i];
            const T b = other.data()[i];
            const T abs_diff = std::abs(a - b);
            const T scale = std::max(std::abs(a), std::abs(b));
            const T tol = atol + rtol * scale;

            if (abs_diff > tol) {
                all_close = false;
                max_diff = std::max(max_diff, abs_diff);
#ifdef BREZEL_DEBUG
                std::cout << "\nNon-matching elements at index " << i << ":"
                          << "\nValue 1: " << a << "\nValue 2: " << b
                          << "\nAbsolute difference: " << abs_diff
                          << "\nTolerance: " << tol << "\n";
#endif
                break;
            }
        }

#ifdef BREZEL_DEBUG
        if (!all_close) {
            std::cout << "Maximum difference: " << max_diff << "\n";
        }
#endif

        return all_close;
    }

    /**
     * @brief Checks if any element satisfies the predicate
     *
     * @tparam Pred Predicate function type
     * @param pred Predicate function
     * @return bool true if any element satisfies the predicate, false otherwise
     */
    template <typename Pred>
    bool any(Pred pred) const {
        const size_t n = numel();

        return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize), false,
            [&](const tbb::blocked_range<size_t>& range, bool init) {
                if (init)
                    return true;

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    if (pred(data()[i]))
                        return true;
                }

                return false;
            },
            std::logical_or<bool>());
    }

    /**
     * @brief Checks if all elements satisfy the predicate
     *
     * @param pred Predicate function
     * @return bool True if all elements satisfy predicate
     */
    template <typename Pred>
    bool all(Pred pred) const {
        const size_t n = numel();

        return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize), true,
            [&](const tbb::blocked_range<size_t>& range, bool init) {
                if (!init)
                    return false;

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    if (!pred(data()[i]))
                        return false;
                }

                return true;
            },
            std::logical_and<bool>());
    }

    /**
     * @brief Checks if tensor contains any NaN values
     *
     * @return bool True if tensor contains NaN values
     */
    bool has_nan() const {
        return any([](T x) { return std::isnan(x); });
    }

    /**
     * @brief Checks if tensor contains any infinite values
     *
     * @return bool True if tensor contains infinite values
     */
    bool has_inf() const {
        return any([](T x) { return std::isinf(x); });
    }

    /**
     * @brief Return indices for maximum elements along specified dimensions
     *
     * @param dim Dimensions to reduce
     * @return Tensor indices of the maximum elements
     */
    BREZEL_NODISCARD Tensor argmax(int64_t dim = -1) const {
        if (dim == -1) {
            T max_val = data()[0];
            size_t max_idx = 0;
            const size_t n = numel();

            for (size_t i = 1; i < n; i++) {
                if (data()[i] > max_val) {
                    max_val = data()[i];
                    max_idx = i;
                }
            }

            return Tensor({1}, static_cast<T>(max_idx));
        }

        BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                      "Dimension out of bounds");

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
                                  T max_val = data()[i * dim_size];
                                  size_t max_idx = 0;

                                  for (size_t j = 1; j < dim_size; ++j) {
                                      if (data()[i * dim_size + j] > max_val) {
                                          max_val = data()[i * dim_size + j];
                                          max_idx = j;
                                      }
                                  }

                                  result.data()[i] = static_cast<T>(max_idx);
                              }
                          });

        return result;
    }

    /**
     * @brief Computes the determinant of a square matrix
     *
     * @details Uses LU decomposition for larger matrices
     *
     * @return T Result of the determinant calculation
     * @throw LogicError if the tensor is not a square matrix
     */
    BREZEL_NODISCARD T det() const {
        BREZEL_ENSURE(ndim() == 2, "Determinant requires a 2D tensor");
        BREZEL_ENSURE(m_shape[0] == m_shape[1], "Matrix must be squared");

        const size_t n = m_shape[0];
        if (n == 1)
            return data()[0];

        if (n == 2)
            return data()[0] * data()[3] - data()[1] * data()[2];

        if (n == 3)
            return data()[0] * (data()[4] * data()[8] - data()[5] * data()[7]) -
                   data()[1] * (data()[3] * data()[8] - data()[5] * data()[6]) +
                   data()[2] * (data()[3] * data()[7] - data()[4] * data()[6]);

        Tensor lu = *this;
        T det_val = T(1);
        std::vector<int64_t> pivots(n);

        for (size_t i = 0; i < n; i++)
            pivots[i] = i;

        // LU decomposition with partial pivoting
        for (size_t k = 0; k < n - 1; k++) {
            T max_val = std::abs(lu.data()[k * n + k]);
            size_t max_idx = k;

            for (size_t i = k + 1; i < n; i++) {
                T val = std::abs(lu.data()[i * n + k]);

                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            if (std::abs(max_val) < std::numeric_limits<T>::epsilon())
                return T(0);

            if (max_idx != k) {
                for (size_t j = 0; j < n; j++)
                    std::swap(lu.data()[k * n + j], lu.data()[max_idx * n + j]);

                std::swap(pivots[k], pivots[max_idx]);
                det_val = -det_val;  // Change sign when we swap rows
            }

            for (size_t i = k + 1; i < n; i++) {
                T mult = lu.data()[i * n + k] / lu.data()[k * n + k];
                lu.data()[i * n + k] = mult;

                for (size_t j = k + 1; j < n; j++)
                    lu.data()[i * n + j] -= mult * lu.data()[k * n + j];
            }
        }

        for (size_t i = 0; i < n; i++)
            det_val *= lu.data()[i * n + i];

        return det_val;
    }

    /**
     * @brief Checks if the matrix is singular (determinant is zero)
     *
     * @return bool True if matrix is singular
     */
    BREZEL_NODISCARD bool is_singular() const {
        return std::abs(det()) < std::numeric_limits<T>::epsilon();
    }

    // Accessors
    /**
     * @brief Gets the raw data pointer
     *
     * @return pointer Raw data pointer
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE pointer data() noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Gets the raw data pointer
     *
     * @return const_pointer Raw data pointer
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const_pointer data() const noexcept {
        return m_storage ? m_storage->data() : nullptr;
    }

    /**
     * @brief Gets the shape of the tensor
     *
     * @return const Shape& Shape of the tensor
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const noexcept {
        return m_shape;
    }

    /**
     * @brief Gets the strides of the tensor
     *
     * @return const Strides& Strides of the tensor
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Strides& strides()
        const noexcept {
        return m_strides;
    }

    /**
     * @brief Gets the number of dimensions of the tensor
     *
     * @return size_t Number of dimensions
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t ndim() const noexcept {
        return m_shape.size();
    }

    /**
     * @brief Gets the total number of elements in the tensor
     *
     * @return size_t Number of elements
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const noexcept {
        return m_shape.numel();
    }

    /**
     * @brief Checks if the tensor is contiguous in memory
     *
     * @return true if contiguous, false otherwise
     */
    BREZEL_NODISCARD bool is_contiguous() const noexcept {
        return m_strides.is_contiguous(m_shape);
    }

    /**
     * @brief Makes the tensor contiguous in memory
     *
     * @return Tensor Contiguous tensor copy
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

    // Debug operations
    /**
     * @brief Formats a floating point number for display
     *
     * @param value Value to format
     * @param precision Decimal precision
     * @return std::string Formatted string
     */
    BREZEL_NODISCARD static std::string format_float(T value,
                                                     int precision = 4) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    /**
     * @brief Returns a detailed string representation of the tensor
     *
     * @param max_items Maximum number of items to show per dimension
     * @return String representation
     */
    std::string repr(size_t max_items = 10) const {
        std::stringstream ss;

        ss << "Tensor<" << typeid(T).name() << ">(";
        ss << "shape=[";

        for (size_t i = 0; i < ndim(); ++i) {
            ss << m_shape[i];
            if (i < ndim() - 1)
                ss << ", ";
        }

        ss << "], ";
        ss << "strides=[";

        for (size_t i = 0; i < ndim(); ++i) {
            ss << m_strides[i];
            if (i < ndim() - 1)
                ss << ", ";
        }

        ss << "], ";

        if (numel() == 0)
            ss << "[]";
        else if (ndim() == 0)
            ss << Tensor::format_float(data()[0]);
        else
            print_tensor_data(ss, 0, std::vector<int64_t>(), max_items);

        ss << "\nMemory: " << (numel() * sizeof(T)) / 1024.0 << " KB, ";
        ss << "Contiguous: " << (is_contiguous() ? "True" : "False") << ")";

        return ss.str();
    }

    /**
     * @brief Prints statistical information about the tensor
     *
     */
    void describe() const {
        std::cout << "\nTensor Statistics:";
        std::cout << "\n------------------";
        std::cout << "\nShape: " << shape_string();
        std::cout << "\nDtype: " << typeid(T).name();
        std::cout << "\nSize: " << numel() << " elements";
        std::cout << "\nMemory: " << (numel() * sizeof(T)) / 1024.0 << " KB";

        if (numel() > 0) {
            auto min_val = min().data()[0];
            auto max_val = max().data()[0];
            auto mean_val = mean().data()[0];
            auto std_val = std().data()[0];

            std::cout << "\nMin: " << format_float(min_val);
            std::cout << "\nMax: " << format_float(max_val);
            std::cout << "\nMean: " << format_float(mean_val);
            std::cout << "\nStd: " << format_float(std_val);

            if (numel() < 1000)
                print_histogram();
        }

        std::cout << std::endl;
    }

    /**
     * @brief Prints the tensor to the standard output with formatting
     *
     * @param max_items Maximum number of items to show per dimension
     */
    void print(size_t max_items = 10) const {
        std::cout << repr(max_items) << std::endl;
    }

    /**
     * @brief Converts tensor to string with minimal formatting
     */
    std::string to_string() const {
        return repr(std::numeric_limits<size_t>::max());
    }

    // Comparison operator
    BREZEL_NODISCARD bool operator==(const Tensor& other) const {
        if (this == &other)
            return true;

        if (m_shape != other.m_shape)
            return false;

        if (numel() != other.numel())
            return false;

        if (empty() && other.empty())
            return true;

        if (empty() || other.empty())
            return false;

        const size_t n = numel();
        return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, n, kDefaultBlockSize), true,
            [&](const tbb::blocked_range<size_t>& range, bool init) {
                if (!init)
                    return false;

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    if (data()[i] != other.data()[i])
                        return false;
                }

                return true;
            },
            std::logical_and<bool>());
    }

    // Arithmetic operators
    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return subtract(other); }
    Tensor operator*(const Tensor& other) const { return multiply(other); }
    Tensor operator/(const Tensor& other) const { return divide(other); }

    Tensor& operator+=(const Tensor& other) { return add_(other); }
    Tensor& operator-=(const Tensor& other) { return subtract_(other); }
    Tensor& operator*=(const Tensor& other) { return multiply_(other); }
    Tensor& operator/=(const Tensor& other) { return divide_(other); }

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

    // Helper functions
    /**
     * @brief Binary opetaro for element-wise operations
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor to operate on
     * @param op Operation to perform
     * @return Tensor Result of the operation
     */
    template <typename BinaryOp>
    BREZEL_NODISCARD Tensor binary_op(const Tensor& other, BinaryOp op) const {
        if (m_shape == other.m_shape) {
            Tensor result(m_shape);
            const size_t n = numel();

            if (is_contiguous() && other.is_contiguous()) {
#if defined(BREZEL_SIMD_AVX512)
                simd_avx512_op(other, result, op);
#elif defined(BREZEL_SIMD_AVX2)
                simd_avx2_op(other, result, op);
#else
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i < range.end(); ++i)
                            result.data()[i] = op(data()[i], other.data()[i]);
                    });
#endif
            } else {
                for (size_t i = 0; i < n; ++i)
                    result.data()[i] = op(data()[i], other.data()[i]);
            }

            return result;
        }

        auto broadcasted = m_shape.broadcast_with(other.m_shape);
        auto lhs_broadcast = broadcast_to(broadcasted);
        auto rhs_broadcast = other.broadcast_to(broadcasted);

        return lhs_broadcast.binary_op(rhs_broadcast, op);
    }

    /**
     * @brief Binary op for in-place element-wise operations
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor to operate on
     * @param op Operation to perform
     * @return Tensor& Reference to the current tensor
     */
    template <typename BinaryOp>
    Tensor& binary_op_inplace(const Tensor& other, BinaryOp op) {
        BREZEL_ENSURE(m_shape == other.m_shape ||
                          m_shape.is_broadcastable_with(other.m_shape),
                      "Incompatible shapes for in-place operation");

        if (m_shape == other.m_shape) {
            const size_t n = numel();

            if (is_contiguous() && other.is_contiguous()) {
                parallel_simd_op(other, *this, op);
            } else {
                for (size_t i = 0; i < n; ++i)
                    data()[i] = op(data()[i], other.data()[i]);
            }
        } else {
            *this = binary_op(other, op);
        }

        return *this;
    }

    /**
     * @brief Reduce operation for element-wise operations
     *
     * @tparam ReduceOp Reduce operation type
     * @param dim Dimension to reduce along
     * @param op Operation to perform
     * @param init Initial value
     * @return Tensor Result of the reduction
     * @throw LogicError if the dimension is out of bounds
     */
    template <typename ReduceOp>
    BREZEL_NODISCARD Tensor reduce(int64_t dim, ReduceOp op, T init) const {
        if (dim == -1) {
            T result = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, numel(), kDefaultBlockSize), init,
                [&](const tbb::blocked_range<size_t>& range, T local_init) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        local_init = op(local_init, data()[i]);
                    }
                    return local_init;
                },
                op);

            return Tensor({1}, result);
        }

        BREZEL_ENSURE(dim >= 0 && dim < static_cast<int64_t>(ndim()),
                      "Dimension out of bounds");

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
                                  T acc = init;

                                  for (size_t j = 0; j < dim_size; ++j)
                                      acc = op(acc, data()[i * dim_size + j]);

                                  result.data()[i] = acc;
                              }
                          });

        return result;
    }

#if defined(BREZEL_SIMD_AVX512)
    /**
     * @brief SIMD implementation for AVX-512
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor to operate on
     * @param result Result tensor
     * @param op Operation to perform
     * @return Tensor Result of the operation
     */
    template <typename BinaryOp>
    void simd_avx512_op(const Tensor& other, Tensor& result,
                        BinaryOp op) const {
        if constexpr (std::is_same_v<T, float>) {
            constexpr size_t simd_width = 16;
            const size_t n = numel();
            const size_t simd_size = n - (n % simd_width);

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, simd_size, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end();
                         i += simd_width) {
                        __m512 a = _mm512_load_ps(&data()[i]);
                        __m512 b = _mm512_load_ps(&other.data()[i]);
                        __m512 c;

                        if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                            c = _mm512_add_ps(a, b);
                        } else if constexpr (std::is_same_v<BinaryOp,
                                                            std::minus<T>>) {
                            c = _mm512_sub_ps(a, b);
                        } else if constexpr (std::is_same_v<
                                                 BinaryOp,
                                                 std::multiplies<T>>) {
                            c = _mm512_mul_ps(a, b);
                        } else if constexpr (std::is_same_v<BinaryOp,
                                                            std::divides<T>>) {
                            c = _mm512_div_ps(a, b);
                        }

                        _mm512_store_ps(&result.data()[i], c);
                    }
                });

            // Handle remaining elements
            for (size_t i = simd_size; i < n; ++i) {
                result.data()[i] = op(data()[i], other.data()[i]);
            }
        }
    }
#endif

#if defined(BREZEL_SIMD_AVX2)
    /**
     * @brief SIMD implementation for AVX2
     *
     * @tparam BinaryOp Binary operation type
     * @param other Other tensor to operate on
     * @param result Result tensor
     * @param op Operation to perform
     * @return Tensor Result of the operation
     */
    template <typename BinaryOp>
    void simd_avx2_op(const Tensor& other, Tensor& result, BinaryOp op) const {
        if constexpr (std::is_same_v<T, float>) {
            constexpr size_t simd_width = 8;
            const size_t n = numel();
            const size_t simd_size = n - (n % simd_width);

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, simd_size, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end();
                         i += simd_width) {
                        __m256 a = _mm256_load_ps(&data()[i]);
                        __m256 b = _mm256_load_ps(&other.data()[i]);
                        __m256 c;

                        if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                            c = _mm256_add_ps(a, b);
                        } else if constexpr (std::is_same_v<BinaryOp,
                                                            std::minus<T>>) {
                            c = _mm256_sub_ps(a, b);
                        } else if constexpr (std::is_same_v<
                                                 BinaryOp,
                                                 std::multiplies<T>>) {
                            c = _mm256_mul_ps(a, b);
                        } else if constexpr (std::is_same_v<BinaryOp,
                                                            std::divides<T>>) {
                            c = _mm256_div_ps(a, b);
                        }

                        _mm256_store_ps(&result.data()[i], c);
                    }
                });

            // Handle remaining elements
            for (size_t i = simd_size; i < n; ++i) {
                result.data()[i] = op(data()[i], other.data()[i]);
            }
        }
    }
#endif

    /**
     * @brief Validates the given indices
     *
     * @param indices Array of indices to validate
     * @throw LogicError if the number of indices is different from the
     * number of dimensions
     * @throw LogicError if the indices are out of bounds
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

    /**
     * @brief Helper function to print tensor data recursively
     *
     */
    void print_tensor_data(std::stringstream& ss, size_t dim,
                           std::vector<int64_t> indices,
                           size_t max_items) const {
        if (dim == ndim()) {
            ss << Tensor::format_float(at(std::span<const int64_t>(indices)));
            return;
        }

        ss << "[";
        size_t size_to_print = m_shape[dim];
        const bool truncated = size_to_print > 2 * max_items;

        if (truncated) {
            size_to_print = max_items;
        }

        for (size_t i = 0; i < size_to_print; ++i) {
            indices.push_back(static_cast<int64_t>(i));
            print_tensor_data(ss, dim + 1, indices, max_items);
            indices.pop_back();

            if (i < size_to_print - 1)
                ss << ", ";

            if (truncated && i == size_to_print - 1)
                ss << ", ...";
        }

        ss << "]";
        if (dim == 0)
            ss << "\n";
    }

    /**
     * @brief Prints a histogram of tensor values
     */
    void print_histogram(size_t bins = 10) const {
        auto min_val = min().data()[0];
        auto max_val = max().data()[0];
        auto range = max_val - min_val;
        auto bin_width = range / bins;

        std::vector<size_t> histogram(bins, 0);
        const size_t n = numel();

        for (size_t i = 0; i < n; ++i) {
            size_t bin =
                std::min(static_cast<size_t>((data()[i] - min_val) / bin_width),
                         bins - 1);
            histogram[bin]++;
        }

        size_t max_count =
            *std::max_element(histogram.begin(), histogram.end());
        const size_t width = 50;

        std::cout << "\nValue Distribution:";
        std::cout << "\n------------------\n";

        for (size_t i = 0; i < bins; ++i) {
            T bin_start = min_val + i * bin_width;
            std::cout << format_float(bin_start, 2) << ": ";

            size_t bar_length = (histogram[i] * width) / max_count;
            std::cout << std::string(bar_length, '#') << " " << histogram[i]
                      << "\n";
        }
    }

    /**
     * @brief Returns shape as formatted string
     */
    std::string shape_string() const {
        std::stringstream ss;
        ss << "(";

        for (size_t i = 0; i < ndim(); ++i) {
            ss << m_shape[i];

            if (i < ndim() - 1)
                ss << ", ";
        }

        ss << ")";
        return ss.str();
    }

    Shape m_shape;
    Strides m_strides;
    std::shared_ptr<Storage> m_storage;
    size_t m_offset{0};  // For view and slices
};
}  // namespace brezel::tensor