#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <brezel/tensor/tensor.hpp>
#include <cmath>
#include <random>

namespace brezel::tensor {

/**
 * @brief Factory class for creating tensors of different types and
 * configurations
 *
 * @details This class provides static factory methods for creating tensors,
 * including zeros, ones, random values, special matrices, etc. It separates
 * the creation logic from the tensor class itself for better organization.
 *
 * @tparam T Data type of tensor elements (must satisfy TensorScalar concept)
 */
template <TensorScalar T>
class BREZEL_API TensorFactory {
    static constexpr size_t kDefaultBlockSize = 1024;

public:
    /**
     * @brief Creates an empty tensor with no elements
     * @return Empty tensor (scalar)
     */
    BREZEL_NODISCARD static Tensor<T> empty() { return Tensor<T>(); }

    /**
     * @brief Creates a tensor with uninitialized memory
     *
     * @param shape Desired shape
     * @return Uninitialized tensor
     */
    BREZEL_NODISCARD static Tensor<T> empty(const Shape& shape) {
        return Tensor<T>(shape);
    }

    /**
     * @brief Creates a tensor with uninitialized memory
     *
     * @param dims Dimensions of the tensor
     * @return Uninitialized tensor
     */
    BREZEL_NODISCARD static Tensor<T> empty(
        std::initializer_list<int64_t> dims) {
        return Tensor<T>(Shape(dims));
    }

    /**
     * @brief Creates a tensor filled with zeros
     *
     * @param shape Desired shape
     * @return Tensor filled with zeros
     */
    BREZEL_NODISCARD static Tensor<T> zeros(const Shape& shape) {
        return Tensor<T>(shape, T(0));
    }

    /**
     * @brief Creates a tensor filled with zeros
     *
     * @param dims Dimensions of the tensor
     * @return Tensor filled with zeros
     */
    BREZEL_NODISCARD static Tensor<T> zeros(
        std::initializer_list<int64_t> dims) {
        return Tensor<T>(Shape(dims), T(0));
    }

    /**
     * @brief Creates a tensor filled with ones
     *
     * @param shape Desired shape
     * @return Tensor filled with ones
     */
    BREZEL_NODISCARD static Tensor<T> ones(const Shape& shape) {
        return Tensor<T>(shape, T(1));
    }

    /**
     * @brief Creates a tensor filled with ones
     *
     * @param dims Dimensions of the tensor
     * @return Tensor filled with ones
     */
    BREZEL_NODISCARD static Tensor<T> ones(
        std::initializer_list<int64_t> dims) {
        return Tensor<T>(Shape(dims), T(1));
    }

    /**
     * @brief Creates a tensor filled with the specified value
     *
     * @param shape Desired shape
     * @param value Value to fill tensor with
     * @return Tensor filled with the specified value
     */
    BREZEL_NODISCARD static Tensor<T> full(const Shape& shape, T value) {
        return Tensor<T>(shape, value);
    }

    /**
     * @brief Creates a tensor filled with the specified value
     *
     * @param dims Dimensions of the tensor
     * @param value Value to fill tensor with
     * @return Tensor filled with the specified value
     */
    BREZEL_NODISCARD static Tensor<T> full(std::initializer_list<int64_t> dims,
                                           T value) {
        return Tensor<T>(Shape(dims), value);
    }

    /**
     * @brief Creates a tensor with uniformly distributed random values
     *
     * @param shape Desired shape
     * @param min Minimum value (default 0)
     * @param max Maximum value (default 1)
     * @return Tensor with random values
     */
    BREZEL_NODISCARD static Tensor<T> rand(const Shape& shape, T min = T(0),
                                           T max = T(1)) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());

        Tensor<T> result(shape);
        const size_t n = result.numel();

        // Different distribution for integral and floating point types
        if constexpr (std::is_integral_v<T>) {
            std::uniform_int_distribution<T> dist(min, max);
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::mt19937 local_gen(rd() + range.begin());

                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] = dist(local_gen);
                    }
                });
        } else {
            std::uniform_real_distribution<T> dist(min, max);
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::mt19937 local_gen(rd() + range.begin());

                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] = dist(local_gen);
                    }
                });
        }

        return result;
    }

    /**
     * @brief Creates a tensor with uniformly distributed random values
     *
     * @param dims Dimensions of the tensor
     * @param min Minimum value (default 0)
     * @param max Maximum value (default 1)
     * @return Tensor with random values
     */
    BREZEL_NODISCARD static Tensor<T> rand(std::initializer_list<int64_t> dims,
                                           T min = T(0), T max = T(1)) {
        return rand(Shape(dims), min, max);
    }

    /**
     * @brief Creates a tensor with normally distributed random values
     *
     * @param shape Desired shape
     * @param mean Mean of the distribution (default 0)
     * @param std Standard deviation of the distribution (default 1)
     * @return Tensor with normally distributed random values
     */
    BREZEL_NODISCARD static Tensor<T> randn(const Shape& shape, T mean = T(0),
                                            T std = T(1)) {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());

        Tensor<T> result(shape);
        const size_t n = result.numel();

        if constexpr (std::is_floating_point_v<T>) {
            std::normal_distribution<T> dist(mean, std);
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::mt19937 local_gen(rd() + range.begin());

                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] = dist(local_gen);
                    }
                });
        } else {
            // For integral types, round the normal distribution
            std::normal_distribution<float> dist(static_cast<float>(mean),
                                                 static_cast<float>(std));

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::mt19937 local_gen(rd() + range.begin());

                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] =
                            static_cast<T>(std::round(dist(local_gen)));
                    }
                });
        }

        return result;
    }

    /**
     * @brief Creates a tensor with normally distributed random values
     *
     * @param dims Dimensions of the tensor
     * @param mean Mean of the distribution (default 0)
     * @param std Standard deviation of the distribution (default 1)
     * @return Tensor with normally distributed random values
     */
    BREZEL_NODISCARD static Tensor<T> randn(std::initializer_list<int64_t> dims,
                                            T mean = T(0), T std = T(1)) {
        return randn(Shape(dims), mean, std);
    }

    /**
     * @brief Creates an identity matrix
     *
     * @param n Size of the identity matrix
     * @return Identity matrix of size n×n
     */
    BREZEL_NODISCARD static Tensor<T> eye(int64_t n) {
        Tensor<T> result(Shape({n, n}), T(0));

        tbb::parallel_for(tbb::blocked_range<int64_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<int64_t>& range) {
                              for (int64_t i = range.begin(); i < range.end();
                                   ++i) {
                                  result.at({i, i}) = T(1);
                              }
                          });

        return result;
    }

    /**
     * @brief Creates a tensor with evenly spaced values
     *
     * @param start Start value
     * @param end End value (exclusive)
     * @param step Step size (default 1)
     * @return Tensor with evenly spaced values
     */
    BREZEL_NODISCARD static Tensor<T> arange(T start, T end, T step = T(1)) {
        BREZEL_ENSURE(step != T(0), "Step cannot be zero");

        int64_t size = static_cast<int64_t>(std::ceil((end - start) / step));
        BREZEL_ENSURE(
            size > 0,
            "Invalid range: end must be greater than start for positive step");

        Tensor<T> result(Shape({size}));

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, size, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    result.data()[i] = start + static_cast<T>(i) * step;
                }
            });

        return result;
    }

    /**
     * @brief Creates a tensor with evenly spaced values over a specified
     * interval
     *
     * @param start Start value
     * @param end End value (inclusive)
     * @param steps Number of steps
     * @return Tensor with evenly spaced values
     */
    BREZEL_NODISCARD static Tensor<T> linspace(T start, T end, int64_t steps) {
        BREZEL_ENSURE(steps > 1, "Number of steps must be greater than 1");

        Tensor<T> result(Shape({steps}));
        const T step = (end - start) / static_cast<T>(steps - 1);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, steps, kDefaultBlockSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    result.data()[i] = start + static_cast<T>(i) * step;
                }
            });

        // Ensure the last element is exactly 'end'
        if (steps > 0) {
            result.data()[steps - 1] = end;
        }

        return result;
    }

    /**
     * @brief Creates a tensor with logarithmically spaced values
     *
     * @param start Start value (base 10 exponent)
     * @param end End value (base 10 exponent, inclusive)
     * @param steps Number of steps
     * @param base Logarithm base (default 10)
     * @return Tensor with logarithmically spaced values
     */
    BREZEL_NODISCARD static Tensor<T> logspace(T start, T end, int64_t steps,
                                               T base = T(10)) {
        BREZEL_ENSURE(steps > 1, "Number of steps must be greater than 1");

        Tensor<T> result(Shape({steps}));
        const T step = (end - start) / static_cast<T>(steps - 1);

        if constexpr (std::is_floating_point_v<T>) {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, steps, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] =
                            std::pow(base, start + static_cast<T>(i) * step);
                    }
                });
        } else {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, steps, kDefaultBlockSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        result.data()[i] = static_cast<T>(
                            std::pow(static_cast<double>(base),
                                     static_cast<double>(start + i * step)));
                    }
                });
        }

        // Ensure the last element is exactly base^end
        if (steps > 0) {
            result.data()[steps - 1] = static_cast<T>(
                std::pow(static_cast<double>(base), static_cast<double>(end)));
        }

        return result;
    }

    /**
     * @brief Creates a diagonal tensor from a 1D tensor
     *
     * @param diagonal Tensor containing diagonal elements
     * @return Diagonal tensor
     */
    BREZEL_NODISCARD static Tensor<T> diag(const Tensor<T>& diagonal) {
        BREZEL_ENSURE(diagonal.ndim() == 1, "Input must be a 1D tensor");

        const int64_t n = diagonal.size(0);
        Tensor<T> result(Shape({n, n}), T(0));

        tbb::parallel_for(tbb::blocked_range<int64_t>(0, n, kDefaultBlockSize),
                          [&](const tbb::blocked_range<int64_t>& range) {
                              for (int64_t i = range.begin(); i < range.end();
                                   ++i) {
                                  result.at({i, i}) = diagonal.at({i});
                              }
                          });

        return result;
    }

    /**
     * @brief Creates a triangular tensor
     *
     * @param n Size of the tensor
     * @param upper Whether to create upper (true) or lower (false) triangular
     * tensor
     * @param value Value to fill the triangular part with (default 1)
     * @return Triangular tensor
     */
    BREZEL_NODISCARD static Tensor<T> tril_or_triu(int64_t n, bool upper,
                                                   T value = T(1)) {
        Tensor<T> result(Shape({n, n}), T(0));

        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, n, kDefaultBlockSize),
            [&](const tbb::blocked_range<int64_t>& range) {
                for (int64_t i = range.begin(); i < range.end(); ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        if ((upper && j >= i) || (!upper && j <= i)) {
                            result.at({i, j}) = value;
                        }
                    }
                }
            });

        return result;
    }

    /**
     * @brief Creates a lower triangular tensor
     *
     * @param n Size of the tensor
     * @param value Value to fill the triangular part with (default 1)
     * @return Lower triangular tensor
     */
    BREZEL_NODISCARD static Tensor<T> tril(int64_t n, T value = T(1)) {
        return tril_or_triu(n, false, value);
    }

    /**
     * @brief Creates an upper triangular tensor
     *
     * @param n Size of the tensor
     * @param value Value to fill the triangular part with (default 1)
     * @return Upper triangular tensor
     */
    BREZEL_NODISCARD static Tensor<T> triu(int64_t n, T value = T(1)) {
        return tril_or_triu(n, true, value);
    }

    /**
     * @brief Creates a tensor matching the shape of another tensor
     *
     * @param other Tensor to get shape from
     * @param value Value to fill the tensor with (default 0)
     * @return New tensor with same shape as other
     */
    BREZEL_NODISCARD static Tensor<T> like(const Tensor<T>& other,
                                           T value = T(0)) {
        return Tensor<T>(other.shape(), value);
    }

    /**
     * @brief Creates a tensor with zeros and the same shape as another tensor
     *
     * @param other Tensor to get shape from
     * @return New tensor with zeros
     */
    BREZEL_NODISCARD static Tensor<T> zeros_like(const Tensor<T>& other) {
        return Tensor<T>(other.shape(), T(0));
    }

    /**
     * @brief Creates a tensor with ones and the same shape as another tensor
     *
     * @param other Tensor to get shape from
     * @return New tensor with ones
     */
    BREZEL_NODISCARD static Tensor<T> ones_like(const Tensor<T>& other) {
        return Tensor<T>(other.shape(), T(1));
    }

    /**
     * @brief Creates a tensor with random values and the same shape as another
     * tensor
     *
     * @param other Tensor to get shape from
     * @param min Minimum value (default 0)
     * @param max Maximum value (default 1)
     * @return New tensor with random values
     */
    BREZEL_NODISCARD static Tensor<T> rand_like(const Tensor<T>& other,
                                                T min = T(0), T max = T(1)) {
        return rand(other.shape(), min, max);
    }

    /**
     * @brief Creates a tensor with normally distributed random values and the
     * same shape as another tensor
     *
     * @param other Tensor to get shape from
     * @param mean Mean of the distribution (default 0)
     * @param std Standard deviation of the distribution (default 1)
     * @return New tensor with random values
     */
    BREZEL_NODISCARD static Tensor<T> randn_like(const Tensor<T>& other,
                                                 T mean = T(0), T std = T(1)) {
        return randn(other.shape(), mean, std);
    }

    /**
     * @brief Creates a tensor with the specified value and the same shape as
     * another tensor
     *
     * @param other Tensor to get shape from
     * @param value Value to fill the tensor with
     * @return New tensor with the specified value
     */
    BREZEL_NODISCARD static Tensor<T> full_like(const Tensor<T>& other,
                                                T value) {
        return Tensor<T>(other.shape(), value);
    }
};

// Convenience namespace alias to make using the factory methods more concise
namespace factory = ::brezel::tensor;

}  // namespace brezel::tensor