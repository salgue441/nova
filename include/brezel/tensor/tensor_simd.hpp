#pragma once

#include <stddef.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <brezel/core/macros.hpp>
#include <type_traits>

#ifdef BREZEL_SIMD_AVX512
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX2)
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX)
#include <immintrin.h>
#endif

namespace brezel::tensor::detail {
/**
 * @brief Applies a binary operation using SIMD instructions for optimized
 * performance on float and double types
 *
 * @details This function provides SIMD (Single Instruction Multiple Data)
 * optimized implementation for binary operations on floating point arrays. For
 * non-floating point types, it falls back to a standard scalar implementation.
 * The function supports both AVX2 and AVX512 instruction sets, automatically
 * selecting the appropriate SIMD width (8 elements for AVX2, 16 for AVX512).
 * Operations are parallelized using Intel TBB for additional performance.
 *
 * Supported SIMD operations:
 * - Addition (std::plus)
 * - Subtraction (std::minus)
 * - Multiplication (std::multiplies)
 * - Division (std::divides)
 * For other operations, falls back to scalar implementation.
 *
 * @note Requires proper memory alignment for SIMD operations.
 * @note The arrays must have sufficient size to handle SIMD width operations.
 *
 * @tparam T Type of the elements (optimized for float/double, falls back to
 * scalar for others)
 * @tparam BinaryOp Type of the binary operation to apply
 * @param[out] result Pointer to the destination array
 * @param[in] a Pointer to the first input array
 * @param[in] b Pointer to the second input array
 * @param[in] size Number of elements to process
 * @param[in] op Binary operation to apply
 * @param[in] block_size Block size for parallel processing
 */
template <typename T, typename BinaryOp>
BREZEL_FORCE_INLINE void apply_simd_op(T* result, const T* a, const T* b,
                                       size_t size, BinaryOp op,
                                       size_t block_size = 1024) {
    // For non-vectorizable types, use scalar implementation
    if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = op(a[i], b[i]);
        }

        return;
    }

#if defined(BREZEL_SIMD_AVX512)
    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t simd_width = 16;
        const size_t simd_size = size - (size % simd_width);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, simd_size, block_size),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end();
                     i += simd_width) {
                    __m512 va = _mm512_loadu_ps(&a[i]);
                    __m512 vb = _mm512_loadu_ps(&b[i]);
                    __m512 vr;

                    if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                        vr = _mm512_add_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::minus<T>>) {
                        vr = _mm512_sub_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::multiplies<T>>) {
                        vr = _mm512_mul_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::divides<T>>) {
                        vr = _mm512_div_ps(va, vb);
                    } else {
                        for (size_t j = 0; j < simd_width; ++j) {
                            result[i + j] = op(a[i + j], b[i + j]);
                        }

                        continue;
                    }

                    _mm512_storeu_ps(&result[i], vr);
                }
            });

        // Process remaining
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = op(a[i], b[i]);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t simd_width = 8;
        const size_t simd_size = size - (size % simd_width);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, simd_size, block_size),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end();
                     i += simd_width) {
                    __m512d va = _mm512_loadu_pd(&a[i]);
                    __m512d vb = _mm512_loadu_pd(&b[i]);
                    __m512d vr;

                    if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                        vr = _mm512_add_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::minus<T>>) {
                        vr = _mm512_sub_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::multiplies<T>>) {
                        vr = _mm512_mul_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::divides<T>>) {
                        vr = _mm512_div_pd(va, vb);
                    } else {
                        for (size_t j = 0; j < simd_width; ++j) {
                            result[i + j] = op(a[i + j], b[i + j]);
                        }

                        continue;
                    }

                    _mm512_storeu_pd(&result[i], vr);
                }
            });

        // Process remaining
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = op(a[i], b[i]);
        }
    }
#elif defined(BREZEL_SIMD_AVX2)
    if constexpr (std::is_same_v<T, float>) {
        constexpr size_t simd_width = 8;
        const size_t simd_size = size - (size % simd_width);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, simd_size, block_size),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end();
                     i += simd_width) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vb = _mm256_loadu_ps(&b[i]);
                    __m256 vr;

                    if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                        vr = _mm256_add_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::minus<T>>) {
                        vr = _mm256_sub_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::multiplies<T>>) {
                        vr = _mm256_mul_ps(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::divides<T>>) {
                        vr = _mm256_div_ps(va, vb);
                    } else {
                        for (size_t j = 0; j < simd_width; ++j) {
                            result[i + j] = op(a[i + j], b[i + j]);
                        }

                        continue;
                    }

                    _mm256_storeu_ps(&result[i], vr);
                }
            });

        // Process remaining
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = op(a[i], b[i]);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        constexpr size_t simd_width = 4;
        const size_t simd_size = size - (size % simd_width);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, simd_size, block_size),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end();
                     i += simd_width) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vb = _mm256_loadu_pd(&b[i]);
                    __m256d vr;

                    if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                        vr = _mm256_add_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::minus<T>>) {
                        vr = _mm256_sub_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::multiplies<T>>) {
                        vr = _mm256_mul_pd(va, vb);
                    } else if constexpr (std::is_same_v<BinaryOp,
                                                        std::divides<T>>) {
                        vr = _mm256_div_pd(va, vb);
                    } else {
                        for (size_t j = 0; j < simd_width; ++j) {
                            result[i + j] = op(a[i + j], b[i + j]);
                        }

                        continue;
                    }

                    _mm256_storeu_pd(&result[i], vr);
                }
            });

        // Process remaining
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = op(a[i], b[i]);
        }
    }
#else
    // No SIMD support, use scalar implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = op(a[i], b[i]);
    }
#endif
}
}  // namespace brezel::tensor::detail