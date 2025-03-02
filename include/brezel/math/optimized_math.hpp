#pragma once

#include <algorithm>
#include <brezel/core/macros.hpp>
#include <cmath>
#include <type_traits>

#if defined(BREZEL_SIMD_AVX2)
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX512)
#include <immintrin.h>
#endif

/**
 * @brief Optimized math operations for tensor computations
 *
 * This namespace provides optimized implementations of math functions that can
 * be used in performance-critical code paths. Functions automatically use SIMD
 * instructions when available
 */
namespace brezel::math {
/**
 * @brief Computes an approximation of the exponential function e^x.
 *
 * This function provides a fast approximation of the exponential function
 * for floating-point types. It uses a polynomial approximation for the
 * range [-ln(2)/2, ln(2)/2] and handles large values by scaling.
 *
 * @tparam T The floating-point type (e.g., float, double).
 * @param x The exponent value.
 * @return The approximate value of e^x.
 *
 * @note The function is optimized for performance and may not provide
 *       the same accuracy as the standard library exp function.
 * @note The input value x is clamped to the range [-87.0f, 88.0f] to
 *       avoid overflow and underflow.
 *
 * @pre T must be a floating-point type.
 * @pre x must be within the range [-87.0f, 88.0f].
 *
 * @warning The function returns 0.0f for x < -87.0f and positive infinity
 *          for x > 88.0f.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_exp(T x) {
    static_assert(std::is_floating_point_v<T>,
                  "fast_exp requires floating point type");

    if (x < -87.0f) {
        return 0.0f;
    }

    if (x > 88.0f) {
        return std::numeric_limits<float>::infinity();
    }

    const float ln2 = 0.6931471805599453f;
    const float half_ln2 = ln2 * 0.5f;

    float f = std::floor(x / ln2 + 0.5f);
    float y = x - f * ln2;
    float p =
        1.0f +
        y * (1.0f +
             y * (0.5f + y * (0.166666667f +
                              y * (0.041666667f +
                                   y * (0.008333333f + y * 0.001388889f)))));

    int exponent = static_cast<int>(f);
    return std::ldexp(static_cast<T>(p), exponent);
}

/**
 * @brief Computes a fast approximation of the natural logarithm of a
 * floating-point number.
 *
 * This function uses a polynomial approximation to compute the natural
 * logarithm of a given floating-point number. It is optimized for performance
 * and may not be as accurate as the standard library's log function.
 *
 * @tparam T The floating-point type of the input value. Must be a
 * floating-point type.
 * @param x The input value for which the natural logarithm is to be computed.
 * Must be greater than 0.
 * @return The approximate natural logarithm of the input value. If the input
 * value is less than or equal to 0, returns negative infinity.
 *
 * @note This function requires the input type to be a floating-point type. If
 * the input value is less than or equal to 0, the function returns negative
 * infinity.
 * @note Polynomial approximation of log(m) where m in [1, 2)
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_log(T x) {
    static_assert(std::is_floating_point_v<T>,
                  "fast_log requires floating point type");

    if (x <= 0) {
        return -std::numeric_limits<T>::infinity();
    }

    int exponent;
    float mantissa = std::frexp(x, &exponent);

    if (mantissa < 1.0f) {
        mantissa *= 2.0f;
        exponent--;
    }

    float y = (mantissa - 1.0f) / (mantissa + 1.0f);
    float y2 = y * y;
    float log_mantissa =
        y * (2.0f + y2 * (2.0f / 3.0f + y2 * (2.0f / 5.0f + y2 * 2.0f / 7.0f)));

    const float ln2 = 0.6931471805599453f;
    return static_cast<T>(log_mantissa + exponent * ln2);
}

/**
 * @brief Computes the square root of a given number using optimized methods.
 *
 * This function provides a fast approximation of the square root for float and
 * double types. It uses SIMD instructions if available (AVX2 or AVX512) for
 * enhanced performance. For float types, if SIMD is not available, it uses a
 * bit-level approximation method. For double types, if SIMD is not available,
 * it falls back to the standard library's sqrt function. For other types, it
 * converts the input to double, computes the square root, and converts back to
 * the original type.
 *
 * @tparam T The type of the input value. Supported types are float, double, and
 * other numeric types.
 * @param x The input value for which the square root is to be computed.
 * @return The computed square root of the input value.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_sqrt(T x) {
    if constexpr (std::is_same_v<T, float>) {
#if defined(BREZEL_SIMD_AVX2) || defined(BREZEL_SIMD_AVX512)
        __m128 value = _mm_set_ss(x);
        __m128 result = _mm_sqrt_ss(value);

        return _mm_cvtss_f32(result);
#else
        union {
            float f;
            u_int32_t i;
        } u;

        u.f = x;
        u.i = (u.i + 0x3f76cf62) >> 1;
        u.f = 0.5f * (u.f + x / u.f);

        return u.f;
#endif
    } else if constexpr (std::is_same_v<T, double>) {
#if defined(BREZEL_SIMD_AVX2) || defined(BREZEL_SIMD_AVX512)
        __m128d value = _mm_set_sd(x);
        __m128d result = _mm_sqrt_sd(value, value);

        return _mm_cvtsd_f64(result);
#else
        return std::sqrt(x);
#endif } else {
        return static_cast<T>(std::sqrt(static_cast<double>(x)));
    }
}

/**
 * @brief Computes a fast approximation of the hyperbolic tangent function.
 *
 * This function provides a fast approximation of the tanh function for floating
 * point types. It uses a polynomial approximation for values of x where x^2
 * <= 15.0f, and returns 1.0 or -1.0 for values of x where x^2 > 15.0f.
 *
 * @tparam T The floating point type of the input value.
 * @param x The input value for which to compute the hyperbolic tangent.
 * @return The approximated hyperbolic tangent of the input value.
 * @note This function requires the input type to be a floating point type.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_tanh(T x) {
    static_assert(std::is_floating_point_v<T>,
                  "fast_tanh requires floating point type");

    const float x2 = x * x;
    if (x2 > 15.0f) {
        return (x > 0.0f) ? T(1.0f) : T(-1.0f);
    }

    return static_cast<T>((x * (27.0f + x2)) / (27.0f + 9.0f * x2));
}

/**
 * @brief Computes a fast approximation of the sigmoid function.
 *
 * This function provides a fast approximation of the sigmoid function,
 * which is defined as 1 / (1 + exp(-x)). It is optimized for performance
 * by using a threshold to return early for large positive or negative values.
 *
 * @tparam T The type of the input value, which must be a floating point type.
 * @param x The input value for which to compute the sigmoid function.
 * @return The sigmoid of the input value, which will be in the range [0, 1].
 *
 * @note This function requires the input type to be a floating point type.
 *       If the input value is greater than 10.0, the function returns 1.0.
 *       If the input value is less than or equal to -10.0, the function returns
 * 0.0.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_sigmoid(T x) {
    static_assert(std::is_floating_point_v<T>,
                  "fast_sigmoid requires floating point type");

    if (x > 10.0f)
        return T(1.0f);

    if (x <= -10.0f)
        return T(0.0f);

    float numerator = 1.0f;
    float denominator = 1.0f + fast_exp(-x);

    return static_cast<T>(numerator / denominator);
}

/**
 * @brief Computes the Rectified Linear Unit (ReLU) function in an optimized
 * manner.
 *
 * The ReLU function is defined as:
 * - f(x) = x if x > 0
 * - f(x) = 0 if x <= 0
 *
 * This function is optimized for performance using inline and nodiscard
 * attributes.
 *
 * @tparam T The type of the input value. Typically, this would be a
 * floating-point type.
 * @param x The input value for which the ReLU function is to be computed.
 * @return The result of the ReLU function applied to the input value.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_relu(T x) {
    return x > T(0) ? x : T(0);
}

/**
 * @brief Computes the Leaky ReLU activation function.
 *
 * This function applies the Leaky Rectified Linear Unit (Leaky ReLU) activation
 * function, which allows a small, non-zero gradient when the unit is not
 * active.
 *
 * @tparam T The type of the input value.
 * @param x The input value.
 * @param alpha The slope of the function for x < 0. Default value is 0.01.
 * @return The result of applying the Leaky ReLU function to the input value.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_leaky_relu(T x, T alpha = T(0.01)) {
    return x > T(0) ? x : alpha * x;
}

/**
 * @brief Computes the power of a number using an optimized method.
 *
 * This function calculates the power of a given base raised to a specified
 * exponent using an efficient algorithm that reduces the number of
 * multiplications.
 *
 * @tparam T The type of the base and the result. Typically, this would be a
 * floating-point or integer type.
 * @param base The base value to be raised to the power of the exponent.
 * @param exponent The exponent value. Can be positive, negative, or zero.
 * @return The result of raising the base to the power of the exponent.
 *         If the exponent is zero, the function returns 1.
 *         If the exponent is negative, the function returns the reciprocal of
 * the base raised to the absolute value of the exponent. If the base is zero
 * and the exponent is positive, the function returns 0.
 *
 * @note This function uses bitwise operations and exponentiation by squaring to
 * achieve better performance. It is marked as BREZEL_NODISCARD and
 * BREZEL_FORCE_INLINE for optimization purposes.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_pow(T base, int exponent) {
    if (exponent == 0)
        return T(1);

    if (exponent == 1)
        return base;

    if (base == T(0))
        return T(0);

    bool is_negative = exponent < 0;
    unsigned int abs_exp = std::abs(exponent);

    T result = T(1);
    T current_power = base;

    while (abs_exp > 0) {
        if (abs_exp & 1) {
            result *= current_power;
        }

        current_power *= current_power;
        abs_exp >>= 1;
    }

    return is_negative ? (T(1) / result) : result;
}

/**
 * @brief Computes the power of a base raised to an exponent using an optimized
 * method.
 *
 * This function calculates the power of a base raised to an exponent using a
 * combination of special cases and mathematical identities for improved
 * performance. The base type must be a floating point.
 *
 * @tparam T The type of the base and exponent, which must be a floating point
 * type.
 * @param base The base value to be raised to the power of the exponent.
 * @param exponent The exponent value to which the base is raised.
 * @return The result of base raised to the power of exponent.
 *
 * @note Special cases:
 * - If the base is 0, the result is 0.
 * - If the base is 1, the result is 1.
 * - If the exponent is 0, the result is 1.
 * - If the exponent is 1, the result is the base.
 * - If the exponent is an integer, the function calls itself with the exponent
 * cast to an integer.
 *
 * @note For non-integer exponents, the function uses the identity: x^y = exp(y
 * * log(x)).
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_pow(T base, T exponent) {
    static_assert(std::is_floating_point_v<T>,
                  "Base type must be floating point");

    if (base == T(0))
        return T(0);

    if (base == T(1))
        return T(1);

    if (exponent == T(0))
        return T(1);

    if (exponent == T(1))
        return base;

    if (std::floor(exponent) == exponent) {
        return fast_pow(base, static_cast<int>(exponent));
    }

    return fast_exp(exponent * fast_log(base));
}

/**
 * @brief Computes the absolute value of a number with optimizations for
 * different types.
 *
 * This function provides a fast implementation of the absolute value
 * computation for floating-point and signed integral types. It uses bitwise
 * operations to clear the sign bit for floating-point numbers and a combination
 * of bitwise and arithmetic operations for signed integral types.
 *
 * @tparam T The type of the input value. Supported types are floating-point
 * (float, double) and signed integral types.
 * @param x The input value for which the absolute value is to be computed.
 * @return The absolute value of the input.
 *
 * @note For floating-point types, the function uses a union to manipulate the
 * bits directly and clear the sign bit. For signed integral types, it uses a
 * bitwise shift and XOR operations to compute the absolute value.
 * @note If the type is not a floating-point or signed integral type, the
 * function falls back to using std::abs.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_abs(T x) {
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (std::is_same_v<T, float>) {
            union {
                float f;
                uint32_t i;
            } u{x};

            // Clear sign bit
            u.i &= 0x7FFFFFFF;
            return u.f;
        } else if constexpr (std::is_same_v<T, double>) {
            union {
                double d;
                uint64_t i;
            } u{x};

            // Clear sign bit
            u.i &= 0x7FFFFFFFFFFFFFFFULL;
            return u.d;
        }
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        const T mask = x >> (sizeof(T) * 8 - 1);
        return (x + mask) ^ mask;
    }

    return std::abs(x);
}

/**
 * @brief Computes the fast inverse square root of a given floating point
 * number.
 *
 * This function uses the Quake III Arena fast inverse square root algorithm for
 * single-precision floating point numbers (float). For other floating point
 * types, it falls back to using the standard library's sqrt function.
 *
 * @tparam T The floating point type of the input value.
 * @param x The input value for which to compute the inverse square root.
 * @return The fast inverse square root of the input value.
 *
 * @note This function requires the input type to be a floating point type.
 *       If the input type is float, it uses a highly optimized algorithm.
 *       For other floating point types, it uses the standard library's sqrt
 * function.
 */
template <typename T>
BREZEL_NODISCARD BREZEL_FORCE_INLINE T fast_inv_sqrt(T x) {
    static_assert(std::is_floating_point_v<T>,
                  "fast_inv_sqrt requires floating point type");

    if constexpr (std::is_same_v<T, float>) {
        union {
            float f;
            uint32_t i;
        } u;

        u.f = x;
        u.i = 0x5F3759DF - (u.i >> 1);

        float xhalf = 0.5f * x;
        u.f = u.f * (1.5f - xhalf * u.f * u.f);

        return u.f;
    } else {
        return 1.0 / std::sqrt(x);
    }
}

// SIMD batch operations for 4/8/16 elements at once
#if defined(BREZEL_SIMD_AVX2) || defined(BREZEL_SIMD_AVX512)

/**
 * @brief Computes the exponential of each element in the input array using SIMD
 * vectorization.
 *
 * This function computes the exponential of each element in the input array
 * `data` using SIMD (Single Instruction, Multiple Data) instructions for
 * performance optimization. It supports both AVX512 and AVX2 instruction sets.
 *
 * @param data Pointer to the input array of floats. The results will be stored
 * in this array.
 * @param count The number of elements in the input array.
 *
 * @note The function uses different implementations based on the available SIMD
 * instruction set:
 *       - For AVX512: Processes 16 elements at a time.
 *       - For AVX2: Processes 8 elements at a time.
 *
 * @note The function handles underflow and overflow conditions:
 *       - Underflow: Values less than -87.0f are set to 0.
 *       - Overflow: Values greater than 88.0f are set to infinity.
 *
 * @note For elements that cannot be processed by SIMD (remaining elements after
 * vectorized processing), the function falls back to a scalar implementation
 * using `fast_exp`.
 *
 * @warning Ensure that the input array `data` has at least `count` elements to
 * avoid out-of-bounds access.
 */
BREZEL_FORCE_INLINE void batch_exp_f32(float* data, size_t count) {
#if defined(BREZEL_SIMD_AVX512)
    size_t vec_count = count / 16;
    size_t i = 0;

    for (; i < vec_count * 16; i += 16) {
        __m512 x = _mm512_load_ps(data + i);

        // Apply the same algorithm as in fast_exp, but vectorized
        __m512 mask_under =
            _mm512_cmp_ps_mask(x, _mm512_set1_ps(-87.0f), _CMP_LT_OS);
        __m512 mask_over =
            _mm512_cmp_ps_mask(x, _mm512_set1_ps(88.0f), _CMP_GT_OS);

        const __m512 ln2 = _mm512_set1_ps(0.6931471805599453f);
        __m512 f = _mm512_floor_ps(_mm512_div_ps(x, ln2));
        __m512 y = _mm512_sub_ps(x, _mm512_mul_ps(f, ln2));

        // Polynomial approximation
        __m512 p = _mm512_add_ps(
            _mm512_set1_ps(1.0f),
            _mm512_mul_ps(
                y,
                _mm512_add_ps(
                    _mm512_set1_ps(1.0f),
                    _mm512_mul_ps(
                        y,
                        _mm512_add_ps(
                            _mm512_set1_ps(0.5f),
                            _mm512_mul_ps(
                                y,
                                _mm512_add_ps(
                                    _mm512_set1_ps(0.166666667f),
                                    _mm512_mul_ps(
                                        y,
                                        _mm512_add_ps(
                                            _mm512_set1_ps(0.041666667f),
                                            _mm512_mul_ps(
                                                y,
                                                _mm512_add_ps(
                                                    _mm512_set1_ps(
                                                        0.008333333f),
                                                    _mm512_mul_ps(
                                                        y,
                                                        _mm512_set1_ps(
                                                            0.001388889f)))))))))))));

        // 2^exponent using integer trick
        __m512i exponent = _mm512_cvtps_epi32(f);
        __m512i exp_shift = _mm512_slli_epi32(
            _mm512_add_epi32(exponent, _mm512_set1_epi32(127)), 23);
        __m512 exp_scale = _mm512_castsi512_ps(exp_shift);

        __m512 result = _mm512_mul_ps(p, exp_scale);

        // Handle underflow/overflow
        result = _mm512_mask_blend_ps(mask_under, result, _mm512_setzero_ps());
        result = _mm512_mask_blend_ps(
            mask_over, result,
            _mm512_set1_ps(std::numeric_limits<float>::infinity()));

        _mm512_store_ps(data + i, result);
    }

    for (; i < count; i++) {
        data[i] = fast_exp(data[i]);
    }
#elif defined(BREZEL_SIMD_AVX2)
    size_t vec_count = count / 8;
    size_t i = 0;

    for (; i < vec_count * 8; i += 8) {
        __m256 x = _mm256_load_ps(data + i);

        // Apply the same algorithm as in fast_exp, but vectorized
        __m256 mask_under =
            _mm256_cmp_ps(x, _mm256_set1_ps(-87.0f), _CMP_LT_OS);
        __m256 mask_over = _mm256_cmp_ps(x, _mm256_set1_ps(88.0f), _CMP_GT_OS);

        const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
        __m256 f = _mm256_floor_ps(_mm256_div_ps(x, ln2));
        __m256 y = _mm256_sub_ps(x, _mm256_mul_ps(f, ln2));

        // Polynomial approximation
        __m256 p = _mm256_add_ps(
            _mm256_set1_ps(1.0f),
            _mm256_mul_ps(
                y,
                _mm256_add_ps(
                    _mm256_set1_ps(1.0f),
                    _mm256_mul_ps(
                        y,
                        _mm256_add_ps(
                            _mm256_set1_ps(0.5f),
                            _mm256_mul_ps(
                                y,
                                _mm256_add_ps(
                                    _mm256_set1_ps(0.166666667f),
                                    _mm256_mul_ps(
                                        y,
                                        _mm256_add_ps(
                                            _mm256_set1_ps(0.041666667f),
                                            _mm256_mul_ps(
                                                y,
                                                _mm256_add_ps(
                                                    _mm256_set1_ps(
                                                        0.008333333f),
                                                    _mm256_mul_ps(
                                                        y,
                                                        _mm256_set1_ps(
                                                            0.001388889f)))))))))))));

        // 2^exponent using integer trick
        __m256i exponent = _mm256_cvtps_epi32(f);
        __m256i exp_shift = _mm256_slli_epi32(
            _mm256_add_epi32(exponent, _mm256_set1_epi32(127)), 23);
        __m256 exp_scale = _mm256_castsi256_ps(exp_shift);

        __m256 result = _mm256_mul_ps(p, exp_scale);

        // Handle underflow/overflow
        result = _mm256_blendv_ps(result, _mm256_setzero_ps(), mask_under);
        result = _mm256_blendv_ps(
            result, _mm256_set1_ps(std::numeric_limits<float>::infinity()),
            mask_over);

        _mm256_store_ps(data + i, result);
    }

    for (; i < count; i++) {
        data[i] = fast_exp(data[i]);
    }
#endif
}

/**
 * @brief Computes the natural logarithm of an array of single-precision
 * floating-point numbers.
 *
 * This function uses SIMD (Single Instruction, Multiple Data) instructions to
 * optimize the computation of the natural logarithm for an array of
 * single-precision floating-point numbers. It supports AVX512 and AVX2
 * instruction sets for batch processing.
 *
 * @param data Pointer to the array of single-precision floating-point numbers.
 * @param count The number of elements in the array.
 *
 * @note The function handles non-positive values by setting the result to
 * negative infinity for such inputs. The remaining elements are processed using
 * a fast logarithm approximation function.
 *
 * @details
 * - For AVX512:
 *   - Processes 16 elements at a time.
 *   - Extracts mantissa and exponent using bit manipulation.
 *   - Scales the mantissa to the [1, 2) range.
 *   - Uses a polynomial approximation to compute the logarithm of the mantissa.
 *   - Converts the exponent to a floating-point number and multiplies by
 * log(2).
 *   - Handles non-positive inputs by setting the result to negative infinity.
 * - For AVX2:
 *   - Processes 8 elements at a time.
 *   - Extracts mantissa and exponent using bit manipulation.
 *   - Scales the mantissa to the [1, 2) range.
 *   - Uses a polynomial approximation to compute the logarithm of the mantissa.
 *   - Converts the exponent to a floating-point number and multiplies by
 * log(2).
 *   - Handles non-positive inputs by setting the result to negative infinity.
 */
BREZEL_FORCE_INLINE void batch_log_f32(float* data, size_t count) {
#if defined(BREZEL_SIMD_AVX512)
    size_t vec_count = count / 16;
    size_t i = 0;

    for (; i < vec_count * 16; i += 16) {
        __m512 x = _mm512_load_ps(data + i);

        // Handle non-positive values
        __mmask16 mask_nonpos =
            _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS);

        // Extract mantissa and exponent using bit manipulation
        __m512i xi = _mm512_castps_si512(x);
        __m512i exponent = _mm512_srli_epi32(
            _mm512_and_si512(xi, _mm512_set1_epi32(0x7F800000)), 23);
        __m512i mantissa_bits =
            _mm512_and_si512(xi, _mm512_set1_epi32(0x007FFFFF));
        __m512 mantissa = _mm512_or_ps(_mm512_castsi512_ps(mantissa_bits),
                                       _mm512_set1_ps(1.0f));

        // Scale mantissa to [1, 2) range
        __mmask16 scale_mask =
            _mm512_cmp_ps_mask(mantissa, _mm512_set1_ps(1.0f), _CMP_LT_OS);
        mantissa = _mm512_mask_mul_ps(mantissa, scale_mask, mantissa,
                                      _mm512_set1_ps(2.0f));
        __m512i exponent_adj = _mm512_mask_sub_epi32(
            exponent, scale_mask, exponent, _mm512_set1_epi32(1));

        // y = (mantissa - 1) / (mantissa + 1)
        __m512 y = _mm512_div_ps(_mm512_sub_ps(mantissa, _mm512_set1_ps(1.0f)),
                                 _mm512_add_ps(mantissa, _mm512_set1_ps(1.0f)));

        __m512 y2 = _mm512_mul_ps(y, y);

        // Polynomial approximation
        __m512 log_mantissa = _mm512_mul_ps(
            y,
            _mm512_add_ps(
                _mm512_set1_ps(2.0f),
                _mm512_mul_ps(
                    y2,
                    _mm512_add_ps(
                        _mm512_set1_ps(2.0f / 3.0f),
                        _mm512_mul_ps(
                            y2, _mm512_add_ps(
                                    _mm512_set1_ps(2.0f / 5.0f),
                                    _mm512_mul_ps(
                                        y2, _mm512_set1_ps(2.0f / 7.0f))))))));

        // Convert exponent to float and multiply by log(2)
        __m512 exponent_f = _mm512_cvtepi32_ps(exponent_adj);
        __m512 result = _mm512_add_ps(
            log_mantissa,
            _mm512_mul_ps(exponent_f, _mm512_set1_ps(0.6931471805599453f)));

        // Handle non-positive inputs
        result = _mm512_mask_blend_ps(
            mask_nonpos, result,
            _mm512_set1_ps(-std::numeric_limits<float>::infinity()));

        _mm512_store_ps(data + i, result);
    }

    // Process remaining elements
    for (; i < count; i++) {
        data[i] = fast_log(data[i]);
    }
#elif defined(BREZEL_SIMD_AVX2)
    // Process 8 elements at a time with AVX2
    size_t vec_count = count / 8;
    size_t i = 0;

    for (; i < vec_count * 8; i += 8) {
        __m256 x = _mm256_load_ps(data + i);

        // Handle non-positive values
        __m256 zero = _mm256_setzero_ps();
        __m256 mask_nonpos = _mm256_cmp_ps(x, zero, _CMP_LE_OS);

        // Extract mantissa and exponent using bit manipulation
        __m256i xi = _mm256_castps_si256(x);
        __m256i exponent = _mm256_srli_epi32(
            _mm256_and_si256(xi, _mm256_set1_epi32(0x7F800000)), 23);
        __m256i mantissa_bits =
            _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF));
        __m256 mantissa = _mm256_or_ps(_mm256_castsi256_ps(mantissa_bits),
                                       _mm256_set1_ps(1.0f));

        // Scale mantissa to [1, 2) range
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 scale_mask = _mm256_cmp_ps(mantissa, one, _CMP_LT_OS);
        __m256 scaled_mantissa = _mm256_mul_ps(mantissa, _mm256_set1_ps(2.0f));
        mantissa = _mm256_blendv_ps(mantissa, scaled_mantissa, scale_mask);

        __m256i one_i = _mm256_set1_epi32(1);
        __m256i adj_exponent = _mm256_sub_epi32(exponent, one_i);
        exponent = _mm256_blendv_epi8(exponent, adj_exponent,
                                      _mm256_castps_si256(scale_mask));

        // y = (mantissa - 1) / (mantissa + 1)
        __m256 num = _mm256_sub_ps(mantissa, one);
        __m256 denom = _mm256_add_ps(mantissa, one);
        __m256 y = _mm256_div_ps(num, denom);

        __m256 y2 = _mm256_mul_ps(y, y);

        // Polynomial approximation
        __m256 poly_term1 = _mm256_set1_ps(2.0f / 3.0f);
        __m256 poly_term2 = _mm256_set1_ps(2.0f / 5.0f);
        __m256 poly_term3 = _mm256_set1_ps(2.0f / 7.0f);

        __m256 poly = _mm256_add_ps(
            _mm256_set1_ps(2.0f),
            _mm256_mul_ps(
                y2,
                _mm256_add_ps(
                    poly_term1,
                    _mm256_mul_ps(
                        y2, _mm256_add_ps(poly_term2,
                                          _mm256_mul_ps(y2, poly_term3))))));

        __m256 log_mantissa = _mm256_mul_ps(y, poly);

        // Convert exponent to float and multiply by log(2)
        __m256 exponent_f = _mm256_cvtepi32_ps(exponent);
        __m256 result = _mm256_add_ps(
            log_mantissa,
            _mm256_mul_ps(exponent_f, _mm256_set1_ps(0.6931471805599453f)));

        // Handle non-positive inputs
        __m256 neg_inf =
            _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        result = _mm256_blendv_ps(result, neg_inf, mask_nonpos);

        _mm256_store_ps(data + i, result);
    }

    // Process remaining elements
    for (; i < count; i++) {
        data[i] = fast_log(data[i]);
    }
#endif
}
#endif
}  // namespace brezel::math