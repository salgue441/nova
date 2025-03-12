#pragma once

#include <brezel/core/macros.hpp>
#include <complex>
#include <concepts>
#include <type_traits>

namespace brezel::tensor::utils {
/**
 * @brief Concept defining valid scalar types for tensor data
 *
 * @details Restricts tensor element types to arithmetic types (integers,
 * floating-point) and boolean values. This ensures that operations like
 * addition, multiplication, etc. are well-defined for tensor elements.
 *
 * @tparam T The type being checked against the concept
 */
template <typename T>
concept TensorScalar = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

/**
 * @brief Concept defining types for which tensor math operations are
 * well-defined
 *
 * @details Restricts math operations to numeric types, excluding boolean values
 * which don't have well-defined mathematical operations like division, sqrt,
 * etc.
 *
 * @tparam T The type being checked against the concept
 */
template <typename T>
concept TensorNumeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

/**
 * @brief Concept defining floating-point types for advanced math operations
 *
 * @details Restricts certain operations like trigonometric functions,
 * exponentials, logarithms, etc. to floating-point types where these operations
 * are well-defined.
 *
 * @tparam T The type being checked against the concept
 */
template <typename T>
concept TensorFloat = std::is_floating_point_v<T>;

/**
 * @brief Concept for complex number types
 *
 * @tparam T The type being checked
 */
template <typename T>
concept TensorComplex = std::is_same_v<T, std::complex<float>> ||
                        std::is_same_v<T, std::complex<double>> ||
                        std::is_same_v<T, std::complex<long double>>;

/**
 * @brief Concept combining all tensor-compatible types
 *
 * @tparam T The type being checked
 */
template <typename T>
concept TensorType = TensorScalar<T> || TensorComplex<T>;

/**
 * @brief Type traits for tenso data types
 *
 * @tparam T Data type
 */
template <typename T>
struct TensorTypeTraits {
    static constexpr bool is_scalar = TensorScalar<T>;
    static constexpr bool is_numeric = TensorNumeric<T>;
    static constexpr bool is_float = TensorFloat<T>;
    static constexpr bool is_complex = TensorComplex<T>;

    using scalar_type = T;
    using real_type = T;
    using high_precision_type = std::conditional_t<
        std::is_same_v<T, float>, double,
        std::conditional_t<std::is_integral_v<T>, double, T>>;

    using accumulator_type = high_precision_type;
    using reduction_type = T;

    template <typename U>
    using binary_op_result_type = std::common_type_t<T, U>;
};

/**
 * @brief Specialization for complex number types
 *
 * @tparam T The underlying real type
 */
template <typename T>
struct TensorTypeTraits<std::complex<T>> {
    static constexpr bool is_scalar = false;
    static constexpr bool is_numeric = true;
    static constexpr bool is_float = TensorFloat<T>;
    static constexpr bool is_complex = true;

    using scalar_type = T;
    using real_type = T;
    using high_precision_type =
        std::complex<typename TensorTypeTraits<T>::high_precision_type>;
    using accumulator_type = high_precision_type;
    using reduction_type = std::complex<T>;

    template <typename U>
    using binary_op_result_type =
        std::conditional_t<TensorComplex<U>,
                           std::complex<std::common_type_t<
                               T, typename TensorTypeTraits<U>::real_type>>,
                           std::complex<std::common_type_t<T, U>>>;
};

/**
 * @brief Get the C++ type name as a string
 *
 * @tparam T Type to get name for
 * @return std::string Type name
 */
template <typename T>
BREZEL_NODISCARD constexpr std::string_view type_name() {
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "int8";
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return "int16";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return "uint16";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return "uint32";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return "uint64";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "complex64";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return "complex128";
    } else {
        return "unknown";
    }
}

/**
 * @brief Get the size of a type in bytes
 *
 * @tparam T Type to get size for
 * @return constexpr size_t Size in bytes
 */
template <typename T>
BREZEL_NODISCARD constexpr size_t type_size() {
    return sizeof(T);
}

/**
 * @brief Check if two types are compatible for binary operations
 *
 * @details For scalar operations, types are compatible if they have a common
 * type
 * @details Complex with scalar is fine if the scalar type is compatible
 * @details Complex with complex is fine 
 *
 * @tparam T First type
 * @tparam U Second type
 * @return constexpr bool True if types are compatible
 */
template <typename T, typename U>
BREZEL_NODISCARD constexpr bool types_compatible() {
    if constexpr (TensorScalar<T> && TensorScalar<U>) {
        return true;
    } else if constexpr (TensorComplex<T> && TensorScalar<U>) {
        return TensorFloat<U> || std::is_integral_v<U>;
    } else if constexpr (TensorScalar<T> && TensorComplex<U>) {
        return TensorFloat<T> || std::is_integral_v<T>;
    } else if constexpr (TensorComplex<T> && TensorComplex<U>) {
        return true;
    } else {
        return false;
    }
}
}  // namespace brezel::tensor::utils