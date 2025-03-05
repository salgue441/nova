#pragma once

#include <concepts>
#include <type_traits>

namespace brezel::tensor {
/**
 * @brief Concept defininig valid scalar types for tensor data
 *
 * @details Restrict tensor element types to arithmetic types (integers,
 * floating-point) and boolean values. This ensures that operations like
 * addition, multiplication, etc. are well-defined for tensor elements
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
 * @brief Helper for constraining operations to specific data types
 *
 * @details Provides static assertions and compile-time type checking for
 * operations that have specific type requirements.
 */
template <typename T>
struct TypeCheck {
    static constexpr bool is_scalar = TensorScalar<T>;
    static constexpr bool is_numeric = TensorNumeric<T>;
    static constexpr bool is_float = TensorFloat<T>;

    static_assert(is_scalar, "Type must satisfy TensorScalar concept");

    template <typename Op>
    static constexpr bool supports_operation() {
        if constexpr (std::is_same_v<Op, std::plus<T>> ||
                      std::is_same_v<Op, std::minus<T>> ||
                      std::is_same_v<Op, std::multiplies<T>>) {
            return is_numeric;

        } else if constexpr (std::is_same_v<Op, std::divides<T>>) {
            return is_numeric;

        } else {
            return is_float;
        }
    }
};
}  // namespace brezel::tensor