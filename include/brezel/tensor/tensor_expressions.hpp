#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <boost/container/small_vector.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <brezel/tensor/tensor_concept.hpp>
#include <brezel/tensor/tensor_simd.hpp>

namespace brezel::tensor {
// Forward declarations
template <TensorScalar T>
class Tensor;

/**
 * @brief Base class for all tensor expressions using CRPT pattern
 * @details This is the core of the expression template system that enables
 * lazy evaluation of complex tensor operations without temporary allocations.
 *
 * @tparam E The derived expression type
 * @tparam T The scalar type of the tensor elements
 */
template <typename E, typename T>
class TensorExpression {
public:
    /**
     * @brief Gets the actual derived expression
     * @return Returns the derived expression
     */
    const E& self() const { return static_cast<const E&>(*this); }

    // Size information
    /**
     * @brief Gets the total number of elements in the expression
     * @return Total number of elements
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const {
        return self().numel();
    }

    /**
     * @brief Gets the shape of the expression result
     * @return Reference to the shape
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const {
        return self().shape();
    }

    /**
     * @brief Access element at a linear index
     *
     * @param i Linear index
     * @return Element value at the index
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T operator[](size_t i) const {
        return self()[i];
    }

    /**
     * @brief Implicit conversion to Tensor
     * @return Evaluated tensor
     */
    BREZEL_NODISCARD operator Tensor<T>() const { return eval(); }

    /**
     * @brief Evaluate the entire expression into a tensor
     * @details This creates a new tensor and fills it with the result of the
     * expression
     *
     * @return Tensor containing the evaluated result
     */
    BREZEL_NODISCARD Tensor<T> eval() const {
        Tensor<T> result(self().numel());
        const size_t n = self().numel();

        if (self().is_contiguous()) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1024),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  for (size_t i = range.begin();
                                       i < range.end(); ++i) {
                                      result.data()[i] = self()[i];
                                  }
                              });
        } else {
            // Handle non-contiguous expressions
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n, 1024),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        boost::container::small_vector<int64_t, 4> indices(
                            self().shape().size());

                        size_t temp = i;
                        for (int64_t j = self().shape().size() - 1; j >= 0;
                             --j) {
                            indices[j] = temp % self().shape()[j];
                            temp /= self().shape()[j];
                        }

                        // Access through at() for non-contiguous data
                        result.at(indices) = self().at(indices);
                    }
                });
        }

        return result;
    }

    /**
     * @brief Checks if the expression result would be contiguous in memory
     * @return True if the result would be contiguous
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous() const {
        return self().is_contiguous();
    }

    /**
     * @brief Addition operator
     * @param rhs Right-hand side expression
     * @return New expression representing the addition
     */
    template <typename RHS>
    BREZEL_NODISCARD auto operator+(const TensorExpression<RHS, T>& rhs) const {
        return make_binary_op(*this, rhs, std::plus<T>());
    }

    /**
     * @brief Subtraction operator
     * @param rhs Right-hand side expression
     * @return New expression representing the subtraction
     */
    template <typename RHS>
    BREZEL_NODISCARD auto operator-(const TensorExpression<RHS, T>& rhs) const {
        return make_binary_op(*this, rhs, std::minus<T>());
    }

    /**
     * @brief Multiplication operator
     * @param rhs Right-hand side expression
     * @return New expression representing the multiplication
     */
    template <typename RHS>
    BREZEL_NODISCARD auto operator*(const TensorExpression<RHS, T>& rhs) const {
        return make_binary_op(*this, rhs, std::multiplies<T>());
    }

    /**
     * @brief Division operator
     * @param rhs Right-hand side expression
     * @return New expression representing the division
     */
    template <typename RHS>
    BREZEL_NODISCARD auto operator/(const TensorExpression<RHS, T>& rhs) const {
        return make_binary_op(*this, rhs, std::divides<T>());
    }

    /**
     * @brief Unary negation operator
     * @return New expression representing the negation
     */
    BREZEL_NODISCARD auto operator-() const {
        return make_unary_op(*this, std::negate<T>());
    }

    /**
     * @brief Element-wise exponential
     * @return New expression representing the exponential
     */
    BREZEL_NODISCARD auto exp() const {
        return make_unary_op(*this, [](T x) { return std::exp(x); });
    }

    /**
     * @brief Element-wise natural logarithm
     * @return New expression representing the logarithm
     */
    BREZEL_NODISCARD auto log() const {
        return make_unary_op(*this, [](T x) { return std::log(x); });
    }

    /**
     * @brief Element-wise square root
     * @return New expression representing the square root
     */
    BREZEL_NODISCARD auto sqrt() const {
        return make_unary_op(*this, [](T x) { return std::sqrt(x); });
    }

    /**
     * @brief Element-wise absolute value
     * @return New expression representing the absolute value
     */
    BREZEL_NODISCARD auto abs() const {
        return make_unary_op(*this, [](T x) { return std::abs(x); });
    }

    /**
     * @brief Element-wise sine
     * @return New expression representing the sine
     */
    BREZEL_NODISCARD auto sin() const {
        return make_unary_op(*this, [](T x) { return std::sin(x); });
    }

    /**
     * @brief Element-wise cosine
     * @return New expression representing the cosine
     */
    BREZEL_NODISCARD auto cos() const {
        return make_unary_op(*this, [](T x) { return std::cos(x); });
    }

    /**
     * @brief Element-wise tangent
     * @return New expression representing the tangent
     */
    BREZEL_NODISCARD auto tan() const {
        return make_unary_op(*this, [](T x) { return std::tan(x); });
    }
};

/**
 * @brief Expression template for binary operations
 * @details Represents operations like a + b, a * b, etc.
 *
 * @tparam LHS Left-hand side expression type
 * @tparam RHS Right-hand side expression type
 * @tparam T Element type
 * @tparam Op Binary operation functor type
 */
template <typename LHS, typename RHS, typename T, typename Op>
class BinaryOpExpression
    : public TensorExpression<BinaryOpExpression<LHS, RHS, T, Op>, T> {
public:
    /**
     * @brief Construct a binary operation expression
     *
     * @param lhs Left-hand side expression
     * @param rhs Right-hand side expressoin
     * @param op Binary operation
     */
    BinaryOpExpression(const LHS& lhs, const RHS& rhs, Op op)
        : m_lhs(lhs), m_rhs(rhs), m_op(op) {
        if (lhs.shape() == rhs.shape()) {
            m_shape = lhs.shape();
        } else {
            m_shape = broadcast_shapes(lhs.shape(), rhs.shape());
        }

        m_is_contiguous = lhs.is_contiguous() && rhs.is_contiguous() &&
                          lhs.shape() == rhs.shape();
    }

    /**
     * @brief Broadcasting helper that follows numpy/PyTorch broadcasting rules
     *
     * @param a First shape to broadcast
     * @param b Second shape to broadcast
     * @return Broadcasted shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    static Shape broadcast_shapes(const Shape& a, const Shape& b) {
        if (!a.is_broadcastable_with(b)) {
            throw core::error::LogicError(
                "Shapes {} and {} cannot be broadcast together", a.to_string(),
                b.to_string());
        }

        return a.broadcast_with(b);
    }

    /**
     * @brief Get total number of elements
     * @return Number of elements in the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const {
        return m_shape.numel();
    }

    /**
     * @brief Get the shape of the expression
     * @return Shape of the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const {
        return m_shape;
    }

    /**
     * @brief Check if expression result is contiguous
     * @return True if the result would be contiguous in memory
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous() const {
        return m_is_contiguous;
    }

    /**
     * @brief Access element by linear index
     *
     * @param i Linear index
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T operator[](size_t i) const {
        if (m_lhs.shape() == m_rhs.shape()) {
            return m_op(m_lhs[i], m_rhs[i]);
        } else {
            return this->at(linear_to_indices(i, m_shape));
        }
    }

    /**
     * @brief Access element by multi-dimensional indices
     *
     * @param indices Multi-dimensional indices
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T
    at(std::span<const int64_t> indices) const {
        std::vector<int64_t> lhs_indices =
            broadcast_indices(indices, m_lhs.shape());
        std::vector<int64_t> rhs_indices =
            broadcast_indices(indices, m_rhs.shape());

        return m_op(m_lhs.at(lhs_indices), m_rhs.at(rhs_indices));
    }

    /**
     * @brief Apply broadcasting rules to indices
     *
     * @param indices Original indices for target shape
     * @param shape Source shape to broadcast from
     * @return Adjusted indices for the source shape
     */
    static std::vector<int64_t> broadcast_indices(
        const std::vector<int64_t>& indices, const Shape& shape) {
        std::vector<int64_t> result(shape.size());
        size_t offset = indices.size() - shape.size();

        for (size_t i = 0; i < shape.size(); ++i) {
            if (i + offset < indices.size()) {
                result[i] = (shape[i] == 1) ? 0 : indices[i + offset];
            } else {
                result[i] = 0;
            }
        }

        return result;
    }

    /**
     * @brief Convert linear index to multi-dimensional indices
     *
     * @param index Linear index
     * @param shape Target shape
     * @return Multi-dimensional indices
     */
    static std::vector<int64_t> linear_to_indices(size_t index,
                                                  const Shape& shape) {
        std::vector<int64_t> indices(shape.size());
        size_t remaining = index;

        for (int64_t i = shape.size() - 1; i >= 0; --i) {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        return indices;
    }

    /**
     * @brief Evaluate with SIMD optimization when possible
     * @return Tensor containing the result
     */
    Tensor<T> eval_simd() const {
        Tensor<T> result(this->shape());

        if (m_is_contiguous && std::is_same_v<LHS, Tensor<T>> &&
            std::is_same_v<RHS, Tensor<T>> &&
            (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
            const auto& lhs_tensor = static_cast<const Tensor<T>&>(m_lhs);
            const auto& rhs_tensor = static_cast<const Tensor<T>&>(m_rhs);

            if (std::is_same_v<Op, std::plus<T>> ||
                std::is_same_v<Op, std::minus<T>> ||
                std::is_same_v<Op, std::multiplies<T>> ||
                std::is_same_v<Op, std::divides<T>>) {
                detail::apply_simd_op(result.data(), lhs_tensor.data(),
                                      rhs_tensor.data(), this->numel(), m_op,
                                      1024);

                return result;
            }
        }

        return this
            ->TensorExpression<BinaryOpExpression<LHS, RHS, T, Op>, T>::eval();
    }

    /**
     * @brief Evaluate the expression into a tensor
     * @return Tensor containing the result
     */
    BREZEL_NODISCARD Tensor<T> eval() const { return eval_simd(); }

private:
    const LHS& m_lhs;
    const RHS& m_rhs;
    Op m_op;
    Shape m_shape;
    bool m_is_contiguous;
};

/**
 * @brief Expression template for unary operations
 * @details Represents operations like -a, exp(a), etc.
 *
 * @tparam Expr The input expression type
 * @tparam T Element type
 * @tparam Op Unary operation functor type
 */
template <typename Expr, typename T, typename Op>
class UnaryOpExpression
    : public TensorExpression<UnaryOpExpression<Expr, T, Op>, T> {
public:
    /**
     * @brief Construct a unary operation expression
     *
     * @param expr Input expression
     * @param op Unary operation
     */
    UnaryOpExpression(const Expr& expr, Op op) : m_expr(expr), m_op(op) {}

    /**
     * @brief Get total number of elements
     * @return Number of elements in the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const {
        return m_expr.numel();
    }

    /**
     * @brief Get the shape of the expression
     * @return Shape of the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const {
        return m_expr.shape();
    }

    /**
     * @brief Check if expression result is contiguous
     * @return True if the result would be contiguous in memory
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous() const {
        return m_expr.is_contiguous();
    }

    /**
     * @brief Access element by linear index
     *
     * @param i Linear index
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T operator[](size_t i) const {
        return m_op(m_expr[i]);
    }

    /**
     * @brief Access element by multi-dimensional indices
     *
     * @param indices Multi-dimensional indices
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T
    at(std::span<const int64_t> indices) const {
        return m_op(m_expr.at(indices));
    }

private:
    const Expr& m_expr;
    Op m_op;
};

/**
 * @brief Expression template for scalar operations (tensor op scalar or scalar
 * op tensor)
 * @details Represents operations like a + 2, 3 * a, etc.
 *
 * @tparam Expr The tensor expression type
 * @tparam T Element type
 * @tparam Op Binary operation functor type
 * @tparam LeftScalar Whether the scalar is on the left side
 */
template <typename Expr, typename T, typename Op, bool LeftScalar>
class ScalarOpExpression
    : public TensorExpression<ScalarOpExpression<Expr, T, Op, LeftScalar>, T> {
public:
    /**
     * @brief Construct a scalar operation expression
     *
     * @param expr Tensor expression
     * @param scalar Scalar value
     * @param op Binary operation
     */
    ScalarOpExpression(const Expr& expr, T scalar, Op op)
        : m_expr(expr), m_scalar(scalar), m_op(op) {}

    /**
     * @brief Get total number of elements
     * @return Number of elements in the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t numel() const {
        return m_expr.numel();
    }

    /**
     * @brief Get the shape of the expression
     * @return Shape of the expression result
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const Shape& shape() const {
        return m_expr.shape();
    }

    /**
     * @brief Check if expression result is contiguous
     * @return True if the result would be contiguous in memory
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous() const {
        return m_expr.is_contiguous();
    }

    /**
     * @brief Access element by linear index
     *
     * @param i Linear index
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T operator[](size_t i) const {
        if constexpr (LeftScalar) {
            return m_op(m_scalar, m_expr[i]);
        } else {
            return m_op(m_expr[i], m_scalar);
        }
    }

    /**
     * @brief Access element by multi-dimensional indices
     *
     * @param indices Multi-dimensional indices
     * @return Computed element value
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE T
    at(std::span<const int64_t> indices) const {
        if constexpr (LeftScalar) {
            return m_op(m_scalar, m_expr.at(indices));
        } else {
            return m_op(m_expr.at(indices), m_scalar);
        }
    }

private:
    const Expr& m_expr;
    T m_scalar;
    Op m_op;
};

// Helper functions to create expression objects
/**
 * @brief Creates a binary operation expression
 *
 * @tparam LHS Left-hand side expression type
 * @tparam RHS Right-hand side expression type
 * @tparam T Element type
 * @tparam Op Binary operation functor type
 * @param lhs Left-hand side expression
 * @param rhs Right-hand side expression
 * @param op Binary operation
 * @return New binary operation expression
 */
template <typename LHS, typename RHS, typename T, typename Op>
BREZEL_NODISCARD auto make_binary_op(const TensorExpression<LHS, T>& lhs,
                                     const TensorExpression<RHS, T>& rhs,
                                     Op op) {
    return BinaryOpExpression<LHS, RHS, T, Op>(lhs.self(), rhs.self(), op);
}

/**
 * @brief Creates a unary operation expression
 *
 * @tparam Expr Input expression type
 * @tparam T Element type
 * @tparam Op Unary operation functor type
 * @param expr Input expression
 * @param op Unary operation
 * @return New unary operation expression
 */
template <typename Expr, typename T, typename Op>
BREZEL_NODISCARD auto make_scalar_op(const TensorExpression<Expr, T>& expr,
                                     Op op) {
    return UnaryOpExpression<Expr, T, Op>(expr.self(), op);
}

/**
 * @brief Creates a tensor-scalar operation expression
 *
 * @tparam Expr Tensor expression type
 * @tparam T Element type
 * @tparam Op Binary operation functor type
 * @param expr Tensor expression
 * @param scalar Scalar value
 * @param op Binary operation
 * @return New scalar operation expression
 */
template <typename Expr, typename T, typename Op>
BREZEL_NODISCARD auto make_scalar_op(const TensorExpression<Expr, T>& expr,
                                     T scalar, Op op) {
    return ScalarOpExpression<Expr, T, Op, false>(expr.self(), scalar, op);
}

/**
 * @brief Creates a scalar-tensor operation expression
 *
 * @tparam Expr Tensor expression type
 * @tparam T Element type
 * @tparam Op Binary operation functor type
 * @param scalar Scalar value
 * @param expr Tensor expression
 * @param op Binary operation
 * @return New scalar operation expression
 */
template <typename Expr, typename T, typename Op>
BREZEL_NODISCARD auto make_scalar_op_left(T scalar,
                                          const TensorExpression<Expr, T>& expr,
                                          Op op) {
    return ScalarOpExpression<Expr, T, Op, true>(expr.self(), scalar, op);
}

/**
 * @brief Add a scalar to a tensor expression
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param expr Tensor expression
 * @param scalar Scalar value
 * @return New expression representing tensor + scalar
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator+(const TensorExpression<E, T>& expr, T scalar) {
    return make_scalar_op(expr, scalar, std::plus<T>());
}

/**
 * @brief Add a tensor expression to a scalar
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param scalar Scalar value
 * @param expr Tensor expression
 * @return New expression representing scalar + tensor
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator+(T scalar, const TensorExpression<E, T>& expr) {
    return make_scalar_op_left(scalar, expr, std::plus<T>());
}

/**
 * @brief Subtract a scalar from a tensor expression
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param expr Tensor expression
 * @param scalar Scalar value
 * @return New expression representing tensor - scalar
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator-(const TensorExpression<E, T>& expr, T scalar) {
    return make_scalar_op(expr, scalar, std::minus<T>());
}

/**
 * @brief Subtract a tensor expression from a scalar
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param scalar Scalar value
 * @param expr Tensor expression
 * @return New expression representing scalar - tensor
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator-(T scalar, const TensorExpression<E, T>& expr) {
    return make_scalar_op_left(scalar, expr, std::minus<T>());
}

/**
 * @brief Multiply a tensor expression by a scalar
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param expr Tensor expression
 * @param scalar Scalar value
 * @return New expression representing tensor * scalar
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator*(const TensorExpression<E, T>& expr, T scalar) {
    return make_scalar_op(expr, scalar, std::multiplies<T>());
}

/**
 * @brief Multiply a scalar by a tensor expression
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param scalar Scalar value
 * @param expr Tensor expression
 * @return New expression representing scalar * tensor
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator*(T scalar, const TensorExpression<E, T>& expr) {
    return make_scalar_op_left(scalar, expr, std::multiplies<T>());
}

/**
 * @brief Divide a tensor expression by a scalar
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param expr Tensor expression
 * @param scalar Scalar value
 * @return New expression representing tensor / scalar
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator/(const TensorExpression<E, T>& expr, T scalar) {
    return make_scalar_op(expr, scalar, std::divides<T>());
}

/**
 * @brief Divide a scalar by a tensor expression
 *
 * @tparam E Expression type
 * @tparam T Element type
 * @param scalar Scalar value
 * @param expr Tensor expression
 * @return New expression representing scalar / tensor
 */
template <typename E, typename T>
BREZEL_NODISCARD auto operator/(T scalar, const TensorExpression<E, T>& expr) {
    return make_scalar_op_left(scalar, expr, std::divides<T>());
}
}  // namespace brezel::tensor