#pragma once

#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/layout.hpp>
#include <brezel/tensor/shape.hpp>
#include <functional>
#include <memory>
#include <span>

namespace brezel::tensor {
template <TensorScalar T>
class Tensor;

template <TensorScalar T>
class TensorView;

namespace detail {
/**
 * @brief Base class for tensor expressions in the tensor framework.
 *
 * This class serves as an abstract base for all tensor expressions, providing
 * a common interface for evaluation and shape information. Tensor expressions
 * represent operations on tensors that can be evaluated lazily.
 *
 * @tparam T The scalar type used in the tensor (must satisfy TensorScalar
 * concept)
 *
 * @details The TensorExpression class defines the core interface that all
 * tensor expressions must implement, including evaluation of elements and shape
 * information. It enables lazy evaluation and expression templates pattern for
 * efficient tensor operations.
 *
 * Key features:
 * - Lazy evaluation through eval() method
 * - Shape and dimensionality information
 * - Virtual interface for polymorphic behavior
 *
 * @note This is an abstract class and cannot be instantiated directly. Derived
 * classes must implement at least eval() and shape() methods.
 *
 * @see TensorScalar
 */
template <TensorScalar T>
class TensorExpression {
public:
    virtual ~TensorExpression() = default;

    // Core evaluation method
    BREZEL_NODISCARD virtual T eval(std::span<const int64_t> indices) const = 0;

    // Shape and dimension information
    BREZEL_NODISCARD virtual Shape shape() const = 0;
    BREZEL_NODISCARD virtual size_t ndim() const { return shape().size(); }
    BREZEL_NODISCARD virtual size_t numel() const { return shape().numel(); }
};

/**
 * @brief A leaf node in the tensor expression tree representing actual tensor
 * data.
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 *
 * This class represents the most basic form of a tensor expression - an actual
 * tensor with data stored in memory. It serves as a leaf node in the expression
 * template tree, providing direct access to tensor data through a specified
 * memory layout.
 *
 * The class maintains a pointer to the raw data and a layout descriptor that
 * defines how the N-dimensional tensor is mapped to linear memory. It
 * implements the tensor expression interface, allowing it to be used in larger
 * tensor expressions.
 *
 * TensorLeafExpression is typically used as the base case in expression
 * templates, representing actual tensor data rather than operations on tensors.
 *
 * @example
 * @code
 * float data[] = {1.0, 2.0, 3.0, 4.0};
 * LayoutDescriptor layout({2, 2}); // 2x2 tensor
 * TensorLeafExpression<float> leaf(data, layout);
 */
template <TensorScalar T>
class TensorLeafExpression : public TensorExpression<T> {
public:
    /**
     * @brief Constructs a TensorLeafExpression with given data and layout.
     *
     * @param data Pointer to the raw tensor data.
     * @param layout The layout descriptor defining the tensor's structure.
     */
    TensorLeafExpression(const T* data, const LayoutDescriptor& layout)
        : m_data(data), m_layout(layout) {}

    /**
     * @brief Evaluates the tensor expression at the specified indices.
     *
     * @param indices A span containing the N-dimensional indices to access the
     * tensor element
     * @return T The value at the specified indices in the tensor
     *
     * This method computes the corresponding linear index from the provided
     * N-dimensional indices using the tensor's memory layout and returns the
     * value at that position in the underlying data array.
     */
    BREZEL_NODISCARD T eval(std::span<const int64_t> indices) const override {
        return m_data[m_layout.get_linear_index(indices)];
    }

    /**
     * @brief Returns the shape of the tensor expression.
     * @return Shape object representing the dimensions of the tensor
     * expression.
     */
    BREZEL_NODISCARD Shape shape() const override { return m_layout.shape(); }

private:
    const T* m_data;
    LayoutDescriptor m_layout;
};

/**
 * @brief A class representing a unary operation on a tensor expression.
 *
 * This class models a unary operation (e.g., negation, absolute value, etc.)
 * applied to a tensor expression. It implements the TensorExpression interface,
 * allowing it to be used in larger tensor expressions.
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @tparam UnaryOp The type of the unary operation to be applied
 *
 * @see TensorExpression
 * @see TensorScalar
 *
 * Example:
 * @code
 * auto tensor = // some tensor expression
 * auto negated = UnaryExpression<float, std::negate<float>>(tensor,
 * std::negate<float>());
 * @endcode
 */
template <TensorScalar T, typename UnaryOp>
class UnaryExpression : public TensorExpression<T> {
public:
    /**
     * @brief Constructs a unary tensor expression with the given operand and
     * operation.
     *
     * @tparam T The data type of the tensor elements
     * @param operand Shared pointer to the tensor expression operand
     * @param op Unary operation to be applied on the operand
     */
    UnaryExpression(std::shared_ptr<TensorExpression<T>> operand, UnaryOp op)
        : m_operand(std::move(operand)), m_op(std::move(op)) {}

    /**
     * @brief Evaluates the unary operation at the given indices
     * @param indices A span containing the indices at which to evaluate the
     * operation
     * @return The result of applying the unary operation to the operand at the
     * given indices
     * @throws None
     */
    BREZEL_NODISCARD T eval(std::span<const int64_t> indices) const override {
        return m_op(m_operand->eval(indices));
    }

    /**
     * @brief Gets the shape of the tensor expression
     * @return Shape object representing the dimensions of the tensor expression
     */
    BREZEL_NODISCARD Shape shape() const override { return m_operand->shape(); }

private:
    std::shared_ptr<TensorExpression<T>> m_operand;
    UnaryOp m_op;
};

template <TensorScalar T, typename BinaryOp>
class BinaryExpression : public TensorExpression<T> {
public:
    /**
     * @brief Constructs a binary expression between two tensor expressions
     *
     * Creates a binary operation between two tensor expressions, computing the
     * resulting shape through broadcasting rules.
     *
     * @tparam T The data type of the tensor elements
     * @param left Left-hand side tensor expression operand
     * @param right Right-hand side tensor expression operand
     * @param op Binary operation to be performed between the tensors
     *
     * @note The resulting shape is calculated using broadcasting rules where
     * smaller tensors are broadcast to match the shape of larger ones where
     * possible
     */
    BinaryExpression(std::shared_ptr<TensorExpression<T>> left,
                     std::shared_ptr<TensorExpression<T>> right, BinaryOp op)
        : m_left(std::move(left)),
          m_right(std::move(right)),
          m_op(std::move(op)) {
        m_result_shape =
            calculate_broadcast_shape(m_left->shape(), m_right->shape());
    }

    /**
     * @brief Evaluates the binary tensor expression at the given indices
     *
     * This function applies the binary operation stored in m_op to the
     * evaluated results of the left and right operands at the given indices.
     * The indices are mapped according to broadcasting rules before evaluation.
     *
     * @param indices Span of indices where the expression should be evaluated
     * @return T Result of applying the binary operation to both operands at the
     * given indices
     * @throws None
     *
     * @see map_indices_for_broadcast
     */
    BREZEL_NODISCARD T eval(std::span<const int64_t> indices) const override {
        auto left_indices = map_indices_for_broadcast(indices, m_left->shape());
        auto right_indices =
            map_indices_for_broadcast(indices, m_right->shape());

        return m_op(m_left->eval(left_indices), m_right->eval(right_indices));
    }

    /**
     * @brief Gets the shape of the tensor expression
     * @return Shape object representing the dimensions of the tensor expression
     */
    BREZEL_NODISCARD Shape shape() const override { return m_result_shape; }

private:
    std::shared_ptr<TensorExpression<T>> m_left;
    std::shared_ptr<TensorExpression<T>> m_right;
    BinaryOp m_op;
    Shape m_result_shape;

    // Helper functions
    /**
     * @brief Calculates the resulting shape when broadcasting two tensors
     * together.
     *
     * This static function determines the output shape when two tensors are
     * broadcast together for element-wise operations. Broadcasting follows
     * standard rules where dimensions must be either equal or one of them must
     * be 1.
     *
     * @param a First tensor shape
     * @param b Second tensor shape
     * @return Shape The resulting broadcast shape
     *
     * @note If either shape is empty, returns the non-empty shape.
     *       Otherwise returns the broadcast-compatible shape of both inputs.
     */
    static Shape calculate_broadcast_shape(const Shape& a, const Shape& b) {
        if (a.empty())
            return b;

        if (b.empty())
            return a;

        return a.broadcast_with(b);
    }

    /**
     * @brief Maps indices from an output tensor to the corresponding indices in
     * a broadcasted input tensor.
     *
     * This function handles the mapping of indices when broadcasting a tensor
     * to a larger shape. It follows numpy-style broadcasting rules where:
     * - Singleton dimensions (size 1) are stretched to match the output size
     * - Dimensions are aligned from right to left
     * - Missing dimensions on the left are treated as size 1
     *
     * @param output_indices Span containing the indices of the output tensor
     * position
     * @param input_shape Shape of the input tensor being broadcasted
     *
     * @return Vector of mapped indices for accessing the input tensor
     *
     * @note For singleton dimensions in the input shape, the corresponding
     * index will be 0
     * @note For dimensions beyond the output tensor's rank, indices will be 0
     */
    std::vector<int64_t> map_indices_for_broadcast(
        std::span<const int64_t> output_indices,
        const Shape& input_shape) const {
        const size_t out_dims = output_indices.size();
        const size_t in_dims = input_shape.size();

        std::vector<int64_t> input_indices(in_dims);
        for (size_t i = 0; i < in_dims; ++i) {
            const size_t out_dim = out_dims - in_dims + i;

            if (out_dim < out_dims) {
                if (input_shape[i] == 1) {
                    input_indices[i] = 0;
                } else {
                    input_indices[i] = output_indices[out_dim];
                }
            } else {
                input_indices[i] = 0;
            }
        }

        return input_indices;
    }
};

/**
 * @brief Creates a unary tensor expression using the provided operand and
 * operation
 *
 * This function creates a shared pointer to a new UnaryExpression object that
 * wraps a tensor expression operand with a unary operation.
 *
 * @tparam T The scalar type of the tensor elements
 * @tparam UnaryOp The type of the unary operation to be applied
 *
 * @param operand The tensor expression to be operated on
 * @param op The unary operation to apply to the tensor expression
 *
 * @return std::shared_ptr<TensorExpression<T>> A shared pointer to the
 * resulting unary expression
 */
template <TensorScalar T, typename UnaryOp>
std::shared_ptr<TensorExpression<T>> make_unary_expr(
    std::shared_ptr<TensorExpression<T>> operand, UnaryOp op) {
    return std::make_shared<UnaryExpression<T, UnaryOp>>(std::move(operand),
                                                         std::move(op));
}

/**
 * @brief Creates a binary expression from two tensor expressions and a binary
 * operation
 *
 * This function creates a shared pointer to a BinaryExpression that combines
 * two tensor expressions using the specified binary operation.
 *
 * @tparam T The scalar type used in the tensor expressions
 * @tparam BinaryOp The type of the binary operation to be performed
 *
 * @param left The left operand tensor expression
 * @param right The right operand tensor expression
 * @param op The binary operation to apply to the operands
 *
 * @return std::shared_ptr<TensorExpression<T>> A shared pointer to the
 * resulting binary expression
 */
template <TensorScalar T, typename BinaryOp>
std::shared_ptr<TensorExpression<T>> make_binary_expr(
    std::shared_ptr<TensorExpression<T>> left,
    std::shared_ptr<TensorExpression<T>> right, BinaryOp op) {
    return std::make_shared<BinaryExpression<T, BinaryOp>>(
        std::move(left), std::move(right), std::move(op));
}
}  // namespace detail
}  // namespace brezel::tensor