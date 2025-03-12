#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape/shape.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace brezel::tensor::utils {
/**
 * @brief Namespace containing error checking and handling utilities for tensor
 * operations
 *
 */
namespace error {
/**
 * @brief Check if two shapes are compatible for broadcasting
 *
 * @param lhs First shape
 * @param rhs Second shape
 * @throws core::error::LogicError if shapes cannot be broadcast together
 */
inline void check_broadcast_shapes(const Shape& lhs, const Shape& rhs) {
    if (!lhs.is_broadcastable_with(rhs)) {
        throw core::error::LogicError("Cannot broadcast shapes {} and {}",
                                      lhs.to_string(), rhs.to_string());
    }
}

/**
 * @brief Check if tensor indices are valid
 *
 * @param indices Indices to check
 * @param shape Shape to check against
 * @throws core::error::LogicError if indices are out of bounds
 */
template <typename Indices>
inline void check_indices(const Indices& indices, const Shape& shape) {
    if (indices.size() != shape.size()) {
        throw core::error::LogicError("Expected {} indices but got {}",
                                      shape.size(), indices.size());
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw core::error::LogicError(
                "Index {} out of bounds for dimension {} with size {}",
                indices[i], i, shape[i]);
        }
    }
}

/**
 * @brief Check if shapes are compatible for element-wise operations
 *
 * @param lhs First shape
 * @param rhs Second shape
 * @param op_name Operation name for error message
 * @throws core::error::LogicError if shapes are not compatible
 */
inline void check_elementwise_compatible(const Shape& lhs, const Shape& rhs,
                                         std::string_view op_name) {
    if (lhs.empty() || rhs.empty()) {
        return;
    }

    if (!lhs.is_broadcastable_with(rhs)) {
        throw core::error::LogicError("Incompatible shapres for {}: {} and {}",
                                      op_name, lhs.to_string(),
                                      rhs.to_string());
    }
}

/**
 * @brief Check if shapes are compatible for matrix multiplication
 *
 * @param lhs First shape
 * @param rhs Second shape
 * @throws core::error::LogicError if shapes are incompatible
 */
inline void check_matmul_shapes(const Shape& lhs, const Shape& rhs) {
    if (lhs.size() < 1 || rhs.size() < 1) {
        throw core::error::LogicError(
            "Both tensors must have at least 1 dimension for matmul");
    }

    const int64_t lhs_dim = lhs.size() >= 2 ? lhs[lhs.size() - 1] : lhs[0];
    const int64_t rhs_dim = rhs.size() >= 2 ? rhs[rhs.size() - 2] : rhs[0];

    if (lhs_dim != rhs_dim) {
        throw core::error::LogicError(
            "Incompatible dimensions for matrix multiplication: {} and {}",
            lhs.to_string(), rhs.to_string());
    }
}

/**
 * @brief Check if a dimension is valid for a given shape
 *
 * @param dim Dimension to check
 * @param ndim Number of dimensions
 * @param allow_negative Whether to interpret negative dims as counting from end
 * @return int64_t Normalized dimension (non-negative)
 * @throws core::error::LogicError if dimension is invalid
 */
inline int64_t check_dim(int64_t dim, int64_t ndim,
                         bool allow_negative = true) {
    if (allow_negative && dim < 0) {
        dim += ndim;
    }

    if (dim < 0 || dim >= ndim) {
        throw core::error::LogicError(
            "Dimension {} out of range for tensor with {} dimensions", dim,
            ndim);
    }

    return dim;
}

/**
 * @brief Check if a set of dimensions are valid for a given shape
 *
 * @param dims Dimensions to check
 * @param ndim Number of dimensions
 * @param allow_negative Whether to interpret negative dims as counting from end
 * @return std::vector<int64_t> Normalized dimensions (non-negative)
 * @throws core::error::LogicError if any dimension is invalid
 */
inline std::vector<int64_t> check_dims(const std::vector<int64_t>& dims,
                                       int64_t ndim,
                                       bool allow_negative = true) {
    std::vector<int64_t> result;
    result.reserve(dims.size());

    for (int64_t dim : dims) {
        result.push_back(check_dim(dim, ndim, allow_negative));
    }

    return result;
}

/**
 * @brief Check if a shape is valid (all dimensions non-negative)
 *
 * @param shape Shape to check
 * @throws core::error::LogicError if shape is invalid
 */
inline void check_shape_valid(const Shape& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            throw core::error::LogicError(
                "Invalid shape: dimension {} is negative ({})", i, shape[i]);
        }
    }
}

/**
 * @brief Check if two shapes have the same number of elements
 *
 * @param from Original shape
 * @param to Target shape
 * @throws core::error::LogicError if element counts don't match
 */
inline void check_reshapable(const Shape& from, const Shape& to) {
    if (from.numel() != to.numel()) {
        throw core::error::LogicError(
            "Cannot reshape tensor of shape {} with {} elements to shape {} "
            "with {} elements",
            from.to_string(), from.numel(), to.to_string(), to.numel());
    }
}

/**
 * @brief Check if two tensors have the same shape
 *
 * @param lhs First shape
 * @param rhs Second shape
 * @param op_name Operation name for error message
 * @throws core::error::LogicError if shapes don't match
 */
inline void check_same_shape(const Shape& lhs, const Shape& rhs,
                             std::string_view op_name) {
    if (lhs != rhs) {
        throw core::error::LogicError(
            "Incompatible shapes for {}: expected {}, got {}", op_name,
            lhs.to_string(), rhs.to_string());
    }
}

/**
 * @brief Check if a shape is compatible with a dimension for reduction
 *
 * @param shape Shape to check
 * @param dim Dimension to reduce along
 * @param keepdim Whether to keep reduced dimensions
 * @return Shape Result shape after reduction
 * @throws core::error::LogicError if dimension is out of bounds
 */
inline Shape check_reduction_shape(const Shape& shape, int64_t dim,
                                   bool keepdim = false) {
    if (dim == -1) {
        if (keepdim) {
            Shape result;

            for (size_t i = 0; i < shape.size(); ++i) {
                result.push_back(1);
            }

            return result;
        } else {
            return Shape({1});
        }
    }

    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw core::error::LogicError(
            "Invalid reduction dimension {} for shape with {} dimensions", dim,
            shape.size());
    }

    Shape result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == static_cast<size_t>(dim)) {
            if (keepdim) {
                result.push_back(1);
            }
        } else {
            result.push_back(shape[i]);
        }
    }

    return result;
}

/**
 * @brief Format a list of dimensions as a string
 *
 * @param dims Dimensions to format
 * @return std::string Formatted string
 */
inline std::string format_dims(const std::vector<int64_t>& dims) {
    std::ostringstream oss;
    oss << "(";

    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }

        oss << dims[i];
    }

    oss << ")";
    return oss.str();
}
}  // namespace error
}  // namespace brezel::tensor::utils