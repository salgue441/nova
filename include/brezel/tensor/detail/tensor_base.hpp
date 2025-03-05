#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/layout.hpp>
#include <brezel/tensor/shape.hpp>
#include <concepts>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace brezel::tensor {
template <TensorScalar T>
class Tensor;

template <TensorScalar T>
class TensorView;

namespace detail {
/**
 * @brief Base class for tensor operations providing a common interface for
 * different tensor implementations
 *
 * @tparam T The scalar type of the tensor elements. Must satisfy TensorScalar
 * concept
 *
 * This class serves as an abstract base class defining the core interface and
 * common functionality for tensor operations. It provides:
 * - Core tensor properties (shape, strides, layout, data access)
 * - Common tensor properties (dimensionality, number of elements, device type,
 * contiguity)
 * - Convenience methods for element access
 * - Type identification utilities
 *
 * The class uses standard C++ type aliases for consistent type handling:
 * - value_type: The underlying scalar type
 * - pointer/const_pointer: Pointer types for data access
 * - reference/const_reference: Reference types for element access
 * - size_type: For size and index representations
 * - difference_type: For pointer arithmetic
 *
 * All derived classes must implement the pure virtual methods defined in the
 * core interface. Common derived properties provide default implementations
 * that can be overridden if needed.
 *
 * Usage example:
 * @code
 * class DenseTensor : public TensorBase<float> {
 *     // Implementation of pure virtual methods
 * };
 * @endcode
 */
template <TensorScalar T>
class TensorBase {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    virtual ~TensorBase() = default;

    // Core interface methods
    virtual const Shape& shape() const noexcept = 0;
    virtual std::span<const int64_t> strides() const noexcept = 0;
    virtual const LayoutDescriptor& layout() const noexcept = 0;
    virtual pointer data() noexcept = 0;
    virtual const_pointer data() const noexcept = 0;
    virtual reference at(std::span<const int64_t> indices) = 0;
    virtual const_reference at(std::span<const int64_t> indices) const = 0;
    virtual Tensor<T> clone() const = 0;

    // Common derived properties
    virtual size_t ndim() const noexcept { return shape().size(); }
    virtual size_t numel() const noexcept { return shape().numel(); }
    virtual DeviceType device() const noexcept { return layout().device(); }
    virtual bool is_contiguous() const noexcept {
        return layout().is_contiguous();
    }

    // Common convenience methods
    BREZEL_NODISCARD inline reference at(
        std::initializer_list<int64_t> indices) {
        return at(std::span<const int64_t>(indices.begin(), indices.size()));
    }

    BREZEL_NODISCARD inline const_reference at(
        std::initializer_list<int64_t> indices) const {
        return at(std::span<const int64_t>(indices.begin(), indices.size()));
    }

    // Type identification
    BREZEL_NODISCARD constexpr std::string_view dtype_name() const noexcept {
        if constexpr (std::is_same_v<T, float>)
            return "float32";

        else if constexpr (std::is_same_v<T, double>)
            return "float64";

        else if constexpr (std::is_same_v<T, int8_t>)
            return "int8";

        else if constexpr (std::is_same_v<T, int16_t>)
            return "int16";

        else if constexpr (std::is_same_v<T, int32_t>)
            return "int32";

        else if constexpr (std::is_same_v<T, int64_t>)
            return "int64";

        else if constexpr (std::is_same_v<T, uint8_t>)
            return "uint8";

        else if constexpr (std::is_same_v<T, uint16_t>)
            return "uint16";

        else if constexpr (std::is_same_v<T, uint32_t>)
            return "uint32";

        else if constexpr (std::is_same_v<T, uint64_t>)
            return "uint64";

        else if constexpr (std::is_same_v<T, bool>)
            return "bool";

        else
            return "unknown";
    }
};
}  // namespace detail
}  // namespace brezel::tensor