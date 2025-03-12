#pragma once

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape/shape.hpp>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace brezel::tensor::shape {

/**
 * @brief Represents the strides of a tensor.
 *
 * @details Handles memory layout and indexing calculation for tensor access.
 * Provides efficient conversion between multi-dimensional and linear indices.
 * Uses small-vector optimization to avoid heap allocations for common tensor
 * ranks (up to 8 dimensions stored inline).
 */
class BREZEL_API Strides {
public:
    using Container = boost::container::small_vector<int64_t, 8>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;
    using reverse_iterator = Container::reverse_iterator;
    using const_reverse_iterator = Container::const_reverse_iterator;

    /**
     * @brief Creates empty strides (for scalar)
     */
    BREZEL_NODISCARD Strides() = default;

    /**
     * @brief Creates strides from shape using C-style (row-major) layout
     *
     * @param shape Tensor shape
     */
    BREZEL_NODISCARD explicit Strides(const Shape& shape) {
        if (shape.empty()) {
            return;
        }

        const size_t ndim = shape.size();
        m_strides.resize(ndim);

        if (ndim == 1) {
            m_strides[0] = 1;
            return;
        }

        m_strides[ndim - 1] = 1;
        for (int64_t i = static_cast<int64_t>(ndim) - 2; i >= 0; --i) {
            m_strides[i] = m_strides[i + 1] * shape[i + 1];
        }
    }

    /**
     * @brief Creates strides from explicit values
     *
     * @param strides Stride values
     */
    BREZEL_NODISCARD Strides(std::initializer_list<int64_t> strides)
        : m_strides(strides) {}

    /**
     * @brief Creates strides from a range of values
     *
     * @tparam Range Range type supporting std::ranges::range
     * @param strides Range of stride values
     */
    template <std::ranges::range Range>
    BREZEL_NODISCARD explicit Strides(const Range& strides) {
        m_strides.reserve(std::ranges::distance(strides));

        for (const auto& stride : strides) {
            m_strides.push_back(static_cast<int64_t>(stride));
        }
    }

    // Element access
    BREZEL_NODISCARD int64_t& operator[](size_t idx) noexcept {
        return m_strides[idx];
    }

    BREZEL_NODISCARD const int64_t& operator[](size_t idx) const noexcept {
        return m_strides[idx];
    }

    BREZEL_NODISCARD int64_t& at(size_t idx) {
        if (idx >= m_strides.size()) {
            throw core::error::LogicError(
                "Index {} out of range for strides with {} dimensions", idx,
                m_strides.size());
        }

        return m_strides[idx];
    }

    BREZEL_NODISCARD const int64_t& at(size_t idx) const {
        if (idx >= m_strides.size()) {
            throw core::error::LogicError(
                "Index {} out of range for strides with {} dimensions", idx,
                m_strides.size());
        }

        return m_strides[idx];
    }

    // Iterators
    BREZEL_NODISCARD iterator begin() noexcept { return m_strides.begin(); }

    BREZEL_NODISCARD const_iterator begin() const noexcept {
        return m_strides.begin();
    }

    BREZEL_NODISCARD const_iterator cbegin() const noexcept {
        return m_strides.cbegin();
    }

    BREZEL_NODISCARD iterator end() noexcept { return m_strides.end(); }

    BREZEL_NODISCARD const_iterator end() const noexcept {
        return m_strides.end();
    }

    BREZEL_NODISCARD const_iterator cend() const noexcept {
        return m_strides.cend();
    }

    BREZEL_NODISCARD reverse_iterator rbegin() noexcept {
        return m_strides.rbegin();
    }

    BREZEL_NODISCARD const_reverse_iterator rbegin() const noexcept {
        return m_strides.rbegin();
    }

    BREZEL_NODISCARD const_reverse_iterator crbegin() const noexcept {
        return m_strides.crbegin();
    }

    BREZEL_NODISCARD reverse_iterator rend() noexcept {
        return m_strides.rend();
    }

    BREZEL_NODISCARD const_reverse_iterator rend() const noexcept {
        return m_strides.rend();
    }

    BREZEL_NODISCARD const_reverse_iterator crend() const noexcept {
        return m_strides.crend();
    }

    // Capacity
    BREZEL_NODISCARD bool empty() const noexcept { return m_strides.empty(); }
    BREZEL_NODISCARD size_t size() const noexcept { return m_strides.size(); }
    BREZEL_NODISCARD size_t ndim() const noexcept { return m_strides.size(); }
    BREZEL_NODISCARD size_t capacity() const noexcept {
        return m_strides.capacity();
    }

    /**
     * @brief Computes linear index from multi-dimensional indices
     *
     * @param indices Array of indices
     * @return Linear memory offset
     * @throws LogicError if number of indices doesn't match stride dimensions
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    get_linear_index(std::span<const int64_t> indices) const {
        if (indices.size() != size()) {
            throw core::error::LogicError("Expected {} indices but got {}",
                                          size(), indices.size());
        }

        if (empty()) {
            return 0;
        }

        // Optimize for common tensor dimensions
        if (size() == 1) {
            return indices[0] * m_strides[0];
        } else if (size() == 2) {
            return indices[0] * m_strides[0] + indices[1] * m_strides[1];
        } else if (size() == 3) {
            return indices[0] * m_strides[0] + indices[1] * m_strides[1] +
                   indices[2] * m_strides[2];
        } else if (size() == 4) {
            return indices[0] * m_strides[0] + indices[1] * m_strides[1] +
                   indices[2] * m_strides[2] + indices[3] * m_strides[3];
        } else {
            size_t offset = 0;
            const size_t n = size();
            size_t i = 0;

            for (; i + 3 < n; i += 4) {
                offset += indices[i] * m_strides[i] +
                          indices[i + 1] * m_strides[i + 1] +
                          indices[i + 2] * m_strides[i + 2] +
                          indices[i + 3] * m_strides[i + 3];
            }

            for (; i < n; ++i) {
                offset += indices[i] * m_strides[i];
            }

            return offset;
        }
    }

    /**
     * @brief Computes linear index from multi-dimensional indices
     *
     * @param indices Initializer list of indices
     * @return Linear memory offset
     * @throws LogicError if number of indices doesn't match stride dimensions
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    get_linear_index(std::initializer_list<int64_t> indices) const {
        return get_linear_index(std::span<const int64_t>(indices));
    }

    /**
     * @brief Computes multi-dimensional indices from linear index
     *
     * @details Calculates indices from right to left (most to least
     * significant)
     *
     * @param linear_index Linear memory offset
     * @param shape Shape of the tensor
     * @return Vector of indices
     * @throws LogicError if shape dimensions doesn't match stride dimensions
     */
    BREZEL_NODISCARD Container get_indices(size_t linear_index,
                                           const Shape& shape) const {
        if (shape.size() != size()) {
            throw core::error::LogicError(
                "Shape dimensions {} don't match stride dimensions {}",
                shape.size(), size());
        }

        Container indices(size(), 0);

        if (size() == 0) {
            return indices;
        }

        if (size() == 1) {
            indices[0] = linear_index / m_strides[0];
            return indices;
        }

        size_t remaining = linear_index;
        for (size_t i = 0; i < size(); ++i) {
            if (m_strides[i] > 0) {
                indices[i] = remaining / m_strides[i];
                remaining %= m_strides[i];
            }
        }

        return indices;
    }

    /**
     * @brief Checks if strides represent a contiguous layout
     *
     * @param shape Associated tensor shape
     * @return true if memory layout is contiguous in row-major order
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous(
        const Shape& shape) const noexcept {
        if (empty()) {
            return true;
        }

        if (size() != shape.size()) {
            return false;
        }

        if (m_strides.back() != 1) {
            return false;
        }

        int64_t expected_stride = 1;
        for (int64_t i = size() - 1; i >= 0; --i) {
            if (m_strides[i] != expected_stride) {
                return false;
            }

            if (i > 0) {
                expected_stride *= shape[i];
            }
        }

        return true;
    }

    /**
     * @brief Checks if strides represent a Fortran-contiguous layout
     * (column-major)
     *
     * @param shape Associated tensor shape
     * @return true if memory layout is contiguous in column-major (F-style)
     * order
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_f_contiguous(
        const Shape& shape) const noexcept {
        if (empty()) {
            return true;
        }

        if (size() != shape.size()) {
            return false;
        }

        if (m_strides.front() != 1) {
            return false;
        }

        int64_t expected_stride = 1;
        for (size_t i = 0; i < size(); ++i) {
            if (m_strides[i] != expected_stride) {
                return false;
            }

            if (i < size() - 1) {
                expected_stride *= shape[i];
            }
        }

        return true;
    }

    /**
     * @brief Creates a C-style (row-major) strides for a given shape
     *
     * @param shape Tensor shape
     * @return Strides object with C-style layout
     */
    BREZEL_NODISCARD static Strides c_contiguous(const Shape& shape) {
        return Strides(shape);
    }

    /**
     * @brief Creates a Fortran-style (column-major) strides for a given
     * shape
     *
     * @param shape Tensor shape
     * @return Strides object with F-style layout
     */
    BREZEL_NODISCARD static Strides f_contiguous(const Shape& shape) {
        if (shape.empty()) {
            return Strides();
        }

        Strides strides;
        strides.m_strides.resize(shape.size());
        strides.m_strides[0] = 1;

        for (size_t i = 1; i < shape.size(); ++i) {
            strides.m_strides[i] = strides.m_strides[i - 1] * shape[i - 1];
        }

        return strides;
    }

    /**
     * @brief Creates strides for a transposed tensor
     *
     * @param shape Original tensor shape
     * @return Strides object for transposed tensor
     */
    BREZEL_NODISCARD static Strides transposed(const Shape& shape) {
        if (shape.size() < 2) {
            return Strides(shape);
        }

        Shape transposed_shape;
        transposed_shape.reserve(shape.size());

        for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
            transposed_shape.push_back(*it);
        }

        return Strides(transposed_shape);
    }

    /**
     * @brief Creates strides for permuted tensor
     *
     * @param shape Original tensor shape
     * @param perm Permutation indices
     * @return Strides object for permuted tensor
     * @throws LogicError if permutation is invalid
     */
    BREZEL_NODISCARD static Strides permuted(const Shape& shape,
                                             const std::vector<size_t>& perm) {
        if (perm.size() != shape.size()) {
            throw core::error::LogicError(
                "Permutation size ({}) does not match shape dimensions ({})",
                perm.size(), shape.size());
        }

        Strides original(shape);
        Strides result;
        result.m_strides.resize(shape.size());

        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] >= shape.size()) {
                throw core::error::LogicError(
                    "Permutation index {} out of range for shape with {} "
                    "dimensions",
                    perm[i], shape.size());
            }

            result.m_strides[i] = original.m_strides[perm[i]];
        }

        return result;
    }

    /**
     * @brief Creates broadcasting strides for a smaller shape to match a
     * larger shape
     *
     * @param shape Original tensor shape
     * @param target_shape Target shape to broadcast to
     * @return Strides object with broadcasting
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD static Strides broadcast(const Shape& shape,
                                              const Shape& target_shape) {
        if (!shape.is_broadcastable_with(target_shape)) {
            throw core::error::LogicError(
                "Shapes {} and {} cannot be broadcast together",
                shape.to_string(), target_shape.to_string());
        }

        Strides result;
        result.m_strides.resize(target_shape.size(), 0);

        if (shape.empty()) {
            return result;
        }

        int64_t shape_idx = shape.size() - 1;
        int64_t target_idx = target_shape.size() - 1;
        Strides orig_strides(shape);

        while (shape_idx >= 0 && target_idx >= 0) {
            if (shape[shape_idx] == target_shape[target_idx]) {
                result.m_strides[target_idx] = orig_strides[shape_idx];
            } else if (shape[shape_idx] == 1) {
                result.m_strides[target_idx] = 0;
            } else {
                throw core::error::LogicError(
                    "Cannot broadcast dimension {} of shape {} to {}",
                    shape_idx, shape[shape_idx], target_shape[target_idx]);
            }

            --shape_idx;
            --target_idx;
        }

        while (target_idx >= 0) {
            result.m_strides[target_idx] = 0;
            --target_idx;
        }

        return result;
    }

    /**
     * @brief Creates strides for a sliced tensor
     *
     * @param original_strides Original tensor strides
     * @param starts Start indices for each dimension
     * @param steps Step size for each dimension
     * @return Strides object for sliced tensor
     * @throws LogicError if slice parameters don't match stride dimensions
     */
    BREZEL_NODISCARD static Strides sliced(const Strides& original_strides,
                                           const std::vector<int64_t>& starts,
                                           const std::vector<int64_t>& steps) {
        if (starts.size() != original_strides.size() ||
            steps.size() != original_strides.size()) {
            throw core::error::LogicError(
                "Slice parameters do not match stride dimensions");
        }

        Strides result;
        result.m_strides.resize(original_strides.size());

        for (size_t i = 0; i < original_strides.size(); ++i) {
            result.m_strides[i] = original_strides.m_strides[i] * steps[i];
        }

        return result;
    }

    // Operators
    /**
     * @brief Equality comparison
     */
    bool operator==(const Strides& other) const noexcept = default;

    // Utility functions
    /**
     * @brief Provides a view into the strides as a span
     * @return std::span<const int64_t> Readonly view of strides
     */
    BREZEL_NODISCARD std::span<const int64_t> as_span() const noexcept {
        return std::span<const int64_t>(m_strides.data(), m_strides.size());
    }

    /**
     * @brief Gets string representation of strides
     * @return Strides as a string
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE std::string to_string() const {
        if (empty()) {
            return "()";
        }

        std::ostringstream oss;
        oss << "(";

        for (size_t i = 0; i < size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }

            oss << m_strides[i];
        }

        oss << ")";
        return oss.str();
    }

private:
    Container m_strides;
};

}  // namespace brezel::tensor::shape