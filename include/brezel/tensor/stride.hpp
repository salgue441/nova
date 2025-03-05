#pragma once

#include <boost/container/small_vector.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <sstream>

namespace brezel::tensor {
/**
 * @brief Represents strides of a tensor
 * @details Handle memory layout and indexing calculations for tensor access.
 * Provides efficient conversion between mutli-dimensional and linear indices.
 */
class BREZEL_API Strides {
public:
    using Container = boost::container::small_vector<int64_t, 4>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;

    /**
     * @brief Creates empty strides
     *
     */
    BREZEL_NODISCARD Strides() = default;

    /**
     * @brief Creates strides from shape using C-style (row-major)
     * @param shape Tensor shape
     */
    BREZEL_NODISCARD explicit Strides(const Shape& shape) {
        if (shape.empty())
            return;

        const size_t ndim = shape.size();
        m_strides.resize(ndim);

        // Compute C-style (row-major) strides
        m_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(ndim) - 2; i >= 0; --i) {
            m_strides[i] = m_strides[i + 1] * shape[i + 1];
        }
    }

    /**
     * @brief Creates strides from explicit values
     * @param strides Stride values
     */
    BREZEL_NODISCARD Strides(std::initializer_list<int64_t> strides)
        : m_strides(strides) {}

    // Element access
    BREZEL_NODISCARD int64_t& operator[](size_t idx) { return m_strides[idx]; }
    BREZEL_NODISCARD const int64_t& operator[](size_t idx) const {
        return m_strides[idx];
    }

    BREZEL_NODISCARD int64_t& at(size_t idx) {
        BREZEL_ENSURE(idx < m_strides.size(),
                      "Index {} out of range for strides with {} dimensions",
                      idx, m_strides.size());

        return m_strides[idx];
    }

    BREZEL_NODISCARD const int64_t& at(size_t idx) const {
        BREZEL_ENSURE(idx < m_strides.size(),
                      "Index {} out of range for strides with {} dimensions",
                      idx, m_strides.size());

        return m_strides[idx];
    }

    // Iterators
    BREZEL_NODISCARD iterator begin() noexcept { return m_strides.begin(); }
    BREZEL_NODISCARD const_iterator begin() const noexcept {
        return m_strides.begin();
    }
    BREZEL_NODISCARD iterator end() noexcept { return m_strides.end(); }
    BREZEL_NODISCARD const_iterator end() const noexcept {
        return m_strides.end();
    }

    // Capacity
    BREZEL_NODISCARD bool empty() const noexcept { return m_strides.empty(); }
    BREZEL_NODISCARD size_t size() const noexcept { return m_strides.size(); }

    /**
     * @brief Computes linear index from multi-dimensional indices
     * @param indices Array of indices
     * @return Linear memory offset
     * @throws LogicError if number of indices doesn't match stride dimensions
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    get_linear_index(std::span<const int64_t> indices) const {
        BREZEL_ENSURE(indices.size() == size(),
                      "Expected {} indices but got {}", size(), indices.size());

        if (empty())
            return 0;

        if (size() == 1)
            return indices[0] * m_strides[0];

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

    /**
     * @brief Checks if strides represent a contiguous layout
     * @param shape Associated tensor shape
     * @return true if memory layout is contiguous
     *
     * For row-major order, strides should decrease from left to right, with
     * the rightmost stride being 1. Each stride should be equal to the product
     * of all the dimensions to its rights.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_contiguous(
        const Shape& shape) const noexcept {
        if (empty())
            return true;

        if (size() != shape.size())
            return false;

        if (m_strides.back() != 1)
            return false;

        int64_t expected_stride = 1;
        for (int64_t i = size() - 1; i >= 0; --i) {
            if (m_strides[i] != expected_stride)
                return false;

            if (i > 0)
                expected_stride *= shape[i];
        }

        return true;
    }

    /**
     * @brief Gets string representation
     * @return Strides as string
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE std::string to_string() const {
        if (empty())
            return "()";

        std::ostringstream oss;
        oss << "(";

        for (size_t i = 0; i < size(); ++i) {
            if (i > 0)
                oss << ", ";

            oss << m_strides[i];
        }

        oss << ")";
        return oss.str();
    }

    // Comparison operators
    bool operator==(const Strides& other) const noexcept = default;

private:
    Container m_strides;
};
}  // namespace brezel::tensor