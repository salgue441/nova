#pragma once

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace brezel::tensor::shape {

/**
 * @brief Represents the shape of a tensor.
 *
 * @details Provides a lightweight wrapper around dimensions with validation and
 * common shape operations. The Shape class efficiently represents tensor
 * dimensions and supports broadcasting, reshaping, and other common operators.
 * Small-vector optimization is used to avoid heap allocations for common tensor
 * dimensions (up to 8 dimensions stored inline).
 */
class BREZEL_API Shape {
public:
    using Container = boost::container::small_vector<int64_t, 8>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;
    using reverse_iterator = Container::reverse_iterator;
    using const_reverse_iterator = Container::const_reverse_iterator;

    /**
     * @brief Creates an empty shape (scalar)
     */
    BREZEL_NODISCARD Shape() = default;

    /**
     * @brief Creates a shape from initializer list
     *
     * @param dims Dimension sizes
     * @throws LogicError if any dimension is negative
     */
    BREZEL_NODISCARD Shape(std::initializer_list<int64_t> dims) {
        m_dims.reserve(dims.size());

        for (auto dim : dims) {
            if (dim < 0) {
                throw core::error::LogicError(
                    "Negative dimension {} is invalid", dim);
            }

            m_dims.push_back(dim);
        }
    }

    /**
     * @brief Creates a shape from a range of dimensions
     *
     * @tparam Range Range type supporting std::ranges::range
     * @param dims Range of dimensions
     * @throws LogicError if any dimension is negative
     */
    template <std::ranges::range Range>
    BREZEL_NODISCARD explicit Shape(const Range& dims) {
        m_dims.reserve(std::ranges::distance(dims));

        for (const auto& dim : dims) {
            int64_t value;

            if constexpr (std::is_same_v<std::ranges::range_value_t<Range>,
                                         int64_t>) {
                value = dim;
            } else {
                value = static_cast<int64_t>(dim);
            }

            if (value < 0) {
                throw core::error::LogicError(
                    "Negative dimension {} is invalid", value);
            }

            m_dims.push_back(value);
        }
    }

    // Element access
    BREZEL_NODISCARD int64_t& operator[](size_t idx) noexcept {
        return m_dims[idx];
    }

    BREZEL_NODISCARD const int64_t& operator[](size_t idx) const noexcept {
        return m_dims[idx];
    }

    BREZEL_NODISCARD int64_t& at(size_t idx) {
        if (idx >= m_dims.size()) {
            throw core::error::LogicError(
                "Index {} out of range for shape with {} dimensions", idx,
                m_dims.size());
        }

        return m_dims[idx];
    }

    BREZEL_NODISCARD const int64_t& at(size_t idx) const {
        if (idx >= m_dims.size()) {
            throw core::error::LogicError(
                "Index {} out of range for shape with {} dimensions", idx,
                m_dims.size());
        }

        return m_dims[idx];
    }

    // Iterators
    BREZEL_NODISCARD iterator begin() noexcept { return m_dims.begin(); }

    BREZEL_NODISCARD const_iterator begin() const noexcept {
        return m_dims.begin();
    }

    BREZEL_NODISCARD const_iterator cbegin() const noexcept {
        return m_dims.cbegin();
    }

    BREZEL_NODISCARD iterator end() noexcept { return m_dims.end(); }

    BREZEL_NODISCARD const_iterator end() const noexcept {
        return m_dims.end();
    }

    BREZEL_NODISCARD const_iterator cend() const noexcept {
        return m_dims.cend();
    }

    BREZEL_NODISCARD reverse_iterator rbegin() noexcept {
        return m_dims.rbegin();
    }

    BREZEL_NODISCARD const_reverse_iterator rbegin() const noexcept {
        return m_dims.rbegin();
    }

    BREZEL_NODISCARD const_reverse_iterator crbegin() const noexcept {
        return m_dims.crbegin();
    }

    BREZEL_NODISCARD reverse_iterator rend() noexcept { return m_dims.rend(); }

    BREZEL_NODISCARD const_reverse_iterator rend() const noexcept {
        return m_dims.rend();
    }

    BREZEL_NODISCARD const_reverse_iterator crend() const noexcept {
        return m_dims.crend();
    }

    // Capacity
    BREZEL_NODISCARD bool empty() const noexcept { return m_dims.empty(); }
    BREZEL_NODISCARD size_t size() const noexcept { return m_dims.size(); }
    BREZEL_NODISCARD size_t ndim() const noexcept { return m_dims.size(); }
    BREZEL_NODISCARD size_t capacity() const noexcept {
        return m_dims.capacity();
    }

    /**
     * @brief Computes total number of elements in a tensor with this shape
     * @return Product of all dimensions
     */
    BREZEL_NODISCARD size_t numel() const noexcept {
        if (empty()) {
            return 1;
        }

        return std::accumulate(
            begin(), end(), size_t{1},
            [](size_t acc, int64_t dim) { return acc * dim; });
    }

    // Modifiers
    void clear() noexcept { m_dims.clear(); }

    void push_back(int64_t dim) {
        if (dim < 0) {
            throw core::error::LogicError("Negative dimension {} is invalid",
                                          dim);
        }

        m_dims.push_back(dim);
    }

    void pop_back() {
        if (empty()) {
            throw core::error::LogicError("Cannot pop from empty shape");
        }

        m_dims.pop_back();
    }

    void reserve(size_t n) { m_dims.reserve(n); }
    void resize(size_t n, int64_t value = 0) { m_dims.resize(n, value); }

    /**
     * @brief Checks if this shape can be broadcast with another
     *
     * @param other Shape to check
     * @return true if shapes are broadcast-compatible, false otherwise
     */
    BREZEL_NODISCARD bool is_broadcastable_with(const Shape& other) const {
        if (empty() || other.empty() || *this == other) {
            return true;
        }

        auto it1 = m_dims.rbegin();
        auto it2 = other.m_dims.rbegin();
        const auto end1 = m_dims.rend();
        const auto end2 = other.m_dims.rend();

        while (it1 != end1 && it2 != end2) {
            if (*it1 != *it2 && *it1 != 1 && *it2 != 1) {
                return false;
            }

            ++it1;
            ++it2;
        }

        return true;
    }

    /**
     * @brief Broadcasts this shape with another
     *
     * @param other Shape to broadcast with
     * @return New broadcasted shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD Shape broadcast_with(const Shape& other) const {
        if (empty()) {
            return other;
        }

        if (other.empty()) {
            return *this;
        }

        if (*this == other) {
            return *this;
        }

        if (!is_broadcastable_with(other)) {
            throw core::error::LogicError(
                "Shapes {} and {} cannot be broadcast together", to_string(),
                other.to_string());
        }

        const size_t max_dims = std::max(size(), other.size());

        Shape result;
        result.reserve(max_dims);

        std::vector<int64_t> padded1(max_dims, 1);
        std::vector<int64_t> padded2(max_dims, 1);

        std::copy(m_dims.begin(), m_dims.end(), padded1.end() - m_dims.size());
        std::copy(other.m_dims.begin(), other.m_dims.end(),
                  padded2.end() - other.m_dims.size());

        for (size_t i = 0; i < max_dims; ++i) {
            if (padded1[i] == padded2[i]) {
                result.push_back(padded1[i]);
            } else if (padded1[i] == 1) {
                result.push_back(padded2[i]);
            } else if (padded2[i] == 1) {
                result.push_back(padded1[i]);
            } else {
                throw core::error::LogicError(
                    "Incompatible dimensions for broadcasting: {} and {}",
                    padded1[i], padded2[i]);
            }
        }

        return result;
    }

    /**
     * @brief Creates a permuted version of this shape
     *
     * @param perm Permutation indices
     * @return Shape New shape with permuted dimensions
     * @throws LogicError if permutation is invalid
     */
    BREZEL_NODISCARD Shape permute(const std::vector<size_t>& perm) const {
        if (perm.size() != size()) {
            throw core::error::LogicError(
                "Permutation size ({}) does not match shape dimensions ({})",
                perm.size(), size());
        }

        for (auto idx : perm) {
            if (idx >= size()) {
                throw core::error::LogicError(
                    "Permutation index {} out of range for shape with {} "
                    "dimensions",
                    idx, size());
            }
        }

        Shape result;
        result.m_dims.reserve(size());

        for (auto idx : perm) {
            result.m_dims.push_back(m_dims[idx]);
        }

        return result;
    }

    /**
     * @brief Squeeze dimensions of size 1 from the shape
     *
     * @param dim Optional specific dimension to squeeze;
     *            if -1, squeeze all dims of size 1
     * @return Shape New shape with squeezed dimensions
     * @throws LogicError if specific dimension is not 1 or is out of range
     */
    BREZEL_NODISCARD Shape squeeze(int64_t dim = -1) const {
        if (dim >= static_cast<int64_t>(size())) {
            throw core::error::LogicError(
                "Dimension {} out of range for shape with {} dimensions", dim,
                size());
        }

        if (dim >= 0) {
            if (m_dims[dim] != 1) {
                throw core::error::LogicError(
                    "Cannot squeeze dimension {} with size {}, must be 1", dim,
                    m_dims[dim]);
            }

            Shape result;
            result.m_dims.reserve(size() - 1);

            for (size_t i = 0; i < size(); ++i) {
                if (i != static_cast<size_t>(dim)) {
                    result.m_dims.push_back(m_dims[i]);
                }
            }

            return result;
        } else {
            Shape result;
            result.m_dims.reserve(size());

            for (auto dim : m_dims) {
                if (dim != 1) {
                    result.m_dims.push_back(dim);
                }
            }

            return result;
        }
    }

    /**
     * @brief Unsqueeze by adding a dimension of size 1
     *
     * @param dim Position to insert the new dimension
     * @return Shape New shape with added dimension
     * @throws LogicError if dimension is out of range
     */
    BREZEL_NODISCARD Shape unsqueeze(int64_t dim) const {
        if (dim < 0) {
            dim += static_cast<int64_t>(size()) + 1;
        }

        if (dim < 0 || dim > static_cast<int64_t>(size())) {
            throw core::error::LogicError(
                "Dimension {} out of range for unsqueezing shape with {} "
                "dimensions",
                dim, size());
        }

        Shape result;
        result.m_dims.reserve(size() + 1);

        for (size_t i = 0; i < size() + 1; ++i) {
            if (i == static_cast<size_t>(dim)) {
                result.m_dims.push_back(1);
            }

            if (i < size()) {
                result.m_dims.push_back(m_dims[i]);
            }
        }

        return result;
    }

    // Operators
    /**
     * @brief Equality comparison
     *
     * @param other Tensor to compare with
     * @return true if they're equal, false otherwise
     */
    bool operator==(const Shape& other) const noexcept = default;

    // Utility functions
    /**
     * @brief Checks if shape can be reshaped to another shape
     *
     * @param other Target shape
     * @return true if shapes have same number of elements, false otherwise
     */
    BREZEL_NODISCARD bool is_reshapable_to(const Shape& other) const noexcept {
        return numel() == other.numel();
    }

    /**
     * @brief Gets string representation of shape
     * @return Shape as a string
     */
    BREZEL_NODISCARD std::string to_string() const {
        if (empty()) {
            return "()";
        }

        std::string result = "(";
        for (size_t i = 0; i < size(); ++i) {
            if (i > 0) {
                result += ", ";
            }

            result += std::to_string(m_dims[i]);
        }

        result += ")";
        return result;
    }

    /**
     * @brief Provides a view into the shape as a span
     * @return std::span<const int64_t> Readonly view of dimensions
     */
    BREZEL_NODISCARD std::span<const int64_t> as_span() const noexcept {
        return std::span<const int64_t>(m_dims.data(), m_dims.size());
    }

private:
    Container m_dims;
};

}  // namespace brezel::tensor::shape