#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace brezel::tensor {

/**
 * @brief Represents the shape of a tensor.
 * @details Provides a lightweight wrapper around dimensions with validation
 * and common shape operations.
 */
class BREZEL_API Shape {
public:
    using Container = std::vector<int64_t>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;

    /**
     * @brief Creates an empty shape (scalar)
     */
    Shape() = default;

    /**
     * @brief Creates a shape from initializer list
     * @param dims Dimension sizes
     * @throws LogicError if any dimension is negative
     */
    Shape(std::initializer_list<int64_t> dims) {
        m_dims.reserve(dims.size());
        for (auto d : dims) {
            if (d < 0) {
                throw core::error::LogicError(
                    "Negative dimension {} is invalid", d);
            }
            m_dims.push_back(d);
        }
    }

    /**
     * @brief Creates a shape from a range of dimensions
     * @tparam Range Range type supporting std::ranges::range
     * @param dims Range of dimensions
     * @throws LogicError if any dimension is negative
     */
    template <std::ranges::range Range>
    explicit Shape(const Range& dims) {
        using brezel::core::error::LogicError;

        m_dims.reserve(std::ranges::distance(dims));
        for (auto d : dims) {
            const int64_t value = static_cast<int64_t>(d);
            if (value < 0) {
                throw LogicError("Negative dimension {} is invalid", value);
            }
            m_dims.push_back(value);
        }
    }

    // Element access
    BREZEL_NODISCARD int64_t& operator[](size_t idx) { return m_dims[idx]; }
    BREZEL_NODISCARD const int64_t& operator[](size_t idx) const {
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
    BREZEL_NODISCARD auto begin() noexcept { return m_dims.begin(); }
    BREZEL_NODISCARD auto begin() const noexcept { return m_dims.begin(); }
    BREZEL_NODISCARD auto end() noexcept { return m_dims.end(); }
    BREZEL_NODISCARD auto end() const noexcept { return m_dims.end(); }

    // Capacity
    BREZEL_NODISCARD bool empty() const noexcept { return m_dims.empty(); }
    BREZEL_NODISCARD size_t size() const noexcept { return m_dims.size(); }

    /**
     * @brief Computes total number of elements in tensor with this shape
     * @return Product of all dimensions
     */
    BREZEL_NODISCARD size_t numel() const noexcept {
        if (empty())
            return 1;
        return std::accumulate(begin(), end(), size_t{1}, std::multiplies<>());
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

    /**
     * @brief Broadcasts this shape with another
     * @param other Shape to broadcast with
     * @return New broadcasted shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD Shape broadcast_with(const Shape& other) const {
        if (empty())
            return other;
        if (other.empty())
            return *this;
        if (*this == other)
            return *this;

        if (!is_broadcastable_with(other)) {
            throw core::error::LogicError(
                "Shapes {} and {} cannot be broadcast together", to_string(),
                other.to_string());
        }

        Shape result;
        const size_t max_dims = std::max(size(), other.size());
        result.m_dims.reserve(max_dims);

        // Right-to-left processing
        auto it1 = m_dims.rbegin();
        auto it2 = other.m_dims.rbegin();
        const auto end1 = m_dims.rend();
        const auto end2 = other.m_dims.rend();

        while (it1 != end1 && it2 != end2) {
            const int64_t dim1 = *it1;
            const int64_t dim2 = *it2;
            result.m_dims.push_back(dim1 == 1 ? dim2 : dim1);
            ++it1;
            ++it2;
        }

        while (it1 != end1) {
            result.m_dims.push_back(*it1);
            ++it1;
        }

        while (it2 != end2) {
            result.m_dims.push_back(*it2);
            ++it2;
        }

        std::reverse(result.m_dims.begin(), result.m_dims.end());
        return result;
    }

    /**
     * @brief Checks if this shape can be broadcast with another
     * @param other Shape to check
     * @return true if shapes are broadcast-compatible
     */
    BREZEL_NODISCARD bool is_broadcastable_with(const Shape& other) const {
        if (empty() || other.empty() || *this == other)
            return true;

        auto it1 = m_dims.rbegin();
        auto it2 = other.m_dims.rbegin();
        const auto end1 = m_dims.rend();
        const auto end2 = other.m_dims.rend();

        while (it1 != end1 && it2 != end2) {
            if (*it1 != *it2 && *it1 != 1 && *it2 != 1)
                return false;
            ++it1;
            ++it2;
        }

        return true;
    }

    /**
     * @brief Gets string representation of shape
     * @return Shape as string
     */
    BREZEL_NODISCARD std::string to_string() const {
        if (empty())
            return "()";

        std::string result = "(";
        for (size_t i = 0; i < size(); ++i) {
            if (i > 0)
                result += ", ";
            result += std::to_string(m_dims[i]);
        }
        result += ")";
        return result;
    }

    bool operator==(const Shape& other) const noexcept = default;

private:
    Container m_dims;
};

}  // namespace brezel::tensor