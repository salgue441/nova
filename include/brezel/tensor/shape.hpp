#pragma once

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <numeric>
#include <ranges>
#include <span>
#include <sstream>
#include <string>

namespace brezel::tensor {
/**
 * @brief Represents the shape of a tensor.
 * @details Provides a lightweight wrapper around dimensions with validation
 * and common shape operations. Uses small vector optimization for efficiency.
 */
class BREZEL_API Shape {
public:
    using Container = boost::container::small_vector<int64_t, 4>;
    using iterator = Container::iterator;
    using const_iterator = Container::const_iterator;

    /**
     * @brief Creates an empty shape (scalar)
     *
     */
    BREZEL_NODISCARD Shape() = default;

    /**
     * @brief Creates a shape from initializer list
     *
     * @param dims Dimension sizes
     * @throws LogicError if any dimension is negative
     */
    BREZEL_NODISCARD Shape(std::initializer_list<int64_t> dims) {
        for (auto dim : dims)
            BREZEL_ENSURE(dim >= 0, "Negative dimension {} is invalid", dim);

        m_dims.insert(m_dims.end(), dims.begin(), dims.end());
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
        std::vector<int64_t> converted;
        converted.reserve(std::ranges::distance(dims));

        for (auto dim : dims) {
            const int64_t value = static_cast<int64_t>(dim);
            BREZEL_ENSURE(value >= 0, "Negative dimension {} is invalid", dim);

            converted.push_back(value);
        }

        m_dims.insert(m_dims.end(), converted.begin(), converted.end());
    }

    // Element access
    BREZEL_NODISCARD int64_t& operator[](size_t idx) { return m_dims[idx]; }
    BREZEL_NODISCARD const int64_t& operator[](size_t idx) const {
        return m_dims[idx];
    }

    BREZEL_NODISCARD int64_t& at(size_t idx) {
        BREZEL_ENSURE(idx < m_dims.size(),
                      "Index {} out of range for shape with {} dimensions", idx,
                      m_dims.size());

        return m_dims[idx];
    }

    BREZEL_NODISCARD const int64_t& at(size_t idx) const {
        BREZEL_ENSURE(idx < m_dims.size(),
                      "Index {} out of range for shape with {} dimensions", idx,
                      m_dims.size());

        return m_dims[idx];
    }

    // Iterators
    BREZEL_NODISCARD iterator begin() noexcept { return m_dims.begin(); }
    BREZEL_NODISCARD const_iterator begin() const noexcept {
        return m_dims.begin();
    }

    BREZEL_NODISCARD iterator end() noexcept { return m_dims.end(); }
    BREZEL_NODISCARD const_iterator end() const noexcept {
        return m_dims.end();
    }

    // Capacity
    BREZEL_NODISCARD bool empty() const noexcept { return m_dims.empty(); }
    BREZEL_NODISCARD size_t size() const noexcept { return m_dims.size(); }

    /**
     * @brief Computes total number of elements in tensor with this shape
     * @return Product of all dimensions
     */
    BREZEL_NODISCARD size_t numel() const noexcept {
        return std::accumulate(begin(), end(), size_t{1}, std::multiplies<>());
    }

    // Modifiers
    void clear() noexcept { m_dims.clear(); }
    void push_back(int64_t dim) {
        BREZEL_ENSURE(dim >= 0, "Negative dimension {} is invalid", dim);
        m_dims.push_back(dim);
    }

    void pop_back() {
        BREZEL_ENSURE(!empty(), "Cannot pop from empty shape");
        m_dims.pop_back();
    }

    /**
     * @brief Broadcasts this shape with another
     *
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

        BREZEL_ENSURE(is_broadcastable_with(other),
                      "Shapes {} and {} cannot be broadcast together",
                      to_string(), other.to_string());

        const size_t max_dims = std::max(size(), other.size());
        Shape result;

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

        while (it2 != end1) {
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
     *
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
     * @brief Gets string representation
     * @return Shape as string
     */
    BREZEL_NODISCARD std::string to_string() const {
        if (empty())
            return "()";

        std::ostringstream oss;
        oss << "(";

        for (size_t i = 0; i < size(); ++i) {
            if (i > 0)
                oss << ", ";

            oss << m_dims[i];
        }

        oss << ")";
        return oss.str();
    }

    // Comparison operator
    bool operator==(const Shape& other) const noexcept = default;

private:
    Container m_dims;
};
}  // namespace brezel::tensor