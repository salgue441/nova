#pragma once

#include <brezel/core/macros.hpp>
#include <brezel/core/error.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/core/span.hpp>
#include <ranges>
#include <optional>
#include <concepts>
#include <numeric>
#include <fmt/format.h>

namespace brezel::core
{
  template <typename T>
  concept DimensionType = std::integral<T> && !std::same_as<T, bool>;

  /**
   * @class Shape
   * @brief Shape class for tensor dimensions
   *
   * @tparam IndexType Integer type for dimensions
   * @tparam InlineCapacity Number of dimensions to store inline
   */
  template <DimensionType IndexType = int64_t, std::size_t InlineCapacity = 4>
  class Shape
  {
  private:
    using container_type = boost::container::
        small_vector<IndexType, InlineCapacity>;

    container_type m_dims;
    container_type m_strides;
    mutable std::optional<IndexType> m_numel_cache;

  public:
    using index_type = IndexType;
    using size_type = typename container_type::size_type;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    // Constructor
    constexpr Shape() noexcept = default;

    /**
     * @brief Iterator constructor for shape dimensions
     *
     * @tparam It Input iterator type for dimensions
     * @param begin Iterator to the beginning of the dimensions
     * @param end Iterator to the end of the dimensions
     * @version 1.0.0
     */
    template <std::input_iterator It>
    explicit Shape(It begin, It end) : m_dims(begin, end)
    {
      validate_and_compute();
    }

    /**
     * @brief Variadic constructor for shape dimensions
     *
     * @tparam Dims Dimension types (must be convertible to IndexType)
     * @param dims Dimensions of the shape
     * @version 1.0.0
     */
    template <typename... Dims>
      requires(std::convertible_to<Dims, IndexType> && ...)
    explicit constexpr Shape(Dims... dims)
        : m_dims{static_cast<IndexType>(dims)...}
    {
      validate_and_compute();
    }

    /**
     * @brief Construct a shape from a range of dimensions (e.g. std::vector)
     *
     * @param dims Range of dimensions
     * @version 1.0.0
     */
    explicit Shape(std::initializer_list<IndexType> dims) : m_dims(dims)
    {
      validate_and_compute();
    }

    // Move operations
    Shape(Shape &&) noexcept = default;
    Shape &operator=(Shape &&) noexcept = default;

    // Copy operations
    Shape(const Shape &) = default;
    Shape &operator=(const Shape &) = default;

    // Iterator and range support
    /**
     * @brief Gets the begin iterator for the dimensions
     *
     * @return Iterator to the beginning of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto begin() noexcept { return m_dims.begin(); }

    /**
     * @brief Gets the end iterator for the dimensions
     *
     * @return Iterator to the end of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto end() noexcept { return m_dims.end(); }

    /**
     * @brief Gets the const begin iterator for the dimensions
     *
     * @return Iterator to the beginning of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto begin() const noexcept { return m_dims.begin(); }

    /**
     * @brief Gets the const end iterator for the dimensions
     *
     * @return Iterator to the end of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto end() const noexcept { return m_dims.end(); }

    /**
     * @brief Gets the const begin iterator for the dimensions
     *
     * @return Iterator to the beginning of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto cbegin() const noexcept { return m_dims.cbegin(); }

    /**
     * @brief Gets the const end iterator for the dimensions
     *
     * @return Iterator to the end of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] auto cend() const noexcept { return m_dims.cend(); }

    // Dimension access
    /**
     * @brief Gets the number of dimensions in the shape
     *
     * @return constexpr size_type Number of dimensions
     * @version 1.0.0
     */
    [[nodiscard]] constexpr size_type ndim() const noexcept
    {
      return m_dims.size();
    }

    /**
     * @brief Checks if the dimensions vector is empty
     *
     * @return true if the dimensions vector is empty, false otherwise
     * @version 1.0.0
     */
    [[nodiscard]] constexpr bool empty() const noexcept
    {
      return m_dims.empty();
    }

    /**
     * @brief Gets a dimension index from the shape
     *
     * @param idx Index of the dimension
     * @return IndexType Dimension at the given index
     */
    [[nodiscard]] IndexType dim(size_type idx) const
    {
      if (idx >= ndim())
      {
        BREZEL_THROW(fmt::format("Index {} out of bounds (max: {})",
                                 idx, ndim()));
      }

      return m_dims[idx];
    }

    // Element access
    /**
     * @brief Element access with bounds checking in debug mode
     *
     * @param idx Index of the element
     * @return constexpr IndexType Element at the given index
     * @version 2.0.0
     */
    [[nodiscard]] constexpr IndexType operator[](size_type idx) const noexcept
    {
      BREZEL_DEBUG_ONLY(
          if (idx >= ndim()){
              BREZEL_THROW("Index {} out of range (ndim = {})", idx, ndim())});

      return m_dims[idx];
    }

    /**
     * @brief Efficient total elements calculation with caching
     *
     * @return IndexType Total number of elements in the shape
     * @version 2.0.0
     */
    [[nodiscard]] IndexType numel() const noexcept
    {
      if (!m_numel_cache)
      {
        if (empty())
          m_numel_cache = 1;

        else
          m_numel_cache = std::accumulate(
              m_dims.begin(), m_dims.end(), static_cast<IndexType>(1),
              std::multiplies<>());
      }

      return *m_numel_cache;
    }

    // View and spans
    /**
     * @brief Gets a span of the dimensions
     *
     * @return boost::span<const IndexType> Span of the dimensions
     * @version 1.0.0
     */
    [[nodiscard]] boost::span<const IndexType> dims() const noexcept
    {
      return {m_dims.data(), m_dims.size()};
    }

    /**
     * @brief Gets a span of the strides
     *
     * @return boost::span<const IndexType> Span of the strides
     * @version 1.0.0
     */
    [[nodiscard]] boost::span<const IndexType> strides() const noexcept
    {
      return {m_strides.data(), m_strides.size()};
    }

    // Shape manipulation
    /**
     * @brief Reshape operation for the shape dimensions
     *
     * @tparam R Range type for the new dimensions (convertible to IndexType)
     * @param new_dims New dimensions
     * @return Reshaped shape
     * @version 2.0.0
     */
    template <std::ranges::input_range R>
      requires std::convertible_to<std::ranges::range_value_t<R>, IndexType>
    Shape reshape(const R &new_dims) const
    {
      const auto old_numel = numel();
      IndexType new_numel = 1;
      container_type reshaped_dims;

      reshaped_dims.reserve(std::ranges::size(new_dims));

      int64_t inferred_dim_idx = -1;
      size_type i = 0;

      for (const auto &dim : new_dims)
      {
        if (dim == -1)
        {
          BREZEL_CHECK(inferred_dim_idx == -1,
                       "Only one dimension can be inferred (-1)");

          inferred_dim_idx = i;
          reshaped_dims.push_back(1);
        }
        else
        {
          BREZEL_CHECK(dim >= 0,
                       fmt::format("Invalid dimension {} at index {}", dim, i));

          new_numel *= dim;
          reshaped_dims.push_back(static_cast<IndexType>(dim));
        }

        ++i;
      }

      if (inferred_dim_idx >= 0)
      {
        BREZEL_CHECK(new_numel != 0,
                     "Cannot infer dimension with zero elements");

        reshaped_dims[inferred_dim_idx] = old_numel / new_numel;
        BREZEL_CHECK(reshaped_dims[inferred_dim_idx] * new_numel == old_numel,
                     "Cannot reshape tensor: incompatible dimensions");
      }
      else
      {
        BREZEL_CHECK(new_numel == old_numel,
                     fmt::format("Cannot reshape tensor of size {} into size {}",
                                 old_numel, new_numel));
      }

      return Shape(std::move(reshaped_dims));
    }

    /**
     * @brief Transpose operation for the shape dimensions
     *
     * @param dim0 First dimension to swap
     * @param dim1 Second dimension to swap
     * @return Transposed shape
     * @version 1.0.0
     */
    Shape transpose(size_type dim0, size_type dim1) const
    {
      BREZEL_CHECK(
          dim0 < ndim() && dim1 < ndim(),
          fmt::format(
              "Dimensions {} and {} out of range (ndim = {})",
              dim0, dim1, ndim()));

      auto new_dims = m_dims;
      std::swap(new_dims[dim0], new_dims[dim1]);

      auto result = Shape(std::move(new_dims));
      auto new_strides = m_strides;
      std::swap(new_strides[dim0], new_strides[dim1]);

      result.m_strides = std::move(new_strides);
      return result;
    }

    /**
     * @brief Checks if the shape is broadcastable to another shape
     *
     * @tparam OtherIndex Index type of the other shape
     * @param other Other shape to check for broadcast compatibility
     * @return true if the shapes are broadcastable, false otherwise
     * @version 2.0.0
     */
    template <DimensionType OtherIndex>
    [[nodiscard]] bool can_broadcast_with(const Shape<OtherIndex> &other)
        const noexcept
    {
      const auto max_dims = std::max(ndim(), other.ndim());
      for (size_type i = 0; i < max_dims; ++i)
      {
        const auto dim1_idx = ndim() - 1 - i;
        const auto dim2_idx = other.ndim() - 1 - i;

        const auto dim1 = dim1_idx < ndim() ? m_dims[dim1_idx] : 1;
        const auto dim2 = dim2_idx < other.ndim()
                              ? static_cast<IndexType>(other[dim2_idx])
                              : 1;

        if (dim1 != 1 && dim2 != 1 && dim1 != dim2)
          return false;
      }

      return true;
    }

    /**
     * @brief Broadcasts the shape to another shape
     *
     * @tparam OtherIndex Index type of the other shape
     * @param a Shape to broadcast
     * @param b Shape to broadcast to
     * @return Broadcasted shape
     * @version 2.0.0
     */
    template <typename OtherIndex>
    static Shape broadcast_shapes(const Shape &a,
                                  const Shape<OtherIndex> &b)
    {
      BREZEL_CHECK(a.can_broadcast_with(b),
                   "Cannot broadcast shapes with incompatible elements");

      const auto max_dims = std::max(a.ndim(), b.ndim());
      container_type new_dims;

      new_dims.reserve(max_dims);
      for (size_type i = 0; i < max_dims; ++i)
      {
        const auto a_idx = a.ndim() - 1 - i;
        const auto b_idx = b.ndim() - 1 - i;

        const auto dim1 = a_idx < a.ndim() ? a.dim(a_idx) : 1;
        const auto dim_b = b_idx < b.ndim()
                               ? static_cast<IndexType>(b[b_idx])
                               : 1;

        new_dims.push_back(std::max(dim1, dim_b));
      }

      return Shape(std::move(new_dims));
    }

    // Comparison operators
    /**
     * @brief Equality operator for shapes
     *
     * @tparam OtherIndex Index type of the other shape
     * @param other Other shape to compare
     * @return true if the shapes are equal, false otherwise
     * @version 1.0.0
     */
    template <typename OtherIndex>
    [[nodiscard]] bool operator==(const Shape<OtherIndex> &other) const
    {
      return std::ranges::equal(m_dims, other.dims());
    }

    /**
     * @brief Inequality operator for shapes
     *
     * @tparam OtherIndex Index type of the other shape
     * @param other Other shape to compare
     * @return true if the shapes are not equal, false otherwise
     * @version 1.0.0
     */
    template <typename OtherIndex>
    [[nodiscard]] bool operator!=(const Shape<OtherIndex> &other) const
    {
      return !(*this == other);
    }

  private:
    /**
     * @brief Wraps the validate dimensions and compute strides functions
     *
     * @version 1.0.0
     */
    void validate_and_compute()
    {
      validate_dimensions();
      compute_strides();

      m_numel_cache.reset();
    }

    /**
     * @brief Validates the dimensions of the shape
     *
     * @throw RuntimeError if any dimension is negative
     * @version 1.0.0
     */
    void validate_dimensions() const
    {
      for (size_type i = 0; i < m_dims.size(); ++i)
      {
        BREZEL_CHECK(
            m_dims[i] >= 0,
            fmt::format("Dimension {} must be non-negative (got {})",
                        i, m_dims[i]));
      }
    }

    /**
     * @brief Compute strides from dimensions of the shape
     *
     * @version 1.0.0
     */
    void compute_strides()
    {
      m_strides.resize(m_dims.size());
      if (m_dims.empty())
        return;

      m_strides.back() = 1;
      for (int i = static_cast<int>(m_dims.size()) - 2; i >= 0; --i)
        m_strides[i] = m_strides[i + 1] * m_dims[i + 1];
    }
  };

  // Type aliases
  using Shape32 = Shape<int32_t>;
  using Shape64 = Shape<int64_t>;

  // Deduction guides
  template <typename... Dims>
  Shape(Dims...) -> Shape<std::common_type_t<Dims...>>;

  // Helper functions
  /**
   * @brief Converts a shape to a string
   *
   * @tparam IndexType Integer type for dimensions
   * @tparam InlineCapacity Number of dimensions to store inline
   * @param shape Shape to convert to a string
   * @return std::string String representation of the shape
   * @version 1.0.0
   */
  template <typename IndexType, size_t InlineCapacity>
  [[nodiscard]] std::string to_string(
      const Shape<IndexType, InlineCapacity> &shape)
  {
    return fmt::format(
        "Shape({})",
        fmt::join(shape.dims(), ", "));
  }
}