#pragma once

#include <nova/core/macros.hpp>
#include <nova/core/error.hpp>
#include <vector>
#include <cstdint>
#include <string>
#include <initializer_list>

namespace nova::core
{
  /**
   * @class Shape
   * @brief Template class representing tensor dimensions and
   *        stride information
   *
   * It is used to store the dimensions of a tensor and the stride
   * information for each dimension. The class provides methods to
   * access the dimensions and strides, as well as to reshape and
   * transpose the shape.
   *
   * @tparam IndexType The type used for dimensions and indices
   * @tparam Allocator Custom allocator for the internal containers
   * @version 1.0.0
   */
  template <
      typename IndexType = int64_t,
      typename Allocator = std::allocator<IndexType>>
  class Shape
  {
    static_assert(std::is_integral_v<IndexType>, "IndexType must be integral");

  public:
    using index_type = IndexType;
    using allocator_type = Allocator;
    using container_type = std::vector<index_type, allocator_type>;
    using size_type = typename container_type::size_type;

    // Constructor
    NOVA_ALWAYS_INLINE Shape() = default;

    /**
     * @brief Construct a new Shape object
     *
     * @tparam Dims Dimension types
     * @param dims The dimensions to use for the shape
     * @version 1.0.0
     */
    template <typename... Dims>
    explicit Shape(Dims... dims) : m_dims{static_cast<index_type>(dims)...}
    {
      compute_strides();
    }

    /**
     * @brief Construct a new Shape object
     *
     * @param dims The dimensions to use for the shape
     * @version 1.0.0
     */
    explicit Shape(const container_type &dims) : m_dims(dims)
    {
      compute_strides();
    }

    /**
     * @brief Construct a new Shape object
     *
     * @param dims The dimensions to use for the shape
     * @version 1.0.0
     */
    explicit Shape(container_type &&dims) noexcept : m_dims(std::move(dims))
    {
      compute_strides();
    }

    // Access methods
    /**
     * @brief Gets the dimensions size
     *
     * @return size_type The number of dimensions
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE size_type ndim() const noexcept { return m_dims.size(); }

    /**
     * @brief Checks if the shape is empty
     *
     * @return true If the shape is empty, false otherwise
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE bool empty() const noexcept { return m_dims.empty(); }

    /**
     * @brief Gets the total number of elements in the shape
     *
     * @return index_type The total number of elements
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE index_type dim(size_type idx) const
    {
      NOVA_CHECK(idx < m_dims.size(),
                 fmt::format("Dimensions index {} out of range (ndim = {})",
                             idx, m_dims.size()));

      return m_dims[idx];
    }

    /**
     * @brief Gets the stride of the shape
     *
     * @return index_type The stride of the shape
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE const container_type &dims() const noexcept
    {
      return m_dims;
    }

    /**
     * @brief Gets the stride of the shape
     *
     * @return index_type The stride of the shape
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE const container_type &strides() const noexcept
    {
      return m_strides;
    }

    // Element access
    /**
     * @brief Operator[] overload for element access
     *
     * @param idx The index of the element
     * @return index_type The element at the given index
     * @since 1.0.0
     */
    NOVA_ALWAYS_INLINE index_type &operator[](size_type idx)
    {
      return m_dims[idx];
    }

    /**
     * @brief Operator[] overload for element access
     *
     * @param idx The index of the element
     * @return index_type The element at the given index
     * @since 1.0.0
     */
    NOVA_ALWAYS_INLINE const index_type &operator[](size_type idx) const
    {
      return m_dims[idx];
    }

    /**
     * @brief Compute total number of elements
     *
     * @return index_type The total number of elements
     * @version 1.0.0
     */
    NOVA_ALWAYS_INLINE index_type numel() const noexcept
    {
      index_type n = 1;

      for (const auto &dim : m_dims)
        n *= dim;

      return n;
    }

    // Shape manipulation
    /**
     * @brief Reshapes the shape object to the specified dimensions
     *
     * @tparam OtherIndexType Index type for the new dimensions
     * @param new_dims The new dimensions to reshape the shape to
     * @return Shape<IndexType, Allocator> The reshaped shape object
     * @version 1.0.0
     */
    template <typename OtherIndexType>
    Shape<IndexType, Allocator> reshape(
        const std::vector<OtherIndexType> &new_dims) const
    {
      const auto old_numel = numel();
      index_type new_numel = 1;
      container_type reshaped_dims;

      reshaped_dims.reserve(new_dims.size());

      int64_t inferred_dim_idx = -1;
      for (size_t i = 0; i < new_dims.size(); ++i)
      {
        if (new_dims[i] == -1)
        {
          NOVA_CHECK(inferred_dim_idx == -1,
                     "Only one dimension can be inferred (-1)");

          inferred_dim_idx = i;
          reshaped_dims.push_back(1);
        }
        else
        {
          NOVA_CHECK(new_dims[i] >= 0,
                     fmt::format("Invalid dimension {} at index {}",
                                 new_dims[i], i));

          new_numel *= new_dims[i];
          reshaped_dims.push_back(static_cast<index_type>(new_dims[i]));
        }
      }

      if (inferred_dim_idx >= 0)
      {
        NOVA_CHECK(new_numel != 0, "Cannot infer dimension with zero elements");

        reshaped_dims[inferred_dim_idx] = old_numel / new_numel;
        NOVA_CHECK(reshaped_dims[inferred_dim_idx] * new_numel == old_numel,
                   "Cannot reshape tensor: incompatible dimensions");
      }
      else
      {
        NOVA_CHECK(new_numel == old_numel,
                   fmt::format("Cannot reshape tensor of size {} into size {}",
                               old_numel, new_numel));
      }

      return Shape(std::move(reshaped_dims));
    }

    /**
     * @brief Transposes the shape object
     *
     * @param dim0 The first dimension to transpose
     * @param dim1 The second dimension to transpose
     * @return Shape<IndexType, Allocator> The transposed shape object
     * @version 1.0.0
     */
    Shape<IndexType, Allocator> transpose(size_type dim0, size_type dim1) const
    {
      NOVA_CHECK(dim0 < ndim() && dim1 < ndim(),
                 fmt::format("Dimensions {} and {} out of range (ndim = {})",
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
     * @brief Checks if the shape can be broadcasted with another shape
     *
     * @tparam OtherIndex Index type for the other shape
     * @tparam OtherAlloc Allocator type for the other shape
     * @param other The other shape to check for broadcast compatibility
     * @return true If the shapes can be broadcasted, false otherwise
     * @version 1.0.0
     */
    template <typename OtherIndex, typename OtherAlloc>
    bool can_broadcast_with(const Shape<OtherIndex, OtherAlloc> &other)
        const noexcept
    {
      const auto max_dims = std::max(ndim(), other.ndim());
      for (size_type i = 0; i < max_dims; ++i)
      {
        const auto dim1_idx = ndim() - 1 - i;
        const auto dim2_idx = other.ndim() - 1 - i;

        const auto dim1 = dim1_idx < ndim() ? m_dims[dim1_idx] : 1;
        const auto dim2 = dim2_idx < other.ndim()
                              ? static_cast<index_type>(other.dims()[dim2_idx])
                              : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
          return false;
      }

      return true;
    }

    /**
     * @brief Broadcasts two shapes together
     *
     * @tparam OtherIndex Index type for the other shape
     * @tparam OtherAlloc Allocator type for the other shape
     * @param other The other shape to broadcast with
     * @return Shape<IndexType, Allocator> The broadcasted shape object
     * @version 1.0.0
     */
    template <typename OtherIndex, typename OtherAlloc>
    static Shape<IndexType, Allocator> broadcast_shapes(
        const Shape<IndexType, Allocator> &a,
        const Shape<OtherIndex, OtherAlloc> &b)
    {
      NOVA_CHECK(a.can_broadcast_with(b),
                 "Cannot broadcast shapes with incompatible dimensions");

      const auto max_dims = std::max(a.ndim(), b.ndim());
      container_type new_dims(max_dims);

      for (size_type i = 0; i < max_dims; ++i)
      {
        const auto a_idx = a.ndim() - 1 - i;
        const auto b_idx = b.ndim() - 1 - i;

        const auto dim_a = a_idx < a.ndim() ? a.dims()[a_idx] : 1;
        const auto dim_b = b_idx < b.ndim()
                               ? static_cast<index_type>(b.dims()[b_idx])
                               : 1;

        new_dims[max_dims - 1 - i] = std::max(dim_a, dim_b);
      }

      return Shape(std::move(new_dims));
    }

    // Comparison operator
    /**
     * @brief Compares two shapes for equality
     *
     * @tparam OtherIndex Index type for the other shape
     * @tparam OtherAlloc Allocator type for the other shape
     * @param other The other shape to compare with
     * @return true If the shapes are equal, false otherwise
     * @version 1.0.0
     */
    template <typename OtherIndex, typename OtherAlloc>
    bool operator==(const Shape<OtherIndex, OtherAlloc> &other) const
    {
      if (ndim() != other.ndim())
        return false;

      for (size_type i = 0; i < ndim(); ++i)
      {
        if (m_dims[i] != static_cast<index_type>(other.dims()[i]))
          return false;
      }

      return true;
    }

    /**
     * @brief Inequality operator for comparing two shapes
     *
     * @tparam OtherIndex Index type for the other shape
     * @tparam OtherAlloc Allocator type for the other shape
     * @param other The other shape to compare with
     * @return true If the shapes are not equal, false otherwise
     * @version 1.0.0
     */
    template <typename OtherIndex, typename OtherAlloc>
    bool operator!=(const Shape<OtherIndex, OtherAlloc> &other) const
    {
      return !(*this == other);
    }

  private:
    container_type m_dims;
    container_type m_strides;

    /**
     * @brief Computes the strides for the shape object
     *
     * @tparam IndexType Index type for the shape
     * @tparam Allocator Allocator type for the shape
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

  // Type aliases for common shape types
  using Shape32 = Shape<int32_t>;
  using Shape64 = Shape<int64_t>;

  // Helper functions
  /**
   * @brief Converts the shape object to a string
   *
   * @tparam IndexType The type used for dimensions and indices
   * @tparam Allocator Custom allocator for the internal containers
   * @param shape The shape object to convert
   * @return std::string The string representation of the shape
   * @version 1.0.0
   */
  template <typename IndexType, typename Allocator>
  std::string to_string(const Shape<IndexType, Allocator> &shape)
  {
    std::string result = "Shape(";

    for (size_t i = 0; i < shape.ndim(); ++i)
    {
      if (i > 0)
        result += ", ";

      result += std::to_string(shape[i]);
    }

    result += ")";
    return result;
  }
} // namespace nova::core