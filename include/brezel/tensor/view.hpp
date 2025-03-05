#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_base.hpp>
#include <brezel/tensor/detail/tensor_iterator.hpp>
#include <brezel/tensor/tensor_impl.hpp>
#include <memory>
#include <span>

namespace brezel::tensor {
template <TensorScalar T>
class TensorView final : public detail::TensorBase<T> {
public:
    using typename detail::TensorBase<T>::value_type;
    using typename detail::TensorBase<T>::pointer;
    using typename detail::TensorBase<T>::const_pointer;
    using typename detail::TensorBase<T>::reference;
    using typename detail::TensorBase<T>::const_reference;
    using typename detail::TensorBase<T>::size_type;
    using typename detail::TensorBase<T>::difference_type;

    using iterator = detail::TensorIterator<T>;
    using const_iterator = detail::TensorIterator<const T>;

    // Constructors
    /**
     * @brief Constructs a view of an existing tensor with a new layout.
     *
     * A TensorView provides a different way to access the data of an existing
     * tensor without copying the underlying data. The view can reorganize how
     * the data is accessed through a new layout, but cannot expand beyond the
     * original tensor's size.
     *
     * @param base_tensor The source tensor to create a view from
     * @param new_layout The layout descriptor defining how to interpret the
     * tensor's data
     * @throws core::error::InvalidArgument if the new layout requires more
     * elements than available in the base tensor
     */
    TensorView(const Tensor<T>& base_tensor, LayoutDescriptor new_layout)
        : m_base_tensor(base_tensor), m_layout(std::move(new_layout)) {
        if (new_layout.numel() > base_tensor.numel()) {
            throw core::error::InvalidArgument(
                "View size ({}) cannot exceed base tensor size ({})",
                new_layout.numel(), base_tensor.numel());
        }
    }

    TensorView() = delete;
    TensorView(const TensorView& other) = default;
    TensorView(TensorView&& other) noexcept = default;
    TensorView& operator=(const TensorView& other) = default;
    TensorView& operator=(TensorView&& other) noexcept = default;
    ~TensorView() override = default;

    // Interface implementation
    /**
     * @brief Accesses an element in the tensor view at the specified indices
     *
     * @param indices A span of indices specifying the element location
     * @return reference A reference to the element at the specified location
     *
     * @throws std::out_of_range If indices are out of bounds
     *
     * This method computes the linear offset using the tensor's layout and
     * returns a reference to the corresponding element in the underlying base
     * tensor's data.
     */
    reference at(std::span<const int64_t> indices) override {
        const size_t offset = m_layout.get_linear_index(indices);
        return m_base_tensor.data()[offset];
    }

    /**
     * @brief Accesses an element in the tensor view at the specified indices
     *
     * @param indices A span of indices specifying the element location
     * @return reference A reference to the element at the specified location
     *
     * @throws std::out_of_range If indices are out of bounds
     *
     * This method computes the linear offset using the tensor's layout and
     * returns a reference to the corresponding element in the underlying base
     * tensor's data.
     */
    const_reference at(std::span<const int64_t> indices) const override {
        const size_t offset = m_layout.get_linear_index(indices);
        return m_base_tensor.data()[offset];
    }

    // Data access (references Tensor's data)
    /**
     * @brief Returns a pointer to the underlying data of the tensor view
     * @return Raw pointer to the first element of the tensor data
     *
     * This method provides direct access to the memory of the base tensor.
     * The returned pointer is non-const and can be used to modify the tensor
     * elements.
     */
    pointer data() noexcept override { return m_base_tensor.data(); }

    /**
     * @brief Returns a pointer to the underlying data of the tensor view
     * @return Raw pointer to the first element of the tensor data
     *
     * This method provides direct access to the memory of the base tensor.
     * The returned pointer is non-const and can be used to modify the tensor
     * elements.
     */
    const_pointer data() const noexcept override {
        return m_base_tensor.data();
    }

    // Layout and shape information
    /**
     * @brief Gets the shape of the tensor view.
     * @return A const reference to the Shape object representing the dimensions
     * of the tensor view.
     */
    const Shape& shape() const noexcept override { return m_layout.shape(); }

    /**
     * @brief Gets the strides of the tensor view.
     *
     * Strides represent the number of elements to skip in memory to move to the
     * next element along each dimension of the tensor.
     *
     * @return A constant span containing the strides for each dimension.
     * @note This is a noexcept operation and returns a read-only view of the
     * strides.
     */
    std::span<const int64_t> strides() const noexcept override {
        return m_layout.strides();
    }

    /**
     * @brief Get the memory layout descriptor of the tensor.
     *
     * @return A constant reference to the LayoutDescriptor object representing
     *         the memory layout of the tensor data.
     */
    const LayoutDescriptor& layout() const noexcept override {
        return m_layout;
    }

    // Clone (materializes the view into a new tensor)
    /**
     * @brief Creates a deep copy of the tensor view
     *
     * Creates a new tensor with the same shape and layout as the current view,
     * copying all elements from the base tensor according to the view's layout.
     * The new tensor is independent from the original view and its base tensor.
     *
     * @return Tensor<T> A new tensor containing a copy of the view's data
     */
    Tensor<T> clone() const override {
        Tensor<T> result(shape(), layout().layout());
        const size_t size = numel();

        auto iter = m_layout.create_iterator();
        auto result_iter = result.layout().create_iterator();
        const_pointer src = m_base_tensor.data();
        pointer dst = result.data();

        for (size_t i = 0; i < size; ++i) {
            dst[result_iter.offset()] = src[iter.offset()];
            iter.next();
            result_iter.next();
        }

        return result;
    }

    // Flat access methods
    /**
     * @brief Access tensor element at given flattened index.
     *
     * @param idx The linear index to access the element at. Must be in range
     * [0, numel()).
     * @return reference Reference to the element at the specified index.
     * @throws core::error::LogicError if index is out of bounds.
     */
    BREZEL_NODISCARD reference item(int64_t idx) {
        if (idx < 0 || idx >= static_cast<int64_t>(numel())) {
            throw core::error::LogicError(
                "Index {} out of bounds for tensor with {} elements", idx,
                numel());
        }

        return m_base_tensor.data()[flat_index_to_offset(idx)];
    }

    /**
     * @brief Access tensor element at given flattened index.
     *
     * @param idx The linear index to access the element at. Must be in range
     * [0, numel()).
     * @return reference Reference to the element at the specified index.
     * @throws core::error::LogicError if index is out of bounds.
     */
    BREZEL_NODISCARD const_reference item(int64_t idx) const {
        if (idx < 0 || idx >= static_cast<int64_t>(numel())) {
            throw core::error::LogicError(
                "Index {} out of bounds for tensor with {} elements", idx,
                numel());
        }

        return m_base_tensor.data()[flat_index_to_offset(idx)];
    }

    // Iterator support
    /**
     * @brief Returns an iterator to the beginning of the tensor view
     * @return Iterator pointing to the first element of the tensor view
     *
     * This method provides access to the beginning of the underlying tensor
     * data through the view's memory layout.
     */
    iterator begin() { return iterator(m_layout, m_base_tensor.data()); }

    /**
     * @brief Returns an iterator to one-past-the-last element of the view
     *
     * Returns an iterator pointing to the position after the last element in
     * the view. This iterator acts as a marker denoting the end of the view's
     * elements and should not be dereferenced.
     *
     * @return iterator End iterator for the view
     */
    iterator end() { return iterator(m_layout, m_base_tensor.data(), numel()); }

    /**
     * @brief Returns a constant iterator to the beginning of the tensor view
     *
     * The iterator provides read-only access to the underlying tensor data
     * using the view's memory layout.
     *
     * @return const_iterator A constant iterator pointing to the first element
     */
    const_iterator begin() const {
        return const_iterator(m_layout, m_base_tensor.data());
    }

    /**
     * @brief Returns a const iterator pointing to one past the last element of
     * the view
     * @return const_iterator - Iterator pointing to one past the last element
     */
    const_iterator end() const {
        return const_iterator(m_layout, m_base_tensor.data(), numel());
    }

    /**
     * @brief Returns a constant iterator pointing to the beginning of the
     * tensor view
     *
     * The iterator provides read-only access to the tensor elements following
     * the view's layout while referencing the base tensor's data.
     *
     * @return const_iterator A constant iterator pointing to the first element
     */
    const_iterator cbegin() const {
        return const_iterator(m_layout, m_base_tensor.data());
    }

    /**
     * @brief Returns a const iterator pointing past the last element of the
     * tensor view
     *
     * Provides a const iterator to the position following the last element in
     * the tensor view. This iterator should not be dereferenced as it points to
     * a position past the end.
     *
     * @return const_iterator A const iterator pointing to the past-the-end
     * element
     */
    const_iterator cend() const {
        return const_iterator(m_layout, m_base_tensor.data(), numel());
    }

    // Expression creation
    /**
     * @brief Converts the tensor view into a tensor expression
     *
     * Creates a leaf expression node that represents this tensor view in the
     * expression tree. This allows the view to be used in tensor expressions
     * and operations.
     *
     * @return std::shared_ptr<detail::TensorExpression<T>> A shared pointer to
     * the tensor expression representing this view
     */
    std::shared_ptr<detail::TensorExpression<T>> as_expression() const {
        return std::make_shared<detail::TensorLeafExpression<T>>(
            m_base_tensor.data(), m_layout);
    }

    // View operations (return new views)
    /**
     * @brief Reshapes the tensor view to a new shape while preserving the total
     * number of elements
     *
     * This method creates a new TensorView with the specified shape. The total
     * number of elements must remain the same after reshaping. For contiguous
     * tensors, it directly creates a new view with the reshaped layout. For
     * non-contiguous tensors, it first creates a contiguous copy and then
     * reshapes it.
     *
     * @param new_shape The target shape to reshape the tensor into
     * @throws LogicError if the total number of elements in new_shape doesn't
     * match the current tensor
     * @return TensorView A new tensor view with the requested shape
     */
    TensorView reshape(const Shape& new_shape) const {
        if (new_shape.numel() != numel()) {
            throw core::error::LogicError(
                "Cannot reshape tensor of size {} to size {}", numel(),
                new_shape.numel());
        }

        if (is_contiguous()) {
            auto new_layout = m_layout.reshape(new_shape);
            return TensorView(m_base_tensor, new_layout);
        }

        return clone().reshape(new_shape);
    }

    /**
     * @brief Creates a view of the tensor with two dimensions transposed
     *
     * This method returns a new TensorView with the specified dimensions
     * swapped. Negative indices are supported and are converted to positive
     * indices by adding the number of dimensions.
     *
     * @param dim0 First dimension to transpose (default = 0)
     * @param dim1 Second dimension to transpose (default = 1)
     * @throws core::error::LogicError if dimensions are out of range
     * @return TensorView A new view with the transposed dimensions
     */
    TensorView transpose(int64_t dim0 = 0, int64_t dim1 = 1) const {
        const size_t dims = ndim();

        if (dim0 < 0)
            dim0 += dims;

        if (dim1 < 0)
            dim1 += dims;

        if (dim0 < 0 || dim0 >= static_cast<int64_t>(dims) || dim1 < 0 ||
            dim1 >= static_cast<int64_t>(dims)) {
            throw core::error::LogicError(
                "Dimension out of range for transpose. Got dimensions {} and "
                "{} "
                "for tensor with {} dimensions",
                dim0, dim1, dims);
        }

        if (dim0 == dim1) {
            return *this;
        }

        auto new_layout = m_layout.transpose(dim0, dim1);
        return TensorView(m_base_tensor, new_layout);
    }

    /**
     * @brief Creates a view into a subset of the tensor based on slice indices.
     *
     * Creates a new TensorView that represents a slice of the original tensor.
     * The slice is defined by start and end indices for each dimension.
     * Negative indices are supported and are interpreted as counting from the
     * end of the dimension.
     *
     * @param indices Vector of pairs where each pair contains (start, end)
     * indices for each dimension The slicing is inclusive for start and
     * exclusive for end indices
     *
     * @return TensorView A new view into the sliced portion of the original
     * tensor
     *
     * @throws core::error::LogicError If number of index pairs doesn't match
     * tensor dimensions
     * @throws core::error::LogicError If start index is out of bounds for any
     * dimension
     * @throws core::error::LogicError If end index is out of bounds or not
     * greater than start index
     */
    TensorView slice(
        const std::vector<std::pair<int64_t, int64_t>>& indices) const {
        const auto& current_shape = shape();
        const size_t dims = ndim();

        if (indices.size() != dims) {
            throw core::error::LogicError(
                "Expected {} indices for slice operation, but got {}", dims,
                indices.size());
        }

        Shape new_shape;
        std::vector<int64_t> new_strides;
        std::vector<int64_t> offsets;

        for (size_t i = 0; i < dims; ++i) {
            const auto& [start, end] = indices[i];
            int64_t dim_size = current_shape[i];
            int64_t real_start = start < 0 ? start + dim_size : start;
            int64_t real_end = end < 0 ? end + dim_size : end;

            if (real_start < 0 || real_start >= dim_size) {
                throw core::error::LogicError(
                    "Slice start index {} out of bounds for dimension {} with "
                    "size {}",
                    start, i, dim_size);
            }

            if (real_end <= real_start || real_end > dim_size) {
                throw core::error::LogicError(
                    "Slice end index {} out of bounds for dimension {} with "
                    "size {}",
                    end, i, dim_size);
            }

            new_shape.push_back(real_end - real_start);
            new_strides.push_back(m_layout.strides()[i]);
            offsets.push_back(real_start * m_layout.strides()[i]);
        }

        size_t total_offset = std::accumulate(offsets.begin(), offsets.end(),
                                              static_cast<size_t>(0));

        LayoutDescriptor new_layout(new_shape, new_strides);
        new_layout.set_offset(m_layout.offset() + total_offset);
        new_layout.set_device(m_layout.device());
        new_layout.set_format(m_layout.format());

        return TensorView(m_base_tensor, new_layout);
    }

    /**
     * @brief Creates a new view with permuted dimensions
     *
     * Rearranges the dimensions of the tensor view according to the provided
     * permutation. For example, if the original tensor has dimensions [2,3,4]
     * and dims=[2,0,1], the resulting view will have dimensions [4,2,3].
     *
     * @param dims Vector specifying the new order of dimensions
     * @return TensorView A new tensor view with permuted dimensions
     * @throws std::invalid_argument if dims contains invalid indices or has
     * wrong size
     */
    TensorView permute(const std::vector<size_t>& dims) const {
        auto new_layout = m_layout.permute(dims);
        return TensorView(m_base_tensor, new_layout);
    }

    /**
     * @brief Removes dimensions of size 1 from the tensor's shape.
     *
     * If a specific dimension is provided, only that dimension is removed if it
     * has size 1. If no dimension is specified, all dimensions of size 1 are
     * removed.
     *
     * @param dim The dimension to squeeze. If -1 (default), all dimensions of
     * size 1 are removed.
     * @return TensorView A new tensor view with the squeezed shape.
     * @throws core::error::LogicError If the specified dimension is out of
     * range or if trying to squeeze a dimension with size not equal to 1.
     */
    TensorView squeeze(int64_t dim = -1) const {
        const auto& current_shape = shape();
        const size_t dims = ndim();

        if (dim >= 0) {
            if (dim >= static_cast<int64_t>(dims)) {
                throw core::error::LogicError(
                    "Dimension {} out of range for tensor with {} dimensions",
                    dim, dims);
            }

            if (current_shape[dim] != 1) {
                throw core::error::LogicError(
                    "Cannot squeeze dimension {} with size {}", dim,
                    current_shape[dim]);
            }

            std::vector<int64_t> new_shape_dims;
            new_shape_dims.reserve(dims - 1);

            for (size_t i = 0; i < dims; ++i) {
                if (i != static_cast<size_t>(dim)) {
                    new_shape_dims.push_back(current_shape[i]);
                }
            }

            return reshape(Shape(new_shape_dims));
        } else {
            std::vector<int64_t> new_shape_dims;
            new_shape_dims.reserve(dims);

            for (size_t i = 0; i < dims; ++i) {
                if (current_shape[i] != 1) {
                    new_shape_dims.push_back(current_shape[i]);
                }
            }

            if (new_shape_dims.empty()) {
                return reshape(Shape{});
            }

            return reshape(Shape(new_shape_dims));
        }
    }

    /**
     * @brief Adds a dimension of size 1 at the specified position.
     *
     * This method creates a new view of the tensor with an additional dimension
     * of size 1 inserted at the specified position. The new dimension can be
     * inserted at any valid position in the tensor's shape.
     *
     * @param dim The position where the new dimension should be inserted.
     *           If negative, it's interpreted as counting from the end:
     *           dim + ndim + 1
     *
     * @return TensorView A new tensor view with the additional dimension
     *
     * @throws LogicError If the dimension is out of valid range
     *         (dim < -(ndim + 1) or dim > ndim)
     */
    TensorView unsqueeze(int64_t dim) const {
        const auto& current_shape = shape();
        const size_t dims = ndim();

        if (dim < 0) {
            dim += dims + 1;
        }

        if (dim < 0 || dim > static_cast<int64_t>(dims)) {
            throw core::error::LogicError(
                "Dimension {} out of range for unsqueeze on tensor with {} "
                "dimensions",
                dim, dims);
        }

        std::vector<int64_t> new_shape_dims;
        new_shape_dims.reserve(dims + 1);

        for (size_t i = 0; i < dims + 1; ++i) {
            if (i == static_cast<size_t>(dim)) {
                new_shape_dims.push_back(1);
            }

            if (i < dims) {
                new_shape_dims.push_back(current_shape[i]);
            }
        }

        return reshape(Shape(new_shape_dims));
    }

    // Operators (create new tensors)
    /**
     * @brief Performs element-wise addition between this tensor view and
     * another tensor view
     *
     * @param other The tensor view to add to this tensor view
     * @return Tensor<T> A new tensor containing the result of the addition
     *
     * @details Creates a binary expression that represents the element-wise
     * addition of two tensor views and returns a new tensor constructed from
     * this expression. The operation is performed lazily through expression
     * templates.
     */
    Tensor<T> operator+(const TensorView& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::plus<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise subtraction between two tensors.
     *
     * This operator creates a new Tensor by subtracting each element of the
     * other TensorView from the corresponding element of this TensorView.
     *
     * @param other The TensorView whose elements will be subtracted
     * @return Tensor<T> A new Tensor containing the result of the subtraction
     *
     * @note The operation is performed element-wise and both tensors must have
     * compatible shapes.
     */
    Tensor<T> operator-(const TensorView& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::minus<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise multiplication between two tensor views
     *
     * @param other The tensor view to multiply with
     * @return Tensor<T> A new tensor containing the element-wise product
     *
     * Creates a binary expression that represents element-wise multiplication
     * and constructs a new tensor from this expression. The operation is
     * performed element by element between the current view and the provided
     * view.
     */
    Tensor<T> operator*(const TensorView& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::multiplies<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise division between this tensor view and
     * another tensor view.
     *
     * Creates a new Tensor by dividing each element of this tensor view by the
     * corresponding element in the other tensor view.
     *
     * @param other The tensor view to divide by
     * @return Tensor<T> A new tensor containing the result of the element-wise
     * division
     *
     * @throws std::invalid_argument If the tensor views have incompatible
     * shapes
     */
    Tensor<T> operator/(const TensorView& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::divides<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Creates a new tensor with all elements negated
     *
     * This operator creates a new tensor where each element is the negation of
     * the corresponding element in the original tensor.
     *
     * @return Tensor<T> A new tensor containing the negated values
     */
    Tensor<T> operator-() const {
        auto expr = detail::make_unary_expr(as_expression(), std::negate<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise addition between two tensors.
     *
     * Creates a new tensor containing the result of adding each element of the
     * current tensor with the corresponding element of the other tensor.
     *
     * @param other The tensor to add with the current tensor
     * @return Tensor<T> A new tensor containing the element-wise sum
     *
     * @throws std::invalid_argument if the tensors have incompatible shapes
     */
    Tensor<T> operator+(const Tensor<T>& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::plus<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise subtraction betwee two tensors.
     *
     * Creates a new tensor containing the result of subtracting each element of
     * the current tensor with the corresponding element of the other tensor.
     *
     * @param other The tensor to subtract with
     * @return Tensor<T> A new tensor containing the element-wise subtraction
     * results
     *
     * @throws std::invalid_argument if the tensors have incompatible shapes
     */
    Tensor<T> operator-(const Tensor<T>& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::minus<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise multiplication between two tensors.
     *
     * Creates a new tensor by multiplying each element of this tensor with the
     * corresponding element in the other tensor. The operation is done
     * element-wise (Hadamard product).
     *
     * @tparam T The data type of the tensor elements
     * @param other The tensor to multiply with
     * @return Tensor<T> A new tensor containing the element-wise multiplication
     * results
     *
     * @note The dimensions of both tensors must be compatible for element-wise
     * operations
     */
    Tensor<T> operator*(const Tensor<T>& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::multiplies<T>{});

        return Tensor<T>(*expr);
    }

    /**
     * @brief Performs element-wise division between two tensors.
     *
     * Creates a new tensor whose elements are the result of dividing the
     * elements of this tensor by the corresponding elements of the other
     * tensor.
     *
     * @param other The tensor to divide by
     * @return Tensor<T> A new tensor containing the element-wise division
     * results
     * @throws std::invalid_argument if the tensors have incompatible shapes
     */
    Tensor<T> operator/(const Tensor<T>& other) const {
        auto expr = detail::make_binary_expr(
            as_expression(), other.as_expression(), std::divides<T>{});

        return Tensor<T>(*expr);
    }

    // Utils
    /**
     * @brief Converts the tensor to a new tensor with a different scalar type.
     *
     * This method creates a new tensor with the same shape as the original
     * tensor but with elements converted to the specified scalar type U. The
     * conversion is performed using static casting.
     *
     * @tparam U The target scalar type for the conversion
     * @return Tensor<U> A new tensor containing the converted elements
     */
    template <TensorScalar U>
    Tensor<U> to() const {
        Tensor<U> result(shape());
        const size_t size = numel();

        auto iter = m_layout.create_iterator();
        auto result_iter = result.layout().create_iterator();
        const_pointer src = m_base_tensor.data();
        typename Tensor<U>::pointer dst = result.data();

        for (size_t i = 0; i < size; ++i) {
            dst[result_iter.offset()] = static_cast<U>(src[iter.offset()]);
            iter.next();
            result_iter.next();
        }

        return result;
    }

    /**
     * @brief Converts the tensor to a string representation
     *
     * Creates a formatted string representation of the tensor contents.
     * Values are organized according to the tensor's layout.
     *
     * @param max_per_line Maximum number of elements to display per line
     * (default: 6)
     * @param precision Number of decimal places to show for floating point
     * values (default: 4)
     * @return std::string Formatted string containing tensor elements
     */
    std::string to_string(int max_per_line = 6, int precision = 4) const {
        return detail::tensor_to_string(m_base_tensor.data(), m_layout,
                                        max_per_line, precision);
    }

    /**
     * @brief Computes the sum of elements along a specified dimension.
     *
     * This method first materializes the view by cloning it, then performs the
     * sum reduction operation on the resulting tensor.
     *
     * @param dim The dimension along which to reduce. Default is -1, which
     * reduces all dimensions.
     * @param keepdim If true, retains the reduced dimension with size 1.
     * Default is false.
     * @return Tensor<T> A new tensor containing the sum of elements along the
     * specified dimension.
     *
     * @note This operation creates a new tensor and does not modify the
     * original view.
     */
    Tensor<T> sum(int64_t dim = -1, bool keepdim = false) const {
        return clone().sum(dim, keepdim);
    }

    /**
     * @brief Computes the mean values along the specified dimension.
     *
     * @tparam T The data type of the tensor elements
     * @param dim The dimension along which to compute the mean. Default is -1
     * (last dimension)
     * @param keepdim Whether to keep the reduced dimension as size 1. Default
     * is false
     * @return Tensor<T> A new tensor containing the mean values
     *
     * For a tensor of shape (a, b, c, d), if dim = 2:
     * - If keepdim = false: Result shape will be (a, b, d)
     * - If keepdim = true: Result shape will be (a, b, 1, d)
     */
    Tensor<T> mean(int64_t dim = -1, bool keepdim = false) const {
        return clone().mean(dim, keepdim);
    }

    /**
     * @brief Computes the maximum value along a specified dimension in the
     * tensor.
     *
     * @tparam T The data type of the tensor elements
     * @param dim The dimension along which to compute the maximum. Default is
     * -1 (last dimension)
     * @param keepdim If true, the output tensor keeps the reduced dimension
     * with size 1. Default is false
     * @return Tensor<T> A new tensor containing the maximum values along the
     * specified dimension
     *
     * @note This is a const member function that returns a new tensor with the
     * result
     */
    Tensor<T> max(int64_t dim = -1, bool keepdim = false) const {
        return clone().max(dim, keepdim);
    }

    /**
     * @brief Computes the minimum value along a specified dimension of the
     * tensor
     *
     * @tparam T The data type of the tensor elements
     * @param dim The dimension along which to compute the minimum. If -1
     * (default), computes the minimum over all elements
     * @param keepdim If true, retains the reduced dimension with size 1.
     * Default is false
     *
     * @return A new Tensor containing the minimum values along the specified
     * dimension
     *
     * @note This is a const member function that returns a new Tensor without
     * modifying the original one
     */
    Tensor<T> min(int64_t dim = -1, bool keepdim = false) const {
        return clone().min(dim, keepdim);
    }

    /**
     * @brief Computes the product of all elements along the specified
     * dimension.
     *
     * @tparam T The data type of the tensor elements.
     * @param dim The dimension along which to compute the product. If -1
     * (default), computes the product over all elements.
     * @param keepdim If true, the output tensor preserves the input dimensions
     * with size 1 at the reduced dimension. If false (default), the dimension
     * is removed.
     * @return Tensor<T> A new tensor containing the products along the
     * specified dimension.
     */
    Tensor<T> prod(int64_t dim = -1, bool keepdim = false) const {
        return clone().prod(dim, keepdim);
    }

    // Matrix operations
    /**
     * @brief Performs matrix multiplication between this tensor view and
     * another tensor view.
     *
     * This operation creates new tensors from both views and performs matrix
     * multiplication. The operation follows standard matrix multiplication
     * rules where the number of columns in the first tensor must match the
     * number of rows in the second tensor.
     *
     * @param other The tensor view to multiply with
     * @return Tensor<T> A new tensor containing the result of the matrix
     * multiplication
     *
     * @throws std::invalid_argument If tensor dimensions are not compatible for
     * matrix multiplication
     */
    Tensor<T> matmul(const TensorView& other) const {
        return clone().matmul(other.clone());
    }

private:
    std::reference_wrapper<const Tensor<T>> m_base_tensor;
    LayoutDescriptor m_layout;

    /**
     * @brief Converts a flat index to a memory offset based on the tensor's
     * layout
     *
     * Given a flat (linear) index, computes the corresponding indices for each
     * dimension and converts them to a memory offset using the tensor's memory
     * layout.
     *
     * @param idx The flat index to convert
     * @return The memory offset corresponding to the flat index
     *
     * @note This function assumes the index is within valid bounds
     */
    BREZEL_NODISCARD inline size_t flat_index_to_offset(size_t idx) const {
        const Shape& shape_val = shape();
        std::vector<int64_t> indices(ndim());

        for (int dim = ndim() - 1; dim >= 0; --dim) {
            indices[dim] = idx % shape_val[dim];
            idx /= shape_val[dim];
        }

        return m_layout.get_linear_index(indices);
    }
};
}  // namespace brezel::tensor