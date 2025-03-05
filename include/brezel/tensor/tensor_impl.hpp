#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_base.hpp>
#include <brezel/tensor/detail/tensor_expression.hpp>
#include <brezel/tensor/detail/tensor_iterator.hpp>
#include <brezel/tensor/detail/tensor_storage.hpp>
#include <brezel/tensor/detail/tensor_utils.hpp>
#include <brezel/tensor/layout.hpp>
#include <brezel/tensor/shape.hpp>
#include <concepts>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#if defined(BREZEL_SIMD_AVX512)
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX2)
#include <immintrin.h>
#endif

namespace brezel::tensor {
// Forward declarations
template <TensorScalar T>
class TensorView;

/**
 * @brief A final class representing a multi-dimensional array (tensor) with
 * type-safe operations
 *
 * The Tensor class provides a comprehensive implementation of a
 * multi-dimensional array with support for various operations, memory layouts,
 * and efficient data manipulation. It manages memory through reference counting
 * and supports both row-major and column-major storage orders.
 *
 * Key features:
 * - Type-safe operations through templated implementation
 * - Reference counting for efficient memory management
 * - Support for different memory layouts (row-major and column-major)
 * - Rich set of constructors for various initialization scenarios
 * - Factory methods for common tensor creation patterns
 * - Iterator support for element traversal
 * - Expression templates for efficient operations
 * - View operations for zero-copy tensor manipulation
 * - Basic arithmetic operations and broadcasting
 * - Reduction operations (sum, mean, max, min, prod)
 * - Matrix operations
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 *
 * Memory management:
 * - Uses shared_ptr for automatic memory management
 * - Implements reference counting for efficient memory sharing
 * - Supports move semantics for efficient tensor manipulation
 *
 * Usage example:
 * @code
 * Tensor<float> a({2, 3}); // Creates a 2x3 tensor
 * Tensor<float> b = Tensor<float>::ones({2, 3}); // Creates a 2x3 tensor filled
 * with ones Tensor<float> c = a + b; // Performs element-wise addition
 * @endcode
 *
 * @note This class inherits from detail::TensorBase<T> and provides a complete
 * implementation of tensor operations
 * @see detail::TensorBase
 * @see TensorScalar
 */
template <TensorScalar T>
class Tensor final : public detail::TensorBase<T> {
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
    using Storage = std::shared_ptr<detail::TensorStorage<T>>;

    // Constructor
    /**
     * @brief Default constructor for Tensor class.
     *
     * Creates an empty tensor with no dimensions (empty shape) and
     * initializes storage with a new shared TensorStorage instance.
     */
    Tensor()
        : m_layout(Shape{}),
          m_storage(std::make_shared<detail::TensorStorage<T>>()) {}

    /**
     * @brief Constructs a tensor with the specified shape and memory layout.
     *
     * Creates a new tensor with uninitialized elements arranged according to
     * the given shape and memory layout specifications.
     *
     * @param shape The shape (dimensions) of the tensor to create
     * @param layout The memory layout to use for storing tensor elements
     * (defaults to RowMajor)
     *
     * @throws std::bad_alloc If memory allocation fails
     */
    explicit Tensor(const Shape& shape,
                    MemoryLayout layout = MemoryLayout::RowMajor)
        : m_layout(shape, layout),
          m_storage(std::make_shared<detail::TensorStorage<T>>(shape.numel())) {
    }

    /**
     * @brief Constructs a tensor with a specific shape, filled with a given
     * value.
     *
     * @tparam T The data type of the tensor elements
     * @param shape The shape (dimensions) of the tensor
     * @param value The value to fill the tensor with
     * @param layout The memory layout of the tensor (default: RowMajor)
     *
     * @details Creates a tensor with the specified shape where all elements are
     * initialized to the given value. The tensor's memory layout can be
     * specified as either row-major or column-major ordering.
     */
    Tensor(const Shape& shape, const T& value,
           MemoryLayout layout = MemoryLayout::RowMajor)
        : m_layout(shape, layout),
          m_storage(std::make_shared<detail::TensorStorage<T>>(shape.numel(),
                                                               value)) {}

    /**
     * @brief Constructs a Tensor from a shape and a range of data.
     *
     * This constructor creates a new Tensor with the specified shape and
     * initializes it with the data from the provided range. The data is copied
     * into the tensor's storage following the specified memory layout.
     *
     * @tparam R A range type where the value type is convertible to T
     * @param shape The shape of the tensor to create
     * @param data A range containing the data to initialize the tensor with
     * @param layout The memory layout to use for the tensor (defaults to
     * RowMajor)
     *
     * @throws core::error::InvalidArgument if the size of the data range
     * doesn't match the number of elements specified by the shape
     *
     * @note The data range must contain exactly the same number of elements as
     * specified by the shape's total number of elements (numel)
     */
    template <std::ranges::range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, T>
    Tensor(const Shape& shape, R&& data,
           MemoryLayout layout = MemoryLayout::RowMajor)
        : m_layout(shape, layout),
          m_storage(std::make_shared<detail::TensorStorage<T>>(shape.numel())) {
        if (std::ranges::distance(data) !=
            static_cast<std::ptrdiff_t>(shape.numel())) {
            throw core::error::InvalidArgument(
                "Data size ({}) does not match shape size ({})",
                std::ranges::distance(data), shape.numel());
        }

        std::ranges::copy(std::forward<R>(data), this->data());
    }

    /**
     * @brief Constructs a Tensor from a nested initializer list
     *
     * This constructor creates a Tensor from nested initializer lists,
     * automatically deducing the shape/dimensions of the tensor from the
     * structure of the initializer list.
     *
     * @tparam U The type of the elements in the initializer list (must be
     * convertible to T)
     * @param data Nested initializer list containing the tensor data
     *
     * @note The structure of the nested initializer list must be regular (all
     * sub-lists at the same level must have the same size)
     *
     * @example
     *   Tensor<float> t = {{1, 2}, {3, 4}}; // Creates a 2x2 tensor
     */
    template <typename U>
    explicit Tensor(std::initializer_list<U> data) {
        std::vector<int64_t> dims;
        detail::compute_dims(data, dims);

        m_layout = LayoutDescriptor(Shape(dims));
        m_storage = std::make_shared<detail::TensorStorage<T>>(std::accumulate(
            dims.begin(), dims.end(), size_t{1}, std::multiplies<>()));

        T* ptr = m_storage->data();
        detail::fill_from_nested_list(data, ptr, dims, 0, 0);
    }

    /**
     * @brief Constructs a Tensor from a tensor expression
     *
     * This constructor evaluates a tensor expression and creates a new tensor
     * with the resulting values. It traverses the expression element by element
     * using multi-dimensional indexing.
     *
     * @tparam T The data type of the tensor elements
     * @param expr The tensor expression to evaluate
     *
     * @details The constructor:
     * 1. Initializes the tensor layout using the expression's shape
     * 2. Allocates storage for the tensor data
     * 3. Evaluates the expression for each index combination
     * 4. Stores the results in the tensor's storage
     *
     * Time complexity: O(n) where n is the total number of elements in the
     * tensor Space complexity: O(d) where d is the number of dimensions (for
     * temporary indices storage)
     */
    explicit Tensor(const detail::TensorExpression<T>& expr)
        : m_layout(expr.shape()),
          m_storage(std::make_shared<detail::TensorStorage<T>>(expr.numel())) {
        const size_t size = numel();
        pointer dst = data();
        const Shape& result_shape = shape();

        std::vector<int64_t> indices(ndim(), 0);
        for (size_t i = 0; i < size; ++i) {
            size_t idx = i;

            for (int64_t d = ndim() - 1; d >= 0; --d) {
                indices[d] = idx % result_shape[d];
                idx /= result_shape[d];
            }

            dst[i] = expr.eval(indices);
        }
    }

    /**
     * @brief Constructs a Tensor with a specified layout and storage
     *
     * @param layout The LayoutDescriptor that defines the tensor's dimensions
     * and strides
     * @param storage The Storage object containing the tensor's data
     *
     * This constructor creates a new Tensor by taking a layout descriptor and a
     * storage object. The storage object is moved into the tensor, while the
     * layout is copied.
     *
     * @details Used internally
     */
    Tensor(const LayoutDescriptor& layout, Storage storage)
        : m_layout(layout), m_storage(std::move(storage)) {}

    /**
     * @brief Copy constructor for the Tensor class
     *
     * Creates a new Tensor that shares the same storage and layout as the
     * source tensor. The reference count of the underlying storage is
     * incremented if it exists.
     *
     * @param other The source tensor to copy from
     */
    Tensor(const Tensor& other)
        : m_layout(other.m_layout), m_storage(other.m_storage) {
        if (m_storage) {
            m_storage->increment_ref();
        }
    }

    /**
     * @brief Move constructor for Tensor class
     *
     * Constructs a new Tensor by transferring ownership of the layout and
     * storage from another Tensor instance.
     *
     * @param other The Tensor to move from
     *
     * @note No need to adjust reference count on moed-from storage
     */
    Tensor(Tensor&& other) noexcept
        : m_layout(std::move(other.m_layout)),
          m_storage(std::move(other.m_storage)) {}

    // Assignments
    /**
     * @brief Assignment operator for tensor class
     *
     * Assigns the content of another tensor to this tensor. This implementation
     * follows the reference counting pattern to manage memory. If this tensor
     * is the last one referencing its current storage, the storage will be
     * automatically cleaned up.
     *
     * @param other The tensor to copy from
     * @return Tensor& Reference to the modified tensor
     */
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (m_storage && m_storage->decrement_ref()) {
            }

            m_layout = other.m_layout;
            m_storage = other.m_storage;

            if (m_storage) {
                m_storage->increment_ref();
            }
        }

        return *this;
    }

    /**
     * @brief Move assignment operator for the Tensor class
     *
     * Assigns the contents of another tensor to this tensor using move
     * semantics. If the current tensor has storage and its reference count
     * reaches zero after decrementing, the storage is deallocated.
     *
     * @param other The tensor to move from
     * @return Tensor& Reference to the current tensor after assignment
     * @note This operation is noexcept
     */
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (m_storage && m_storage->decrement_ref()) {
            }

            m_layout = std::move(other.m_layout);
            m_storage = std::move(other.m_storage);
        }

        return *this;
    }

    // Destructor
    /**
     * @brief Destructor for Tensor class that manages the reference counting of
     * the tensor storage
     *
     * If a storage object exists and its reference count reaches zero after
     * decrementing, the storage will be automatically deallocated.
     */
    ~Tensor() override {
        if (m_storage && m_storage->decrement_ref()) {
        }
    }

    // Factory methods
    /**
     * @brief Creates a tensor filled with zeros
     *
     * Creates a new tensor with the specified shape and memory layout,
     * initializing all elements to zero.
     *
     * @param shape The shape of the tensor to create
     * @param layout The memory layout of the tensor (default: RowMajor)
     * @return A new tensor filled with zeros
     */
    static Tensor zeros(const Shape& shape,
                        MemoryLayout layout = MemoryLayout::RowMajor) {
        return Tensor(shape, T(0), layout);
    }

    /**
     * @brief Creates a tensor filled with ones with the specified shape and
     * memory layout.
     *
     * @param shape The shape of the tensor to be created
     * @param layout The memory layout of the tensor (default: RowMajor)
     * @return Tensor A new tensor of the specified shape filled with ones
     */
    static Tensor ones(const Shape& shape,
                       MemoryLayout layout = MemoryLayout::RowMajor) {
        return Tensor(shape, T(1), layout);
    }

    /**
     * @brief Creates a tensor filled with a specified value
     *
     * @tparam T Type of the tensor elements
     * @param shape The shape of the tensor to create
     * @param value The value to fill the tensor with
     * @param layout Memory layout of the tensor (default: RowMajor)
     * @return Tensor A new tensor with the specified shape filled with the
     * given value
     */
    static Tensor full(const Shape& shape, const T& value,
                       MemoryLayout layout = MemoryLayout::RowMajor) {
        return Tensor(shape, value, layout);
    }

    /**
     * @brief Creates a 2D identity tensor with ones on the diagonal and zeros
     * elsewhere.
     *
     * This function generates a square tensor of size n x n where all elements
     * are zero except for the main diagonal elements which are set to one. The
     * operation is parallelized using Intel's Thread Building Blocks (TBB) for
     * better performance.
     *
     * @param n The size of the square tensor (n x n)
     * @throws core::error::InvalidArgument if n is less than or equal to zero
     * @return Tensor A square tensor of size n x n with ones on the main
     * diagonal
     */
    static Tensor eye(int64_t n) {
        if (n <= 0) {
            throw core::error::InvalidArgument("Size must be positive, got {}",
                                               n);
        }

        Tensor result = zeros({n, n});
        tbb::parallel_for(tbb::blocked_range<int64_t>(0, n),
                          [&result](const tbb::blocked_range<size_t>& range) {
                              for (int64_t i = range.begin(); i < range.end();
                                   ++i) {
                                  result.at({i, i}) = T(1);
                              }
                          });

        return result;
    }

    /**
     * @brief Creates a one-dimensional tensor with values evenly spaced over a
     * specified interval.
     *
     * Generates a tensor of shape (steps,) containing evenly spaced values from
     * start to end, inclusive. The values are computed in parallel using Intel
     * TBB.
     *
     * @tparam T The data type of the tensor elements
     * @param start The starting value of the sequence
     * @param end The ending value of the sequence
     * @param steps The number of points to generate in the sequence
     * @return Tensor A new tensor containing the evenly spaced values
     * @throws core::error::InvalidArgument if steps is not positive
     *
     * @note If steps == 1, returns a tensor containing only the start value
     * @note The spacing between values is calculated as (end - start) / (steps
     * - 1)
     */
    static Tensor linspace(T start, T end, int64_t steps) {
        if (steps <= 0) {
            throw core::error::InvalidArgument(
                "Number of steps must be positive");
        }

        Tensor result({steps});
        T* data_ptr = result.data();

        if (steps == 1) {
            data_ptr[0] = start;
            return result;
        }

        const T step = (end - start) / static_cast<T>(steps - 1);
        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, steps),
            [=](const tbb::blocked_range<int64_t>& range) {
                for (int64_t i = range.begin(); i < range.end(); ++i) {
                    data_ptr[i] = start + step * static_cast<T>(i);
                }
            });

        return result;
    }

    /**
     * @brief Creates a tensor with evenly spaced values within the given
     * interval.
     *
     * @tparam T The data type of the tensor elements
     * @param start The starting value of the sequence
     * @param end The ending value of the sequence (exclusive)
     * @param step The spacing between values (default: 1)
     * @return Tensor A new tensor containing the evenly spaced values
     *
     * @throws core::error::InvalidArgument if step is zero
     *
     * @note If step is positive, start must be less than end to generate values
     * @note If step is negative, start must be greater than end to generate
     * values
     * @note If conditions are not met, returns a tensor with shape {0}
     */
    static Tensor arange(T start, T end, T step = T(1)) {
        if (step == T(0)) {
            throw core::error::InvalidArgument("Step cannot be zero");
        }

        if ((step > T(0) && start >= end) || (step < T(0) && start <= end)) {
            return Tensor({0});
        }

        const int64_t steps =
            static_cast<int64_t>(std::ceil((end - start) / step));
        Tensor result({steps});
        T* data_ptr = result.data();

        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, steps),
            [=](const tbb::blocked_range<int64_t>& range) {
                for (int64_t i = range.begin(); i < range.end(); ++i) {
                    data_ptr[i] = start + step * static_cast<T>(i);
                }
            });

        return result;
    }

    /**
     * @brief Creates a tensor with random values uniformly distributed between
     * min and max.
     *
     * This function generates a tensor of specified shape filled with random
     * values using thread-local random number generators for parallel
     * computation. The generation is performed in parallel using Intel TBB.
     *
     * @tparam T The data type of the tensor elements (must be arithmetic)
     * @param shape The shape of the tensor to be created
     * @param min The minimum value for random generation (default: 0)
     * @param max The maximum value for random generation (default: 1)
     * @param layout The memory layout of the tensor (default: RowMajor)
     *
     * @return A new Tensor object with random values
     *
     * @throws static_assert If T is not an arithmetic type
     *
     * @note The random generation is performed in parallel using thread-local
     * random generators to ensure thread safety and performance
     */
    static Tensor random(const Shape& shape, T min = T(0), T max = T(1),
                         MemoryLayout layout = MemoryLayout::RowMajor) {
        Tensor result(shape, layout);
        T* data_ptr = result.data();
        const size_t size = result.numel();

        if constexpr (!std::is_arithmetic_v<T>) {
            static_assert(std::is_arithmetic_v<T>,
                          "random() requires arithmetic type");

            return result;
        }

        detail::ThreadLocalRandomGeneratorFactory<T> generator_factory(min,
                                                                       max);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, size, 1024),
            [&](const tbb::blocked_range<size_t>& range) {
                auto gen = generator_factory.get_generator(
                    std::hash<std::thread::id>{}(std::this_thread::get_id()));

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    data_ptr[i] = gen();
                }
            });

        return result;
    }

    /**
     * @brief Creates a tensor filled with random numbers from a normal
     * (Gaussian) distribution.
     *
     * This function generates a tensor with the specified shape where elements
     * are sampled from a normal distribution with the given mean and standard
     * deviation. The generation is performed in parallel using Intel TBB for
     * improved performance.
     *
     * @tparam T The data type of the tensor elements (must be a floating-point
     * type)
     * @param shape The shape of the tensor to create
     * @param mean The mean (μ) of the normal distribution (default: 0)
     * @param std_dev The standard deviation (σ) of the normal distribution
     * (default: 1)
     * @param layout Memory layout for the tensor (default: RowMajor)
     *
     * @return A new Tensor filled with random numbers from the specified normal
     * distribution
     *
     * @throws std::static_assertion_failure if T is not a floating-point type
     *
     * @note The random number generation is thread-safe and uses thread-local
     * generators to ensure reproducibility and performance in parallel
     * execution.
     */
    static Tensor radn(const Shape& shape, T mean = T(0), T std_dev = T(1),
                       MemoryLayout layout = MemoryLayout::RowMajor) {
        static_assert(std::is_floating_point_v<T>,
                      "randn() requires floating point type");

        Tensor result(shape, layout);
        T* data_ptr = result.data();
        const size_t size = result.numel();

        detail::ThreadLocalNormalGeneratorFactory<T> generator_factory(mean,
                                                                       std_dev);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, size, 1024),
            [&](const tbb::blocked_range<size_t>& range) {
                auto gen = generator_factory.get_generator(
                    std::hash<std::thread::id>{}(std::this_thread::get_id()));

                for (size_t i = range.begin(); i < range.end(); ++i) {
                    data_ptr[i] = gen();
                }
            });

        return result;
    }

    // Interface implementation
    /**
     * @brief Access tensor element at given indices
     *
     * @param indices Span containing the indices to access the element
     * @return reference Reference to the element at the specified indices
     * @throws std::out_of_range If indices are out of bounds
     *
     * Provides multi-dimensional array access using layout-specific linear
     * indexing. The number of indices must match tensor's rank.
     */
    reference at(std::span<const int64_t> indices) override {
        return data()[m_layout.get_linear_index(indices)];
    }

    /**
     * @brief Access tensor element at given indices
     *
     * @param indices Span containing the indices to access the element
     * @return reference Reference to the element at the specified indices
     * @throws std::out_of_range If indices are out of bounds
     *
     * Provides multi-dimensional array access using layout-specific linear
     * indexing. The number of indices must match tensor's rank.
     */
    const_reference at(std::span<const int64_t> indices) const override {
        return data()[m_layout.get_linear_index(indices)];
    }

    // Direct data access
    /**
     * @brief Returns a pointer to the underlying data storage.
     * @return Raw pointer to the first element of the tensor's data.
     * @note This function is noexcept and overrides a virtual function.
     */
    pointer data() noexcept override { return m_storage->data(); }

    /**
     * @brief Returns a const pointer to the underlying data storage.
     * @return Const pointer to the first element of the tensor's data.
     * @note This function is noexcept and overrides a virtual function.
     */
    const_pointer data() const noexcept override { return m_storage->data(); }

    // Layout shape and information
    /**
     * @brief Gets the shape of the tensor
     * @return A const reference to the Shape object representing the dimensions
     * of the tensor
     * @note This is a noexcept operation that provides read-only access to the
     * tensor's shape
     */
    const Shape& shape() const noexcept override { return m_layout.shape(); }

    /**
     * @brief Gets the strides of the tensor.
     *
     * Strides determine the number of elements to skip to move to the next
     * element along each dimension. For example, for a 2x3 matrix stored in
     * row-major order, the strides would be [3,1], meaning you skip 3 elements
     * to move to the next row, and 1 element to move to the next column.
     *
     * @return A span containing the strides for each dimension of the tensor
     */
    std::span<const int64_t> strides() const noexcept override {
        return m_layout.strides();
    }

    /**
     * @brief Get the layout descriptor of the tensor.
     *
     * @return const LayoutDescriptor& A constant reference to the tensor's
     * layout descriptor.
     *
     * @note This function is noexcept and will not throw any exceptions.
     */
    const LayoutDescriptor& layout() const noexcept override {
        return m_layout;
    }

    // Clone implementation (deep copy)
    /**
     * @brief Creates a deep copy of the tensor.
     *
     * This method creates a new tensor with the same shape and layout as the
     * original, and copies all elements from the source tensor to the new
     * tensor. The copy operation is optimized for contiguous tensors using
     * memcpy, while non-contiguous tensors are copied element by element using
     * iterators.
     *
     * @return Tensor A new tensor containing a copy of the data from this
     * tensor
     */
    Tensor clone() const override {
        Tensor result(shape(), layout().layout());
        const size_t size = numel();

        if (is_contiguous() && result.is_contiguous()) {
            std::memcpy(result.data(), data(), size * sizeof(T));
        } else {
            auto iter = m_layout.create_iterator();
            auto result_iter = result.layout().create_iterator();

            for (size_t i = 0; i < size; ++i) {
                result.data()[result_iter.offset()] = data()[iter.offset()];
                iter.next();
                result_iter.next();
            }
        }

        return result;
    }

    // Flat access method
    /**
     * @brief Access a tensor element by flattened index
     *
     * Provides access to a single element in the tensor using a linear
     * (flattened) index. The element access follows the tensor's layout
     * ordering.
     *
     * @param idx Linear index of the element to access (0-based)
     * @throws core::error::LogicError if index is out of bounds
     * @return Reference to the element at the specified index
     */
    BREZEL_NODISCARD reference item(int64_t idx) {
        if (idx < 0 || idx >= static_cast<int64_t>(numel())) {
            throw core::error::LogicError(
                "Index {} out of bounds for tensor with {} elements", idx,
                numel());
        }

        return data()[layout_index_to_storage_index(idx)];
    }

    /**
     * @brief Access a tensor element by flattened index
     *
     * Provides access to a single element in the tensor using a linear
     * (flattened) index. The element access follows the tensor's layout
     * ordering.
     *
     * @param idx Linear index of the element to access (0-based)
     * @throws core::error::LogicError if index is out of bounds
     * @return Reference to the element at the specified index
     */
    BREZEL_NODISCARD const_reference item(int64_t idx) const {
        if (idx < 0 || idx >= static_cast<int64_t>(numel())) {
            throw core::error::LogicError(
                "Index {} out of bounds for tensor with {} elements", idx,
                numel());
        }

        return data()[layout_index_to_storage_index(idx)];
    }

    // Iterator support
    /**
     * @brief Returns an iterator to the beginning of the tensor
     * @return iterator pointing to the first element of the tensor
     */
    iterator begin() { return iterator(m_layout, data()); }

    /**
     * @brief Returns an iterator pointing to one past the last element of the
     * tensor
     * @return iterator End iterator for the tensor's data
     */
    iterator end() { return iterator(m_layout, data(), numel()); }

    /**
     * @brief Returns a constant iterator pointing to the beginning of the
     * tensor.
     * @return const_iterator A constant iterator to the first element of the
     * tensor.
     */
    const_iterator begin() const { return const_iterator(m_layout, data()); }

    /**
     * @brief Returns a const_iterator pointing to one past the last element of
     * the tensor
     *
     * @return const_iterator Iterator positioned after the last element
     */
    const_iterator end() const {
        return const_iterator(m_layout, data(), numel());
    }

    /**
     * @brief Returns a constant iterator pointing to the beginning of the
     * tensor.
     *
     * @return const_iterator An immutable iterator to the first element of the
     * tensor.
     */
    const_iterator cbegin() const { return const_iterator(m_layout, data()); }

    /**
     * @brief Returns a const iterator pointing to one past the last element of
     * the tensor
     *
     * @return const_iterator A constant iterator positioned after the last
     * element
     *
     * This method provides STL-style const_iterator for accessing tensor
     * elements. The returned iterator can be used for read-only access to
     * tensor elements in a sequential manner.
     */
    const_iterator cend() const {
        return const_iterator(m_layout, data(), numel());
    }

    // Expression creation
    /**
     * @brief Converts the tensor to a tensor expression
     * @return A shared pointer to a TensorExpression representing this tensor
     * @details Creates a leaf expression node from the tensor's data and
     * layout. This allows the tensor to be used in tensor expressions for lazy
     * evaluation
     */
    std::shared_ptr<detail::TensorExpression<T>> as_expression() const {
        return std::make_shared<detail::TensorLeafExpression<T>>(data(),
                                                                 m_layout);
    }

    // View operations
    TensorView<T> reshape(const Shape& new_shape) const;
    TensorView<T> transpose(int64_t dim0 = 0, int64_t dim = 1) const;
    TensorView<T> slice(
        const std::vector<std::pair<int64_t, int64_t>>& indices) const;
    TensorView<T> permute(const std::vector<size_t>& dims) const;
    TensorView<T> squeeze(int64_t dim = -1) const;
    TensorView<T> unsqueeze(int64_t dim) const;

    // Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    // Utility methods
    Tensor& fill_(const T& value);

    template <TensorScalar U>
    Tensor<U> to() const;

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
        return detail::tensor_to_string(data(), m_layout, max_per_line,
                                        precision);
    }

    Tensor<T> contiguous() const {
        if (is_contiguous()) {
            return *this;
        }

        Tensor result(shape());
        const auto size = numel();

        auto iter = m_layout.create_iterator();
        T* dst = result.data();
        const T* src = data();

        if (size > 1024) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, size),
                              [&](const tbb::blocked_range<size_t>& range) {
                                  auto local_iter = m_layout.create_iterator();
                                  for (size_t i = 0; i < range.begin(); ++i) {
                                      local_iter.next();
                                  }

                                  for (size_t i = range.begin();
                                       i < range.end(); ++i) {
                                      dst[i] = src[local_iter.offset()];
                                      local_iter.next();
                                  }
                              });
        } else {
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[iter.offset()];
                iter.next();
            }
        }

        return result;
    }

    // Reduction operations
    Tensor sum(int64_t dim = -1, bool keepdim = false) const;
    Tensor mean(int64_t dim = -1, bool keepdim = false) const;
    Tensor max(int64_t dim = -1, bool keepdim = false) const;
    Tensor min(int64_t dim = -1, bool keepdim = false) const;
    Tensor prod(int64_t dim = -1, bool keepdim = false) const;

    // Matrix operations
    Tensor matmul(const Tensor& other) const;

private:
    LayoutDescriptor m_layout;
    Storage m_storage;

    /**
     * @brief Converts a layout index to a storage index in the tensor's
     * memory layout
     *
     * This function maps a linear index in the tensor's layout space to its
     * corresponding storage index in memory. For contiguous tensors with
     * zero offset, the mapping is direct (identity). For non-contiguous or
     * offset tensors, the function performs dimension-wise index
     * calculation.
     *
     * @param idx The input layout index to convert
     * @return The corresponding storage index in memory
     *
     * @note For contiguous tensors with zero offset, this is an O(1)
     * operation. For non-contiguous or offset tensors, this is an O(n)
     * operation where n is ndim().
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    layout_index_to_storage_index(size_t idx) const {
        if (is_contiguous() && m_layout.offset() == 0) {
            return idx;
        } else {
            std::vector<int64_t> indices(ndim());

            m_layout.get_indices(idx, indices);
            return m_layout.get_linear_index(indices);
        }
    }
};
}  // namespace brezel::tensor