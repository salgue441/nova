#pragma once

#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/layout.hpp>
#include <concepts>
#include <iterator>
#include <optional>

namespace brezel::tensor::detail {
// Forward declaration
template <typename IterValue>
class TensorIterator;

/**
 * @brief Implementation class for tensor iteration
 *
 * @tparam IterValue The value type being iterated over in the tensor
 *
 * @details TensorIteratorImpl provides a random access iterator implementation
 * for tensors. It supports forward and backward iteration, random access
 * operations, and comparison operations. The iterator traverses tensor elements
 * according to the specified layout descriptor, which defines the memory access
 * pattern.
 *
 * The iterator maintains its position and can be:
 * - Incremented/decremented (++, --)
 * - Advanced by n positions (+=, -=)
 * - Compared with other iterators (<, <=, >, >=, ==, !=)
 * - Used for random access ([])
 * - Used to calculate distances between iterators (-)
 *
 * The implementation uses a lazy initialization strategy for the internal
 * iterator state to optimize performance when the iterator is created but not
 * immediately used.
 *
 * This class follows the requirements of C++'s RandomAccessIterator concept.
 *
 * @note This class is meant to be used through the TensorIterator class, which
 * is its friend class.
 */
template <typename IterValue>
class TensorIteratorImpl {
    friend class TensorIterator<IterValue>;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<IterValue>;
    using difference_type = std::ptrdiff_t;
    using pointer = IterValue*;
    using reference = IterValue&;

    // Constructor
    TensorIteratorImpl() = default;

    /**
     * @brief Constructs a tensor iterator implementation
     *
     * @param layout The layout descriptor that defines the iteration pattern
     * @param data Pointer to the underlying data
     * @param pos Initial position of the iterator (default = 0)
     *
     * @details If the initial position is greater than 0, creates an internal
     * layout iterator and advances it to the specified position by calling
     * next() pos times.
     */
    TensorIteratorImpl(const LayoutDescriptor& layout, pointer data,
                       size_t pos = 0)
        : m_layout(layout), m_data(data), m_position(pos) {
        if (pos > 0) {
            m_iter = layout.create_iterator();

            for (size_t i = 0; i < pos; ++i) {
                m_iter.next();
            }
        }
    }

    // Operators
    /**
     * @brief Dereference operator to access tensor element at current iterator
     * position
     *
     * @return reference Reference to the tensor element at current position
     *
     * @details If iterator is at initial position (m_position = 0) and has no
     * valid iteration state, returns element at base layout offset. Otherwise
     * returns element at current iteration offset.
     */
    BREZEL_NODISCARD reference operator*() const {
        if (m_position == 0 && !m_iter.has_value()) {
            return m_data[m_layout.offset()];
        }

        return m_data[m_iter->offset()];
    }

    /**
     * @brief Overloaded arrow operator for accessing the current element.
     *
     * Returns a pointer to the current element in the tensor.
     * If at the initial position (0) and no iterator exists, returns pointer
     * to first element using base layout offset.
     * Otherwise returns pointer to element at current iterator position.
     *
     * @return pointer Pointer to the current element in tensor memory
     */
    BREZEL_NODISCARD pointer operator->() const {
        if (m_position == 0 && !m_iter.has_value()) {
            return m_data + m_layout.offset();
        }

        return m_data + m_iter->offset();
    }

    /**
     * @brief Prefix increment operator for tensor iterator.
     *
     * Advances the iterator to the next position in the tensor.
     * Ensures iterator is properly initialized before advancing.
     *
     * @return Reference to this iterator after increment
     *
     * @pre Iterator must be dereferenceable
     * @post Position is advanced by one and internal iterator state is updated
     */
    TensorIteratorImpl& operator++() {
        ensure_iter_initialized();

        m_iter->next();
        ++m_position;

        return *this;
    }

    /**
     * @brief Post-increment operator overload for tensor iterator
     *
     * Creates a copy of the current iterator state, then increments the
     * iterator, and returns the copy (original state). This provides the
     * standard post-increment semantics.
     *
     * @return TensorIteratorImpl A copy of the iterator before increment
     */
    TensorIteratorImpl& operator++(int) {
        TensorIteratorImpl temp(*this);

        ++(*this);
        return temp;
    }

    /**
     * @brief Prefix decrement operator for tensor iterator
     *
     * Decrements the iterator's position and resets the internal layout
     * iterator to match the new position. The internal iterator is then
     * advanced to the new position.
     *
     * @return Reference to the modified iterator
     *
     * @note If current position is 0, the operation has no effect
     */
    TensorIteratorImpl& operator--() {
        if (m_position > 0) {
            --m_position;
            m_iter = m_layout.create_iterator();

            for (size_t i = 0; i < m_position; ++i) {
                m_iter->next();
            }
        }

        return *this;
    }

    /**
     * @brief Post-decrement operator for tensor iterator
     *
     * Decrements the iterator's position after returning a copy of the original
     * iterator.
     *
     * @return TensorIteratorImpl Copy of the iterator before decrementing
     */
    TensorIteratorImpl operator--(int) {
        TensorIteratorImpl temp(*this);

        --(*this);
        return temp;
    }

    /**
     * @brief Advances the iterator by n positions.
     *
     * If n is positive, moves forward n times.
     * If n is negative, moves backward by calling operator-= with positive n.
     * Does nothing if n is zero.
     *
     * @param n Number of positions to advance (can be negative)
     * @return Reference to this iterator after advancing
     */
    TensorIteratorImpl& operator+=(difference_type n) {
        if (n > 0) {
            ensure_iter_initialized();

            for (difference_type i = 0; i < n; ++i) {
                m_iter->next();
            }

            m_position += n;
        } else if (n < 0) {
            return *this -= -n;
        }

        return *this;
    }

    /**
     * @brief Subtracts n positions from the iterator's current position
     *
     * Moves the iterator backwards by n positions. If n is negative,
     * the operation is delegated to operator+=(-n).
     *
     * @param n Number of positions to move backwards
     * @throws std::out_of_range if the subtraction would move before the start
     * position
     * @return Reference to this iterator after moving
     */
    TensorIteratorImpl& operator-=(difference_type n) {
        if (n > 0) {
            if (static_cast<difference_type>(m_position) < n) {
                throw std::out_of_range(
                    "Iterator subtraction would go beyond start");
            }

            m_position -= n;
            m_iter = m_layout.create_iterator();

            for (size_t i = 0; i < m_position; ++i) {
                m_iter->next();
            }
        } else if (n < 0) {
            return *this += -n;
        }

        return *this;
    }

    /**
     * @brief Random access operator for tensor iterator
     *
     * Provides random access to elements at offset n from current iterator
     * position. Creates a temporary iterator, advances it by n positions, and
     * returns the referenced element.
     *
     * @param n Offset from current position
     * @return reference Reference to the element at position current + n
     */
    BREZEL_NODISCARD reference operator[](difference_type n) const {
        TensorIteratorImpl temp(*this);

        temp += n;
        return *temp;
    }

    // Comparison operators
    /**
     * @brief Equality comparison operator for tensor iterators
     * @param other The tensor iterator to compare with
     * @return true if the iterators point to the same position, false otherwise
     *
     * Compares two tensor iterators by checking if they point to the same
     * position in memory.
     */
    BREZEL_NODISCARD bool operator==(const TensorIteratorImpl& other) const {
        return m_position == other.m_position;
    }

    /**
     * @brief Inequality comparison operator
     *
     * Checks if this iterator is not equal to another iterator.
     * Implemented in terms of operator==.
     *
     * @param other The iterator to compare against
     * @return true if the iterators are not equal
     * @return false if the iterators are equal
     */
    BREZEL_NODISCARD bool operator!=(const TensorIteratorImpl& other) const {
        return !(*this == other);
    }

    /**
     * @brief Compares if this iterator's position is less than another
     * iterator's position
     *
     * @param other Another TensorIteratorImpl to compare with
     * @return true if this iterator's position is less than other's position,
     * false otherwise
     */
    BREZEL_NODISCARD bool operator<(const TensorIteratorImpl& other) const {
        return m_position < other.m_position;
    }

    /**
     * @brief Checks if this iterator's position is less than or equal to
     * another iterator's position
     *
     * @param other The iterator to compare against
     * @return true if this iterator's position is less than or equal to the
     * other iterator's position, false otherwise
     */
    BREZEL_NODISCARD bool operator<=(const TensorIteratorImpl& other) const {
        return m_position <= other.m_position;
    }

    /**
     * @brief Checks if this iterator points to a position after another
     * iterator
     *
     * @param other The iterator to compare with
     * @return true if this iterator's position is greater than the other's
     * position
     */
    BREZEL_NODISCARD bool operator>(const TensorIteratorImpl& other) const {
        return m_position > other.m_position;
    }

    /**
     * @brief Compares if this iterator's position is greater than or equal to
     * another iterator's position
     *
     * @param other The iterator to compare against
     * @return true if this iterator's position is greater than or equal to the
     * other iterator's position, false otherwise
     */
    BREZEL_NODISCARD bool operator>=(const TensorIteratorImpl& other) const {
        return m_position >= other.m_position;
    }

    BREZEL_NODISCARD difference_type
    /**
     * @brief Calculates the distance between two iterators
     *
     * Subtracts the positions of two iterators to determine how many elements
     * are between them.
     *
     * @param other The iterator to subtract from this one
     * @return difference_type The number of elements between the two iterators
     */
    operator-(const TensorIteratorImpl & other) const {
        return static_cast<difference_type>(m_position) -
               static_cast<difference_type>(other.m_position);
    }

private:
    LayoutDescriptor m_layout;
    pointer m_data = nullptr;
    size_t m_position = 0;
    std::optional<StridedIndex> m_iter;

    /**
     * @brief Ensures that the internal tensor iterator is initialized
     *
     * If the iterator hasn't been created yet (m_iter is empty), creates a new
     * iterator using the tensor's layout. This method guarantees that the
     * iterator is available for tensor traversal operations.
     */
    void ensure_iter_initialized() {
        if (!m_iter.has_value()) {
            m_iter = m_layout.create_iterator();
        }
    }
};

/**
 * @brief Template class implementing a random access iterator for Tensor
 * objects
 * @tparam IterValue The value type that the iterator will traverse
 *
 * TensorIterator provides a standard-compliant random access iterator interface
 * for traversing elements in a tensor. It wraps around a TensorIteratorImpl
 * which handles the actual iteration logic based on the tensor's memory layout.
 *
 * The iterator supports all standard random access iterator operations
 * including:
 * - Forward and backward traversal
 * - Random access indexing
 * - Iterator arithmetic (addition and subtraction)
 * - Comparison operations
 *
 * This iterator satisfies the requirements of LegacyRandomAccessIterator and
 * provides the following type aliases:
 * - iterator_category: std::random_access_iterator_tag
 * - value_type: The underlying value type without cv-qualifiers
 * - difference_type: std::ptrdiff_t for pointer arithmetic
 * - pointer: Pointer to the value type
 * - reference: Reference to the value type
 *
 * @note The iterator's behavior is determined by the underlying layout
 * descriptor which defines how the tensor's elements are arranged in memory.
 *
 * @example
 * @code
 * TensorIterator<float> it(layout, data_ptr);
 * float value = *it;  // Access current element
 * ++it;               // Move to next element
 * it += 5;            // Advance 5 positions
 * @endcode
 */
template <typename IterValue>
class TensorIterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::remove_cv_t<IterValue>;
    using difference_type = std::ptrdiff_t;
    using pointer = IterValue*;
    using reference = IterValue&;

    // Constructors
    TensorIterator() = default;

    /**
     * @brief Constructs a new Tensor Iterator object
     *
     * @param layout The layout descriptor that defines the memory layout of the
     * tensor
     * @param data Pointer to the underlying data
     * @param pos Initial position in the iteration sequence (defaults to 0)
     */
    TensorIterator(const LayoutDescriptor& layout, pointer data, size_t pos = 0)
        : m_impl(layout, data, pos) {}

    // Iterator operations - forward implementations
    /**
     * @brief Dereference operator for tensor iterator.
     * @return A reference to the value pointed to by the iterator.
     *
     * Provides access to the current element being pointed to by the iterator
     * through dereferencing the internal implementation pointer.
     */
    reference operator*() const { return *m_impl; }

    /**
     * @brief Returns a pointer to the current element.
     * @return A pointer to the element referenced by the iterator.
     *
     * Delegates to the underlying implementation's operator->().
     */
    pointer operator->() const { return m_impl.operator->(); }

    /**
     * @brief Pre-increment operator for TensorIterator.
     *
     * Advances the iterator to the next element in the tensor.
     *
     * @return Reference to the incremented iterator.
     */
    TensorIterator& operator++() {
        ++m_impl;

        return *this;
    }

    /**
     * @brief Post-increment operator for TensorIterator.
     *
     * Creates a copy of the current iterator and then increments the original.
     *
     * @return TensorIterator A copy of the iterator before increment.
     */
    TensorIterator operator++(int) { return TensorIterator(m_impl++); }

    /**
     * @brief Prefix decrement operator that moves the iterator backward by one
     * position.
     *
     * @return TensorIterator& Reference to the decremented iterator.
     */
    TensorIterator operator--() {
        --m_impl;

        return *this;
    }

    /**
     * @brief Post-decrement operator for the TensorIterator class.
     * @details Decrements the internal implementation iterator after returning
     * the current value.
     * @return TensorIterator A copy of the iterator before decrementing.
     */
    TensorIterator operator--(int) { return TensorIterator(m_impl--); }

    /**
     * @brief Adds a given number of steps to the iterator's position
     * @param n The number of steps to advance the iterator
     * @return Reference to this iterator after advancing
     *
     * Advances the internal implementation of the iterator by n positions
     * and returns a reference to the modified iterator.
     */
    TensorIterator& operator+=(difference_type n) {
        m_impl += n;

        return *this;
    }

    /**
     * @brief Adds a number of steps to the iterator's current position
     *
     * Creates a new iterator that is advanced by n positions from the current
     * position. This is a non-mutating operation that returns a new iterator
     * instance.
     *
     * @param n Number of positions to advance
     * @return TensorIterator New iterator positioned n steps ahead
     */
    TensorIterator operator+(difference_type n) const {
        TensorIterator result(*this);
        result += n;

        return result;
    }

    /**
     * @brief Subtracts a number of steps from the iterator's position
     * @param n The number of steps to subtract
     * @return Reference to this iterator after subtraction
     */
    TensorIterator& operator-=(difference_type n) {
        m_impl -= n;

        return *this;
    }

    /**
     * @brief Subtracts n positions from the current iterator position
     *
     * Creates a new iterator that points to a position n elements before the
     * current position
     *
     * @param n Number of positions to move backwards
     * @return A new TensorIterator pointing to the position n elements before
     * the current position
     */
    TensorIterator operator-(difference_type n) const {
        TensorIterator result(*this);
        result -= n;

        return result;
    }

    /**
     * @brief Returns a reference to the element at the specified position.
     *
     * @param n Position of the element to access
     * @return reference Reference to the element at position n
     *
     * @note This operator provides unchecked access to the tensor elements
     */
    reference operator[](difference_type n) const { return m_impl[n]; }

    // Comparison operators
    /**
     * @brief Equality comparison operator for TensorIterator
     * @param other The TensorIterator to compare with
     * @return true if the iterators point to the same position, false otherwise
     *
     * Compares the underlying implementation of two TensorIterators for
     * equality.
     */
    bool operator==(const TensorIterator& other) const {
        return m_impl == other.m_impl;
    }

    /**
     * @brief Inequality operator to compare two tensor iterators
     * @param other The TensorIterator to compare against
     * @return true if the iterators are not equal, false otherwise
     *
     * Compares the underlying implementation of two tensor iterators to
     * determine if they are different.
     */
    bool operator!=(const TensorIterator& other) const {
        return m_impl != other.m_impl;
    }

    /**
     * @brief Compares if this iterator is less than another iterator
     * @param other The iterator to compare against
     * @return true if this iterator is less than the other iterator, false
     * otherwise
     */
    bool operator<(const TensorIterator& other) const {
        return m_impl < other.m_impl;
    }

    /**
     * @brief Less than or equal operator for tensor iterator comparison
     * @param other The tensor iterator to compare with
     * @return true if this iterator is less than or equal to the other iterator
     * @note Compares the underlying implementations of the iterators
     */
    bool operator<=(const TensorIterator& other) const {
        return m_impl <= other.m_impl;
    }

    /**
     * @brief Greater than comparison operator for TensorIterator
     * @param other The TensorIterator to compare with
     * @return true if this iterator points to a position after other, false
     * otherwise
     */
    bool operator>(const TensorIterator& other) const {
        return m_impl > other.m_impl;
    }

    /**
     * @brief Greater than or equal comparison operator for TensorIterator
     *
     * Compares this iterator with another iterator by comparing their internal
     * implementations
     *
     * @param other The iterator to compare with
     * @return true if this iterator points to a position greater than or equal
     * to the other iterator
     * @return false otherwise
     */
    bool operator>=(const TensorIterator& other) const {
        return m_impl >= other.m_impl;
    }

    /**
     * @brief Calculates the distance between two iterators.
     * @param other The iterator to subtract from this iterator.
     * @return The number of elements between this iterator and the other
     * iterator.
     *
     * Computes the distance by subtracting the underlying implementation
     * iterators. Returns a positive value if this iterator is after other,
     * negative if before.
     */
    difference_type operator-(const TensorIterator& other) const {
        return m_impl - other.m_impl;
    }

private:
    TensorIteratorImpl<IterValue> m_impl;
};

/**
 * @brief Overloads operator+ to support n + iterator syntax for TensorIterator
 *
 * @tparam IterValue The value type being iterated over
 * @param n The number of positions to advance the iterator
 * @param it The iterator to advance
 * @return TensorIterator<IterValue> A new iterator advanced by n positions
 *
 * This non-member operator allows the commutative property of addition with
 * iterators, enabling syntax like "5 + iterator" in addition to "iterator + 5"
 */
template <typename IterValue>
TensorIterator<IterValue> operator+(
    typename TensorIterator<IterValue>::difference_type n,
    const TensorIterator<IterValue>& it) {
    return it + n;
}

}  // namespace brezel::tensor::detail