#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/pfr.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape.hpp>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(BREZEL_SIMD_AVX2)
#include <immintrin.h>
#elif defined(BREZEL_SIMD_AVX512)
#include <immintrin.h>
#endif

namespace brezel::tensor {
namespace detail {
constexpr size_t kSmallVectorSize = 8;

template <typename T>
using SmallVector = boost::container::small_vector<T, kSmallVectorSize>;

template <typename T, size_t N>
using StaticVector = boost::container::static_vector<T, N>;

/**
 * @brief A utility function that provides a hint to the compiler that the given
 * condition is likely to be true.
 *
 * This function uses the `BREZEL_PREDICT_TRUE` macro to indicate that the
 * condition is expected to be true most of the time, which can help the
 * compiler optimize the code for better performance.
 *
 * @tparam T The type of the condition to be evaluated.
 * @param val The condition to be evaluated.
 * @return true if the condition is likely to be true, false otherwise.
 */
template <typename T>
inline constexpr bool likely(T&& val) noexcept {
    return BREZEL_PREDICT_TRUE(static_cast<bool>(std::forward<T>(val)));
}

/**
 * @brief A utility function that hints the compiler to optimize for the
 * unlikely case.
 *
 * This function uses a compiler-specific built-in function to provide a hint
 * that the condition is expected to be false most of the time. This can help
 * the compiler generate more efficient code by optimizing for the common case
 * where the condition is false.
 *
 * @tparam T The type of the value being evaluated.
 * @param val The value to be evaluated. This is a forwarding reference to allow
 * perfect forwarding.
 * @return true if the condition is unlikely, false otherwise.
 */
template <typename T>
inline constexpr bool unlikely(T&& val) noexcept {
    return BREZEL_PREDICT_FALSE(static_cast<bool>(std::forward<T>(val)));
}

/**
 * @brief Fast bit-counting utilities
 */
/**
 * @brief Counts the number of leading zeros in a 64-bit unsigned integer.
 *
 * This function uses compiler-specific built-in functions or a manual
 * implementation to count the number of leading zeros in the given 64-bit
 * unsigned integer `x`.
 *
 * @param x The 64-bit unsigned integer to count leading zeros in.
 * @return The number of leading zeros in `x`. If `x` is zero, the function
 *         returns 64.
 *
 * @note This function is marked as `constexpr` and `noexcept`.
 * @note The implementation uses compiler-specific built-ins for GCC, Clang,
 *       and MSVC. If none of these compilers are detected, a manual
 *       implementation is used.
 */
constexpr int cound_leading_zeros(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#elif defined(_MSC_VER)
    unsigned long result;
    return _BitScanReverse64(&result, x) ? 63 - result : 64;
#else
    if (x == 0)
        return;

    int count = 0;
    while ((x & (uint64_t(1) << 63)) == 0) {
        count++;
        x <<= 1;
    }

    return count;
#endif
}

/**
 * @brief Computes the ceiling of the division of two numbers.
 *
 * This function returns the smallest integer greater than or equal to the
 * result of the division of @p a by @p b.
 *
 * @tparam T The type of the input parameters and return value. Must be an
 * integral type.
 * @param a The dividend.
 * @param b The divisor. Must be non-zero.
 * @return The ceiling of the division of @p a by @p b.
 */
template <typename T>
constexpr T ceil_div(T a, T b) noexcept {
    return (a + b - 1) / b;
}

/**
 * @brief Combines a hash value with an existing seed.
 *
 * This function takes an existing seed and a value, computes the hash of the
 * value, and combines it with the seed using a specific bit manipulation
 * technique. This is useful for creating a single hash value from multiple
 * inputs.
 *
 * @tparam T The type of the value to be hashed.
 * @param seed A reference to the existing seed to be combined with the hash of
 * the value.
 * @param val The value to be hashed and combined with the seed.
 */
template <typename T>
inline void hash_combine(size_t& seed, const T& val) {
    seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace detail

/**
 * @brief Memory layout enumeration
 * @details Define the order in which tensor elements are stored in memory
 */
enum class MemoryLayout : uint8_t {
    RowMajor,     // Dimensions are stored from slowest to fastest
    ColumnMajor,  // Col-major, dimensions are stored from fastest to slowest
    Strided       // Custom strided layout with explicit stride values
};

/**
 * @brief Converts a memory layout to a string
 *
 * @param layout The memory layout enum value
 * @return String representation of the layout
 */
constexpr std::string_view to_string(MemoryLayout layout) noexcept {
    switch (layout) {
        case MemoryLayout::RowMajor:
            return "RowMajor";

        case MemoryLayout::ColumnMajor:
            return "ColumnMajor";

        case MemoryLayout::Strided:
            return "Strided";

        default:
            return "Unkown";
    }
}

/**
 * @brief Device type enumeration
 * @details Represents where tensor data is physically stored
 */
enum class DeviceType : uint8_t {
    CPU,     ///< CPU memory
    CUDA,    ///< NVIDIA CUDA GPU memory
    OpenCL,  ///< OpenCL device memory
    Metal,   ///< Apple Metal GPU memory
    Custom   ///< Custom device type
};

/**
 * @brief Converts a device type to a string
 *
 * @param device The device type enum value
 * @return String representation of the device
 */
constexpr std::string_view to_string(DeviceType device) noexcept {
    switch (device) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        case DeviceType::OpenCL:
            return "OpenCL";
        case DeviceType::Metal:
            return "Metal";
        case DeviceType::Custom:
            return "Custom";
        default:
            return "Unknown";
    }
}

/**
 * @brief Memory format enumeration
 * @details Defines special memory layouts for specific tensor types
 */
enum class MemoryFormat : uint8_t {
    Contiguous,     ///< Standard contiguous memory
    ChannelsLast,   ///< NHWC layout for images (for efficient convolutions)
    ChannelsFirst,  ///< NCHW layout for images (PyTorch default)
    Sparse,         ///< Sparse storage format
    Quantized,      ///< Quantized storage (reduced precision)
    Custom          ///< Custom storage format
};

/**
 * @brief Converts a memory format to a string
 *
 * @param format The memory format enum value
 * @return String representation of the format
 */
constexpr std::string_view to_string(MemoryFormat format) noexcept {
    switch (format) {
        case MemoryFormat::Contiguous:
            return "Contiguous";

        case MemoryFormat::ChannelsLast:
            return "ChannelsLast";

        case MemoryFormat::ChannelsFirst:
            return "ChannelsFirst";

        case MemoryFormat::Sparse:
            return "Sparse";

        case MemoryFormat::Quantized:
            return "Quantized";

        case MemoryFormat::Custom:
            return "Custom";

        default:
            return "Unknown";
    }
}

// Forward declarations for StridedIndex
class LayoutDescriptor;

/**
 * @brief Multi-dimensional index with stride-based traversal
 * @details Efficiently handles indexed access to tensors with various memory
 * layouts. Uses compile-time optimization with constexpr where possible
 */
class BREZEL_API StridedIndex {
public:
    /**
     * @brief Create a strided index for a specific layout
     * @param layout The layout descriptor defining strides
     */
    explicit StridedIndex(const LayoutDescriptor&);

    /**
     * @brief Resets the tensor layout to its initial state.
     *
     * This function resets the position and offset of the tensor layout to
     * their initial values. It also sets all coordinates to zero.
     *
     * @note This function is marked as `noexcept`, indicating that it does not
     * throw any exceptions.
     */
    BREZEL_FORCE_INLINE void reset() noexcept {
        m_position = 0;
        m_offset = m_initial_offset;

        std::fill(m_coordinates.begin(), m_coordinates.end(), 0);
    }

    /**
     * @brief Advances to the next position in a multidimensional tensor layout.
     *
     * This function increments the current position within the tensor layout.
     * If the position exceeds the total size of the tensor, it returns false.
     * Otherwise, it updates the coordinates and offset based on the tensor's
     * shape and strides.
     *
     * @return true if the position was successfully incremented and is within
     * bounds.
     * @return false if the position exceeds the total size of the tensor.
     */
    BREZEL_FORCE_INLINE bool next() noexcept {
        if (++m_position >= m_total_size) {
            return false;
        }

        for (int64_t dim = m_ndim - 1; dim >= 0; --dim) {
            if (++m_coordinates[dim] < m_shape[dim]) {
                m_offset += m_strides[dim];
                return true;
            }

            m_coordinates[dim] = 0;
            m_offset -= m_strides[dim] * (m_shape[dim] - 1);
        }

        return true;
    }

    /**
     * @brief Moves the tensor's internal position to the specified coordinates.
     *
     * This function updates the tensor's internal position and offset based on
     * the provided coordinates. It ensures that the coordinates match the
     * tensor's dimensions and are within the valid range of the tensor's shape.
     *
     * @param coords A span of integers representing the coordinates to move to.
     *               The size of the span must match the number of dimensions of
     *               the tensor.
     *
     * @throws brezel::core::error::LogicError If the size of the coordinates
     * does not match the number of dimensions of the tensor.
     * @throws brezel::core::error::LogicError If any coordinate is out of
     * bounds for the tensor's shape.
     */
    BREZEL_FORCE_INLINE void move_to(std::span<const int64_t> coords) {
        if (coords.size() != m_ndim) {
            throw brezel::core::error::LogicError(
                "Coordinate size doesn't match tensor dimensions");
        }

        for (size_t i = 0; i < m_ndim; ++i) {
            if (coords[i] < 0 || coords[i] >= m_shape[i]) {
                throw brezel::core::error::LogicError(
                    "Coordinate out of bounds");
            }
        }

        size_t position = 0;
        size_t multiplier = 1;

        for (int64_t i = m_ndim - 1; i >= 0; --i) {
            position += coords[i] * multiplier;
            multiplier *= m_shape[i];
        }

        m_position = position;
        m_offset = m_initial_offset;

        for (size_t i = 0; i < m_ndim; ++i) {
            m_coordinates[i] = coords[i];
            m_offset += coords[i] * m_strides[i];
        }
    }

    /**
     * @brief Returns the current offset within the tensor.
     *
     * This function provides the current offset within the tensor, which
     * represents the position in the underlying memory layout.
     *
     * @return The current offset within the tensor.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t offset() const noexcept {
        return m_offset;
    }

    /**
     * @brief Returns the current coordinates within the tensor.
     *
     * This function provides a span of the current coordinates within the
     * tensor. The coordinates represent the multi-dimensional position within
     * the tensor.
     *
     * @return A span of the current coordinates within the tensor.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE std::span<const int64_t> coordinates()
        const noexcept {
        return m_coordinates;
    }

    /**
     * @brief Returns a pointer to the current coordinates within the tensor.
     *
     * This function provides a pointer to the current coordinates within the
     * tensor. The coordinates represent the multi-dimensional position within
     * the tensor.
     *
     * @return A pointer to the current coordinates within the tensor.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE const int64_t* coordinates_ptr()
        const noexcept {
        return m_coordinates.data();
    }

    /**
     * @brief Checks if the current position is at the beginning of the tensor.
     *
     * This function checks if the current position within the tensor is at the
     * beginning (i.e., the first element).
     *
     * @return true if the current position is at the beginning, false
     * otherwise.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_begin() const noexcept {
        return m_position == 0;
    }

    /**
     * @brief Checks if the current position is at the end of the tensor.
     *
     * This function checks if the current position within the tensor is at the
     * end (i.e., past the last element).
     *
     * @return true if the current position is at the end, false otherwise.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE bool is_end() const noexcept {
        return m_position >= m_total_size;
    }

    /**
     * @brief Returns the total size of the tensor.
     *
     * This function provides the total number of elements in the tensor.
     *
     * @return The total size of the tensor.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t size() const noexcept {
        return m_total_size;
    }

    /**
     * @brief Returns the current position within the tensor.
     *
     * This function provides the current position within the tensor, which
     * represents the index of the current element.
     *
     * @return The current position within the tensor.
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t position() const noexcept {
        return m_position;
    }

private:
    detail::SmallVector<int64_t> m_coordinates;
    detail::SmallVector<int64_t> m_strides;
    detail::SmallVector<int64_t> m_shape;

    size_t m_offset = 0;
    size_t m_position = 0;
    size_t m_total_size = 0;
    size_t m_initial_offset = 0;
    size_t m_ndim = 0;
};

/**
 * @brief Layout descriptor for tensor memory organization
 * @details Encapsulates all information about how tensor data is arranged in
 * memory.
 */
class BREZEL_API LayoutDescriptor {
public:
    /**
     * @brief Constructs a simple contiguous layout descriptor
     *
     * @param shape The tensor shape
     * @param layout Memory layout style (row or column major)
     * @throws brezel::core::error::InvalidArgument if shape is invalid
     */
    BREZEL_NODISCARD explicit LayoutDescriptor(
        const Shape& shape, MemoryLayout layout = MemoryLayout::RowMajor)
        : m_shape(shape),
          m_layout(layout),
          m_format(MemoryFormat::Contiguous),
          m_device(DeviceType::CPU) {
        validate_shape(shape);
        compute_strides();
    }

    /**
     * @brief Construct a layout descriptor with explicit strides
     *
     * @param shape The tensor shape
     * @param strides The explicit stride values
     * @throws brezel::core::error::InvalidArgument if strides and shape don't
     *                                              match or have invalid values
     */
    BREZEL_NODISCARD LayoutDescriptor(const Shape& shape,
                                      std::span<const int64_t> strides)
        : m_shape(shape),
          m_layout(MemoryLayout::Strided),
          m_format(MemoryFormat::Contiguous),
          m_device(DeviceType::CPU) {
        if (strides.size() != shape.size()) {
            throw brezel::core::error::InvalidArgument(
                "Strides size ({}) must match shape size ({})", strides.size(),
                shape.size());
        }

        validate_shape(shape);
        validate_strides(strides);

        m_strides.assign(strides.begin(), strides.end());
    }

    /**
     * @brief Constructs a layout descriptor for a specific memory format
     *
     * @param shape The tensor shape
     * @param format Special memory format
     * @param device Device where the tensor is stored
     * @throws InvalidArgument if shape is invalid or format is incompatible
     * with shape
     */
    BREZEL_NODISCARD LayoutDescriptor(const Shape& shape, MemoryFormat format,
                                      DeviceType device = DeviceType::CPU)
        : m_shape(shape), m_format(format), m_device(device) {
        validate_shape(shape);

        if (format == MemoryFormat::ChannelsLast) {
            if (shape.size() == 4) {
                m_layout = MemoryLayout::Strided;

                // Compute NHWC strides (N x H x W x C)
                int64_t h = shape[1];
                int64_t w = shape[2];
                int64_t c = shape[3];

                m_strides.resize(4);
                m_strides[0] = h * w * c;  // N stride
                m_strides[1] = w * c;      // H stride
                m_strides[2] = c;          // W stride
                m_strides[3] = 1;          // C stride
            } else {
                throw brezel::core::error::InvalidArgument(
                    "ChannelsLast format requires a 4D tensor, but got {}D",
                    shape.size());
            }
        } else if (format == MemoryFormat::ChannelsFirst) {
            if (shape.size() == 4) {
                m_layout = MemoryLayout::RowMajor;
                compute_strides();
            } else {
                throw brezel::core::error::InvalidArgument(
                    "ChannelsFirst format requires a 4D tensor, but got {}D",
                    shape.size());
            }
        } else if (format == MemoryFormat::Sparse) {
            m_layout = MemoryLayout::Strided;
            m_strides.resize(shape.size(), 0);
        } else {
            m_layout = MemoryLayout::RowMajor;
            compute_strides();
        }
    }

    // Copy && Move constructor
    LayoutDescriptor(const LayoutDescriptor& other) = default;
    LayoutDescriptor(LayoutDescriptor&& other) noexcept = default;
    LayoutDescriptor& operator=(const LayoutDescriptor& other) = default;
    LayoutDescriptor& operator=(LayoutDescriptor&& other) noexcept = default;

    /**
     * @brief Returns the shape of the tensor
     * @return Reference to the shape
     */
    BREZEL_NODISCARD const Shape& shape() const noexcept { return m_shape; }

    /**
     * @brief Returns the strides of the tensor
     * @return Span of stride values
     */
    BREZEL_NODISCARD std::span<const int64_t> strides() const noexcept {
        return m_strides;
    }

    /**
     * @brief Returns the memory layout type
     * @return Memory layout enum value
     */
    BREZEL_NODISCARD MemoryLayout layout() const noexcept { return m_layout; }

    /**
     * @brief Returns the memory format
     * @return Memory format enum value
     */
    BREZEL_NODISCARD MemoryFormat format() const noexcept { return m_format; }

    /**
     * @brief Returns the device type
     * @return Device type enum value
     */
    BREZEL_NODISCARD DeviceType device() const noexcept { return m_device; }

    /**
     * @brief Gets the number of dimensions
     * @return Number of dimensions
     */
    BREZEL_NODISCARD size_t ndim() const noexcept { return m_shape.size(); }

    /**
     * @brief Gets the offset into the memory buffer
     * @return Memory offset in elements
     */
    BREZEL_NODISCARD size_t offset() const noexcept { return m_offset; }

    /**
     * @brief Sets the memory offset
     * @param offset Byte offset into the underlying storage
     */
    void set_offset(size_t offset) noexcept { m_offset = offset; }

    /**
     * @brief Sets the memory format
     * @param format Memory format to set
     */
    void set_format(MemoryFormat format) noexcept { m_format = format; }

    /**
     * @brief Sets the device type
     * @param device Device type to set
     */
    void set_device(DeviceType device) noexcept { m_device = device; }

    /**
     * @brief Computes linear index from multi-dimensional indices
     *
     * @param indices Array of indices for each dimension
     * @return Linear memory offset
     * @throw brezel::core::error::LogicError if indices size doesn't match
     *                                        shape or indices are out of bounds
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t
    get_linear_index(std::span<const int64_t> indices) const {
        if (detail::unlikely(indices.size() != m_shape.size())) {
            throw brezel::core::error::InvalidArgument(
                "Indices size must match shape size");
        }

#ifdef BREZEL_CONFIG_DEBUG
        for (size_t i = 0; i < indices.size(); ++i) {
            if (detail::unlikely(indices[i] < 0 || indices[i] >= m_shape[i])) {
                throw brezel::core::error::LogicError("Index out of bounds");
            }
        }
#endif

        size_t offset = m_offset;
        const size_t ndim = m_shape.size();

        // Optimization for common tensor dimensions with unrolled loops
        if (ndim <= 4) {
            switch (ndim) {
                case 0:
                    return offset;

                case 1:
                    return offset + indices[0] * m_strides[0];

                case 2:
                    return offset + indices[0] * m_strides[0] +
                           indices[1] * m_strides[1];

                case 3:
                    return offset + indices[0] * m_strides[0] +
                           indices[1] * m_strides[1] +
                           indices[2] * m_strides[2];

                case 4:
                    return offset + indices[0] * m_strides[0] +
                           indices[1] * m_strides[1] +
                           indices[2] * m_strides[2] +
                           indices[3] * m_strides[3];
            }
        }

        for (size_t i = 0; i < ndim; ++i) {
            offset += indices[i] * m_strides[i];
        }

        return offset;
    }

    /**
     * @brief SIMD-accelerated dot product for index calculation
     *
     * @param a First vector
     * @param b Second vector
     * @return Dot product of the two vectors
     */
    BREZEL_NODISCARD BREZEL_FORCE_INLINE size_t simd_dot_product(
        std::span<const int64_t> a, std::span<const int64_t> b) const noexcept {
        size_t result = 0;

#if defined(BREZEL_SIMD_AVX512)
        if (a.size() >= 8) {
            const size_t simd_size = a.size() - (a.size() % 8);
            for (size_t i = 0; i < simd_size; i += 8) {
                __m512i a_vec = _mm512_loadu_epi64(a.data() + i);
                __m512i b_vec = _mm512_loadu_epi64(b.data() + i);
                __m512i prod = _mm512_mullo_epi64(a_vec, b_vec);
                result += _mm512_reduce_add_epi64(prod);
            }

            // Process remaining elements
            for (size_t i = simd_size; i < a.size(); ++i) {
                result += a[i] * b[i];
            }
        } else {
            result =
                std::inner_product(a.begin(), a.end(), b.begin(), size_t{0});
        }
#elif defined(BREZEL_SIMD_AVX2)
        if (a.size() >= 4) {
            const size_t simd_size = a.size() - (a.size() % 4);

            for (size_t i = 0; i < simd_size; i += 4) {
                __m256i a_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a.data() + i));
                __m256i b_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(b.data() + i));

                __m256i prod = _mm256_mul_epi32(a_vec, b_vec);

                int64_t sum_array[4];
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_array),
                                    prod);

                for (int j = 0; j < 4; ++j) {
                    result += sum_array[j];
                }
            }

            // Process remaining elements
            for (size_t i = simd_size; i < a.size(); ++i) {
                result += a[i] * b[i];
            }
        } else {
            result =
                std::inner_product(a.begin(), a.end(), b.begin(), size_t{0});
        }
#else
        // No SIMD support, std
        result = std::inner_product(a.begin(), a.end(), b.begin(), size_t{0});
#endif

        return result;
    }

    /**
     * @brief Converts linear index to multi-dimensional indices
     *
     * @param linear_index Linear memory index
     * @param indices Output array to store the calculated indices
     * @throws brezel::core::error::InvalidArgument if output indices array size
     *                                              doesn't match shape
     */
    BREZEL_FORCE_INLINE void get_indices(size_t linear_index,
                                         std::span<int64_t> indices) const {
        if (detail::unlikely(indices.size() != m_shape.size())) {
            throw brezel::core::error::InvalidArgument(
                "Output indices size ({}) must match shape size ({})",
                indices.size(), m_shape.size());
        }

        size_t remaining = linear_index - m_offset;
        if (m_layout == MemoryLayout::RowMajor) {
            const size_t ndim = m_shape.size();

            if (ndim <= 4) {
                switch (ndim) {
                    case 0:
                        return;

                    case 1:
                        indices[0] = remaining / m_strides[0];
                        return;

                    case 2:
                        indices[1] = remaining % m_strides[0];
                        indices[0] = remaining / m_strides[0];
                        return;

                    case 3:
                        indices[2] = remaining % m_strides[1];
                        remaining /= m_strides[1];
                        indices[1] = remaining % (m_strides[0] / m_strides[1]);
                        indices[0] = remaining / (m_strides[0] / m_strides[1]);
                        return;

                    case 4:
                        indices[3] = remaining % m_strides[2];
                        remaining /= m_strides[2];
                        indices[2] = remaining % (m_strides[1] / m_strides[2]);
                        remaining /= (m_strides[1] / m_strides[2]);
                        indices[1] = remaining % (m_strides[0] / m_strides[1]);
                        indices[0] = remaining / (m_strides[0] / m_strides[1]);
                        return;
                }
            }

            for (size_t i = 0; i < ndim; ++i) {
                indices[i] = remaining / m_strides[i];
                remaining %= m_strides[i];
            }

            return;
        }

        if (m_layout == MemoryLayout::ColumnMajor) {
            const size_t ndim = m_shape.size();

            if (ndim <= 4) {
                switch (ndim) {
                    case 0:
                        return;

                    case 1:
                        indices[0] = remaining / m_strides[0];
                        return;

                    case 2:
                        indices[0] = remaining % m_strides[1];
                        indices[1] = remaining / m_strides[1];
                        return;

                    case 3:
                        indices[0] = remaining % m_strides[1];
                        remaining /= m_strides[1];
                        indices[1] = remaining % (m_strides[2] / m_strides[1]);
                        indices[2] = remaining / (m_strides[2] / m_strides[1]);
                        return;

                    case 4:
                        indices[0] = remaining % m_strides[1];
                        remaining /= m_strides[1];
                        indices[1] = remaining % (m_strides[2] / m_strides[1]);
                        remaining /= (m_strides[2] / m_strides[1]);
                        indices[2] = remaining % (m_strides[3] / m_strides[2]);
                        indices[3] = remaining / (m_strides[3] / m_strides[2]);
                        return;
                }
            }

            for (int64_t i = ndim - 1; i >= 0; --i) {
                indices[i] = remaining / m_strides[i];
                remaining %= m_strides[i];
            }
            return;
        }

        compute_indices_from_offset(remaining, indices);
    }

    /**
     * @brief Compute indices from a linear offset for arbitrary strides
     *
     * @param offset Linear offset
     * @param indices Output indices array
     */
    BREZEL_FORCE_INLINE void compute_indices_from_offset(
        size_t offset, std::span<int64_t> indices) const {
        std::fill(indices.begin(), indices.end(), 0);
        if (offset == 0) {
            return;
        }

        const size_t ndim = m_shape.size();
        if (ndim == 0) {
            return;
        }

        size_t remaining = offset;
        bool progress = true;

        // Greedy algorithm for computing indices
        while (remaining > 0 && progress) {
            progress = false;

            for (size_t i = 0; i < ndim; ++i) {
                if (m_strides[i] > 0 && indices[i] < m_shape[i] - 1 &&
                    remaining >= m_strides[i]) {
                    remaining -= m_strides[i];
                    indices[i]++;

                    progress = true;
                }
            }
        }
    }

    /**
     * @brief Checks if the tensor's memory layout is contiguous.
     *
     * This function determines whether the tensor's memory layout is
     * contiguous. A contiguous memory layout means that the elements of the
     * tensor are stored in a single, continuous block of memory.
     *
     * @return true if the tensor's memory layout is contiguous, false
     * otherwise.
     *
     * @note A tensor with an empty shape (size 0) is considered contiguous.
     * @note For RowMajor and ColumnMajor memory layouts, the tensor is
     * contiguous if the offset is 0.
     * @note For other memory layouts, the tensor is contiguous if the offset is
     * 0 and the strides are row-contiguous.
     */
    BREZEL_NODISCARD bool is_contiguous() const noexcept {
        if (m_shape.size() == 0)
            return true;

        if (m_layout == MemoryLayout::RowMajor ||
            m_layout == MemoryLayout::ColumnMajor) {
            return m_offset == 0;
        }

        return m_offset == 0 && is_row_contiguous_strides();
    }

    /**
     * @brief Checks if the tensor strides are row-contiguous.
     *
     * This function determines if the strides of the tensor are row-contiguous,
     * meaning that the memory layout of the tensor is such that elements in the
     * same row are contiguous in memory.
     *
     * @return true if the tensor strides are row-contiguous, false otherwise.
     */
    BREZEL_NODISCARD bool is_row_contiguous_strides() const noexcept {
        if (m_shape.size() == 0)
            return true;

        detail::SmallVector<int64_t> expected_strides(m_shape.size());

        expected_strides.back() = 1;
        for (int64_t i = m_shape.size() - 2; i >= 0; --i) {
            expected_strides[i] = expected_strides[i + 1] * m_shape[i + 1];
        }

        return std::equal(m_strides.begin(), m_strides.end(),
                          expected_strides.begin());
    }

    /**
     * @brief Checks if the tensor is row contiguous.
     *
     * This function determines whether the tensor's data is stored in a
     * row-contiguous manner. A tensor is considered row contiguous if its
     * memory layout is row-major and its offset is zero. If the memory layout
     * is not row-major, it checks if the tensor's strides are row contiguous.
     *
     * @return true if the tensor is row contiguous, false otherwise.
     */
    BREZEL_NODISCARD bool is_row_contiguous() const noexcept {
        if (m_shape.size() == 0)
            return true;

        if (m_layout == MemoryLayout::RowMajor) {
            return m_offset == 0;
        }

        return m_offset == 0 && is_row_contiguous_strides();
    }

    /**
     * @brief Checks if the tensor has column-contiguous strides.
     *
     * This function determines whether the strides of the tensor are contiguous
     * in column-major order. In other words, it checks if the strides follow
     * the pattern expected for a tensor stored in column-major order.
     *
     * @return true if the tensor has column-contiguous strides, false
     * otherwise.
     */
    BREZEL_NODISCARD bool is_column_contiguous_strides() const noexcept {
        if (m_shape.size() == 0)
            return true;

        detail::SmallVector<int64_t> expected_strides(m_shape.size());

        expected_strides.front() = 1;
        for (size_t i = 1; i < m_shape.size(); ++i) {
            expected_strides[i] = expected_strides[i - 1] * m_shape[i - 1];
        }

        return std::equal(m_strides.begin(), m_strides.end(),
                          expected_strides.begin());
    }

    /**
     * @brief Checks if the tensor is column contiguous.
     *
     * This function determines whether the tensor's memory layout is contiguous
     * in column-major order.
     *
     * @return true if the tensor is column contiguous, false otherwise.
     *
     * @note A tensor is considered column contiguous if it has no shape (empty
     * tensor) or if it is in column-major layout with an offset of zero. If the
     * layout is not column-major, it also checks if the strides are column
     * contiguous.
     */
    BREZEL_NODISCARD bool is_column_contiguous() const noexcept {
        if (m_shape.size() == 0)
            return true;

        if (m_layout == MemoryLayout::ColumnMajor) {
            return m_offset == 0;
        }

        return m_offset == 0 && is_column_contiguous_strides();
    }

    /**
     * @brief Creates a new iterator
     * @return StridedIndex iterator
     */
    BREZEL_NODISCARD StridedIndex create_iterator() const {
        return StridedIndex(*this);
    }

    /**
     * @brief Converts the memory layout to RowMajor
     * @return LayoutDescriptor with row major layout
     */
    BREZEL_NODISCARD LayoutDescriptor to_row_major() const {
        LayoutDescriptor result(m_shape, MemoryLayout::RowMajor);
        result.set_format(m_format);
        result.set_device(m_device);

        return result;
    }

    /**
     * @brief Converts the memory layout to ColumnMajor
     * @return LayoutDescriptor with col major layout
     */
    BREZEL_NODISCARD LayoutDescriptor to_column_major() const {
        LayoutDescriptor result(m_shape, MemoryLayout::ColumnMajor);
        result.set_format(m_format);
        result.set_device(m_device);

        return result;
    }

    /**
     * @brief Transposes the layout by swapping two specified dimensions.
     *
     * This function creates a new LayoutDescriptor by swapping the specified
     * dimensions `dim0` and `dim1` in the shape and strides of the current
     * layout. If the specified dimensions are out of bounds or are the same,
     * appropriate actions are taken.
     *
     * @param dim0 The first dimension to swap.
     * @param dim1 The second dimension to swap.
     * @return A new LayoutDescriptor with the specified dimensions transposed.
     * @throws brezel::core::error::InvalidArgument if `dim0` or `dim1` are out
     * of bounds.
     */
    BREZEL_NODISCARD LayoutDescriptor transpose(size_t dim0,
                                                size_t dim1) const {
        if (dim0 >= m_shape.size() || dim1 >= m_shape.size()) {
            throw brezel::core::error::InvalidArgument(
                "Transposition dimensions ({}, {}) out of bounds for shape of "
                "size {}",
                dim0, dim1, m_shape.size());
        }

        if (dim0 == dim1) {
            return *this;
        }

        Shape new_shape = m_shape;
        std::swap(new_shape[dim0], new_shape[dim1]);

        auto new_strides = m_strides;
        std::swap(new_strides[dim0], new_strides[dim1]);

        auto result = LayoutDescriptor(new_shape, new_strides);
        result.set_offset(m_offset);
        result.set_format(m_format);
        result.set_device(m_device);

        return result;
    }

    /**
     * @brief Creates a permuted layout
     * @param permutation New dimension order
     * @return New layout descriptor with permuted dimensions
     * @throws InvalidArgument if permutation is invalid
     */
    LayoutDescriptor permute(const std::vector<size_t>& dims) const {
        if (dims.size() != m_shape.size()) {
            throw core::error::InvalidArgument(
                "Permutation dimensions ({}) must match shape size ({})",
                dims.size(), m_shape.size());
        }

        std::vector<bool> used(m_shape.size(), false);
        for (size_t dim : dims) {
            if (dim >= m_shape.size()) {
                throw core::error::InvalidArgument(
                    "Dimension {} is out of bounds for shape with {} "
                    "dimensions",
                    dim, m_shape.size());
            }

            if (used[dim]) {
                throw core::error::InvalidArgument(
                    "Duplicate dimension {} in permutation", dim);
            }

            used[dim] = true;
        }

        Shape new_shape;
        for (size_t dim : dims) {
            new_shape.push_back(m_shape[dim]);
        }

        std::vector<int64_t> new_strides(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            new_strides[i] = m_strides[dims[i]];
        }

        auto result = LayoutDescriptor(new_shape, new_strides);
        result.set_format(m_format);
        result.set_device(m_device);
        result.set_offset(m_offset);

        return result;
    }

    /**
     * @brief Reshapes the tensor to the specified new shape.
     *
     * This function reshapes the tensor to the given new shape while ensuring
     * that the total number of elements remains the same. If the tensor is
     * contiguous, the new layout will retain the original layout properties
     * such as offset, format, and device. If the tensor is not contiguous, the
     * new layout will be set to row-major order.
     *
     * @param new_shape The target shape to reshape the tensor to.
     * @return A new LayoutDescriptor with the specified shape.
     * @throws brezel::core::error::InvalidArgument if the number of elements in
     * the new shape does not match the number of elements in the original
     * shape.
     */
    BREZEL_NODISCARD LayoutDescriptor reshape(const Shape& new_shape) const {
        if (new_shape.numel() != m_shape.numel()) {
            throw brezel::core::error::InvalidArgument(
                "Reshape target shape must have the same number of elements. "
                "Original: {}, New: {}",
                m_shape.numel(), new_shape.numel());
        }

        validate_shape(new_shape);
        if (is_contiguous()) {
            auto result = LayoutDescriptor(new_shape, m_layout);
            result.set_offset(m_offset);
            result.set_format(m_format);
            result.set_device(m_device);

            return result;
        }

        return LayoutDescriptor(new_shape, MemoryLayout::RowMajor);
    }

    /**
     * @brief Checks if the current shape is broadcastable to the target shape.
     *
     * This function determines whether the current shape can be broadcasted to
     * the target shape according to the broadcasting rules. A shape is
     * broadcastable to another shape if:
     * - The current shape has fewer dimensions than or equal to the target
     * shape.
     * - For each dimension, starting from the trailing dimensions, the
     * dimension sizes must either be equal or one of them must be 1.
     *
     * @param target The target shape to which the current shape is to be
     * broadcasted.
     * @return true if the current shape is broadcastable to the target shape,
     * false otherwise.
     */
    BREZEL_NODISCARD bool is_broadcastable_to(const Shape& target) const {
        if (m_shape.size() > target.size()) {
            return false;
        }

        const size_t dim_diff = target.size() - m_shape.size();
        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (m_shape[i] != target[i + dim_diff] && m_shape[i] != 1) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Broadcasts the current layout to the specified target shape.
     *
     * This function attempts to broadcast the current tensor layout to the
     * given target shape. If the current shape is not broadcastable to the
     * target shape, an InvalidArgument exception is thrown.
     *
     * @param target_shape The shape to which the current layout should be
     * broadcasted.
     * @return LayoutDescriptor The new layout descriptor with the broadcasted
     * shape and strides.
     * @throws brezel::core::error::InvalidArgument If the current shape cannot
     * be broadcasted to the target shape.
     */
    BREZEL_NODISCARD LayoutDescriptor
    broadcast_to(const Shape& target_shape) const {
        if (!is_broadcastable_to(target_shape)) {
            throw brezel::core::error::InvalidArgument(
                "Cannot broadcast shape {} to target shape {}",
                m_shape.to_string(), target_shape.to_string());
        }

        validate_shape(target_shape);
        detail::SmallVector<int64_t> new_strides(target_shape.size(), 0);
        const size_t dim_diff = target_shape.size() - m_shape.size();

        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (m_shape[i] == 1) {
                new_strides[i + dim_diff] = 0;
            } else {
                new_strides[i + dim_diff] = m_strides[i];
            }
        }

        auto result = LayoutDescriptor(target_shape, new_strides);
        result.set_offset(m_offset);
        result.set_format(m_format);
        result.set_device(m_device);

        return result;
    }

    /**
     * @brief Equality operator for LayoutDescriptor.
     *
     * This operator compares two LayoutDescriptor objects for equality.
     *
     * @param other The LayoutDescriptor object to compare with.
     * @return true if all member variables (m_shape, m_layout, m_format,
     * m_device, m_offset, m_strides) are equal between the two objects, false
     * otherwise.
     */
    BREZEL_NODISCARD bool operator==(
        const LayoutDescriptor& other) const noexcept {
        return m_shape == other.m_shape && m_layout == other.m_layout &&
               m_format == other.m_format && m_device == other.m_device &&
               m_offset == other.m_offset && m_strides == other.m_strides;
    }

    /**
     * @brief Inequality operator for comparing two LayoutDescriptor objects.
     *
     * This operator checks if the current LayoutDescriptor object is not equal
     * to another LayoutDescriptor object.
     *
     * @param other The LayoutDescriptor object to compare with.
     * @return true if the objects are not equal, false otherwise.
     */
    BREZEL_NODISCARD bool operator!=(
        const LayoutDescriptor& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Returns the number of elements in the tensor.
     *
     * This function calculates and returns the total number of elements
     * in the tensor based on its shape.
     *
     * @return size_t The total number of elements in the tensor.
     */
    BREZEL_NODISCARD size_t numel() const noexcept { return m_shape.numel(); }

    std::string to_string() const {
        std::string result = "Layout(shape=[";
        for (size_t i = 0; i < m_shape.size(); ++i) {
            result += std::to_string(m_shape[i]);

            if (i < m_shape.size() - 1)
                result += ", ";
        }

        result += "], strides=[";
        for (size_t i = 0; i < m_strides.size(); ++i) {
            result += std::to_string(m_strides[i]);

            if (i < m_strides.size() - 1)
                result += ", ";
        }

        result += "], layout=";
        result += std::string(brezel::tensor::to_string(m_layout));

        result += ", format=";
        result += std::string(brezel::tensor::to_string(m_format));

        result += ", device=";
        result += std::string(brezel::tensor::to_string(m_device));

        if (m_offset > 0) {
            result += ", offset=" + std::to_string(m_offset);
        }

        result += ")";
        return result;
    }

private:
    Shape m_shape;
    detail::SmallVector<int64_t> m_strides;
    MemoryLayout m_layout;
    MemoryFormat m_format;
    DeviceType m_device;
    size_t m_offset = 0;

    // Helper functions
    /**
     * @brief Computes the strides for the tensor based on its shape and
     * memory layout.
     *
     * This function resizes the strides vector to match the shape size and
     * calculates the strides based on the specified memory layout (RowMajor
     * or ColumnMajor).
     */
    void compute_strides() {
        m_strides.resize(m_shape.size());
        if (m_shape.size() == 0)
            return;

        if (m_layout == MemoryLayout::RowMajor) {
            m_strides.back() = 1;
            for (int64_t i = static_cast<int64_t>(m_shape.size()) - 2; i >= 0;
                 --i) {
                m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
            }
        } else if (m_layout == MemoryLayout::ColumnMajor) {
            m_strides.front() = 1;
            for (size_t i = 1; i < m_shape.size(); ++i) {
                m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
            }
        }
    }

    /**
     * @brief Validates the given shape to ensure all dimensions are
     * non-negative.
     *
     * @param shape The shape to validate.
     * @throws brezel::core::error::InvalidArgument if any dimension in the
     * shape is negative.
     */
    void validate_shape(const Shape& shape) const {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] < 0) {
                throw brezel::core::error::InvalidArgument(
                    "Shape dimension {} has invalid negative size: {}", i,
                    shape[i]);
            }
        }
    }

    /**
     * @brief Validates the given strides to ensure all strides are
     * non-negative.
     *
     * @param strides The strides to validate.
     * @throws brezel::core::error::InvalidArgument if any stride is
     * negative.
     */
    void validate_strides(std::span<const int64_t> strides) const {
        for (size_t i = 0; i < strides.size(); ++i) {
            if (strides[i] < 0) {
                throw brezel::core::error::InvalidArgument(
                    "Negative stride {} at dimension {} is invalid", strides[i],
                    i);
            }
        }
    }

    /**
     * @brief Validates the given permutation to ensure it is a valid
     * permutation of the shape dimensions.
     *
     * @param permutation The permutation to validate.
     * @throws brezel::core::error::InvalidArgument if any index in the
     * permutation is out of bounds or duplicated.
     */
    void validate_permutation(std::span<const size_t> permutation) const {
        std::vector<bool> used(m_shape.size(), false);
        for (size_t idx : permutation) {
            if (idx >= m_shape.size()) {
                throw brezel::core::error::InvalidArgument(
                    "Permutation index {} out of bounds for shape of size "
                    "{}",
                    idx, m_shape.size());
            }

            if (used[idx]) {
                throw brezel::core::error::InvalidArgument(
                    "Duplicate dimension {} in permutation", idx);
            }

            used[idx] = true;
        }
    }

    /**
     * @brief Validates the slice parameters (start, end, and step) to
     * ensure they are valid for the tensor shape.
     *
     * @param start The start indices for the slice.
     * @param end The end indices for the slice.
     * @param step The step sizes for the slice.
     * @throws brezel::core::error::InvalidArgument if the sizes of start,
     * end, and step do not match the shape size, or if any step size is
     * non-positive.
     */
    void validate_slice_parameters(std::span<const int64_t> start,
                                   std::span<const int64_t> end,
                                   std::span<const int64_t> step) const {
        if (start.size() != m_shape.size() || end.size() != m_shape.size() ||
            step.size() != m_shape.size()) {
            throw brezel::core::error::InvalidArgument(
                "Start, end, and step must all have the same size as "
                "shape");
        }

        for (size_t i = 0; i < step.size(); ++i) {
            if (step[i] <= 0) {
                throw brezel::core::error::InvalidArgument(
                    "Step sizes must be positive, but got {} at dimension "
                    "{}",
                    step[i], i);
            }
        }
    }
};

/**
 * @brief Implementation of StridedIndex constructor
 *
 * @param layout Layout memory descriptor
 */
inline StridedIndex::StridedIndex(const LayoutDescriptor& layout) {
    m_ndim = layout.ndim();
    m_coordinates.resize(m_ndim, 0);
    m_shape.assign(layout.shape().begin(), layout.shape().end());
    m_strides.assign(layout.strides().begin(), layout.strides().end());
    m_initial_offset = layout.offset();
    m_offset = m_initial_offset;
    m_position = 0;

    m_total_size = 1;
    for (size_t i = 0; i < m_ndim; ++i) {
        m_total_size *= m_shape[i];
    }
}
}  // namespace brezel::tensor

// std::hash specialization for LayoutDescriptor
namespace std {
template <>
struct hash<brezel::tensor::LayoutDescriptor> {
    size_t operator()(const brezel::tensor::LayoutDescriptor& layout) const {
        size_t seed = 0;
        boost::hash_combine(seed, layout.shape().size());

        for (size_t i = 0; i < layout.shape().size(); ++i) {
            boost::hash_combine(seed, layout.shape()[i]);
        }

        boost::hash_combine(seed, layout.strides().size());
        for (size_t i = 0; i < layout.strides().size(); ++i) {
            boost::hash_combine(seed, layout.strides()[i]);
        }

        boost::hash_combine(seed, static_cast<uint8_t>(layout.layout()));
        boost::hash_combine(seed, static_cast<uint8_t>(layout.format()));
        boost::hash_combine(seed, static_cast<uint8_t>(layout.device()));
        boost::hash_combine(seed, layout.offset());

        return seed;
    }
};
}  // namespace std