#pragma once

#include <brezel/core/macros.hpp>
#include <brezel/tensor/detail/tensor_concept.hpp>
#include <brezel/tensor/layout.hpp>
#include <concepts>
#include <initializer_list>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace brezel::tensor::detail {
/**
 * @brief Recursively computes the dimensions of a nested container structure.
 *
 * This function determines the shape/dimensions of nested containers like
 * vectors or initializer lists by recursively traversing the structure and
 * recording the size at each nesting level.
 *
 * @tparam T The container type to analyze
 * @param data The container whose dimensions need to be computed
 * @param dims Vector that will store the computed dimensions
 *
 * The function:
 * 1. Pushes the size of current container level to dims vector
 * 2. If container elements are themselves containers, recursively processes
 * first element
 * 3. Handles both std::initializer_list and other container types that meet
 * requirements
 */
template <typename T>
void compute_dims(const T& data, std::vector<int64_t>& dims) {
    dims.push_back(static_cast<int64_t>(data.size()));

    if constexpr (std::is_same_v<
                      std::remove_cvref_t<decltype(*data.begin())>,
                      std::initializer_list<std::remove_cvref_t<T>>> ||
                  requires(T t) {
                      {
                          *t.begin()
                      } -> std::same_as<
                          std::initializer_list<std::remove_cvref_t<T>>&>;
                  }) {
        if (!data.empty()) {
            compute_dims(*data.begin(), dims);
        }
    }
}

/**
 * @brief Fills a contiguous memory block with data from a nested list
 * structure.
 *
 * This function recursively traverses a nested list structure and fills a
 * contiguous memory block according to the specified tensor dimensions. It
 * handles arbitrary nesting levels and performs type conversion from the input
 * type to the tensor's scalar type.
 *
 * @tparam U The type of the nested list structure (should be a nested
 * container)
 * @tparam T The target scalar type that satisfies the TensorScalar concept
 *
 * @param data The nested list structure containing the source data
 * @param ptr Pointer to the contiguous memory block to fill
 * @param dims Vector containing the dimensions of the tensor
 * @param dim_idx Current dimension index in the recursive process
 * @param offset Current offset in the contiguous memory block
 *
 * @return The next memory offset after filling the current nested level
 *
 * @note If the input nested list contains more elements than specified in dims,
 *       excess elements are ignored.
 */
template <typename U, TensorScalar T>
size_t fill_from_nested_list(const U& data, T* ptr,
                             const std::vector<int64_t>& dims, size_t dim_idx,
                             size_t offset) {
    if (dim_idx == dims.size() - 1) {
        size_t i = 0;

        for (const auto& val : data) {
            if (i >= static_cast<size_t>(dims[dim_idx])) {
                break;
            }

            ptr[offset + i] = static_cast<T>(val);
            ++i;
        }

        return offset + dims[dim_idx];
    } else {
        size_t i = 0;
        size_t new_offset = offset;
        size_t stride = 1;

        for (size_t j = dim_idx + 1; j < dims.size(); ++i) {
            stride *= dims[j];
        }

        for (const auto& sublist : data) {
            if (i >= static_cast<size_t>(dims[dim_idx])) {
                break;
            }

            new_offset = fill_from_nested_list(sublist, ptr, dims, dim_idx + 1,
                                               offset + i * stride);
            ++i;
        }

        return new_offset;
    }
}

/**
 * @brief Formats a tensor element into a string representation.
 *
 * For floating point numbers:
 * - Uses fixed precision formatting
 * - Removes trailing zeros after decimal point
 * - Removes decimal point if no decimals remain
 *
 * For integral types:
 * - Converts directly to string
 *
 * For boolean values:
 * - Returns "true" or "false"
 *
 * For other types:
 * - Uses default string conversion
 *
 * @tparam T The scalar type of the tensor element
 * @param value The value to format
 * @param precision Number of decimal places for floating point numbers
 * (default: 4)
 * @return std::string The formatted string representation of the value
 */
template <TensorScalar T>
std::string format_element(const T& value, int precision = 4) {
    std::ostringstream oss;

    if constexpr (std::is_floating_point_v<T>) {
        oss << std::fixed << std::setprecision(precision) << value;

        std::string str = oss.str();
        size_t dot_pos = str.find('.');

        if (dot_pos != std::string::npos) {
            size_t last_non_zero = str.find_last_not_of('0');
            if (last_non_zero != std::string::npos && last_non_zero > dot_pos) {
                str.erase(last_non_zero + 1);
            } else if (last_non_zero == dot_pos) {
                str.erase(dot_pos);
            }
        }

        return str;
    } else if constexpr (std::is_integral_v<T>) {
        oss << value;
        return oss.str();
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        oss << value;
        return oss.str();
    }
}

/**
 * @brief Recursively prints a tensor to an output stream with formatting.
 *
 * This function handles the recursive printing of a multi-dimensional tensor,
 * with support for truncation of large dimensions and precision control.
 *
 * @param os Output stream to write to
 * @param data Pointer to the tensor data array
 * @param layout Layout descriptor containing stride information
 * @param dims Vector containing the dimensions of the tensor
 * @param dim_idx Current dimension index in the recursion
 * @param current_indices Vector storing current indices during traversal
 * @param max_per_line Maximum number of elements to print per line before
 * truncating (0 for no truncation)
 * @param precision Number of decimal places to show for floating point numbers
 *
 * @details
 * - For inner dimensions, prints elements in a single line with commas
 * - For outer dimensions, prints subarrays with newlines and proper indentation
 * - If max_per_line > 0, truncates output showing first and last max_per_line
 * elements
 * - Uses layout.get_linear_index() to compute correct memory locations
 * - Formats output with brackets [], commas, and ellipsis for truncation
 */
template <TensorScalar T>
void print_tensor_recursive(std::ostream& os, const T* data,
                            const LayoutDescriptor& layout,
                            const std::vector<int64_t>& dims, size_t dim_idx,
                            std::vector<int64_t>& current_indices,
                            int max_per_line, int precision) {
    if (dim_idx == dims.size() - 1) {
        oss << "[";

        bool truncated = false;
        size_t line_size = dims[dim_idx];

        if (max_per_line > 0 &&
            line_size > static_cast<size_t>(2 * max_per_line + 1)) {
            line_size = 2 * max_per_line + 1;
            truncated = true;
        }

        for (size_t i = 0; i < line_size; ++i) {
            if (truncated && i == max_per_line) {
                os << " ... ";

                i = dims[dim_idx] - max_per_line - 1;
                continue;
            }

            current_indices[dim_idx] = static_cast<int64_t>(i);
            size_t linear_index = layout.get_linear_index(current_indices);

            os << format_element(data[linear_idx], precision);
            if (i < line_size - 1) {
                os << ", ";
            }
        }

        os << "]";
    } else {
        os << "[";

        bool truncated = false;
        size_t subarray_count = dims[dim_idx];

        if (max_per_line > 0 &&
            subarray_count > static_cast<size_t>(2 * max_per_line + 1)) {
            subarray_count = 2 * max_per_line + 1;
            truncated = true;
        }

        for (size_t i = 0; i < subarray_count; ++i) {
            if (truncated && i == max_per_line) {
                os << " ... ";

                i = dims[dim_idx] - max_per_line - 1;
                continue;
            }

            current_indices[dim_idx] = static_cast<int64_t>(i);
            print_tensor_recursive(os, data, layout, dims, dim_idx + 1,
                                   current_indices, max_per_line, precision);

            if (i < subarray_count - 1) {
                os << ",\n" << std::string(dim_idx + 1, ' ');
            }
        }

        os << "]";
    }
}

/**
 * @brief Converts a tensor's data to a string representation.
 *
 * This function takes a pointer to tensor data and its layout information and
 * generates a formatted string representation of the tensor's contents. The
 * output format includes proper indentation and line breaks for
 * multidimensional tensors.
 *
 * @tparam T The scalar type of the tensor elements (must satisfy TensorScalar
 * concept)
 * @param data Pointer to the tensor's data
 * @param layout The layout descriptor containing shape and stride information
 * @param max_per_line Maximum number of elements to print per line (default: 6)
 * @param precision Number of decimal places for floating-point numbers
 * (default: 4)
 *
 * @return std::string A formatted string representation of the tensor
 *
 * @note For 0-dimensional tensors (scalars), returns the single value formatted
 *       according to the specified precision.
 */
template <TensorScalar T>
std::string tensor_to_string(const T* data, const LayoutDescriptor& layout,
                             int max_per_line = 6, int precision = 4) {
    std::ostringstream os;
    const auto& shape = layout.shape();
    std::vector<int64_t> dims;

    dims.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        dims.push_back(shape[i]);
    }

    if (dims.empty()) {
        return format_element(data[0], precision);
    }

    std::vector<int64_t> current_indices(dims.size(), 0);
    print_tensor_recursive(os, data, layout, dims, 0, current_indices,
                           max_per_line, precision);

    return os.str();
}

/**
 * @brief A generic random number generator for numeric types
 *
 * @tparam T The numeric type for random generation (must be floating point or
 * integral)
 *
 * @details This class provides functionality to generate random numbers of a
 * specified numeric type within a given range. It supports both floating point
 * and integral types. The generator uses std::uniform_real_distribution for
 * floating point types and std::uniform_int_distribution for integral types.
 *
 * The class provides two constructors:
 * - One that uses a random device to seed the generator
 * - One that accepts a specific seed value
 *
 * It also provides methods to:
 * - Generate single random values via operator()
 * - Fill arrays with random values via the fill() method
 *
 * @example
 * @code
 * RandomGenerator<float> rng(0.0f, 1.0f);  // Creates generator for floats
 * [0,1] float random_value = rng();              // Generates single random
 * float float array[100]; rng.fill(array, 100);                    // Fills
 * array with random values
 * @endcode
 *
 * @throws static_assertion If template parameter T is not a floating point or
 * integral type
 *
 * @note Thread safety is not guaranteed when the same generator instance is
 * used from multiple threads simultaneously
 */
template <typename T>
class RandomGenerator {
public:
    /**
     * @brief Constructs a random number generator for numeric types
     *
     * Creates a random number generator that produces values in the range
     * [min_val, max_val]. The generator works with both floating point and
     * integral types.
     *
     * @tparam T The numeric type (must be either floating point or integral)
     * @param min_val The minimum value in the range (inclusive)
     * @param max_val The maximum value in the range (inclusive)
     *
     * @throws static_assert If T is not a floating point or integral type
     *
     * @note Uses std::uniform_real_distribution for floating point types
     * @note Uses std::uniform_int_distribution for integral types
     */
    RandomGenerator(T min_val, T max_val)
        : m_min(min_val), m_max(max_val), m_device(), m_generator(m_device()) {
        if constexpr (std::is_floating_point_v<T>) {
            m_distribution =
                std::uniform_real_distribution<T>(min_val, max_val);
        } else if constexpr (std::is_integral_v<T>) {
            m_distribution = std::uniform_int_distribution<T>(min_val, max_val);
        } else {
            static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                          "RandomGenerator only works with numeric types");
        }
    }

    /**
     * @brief Constructor for RandomGenerator class that generates random
     * numbers within a specified range
     *
     * @tparam T The numeric type for the random values (must be floating point
     * or integral)
     * @param min_val The minimum value of the range (inclusive)
     * @param max_val The maximum value of the range (inclusive)
     * @param seed The seed value for the random number generator
     *
     * @throws static_assertion If template parameter T is not a floating point
     * or integral type
     *
     * @details Creates a uniform distribution of either real or integer values
     * based on the template parameter type T. For floating point types, uses
     * uniform_real_distribution. For integral types, uses
     * uniform_int_distribution.
     */
    RandomGenerator(T min_val, T max_val, uint32_t seed)
        : m_min(min_val), m_max(max_val), m_device(), m_generator(seed) {
        if constexpr (std::is_floating_point_v<T>) {
            m_distribution =
                std::uniform_real_distribution<T>(min_val, max_val);
        } else if constexpr (std::is_integral_v<T>) {
            m_distribution = std::uniform_int_distribution<T>(min_val, max_val);
        } else {
            static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                          "RandomGenerator only works with numeric types");
        }
    }

    /**
     * @brief Generates a random value based on the configured distribution.
     *
     * @return T A random value of type T generated according to the internal
     * distribution using the stored random number generator.
     */
    BREZEL_NODISCARD T operator()() { return m_distribution(m_generator); }

    /**
     * @brief Fills a contiguous block of memory with random values
     *
     * This function populates an array with random values generated from
     * the internal distribution and random number generator.
     *
     * @param data Pointer to the beginning of the memory block to fill
     * @param size Number of elements to fill
     *
     * @note The function assumes the memory pointed by data is valid and
     *       has enough space for size elements
     */
    void fill(T* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = m_distribution(m_generator);
        }
    }

private:
    T m_min;
    T m_max;
    std::random_device m_device;
    std::mt19937 m_generator;

    // Conditional type based on T
    std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
        std::conditional_t<std::is_integral_v<T>,
                           std::uniform_int_distribution<T>, void>>
        m_distribution;
};

/**
 * @brief A class that generates normally-distributed random numbers
 *
 * NormalGenerator provides functionality to generate random numbers that follow
 * a normal (Gaussian) probability distribution with configurable mean and
 * standard deviation parameters.
 *
 * @tparam T The floating point type used for the distribution parameters and
 * generated values
 *
 * Key features:
 * - Generates normally distributed random numbers
 * - Configurable mean and standard deviation
 * - Optional seed value for reproducible sequences
 * - Can generate single values or fill arrays
 * - Thread-safe random number generation
 *
 * @example
 * @code
 * NormalGenerator<float> generator(0.0f, 1.0f); // mean=0, stddev=1
 * float random_value = generator();  // Get single random value
 *
 * float array[100];
 * generator.fill(array, 100);  // Fill array with random values
 * @endcode
 *
 * @note Only works with floating point types (float, double, long double)
 * @see std::normal_distribution
 */
template <typename T>
class NormalGenerator {
public:
    /**
     * @brief Constructs a normal (Gaussian) random number generator
     *
     * Creates a generator that produces random numbers following a normal
     * distribution with specified mean and standard deviation parameters.
     *
     * @tparam T The floating point type used for the distribution parameters
     * @param mean The mean (average) of the normal distribution
     * @param stddev The standard deviation (spread) of the normal distribution
     *
     * @throws static_assert Failed if T is not a floating point type
     */
    NormalGenerator(T mean, T stddev)
        : m_mean(mean),
          m_stddev(stddev),
          m_device(),
          m_generator(m_device()),
          m_distribution(mean, stddev) {
        static_assert(std::is_floating_point_v<T>,
                      "NormalGenerator only works with floating point types");
    }

    /**
     * @brief Constructs a Normal (Gaussian) distribution random number
     * generator
     *
     * Creates a generator that produces random numbers following a normal
     * distribution with specified mean and standard deviation.
     *
     * @tparam T Floating point type for the distribution parameters and
     * generated values
     * @param mean The mean (average) of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     * @param seed The seed value for the random number generator
     *
     * @throws static_assertion Fails if T is not a floating point type
     */
    NormalGenerator(T mean, T stddev, uint32_t seed)
        : m_mean(mean),
          m_stddev(stddev),
          m_device(),
          m_generator(seed),
          m_distribution(mean, stddev) {
        static_assert(std::is_floating_point_v<T>,
                      "NormalGenerator only works with floating point types");
    }

    /**
     * @brief Generates a random value using the stored distribution and
     * generator
     * @return A random value of type T according to the configured distribution
     *
     * Function call operator that produces random numbers based on the internal
     * random number generator and distribution objects.
     */
    BREZEL_NODISCARD T operator()() { return m_distribution(m_generator); }

    /**
     * @brief Fills an array with random values based on the configured
     * distribution
     *
     * @param data Pointer to the array to be filled with random values
     * @param size Number of elements to fill in the array
     *
     * @pre data must point to a valid array with at least 'size' elements
     * @pre size must be greater than 0
     *
     * @note This function uses the internally stored random number generator
     * and distribution to generate the random values
     */
    void fill(T* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = m_distribution(m_generator);
        }
    }

private:
    T m_mean;
    T m_stddev;
    std::random_device m_device;
    std::mt19937 m_generator;
    std::normal_distribution<T> m_distribution;
};

/**
 * @brief A factory class for creating thread-local random number generators
 *
 * @tparam T The numeric type for the random numbers, must satisfy TensorScalar
 * concept
 *
 * @details This class provides a thread-safe way to create random number
 * generators with consistent seeding across different threads. It ensures that:
 * - Each thread gets its own unique random number generator
 * - The random sequences are deterministic and reproducible
 * - The generated numbers fall within a specified range [min_val, max_val]
 *
 * The factory uses a base seed combined with thread IDs to create unique but
 * deterministic seeds for each thread's generator. This approach maintains
 * reproducibility while avoiding contention between threads.
 *
 * @example
 * @code
 * ThreadLocalRandomGeneratorFactory<float> factory(-1.0f, 1.0f);
 * auto generator = factory.get_generator(thread_id);
 * @endcode
 *
 * @note The template parameter T must satisfy the TensorScalar concept, which
 * typically includes fundamental numeric types like float, double, etc.
 */
template <TensorScalar T>
class ThreadLocalRandomGeneratorFactory {
public:
    /**
     * @brief Constructs a thread-local random number generator factory with
     * specified range
     *
     * @tparam T The numeric type for the random number range
     * @param min_val The minimum value of the random number range (inclusive)
     * @param max_val The maximum value of the random number range (inclusive)
     *
     * @details Initializes the factory with a range [min_val, max_val] and
     * generates a base seed using a random device. This base seed will be used
     * to create thread-local random number generators.
     */
    ThreadLocalRandomGeneratorFactory(T min_val, T max_val)
        : m_min(min_val), m_max(max_val), m_base_seed(std::random_device{}()) {}

    /**
     * @brief Creates a random number generator for a specific thread
     *
     * @param thread_id The ID of the thread requesting the generator
     * @return RandomGenerator<T> A random number generator initialized with
     * thread-specific seed
     *
     * @details Creates a new random number generator instance using the base
     * seed plus thread ID to ensure unique but deterministic random sequences
     * across threads. The generator is configured with the min/max bounds
     * stored in the class.
     */
    RandomGenerator<T> get_generator(size_t thread_id) const {
        uint32_t seed = m_base_seed + static_cast<uint32_t>(thread_id);
        return RandomGenerator<T>(m_min, m_max, seed);
    }

private:
    T m_min;
    T m_max;
    uint32_t m_base_seed;
};

/**
 * @brief Factory class for creating thread-local normal distribution generators
 *
 * @tparam T The scalar type for the random number generation (must satisfy
 * TensorScalar concept)
 *
 * @details This class provides a thread-safe way to create random number
 * generators that follow a normal (Gaussian) distribution. Each thread gets its
 * own unique generator instance with a thread-specific seed derived from a base
 * seed.
 *
 * The factory ensures that:
 * - Each thread gets a uniquely seeded generator
 * - The random number sequence is reproducible for each thread
 * - The generated numbers follow the specified normal distribution parameters
 *
 * Usage example:
 * @code
 * ThreadLocalNormalGeneratorFactory<float> factory(0.0f, 1.0f);
 * auto generator = factory.get_generator(thread_id);
 * @endcode
 */
template <TensorScalar T>
class ThreadLocalNormalGeneratorFactory {
public:
    /**
     * @brief Constructor for ThreadLocalNormalGeneratorFactory that creates
     * normally distributed random number generators
     *
     * @tparam T The numeric type for the mean and standard deviation (typically
     * float or double)
     * @param mean The mean value of the normal distribution
     * @param stddev The standard deviation of the normal distribution
     *
     * @details Initializes a factory for thread-local normal distribution
     * generators with specified parameters. Uses a random device to generate
     * the base seed for the random number generation.
     */
    ThreadLocalNormalGeneratorFactory(T mean, T stddev)
        : m_mean(mean), m_stddev(stddev), m_base_seed(std::random_device{}()) {}

    /**
     * @brief Gets a normal distribution generator for a specific thread
     *
     * Creates and returns a normal distribution generator with the stored mean
     * and standard deviation values, using a unique seed based on the thread
     * ID.
     *
     * @param thread_id The ID of the thread requesting the generator
     * @return NormalGenerator<T> A normal distribution generator initialized
     * with mean, standard deviation and thread-specific seed
     */
    NormalGenerator<T> get_generator(size_t thread_id) const {
        uint32_t seed = m_base_seed + static_cast<uint32_t>(thread_id);
        return NormalGenerator<T>(m_mean, m_stddev, seed);
    }

private:
    T m_mean;
    T m_stddev;
    uint32_t m_base_seed;
};
}  // namespace brezel::tensor::detail