#pragma once

#include <boost/compute/algorithm.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/thread_pool.hpp>
#include <brezel/math/optimized_math.hpp>
#include <brezel/tensor/tensor.hpp>
#include <eigen3/Eigen/Dense>

namespace brezel::tensor {
/**
 * @brief Extends the Tensor class with optimized mathematical operations
 *
 * This namespace provides specialized mathematical functions that leverage
 * optimized implementations, SIMD instructions, and parallelization
 */
namespace math {
/**
 * @brief Thread pool for parallel math operations
 */

/**
 * @brief Apply optimized exponential function to each element of the tensor
 *
 * @tparam T Tensor element type
 * @param input Input tensor
 * @return New tensor with exponential applied to each element
 */
}  // namespace math
}  // namespace brezel::tensor