#pragma once

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brezel::autograd {
/**
 * @brief Represents a node in the computation graph.
 * @details Tracks operation history and gradients for automatic differentiation
 *
 */
class BREZEL_API Node{};

/**
 * @brief Variable class that wraps a tensor for automatic differentiation
 *
 */
class BREZEL_API Variable{};
}  // namespace brezel::autograd