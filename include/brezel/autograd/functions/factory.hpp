#pragma once

#include <brezel/autograd/variable.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <memory>

namespace brezel::autograd {
namespace functions {

/**
 * @brief Create a variable without gradient tracking
 *
 * @param tensor Input tensor
 * @return Variable wrapping the tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> tensor(
    const tensor::Tensor<float>& tensor) {
    return Variable::create(tensor);
}

/**
 * @brief Create a variable with optional gradient tracking
 *
 * @param tensor Input tensor
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping the tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> tensor(
    const tensor::Tensor<float>& tensor, bool requires_grad) {
    return Variable::create(tensor, requires_grad);
}

/**
 * @brief Create a variable filled with zeros
 *
 * @param shape Shape of the tensor
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a zeros tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> zeros(
    const tensor::Shape& shape, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::zeros(shape), requires_grad);
}

/**
 * @brief Create a variable filled with ones

 * @param shape Shape of the tensor
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a ones tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> ones(
    const tensor::Shape& shape, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::ones(shape), requires_grad);
}

/**
 * @brief Create a variable filled with random values

 * @param shape Shape of the tensor
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a random tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> rand(
    const tensor::Shape& shape, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::random_uniform(shape),
                            requires_grad);
}

/**
 * @brief Create a variable filled with normally distributed random values

 * @param shape Shape of the tensor
 * @param mean Mean of the normal distribution
 * @param std Standard deviation of the normal distribution
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a random normal tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> randn(
    const tensor::Shape& shape, float mean = 0.0f, float std = 1.0f,
    bool requires_grad = false) {
    return Variable::create(
        tensor::Tensor<float>::random_normal(shape, mean, std), requires_grad);
}

/**
 * @brief Create a variable representing an identity matrix

 * @param size Size of the identity matrix
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping an identity matrix
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> eye(
    size_t size, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::eye(size), requires_grad);
}

/**
 * @brief Create a variable filled with a specific value

 * @param shape Shape of the tensor
 * @param value Value to fill the tensor with
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a filled tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> full(
    const tensor::Shape& shape, float value, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::full(shape, value),
                            requires_grad);
}

/**
 * @brief Create a variable from a range of values

 * @param start Start value
 * @param end End value
 * @param step Step between values
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping an arange tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> arange(
    float start, float end, float step = 1.0f, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::arange(start, end, step),
                            requires_grad);
}

/**
 * @brief Create a variable with linearly spaced values

 * @param start Start value
 * @param end End value
 * @param steps Number of steps
 * @param requires_grad Whether to track gradients
 * @return Variable wrapping a linspace tensor
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> linspace(
    float start, float end, size_t steps, bool requires_grad = false) {
    return Variable::create(tensor::Tensor<float>::linspace(start, end, steps),
                            requires_grad);
}

}  // namespace functions
}  // namespace brezel::autograd