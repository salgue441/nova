#pragma once

#include <brezel/autograd/context.hpp>
#include <brezel/autograd/op_type.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <memory>
#include <string>
#include <vector>

namespace brezel::autograd {
// Forward declarations
class Variable;

/**
 * @brief Base class for all gradient functions
 *
 * Acts as a node in the computation graph, storing inputs,
 * outputs, and implementing backward propagation logic
 */
class BREZEL_API GradFunction {
public:
    virtual ~GradFunction() = default;

    /**
     * @brief Apply the gradient function in forward direction
     *
     * @param inputs Input tensors
     * @return Output tensor after applying the function
     */
    virtual tensor::Tensor<float> apply(
        const std::vector<tensor::Tensor<float>>& inputs) = 0;

    /**
     * @brief Compute gradients for all inputs given output gradient
     *
     * @param grad_output Gradient flowing back from downstream
     * @return Vector of gradients for each input
     */
    virtual std::vector<tensor::Tensor<float>> backward(
        const tensor::Tensor<float>& grad_output) = 0;

    /**
     * @brief Set inputs to this grad function
     *
     * @param inputs Vector of input variables
     */
    void set_inputs(const std::vector<std::shared_ptr<Variable>>& inputs) {
        m_inputs.clear();

        for (const auto& var : inputs)
            m_inputs.push_back(var);
    }

    /**
     * @brief Set the next variable in the chain
     *
     * @param output Output variable
     */
    void set_output(std::shared_ptr<Variable> output) { m_output = output; }

    /**
     * @brief Get the operation type of this function
     *
     * @return Operation type
     */
    OpType op_type() const { return m_op_type; }

    /**
     * @brief Get the name of this function
     *
     * @return Function name
     */
    std::string name() const { return m_name; }

    /**
     * @brief Get the input variables (for topological sort)
     */
    const std::vector<std::weak_ptr<Variable>>& get_inputs() const {
        return m_inputs;
    }

protected:
    OpType m_op_type = OpType::None;
    std::string m_name = "GradFunction";
    std::vector<std::weak_ptr<Variable>> m_inputs;
    std::weak_ptr<Variable> m_output;
    AutogradContext m_ctx;
};

/**
 * @brief Base template for creating operation-specific gradient functions
 * @tparam Derived CRPT derived class
 */
template <typename Derived>
class GradFunctionImpl : public GradFunction {
public:
    /**
     * @brief Apply the gradient function in forward direction
     *
     * @param inputs Input tensors
     * @return Output tensor after applying the function
     */
    tensor::Tensor<float> apply(
        const std::vector<tensor::Tensor<float>>& inputs) override {
        return static_cast<Derived*>(this)->forward(inputs, m_ctx);
    }

    /**
     * @brief Compute gradients for all inputs given output gradient
     *
     * @param grad_output Gradient flowing back from downstream
     * @return Vector of gradients for each input
     */
    std::vector<tensor::Tensor<float>> backward(
        const tensor::Tensor<float>& grad_output) override {
        return static_cast<Derived*>(this)->backward_impl(grad_output, m_ctx);
    }

    virtual tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) = 0;

    virtual std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) = 0;
};
}  // namespace brezel::autograd