#pragma once

#include <brezel/autograd/function.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace brezel::autograd {
// Forward declaration
class GradFunction;

/**
 * @brief Tensor with autograd capabilities
 *
 * Wraps a tensor and tracks gradients. This is the primary
 * interface for users of the autograd system.
 */
class BREZEL_API Variable : public std::enable_shared_from_this<Variable> {
public:
    /**
     * @brief Creates a variable without gradient tracking
     *
     * @param data Underlying tensor data
     */
    BREZEL_NODISCARD static std::shared_ptr<Variable> create(
        const tensor::Tensor<float>& data) {
        return std::shared_ptr<Variable>(new Variable(data, false));
    }

    /**
     * @brief Creates a variable with gradient tracking
     *
     * @param data Underlying tensor data
     * @param requires_grad Whether to track gradients
     */
    BREZEL_NODISCARD static std::shared_ptr<Variable> create(
        const tensor::Tensor<float>& data, bool requires_grad) {
        return std::shared_ptr<Variable>(new Variable(data, requires_grad));
    }

    /**
     * @brief Creates a variable from an operation on other variables
     *
     * @param data Resulting tensor data
     * @param grad_fn Gradient function for this operation
     * @param inputs Input variables to the operation
     */
    BREZEL_NODISCARD static std::shared_ptr<Variable> create(
        const tensor::Tensor<float>& data,
        std::shared_ptr<GradFunction> grad_fn,
        const std::vector<std::shared_ptr<Variable>>& inputs) {
        auto var = std::shared_ptr<Variable>(new Variable(data, true));

        var->m_grad_fn = grad_fn;
        grad_fn->set_inputs(inputs);
        grad_fn->set_output(var);

        return var;
    }

    // Constructors
    /**
     * @brief Construct a variable wrapping tensor
     *
     * @param data Underlying tensor data
     * @param requires_grad Whether to track gradients
     */
    Variable(const tensor::Tensor<float>& data, bool requires_grad)
        : m_data(data), m_requires_grad(requires_grad) {
        if (requires_grad)
            m_grad = tensor::Tensor<float>::zeros(data.shape());
    }

    // Accesors
    /**
     * @brief Get the underlying tensor data
     *
     * @return Tensor data
     */
    BREZEL_NODISCARD const tensor::Tensor<float>& data() const {
        return m_data;
    }

    /**
     * @brief Get the gradient tensor
     *
     * @return Gradient tensor
     */
    BREZEL_NODISCARD const tensor::Tensor<float>& grad() const {
        BREZEL_ENSURE(m_requires_grad, "Variable does not require gradients");
        return m_grad;
    }

    /**
     * @brief Check if gradient tracking is enabled
     *
     * @return Whether gradient tracking is enabled
     */
    BREZEL_NODISCARD bool requires_grad() const { return m_requires_grad; }

    /**
     * @brief Enable or disable gradient tracking
     *
     * @param requires_grad Whether to track gradients
     */
    void set_requires_grad(bool requires_grad) {
        m_requires_grad = requires_grad;

        if (requires_grad && m_grad.numel() == 0)
            m_grad = tensor::Tensor<float>::zeros(m_data.shape());
    }

    /**
     * @brief Get the gradient function
     *
     * @return Gradient function
     */
    BREZEL_NODISCARD std::shared_ptr<GradFunction> grad_fn() const {
        return m_grad_fn;
    }

    // Methods
    /**
     * @brief Compute and propagate gradients
     *
     * @param grad Optional external gradient to start backpropagation from
     */
    void backward(const tensor::Tensor<float>& grad = tensor::Tensor<float>()) {
        BREZEL_ENSURE(
            m_requires_grad,
            "Calling backward on a variable that doesn't require gradients");

        bool is_grad_empty = (grad.numel() == 0 || grad.data() == nullptr);
        BREZEL_ENSURE(
            m_data.numel() == 1 || !is_grad_empty,
            "Gradient can be implicitly created only for scalar outputs");

        tensor::Tensor<float> grad_output = grad;
        if (is_grad_empty) {
            BREZEL_ENSURE(m_data.numel() == 1,
                          "Non-scalar output requires explicit gradient");

            grad_output = tensor::Tensor<float>({1}, 1.0f);
        } else {
            grad_output = grad;
        }

        BREZEL_ENSURE(grad_output.shape() == m_data.shape(),
                      "Gradient shape mismatch: expected {}, got {}",
                      m_data.shape().to_string(),
                      grad_output.shape().to_string());

        std::vector<std::shared_ptr<Variable>> execution_plan =
            build_execution_plan();

        std::unordered_map<std::shared_ptr<Variable>, tensor::Tensor<float>>
            grads;

        grads[shared_from_this()] = grad_output;

        for (auto& var : execution_plan) {
            if (!var->requires_grad() || !var->grad_fn())
                continue;

            auto it = grads.find(var);
            if (it == grads.end())
                continue;

            auto& grad_output = it->second;
            auto grad_fn = var->grad_fn();
            auto input_grads = grad_fn->backward(grad_output);
            auto inputs = get_inputs_from_grad_fn(grad_fn);

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (!inputs[i]->requires_grad())
                    continue;

                auto input_grad = input_grads[i];
                if (grads.find(inputs[i]) == grads.end())
                    grads[inputs[i]] = input_grad;

                else
                    grads[inputs[i]] = grads[inputs[i]] + input_grad;
            }
        }

        for (auto& [var, grad] : grads) {
            if (var->requires_grad())
                var->m_grad = grad;
        }
    }

    /**
     * @brief Zero out the gradient
     */
    void zero_grad() {
        if (m_requires_grad)
            m_grad = tensor::Tensor<float>::zeros(m_data.shape());
    }

    /**
     * @brief Detach the variable from its computation history
     *
     * @return New variable with same data but no gradient history
     */
    BREZEL_NODISCARD std::shared_ptr<Variable> detach() const {
        return create(m_data, false);
    }

private:
    tensor::Tensor<float> m_data;
    tensor::Tensor<float> m_grad;
    bool m_requires_grad{false};
    std::shared_ptr<GradFunction> m_grad_fn = nullptr;

    // Utilities
    /**
     * @brief Build execution plan for backward propagation
     *
     * @return Topologically sorted list of variables
     */
    std::vector<std::shared_ptr<Variable>> build_execution_plan() {
        std::vector<std::shared_ptr<Variable>> result;
        std::unordered_set<std::shared_ptr<Variable>> visited;
        std::function<void(std::shared_ptr<Variable>)> dfs;

        dfs = [&](std::shared_ptr<Variable> var) {
            if (visited.find(var) != visited.end())
                return;

            visited.insert(var);
            auto inputs = get_inputs_from_grad_fn(var->grad_fn());

            for (const auto& input : inputs) {
                if (input->requires_grad())
                    dfs(input);
            }

            result.push_back(var);
        };

        dfs(shared_from_this());
        return result;
    }

    /**
     * @brief Get input variables from a gradient function
     *
     * @param grad_fn Gradient function
     * @return Vector of input variables
     */
    std::vector<std::shared_ptr<Variable>> get_inputs_from_grad_fn(
        std::shared_ptr<GradFunction> grad_fn) {
        if (!grad_fn)
            return {};

        std::vector<std::shared_ptr<Variable>> inputs;
        for (const auto& weak_input : grad_fn->get_inputs()) {
            if (auto input = weak_input.lock())
                inputs.push_back(input);
        }

        return inputs;
    }
};
}  // namespace brezel::autograd