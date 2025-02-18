#pragma once

#include <brezel/autograd/function.hpp>
#include <brezel/autograd/variable.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <memory>
#include <vector>

namespace brezel::autograd {
/**
 * @brief Gradient function for addition operation
 */
class BREZEL_API AddFunction : public GradFunctionImpl<AddFunction> {
public:
    AddFunction() {
        m_op_type = OpType::Add;
        m_name = "AddFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 2,
                      "AddFunction expects 2 inputs, got {}", inputs.size());

        return inputs[0] + inputs[1];
    }

    /**
     * @brief Computes backward implementation for the multiply function
     *
     * @details Gradient flow to both inputs unchanged
     *
     * @param grad_output Grad output tensor
     * @param ctx Autograd context object
     * @return std::vector<tensor::Tensor<float>> Result from backward pass
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        return {grad_output, grad_output};
    }
};

/**
 * @brief Gradient function for multiply operation
 */
class BREZEL_API MultiplyFunction : public GradFunctionImpl<MultiplyFunction> {
public:
    MultiplyFunction() {
        m_op_type = OpType::Multiply;
        m_name = "MultiplyFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 2,
                      "MultiplyFunction expects 2 inputs, got {}",
                      inputs.size());

        ctx.save_for_backward("input0", inputs[0]);
        ctx.save_for_backward("input1", inputs[1]);

        return inputs[0] + inputs[1];
    }

    /**
     * @brief Computes backward implementation for the multiply function
     *
     * @details Chain rule: d(a*b)/da = b * d(a*b)/d(a*b) = b * grad_output
     * @details Chain rule: d(a*b)/db = a * d(a*b)/d(a*b) = a * grad_output
     *
     * @param grad_output Gradient output tensor
     * @param ctx Autograd context object
     * @return std::vector<tensor::Tensor<float>> Result of the backwards pass
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& input0 = ctx.get_saved_tensor("input0");
        const auto& input1 = ctx.get_saved_tensor("input1");

        return {input1 * grad_output, input0 * grad_output};
    }
};

/**
 * @brief Gradient function for matrix multiplication
 */
class BREZEL_API MatMulFunction : public GradFunctionImpl<MatMulFunction> {
public:
    MatMulFunction() {
        m_op_type = OpType::MatMul;
        m_name = "MatMulFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 2,
                      "MatMulFunction expects 2 inputs, got {}", inputs.size());

        ctx.save_for_backward("input0", inputs[0]);
        ctx.save_for_backward("input1", inputs[1]);

        return inputs[0].matmul(inputs[1]);
    }

    /**
     * @brief Computes backward implementation for the multiply function
     *
     * @details For matrix multiplication C = A @ B
     * @details dL/dA = dL/dC * B.T
     * @details dL/dB = A.T * dL/dC
     *
     * @param grad_output Grad output tensor
     * @param ctx Autograd context object
     * @return std::vector<tensor::Tensor<float>> Result from backward pass
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& input0 = ctx.get_saved_tensor("input0");
        const auto& input1 = ctx.get_saved_tensor("input1");

        auto grad_input0 = grad_output.matmul(input1.transpose());
        auto grad_input1 = input0.transpose().matmul(grad_output);

        return {grad_input0, grad_input1};
    }
};

namespace functions {
/**
 * @brief Add two variables
 *
 * @param a First variable
 * @param b Second variable
 * @return Result variable
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE std::shared_ptr<Variable> add(
    const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto grad_fn = std::make_shared<AddFunction>();
    auto result = grad_fn->apply({a->data(), b->data()});

    if (a->requires_grad() || b->requires_grad())
        return Variable::create(result, grad_fn, {a, b});

    else
        return Variable::create(result, false);
}

/**
 * @brief Multiply two variables
 *
 * @param a First variable
 * @param b Second variable
 * @return Result variable
 */
BREZEL_NODISCARD BREZEL_FORCE_INLINE std::shared_ptr<Variable> multiply(
    const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto grad_fn = std::make_shared<MultiplyFunction>();
    auto result = grad_fn->apply({a->data(), b->data()});

    if (a->requires_grad() || b->requires_grad())
        return Variable::create(result, grad_fn, {a, b});

    else
        return Variable::create(result, false);
}

/**
 * @brief Matrix multiply two variables
 *
 * @param a First variable
 * @param b Second variable
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> matmul(
    const std::shared_ptr<Variable>& a, const std::shared_ptr<Variable>& b) {
    auto grad_fn = std::make_shared<MatMulFunction>();
    auto result = grad_fn->apply({a->data(), b->data()});

    if (a->requires_grad() || b->requires_grad())
        return Variable::create(result, grad_fn, {a, b});

    else
        return Variable::create(result, false);
}
}  // namespace functions
}  // namespace brezel::autograd