#pragma once

#include <brezel/autograd/function.hpp>
#include <brezel/autograd/variable.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <limits>
#include <memory>
#include <vector>

namespace brezel::autograd {
/**
 * @brief Gradient function for the ReLU activation function
 *
 */
class BREZEL_API ReLUFunction : public GradFunctionImpl<ReLUFunction> {
public:
    ReLUFunction() {
        m_op_type = OpType::ReLU;
        m_name = "ReLUFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1,
                      "ReLU Function expects 1 input, got {}", inputs.size());

        auto input = inputs[0];
        auto output = input.clip(0.0f, std::numeric_limits<float>::max());

        ctx.save_for_backward("input", input);
        return output;
    }

    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& input = ctx.get_saved_tensor("input");
        auto mask = input.sign().clip(0.0f, 1.0f);

        return {mask * grad_output};
    }
};

/**
 * @brief Gradient function for the Sigmoid activation function
 */
class BREZEL_API SigmoidFunction : public GradFunctionImpl<SigmoidFunction> {
public:
    SigmoidFunction() {
        m_op_type = OpType::Sigmoid;
        m_name = "SigmoidFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1,
                      "SigmoidFunction expects 1 input, got {}", inputs.size());

        auto input = inputs[0];
        auto neg_one = tensor::Tensor<float>({1}, -1.0f);
        auto output = tensor::Tensor<float>::ones(input.shape())
                          .divide(tensor::Tensor<float>::ones(input.shape()) +
                                  (input.multiply(neg_one)).exp());

        ctx.save_for_backward("output", output);
        return output;
    }

    /**
     * @brief
     *
     * @details Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
     *
     * @param grad_output
     * @param ctx
     * @return std::vector<tensor::Tensor<float>>
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& output = ctx.get_saved_tensor("output");

        auto ones = tensor::Tensor<float>::ones(output.shape());
        auto sigmoid_grad = output.multiply(ones.subtract(output));
        auto grad_input = sigmoid_grad.multiply(grad_output);

        return {grad_input};
    }
};

/**
 * @brief Gradient function for the Tanh activation function
 */
class BREZEL_API TanhFunction : public GradFunctionImpl<TanhFunction> {
public:
    TanhFunction() {
        m_op_type = OpType::Tanh;
        m_name = "TanhFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1,
                      "TanhFunction expects 1 input, got {}", inputs.size());

        auto input = inputs[0];
        auto pos_exp = input.exp();
        auto neg_exp =
            (input.multiply(tensor::Tensor<float>({1}, -1.0f))).exp();
        auto output = (pos_exp - neg_exp).divide(pos_exp + neg_exp);

        ctx.save_for_backward("output", output);
        return output;
    }

    /**
     * @brief
     *
     * @details Tahn gradient: 1 - tanh(x)²
     *
     * @param grad_output
     * @param ctx
     * @return std::vector<tensor::Tensor<float>>
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& output = ctx.get_saved_tensor("output");

        auto grad_input =
            (tensor::Tensor<float>::ones(output.shape()) - (output * output)) *
            grad_output;

        return {grad_input};
    }
};

/**
 * @brief Gradient function for the LeakyReLU activation function
 */
class BREZEL_API LeakyReLUFunction
    : public GradFunctionImpl<LeakyReLUFunction> {
public:
    LeakyReLUFunction(float negative_slope = 0.01)
        : m_negative_slope(negative_slope) {
        m_op_type = OpType::ReLU;
        m_name = "LeakyReLUFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1,
                      "LeakyReLUFunction expects 1 input, got {}",
                      inputs.size());

        auto input = inputs[0];
        auto positive_part =
            input.clip(0.0f, std::numeric_limits<float>::max());
        auto negative_part =
            input.clip(std::numeric_limits<float>::lowest(), 0.0f)
                .multiply(tensor::Tensor<float>({1}, m_negative_slope));
        auto output = positive_part + negative_part;

        ctx.save_for_backward("input", input);
        ctx.save_for_backward("slope",
                              static_cast<int64_t>(static_cast<int64_t>(
                                  std::round(m_negative_slope * 1000.0f))));

        return output;
    }

    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& input = ctx.get_saved_tensor("input");
        float slope =
            static_cast<float>(ctx.get_saved_scalar("slope")) / 1000.0f;

        auto pos_mask = input.sign().clip(0.0f, 1.0f);
        auto neg_mask = tensor::Tensor<float>::ones(input.shape()) - pos_mask;

        auto slope_tensor = tensor::Tensor<float>({1}, slope);
        auto grad_mask = pos_mask + neg_mask * slope_tensor;

        return {grad_mask * grad_output};
    }

private:
    float m_negative_slope;
};

/**
 * @brief Gradient function for the ELU (Exponential Linear Unit) activation
 * function
 */
class BREZEL_API ELUFunction : public GradFunctionImpl<ELUFunction> {
public:
    ELUFunction(float alpha = 1.0) : m_alpha(alpha) {
        m_op_type = OpType::None;  // Custom activation
        m_name = "ELUFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1, "ELUFunction expects 1 input, got {}",
                      inputs.size());

        auto input = inputs[0];
        auto positive_part =
            input.clip(0.0f, std::numeric_limits<float>::max());
        auto negative_part =
            (input.clip(std::numeric_limits<float>::lowest(), 0.0f).exp() -
             tensor::Tensor<float>({1}, 1.0f));
        auto output = positive_part + negative_part;

        ctx.save_for_backward("output", output);
        ctx.save_for_backward("input", input);
        ctx.save_for_backward(
            "alpha", static_cast<int64_t>(
                         static_cast<int64_t>(std::round(m_alpha * 1000.0f))));

        return output;
    }

    /**
     * @brief
     *
     * @details gradient is 1 for x > 0, and alpha * exp(x) for x <= 0
     *
     * @param grad_output
     * @param ctx
     * @return std::vector<tensor::Tensor<float>>
     */
    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& output = ctx.get_saved_tensor("output");
        const auto& input = ctx.get_saved_tensor("input");
        float alpha =
            static_cast<float>(ctx.get_saved_scalar("alpha")) / 1000.0f;

        auto pos_mask = input.sign().clip(0.0f, 1.0f);
        auto neg_mask = tensor::Tensor<float>::ones(input.shape()) - pos_mask;

        auto pos_grad = pos_mask;
        auto alpha_tensor = tensor::Tensor<float>({1}, alpha);
        auto neg_grad = neg_mask * (output + alpha_tensor);

        auto grad_mask = pos_grad + neg_grad;

        return {grad_mask * grad_output};
    }

private:
    float m_alpha;
};

namespace functions {
/**
 * @brief Apply ReLU activation to a variable
 *
 * @param input Input variable
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> relu(
    const std::shared_ptr<Variable>& input) {
    auto grad_fn = std::make_shared<ReLUFunction>();
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Apply sigmoid activation to a variable
 *
 * @param input Input variable
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> sigmoid(
    const std::shared_ptr<Variable>& input) {
    auto grad_fn = std::make_shared<SigmoidFunction>();
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Apply tanh activation to a variable
 *
 * @param input Input variable
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> tanh(
    const std::shared_ptr<Variable>& input) {
    auto grad_fn = std::make_shared<TanhFunction>();
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Apply LeakyReLU activation to a variable
 *
 * @param input Input variable
 * @param negative_slope Slope for negative inputs (default 0.01)
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> leaky_relu(
    const std::shared_ptr<Variable>& input, float negative_slope = 0.01) {
    auto grad_fn = std::make_shared<LeakyReLUFunction>(negative_slope);
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Apply ELU (Exponential Linear Unit) activation to a variable
 *
 * @param input Input variable
 * @param alpha Scale parameter for negative values (default 1.0)
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> elu(
    const std::shared_ptr<Variable>& input, float alpha = 1.0) {
    auto grad_fn = std::make_shared<ELUFunction>(alpha);
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}
}  // namespace functions
}  // namespace brezel::autograd