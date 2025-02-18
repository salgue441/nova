#pragma once

#include <brezel/autograd/function.hpp>
#include <brezel/autograd/variable.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <memory>
#include <vector>

namespace brezel::autograd {
/**
 * @brief Gradient function for the Sum reduction
 */
class BREZEL_API SumFunction : public GradFunctionImpl<SumFunction> {
public:
    SumFunction(int64_t dim = -1) : m_dim(dim) {
        m_op_type = OpType::Sum;
        m_name = "SumFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1, "SumFunction expects 1 input, got {}",
                      inputs.size());

        auto input = inputs[0];
        ctx.save_for_backward("input_shape",
                              static_cast<int64_t>(input.numel()));
        ctx.save_for_backward("input_ndim", static_cast<int64_t>(input.ndim()));
        ctx.save_for_backward("dim", m_dim);

        if (m_dim != -1) {
            for (size_t i = 0; i < input.ndim(); ++i)
                ctx.save_for_backward("shape_" + std::to_string(i),
                                      static_cast<int64_t>(input.shape()[i]));
        }

        return input.sum(m_dim);
    }

    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto input_size = ctx.get_saved_scalar("input_shape");
        const auto input_ndim = ctx.get_saved_scalar("input_ndim");
        const auto dim = ctx.get_saved_scalar("dim");

        if (dim == 1) {
            auto grad_scalar =
                tensor::Tensor<float>({1}, grad_output.data()[0]);
            auto grad_input =
                tensor::Tensor<float>::ones({static_cast<int64_t>(input_size)})
                    .multiply(grad_scalar);

            return {grad_input};
        } else {
            tensor::Shape original_shape;
            for (int64_t i = 0; i < input_ndim; ++i)
                original_shape.push_back(
                    ctx.get_saved_scalar("shape_" + std::to_string(i)));

            auto grad_input = tensor::Tensor<float>::ones(original_shape);
            auto grad_scalar =
                tensor::Tensor<float>({1}, grad_output.data()[0]);

            grad_input = grad_input.multiply(grad_scalar);
            return {grad_input};
        }
    }

private:
    int64_t m_dim;
};

/**
 * @brief Gradient function for the Mean reduction
 */
class BREZEL_API MeanFunction : public GradFunctionImpl<MeanFunction> {
public:
    MeanFunction(int64_t dim = -1) : m_dim(dim) {
        m_op_type = OpType::Mean;
        m_name = "MeanFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1,
                      "MeanFunction expects 1 input, got {}", inputs.size());

        auto input = inputs[0];
        ctx.save_for_backward("input_shape",
                              static_cast<int64_t>(input.numel()));
        ctx.save_for_backward("input_ndim", static_cast<int64_t>(input.ndim()));
        ctx.save_for_backward("dim", m_dim);

        if (m_dim != -1) {
            for (size_t i = 0; i < input.ndim(); i++) {
                ctx.save_for_backward("shape_" + std::to_string(i),
                                      static_cast<int64_t>(input.shape()[i]));
            }

            ctx.save_for_backward("dim_size",
                                  static_cast<int64_t>(input.shape()[m_dim]));
        }

        return input.mean(m_dim);
    }

    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto input_size = ctx.get_saved_scalar("input_shape");
        const auto input_ndim = ctx.get_saved_scalar("input_ndim");
        const auto dim = ctx.get_saved_scalar("dim");

        if (dim == -1) {
            float scale = 1.0f / input_size;
            auto grad_scalar =
                tensor::Tensor<float>({1}, grad_output.data()[0] * scale);

            auto grad_input =
                tensor::Tensor<float>::ones({static_cast<int64_t>(input_size)})
                    .multiply(grad_scalar);

            return {grad_input};
        } else {
            tensor::Shape original_shape;
            for (int64_t i = 0; i < input_ndim; i++) {
                original_shape.push_back(
                    ctx.get_saved_scalar("shape_" + std::to_string(i)));
            }

            const auto dim_size = ctx.get_saved_scalar("dim_size");
            float scale = 1.0f / dim_size;

            auto scaled_grad =
                tensor::Tensor<float>({1}, grad_output.data()[0] * scale);
            auto grad_input = tensor::Tensor<float>::ones(original_shape)
                                  .multiply(scaled_grad);

            return {grad_input};
        }
    }

private:
    int64_t m_dim;
};

/**
 * @brief Gradient function for the Max reduction
 */
class BREZEL_API MaxFunction : public GradFunctionImpl<MaxFunction> {
public:
    MaxFunction(int64_t dim = -1) : m_dim(dim) {
        m_op_type = OpType::Max;
        m_name = "MaxFunction";
    }

    tensor::Tensor<float> forward(
        const std::vector<tensor::Tensor<float>>& inputs,
        AutogradContext& ctx) override {
        BREZEL_ENSURE(inputs.size() == 1, "MaxFunction expects 1 input, got {}",
                      inputs.size());

        auto input = inputs[0];

        ctx.save_for_backward("input", input);
        ctx.save_for_backward("dim", m_dim);

        return input.max(m_dim);
    }

    std::vector<tensor::Tensor<float>> backward_impl(
        const tensor::Tensor<float>& grad_output,
        const AutogradContext& ctx) override {
        const auto& input = ctx.get_saved_tensor("input");
        const auto dim = ctx.get_saved_scalar("dim");

        if (dim == -1) {
            auto max_val = input.max().data()[0];
            auto ones = tensor::Tensor<float>::ones(input.shape());
            auto max_tensor = tensor::Tensor<float>({1}, max_val);
            auto equals_max = input.subtract(max_tensor).abs().clip(0.0f, 1e-5);
            auto mask = ones.subtract(equals_max.multiply(
                                          tensor::Tensor<float>({1}, 1e5)))
                            .clip(0.0f, 1.0f);

            auto grad_scalar =
                tensor::Tensor<float>({1}, grad_output.data()[0]);
            auto grad_input = mask.multiply(grad_scalar);

            return {grad_input};
        } else {
            auto max_vals = input.max(dim);
            auto grad_input = tensor::Tensor<float>::zeros(input.shape());
            const size_t n = input.numel();
            const size_t ndim = input.ndim();

            for (size_t i = 0; i < n; ++i) {
                std::vector<int64_t> indices(ndim);
                size_t temp = i;

                for (size_t j = 0; j < ndim; ++j) {
                    indices[j] = temp % input.shape()[j];
                    temp /= input.shape()[j];
                }

                std::vector<int64_t> max_indices = indices;
                max_indices.erase(max_indices.begin() + dim);

                if (std::abs(input.at(indices) - max_vals.at(max_indices)) <
                    1e-5) {
                    // Max, propagate gradient
                    std::vector<int64_t> grad_indices = max_indices;
                    grad_input.at(indices) = grad_output.at(grad_indices);
                }
            }

            return {grad_input};
        }
    }

private:
    int64_t m_dim;
};

namespace functions {

/**
 * @brief Sum reduction operation on a variable
 *
 * @param input Input variable
 * @param dim Dimension to reduce along
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> sum(
    const std::shared_ptr<Variable>& input, int64_t dim = -1) {
    auto grad_fn = std::make_shared<SumFunction>(dim);
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Mean reduction operation on a variable
 *
 * @param input Input variable
 * @param dim Dimension to reduce along
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> mean(
    const std::shared_ptr<Variable>& input, int64_t dim = -1) {
    auto grad_fn = std::make_shared<MeanFunction>(dim);
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

/**
 * @brief Max reduction operation on a variable
 *
 * @param input Input variable
 * @param dim Dimension to reduce along
 * @return Result variable
 */
BREZEL_NODISCARD inline std::shared_ptr<Variable> max(
    const std::shared_ptr<Variable>& input, int64_t dim = -1) {
    auto grad_fn = std::make_shared<MaxFunction>(dim);
    auto result = grad_fn->apply({input->data()});

    if (input->requires_grad())
        return Variable::create(result, grad_fn, {input});

    else
        return Variable::create(result, false);
}

}  // namespace functions
}  // namespace brezel::autograd