#pragma once

#include <tbb/concurrent_vector.h>

#include <boost/container/small_vector.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <functional>
#include <memory>
#include <vector>

namespace brezel::autograd {
/**
 * @brief Node in the computation graph that tracks operations and gradients
 *
 */
class BREZEL_API Node {
public:
    using GradientFunction = std::function<void(const tensor::Tensor<float>&)>;
    using NodePtr = std::shared_ptr<Node>;
    using WeakPtr = std::weak_ptr<Node>;

    /**
     * @brief Construct a new Node object with given tensor data
     * @param data Input tensor data
     */
    explicit Node(const tensor::Tensor<float>& data)
        : m_data(std::move(data)), m_requires_grad(false), m_is_leaf(true) {}

    // Allow copying and moving
    Node(const Node&) = default;
    Node& operator=(const Node&) = default;
    Node(Node&&) noexcept = default;
    Node& operator=(Node&&) noexcept = default;

    /**
     * @brief Sets whether node requires gradient computation
     * @param requires_grad If true, enable gradient computation
     */
    void set_requires_grad(bool requires_grad) noexcept {
        m_requires_grad = requires_grad;
        if (requires_grad && !m_grad) {
            m_grad = std::make_unique<tensor::Tensor<float>>(m_data.shape());
        }
    }

    /**
     * @brief Sets the gradient function for backward pass
     * @param grad_fn Function to compute gradients
     * @param prev_nodes Previous nodes in computation graph
     */
    void set_grad_fn(GradientFunction grad_fn,
                     std::vector<NodePtr> prev_nodes = {}) {
        m_grad_fn = std::move(grad_fn);
        m_prev_nodes = std::move(prev_nodes);
        m_is_leaf = false;
    }

    /**
     * @brief Retains gradient computation graph
     * Prevents computation graph from being freed before backward pass
     */
    void retain_grad() noexcept { m_retain_grad = true; }

    /**
     * @brief Accumulates gradients during backward pass
     * @param grad Gradient to accumulate
     */
    void accumulate_grad(const tensor::Tensor<float>& grad) {
        if (!m_grad) {
            m_grad = std::make_unique<tensor::Tensor<float>>(grad);
        } else {
            *m_grad = m_grad->add(grad);
        }
    }

    /**
     * @brief Computes gradients using stored gradient function
     */
    void backward() {
        if (m_grad_fn && m_grad) {
            m_grad_fn(*m_grad);
        }
    }

    // Accessors
    BREZEL_NODISCARD const tensor::Tensor<float>& data() const noexcept {
        return m_data;
    }

    BREZEL_NODISCARD const tensor::Tensor<float>* grad() const noexcept {
        return m_grad.get();
    }

    BREZEL_NODISCARD bool requires_grad() const noexcept {
        return m_requires_grad;
    }

    BREZEL_NODISCARD bool is_leaf() const noexcept { return m_is_leaf; }
    BREZEL_NODISCARD const std::vector<NodePtr>& prev_nodes() const noexcept {
        return m_prev_nodes;
    }

private:
    tensor::Tensor<float> m_data;
    std::unique_ptr<tensor::Tensor<float>> m_grad;
    GradientFunction m_grad_fn;
    std::vector<NodePtr> m_prev_nodes;
    bool m_requires_grad;
    bool m_is_leaf;
    bool m_retain_grad{false};
};
}  // namespace brezel::autograd