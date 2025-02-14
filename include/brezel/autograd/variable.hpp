#pragma once

#include <brezel/autograd/node.hpp>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <memory>
#include <unordered_set>
#include <vector>

namespace brezel::autograd {

/**
 * @brief Variable class that wraps tensors for automatic differentiation
 */
class BREZEL_API Variable {
public:
    /**
     * @brief Creates a variable from tensor data
     *
     * @param data Input tensor
     * @param requires_grad If true, enable gradient computation
     */
    explicit Variable(tensor::Tensor<float> data, bool requires_grad = false)
        : m_node(std::make_shared<Node>(std::move(data))) {
        m_node->set_requires_grad(requires_grad);
    }

    // Value accesor
    BREZEL_NODISCARD const tensor::Tensor<float>& data() const noexcept {
        return m_node->data();
    }

    BREZEL_NODISCARD const tensor::Tensor<float>* grad() const noexcept {
        return m_node->grad();
    }

    BREZEL_NODISCARD bool requires_grad() const noexcept {
        return m_node->requires_grad();
    }

    BREZEL_NODISCARD bool is_leaf() const noexcept { return m_node->is_leaf(); }

    /**
     * @brief Initializes backward pass through computation graph
     *
     */
    void backward() {
        BREZEL_ENSURE(
            requires_grad(),
            "Calling backward on a variable that doesn't require gradients");

        BREZEL_ENSURE(data().numel() == 1,
                      "Grad can be implicitly created only for scalar outputs");

        tensor::Tensor<float> grad({1}, 1.0f);
        m_node->accumulate_grad(grad);

        std::vector<std::shared_ptr<Node>> topo_sorted;
        std::unordered_set<Node*> visited;
        topo_sort(m_node, topo_sorted, visited);

        for (auto& node : topo_sorted)
            node->backward();
    }

    /**
     * @brief Retains computation graph
     */
    void retain_grad() noexcept { m_node->retain_grad(); }

    // Friend declaration for non-member functions
    friend Variable matmul(const Variable& lhs, const Variable& rhs);

private:
    /**
     * @brief Performs topological sort of computation graph
     *
     * @param node Current node
     * @param sorted Output vector of sorted nodes
     * @param visited Set of visited nodes
     */
    static void topo_sort(const std::shared_ptr<Node>& node,
                          std::vector<std::shared_ptr<Node>>& sorted,
                          std::unordered_set<Node*>& visited) {
        if (visited.insert(node.get()).second) {
            for (const auto& child : node->prev_nodes())
                topo_sort(child, sorted, visited);

            sorted.push_back(node);
        }
    }

    std::shared_ptr<Node> m_node;
};

/**
 * @brief Matrix multiplication operation
 *
 * @param lhs Left-hand side variable
 * @param rhs Right-hand side variable
 * @return Resulting variable
 */
BREZEL_NODISCARD inline Variable matmul(const Variable& lhs,
                                        const Variable& rhs) {
    auto result_tensor = lhs.data().multiply(rhs.data());
    Variable result(std::move(result_tensor),
                    lhs.requires_grad() || rhs.requires_grad());

    if (lhs.requires_grad()) {
        auto lhs_node = lhs.m_node;
        auto rhs_node = rhs.m_node;

        result.m_node->set_grad_fn(
            [lhs_node, rhs_node](const tensor::Tensor<float>& grad) {
                if (lhs_node->requires_grad()) {
                    lhs_node->accumulate_grad(
                        grad.matmul(rhs_node->data().transpose()));
                }

                if (rhs_node->requires_grad()) {
                    rhs_node->accumulate_grad(
                        lhs_node->data().transpose().multiply(grad));
                }
            },
            {lhs_node, rhs_node});
    }

    return result;
}
}  // namespace brezel::autograd