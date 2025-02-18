#pragma once

#include <tbb/concurrent_unordered_map.h>

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/tensor.hpp>
#include <string>
#include <unordered_map>

namespace brezel::autograd {
/**
 * @brief Context for storing intermediate values required for gradient
 * computation.
 *
 * This class provides a storage mechanism for tensors and scalars that need to
 * be stored during the forward pass and retrieved during the backward pass.
 *
 */
struct BREZEL_API AutogradContext {
    std::unordered_map<std::string, tensor::Tensor<float>> saved_tensors;
    std::unordered_map<std::string, int64_t> saved_scalars;

    /**
     * @brief Save a tensor for backward computation
     *
     * @param name Key to identify the tensor
     * @param tensor Tensor to save
     */
    void save_for_backward(const std::string& name,
                           const tensor::Tensor<float>& tensor) {
        saved_tensors[name] = tensor;
    }

    /**
     * @brief Save a scalar for backward computation
     *
     * @param name Key to identify the scalar
     * @param value Scalar value to save
     */
    void save_for_backward(const std::string& name, int64_t value) {
        saved_scalars[name] = value;
    }

    /**
     * @brief Get a saved tensor
     *
     * @param name Key identifying the tensor
     * @return Reference to the saved tensor
     * @throws LogicError if the tensor wasn't found
     */
    const tensor::Tensor<float>& get_saved_tensor(
        const std::string& name) const {
        auto it = saved_tensors.find(name);
        BREZEL_ENSURE(it != saved_tensors.end(),
                      "Tensor '{}' not found in saved_tensors", name);

        return it->second;
    }

    /**
     * @brief Get a saved scalar
     *
     * @param name Key identifying the scalar
     * @return The saved scalar value
     * @throws LogicError if the scalar wasn't found
     */
    int64_t get_saved_scalar(const std::string& name) const {
        auto it = saved_scalars.find(name);
        BREZEL_ENSURE(it != saved_scalars.end(),
                      "Scalar '{}' not found in saved_scalars", name);

        return it->second;
    }
};
}  // namespace brezel::autograd