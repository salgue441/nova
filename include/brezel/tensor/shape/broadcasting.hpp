#pragma once

#include <algorithm>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape/shape.hpp>
#include <brezel/tensor/shape/strides.hpp>
#include <vector>

namespace brezel::tensor::shape {

/**
 * @brief Provides broadcasting utilities for tensor operations.
 *
 * @details Broadcasting is a powerful mechanism that allows arrays with
 * different shapes to be combined in operations. The rules follow NumPy/PyTorch
 * semantics.
 */
class BREZEL_API Broadcasting {
public:
    /**
     * @brief Calculate the broadcasted shape from two input shapes
     *
     * @param shape1 First shape
     * @param shape2 Second shape
     * @return Shape The resulting broadcasted shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD static Shape broadcast_shapes(const Shape& shape1,
                                                   const Shape& shape2) {
        return shape1.broadcast_with(shape2);
    }

    /**
     * @brief Calculate the broadcasted shape from multiple input shapes
     *
     * @param shapes Vector of shapes to broadcast together
     * @return Shape The resulting broadcasted shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD static Shape broadcast_shapes(
        const std::vector<Shape>& shapes) {
        if (shapes.empty()) {
            return Shape();
        }

        if (shapes.size() == 1) {
            return shapes[0];
        }

        Shape result = shapes[0];
        for (size_t i = 1; i < shapes.size(); ++i) {
            result = result.broadcast_with(shapes[i]);
        }

        return result;
    }

    /**
     * @brief Calculate broadcasting strides for a shape to a target shape
     *
     * @param shape Original shape
     * @param target Target shape to broadcast to
     * @return Strides Broadcasting strides
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD static Strides broadcast_strides(const Shape& shape,
                                                      const Shape& target) {
        return Strides::broadcast(shape, target);
    }

    /**
     * @brief Checks if an index is valid for broadcasted access
     *
     * @param index Multi-dimensional index
     * @param shape Shape to check against
     * @param broadcasted_shape Broadcasted shape
     * @return bool True if the index is valid
     */
    BREZEL_NODISCARD static bool is_valid_broadcast_index(
        const std::vector<int64_t>& index, const Shape& shape,
        const Shape& broadcasted_shape) {
        const size_t dim_diff = broadcasted_shape.size() - shape.size();

        for (size_t i = 0; i < shape.size(); ++i) {
            const size_t target_dim = i + dim_diff;

            if (shape[i] != 1 &&
                (index[target_dim] < 0 || index[target_dim] >= shape[i])) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Calculate an original index from a broadcasted index
     *
     * @param broadcasted_index Index in the broadcasted shape
     * @param original_shape Original shape
     * @param broadcasted_shape Broadcasted shape
     * @return std::vector<int64_t> Corresponding index in the original shape
     */
    BREZEL_NODISCARD static std::vector<int64_t> get_original_index(
        const std::vector<int64_t>& broadcasted_index,
        const Shape& original_shape, const Shape& broadcasted_shape) {
        std::vector<int64_t> original_index(original_shape.size());

        const size_t dim_diff =
            broadcasted_shape.size() - original_shape.size();

        for (size_t i = 0; i < original_shape.size(); ++i) {
            const size_t target_dim = i + dim_diff;

            original_index[i] =
                (original_shape[i] == 1) ? 0 : broadcasted_index[target_dim];
        }

        return original_index;
    }

    /**
     * @brief Calculate the memory offset for a broadcasted index
     *
     * @param broadcasted_index Index in the broadcasted shape
     * @param original_shape Original shape
     * @param original_strides Original strides
     * @param broadcasted_shape Broadcasted shape
     * @return size_t Memory offset in the original data
     */
    BREZEL_NODISCARD static size_t get_broadcast_offset(
        const std::vector<int64_t>& broadcasted_index,
        const Shape& original_shape, const Strides& original_strides,
        const Shape& broadcasted_shape) {
        std::vector<int64_t> original_index = get_original_index(
            broadcasted_index, original_shape, broadcasted_shape);

        return original_strides.get_linear_index(original_index);
    }

    /**
     * @brief Create efficient broadcast mapping for iteration
     *
     * @details Generates mapping information for efficiently iterating over
     * broadcasted tensors. This is particularly useful for operations on
     * tensors with different shapes.
     *
     * @param shapes Vector of shapes to broadcast
     * @return std::vector<Strides> Vector of broadcast strides for each shape
     * @throws LogicError if shapes cannot be broadcast together
     */
    BREZEL_NODISCARD static std::vector<Strides> create_broadcast_mapping(
        const std::vector<Shape>& shapes) {
        if (shapes.empty()) {
            return {};
        }

        Shape broadcasted_shape = broadcast_shapes(shapes);

        std::vector<Strides> broadcast_strides;
        broadcast_strides.reserve(shapes.size());

        for (const auto& shape : shapes) {
            broadcast_strides.push_back(
                Strides::broadcast(shape, broadcasted_shape));
        }

        return broadcast_strides;
    }

    /**
     * @brief Broadcast dimension information for optimized iteration
     */
    struct BroadcastDimension {
        size_t size;  // Size of this dimension

        std::vector<int64_t>
            strides;         // Stride for each tensor in this dimension
        bool is_contiguous;  // Whether this dimension is contiguous for tensors
    };

    /**
     * @brief Broadcast iteration plan for optimized execution
     */
    struct BroadcastPlan {
        Shape broadcasted_shape;                     // Final broadcasted shape
        std::vector<BroadcastDimension> dimensions;  // Optimized dimensions
        size_t total_elements;                       // Total elements
    };

    /**
     * @brief Generate optimized iteration plan for broadcasting
     *
     * @details For better performance, we can optimize the iteration by
     * identifying continuous memory regions and creating an optimized iteration
     * plan.
     *
     * @param shapes Vector of shapes to broadcast
     * @param strides Vector of strides for each shape
     * @return BroadcastPlan Optimized iteration plan
     */
    BREZEL_NODISCARD static BroadcastPlan create_broadcast_plan(
        const std::vector<Shape>& shapes, const std::vector<Strides>& strides) {
        if (shapes.empty() || strides.empty() ||
            shapes.size() != strides.size()) {
            throw core::error::InvalidArgument(
                "Invalid input for broadcast plan creation");
        }

        Shape broadcasted_shape = broadcast_shapes(shapes);

        BroadcastPlan plan;
        plan.broadcasted_shape = broadcasted_shape;
        plan.total_elements = broadcasted_shape.numel();

        plan.dimensions.reserve(broadcasted_shape.size());

        for (size_t dim = 0; dim < broadcasted_shape.size(); ++dim) {
            BroadcastDimension dimension;
            dimension.size = broadcasted_shape[dim];
            dimension.strides.resize(shapes.size());

            bool is_contiguous = true;

            for (size_t tensor_idx = 0; tensor_idx < shapes.size();
                 ++tensor_idx) {
                if (dim >=
                    broadcasted_shape.size() - shapes[tensor_idx].size()) {
                    size_t original_dim = dim - (broadcasted_shape.size() -
                                                 shapes[tensor_idx].size());

                    if (shapes[tensor_idx][original_dim] == 1) {
                        dimension.strides[tensor_idx] = 0;
                        is_contiguous = false;
                    } else {
                        dimension.strides[tensor_idx] =
                            strides[tensor_idx][original_dim];
                    }
                } else {
                    dimension.strides[tensor_idx] = 0;
                    is_contiguous = false;
                }
            }

            dimension.is_contiguous = is_contiguous;
            plan.dimensions.push_back(dimension);
        }

        return plan;
    }

    /**
     * @brief Pack multiple broadcasting dimensions into a single dimension when
     * possible
     *
     * @details This optimization identifies consecutive dimensions that can be
     * merged into a single dimension to reduce loop overhead and improve cache
     * locality.
     *
     * @param plan Original broadcast plan
     * @return BroadcastPlan Optimized broadcast plan with packed dimensions
     */
    BREZEL_NODISCARD static BroadcastPlan optimize_broadcast_plan(
        const BroadcastPlan& plan) {
        if (plan.dimensions.empty()) {
            return plan;
        }

        BroadcastPlan optimized;
        optimized.broadcasted_shape = plan.broadcasted_shape;
        optimized.total_elements = plan.total_elements;

        size_t num_tensors = plan.dimensions[0].strides.size();

        size_t current_dim = 0;
        while (current_dim < plan.dimensions.size()) {
            BroadcastDimension merged = plan.dimensions[current_dim];
            size_t next_dim = current_dim + 1;

            while (next_dim < plan.dimensions.size()) {
                const BroadcastDimension& next = plan.dimensions[next_dim];

                bool can_merge = true;
                for (size_t i = 0; i < num_tensors; ++i) {
                    if (!((merged.strides[i] == 0 && next.strides[i] == 0) ||
                          (merged.strides[i] != 0 && next.strides[i] != 0 &&
                           merged.strides[i] * merged.size ==
                               next.strides[i]))) {
                        can_merge = false;
                        break;
                    }
                }

                if (!can_merge) {
                    break;
                }

                for (size_t i = 0; i < num_tensors; ++i) {
                    if (merged.strides[i] == 0) {
                        merged.strides[i] = 0;
                    }
                }

                merged.size *= next.size;
                merged.is_contiguous =
                    merged.is_contiguous && next.is_contiguous;

                next_dim++;
            }

            optimized.dimensions.push_back(merged);
            current_dim = next_dim;
        }

        return optimized;
    }

    /**
     * @brief Apply a function to each element in the broadcasted tensors
     *
     * @details This is a high-level utility that handles all the broadcasting
     * details and applies a function to each element in the broadcasted result.
     *
     * @tparam Func Function type to apply
     * @tparam DataPtrs Pointer types to tensor data
     * @param func Function to apply
     * @param shapes Vector of shapes
     * @param strides Vector of strides
     * @param data_ptrs Pointers to tensor data
     */
    template <typename Func, typename... DataPtrs>
    static void apply_broadcasted(Func&& func, const std::vector<Shape>& shapes,
                                  const std::vector<Strides>& strides,
                                  DataPtrs... data_ptrs) {
        static_assert(sizeof...(DataPtrs) > 0,
                      "At least one data pointer required");

        if (shapes.empty() || shapes.size() != sizeof...(DataPtrs)) {
            throw core::error::InvalidArgument(
                "Number of shapes must match number of data pointers");
        }

        if (strides.size() != shapes.size()) {
            throw core::error::InvalidArgument(
                "Number of strides must match number of shapes");
        }

        BroadcastPlan plan = create_broadcast_plan(shapes, strides);
        plan = optimize_broadcast_plan(plan);

        apply_broadcast_plan(
            std::forward<Func>(func), plan, 0,
            std::vector<int64_t>(plan.broadcasted_shape.size(), 0), 0,
            data_ptrs...);
    }

private:
    /**
     * @brief Recursive implementation of applying a function to broadcasted
     * tensors
     *
     * @tparam Func Function type
     * @tparam DataPtrs Data pointer types
     * @param func Function to apply
     * @param plan Broadcast plan
     * @param dim Current dimension
     * @param indices Current indices
     * @param offset Current data offset
     * @param data_ptrs Pointers to tensor data
     */
    template <typename Func, typename... DataPtrs>
    static void apply_broadcast_plan(Func&& func, const BroadcastPlan& plan,
                                     size_t dim, std::vector<int64_t>& indices,
                                     size_t offset, DataPtrs... data_ptrs) {
        if (dim == plan.dimensions.size()) {
            func(data_ptrs...);
            return;
        }

        const BroadcastDimension& dimension = plan.dimensions[dim];

        std::array<int64_t, sizeof...(DataPtrs)> strides{
            plan.dimensions[dim].strides[static_cast<size_t>(
                &data_ptrs - &std::get<0>(std::make_tuple(data_ptrs...)))]...};

        for (size_t i = 0; i < dimension.size; ++i) {
            indices[dim] = i;

            if (dim == plan.dimensions.size() - 1) {
                func((data_ptrs +
                      (strides[&data_ptrs -
                               &std::get<0>(std::make_tuple(data_ptrs...))] *
                       i))...);
            } else {
                apply_broadcast_plan(
                    std::forward<Func>(func), plan, dim + 1, indices, offset,
                    (data_ptrs +
                     (strides[&data_ptrs -
                              &std::get<0>(std::make_tuple(data_ptrs...))] *
                      i))...);
            }
        }
    }
};

}  // namespace brezel::tensor::shape