#pragma once

namespace brezel::autograd {

/**
 * @brief Enumeration of operation types for autograd function nodes
 *
 * Defines all the possible operations supported by the autograd system.
 * This helps with debugging and makes it easier to track the computation graph.
 */
enum class OpType {
    None,       ///< No operation or undefined
    Add,        ///< Addition operation
    Subtract,   ///< Subtraction operation
    Multiply,   ///< Multiplication operation
    Divide,     ///< Division operation
    MatMul,     ///< Matrix multiplication
    Pow,        ///< Power operation
    Exp,        ///< Exponential function
    Log,        ///< Natural logarithm
    Sum,        ///< Sum reduction
    Mean,       ///< Mean reduction
    Max,        ///< Maximum reduction
    Sigmoid,    ///< Sigmoid activation function
    Tanh,       ///< Tanh activation function
    ReLU,       ///< ReLU activation function
    Reshape,    ///< Reshape operation
    Transpose,  ///< Transpose operation
    View,       ///< View operation
    Custom      ///< Custom user-defined operation
};

}  // namespace brezel::autograd