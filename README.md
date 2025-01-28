# NOVA

NOVA is a modern, high-performance tensor computation framework written in C++20. It provides a flexible and intuitive API for building and training neural networks, with seamless support for both CPU and GPU acceleration.

## Features

- **Modern C++ Design**: Built using C++20 features for improved performance and developer experience.
- **Hardware Acceleration**: Seamless CPU and CUDA AGPU support with unified API.
- **Automatic Differentiation**: Dynamic computational graph with reverse-mode automatic differentiation.
- **Neural Network API**: High-level modules for quick model prototyping and training.
- **Performance**: Optimized tensor operations with SIMD and thread pooling.
- **Extensible**: Easy to add custom operations and layers.
- **Cross-Platform**: Supports Windows, macOS, and Linux.

## Quick Start

```cpp
#include <nova/nova.hpp>
namespace nv = nova;

int main() {
    // Create a tensor on CPU
    auto x = nv::tensor::ones({2, 3});

    // Move to GPU if available
    x = x.cuda();

    // Basic operations
    auto y = nv::sin(x) + nv::cos(x);

    // Create a simple neural network
    auto model = nv::nn::Sequential({
        nv::nn::Linear(784, 256),
        nv::nn::ReLU(),
        nv::nn::Linear(256, 10),
        nv::nn::LogSoftmax()
    });

    // Training loop example
    auto optimizer = nv::optim::Adam(model.parameters());

    for (const auto& batch : dataloader) {
        optimizer.zero_grad();

        auto output = model(batch.input);
        auto loss = nv::nn::nll_loss(output, batch.target);

        loss.backward();
        optimizer.step();
    }

    return 0;
}
```

## Building from source

### Prerequisites

- CMake 3.20 or higher
- C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 19.29+)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- Google Test (for testing)
- Google Benchmark (for benchmarking)
- fmt library
- Doxygen (for documentation)

### Build instructions

```bash
# Clone the repository
git clone https://github.com/salgue441/nova.git
cd nova

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --config Release -j8

# Run tests
ctest --output-on-failure

# Generate documentation
cmake --build . --target docs
```

### Project Structure

```bash
nova/
├── benchmarks/          # Performance benchmarks
├── cmake/              # CMake modules and utilities
├── docs/               # Documentation
├── examples/           # Example projects and notebooks
├── include/            # Public headers
│   └── nova/
│       ├── core/       # Core tensor operations
│       ├── nn/         # Neural network modules
│       ├── optim/      # Optimizers
│       └── cuda/       # CUDA operations
├── src/               # Implementation files
├── tests/             # Test suite
└── tools/             # Development tools
```

## Documentation

Comprehensive documentation is available at [https://salgue441.github.io/nova/](https://salgue441.github.io/nova/).

- API Reference
- Tutorials
- Examples
- Performance Guide
- Contributing Guidelines

## Testing

The project uses Google Test for unit testing and integration testing. Tests are organized by component:

```bash
./bin/nova_tests

# Specific test suite
./bin/nova_tests --gtest_filter="TensorTest.*"
```

## Benchmarks

Performance benchmarks are implemented using Google Benchmark:

```bash
./bin/nova_benchmarks

# Specific benchmark
./bin/nova_benchmarks --benchmark_filter="TensorOps"
```

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CUDA integration is inspired by PyTorch's ATen library.
- Testing infrastucture is based on Google Test and Google Benchmark.
- Performance optimizations are based on Eigen and TensorFlow.
