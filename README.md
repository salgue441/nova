# BREZEL

BREZEL is a modern, high-performance tensor computation framework written in C++20. It provides a flexible and intuitive API for building and training neural networks, with seamless support for both CPU and GPU acceleration.

## Features

- **Modern C++ Design**: Built using C++20 features for improved performance and developer experience.
- **Hardware Acceleration**: Seamless CPU and CUDA GPU support with unified API.
- **Automatic Differentiation**: Dynamic computational graph with reverse-mode automatic differentiation.
- **Neural Network API**: High-level modules for quick model prototyping and training.
- **Performance**: Optimized tensor operations with SIMD and thread pooling.
- **Extensible**: Easy to add custom operations and layers.
- **Cross-Platform**: Supports Windows, macOS, and Linux.

## Quick Start

```cpp
#include <brezel/brezel.hpp>
namespace bz = brezel;

int main() {
    // Create a tensor on CPU
    auto x = bz::tensor::ones({2, 3});

    // Move to GPU if available
    x = x.cuda();

    // Basic operations
    auto y = bz::sin(x) + bz::cos(x);

    // Create a simple neural network
    auto model = bz::nn::Sequential({
        bz::nn::Linear(784, 256),
        bz::nn::ReLU(),
        bz::nn::Linear(256, 10),
        bz::nn::LogSoftmax()
    });

    // Training loop example
    auto optimizer = bz::optim::Adam(model.parameters());

    for (const auto& batch : dataloader) {
        optimizer.zero_grad();

        auto output = model(batch.input);
        auto loss = bz::nn::nll_loss(output, batch.target);

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
- Ninja build system (optional, recommended)

### Build instructions

```bash
# Clone the repository
git clone https://github.com/salgue441/brezel.git
cd brezel

# Basic build
./scripts/build.sh

# Debug build with CUDA support
./scripts/build.sh -t Debug -g

# Release build with all features
./scripts/build.sh -t Release -g -b -d -l -j 8 --prefix /usr/local

# Show all build options
./scripts/build.sh --help
```

### Available Build Options

- `-t, --type`: Build type (Debug|Release|RelWithDebInfo) [default: Release]
- `-c, --clean`: Clean build directory before building
- `-g, --cuda`: Enable CUDA support
- `-j, --jobs`: Number of parallel jobs
- `-p, --prefix`: Installation prefix
- `-b, --benchmarks`: Enable benchmarks build
- `-d, --docs`: Enable documentation build
- `-s, --sanitizer`: Enable sanitizer instrumentation
- `-l, --lto`: Enable Link Time Optimization

### Project Structure

```bash
brezel/
├── benchmarks/          # Performance benchmarks
├── cmake/              # CMake modules and utilities
├── docs/               # Documentation
├── examples/           # Example projects and notebooks
├── include/            # Public headers
│   └── brezel/
│       ├── core/       # Core tensor operations
│       ├── nn/         # Neural network modules
│       ├── optim/      # Optimizers
│       └── cuda/       # CUDA operations
├── scripts/           # Build and utility scripts
├── src/               # Implementation files
├── tests/             # Test suite
└── tools/             # Development tools
```

## Development

### Code Formatting

The project uses clang-format for code formatting. To format your code:

```bash
# Format all source files
./scripts/format.sh

# Check formatting without making changes
./scripts/format.sh -c

# Format specific files or directories
./scripts/format.sh src/ include/
```

### Testing

The project uses Google Test for unit testing and integration testing. Tests can be run using:

```bash
# Run all tests
./scripts/test.sh

# Run only unit tests
./scripts/test.sh -t unit

# Run tests matching pattern
./scripts/test.sh -f "TensorTest*"

# Run with code coverage report
./scripts/test.sh -c
```

### Running Examples and Benchmarks

To run examples and benchmarks:

```bash
# Run a specific example
./scripts/run.sh examples/basic_tensor

# Run with arguments
./scripts/run.sh examples/neural_net -- --input data.txt --epochs 100

# Run benchmarks with profiling
./scripts/run.sh -p benchmarks/tensor_ops
```

## Documentation

Comprehensive documentation is available at [https://salgue441.github.io/brezel/](https://salgue441.github.io/brezel/).

- API Reference
- Tutorials
- Examples
- Performance Guide
- Contributing Guidelines

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CUDA integration is inspired by PyTorch's ATen library.
- Testing infrastructure is based on Google Test and Google Benchmark.
- Performance optimizations are based on Eigen and TensorFlow.
