#!/bin/bash

# Set exit on any error and error pipeline cache
set -e
set -o pipefail

# Default values
BUILD_DIR="build"
BUILD_TYPE="Release"
USA_CUDA="OFF"
NUM_CORES=$(nproc)

# Help message
usage() {
  echo "Usage: $0 [-b BUILD_DIR] [-t BUILD_TYPE] [-c USE_CUDA] [-j NUM_CORES]"
  echo "  -b BUILD_DIR   Set the build directory (default: 'build')"
  echo "  -t BUILD_TYPE  Choose 'Debug' or 'Release' (default: 'Release')"
  echo "  -c USE_CUDA    Enable CUDA support: ON or OFF (default: OFF)"
  echo "  -j NUM_CORES   Number of cores for parallel build (default: auto-detect)"
  exit 1
}

# Parse command-line arguments
while getopts "b:t:c:j:h" opt; do
  case ${opt} in
  b) BUILD_DIR="$OPTARG" ;;
  t) BUILD_TYPE="$OPTARG" ;;
  c) USE_CUDA="$OPTARG" ;;
  j) NUM_CORES="$OPTARG" ;;
  h) usage ;;
  *) usage ;;
  esac
done

# Ensure valid build type
if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
  echo "Error: Invalid build type '$BUILD_TYPE'. Use 'Debug' or 'Release'."
  exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run CMake configuration
echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DUSE_CUDA="$USE_CUDA"

# Build the project
echo "Building project with $NUM_CORES cores..."
cmake --build . --parallel "$NUM_CORES"

echo "Build completed successfully!"
