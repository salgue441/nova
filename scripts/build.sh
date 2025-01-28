#!/bin/bash
set -e

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX="install"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CUDA_OPTION="-DUSE_CUDA=OFF"

# Help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help            Show this help message"
  echo "  -t, --type TYPE       Build type (Debug|Release|RelWithDebInfo)"
  echo "  -b, --build-dir DIR   Build directory"
  echo "  -i, --install-dir DIR Installation directory"
  echo "  -j, --jobs N          Number of parallel jobs"
  echo "  --cuda                Enable CUDA support (disabled by default)"
  echo "  --clean               Clean build directory"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  -t | --type)
    BUILD_TYPE="$2"
    shift 2
    ;;
  -b | --build-dir)
    BUILD_DIR="$2"
    shift 2
    ;;
  -i | --install-dir)
    INSTALL_PREFIX="$2"
    shift 2
    ;;
  -j | --jobs)
    JOBS="$2"
    shift 2
    ;;
  --cuda)
    CUDA_OPTION="-DUSE_CUDA=ON"
    shift
    ;;
  --clean)
    rm -rf "${BUILD_DIR}"
    shift
    ;;
  *)
    echo "Unknown option: $1"
    show_help
    exit 1
    ;;
  esac
done

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo "Configuring with options: -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${CUDA_OPTION}"
cmake .. \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  ${CUDA_OPTION}

# Build
echo "Building with ${JOBS} jobs"
cmake --build . -j "${JOBS}"
