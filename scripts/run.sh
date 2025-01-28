#!/bin/bash
set -e

# Default values
BUILD_DIR="build"
BINARY=""

# Help message
show_help() {
  echo "Usage: $0 [options] binary_name [binary_args]"
  echo "Options:"
  echo "  -h, --help            Show this help message"
  echo "  -b, --build-dir DIR   Build directory"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  -b | --build-dir)
    BUILD_DIR="$2"
    shift 2
    ;;
  *)
    BINARY="$1"
    shift
    BINARY_ARGS="$@"
    break
    ;;
  esac
done

# Check binary name
if [ -z "${BINARY}" ]; then
  echo "Error: No binary specified"
  show_help
  exit 1
fi

# Check build directory
if [ ! -d "${BUILD_DIR}" ]; then
  echo "Error: Build directory not found"
  echo "Please run build.sh first"
  exit 1
fi

# Run binary
"${BUILD_DIR}/bin/${BINARY}" ${BINARY_ARGS}
