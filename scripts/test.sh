#!/bin/bash

# Set exit on any error and error pipeline cache
set -e
set -o pipefail

BUILD_DIR="build"
VERBOSE="OFF"
TEST_NAME=""

# Help message
usage() {
  echo "Usage: $0 [-b BUILD_DIR] [-v] [-t TEST_NAME]"
  echo "  -b BUILD_DIR   Set the build directory (default: 'build')"
  echo "  -v             Enable verbose output"
  echo "  -t TEST_NAME   Run a specific test"
  exit 1
}

# Parse command-line arguments
while getopts "b:vt:h" opt; do
  case ${opt} in
  b) BUILD_DIR="$OPTARG" ;;
  v) VERBOSE="ON" ;;
  t) TEST_NAME="$OPTARG" ;;
  h) usage ;;
  *) usage ;;
  esac
done

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
  echo "Error: Build directory '$BUILD_DIR' does not exist. Run './build.sh' first."
  exit 1
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Run tests
echo "Running tests..."
if [ -n "$TEST_NAME" ]; then
  ctest -R "$TEST_NAME" --output-on-failure
else
  if [ "$VERBOSE" == "ON" ]; then
    ctest --output-on-failure --verbose
  else
    ctest --output-on-failure
  fi
fi

echo "Tests completed!"
