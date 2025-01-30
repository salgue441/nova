#!/bin/bash

# Set exit on any error and error pipeline cache
set -e
set -o pipefail

BUILD_DIR="build"
EXECUTABLE=""
ARGS=""

# Help message
usage() {
  echo "Usage: $0 -e EXECUTABLE [-b BUILD_DIR] [-- ARGS...]"
  echo "  -e EXECUTABLE  Name of the executable to run (required)"
  echo "  -b BUILD_DIR   Set the build directory (default: 'build')"
  echo "  -- ARGS...     Additional arguments to pass to the executable"
  exit 1
}

# Parse command-line arguments
while getopts "e:b:h" opt; do
  case ${opt} in
  e) EXECUTABLE="$OPTARG" ;;
  b) BUILD_DIR="$OPTARG" ;;
  h) usage ;;
  *) usage ;;
  esac
done

# Shift to positional arguments (executable arguments)
shift $((OPTIND - 1))
ARGS="$@"

# Ensure executable is provided
if [ -z "$EXECUTABLE" ]; then
  echo "Error: No executable specified."
  usage
fi

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
  echo "Error: Build directory '$BUILD_DIR' does not exist. Run './build.sh' first."
  exit 1
fi

# Check if executable exists
EXEC_PATH="$BUILD_DIR/$EXECUTABLE"
if [ ! -f "$EXEC_PATH" ]; then
  echo "Error: Executable '$EXEC_PATH' not found."
  exit 1
fi

# Run executable
echo "Running '$EXECUTABLE' with arguments: $ARGS"
"$EXEC_PATH" $ARGS
