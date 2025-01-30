#!/usr/bin/env bash

# Strict mode
set -euo pipefail
IFS=$'\n\t'

# Terminal colors and styles
readonly BOLD='\033[1m'
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly GRAY='\033[0;90m'
readonly NC='\033[0m'

# Configuration variables with defaults
declare -A CONFIG=(
  [BUILD_DIR]="build"
  [BUILD_TYPE]="Release"
  [USE_CUDA]="OFF"
  [CLEAN_BUILD]="OFF"
  [JOBS]="$(nproc)"
  [INSTALL_PREFIX]=""
  [VCPKG_ROOT]="${VCPKG_ROOT:-}"
  [ENABLE_TESTING]="ON"
  [ENABLE_BENCHMARKS]="OFF"
  [ENABLE_DOCS]="OFF"
  [ENABLE_SANITIZER]="OFF"
  [ENABLE_LTO]="OFF"
  [BUILD_GENERATOR]="Ninja"
)

# Logging functions with emojis
log_header() {
  printf "\n${BOLD}${BLUE}=== %s ===${NC}\n" "$1"
}

log_info() {
  printf "${BLUE}ℹ️  %s${NC}\n" "$1"
}

log_success() {
  printf "${GREEN}✅ %s${NC}\n" "$1"
}

log_warning() {
  printf "${YELLOW}⚠️  %s${NC}\n" "$1"
}

log_error() {
  printf "${RED}❌ %s${NC}\n" "$1" >&2
}

log_step() {
  printf "${MAGENTA}🔨 %s${NC}\n" "$1"
}

# Error handler
trap 'log_error "Script failed on line $LINENO\n"' ERR

# Show spinner while a command is running
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
  while ps -p $pid >/dev/null; do
    local temp=${spinstr#?}
    printf " ${CYAN}%c${NC}" "$spinstr"
    local spinstr=$temp${spinstr%"$temp"}
    sleep $delay
    printf "\b\b"
  done
  printf "  \b\b"
}

# Help message
show_help() {
  local build_dir="${CONFIG[BUILD_DIR]:-build}"
  printf "%b" "
${BOLD}Brezel Tensor Framework Build Script${NC}

${BOLD}Usage:${NC}
    $(basename "$0") [options]

${BOLD}Options:${NC}
    ${CYAN}-t, --type${NC} TYPE          Build type (Debug|Release|RelWithDebInfo) [default: Release]
    ${CYAN}-c, --clean${NC}              Clean build directory before building
    ${CYAN}-g, --cuda${NC}               Enable CUDA support
    ${CYAN}-j, --jobs${NC} N             Number of parallel jobs [default: $(nproc)]
    ${CYAN}-p, --prefix${NC} PATH        Installation prefix
    ${CYAN}-v, --vcpkg${NC} PATH         Path to vcpkg root
    ${CYAN}-b, --benchmarks${NC}         Enable benchmarks build
    ${CYAN}-d, --docs${NC}               Enable documentation build
    ${CYAN}-s, --sanitizer${NC}          Enable sanitizer instrumentation
    ${CYAN}-l, --lto${NC}                Enable Link Time Optimization
    ${CYAN}-h, --help${NC}               Show this help message

${BOLD}Examples:${NC}
    # Basic build
    ./$(basename "$0")

    # Debug build with CUDA
    ./$(basename "$0") -t Debug -g

    # Release build with all features
    ./$(basename "$0") -t Release -g -b -d -l -j 8 --prefix /usr/local

${BOLD}Note:${NC}
    Build artifacts will be placed in the '${build_dir}' directory.
\n"
  exit 0
}

# Parse command line arguments
# Parse command line arguments
parse_args() {
  # Show help if no arguments provided
  if [[ $# -eq 0 ]]; then
    show_help
  fi

  while [[ $# -gt 0 ]]; do
    case $1 in
    -t | --type)
      [[ -z "${2:-}" ]] && log_error "Build type required" && exit 1
      CONFIG[BUILD_TYPE]="$2"
      shift 2
      ;;
    -c | --clean)
      CONFIG[CLEAN_BUILD]="ON"
      shift
      ;;
    -g | --cuda)
      CONFIG[USE_CUDA]="ON"
      shift
      ;;
    -j | --jobs)
      [[ -z "${2:-}" ]] && log_error "Number of jobs required" && exit 1
      CONFIG[JOBS]="$2"
      shift 2
      ;;
    -p | --prefix)
      [[ -z "${2:-}" ]] && log_error "Installation prefix required" && exit 1
      CONFIG[INSTALL_PREFIX]="$2"
      shift 2
      ;;
    -v | --vcpkg)
      [[ -z "${2:-}" ]] && log_error "vcpkg path required" && exit 1
      CONFIG[VCPKG_ROOT]="$2"
      shift 2
      ;;
    -b | --benchmarks)
      CONFIG[ENABLE_BENCHMARKS]="ON"
      shift
      ;;
    -d | --docs)
      CONFIG[ENABLE_DOCS]="ON"
      shift
      ;;
    -s | --sanitizer)
      CONFIG[ENABLE_SANITIZER]="ON"
      shift
      ;;
    -l | --lto)
      CONFIG[ENABLE_LTO]="ON"
      shift
      ;;
    -h | --help)
      show_help # This will exit directly
      ;;
    *)
      log_error "Unknown option: $1"
      show_help # This will exit directly
      ;;
    esac
  done
}

# Validate configuration
validate_config() {
  local valid_build_types=("Debug" "Release" "RelWithDebInfo" "MinSizeRel")

  if [[ ! " ${valid_build_types[*]} " =~ ${CONFIG[BUILD_TYPE]} ]]; then
    log_error "Invalid build type: ${CONFIG[BUILD_TYPE]}"
    exit 1
  fi

  if [[ ${CONFIG[JOBS]} -lt 1 ]]; then
    log_error "Invalid number of jobs: ${CONFIG[JOBS]}"
    exit 1
  fi

  # Validate CUDA configuration
  if [[ ${CONFIG[USE_CUDA]} == "ON" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
      log_error "CUDA support requested but nvcc not found"
      exit 1
    fi
  fi

  # Validate installation prefix if specified
  if [[ -n ${CONFIG[INSTALL_PREFIX]} ]]; then
    if [[ ! -d $(dirname ${CONFIG[INSTALL_PREFIX]}) ]]; then
      log_error "Parent directory of install prefix does not exist: ${CONFIG[INSTALL_PREFIX]}"
      exit 1
    fi
  fi

  # Validate vcpkg path if specified
  if [[ -n ${CONFIG[VCPKG_ROOT]} ]]; then
    if [[ ! -f "${CONFIG[VCPKG_ROOT]}/scripts/buildsystems/vcpkg.cmake" ]]; then
      log_error "Invalid vcpkg root path: ${CONFIG[VCPKG_ROOT]}"
      exit 1
    fi
  fi
}

# Check system dependencies
check_dependencies() {
  log_step "Checking system dependencies..."

  # Required dependencies
  local missing_required=()
  if ! command -v cmake >/dev/null 2>&1; then
    missing_required+=("cmake")
  fi
  if ! command -v git >/dev/null 2>&1; then
    missing_required+=("git")
  fi

  # Handle required dependencies
  if [[ ${#missing_required[@]} -ne 0 ]]; then
    log_error "Missing required dependencies: ${missing_required[*]}"
    log_info "Please install the missing dependencies:"
    if [[ -f /etc/debian_version ]]; then
      log_info "  sudo apt-get install ${missing_required[*]}"
    elif [[ -f /etc/fedora-release ]]; then
      log_info "  sudo dnf install ${missing_required[*]}"
    elif [[ -f /etc/arch-release ]]; then
      log_info "  sudo pacman -S ${missing_required[*]}"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
      log_info "  brew install ${missing_required[*]}"
    else
      log_info "  Please install: ${missing_required[*]}"
    fi
    exit 1
  fi

  # Optional: Ninja build system
  if ! command -v ninja >/dev/null 2>&1; then
    log_warning "Ninja build system not found, falling back to Make"
    CONFIG[BUILD_GENERATOR]="Unix Makefiles"
  else
    CONFIG[BUILD_GENERATOR]="Ninja"
  fi

  # CUDA dependency check
  if [[ ${CONFIG[USE_CUDA]} == "ON" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
      log_error "CUDA support requested but nvcc not found"
      log_info "Please install the CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
      exit 1
    fi
  fi

  log_success "All required dependencies satisfied"
}

# Setup build directory
setup_build_dir() {
  if [[ ${CONFIG[CLEAN_BUILD]} == "ON" ]]; then
    log_step "Cleaning build directory..."
    rm -rf "${CONFIG[BUILD_DIR]}"
  fi

  mkdir -p "${CONFIG[BUILD_DIR]}"
}

# Configure project
configure_project() {
  log_step "Configuring project..."

  local cmake_args=(
    "-DCMAKE_BUILD_TYPE=${CONFIG[BUILD_TYPE]}"
    "-DUSE_CUDA=${CONFIG[USE_CUDA]}"
    "-DBUILD_TESTING=${CONFIG[ENABLE_TESTING]}"
    "-DBUILD_BENCHMARKS=${CONFIG[ENABLE_BENCHMARKS]}"
    "-DBUILD_DOCS=${CONFIG[ENABLE_DOCS]}"
    "-DENABLE_SANITIZER=${CONFIG[ENABLE_SANITIZER]}"
    "-DENABLE_LTO=${CONFIG[ENABLE_LTO]}"
    "-G${CONFIG[BUILD_GENERATOR]}"
  )

  if [[ -n ${CONFIG[INSTALL_PREFIX]} ]]; then
    cmake_args+=("-DCMAKE_INSTALL_PREFIX=${CONFIG[INSTALL_PREFIX]}")
  fi

  if [[ -n ${CONFIG[VCPKG_ROOT]} ]]; then
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${CONFIG[VCPKG_ROOT]}/scripts/buildsystems/vcpkg.cmake")
  fi

  # Create a temporary file for CMake output
  local cmake_output=$(mktemp)

  # Run CMake and capture output
  if ! cmake -S . -B "${CONFIG[BUILD_DIR]}" "${cmake_args[@]}" >"${cmake_output}" 2>&1; then
    log_error "CMake configuration failed. Error output:"
    cat "${cmake_output}"
    rm "${cmake_output}"
    exit 1
  fi

  # Check for warnings in the output
  if grep -i "warning:" "${cmake_output}" >/dev/null; then
    log_warning "CMake configuration completed with warnings:"
    grep -i "warning:" "${cmake_output}"
  fi

  rm "${cmake_output}"
  log_success "Project configured successfully"
}

# Build project
build_project() {
  log_step "Building project..."

  # Create a temporary file for build output
  local build_output=$(mktemp)

  # Run build and capture output
  if ! cmake --build "${CONFIG[BUILD_DIR]}" --parallel "${CONFIG[JOBS]}" >"${build_output}" 2>&1; then
    log_error "Build failed. Error output:"
    cat "${build_output}"
    rm "${build_output}"
    exit 1
  fi

  # Check for warnings in the output
  if grep -i "warning:" "${build_output}" >/dev/null; then
    log_warning "Build completed with warnings:"
    grep -i "warning:" "${build_output}"
  fi

  rm "${build_output}"
  log_success "Build completed successfully"
}

# Print build summary
print_summary() {
  log_header "Build Configuration"
  printf "${GRAY}"
  printf "  %-20s %s\n" "Build Type:" "${CONFIG[BUILD_TYPE]}"
  printf "  %-20s %s\n" "CUDA Support:" "${CONFIG[USE_CUDA]}"
  printf "  %-20s %s\n" "Testing:" "${CONFIG[ENABLE_TESTING]}"
  printf "  %-20s %s\n" "Benchmarks:" "${CONFIG[ENABLE_BENCHMARKS]}"
  printf "  %-20s %s\n" "Documentation:" "${CONFIG[ENABLE_DOCS]}"
  printf "  %-20s %s\n" "Sanitizer:" "${CONFIG[ENABLE_SANITIZER]}"
  printf "  %-20s %s\n" "LTO:" "${CONFIG[ENABLE_LTO]}"
  printf "  %-20s %s\n" "Jobs:" "${CONFIG[JOBS]}"
  [[ -n ${CONFIG[INSTALL_PREFIX]} ]] && printf "  %-20s %s\n" "Install Prefix:" "${CONFIG[INSTALL_PREFIX]}"
  printf "${NC}\n"
}

# Main function
main() {
  log_header "Brezel Tensor Framework Build Script"

  parse_args "$@"
  validate_config
  print_summary
  check_dependencies
  setup_build_dir
  configure_project
  build_project

  log_header "Build Completed! 🎉"
  printf "${BLUE}ℹ️  Build artifacts can be found in:${NC} ${CONFIG[BUILD_DIR]}\n"
}

# Execute main function
main "$@"
