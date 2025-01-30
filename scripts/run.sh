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

# Configuration
declare -A CONFIG=(
  [BUILD_DIR]="build"
  [TARGET]=""
  [ARGS]=""
  [VERBOSE]="OFF"
  [PROFILE]="OFF"
  [LOG_OUTPUT]="OFF"
  [LOG_DIR]="logs"
)

# Log functions
log_header() { printf "\n${BOLD}${BLUE}=== %s ===${NC}\n" "$1"; }
log_info() { printf "${BLUE}ℹ️  %s${NC}\n" "$1"; }
log_success() { printf "${GREEN}✅ %s${NC}\n" "$1"; }
log_warning() { printf "${YELLOW}⚠️  %s${NC}\n" "$1"; }
log_error() { printf "${RED}❌ %s${NC}\n" "$1" >&2; }
log_step() { printf "${MAGENTA}🔨 %s${NC}\n" "$1"; }

# Error handler
trap 'log_error "Script failed on line $LINENO"' ERR

# Help message
show_help() {
  printf "%b" "
${BOLD}Brezel Tensor Framework Runner${NC}

${BOLD}Usage:${NC}
    $(basename "$0") [options] <target> [-- target_args]

${BOLD}Options:${NC}
    ${CYAN}-b, --build-dir${NC} DIR   Build directory [default: build]
    ${CYAN}-v, --verbose${NC}         Show verbose output
    ${CYAN}-p, --profile${NC}         Run with profiling enabled
    ${CYAN}-l, --log${NC}             Save output to log file
    ${CYAN}-h, --help${NC}            Show this help message

${BOLD}Examples:${NC}
    # Run a specific example
    ./$(basename "$0") examples/basic_tensor

    # Run with arguments
    ./$(basename "$0") examples/neural_net -- --input data.txt --epochs 100

    # Run with profiling
    ./$(basename "$0") -p benchmarks/tensor_ops

    # Run with logging
    ./$(basename "$0") -l examples/training -- --dataset mnist
"
  return 0
}

# Parse command line arguments
parse_args() {
  # Show help if no arguments
  if [[ $# -eq 0 ]]; then
    log_error "No target specified"
    show_help
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case $1 in
    -b | --build-dir)
      [[ -z "${2:-}" ]] && log_error "Build directory required" && exit 1
      CONFIG[BUILD_DIR]="$2"
      shift 2
      ;;
    -v | --verbose)
      CONFIG[VERBOSE]="ON"
      shift
      ;;
    -p | --profile)
      CONFIG[PROFILE]="ON"
      shift
      ;;
    -l | --log)
      CONFIG[LOG_OUTPUT]="ON"
      shift
      ;;
    -h | --help)
      show_help
      exit 0
      ;;
    --)
      shift
      CONFIG[ARGS]="$*"
      break
      ;;
    -*)
      log_error "Unknown option: $1"
      show_help
      exit 1
      ;;
    *)
      CONFIG[TARGET]="$1"
      shift
      ;;
    esac
  done

  # Check if target is specified
  if [[ -z "${CONFIG[TARGET]}" ]]; then
    log_error "No target specified"
    show_help
    exit 1
  fi
}

# Validate configuration
validate_config() {
  # Check build directory
  if [[ ! -d "${CONFIG[BUILD_DIR]}" ]]; then
    log_error "Build directory not found: ${CONFIG[BUILD_DIR]}"
    log_info "Please build the project first:"
    log_info "  ./build.sh"
    exit 1
  fi

  # Find target executable
  local target_path="${CONFIG[BUILD_DIR]}/${CONFIG[TARGET]}"
  if [[ ! -f "$target_path" ]]; then
    # Try with .exe extension for Windows
    target_path="${target_path}.exe"
  fi

  if [[ ! -f "$target_path" ]]; then
    log_error "Target not found: ${CONFIG[TARGET]}"
    log_info "Available targets:"
    find "${CONFIG[BUILD_DIR]}" -type f -executable -print0 |
      while IFS= read -r -d '' file; do
        local rel_path="${file#${CONFIG[BUILD_DIR]}/}"
        echo "  $rel_path"
      done
    exit 1
  fi

  CONFIG[TARGET]="$target_path"

  # Create log directory if needed
  if [[ "${CONFIG[LOG_OUTPUT]}" == "ON" ]]; then
    mkdir -p "${CONFIG[LOG_DIR]}"
  fi
}

# Setup profiling if enabled
setup_profiling() {
  if [[ "${CONFIG[PROFILE]}" == "ON" ]]; then
    if command -v perf &>/dev/null; then
      log_info "Using perf for profiling"
      CONFIG[PREFIX]="perf record -g --call-graph dwarf"
    else
      log_warning "perf not found, profiling disabled"
      CONFIG[PROFILE]="OFF"
      CONFIG[PREFIX]=""
    fi
  else
    CONFIG[PREFIX]=""
  fi
}

# Run the target
run_target() {
  local cmd=()

  # Add profiling prefix if enabled
  if [[ -n "${CONFIG[PREFIX]:-}" ]]; then
    read -ra prefix_args <<<"${CONFIG[PREFIX]}"
    cmd+=("${prefix_args[@]}")
  fi

  # Add target and its arguments
  cmd+=("${CONFIG[TARGET]}")
  if [[ -n "${CONFIG[ARGS]}" ]]; then
    read -ra target_args <<<"${CONFIG[ARGS]}"
    cmd+=("${target_args[@]}")
  fi

  # Prepare output handling
  local log_file=""
  if [[ "${CONFIG[LOG_OUTPUT]}" == "ON" ]]; then
    local target_name
    target_name=$(basename "${CONFIG[TARGET]}")
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="${CONFIG[LOG_DIR]}/${target_name}_${timestamp}.log"
  fi

  # Run command
  log_step "Running ${CONFIG[TARGET]##*/}..."
  if [[ "${CONFIG[VERBOSE]}" == "ON" ]]; then
    log_info "Command: ${cmd[*]}"
  fi

  local start_time
  start_time=$(date +%s)

  if [[ "${CONFIG[LOG_OUTPUT]}" == "ON" ]]; then
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
  else
    "${cmd[@]}"
    local status=$?
  fi

  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))

  # Print summary
  if [[ $status -eq 0 ]]; then
    log_success "Execution completed successfully"
  else
    log_error "Execution failed with status $status"
  fi

  printf "${BLUE}⏱️  Execution time:${NC} %ds\n" "$duration"

  if [[ "${CONFIG[LOG_OUTPUT]}" == "ON" ]]; then
    log_info "Output saved to: $log_file"
  fi

  if [[ "${CONFIG[PROFILE]}" == "ON" ]]; then
    log_info "Profile data saved. View with:"
    log_info "  perf report"
  fi

  return $status
}

# Main function
main() {
  log_header "Brezel Runner"

  parse_args "$@"
  validate_config
  setup_profiling

  # Print configuration
  if [[ "${CONFIG[VERBOSE]}" == "ON" ]]; then
    log_info "Configuration:"
    printf "  %-20s %s\n" "Target:" "${CONFIG[TARGET]}"
    printf "  %-20s %s\n" "Build Dir:" "${CONFIG[BUILD_DIR]}"
    printf "  %-20s %s\n" "Arguments:" "${CONFIG[ARGS]:-none}"
    printf "  %-20s %s\n" "Profiling:" "${CONFIG[PROFILE]}"
    printf "  %-20s %s\n" "Logging:" "${CONFIG[LOG_OUTPUT]}"
    echo
  fi

  run_target
}

# Execute main
main "$@"
