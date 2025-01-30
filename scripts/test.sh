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
  [TEST_FILTER]=""
  [TEST_TYPE]="all" # all, unit, integration, benchmark
  [VERBOSE]="OFF"
  [JOBS]="$(nproc)"
  [REPEAT]="1"
  [OUTPUT_XML]="OFF"
  [OUTPUT_DIR]="test_results"
  [COVERAGE]="OFF"
)

# Test Categories
declare -a UNIT_TESTS=()
declare -a INTEGRATION_TESTS=()
declare -a BENCHMARKS=()

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
${BOLD}Brezel Tensor Framework Test Runner${NC}

${BOLD}Usage:${NC}
    $(basename "$0") [options]

${BOLD}Options:${NC}
    ${CYAN}-t, --type${NC} TYPE       Test type (all|unit|integration|benchmark) [default: all]
    ${CYAN}-f, --filter${NC} PATTERN  Only run tests matching pattern
    ${CYAN}-b, --build-dir${NC} DIR   Build directory [default: build]
    ${CYAN}-v, --verbose${NC}         Show verbose output
    ${CYAN}-j, --jobs${NC} N         Number of parallel jobs [default: $(nproc)]
    ${CYAN}-r, --repeat${NC} N       Repeat tests N times [default: 1]
    ${CYAN}-x, --xml${NC}            Generate XML test results
    ${CYAN}-c, --coverage${NC}       Generate coverage report
    ${CYAN}-h, --help${NC}           Show this help message

${BOLD}Examples:${NC}
    # Run all tests
    ./$(basename "$0")

    # Run only unit tests
    ./$(basename "$0") -t unit

    # Run tests matching pattern
    ./$(basename "$0") -f \"TensorTest*\"

    # Run with coverage
    ./$(basename "$0") -c
\n"
  exit 0
}

# Parse command line arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
    -t | --type)
      [[ -z "${2:-}" ]] && log_error "Test type required" && exit 1
      CONFIG[TEST_TYPE]="$2"
      shift 2
      ;;
    -f | --filter)
      [[ -z "${2:-}" ]] && log_error "Filter pattern required" && exit 1
      CONFIG[TEST_FILTER]="$2"
      shift 2
      ;;
    -b | --build-dir)
      [[ -z "${2:-}" ]] && log_error "Build directory required" && exit 1
      CONFIG[BUILD_DIR]="$2"
      shift 2
      ;;
    -v | --verbose)
      CONFIG[VERBOSE]="ON"
      shift
      ;;
    -j | --jobs)
      [[ -z "${2:-}" ]] && log_error "Number of jobs required" && exit 1
      CONFIG[JOBS]="$2"
      shift 2
      ;;
    -r | --repeat)
      [[ -z "${2:-}" ]] && log_error "Repeat count required" && exit 1
      CONFIG[REPEAT]="$2"
      shift 2
      ;;
    -x | --xml)
      CONFIG[OUTPUT_XML]="ON"
      shift
      ;;
    -c | --coverage)
      CONFIG[COVERAGE]="ON"
      shift
      ;;
    -h | --help)
      show_help
      ;;
    *)
      log_error "Unknown option: $1"
      show_help
      ;;
    esac
  done
}

# Validate configuration
validate_config() {
  # Validate test type
  if [[ ! "${CONFIG[TEST_TYPE]}" =~ ^(all|unit|integration|benchmark)$ ]]; then
    log_error "Invalid test type: ${CONFIG[TEST_TYPE]}"
    log_info "Valid types are: all, unit, integration, benchmark"
    exit 1
  fi

  # Validate build directory
  if [[ ! -d "${CONFIG[BUILD_DIR]}" ]]; then
    log_error "Build directory not found: ${CONFIG[BUILD_DIR]}"
    log_info "Please build the project first:"
    log_info "  ./build.sh         # Default build"
    log_info "  ./build.sh -t Debug # Debug build with tests"
    exit 1
  fi

  # Validate jobs
  if [[ ${CONFIG[JOBS]} -lt 1 ]]; then
    log_error "Invalid number of jobs: ${CONFIG[JOBS]}"
    log_info "Number of jobs must be at least 1"
    exit 1
  fi

  # Validate repeat count
  if [[ ${CONFIG[REPEAT]} -lt 1 ]]; then
    log_error "Invalid repeat count: ${CONFIG[REPEAT]}"
    log_info "Repeat count must be at least 1"
    exit 1
  fi

  # Validate output directory for XML
  if [[ "${CONFIG[OUTPUT_XML]}" == "ON" ]]; then
    mkdir -p "${CONFIG[OUTPUT_DIR]}"
  fi

  # Check for coverage tools if coverage is enabled
  if [[ "${CONFIG[COVERAGE]}" == "ON" ]]; then
    if ! command -v gcovr &>/dev/null; then
      log_warning "Coverage requested but gcovr not found"
      log_info "Install gcovr for coverage support:"
      log_info "  pip install gcovr"
      CONFIG[COVERAGE]="OFF"
    fi
  fi
}

# Find tests in build directory
find_tests() {
  log_step "Discovering tests..."

  # Find unit tests
  while IFS= read -r -d '' test_file; do
    if [[ $test_file =~ .*_test$ || $test_file =~ .*_test.exe$ ]]; then
      UNIT_TESTS+=("$test_file")
    fi
  done < <(find "${CONFIG[BUILD_DIR]}/tests" -type f -executable -print0 2>/dev/null || true)

  # Find integration tests
  while IFS= read -r -d '' test_file; do
    if [[ $test_file =~ .*_integration$ || $test_file =~ .*_integration.exe$ ]]; then
      INTEGRATION_TESTS+=("$test_file")
    fi
  done < <(find "${CONFIG[BUILD_DIR]}/tests" -type f -executable -print0 2>/dev/null || true)

  # Find benchmarks
  while IFS= read -r -d '' bench_file; do
    if [[ $bench_file =~ .*_bench$ || $bench_file =~ .*_bench.exe$ ]]; then
      BENCHMARKS+=("$bench_file")
    fi
  done < <(find "${CONFIG[BUILD_DIR]}/benchmarks" -type f -executable -print0 2>/dev/null || true)

  # Report findings
  log_info "Found:"
  printf "  %-20s %d\n" "Unit Tests:" "${#UNIT_TESTS[@]}"
  printf "  %-20s %d\n" "Integration Tests:" "${#INTEGRATION_TESTS[@]}"
  printf "  %-20s %d\n" "Benchmarks:" "${#BENCHMARKS[@]}"
}

# Run a single test
run_test() {
  local test_file="$1"
  local test_name
  test_name=$(basename "$test_file")
  local output_file
  output_file=$(mktemp)
  local status=0

  # Prepare test command
  local cmd=("$test_file")
  [[ -n "${CONFIG[TEST_FILTER]}" ]] && cmd+=("--gtest_filter=${CONFIG[TEST_FILTER]}")
  [[ "${CONFIG[VERBOSE]}" == "ON" ]] && cmd+=("--gtest_color=yes" "--gtest_print_time=1")
  [[ "${CONFIG[OUTPUT_XML]}" == "ON" ]] && cmd+=("--gtest_output=xml:${CONFIG[OUTPUT_DIR]}/${test_name}.xml")

  # Run test
  printf "Running %s..." "$test_name"
  if ! "${cmd[@]}" >"$output_file" 2>&1; then
    status=1
    printf "${RED}FAILED${NC}\n"
    if [[ "${CONFIG[VERBOSE]}" == "ON" ]]; then
      cat "$output_file"
    else
      grep -E "FAILED|ERROR" "$output_file" || true
    fi
  else
    printf "${GREEN}PASSED${NC}\n"
    [[ "${CONFIG[VERBOSE]}" == "ON" ]] && cat "$output_file"
  fi

  rm "$output_file"
  return $status
}

# Run a single benchmark
run_benchmark() {
  local bench_file="$1"
  local bench_name
  bench_name=$(basename "$bench_file")
  local output_file
  output_file=$(mktemp)
  local status=0

  # Prepare benchmark command
  local cmd=("$bench_file")
  [[ -n "${CONFIG[TEST_FILTER]}" ]] && cmd+=("--benchmark_filter=${CONFIG[TEST_FILTER]}")
  [[ "${CONFIG[VERBOSE]}" == "ON" ]] && cmd+=("--benchmark_color=yes")
  cmd+=("--benchmark_out=${CONFIG[OUTPUT_DIR]}/${bench_name}.json")
  cmd+=("--benchmark_out_format=json")

  # Run benchmark
  printf "Running %s..." "$bench_name"
  if ! "${cmd[@]}" >"$output_file" 2>&1; then
    status=1
    printf "${RED}FAILED${NC}\n"
    cat "$output_file"
  else
    printf "${GREEN}COMPLETED${NC}\n"
    cat "$output_file"
  fi

  rm "$output_file"
  return $status
}

# Run test suite
run_tests() {
  local failed=0
  local total=0
  local success_rate=0

  # Create output directory if needed
  [[ "${CONFIG[OUTPUT_XML]}" == "ON" ]] && mkdir -p "${CONFIG[OUTPUT_DIR]}"

  # Exit early if no tests found
  if [[ ${#UNIT_TESTS[@]} -eq 0 && ${#INTEGRATION_TESTS[@]} -eq 0 && ${#BENCHMARKS[@]} -eq 0 ]]; then
    log_warning "No tests found in ${CONFIG[BUILD_DIR]}"
    return 0
  fi

  # Run unit tests
  if [[ "${CONFIG[TEST_TYPE]}" =~ ^(all|unit)$ && ${#UNIT_TESTS[@]} -gt 0 ]]; then
    log_step "Running unit tests..."
    for test in "${UNIT_TESTS[@]}"; do
      ((total++))
      run_test "$test" || ((failed++))
    done
  fi

  # Run integration tests
  if [[ "${CONFIG[TEST_TYPE]}" =~ ^(all|integration)$ && ${#INTEGRATION_TESTS[@]} -gt 0 ]]; then
    log_step "Running integration tests..."
    for test in "${INTEGRATION_TESTS[@]}"; do
      ((total++))
      run_test "$test" || ((failed++))
    done
  fi

  # Run benchmarks
  if [[ "${CONFIG[TEST_TYPE]}" =~ ^(all|benchmark)$ && ${#BENCHMARKS[@]} -gt 0 ]]; then
    log_step "Running benchmarks..."
    for bench in "${BENCHMARKS[@]}"; do
      ((total++))
      run_benchmark "$bench" || ((failed++))
    done
  fi

  # Generate coverage report
  if [[ "${CONFIG[COVERAGE]}" == "ON" ]]; then
    log_step "Generating coverage report..."
    if command -v gcovr &>/dev/null; then
      gcovr -r . \
        --html --html-details \
        -o "${CONFIG[OUTPUT_DIR]}/coverage.html" \
        --exclude-directories=tests \
        --exclude-directories=third_party
      log_info "Coverage report generated at ${CONFIG[OUTPUT_DIR]}/coverage.html"
    else
      log_warning "gcovr not found, skipping coverage report"
    fi
  fi

  # Calculate success rate
  if [[ $total -gt 0 ]]; then
    success_rate=$(((total - failed) * 100 / total))
  fi

  # Print summary
  log_header "Test Summary"
  printf "  %-20s %d\n" "Total Tests:" "$total"
  printf "  %-20s %d\n" "Failed:" "$failed"
  printf "  %-20s %d\n" "Passed:" "$((total - failed))"
  if [[ $total -gt 0 ]]; then
    printf "  %-20s %d%%\n" "Success Rate:" "$success_rate"
  fi

  return $failed
}

# Main function
main() {
  log_header "Brezel Test Runner"

  parse_args "$@"
  validate_config

  # Print configuration
  log_info "Configuration:"
  printf "  %-20s %s\n" "Test Type:" "${CONFIG[TEST_TYPE]}"
  printf "  %-20s %s\n" "Filter:" "${CONFIG[TEST_FILTER]:-none}"
  printf "  %-20s %s\n" "Build Dir:" "${CONFIG[BUILD_DIR]}"
  printf "  %-20s %s\n" "Verbose:" "${CONFIG[VERBOSE]}"
  printf "  %-20s %s\n" "Jobs:" "${CONFIG[JOBS]}"
  printf "  %-20s %s\n" "Repeat:" "${CONFIG[REPEAT]}"
  printf "  %-20s %s\n" "XML Output:" "${CONFIG[OUTPUT_XML]}"
  printf "  %-20s %s\n" "Coverage:" "${CONFIG[COVERAGE]}"
  echo

  find_tests

  if [[ ${#UNIT_TESTS[@]} -eq 0 && ${#INTEGRATION_TESTS[@]} -eq 0 && ${#BENCHMARKS[@]} -eq 0 ]]; then
    log_warning "No tests found in ${CONFIG[BUILD_DIR]}"
    log_info "Make sure:"
    log_info "1. You've built the project with tests enabled: ./build.sh"
    log_info "2. Tests are located in the correct directory: ${CONFIG[BUILD_DIR]}/tests"
    log_info "3. Test files are named correctly: *_test, *_integration, *_bench"
    return 0
  fi

  local start_time
  start_time=$(date +%s)

  # Run tests multiple times if requested
  local failed=0
  for ((i = 1; i <= ${CONFIG[REPEAT]}; i++)); do
    if [[ ${CONFIG[REPEAT]} -gt 1 ]]; then
      log_header "Test Run $i/${CONFIG[REPEAT]}"
    fi
    run_tests || ((failed++))
  done

  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))

  log_header "Test Execution Complete"
  printf "${BLUE}⏱️  Total time:${NC} %ds\n" "$duration"

  return $failed
}

# Execute main
main "$@"
