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

# Configuration variables
declare -A CONFIG=(
  [CHECK_ONLY]="OFF"
  [VERBOSE]="OFF"
  [JOBS]="$(nproc)"
  [FORMAT_STYLE]="file" # Use .clang-format file
)

# Array for paths (declare separately since we can't nest arrays)
declare -a PATHS=()

# File extensions to process
declare -a CPP_EXTENSIONS=(
  "cpp" "hpp"
  "cc" "h"
  "cxx" "hxx"
  "cu" "cuh"
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
${BOLD}Brezel Tensor Framework Code Formatter${NC}

${BOLD}Usage:${NC}
    $(basename "$0") [options] [paths...]

${BOLD}Options:${NC}
    ${CYAN}-c, --check${NC}         Check formatting without making changes
    ${CYAN}-v, --verbose${NC}       Show verbose output
    ${CYAN}-j, --jobs${NC} N        Number of parallel jobs [default: $(nproc)]
    ${CYAN}-s, --style${NC} STYLE   Formatting style (default: file)
    ${CYAN}-h, --help${NC}          Show this help message

${BOLD}Examples:${NC}
    # Format all source files
    ./$(basename "$0")

    # Check formatting in specific directory
    ./$(basename "$0") -c src/

    # Format specific files with custom style
    ./$(basename "$0") -s=Google file1.cpp file2.hpp

${BOLD}Note:${NC}
    By default, uses .clang-format file in project root.
    If no paths are specified, formats all source files.
\n"
  exit 0
}

# Parse command line arguments
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
    -c | --check)
      CONFIG[CHECK_ONLY]="ON"
      shift
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
    -s | --style)
      [[ -z "${2:-}" ]] && log_error "Style name required" && exit 1
      CONFIG[FORMAT_STYLE]="$2"
      shift 2
      ;;
    -h | --help)
      show_help
      ;;
    -*)
      log_error "Unknown option: $1"
      show_help
      ;;
    *)
      PATHS+=("$1")
      shift
      ;;
    esac
  done
}

# Check dependencies
check_dependencies() {
  log_step "Checking dependencies..."

  if ! command -v clang-format &>/dev/null; then
    log_error "clang-format not found. Please install clang-format."
    if [[ -f /etc/debian_version ]]; then
      log_info "Install with: sudo apt-get install clang-format"
    elif [[ -f /etc/fedora-release ]]; then
      log_info "Install with: sudo dnf install clang-format"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
      log_info "Install with: brew install clang-format"
    fi
    exit 1
  fi

  # Check clang-format version
  local version
  version=$(clang-format --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0")
  log_info "Found clang-format version $version"
}

# Find source files
find_source_files() {
  local find_paths=()

  # Use provided paths or default to current directory
  if [ ${#PATHS[@]} -eq 0 ]; then
    find_paths=(".")
  else
    find_paths=("${PATHS[@]}")
  fi

  local find_command="find"
  local extension_pattern=""

  # Build extension pattern
  for ext in "${CPP_EXTENSIONS[@]}"; do
    if [ -z "$extension_pattern" ]; then
      extension_pattern="-name \"*.$ext\""
    else
      extension_pattern="$extension_pattern -o -name \"*.$ext\""
    fi
  done

  # Build and execute find command
  local cmd="$find_command ${find_paths[*]} \\( $extension_pattern \\) \
        -type f \
        -not -path \"*/build/*\" \
        -not -path \"*/\\.*\" \
        -not -path \"*/third_party/*\" \
        -not -path \"*/external/*\""

  if [[ ${CONFIG[VERBOSE]} == "ON" ]]; then
    log_info "Executing: $cmd"
  fi

  eval "$cmd"
}

# Format a single file
format_file() {
  local file="$1"
  local status=0
  local output

  if [[ ${CONFIG[CHECK_ONLY]} == "ON" ]]; then
    if ! output=$(clang-format --style="${CONFIG[FORMAT_STYLE]}" "$file" | diff -u "$file" - 2>&1); then
      printf "${RED}%s needs formatting${NC}\n" "$file"
      if [[ ${CONFIG[VERBOSE]} == "ON" ]]; then
        echo "$output"
      fi
      return 1
    fi
  else
    if ! output=$(clang-format -i --style="${CONFIG[FORMAT_STYLE]}" "$file" 2>&1); then
      log_error "Failed to format $file:"
      echo "$output"
      return 1
    elif [[ ${CONFIG[VERBOSE]} == "ON" ]]; then
      log_success "Formatted $file"
    fi
  fi
  return 0
}

# Format all files
format_files() {
  log_step "Finding source files..."
  local files=()
  mapfile -t files < <(find_source_files)

  if [[ ${#files[@]} -eq 0 ]]; then
    log_warning "No source files found"
    return 0
  fi

  log_info "Found ${#files[@]} source files"

  local failed=0
  local formatted=0

  log_step "Processing files..."
  for file in "${files[@]}"; do
    if format_file "$file"; then
      ((formatted++))
    else
      ((failed++))
    fi
  done

  if [[ $failed -eq 0 ]]; then
    if [[ ${CONFIG[CHECK_ONLY]} == "ON" ]]; then
      log_success "All $formatted files are properly formatted"
    else
      log_success "Successfully formatted $formatted files"
    fi
    return 0
  else
    log_error "$failed files need formatting"
    return 1
  fi
}

# Create/check .clang-format file
setup_format_config() {
  if [[ ${CONFIG[FORMAT_STYLE]} == "file" ]] && [[ ! -f .clang-format ]]; then
    log_warning "No .clang-format file found, creating default config..."
    cat >.clang-format <<'EOF'
---
Language: Cpp
BasedOnStyle: Google
AccessModifierOffset: -4
AlignAfterOpenBracket: Align
AlignConsecutiveAssignments: false
AlignConsecutiveDeclarations: false
AlignEscapedNewlines: Left
AlignOperands: true
AlignTrailingComments: true
AllowAllParametersOfDeclarationOnNextLine: true
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
AlwaysBreakAfterReturnType: None
AlwaysBreakBeforeMultilineStrings: true
AlwaysBreakTemplateDeclarations: Yes
BinPackArguments: true
BinPackParameters: true
BreakBeforeBinaryOperators: None
BreakBeforeBraces: Attach
BreakBeforeInheritanceComma: false
BreakBeforeTernaryOperators: true
BreakConstructorInitializersBeforeComma: false
BreakConstructorInitializers: BeforeColon
BreakStringLiterals: true
ColumnLimit: 80
CompactNamespaces: false
ConstructorInitializerAllOnOneLineOrOnePerLine: true
ConstructorInitializerIndentWidth: 4
ContinuationIndentWidth: 4
Cpp11BracedListStyle: true
DerivePointerAlignment: false
ExperimentalAutoDetectBinPacking: false
FixNamespaceComments: true
IncludeBlocks: Regroup
IndentCaseLabels: true
IndentPPDirectives: None
IndentWidth: 4
IndentWrappedFunctionNames: false
KeepEmptyLinesAtTheStartOfBlocks: false
MaxEmptyLinesToKeep: 1
NamespaceIndentation: None
PointerAlignment: Left
ReflowComments: true
SortIncludes: true
SortUsingDeclarations: true
SpaceAfterCStyleCast: false
SpaceAfterTemplateKeyword: true
SpaceBeforeAssignmentOperators: true
SpaceBeforeParens: ControlStatements
SpaceInEmptyParentheses: false
SpacesBeforeTrailingComments: 2
SpacesInAngles: false
SpacesInContainerLiterals: false
SpacesInCStyleCastParentheses: false
SpacesInParentheses: false
SpacesInSquareBrackets: false
Standard: c++20
TabWidth: 4
UseTab: Never
EOF
    log_success "Created .clang-format file"
  fi
}

# Main function
main() {
  log_header "Brezel Code Formatter"

  parse_args "$@"
  check_dependencies
  setup_format_config

  # Print configuration
  log_info "Configuration:"
  printf "  %-20s %s\n" "Check only:" "${CONFIG[CHECK_ONLY]}"
  printf "  %-20s %s\n" "Style:" "${CONFIG[FORMAT_STYLE]}"
  printf "  %-20s %s\n" "Verbose:" "${CONFIG[VERBOSE]}"
  if [[ ${#PATHS[@]} -gt 0 ]]; then
    printf "  %-20s %s\n" "Paths:" "${PATHS[*]}"
  fi
  echo

  # Format files
  if format_files; then
    exit 0
  else
    exit 1
  fi
}

# Execute main
main "$@"
