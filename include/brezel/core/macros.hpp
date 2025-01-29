#pragma once

// Version information
#define BREZEL_MAJOR_VERSION 0
#define BREZEL_MINOR_VERSION 1
#define BREZEL_PATCH_VERSION 0

// Platform Detection
#if defined(_WIN32) || defined(_WIN64)
#define BREZEL_PLATFORM_WINDOWS
#if defined(_WIN64)
#define BREZEL_64_BIT
#else
#define BREZEL_32_BIT
#endif
#elif defined(__linux__)
#define BREZEL_PLATFORM_LINUX
#if defined(__x86_64__) || defined(__ppc64__)
#define BREZEL_64_BIT
#else
#define BREZEL_32_BIT
#endif
#elif defined(__APPLE__)
#define BREZEL_PLATFORM_MACOS
#define BREZEL_64_BIT
#endif

// Compiler detection
#if defined(_MSC_VER)
#define BREZEL_COMPILER_MSVC
#elif defined(__clang__)
#define BREZEL_COMPILER_CLANG
#elif defined(__GNUC__)
#define BREZEL_COMPILER_GCC
#endif

// Visibility macros
#if defined(BREZEL_PLATFORM_WINDOWS)
#define BREZEL_EXPORT __declspec(dllexport)
#define BREZEL_IMPORT __declspec(dllimport)
#if defined(BREZEL_BUILD_SHARED)
#define BREZEL_API BREZEL_EXPORT
#else
#define BREZEL_API BREZEL_IMPORT
#endif
#else
#define BREZEL_EXPORT __attribute__((visibility("default")))
#define BREZEL_IMPORT
#if defined(BREZEL_BUILD_SHARED)
#define BREZEL_API BREZEL_EXPORT
#else
#define BREZEL_API
#endif
#endif

// Function inlining
#if defined(BREZEL_COMPILER_MSVC)
#define BREZEL_ALWAYS_INLINE __forceinline
#define BREZEL_NEVER_INLINE __declspec(noinline)
#else
#define BREZEL_ALWAYS_INLINE inline __attribute__((always_inline))
#define BREZEL_NEVER_INLINE __attribute__((noinline))
#endif

// Debugging macros
#if defined(BREZEL_DEBUG)
#define BREZEL_DEBUG_ONLY(x) x
#else
#define BREZEL_DEBUG_ONLY(x)
#endif

// CUDA support
#if defined(BREZEL_WITH_CUDA)
#define BREZEL_CUDA_EXPORT __host__ __device__
#define BREZEL_CUDA_ONLY __device__
#define BREZEL_HOST_ONLY __host__
#else
#define BREZEL_CUDA_EXPORT
#define BREZEL_CUDA_ONLY
#define BREZEL_HOST_ONLY
#endif

// Alignment
#define BREZEL_ALIGN(x) alignas(x)

// Unused variable
#define BREZEL_UNUSED(x) (void)(x)

// Function name macro
#if defined(BREZEL_COMPILER_MSVC)
#define BREZEL_FUNCTION __FUNCSIG__
#else
#define BREZEL_FUNCTION __PRETTY_FUNCTION__
#endif

// Stringify macros
#define BREZEL_STRINGIFY_IMPL(x) #x
#define BREZEL_STRINGIFY(x) BREZEL_STRINGIFY_IMPL(x)

// Concatenation macros
#define BREZEL_CONCAT_IMPL(x, y) x##y
#define BREZEL_CONCAT(x, y) BREZEL_CONCAT_IMPL(x, y)

// Disable copy
#define BREZEL_DISABLE_COPY(Class) \
  Class(const Class &) = delete;   \
  Class &operator=(const Class &) = delete

// Disable move
#define BREZEL_DISABLE_MOVE(Class) \
  Class(Class &&) = delete;        \
  Class &operator=(Class &&) = delete

// Disable copy and move
#define BREZEL_DISABLE_COPY_AND_MOVE(Class) \
  BREZEL_DISABLE_COPY(Class)                \
  BREZEL_DISABLE_MOVE(Class)

// Enable move only
#define BREZEL_MOVE_ONLY(Class)       \
  BREZEL_DISABLE_COPY(Class)          \
  Class(Class &&) noexcept = default; \
  Class &operator=(Class &&) noexcept = default