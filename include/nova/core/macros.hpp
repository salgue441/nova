#pragma once

// Version information
#define NOVA_MAJOR_VERSION 0
#define NOVA_MINOR_VERSION 1
#define NOVA_PATCH_VERSION 0

// Platform Detection
#if defined(_WIN32) || defined(_WIN64)
#define NOVA_PLATFORM_WINDOWS
#if defined(_WIN64)
#define NOVA_64_BIT
#else
#define NOVA_32_BIT
#endif
#elif defined(__linux__)
#define NOVA_PLATFORM_LINUX
#if defined(__x86_64__) || defined(__ppc64__)
#define NOVA_64_BIT
#else
#define NOVA_32_BIT
#endif
#elif defined(__APPLE__)
#define NOVA_PLATFORM_MACOS
#define NOVA_64_BIT
#endif

// Compiler detection
#if defined(_MSC_VER)
#define NOVA_COMPILER_MSVC
#elif defined(__clang__)
#define NOVA_COMPILER_CLANG
#elif defined(__GNUC__)
#define NOVA_COMPILER_GCC
#endif

// Visibility macros
#if defined(NOVA_PLATFORM_WINDOWS)
#define NOVA_EXPORT __declspec(dllexport)
#define NOVA_IMPORT __declspec(dllimport)
#if defined(NOVA_BUILD_SHARED)
#define NOVA_API NOVA_EXPORT
#else
#define NOVA_API NOVA_IMPORT
#endif
#else
#define NOVA_EXPORT __attribute__((visibility("default")))
#define NOVA_IMPORT
#if defined(NOVA_BUILD_SHARED)
#define NOVA_API NOVA_EXPORT
#else
#define NOVA_API
#endif
#endif

// Function inlining
#if defined(NOVA_COMPILER_MSVC)
#define NOVA_ALWAYS_INLINE __forceinline
#define NOVA_NEVER_INLINE __declspec(noinline)
#else
#define NOVA_ALWAYS_INLINE inline __attribute__((always_inline))
#define NOVA_NEVER_INLINE __attribute__((noinline))
#endif

// Debugging macros
#if defined(NOVA_DEBUG)
#define NOVA_DEBUG_ONLY(x) x
#else
#define NOVA_DEBUG_ONLY(x)
#endif

// CUDA support
#if defined(NOVA_WITH_CUDA)
#define NOVA_CUDA_EXPORT __host__ __device__
#define NOVA_CUDA_ONLY __device__
#define NOVA_HOST_ONLY __host__
#else
#define NOVA_CUDA_EXPORT
#define NOVA_CUDA_ONLY
#define NOVA_HOST_ONLY
#endif

// Alignment
#define NOVA_ALIGN(x) alignas(x)

// Unused variable
#define NOVA_UNUSED(x) (void)(x)

// Function name macro
#if defined(NOVA_COMPILER_MSVC)
#define NOVA_FUNCTION __FUNCSIG__
#else
#define NOVA_FUNCTION __PRETTY_FUNCTION__
#endif

// Stringify macros
#define NOVA_STRINGIFY_IMPL(x) #x
#define NOVA_STRINGIFY(x) NOVA_STRINGIFY_IMPL(x)

// Concatenation macros
#define NOVA_CONCAT_IMPL(x, y) x##y
#define NOVA_CONCAT(x, y) NOVA_CONCAT_IMPL(x, y)

// Disable copy
#define NOVA_DISABLE_COPY(Class) \
  Class(const Class &) = delete; \
  Class &operator=(const Class &) = delete

// Disable move
#define NOVA_DISABLE_MOVE(Class) \
  Class(Class &&) = delete;      \
  Class &operator=(Class &&) = delete

// Disable copy and move
#define NOVA_DISABLE_COPY_AND_MOVE(Class) \
  NOVA_DISABLE_COPY(Class)                \
  NOVA_DISABLE_MOVE(Class)

// Enable move only
#define NOVA_MOVE_ONLY(Class)         \
  NOVA_DISABLE_COPY(Class)            \
  Class(Class &&) noexcept = default; \
  Class &operator=(Class &&) noexcept = default