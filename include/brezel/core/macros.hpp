#pragma once

#include <concepts>
#include <version>

namespace brezel::detail {
// Forward declarations for assert handling
[[noreturn]] void assert_failure(const char* condition, const char* message,
                                 const char* file, unsigned line);
}  // namespace brezel::detail

// Version information
#define BREZEL_VERSION_MAJOR 0
#define BREZEL_VERSION_MINOR 1
#define BREZEL_VERSION_PATCH 0
#define BREZEL_VERSION_STRING "0.1.0"

// Function attributes
#define BREZEL_NODISCARD [[nodiscard]]
#define BREZEL_MAYBE_UNUSED [[maybe_unused]]
#define BREZEL_DEPRECATED(msg) [[deprecated(msg)]]
#define BREZEL_FALLTHROUGH [[fallthrough]]
#define BREZEL_NORETURN [[noreturn]]
#define BREZEL_CARRIES_DEPENDENCY [[carries_dependency]]
#define BREZEL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#define BREZEL_LIKELY [[likely]]
#define BREZEL_UNLIKELY [[unlikely]]

// Platform and architecture detection
#if defined(_WIN32) || defined(_WIN64)
#define BREZEL_PLATFORM_WINDOWS
#define BREZEL_PLATFORM_NAME "Windows"
#if defined(_WIN64) || defined(__x86_64__)
#define BREZEL_ARCH_X64
#else
#define BREZEL_ARCH_X86
#endif
#elif defined(__linux__)
#define BREZEL_PLATFORM_LINUX
#define BREZEL_PLATFORM_NAME "Linux"
#if defined(__x86_64__)
#define BREZEL_ARCH_X64
#else
#define BREZEL_ARCH_X86
#endif
#elif defined(__APPLE__)
#define BREZEL_PLATFORM_MACOS
#define BREZEL_PLATFORM_NAME "macOS"
#define BREZEL_ARCH_X64
#else
#error "Unsupported platform"
#endif

// Compiler detection and features
#if defined(_MSC_VER)
#define BREZEL_COMPILER_MSVC
#define BREZEL_COMPILER_NAME "MSVC"
#define BREZEL_FORCE_INLINE [[msvc::forceinline]]
#define BREZEL_NO_INLINE __declspec(noinline)
#define BREZEL_RESTRICT __restrict
#elif defined(__clang__)
#define BREZEL_COMPILER_CLANG
#define BREZEL_COMPILER_NAME "Clang"
#define BREZEL_FORCE_INLINE [[clang::always_inline]] inline
#define BREZEL_NO_INLINE [[clang::noinline]]
#define BREZEL_RESTRICT __restrict
#elif defined(__GNUC__)
#define BREZEL_COMPILER_GCC
#define BREZEL_COMPILER_NAME "GCC"
#define BREZEL_FORCE_INLINE [[gnu::always_inline]] inline
#define BREZEL_NO_INLINE [[gnu::noinline]]
#define BREZEL_RESTRICT __restrict
#endif

// Build configuration
#if defined(BREZEL_DEBUG) || defined(_DEBUG)
#define BREZEL_CONFIG_DEBUG
#define BREZEL_CONFIG_NAME "Debug"
#else
#define BREZEL_CONFIG_RELEASE
#define BREZEL_CONFIG_NAME "Release"
#endif

// Memory and cache optimizations
#define BREZEL_CACHE_LINE_SIZE 64
#define BREZEL_ALIGN_CACHE alignas(BREZEL_CACHE_LINE_SIZE)
#define BREZEL_ALIGN(x) alignas(x)

// Memory prefetch hints
#if defined(BREZEL_COMPILER_GCC) || defined(BREZEL_COMPILER_CLANG)
#define BREZEL_PREFETCH_READ(ptr) __builtin_prefetch((ptr), 0, 3)
#define BREZEL_PREFETCH_WRITE(ptr) __builtin_prefetch((ptr), 1, 3)
#define BREZEL_ASSUME_ALIGNED(ptr, alignment) \
    __builtin_assume_aligned((ptr), (alignment))
#else
#define BREZEL_PREFETCH_READ(ptr) ((void)(ptr))
#define BREZEL_PREFETCH_WRITE(ptr) ((void)(ptr))
#define BREZEL_ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

// Branch prediction hints
#if defined(BREZEL_COMPILER_GCC) || defined(BREZEL_COMPILER_CLANG)
#define BREZEL_EXPECT(expr, value) __builtin_expect(!!(expr), (value))
#define BREZEL_PREDICT_TRUE(expr) BREZEL_EXPECT(!!(expr), 1)
#define BREZEL_PREDICT_FALSE(expr) BREZEL_EXPECT(!!(expr), 0)
#else
#define BREZEL_EXPECT(expr, value) (expr)
#define BREZEL_PREDICT_TRUE(expr) (expr)
#define BREZEL_PREDICT_FALSE(expr) (expr)
#endif

// Modern C++20 feature detection
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
#define BREZEL_HAS_CONCEPTS
#endif

#if defined(__cpp_constexpr) && __cpp_constexpr >= 201907L
#define BREZEL_HAS_CONSTEXPR_DYNAMIC
#endif

#if defined(__cpp_modules) && __cpp_modules >= 201907L
#define BREZEL_HAS_MODULES
#endif

#if defined(__cpp_consteval) && __cpp_consteval >= 201811L
#define BREZEL_HAS_CONSTEVAL
#define BREZEL_CONSTEVAL consteval
#else
#define BREZEL_CONSTEVAL constexpr
#endif

// SIMD support detection
#if defined(__AVX512F__)
#define BREZEL_SIMD_AVX512
#elif defined(__AVX2__)
#define BREZEL_SIMD_AVX2
#elif defined(__AVX__)
#define BREZEL_SIMD_AVX
#elif defined(__SSE4_2__)
#define BREZEL_SIMD_SSE4_2
#endif

// Class utilities with modern attributes
#define BREZEL_MOVEABLE(Class)                          \
    BREZEL_NODISCARD Class(Class&&) noexcept = default; \
    Class& operator=(Class&&) noexcept = default;       \
    Class(const Class&) = delete;                       \
    Class& operator=(const Class&) = delete

#define BREZEL_COPYABLE(Class)                          \
    BREZEL_NODISCARD Class(const Class&) = default;     \
    Class& operator=(const Class&) = default;           \
    BREZEL_NODISCARD Class(Class&&) noexcept = default; \
    Class& operator=(Class&&) noexcept = default

#define BREZEL_IMMOVABLE(Class) \
    Class(Class&&) = delete;    \
    Class& operator=(Class&&) = delete

#define BREZEL_UNCOPYABLE(Class)  \
    Class(const Class&) = delete; \
    Class& operator=(const Class&) = delete

// API visibility with platform-specific attributes
#if defined(BREZEL_PLATFORM_WINDOWS)
#define BREZEL_API_EXPORT __declspec(dllexport)
#define BREZEL_API_IMPORT __declspec(dllimport)
#else
#define BREZEL_API_EXPORT [[gnu::visibility("default")]]
#define BREZEL_API_IMPORT [[gnu::visibility("default")]]
#endif

#if defined(BREZEL_BUILD_SHARED)
#define BREZEL_API BREZEL_API_EXPORT
#else
#define BREZEL_API BREZEL_API_IMPORT
#endif

// Function optimization hints
#define BREZEL_HOT [[gnu::hot]]
#define BREZEL_COLD [[gnu::cold]]
#define BREZEL_PURE [[gnu::pure]]
#define BREZEL_CONST [[gnu::const]]

// Memory alignment and vectorization hints
#define BREZEL_VECTORIZE [[gnu::vectorize]]
#define BREZEL_LOOP_VECTORIZE [[clang::loop_vectorize]]
#define BREZEL_ALIGNED(x) [[gnu::aligned(x)]]

// Optimization control
#define BREZEL_OPTIMIZE(level) [[gnu::optimize(level)]]
#define BREZEL_NO_OPTIMIZE [[gnu::optimize("O0")]]

// Debug configuration
#if defined(BREZEL_ENABLE_DEBUG) || defined(_DEBUG)
#define BREZEL_DEBUG_MODE
#endif

// Debug utilities with enhanced error reporting - only enabled when explicitly
// requested
#ifdef BREZEL_DEBUG_MODE
#define BREZEL_DEBUG_ONLY(x) x
#define BREZEL_ASSERT(condition, message)                                   \
    do {                                                                    \
        if (BREZEL_PREDICT_FALSE(!(condition))) {                           \
            ::brezel::detail::assert_failure(#condition, message, __FILE__, \
                                             __LINE__);                     \
        }                                                                   \
    } while (0)

#define BREZEL_VERIFY(condition, message) BREZEL_ASSERT(condition, message)
#else
#define BREZEL_DEBUG_ONLY(x)
#define BREZEL_ASSERT(condition, message) ((void)0)
#define BREZEL_VERIFY(condition, message)         \
    do {                                          \
        if (BREZEL_PREDICT_FALSE(!(condition))) { \
            std::terminate();                     \
        }                                         \
    } while (0)
#endif