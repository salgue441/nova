#pragma once

#include <brezel/core/macros.hpp>
#include <stdexcept>
#include <string>
#include <fmt/format.h>

namespace brezel::core
{
  /**
   * @class BREZEL_API
   * @brief Base class for all brezel exceptions
   *
   */
  class BREZEL_API Exception : public std::runtime_error
  {
  public:
    explicit Exception(const std::string &message)
        : std::runtime_error(message), m_message(message) {}

    explicit Exception(const char *message)
        : std::runtime_error(message), m_message(message) {}

    const char *what() const noexcept override { return m_message.c_str(); }

  private:
    std::string m_message;
  };

  /**
   * @brief Exception for runtime errors
   */
  class BREZEL_API RuntimeError : public Exception
  {
  public:
    explicit RuntimeError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for shape mismatches
   */
  class BREZEL_API ShapeError : public Exception
  {
  public:
    explicit ShapeError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for device-related errors
   */
  class BREZEL_API DeviceError : public Exception
  {
  public:
    explicit DeviceError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for memory-related errors
   */
  class BREZEL_API MemoryError : public Exception
  {
  public:
    explicit MemoryError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for CUDA-related errors
   */
  class BREZEL_API CUDAError : public Exception
  {
  public:
    explicit CUDAError(const std::string &message) : Exception(message) {}
  };

  // Error checking macros
#define BREZEL_THROW(message)                                    \
  throw ::brezel::core::RuntimeError(::fmt::format("[{}:{}] {}", \
                                                   __FILE__, __LINE__, message))

#define BREZEL_CHECK(condition, message) \
  do                                     \
  {                                      \
    if (!(condition))                    \
    {                                    \
      BREZEL_THROW(message);             \
    }                                    \
  } while (0)

#define BREZEL_CHECK_SHAPE(condition, message)                                      \
  do                                                                                \
  {                                                                                 \
    if (!(condition))                                                               \
    {                                                                               \
      throw ::brezel::core::ShapeError(::fmt::format("[{}:{}] {}",                  \
                                                     __FILE__, __LINE__, message)); \
    }                                                                               \
  } while (0)

#define BREZEL_CHECK_DEVICE(condition, message)                                      \
  do                                                                                 \
  {                                                                                  \
    if (!(condition))                                                                \
    {                                                                                \
      throw ::brezel::core::DeviceError(::fmt::format("[{}:{}] {}",                  \
                                                      __FILE__, __LINE__, message)); \
    }                                                                                \
  } while (0)

#define BREZEL_CHECK_MEMORY(condition, message)                                      \
  do                                                                                 \
  {                                                                                  \
    if (!(condition))                                                                \
    {                                                                                \
      throw ::brezel::core::MemoryError(::fmt::format("[{}:{}] {}",                  \
                                                      __FILE__, __LINE__, message)); \
    }                                                                                \
  } while (0)

// CUDA error checking
#ifdef BREZEL_WITH_CUDA
#define BREZEL_CUDA_CHECK(call)                                                                      \
  do                                                                                                 \
  {                                                                                                  \
    cudaError_t error = call;                                                                        \
    if (error != cudaSuccess)                                                                        \
    {                                                                                                \
      throw ::brezel::core::CUDAError(::fmt::format("[{}:{}] CUDA error: {}",                        \
                                                    __FILE__, __LINE__, cudaGetErrorString(error))); \
    }                                                                                                \
  } while (0)
#endif
} // namespace brezel