#pragma once

#include "nova/core/macros.hpp"
#include <stdexcept>
#include <string>
#include <fmt/format.h>

namespace nova::core
{
  /**
   * @class NOVA_API
   * @brief Base class for all NOVA exceptions
   *
   */
  class NOVA_API Exception : public std::runtime_error
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
  class NOVA_API RuntimeError : public Exception
  {
  public:
    explicit RuntimeError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for shape mismatches
   */
  class NOVA_API ShapeError : public Exception
  {
  public:
    explicit ShapeError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for device-related errors
   */
  class NOVA_API DeviceError : public Exception
  {
  public:
    explicit DeviceError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for memory-related errors
   */
  class NOVA_API MemoryError : public Exception
  {
  public:
    explicit MemoryError(const std::string &message) : Exception(message) {}
  };

  /**
   * @brief Exception for CUDA-related errors
   */
  class NOVA_API CUDAError : public Exception
  {
  public:
    explicit CUDAError(const std::string &message) : Exception(message) {}
  };

  // Error checking macros
#define NOVA_THROW(message)                              \
  throw ::nova::RuntimeError(::fmt::format("[{}:{}] {}", \
                                           __FILE__, __LINE__, message))

#define NOVA_CHECK(condition, message) \
  do                                   \
  {                                    \
    if (!(condition))                  \
    {                                  \
      NOVA_THROW(message);             \
    }                                  \
  } while (0)

#define NOVA_CHECK_SHAPE(condition, message)                                \
  do                                                                        \
  {                                                                         \
    if (!(condition))                                                       \
    {                                                                       \
      throw ::nova::ShapeError(::fmt::format("[{}:{}] {}",                  \
                                             __FILE__, __LINE__, message)); \
    }                                                                       \
  } while (0)

#define NOVA_CHECK_DEVICE(condition, message)                                \
  do                                                                         \
  {                                                                          \
    if (!(condition))                                                        \
    {                                                                        \
      throw ::nova::DeviceError(::fmt::format("[{}:{}] {}",                  \
                                              __FILE__, __LINE__, message)); \
    }                                                                        \
  } while (0)

#define NOVA_CHECK_MEMORY(condition, message)                                \
  do                                                                         \
  {                                                                          \
    if (!(condition))                                                        \
    {                                                                        \
      throw ::nova::MemoryError(::fmt::format("[{}:{}] {}",                  \
                                              __FILE__, __LINE__, message)); \
    }                                                                        \
  } while (0)

// CUDA error checking
#ifdef NOVA_WITH_CUDA
#define NOVA_CUDA_CHECK(call)                                                                \
  do                                                                                         \
  {                                                                                          \
    cudaError_t error = call;                                                                \
    if (error != cudaSuccess)                                                                \
    {                                                                                        \
      throw ::nova::CUDAError(::fmt::format("[{}:{}] CUDA error: {}",                        \
                                            __FILE__, __LINE__, cudaGetErrorString(error))); \
    }                                                                                        \
  } while (0)
#endif
} // namespace nova