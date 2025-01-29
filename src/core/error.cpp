#include <brezel/core/error.hpp>

namespace brezel::core
{
  namespace detail
  {
    /**
     * @brief Get the source location object
     * @note Helper function
     *
     * @param file File name where the error occurred
     * @param line Line number where the error occurred
     * @return std::string Source location in the format [file:line]
     * @version 1.0.0
     */
    std::string get_source_location(const char *file, int line)
    {
      return fmt::format("[{}:{}]", file, line);
    }

    /**
     * @brief Formats error messages
     * @note Helper function
     *
     * @param file File name where the error occurred
     * @param line Line number where the error occurred
     * @param message Error message
     * @return std::string Formatted error message
     * @version 1.0.0
     */
    std::string format_error_message(const char *file, int line,
                                     const std::string &message)
    {
      return fmt::format("[{}] {}", get_source_location(file, line), message);
    }
  } // namespace detail

#ifdef BREZEL_WITH_CUDA
  namespace cuda
  {
    /**
     * @brief Get the CUDA error string
     * @note Helper function
     *
     * @param error CUDA error code
     * @return std::string CUDA error string
     * @version 1.0.0
     */
    std::string get_cuda_error_string(cudaError_t error)
    {
      return cudaGetErrorString(error);
    }

    /**
     * @brief Check for CUDA errors
     * @note Helper function
     *
     * @param file File name where the error occurred
     * @param line Line number where the error occurred
     * @version 1.0.0
     */
    void check_cuda_error(const char *file, int line, cudaError_t error)
    {
      if (error != cudaSuccess)
      {
        throw CUDAError(detail::format_error_message(file, line,
                                                     fmt::format("CUDA error: {}", get_cuda_error_string(error))));
      }
    }

    /**
     * @brief Check for CUDA errors
     * @note Helper function
     *
     * @param file File name where the error occurred
     * @param line Line number where the error occurred
     * @param error CUDA error code
     * @version 1.0.0
     */
    void check_cuda_error(const char *file, int line, cudaError_t error)
    {
      if (error != cudaSuccess)
      {
        throw CUDAError(detail::format_error_message(file, line,
                                                     fmt::format("CUDA error: {}", get_cuda_error_string(error))));
      }
    }
  } // namespace cuda
#endif
}