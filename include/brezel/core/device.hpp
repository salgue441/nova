#pragma once

#include <brezel/core/error.hpp>
#include <brezel/core/macros.hpp>
#include <cstring>
#include <memory>
#include <optional>
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/core/span.hpp>
#include <tbb/concurrent_queue.h>
#include <fmt/format.h>

#ifdef BREZEL_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace brezel::core
{
  /**
   * @class DeviceMemoryPool
   * @brief Thread-safe memory pool for device allocators
   *
   * @tparam Allocator Allocator type for memory allocation
   * @version 2.0.0
   */
  template <typename Allocator>
  class DeviceMemoryPool
  {
  public:
    /**
     * @brief Allocates memory from the pool or the device
     *
     * @param size Size of memory to allocate
     * @return std::pair<void *, bool> Pointer to allocated memory and
     *         flag indicating if memory was allocated from the pool
     * @version 1.0.0
     */
    std::pair<void *, bool> allocate(size_t size)
    {
      MemoryBlock block{nullptr, 0};
      if (m_memory_pool.try_pop(block) && block.size >= size)
        return {block.ptr, true};

      return {Allocator::allocate(size), false};
    }

    /**
     * @brief Deallocates memory to the pool or the device
     *
     * @param ptr Pointer to memory to deallocate
     * @param size Size of memory to deallocate
     * @param from_pool Flag indicating if memory was allocated from the pool
     * @version 1.0.0
     */
    void deallocate(void *ptr, size_t size, bool from_pool = false)
    {
      if (from_pool)
      {
        size_t current_size = 0;
        m_memory_pool.size(current_size);

        if (current_size < MAX_POOL_SIZE)
        {
          m_memory_pool.push({ptr, size});
          return;
        }
      }

      Allocator::free(ptr);
    }

    /**
     * @brief Destructor
     * @version 1.0.0
     */
    ~DeviceMemoryPool()
    {
      MemoryBlock block;
      while (m_memory_pool.try_pop(block))
        Allocator::free(block.ptr);
    }

  private:
    struct MemoryBlock
    {
      void *ptr;
      size_t size;
    };

    tbb::concurrent_queue<MemoryBlock> m_memory_pool;
    static constexpr size_t MAX_POOL_SIZE = 1024;
  };

  /**
   * @class CPUAllocator
   * @brief Aligned memory allocator for CPU
   *
   * @version 1.0.0
   */
  class BREZEL_API CPUAllocator
  {
    static constexpr size_t ALIGNMENT = 64;
    using AlignedAllocator = boost::alignment::aligned_allocator<uint8_t, ALIGNMENT>;

  public:
    /**
     * @brief Allocates memory on the CPU
     *
     * @param size Size of memory to allocate
     * @return void * Pointer to allocated memory
     * @version 1.0.0
     */
    static void *allocate(size_t size)
    {
      try
      {
        return AlignedAllocator{}.allocate(size);
      }
      catch (const std::bad_alloc &)
      {
        BREZEL_THROW(
            fmt::format("Failed to allocate {} bytes of memory",
                        size));
      }
    }

    /**
     * @brief Deallocates memory on the CPU
     *
     * @param ptr Pointer to memory to deallocate
     * @version 1.0.0
     */
    static void free(void *ptr) noexcept
    {
      if (ptr)
        AlignedAllocator{}.deallocate(static_cast<uint8_t *>(ptr), 0);
    }
  };

  /**
   * @class CPUDevice
   * @brief CPU Device implementation with memory pool
   *
   * @version 2.0.0
   */
  class BREZEL_API CPUDevice
  {
  public:
    enum class MemcpyKind
    {
      HostToHost,
      HostToDevice,
      DeviceToHost,
      DeviceToDevice
    };

    /**
     * @brief Allocates memory on the CPU
     *
     * @param size Size of memory to allocate
     * @version 1.0.0
     */
    static void *allocate(size_t size)
    {
      auto [ptr, from_pool] = get_pool().allocate(size);
      return ptr;
    }

    /**
     * @brief Frees memory on the CPU
     *
     * @param ptr Pointer to memory to deallocate
     * @version 1.0.0
     */
    static void free(void *ptr) noexcept
    {
      if (ptr)
        get_pool().deallocate(ptr, 0);
    }

    /**
     * @brief Memory copy operation on the CPU
     *
     * @param ptr Pointer to destination memory
     * @param value Value to copy
     * @param size Size of memory to copy
     * @version 1.0.0
     */
    static void memset(void *ptr, int value, size_t size)
    {
      std::memset(ptr, value, size);
    }

    /**
     * @brief Memory copy operation on the CPU
     *
     * @tparam T Data type of memory elements
     * @param dst Destination memory span
     * @param src Source memory span
     * @param kind Kind of memory copy operation
     * @version 1.0.0
     */
    template <typename T>
    static void memcpy(boost::span<T> dst, boost::span<const T> src,
                       MemcpyKind kind)
    {
      BREZEL_CHECK(dst.size() == src.size(),
                   "Span sizes must match for memcpy");

      std::memcpy(dst.data(), src.data(), dst.size_bytes());
    }

    /**
     * @brief Memory copy operation on the CPU
     *
     * @param dst Destination memory pointer
     * @param src Source memory pointer
     * @param size Size of memory to copy
     * @param kind Kind of memory copy operation
     * @version 1.0.0
     */
    static void memcpy(void *dst, const void *src, size_t size,
                       MemcpyKind kind)
    {
      std::memcpy(dst, src, size);
    }

    // SIMD-optimized operations
    /**
     * @brief Vectorized memory copy operation on the CPU
     *
     * @tparam T Data type of memory elements
     * @param dst Destination memory span
     * @param src Source memory span
     * @version 1.0.0
     */
    template <typename T>
    static void vectorized_copy(boost::span<T> dst, boost::span<const T> src)
    {
      BREZEL_CHECK(dst.size() == src.size(),
                   "Span sizes must match for vectorized_copy");

#if defined(__AVX2__)
      const size_t vec_size = 32 / sizeof(T);
      size_t vec_length = (dst.size() / vec_size) * vec_size;

      for (size_t i = 0; i < vec_length; i += vec_size)
      {
        __m256i vec = _mm256_load_si256((__m256i *)&src[i]);
        _mm256_store_si256((__m256i *)&dst[i], vec);
      }

      // Remaining elements
      for (size_t i = vec_length; i < dst.size(); ++i)
        dst[i] = src[i];
#else
      std::memcpy(dst.data(), src.data(), dst.size_bytes());
#endif
    }

  private:
    /**
     * @brief Get the pool object
     *
     * @return DeviceMemoryPool<CPUAllocator> & Reference to the memory pool
     * @version 1.0.0
     */
    static DeviceMemoryPool<CPUAllocator> &get_pool()
    {
      static DeviceMemoryPool<CPUAllocator> pool;
      return pool;
    }
  };

#ifdef BREZEL_WITH_CUDA
  /**
   * @class CUDAAllocator
   * @brief Cuda memory allocator with advanced memory management
   *
   * @version 2.0.0
   */
  class BREZEL_API CUDAAllocator
  {
  public:
    /**
     * @brief Allocates memory on CUDA device with optional memory pool
     *
     * @param size Size in bytes to allocated
     * @return Pointer to allocated memory
     * @version 1.0.0
     */
    static void *allocate(size_t size)
    {
      void *ptr = nullptr;

      // Try using CUDA memory pool first (Cuda 11.2 +)
#if CUDART_VERSION >= 11020
      if (auto status = cudaMallocAsync(&ptr, size, getCurrentStream());
          status == cudaSuccess)
        return ptr;
#endif

      // Fall back to regular cudaMalloc
      auto status = cudaMalloc(&ptr, size);
      if (status != cudaSuccess)
        BREZEL_THROW(fmt::format(
            "CUDA allocation failed: {} (size: {} bytes)",
            cudaGetErrorString(status), size));

      return ptr;
    }

    /**
     * @brief Frees CUDA device memory
     *
     * @param ptr Pointer to memory to free
     * @version 1.0.0
     */
    static void free(void *ptr) noexcept
    {
      if (!ptr)
        return;

#if CUDART_VERSION >= 11020
      if (cudaFreeAsync(ptr, getCurrentStream()) == cudaSuccess)
        return;
#endif

      cudaFree(ptr);
    }

  private:
    /**
     * @brief Gets the current CUDA stream
     *
     * @return cudaStream_t Current CUDA stream
     * @version 1.0.0
     */
    static cudaStream_t getCurrentStream()
    {
      cudaStream_t stream;
      cudaStreamGetCurrent(&stream);

      return stream;
    }
  }

  /**
   * @class CUDADevice
   * @brief CUDA device implementation
   *
   * @version 2.0.0
   */
  class BREZEL_API CUDADevice
  {
  public:
    enum class MemcpyKind
    {
      HostToHost = cudaMemcpyHostToHost,
      HostToDevice = cudaMemcpyHostToDevice,
      DeviceToHost = cudaMemcpyDeviceToHost,
      DeviceToDevice = cudaMemcpyDeviceToDevice
    };

  private:
    /**
     * @brief Get the pool object
     *
     * @return DeviceMemoryPool<CUDAAllocator> & Reference to the memory pool
     * @version 1.0.0
     */
    static DeviceMemoryPool<CUDAAllocator> &get_pool()
    {
      static DeviceMemoryPool<CUDAAllocator> pool;
      return pool;
    }

    /**
     * @struct StreamPool
     * @brief Pool of CUDA streams for asynchronous operations
     *
     * @version 1.0.0
     */
    struct StreamPool
    {
      static constexpr size_t POOL_SIZE = 8;
      boost::container::small_vecor<cudaStream_t, POOL_SIZE> streams;
      size_t current_idx = 0;

      /**
       * @brief Construct a new Stream Pool object
       *
       * @version 1.0.0
       */
      StreamPool()
      {
        streams.reserve(POOL_SIZE);
        for (size_t i = 0; i < POOL_SIZE; ++i)
        {
          cudaStream_t stream;

          BREZEL_CUDA_CHECK(cudaStreamCreate(&stream));
          streams.push_back(stream);
        }
      }

      /**
       * @brief Destroy the Stream Pool object
       *
       * @version 1.0.0
       */
      ~StreamPool()
      {
        for (auto stream : streams)
          cudaStreamDestroy(stream);
      }

      /**
       * @brief Gets the current CUDA stream
       *
       * @return cudaStream_t Current CUDA stream
       * @version 1.0.0
       */
      cudaStream_t get_stream()
      {
        current_idx = (current_idx + 1) % POOL_SIZE;
        return streams[current_idx];
      }
    };

    /**
     * @brief Get the stream pool object
     *
     * @return StreamPool & Reference to the stream pool
     * @version 1.0.0
     */
    static StreamPool &get_stream_pool()
    {
      static StreamPool pool;
      return pool;
    }

  public:
    /**
     * @brief Get CUDA device properties
     *
     * @return std::optional<cudaDeviceProp> Device properties if available
     * @version 1.0.0
     */
    static std::optional<cudaDeviceProp> get_device_properties() noexcept
    {
      try
      {
        int device;
        BREZEL_CUDA_CHECK(cudaGetDevice(&device));

        cudaDeviceProp props;
        BREZEL_CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        return props;
      }
      catch (...)
      {
        return std::nullopt;
      }
    }

    /**
     * @brief Checks if CUDA is available
     *
     * @return True if CUDA device is available, false otherwise
     * @version 1.0.0
     */
    static bool is_available() noexcept
    {
      int device_count = 0;

      return cudaGetDeviceCount(&device_count) == cudaSuccess &&
             device_count > 0;
    }

    /**
     * @brief Get current CUDA stream
     *
     * @return cudaStream_t Current CUDA stream
     * @version 1.0.0
     */
    static cudaStream_t get_stream()
    {
      return get_stream_pool().get_stream();
    }

    /**
     * @brief Allocates memory on CUDA device
     *
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory
     * @version 1.0.0
     */
    static void *allocate(size_t size)
    {
      auto [ptr, from_pool] = get_pool().allocate(size);
      return ptr;
    }

    /**
     * @brief Frees CUDA device memory
     *
     * @param ptr Pointer to memory to free
     * @version 1.0.0
     */
    static void free(void *ptr) noexcept
    {
      if (ptr)
        get_pool().deallocate(ptr, 0);
    }

    /**
     * @brief Sets CUDA memory to a value
     *
     * @param ptr Pointer to memory
     * @param value Value to set
     * @param size Size in bytes
     * @version 1.0.0
     */
    static void memset(void *ptr, int value, size_t size)
    {
      BREZEL_CUDA_CHECK(cudaMemsetAsync(ptr, value, size, get_stream()));
    }

    /**
     * @brief Copies memory between locations with span support
     *
     * @param dst Destination span
     * @param src Source span
     * @param kind Type of memory transfer
     * @version 1.0.0
     */
    template <typename T>
    static void memcpy(boost::span<T> dst,
                       boost::span<const T> src, MemcpyKind kind)
    {
      BREZEL_CHECK(dst.size() == src.size(),
                   "Span sizes must match for memcpy");

      BREZEL_CUDA_CHECK(cudaMemcpyAsync(
          dst.data(), src.data(),
          dst.size_bytes(),
          static_cast<cudaMemcpyKind>(kind),
          get_stream()));
    }

    /**
     * @brief Copies memory between locations
     *
     * @param dst Destination pointer
     * @param src Source pointer
     * @param size Size in bytes
     * @param kind Type of memory transfer
     * @version 1.0.0
     */
    static void memcpy(void *dst, const void *src, size_t size, MemcpyKind kind)
    {
      BREZEL_CUDA_CHECK(cudaMemcpyAsync(
          dst, src, size,
          static_cast<cudaMemcpyKind>(kind),
          get_stream()));
    }

    /**
     * @brief Synchronizes all CUDA operations
     *
     * @version 1.0.0
     */
    static void synchronize()
    {
      BREZEL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    /**
     * @brief Synchronizes current CUDA stream
     *
     * @version 1.0.0
     */
    static void stream_synchronize()
    {
      BREZEL_CUDA_CHECK(cudaStreamSynchronize(get_stream()));
    }
  };
#endif
} // namespace brezel::core