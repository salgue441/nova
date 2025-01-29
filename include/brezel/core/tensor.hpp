#pragma once

#include <brezel/core/shape.hpp>
#include <brezel/core/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/core/device.hpp>
#include <memory>
#include <vector>
#include <type_traits>
#include <concepts>
#include <optional>
#include <ranges>
#include <boost/container/small_vector.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/core/span.hpp>
#include <fmt/format.h>

namespace brezel::core
{
  // Concept for valid tensor data types with additional constraints
  template <typename T>
  concept TensorScalar = std::is_arithmetic_v<T> &&
                         !std::is_same_v<T, bool> &&
                         std::is_trivially_copyable_v<T>;

  /**
   * @class Tensor
   * @brief Base tensor class for multi-dimensional arrays
   *
   * @tparam T Data type of tensor elements (must satisfy TensorScalar concept)
   * @tparam DeviceType Device type for the tensor (e.g. CPU, CUDA)
   * @version 1.0.0
   */
  template <TensorScalar T, typename DeviceType = CPUDevice>
  class BREZEL_API Tensor
  {
  public:
    using value_type = T;
    using device_type = DeviceType;
    using size_type = typename Shape64::size_type;
    using shape_type = Shape64;
    using storage_type = boost::container::small_vector<T, 16>;

  private:
    /**
     * @class TensorImpl
     * @brief Inner implementation class for the tensor
     *
     * @version 1.0.0
     */
    class TensorImpl
    {
    public:
      shape_type shape;
      std::unique_ptr<T, void (*)(T *)> data;
      bool requires_grad;
      std::optional<storage_type> grad_storage;

      /**
       * @brief Construct a new TensorImpl object
       *
       * @param s Shape of the tensor
       * @param d Data pointer of the tensor
       * @param grad Flag indicating if the tensor requires gradients
       */
      TensorImpl(shape_type s,
                 std::unique_ptr<T, void (*)(T *)> d, bool grad = false)
          : shape(std::move(s)), data(std::move(d)), requires_grad(grad)
      {
        grad_storage = grad ? std::make_optional<storage_type>(s.numel())
                            : std::nullopt;
      }
    };

    std::shared_ptr<TensorImpl> m_impl;
    static thread_local boost::container::small_vector<
        std::unique_ptr<T[]>, 8>
        m_memory_pool;

  public:
    // Constructors
    /**
     * @brief Constructor with std::in_place
     *
     * @tparam Args Argument types for the tensor constructor
     * @param args Arguments for the tensor constructor
     * @version 1.0.0
     */
    template <typename... Args>
    explicit Tensor(std::in_place_t, Args &&...args)
        : m_impl(std::make_shared<TensorImpl>(std::forward<Args>(args)...)) {}

    /**
     * @brief Default constructor with empty state
     *
     * @version 1.0.0
     */
    Tensor() : Tensor(std::in_place,
                      shape_type{},
                      std::unique_ptr<T, void (*)(T *)>(nullptr, [](T *p)
                                                        { DeviceType::free(p); })) {}

    /**
     * @brief Shape constructor with optional gradient tracking
     *
     * @param shape Shape of the tensor
     * @param requires_grad Flag indicating if the tensor requires gradients
     * @version 1.0.0
     */
    explicit Tensor(const shape_type &shape, bool requires_grad = false)
        : Tensor(std::in_place, shape, allocate_memory(shape.numel()),
                 requires_grad) {}

    // Move operations
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    // Copy operations
    /**
     * @brief Copies the tensor data to a new one
     *
     * @param other Other tensor to copy
     * @return Tensor<T, DeviceType>& Reference to the copied tensor
     * @version 1.0.0
     */
    Tensor(const Tensor &other) : Tensor(other.shape())
    {
      DeviceType::memcpy(data(), other.data(), numel() * sizeof(T),
                         DeviceType::MemcpyKind::HostToDevice);

      if (other.requires_grad() && other.m_impl->grad_storage)
      {
        m_impl->requires_grad = true;
        m_impl->grad_storage = other.m_impl->grad_storage;
      }
    }

    // Static factory methods with boost::span support
    /**
     * @brief From a range of elements with a given shape
     *
     * @param range Range of elements to copy
     * @param shape Shape of the tensor
     * @return Tensor Tensor with the copied elements
     * @version 1.0.0
     */
    static Tensor from_range(std::ranges::input_range auto &&range,
                             const shape_type &shape)
    {
      Tensor result(shape);
      auto span = boost::span(result.data(), result.numel());

      std::ranges::copy(range, span.begin());
      return result;
    }

    /**
     * @brief Creates a tensor with zeros
     *
     * @param shape Shape of the tensor
     * @return Tensor Tensor with zeros as elements
     * @version 1.0.0
     */
    static Tensor zeros(const shape_type &shape)
    {
      Tensor result(shape);
      DeviceType::memset(result.data(), 0, result.numel() * sizeof(T));

      return result;
    }

    /**
     * @brief Creates a tensor with ones
     *
     * @param shape Shape of the tensor
     * @return Tensor Tensor with ones as elements
     * @version 1.0.0
     */
    static Tensor ones(const shape_type &shape)
    {
      Tensor result(shape);
      auto span = boost::span(result.data(), result.numel());

      std::ranges::fill(span, T(1));
      return result;
    }

    // View operations with boost::span
    /**
     * @brief Gets a view of the tensor data
     *
     * @return boost::span<T> Span of the tensor data
     * @version 1.0.0
     */
    [[nodiscard]] boost::span<T> view() noexcept
    {
      return {data(), numel()};
    }

    /**
     * @brief Gets a view of the tensor data
     *
     * @return boost::span<T> Span of the tensor data
     * @version 1.0.0
     */
    [[nodiscard]] boost::span<const T> view() const noexcept
    {
      return {data(), numel()};
    }

    // Range-based iteration support
    /**
     * @brief Gets the begin iterator of the tensor
     *
     * @return auto Begin iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto begin() noexcept { return view().begin(); }

    /**
     * @brief Gets the end iterator of the tensor
     *
     * @return auto End iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto end() noexcept { return view().end(); }

    /**
     * @brief Gets the begin iterator of the tensor
     *
     * @return auto Begin iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto begin() const noexcept { return view().begin(); }

    /**
     * @brief Gets the end iterator of the tensor
     *
     * @return auto End iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto end() const noexcept { return view().end(); }

    /**
     * @brief Gets the const begin iterator of the tensor
     *
     * @return auto Const begin iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto cbegin() const noexcept { return view().cbegin(); }

    /**
     * @brief Gets the const end iterator of the tensor
     *
     * @return auto Const end iterator of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] auto cend() const noexcept { return view().cend(); }

    // Property accessors
    /**
     * @brief Gets the shape of the tensor
     *
     * @return const shape_type& Shape of the tensor
     * @version 1.0.0
     */
    [[nodiscard]] const shape_type &shape() const noexcept
    {
      return m_impl->shape;
    }

    /**
     * @brief Number of elements in the tensor
     *
     * @return size_type Number of elements in the tensor
     * @version 1.0.0
     */
    [[nodiscard]] size_type numel() const noexcept
    {
      return m_impl->shape.numel();
    }

    /**
     * @brief Number of dimensions of the tensor
     *
     * @return size_type Number of dimensions
     * @version 1.0.0
     */
    [[nodiscard]] size_type ndim() const noexcept
    {
      return m_impl->shape.ndim();
    }

    /**
     * @brief Checks if the tensor requires grad
     *
     * @return true if the tensor requires grad, false otherwise
     * @version 1.0.0
     */
    [[nodiscard]] bool requires_grad() const noexcept
    {
      return m_impl->requires_grad;
    }

    /**
     * @brief Gets the data stored in the tensor
     *
     * @return T* Pointer to the tensor data
     * @version 1.0.0
     */
    [[nodiscard]] T *data() noexcept { return m_impl->data.get(); }

    /**
     * @brief Gets the data stored in the tensor
     *
     * @return const T* Pointer to the tensor data
     * @version 1.0.0
     */
    [[nodiscard]] const T *data() const noexcept { return m_impl->data.get(); }

    // Device transfer operations with std::optional status
    /**
     * @brief Tries to move the tensor to a cpu device
     *
     * @return std::optional Tensor<T, CPUDevice> Optional moved tensor
     * @version 1.0.0
     */
    [[nodiscard]] std::optional<Tensor> try_cpu() const noexcept
    {
      try
      {
        if constexpr (std::is_same_v<DeviceType, CPUDevice>)
          return *this;

        else
          return to_device<CPUDevice>();
      }
      catch (...)
      {
        return std::nullopt;
      }
    }

    /**
     * @brief Moves the tensor to a cpu device
     *
     * @return Tensor<T, CPUDevice> Moved tensor
     * @version 1.0.0
     */
    [[nodiscard]] Tensor cpu() const
    {
      return try_cpu().value_or(BREZEL_THROW("Failed to move tensor to CPU"));
    }

    /**
     * @brief Tries to move the tensor to a cuda device
     *
     * @return std::optional Tensor<T, CUDADevice> Optional moved tensor
     * @version 1.0.0
     */
    [[nodiscard]] std::optional<Tensor> try_cuda() const noexcept
    {
      try
      {
        if constexpr (std::is_same_v<DeviceType, CUDADevice>)
          return *this;

        else
          return to_device<CUDADevice>();
      }
      catch (...)
      {
        return std::nullopt;
      }
    }

    /**
     * @brief Moves the tensor to a cuda device
     *
     * @return Tensor<T, CUDADevice> Moved tensor
     * @version 1.0.0
     */
    [[nodiscard]] Tensor cuda() const
    {
      return try_cuda().value_or(BREZEL_THROW("Failed to move tensor to CUDA"));
    }

  private:
    /**
     * @brief Allocates memory with a pool
     *
     * @param size Size of memory to allocate
     * @return Unique pointer to allocated memory
     * @version 1.0.0
     */
    static std::unique_ptr<T, void (*)(T *)> allocate_memory(size_t size)
    {
      if (size == 0)
        return {nullptr, [](T *) {}};

      if (size <= 16 && !m_memory_pool.empty())
      {
        auto ptr = std::move(m_memory_pool.back());
        m_memory_pool.pop_back();

        return {ptr.release(), [](T *p)
                {
                  m_memory_pool.emplace_back(p);
                }};
      }

      return {
          static_cast<T *>(DeviceType::allocate(size * sizeof(T))),
          [](T *p)
          { DeviceType::free(p); }};
    }

    /**
     * @brief Moves the tensor to a device
     *
     * @tparam TargetDevice Target device type
     * @return Tensor<T, DeviceType> Moved tensor
     * @version 1.0.0
     */
    template <typename TargetDevice>
    [[nodiscard]] Tensor<T, TargetDevice> to_device() const
    {
      Tensor<T, TargetDevice> result(shape());
      TargetDevice::memcpy(
          result.data(),
          data(),
          numel() * sizeof(T),
          TargetDevice::MemcpyKind::HostToDevice);

      return result;
    }
  };

  // Initialize thread_local memory pool
  template <TensorScalar T, typename DeviceType>
  thread_local boost::container::small_vector<
      std::unique_ptr<T[]>, 8>
      Tensor<T, DeviceType>::m_memory_pool;

  // Type aliases for common tensor types
  using FloatTensor = Tensor<float>;
  using DoubleTensor = Tensor<double>;
  using IntTensor = Tensor<int32_t>;
  using LongTensor = Tensor<int64_t>;
} // namespace brezel::core