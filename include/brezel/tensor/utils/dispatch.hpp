#pragma once

#include <array>
#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/utils/type_traits.hpp>
#include <functional>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>
#include <variant>

namespace brezel::tensor::utils {
/**
 * @brief Type-based dispatch system for tensor operations
 *
 * @details Provides mechanisms to dispatch operations based on
 * runtime type information, enabling a consistent interface across
 * different tensor element types.
 */
namespace dispatch {
/**
 * @brief Type erased function wrapper
 *
 * @tparam Ret Return type of the function
 * @tparam Args Argument types of the function
 */
template <typename Ret, typename... Args>
class TypeErasedFunction {
public:
    /**
     * @brief Default constructor
     * @notes Creates a function that will error when called
     */
    TypeErasedFunction() : m_impl(nullptr) {}

    /**
     * @brief Constructor from a callable object
     *
     * @tparam F Function type
     * @param f Function to wrap
     */
    template <typename F>
    TypeErasedFunction(F&& f) : m_impl(std::forward<F>(f)) {}

    /**
     * @brief Call operator
     *
     * @param args Arguments
     * @return Ret Return value
     */
    Ret operator()(Args... args) const {
        if (!m_impl) {
            throw core::error::LogicError(
                "Attempt to call uninitialized type-erased function");
        }

        return m_impl(std::forward<Args>(args)...);
    }

    /**
     * @brief Check if function is callable
     *
     * @return true if callable, false otherwise
     */
    explicit operator bool() const { return static_cast<bool>(m_impl); }

private:
    std::function<Ret(Args...)> m_impl;
};

/**
 * @brief Type dispatch registry for functions with a specific signature
 *
 * @tparam Ret Return type of the functions
 * @tparam Args Argument types of the functions
 */
template <typename Ret, typename... Args>
class TypeDispatchRegistry {
public:
    using FunctionType = TypeErasedFunction<Ret, Args...>;

    /**
     * @brief Register a function for a specific type
     *
     * @tparam T Type to register for
     * @tparam F Function type
     * @param func Function to register
     */
    template <typename T, typename F>
    void register_function(F&& func) {
        auto type_idx = std::type_index(typeid(T));
        m_functions[type_idx] = FunctionType(std::forward<F>(func));
    }

    /**
     * @brief Get function for a specific type
     *
     * @tparam T Type to get function for
     * @return const FunctionType& Function
     * @throws core::error::LogicError if type not registered
     */
    template <typename T>
    const FunctionType& get_function() const {
        auto type_idx = std::type_index(typeid(T));
        auto it = m_functions.find(type_idx);

        if (it == m_functions.end()) {
            throw core::error::LogicError("No function registered for type {}",
                                          typeid(T).name());
        }

        return it->second;
    }

    /**
     * @brief Check if a function is registered for a type
     *
     * @tparam T Type to check
     * @return true if registered
     */
    template <typename T>
    bool has_function() const {
        auto type_idx = std::type_index(typeid(T));
        return m_functions.find(type_idx) != m_functions.end();
    }

private:
    std::unordered_map<std::type_index, FunctionType> m_functions;
};

/**
 * @brief Type dispatch registry for binary operations
 *
 * @tparam Ret Return type
 * @tparam Arg1 First argument type
 * @tparam Arg2 Second argument type
 * @tparam Args Additional argument types
 */
template <typename Ret, typename Arg1, typename Arg2, typename... Args>
class BinaryTypeDispatchRegistry {
public:
    using FunctionType = TypeErasedFunction<Ret, Arg1.Arg2, Args...>;

    /**
     * @brief Register a function for a specific pair of types
     *
     * @tparam T1 First type
     * @tparam T2 Second type
     * @tparam F Function type
     * @param func Function to register
     */
    template <typename T1, typename T2, typename F>
    void register_function(F&& func) {
        auto key = std::make_pair(std::type_index(typeid(T1)),
                                  std::type_index(typeid(T2)));

        m_functions[key] = FunctionType(std::forward<F>(func));
    }

    /**
     * @brief Get function for a specific pair of types
     *
     * @tparam T1 First type
     * @tparam T2 Second type
     * @return const FunctionType& Function
     * @throws core::error::LogicError if type pair not registered
     */
    template <typename T1, typename T2>
    const FunctionType& get_function() const {
        auto key = std::make_pair(std::type_index(typeid(T1)),
                                  std::type_index(typeid(T2)));

        auto it = m_functions.find(key);

        if (it == m_functions.end()) {
            throw core::error::LogicError(
                "No function registered for types {} and {}", typeid(T1).name(),
                typeid(T2).name());
        }

        return it->second;
    }

    /**
     * @brief Check if a function is registered for a pair of types
     *
     * @tparam T1 First type
     * @tparam T2 Second type
     * @return true if registered
     */
    template <typename T1, typename T2>
    bool has_function() const {
        auto key = std::make_pair(std::type_index(typeid(T1)),
                                  std::type_index(typeid(T2)));

        return m_functions.find(key) != m_functions.end();
    }

private:
    using KeyType = std::pair<std::type_index, std::type_index>;
    struct KeyHash {
        std::size_t operator()(const KeyType& key) const {
            return std::hash<std::type_index>()(key.first) ^
                   std::hash<std::type_index>()(key.second);
        }
    };

    struct KeyEqual {
        bool operator()(const KeyType& lhs, const KeyType& rhs) const {
            return lhs.first == rhs.first && lhs.second == rhs.second;
        }
    };

    std::unordered_map<KeyType, FunctionType, KeyHash, KeyEqual> m_functions;
};

/**
 * @brief Enumeration of supported data types
 */
enum class DataType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    Complex64,
    Complex128,
    Unknown
};

/**
 * @brief Convert a data type to a string
 *
 * @param dtype Data type
 * @return std::string_view Type name
 */
BREZEL_NODISCARD constexpr std::string_view data_type_name(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return "float32";
        case DataType::Float64:
            return "float64";
        case DataType::Int8:
            return "int8";
        case DataType::Int16:
            return "int16";
        case DataType::Int32:
            return "int32";
        case DataType::Int64:
            return "int64";
        case DataType::UInt8:
            return "uint8";
        case DataType::UInt16:
            return "uint16";
        case DataType::UInt32:
            return "uint32";
        case DataType::UInt64:
            return "uint64";
        case DataType::Bool:
            return "bool";
        case DataType::Complex64:
            return "complex64";
        case DataType::Complex128:
            return "complex128";
        case DataType::Unknown:
            return "unknown";
        default:
            return "unknown";
    }
}

/**
 * @brief Get the DataType corresponding to the template type T.
 *
 * This function uses constexpr if statements to determine the DataType
 * based on the type T. It supports various fundamental types, including
 * floating-point, integer, unsigned integer, boolean, and complex types.
 * If the type T does not match any of the supported types, it returns
 * DataType::Unknown.
 *
 * @tparam T The type for which to get the corresponding DataType.
 * @return DataType The DataType corresponding to the type T.
 */
template <typename T>
constexpr DataType get_data_type() {
    if constexpr (std::is_same_v<T, float>) {
        return DataType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DataType::Float64;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DataType::Int8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return DataType::Int16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DataType::Int64;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DataType::UInt8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DataType::UInt16;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UInt32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return DataType::UInt64;
    } else if constexpr (std::is_same_v<T, bool>) {
        return DataType::Bool;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return DataType::Complex64;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return DataType::Complex128;
    } else {
        return DataType::Unknown;
    }
}

/**
 * @class TypeErased
 * @brief A class for type-erased storage of various data types, including
 * complex numbers.
 *
 * The TypeErased class provides a mechanism to store and manipulate values of
 * different data types in a type-erased manner. It supports both real and
 * complex numbers, and provides methods to retrieve the stored value in a
 * specified type.
 *
 * The class uses a variant to hold the data and provides template methods to
 * handle different types at compile time. It also includes functionality to
 * check if the stored value is complex and to convert the stored data to a
 * string representation.
 *
 * @note The class supports the following data types:
 * - float
 * - double
 * - int8_t
 * - int16_t
 * - int32_t
 * - int64_t
 * - uint8_t
 * - uint16_t
 * - uint32_t
 * - uint64_t
 * - bool
 * - std::complex<float>
 * - std::complex<double>
 *
 * @throws core::error::LogicError if an unsupported type is used for type
 * erasure or conversion.
 */
class TypeErased {
public:
    TypeErased()
        : m_dtype(DataType::Unknown),
          m_data(0),
          m_is_complex(false),
          m_complex_imag(0.0) {}

    /**
     * @brief Constructs a TypeErased object from a given value.
     *
     * This constructor template initializes the TypeErased object with the
     * provided value. It determines the data type of the value using the
     * get_data_type<T>() function and stores the value in the m_data member. If
     * the value is of a complex type, it also sets the m_is_complex flag and
     * stores the imaginary part separately.
     *
     * @tparam T The type of the value to be type-erased.
     * @param value The value to be type-erased and stored in the TypeErased
     * object.
     *
     * @throws core::error::LogicError if the type of the value is not supported
     * for type erasure.
     */
    template <typename T>
    explicit TypeErased(T value) : m_dtype(get_data_type<T>()) {
        if constexpr (std::is_same_v<T, float>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, double>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, bool>) {
            m_data = value;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            m_data = value.real();

            m_is_complex = true;
            m_complex_imag = value.imag();
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            m_data = value.real();

            m_is_complex = true;
            m_complex_imag = value.imag();
        } else {
            throw core::error::LogicError(
                "Unsupported type for type erasure: {}", typeid(T).name());
        }
    }

    /**
     * @brief Retrieves the data type of the tensor.
     *
     * @return DataType The data type of the tensor.
     */
    DataType dtype() const noexcept { return m_dtype; }

    /**
     * @brief Checks if the tensor is complex.
     *
     * This function returns a boolean value indicating whether the tensor
     * is of a complex data type.
     *
     * @return true if the tensor is complex, false otherwise.
     */
    bool is_complex() const noexcept { return m_is_complex; }

    /**
     * @brief Converts the current object to the specified type T.
     *
     * This function template converts the current object to the specified type
     * T. It handles both complex and non-complex tensor types.
     *
     * @tparam T The type to which the current object will be converted.
     *
     * @return The current object converted to the specified type T.
     *
     * If T is a complex tensor type and the current object is not complex,
     * the imaginary part will be set to 0. If the current object is complex,
     * the real and imaginary parts will be used to construct the complex type.
     *
     * If T is not a complex tensor type, the real part of the current object
     * will be returned regardless of whether the current object is complex or
     * not.
     */
    template <typename T>
    T as() const {
        if constexpr (TensorComplex<T>) {
            if (!m_is_complex) {
                return T(get_real<typename TensorTypeTraits<T>::real_type>(),
                         0);
            } else {
                return T(get_real<typename TensorTypeTraits<T>::real_type>(),
                         static_cast<typename TensorTypeTraits<T>::real_type>(
                             m_complex_imag));
            }
        } else {
            if (m_is_complex) {
                return get_real<T>();
            } else {
                return get_real<T>();
            }
        }
    }

    /**
     * @brief Retrieves the value stored in the variant `m_data` as the
     * specified type `T`.
     *
     * This function attempts to convert the value stored in `m_data` to the
     * specified type `T`. It supports conversion to `float`, `double`, integral
     * types, and `bool`.
     *
     * @tparam T The type to which the value should be converted.
     * @return T The value stored in `m_data` converted to type `T`.
     *
     * @throws core::error::LogicError If the conversion to type `T` is not
     * possible.
     *
     * @note The function uses `if constexpr` to handle different types at
     * compile time. It checks the type of `m_data` using
     * `std::holds_alternative` and retrieves the value using `std::get`,
     * performing necessary type conversions.
     *
     * @note For `bool` type, if `m_data` holds a boolean, it returns the
     * boolean value. Otherwise, it uses `std::visit` to check if the stored
     * value is non-zero.
     */
    template <typename T>
    T get_real() const {
        if constexpr (std::is_same_v<T, float>) {
            if (std::holds_alternative<float>(m_data)) {
                return std::get<float>(m_data);
            } else if (std::holds_alternative<double>(m_data)) {
                return static_cast<T>(std::get<double>(m_data));
            } else if (std::holds_alternative<int8_t>(m_data)) {
                return static_cast<T>(std::get<int8_t>(m_data));
            } else if (std::holds_alternative<int16_t>(m_data)) {
                return static_cast<T>(std::get<int16_t>(m_data));
            } else if (std::holds_alternative<int32_t>(m_data)) {
                return static_cast<T>(std::get<int32_t>(m_data));
            } else if (std::holds_alternative<int64_t>(m_data)) {
                return static_cast<T>(std::get<int64_t>(m_data));
            } else if (std::holds_alternative<uint8_t>(m_data)) {
                return static_cast<T>(std::get<uint8_t>(m_data));
            } else if (std::holds_alternative<uint16_t>(m_data)) {
                return static_cast<T>(std::get<uint16_t>(m_data));
            } else if (std::holds_alternative<uint32_t>(m_data)) {
                return static_cast<T>(std::get<uint32_t>(m_data));
            } else if (std::holds_alternative<uint64_t>(m_data)) {
                return static_cast<T>(std::get<uint64_t>(m_data));
            } else if (std::holds_alternative<bool>(m_data)) {
                return std::get<bool>(m_data) ? T(1) : T(0);
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (std::holds_alternative<float>(m_data)) {
                return static_cast<T>(std::get<float>(m_data));
            } else if (std::holds_alternative<double>(m_data)) {
                return std::get<double>(m_data);
            } else if (std::holds_alternative<int8_t>(m_data)) {
                return static_cast<T>(std::get<int8_t>(m_data));
            } else if (std::holds_alternative<int16_t>(m_data)) {
                return static_cast<T>(std::get<int16_t>(m_data));
            } else if (std::holds_alternative<int32_t>(m_data)) {
                return static_cast<T>(std::get<int32_t>(m_data));
            } else if (std::holds_alternative<int64_t>(m_data)) {
                return static_cast<T>(std::get<int64_t>(m_data));
            } else if (std::holds_alternative<uint8_t>(m_data)) {
                return static_cast<T>(std::get<uint8_t>(m_data));
            } else if (std::holds_alternative<uint16_t>(m_data)) {
                return static_cast<T>(std::get<uint16_t>(m_data));
            } else if (std::holds_alternative<uint32_t>(m_data)) {
                return static_cast<T>(std::get<uint32_t>(m_data));
            } else if (std::holds_alternative<uint64_t>(m_data)) {
                return static_cast<T>(std::get<uint64_t>(m_data));
            } else if (std::holds_alternative<bool>(m_data)) {
                return std::get<bool>(m_data) ? T(1) : T(0);
            }
        } else if constexpr (std::is_integral_v<T>) {
            if (std::holds_alternative<float>(m_data)) {
                return static_cast<T>(std::get<float>(m_data));
            } else if (std::holds_alternative<double>(m_data)) {
                return static_cast<T>(std::get<double>(m_data));
            } else if (std::holds_alternative<int8_t>(m_data)) {
                return static_cast<T>(std::get<int8_t>(m_data));
            } else if (std::holds_alternative<int16_t>(m_data)) {
                return static_cast<T>(std::get<int16_t>(m_data));
            } else if (std::holds_alternative<int32_t>(m_data)) {
                return static_cast<T>(std::get<int32_t>(m_data));
            } else if (std::holds_alternative<int64_t>(m_data)) {
                return static_cast<T>(std::get<int64_t>(m_data));
            } else if (std::holds_alternative<uint8_t>(m_data)) {
                return static_cast<T>(std::get<uint8_t>(m_data));
            } else if (std::holds_alternative<uint16_t>(m_data)) {
                return static_cast<T>(std::get<uint16_t>(m_data));
            } else if (std::holds_alternative<uint32_t>(m_data)) {
                return static_cast<T>(std::get<uint32_t>(m_data));
            } else if (std::holds_alternative<uint64_t>(m_data)) {
                return static_cast<T>(std::get<uint64_t>(m_data));
            } else if (std::holds_alternative<bool>(m_data)) {
                return std::get<bool>(m_data) ? T(1) : T(0);
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (std::holds_alternative<bool>(m_data)) {
                return std::get<bool>(m_data);
            } else {
                return std::visit([](auto&& arg) { return arg != 0; }, m_data);
            }
        }

        throw core::error::LogicError("Cannot convert type {} to {}",
                                      data_type_name(m_dtype),
                                      typeid(T).name());
    }

    /**
     * @brief Returns the imaginary part of the complex number.
     *
     * This function returns the imaginary component of a complex number
     * stored in the object. It is marked as noexcept, indicating that it
     * does not throw any exceptions.
     *
     * @return double The imaginary part of the complex number.
     */
    double imag() const noexcept { return m_complex_imag; }

    /**
     * @brief Converts the stored data to a string representation.
     *
     * This function generates a string representation of the stored data. If
     * the data is complex, it includes both the real and imaginary parts in the
     * format "real+imagj" or "real-imagj". If the data is not complex, it
     * simply converts the data to a string.
     *
     * @return A string representation of the stored data.
     */
    std::string to_string() const {
        std::stringstream ss;

        if (m_is_complex) {
            ss << std::visit([](auto&& arg) { return std::to_string(arg); },
                             m_data);

            if (m_complex_imag >= 0.0) {
                ss << "+" << m_complex_imag << "j";
            } else {
                ss << m_complex_imag << "j";
            }
        } else {
            std::visit([&ss](auto&& arg) { ss << arg; }, m_data);
        }

        return ss.str();
    }

private:
    DataType m_dtype = DataType::Unknown;
    std::variant<float, double, int8_t, int16_t, int32_t, int64_t, uint8_t,
                 uint16_t, uint32_t, uint64_t, bool>
        m_data;

    bool m_is_complex = false;
    double m_complex_imag = 0.0;
};

/**
 * @class Dispatcher
 * @brief Singleton class responsible for registering standard and binary
 * operations for various types.
 *
 * The Dispatcher class provides methods to register standard types and binary
 * operations for a variety of types including floating point types, integer
 * types, boolean, and complex types. It ensures that the registration is done
 * in a singleton instance to maintain a single point of control.
 *
 * Usage:
 * @code
 * Dispatcher::instance().register_standard_types<Op>(registry);
 * Dispatcher::instance().register_binary_standard_types<Op>(binary_registry);
 * @endcode
 *
 * The class is non-copyable and non-movable to ensure the singleton property.
 */
class Dispatcher {
public:
    static Dispatcher& instance() {
        static Dispatcher instance;
        return instance;
    }

    /**
     * @brief Registers standard types with the given TypeDispatchRegistry.
     *
     * This function registers a set of standard types (such as float, double,
     * various integer types, boolean, and complex numbers) with the provided
     * TypeDispatchRegistry. It uses the provided template operation (Op) to
     * register functions for each type.
     *
     * @tparam Op The template operation to be registered for each type.
     * @tparam Ret The return type of the functions to be registered.
     * @tparam Args The argument types of the functions to be registered.
     * @param registry The TypeDispatchRegistry instance where the functions
     * will be registered.
     */
    template <template <typename> class Op, typename Ret, typename... Args>
    void register_standard_types(TypeDispatchRegistry<Ret, Args...>& registry) {
        registry.template register_function<float, Op<float>>();
        registry.template register_function<double, Op<double>>();

        registry.template register_function<int8_t, Op<int8_t>>();
        registry.template register_function<int16_t, Op<int16_t>>();
        registry.template register_function<int32_t, Op<int32_t>>();
        registry.template register_function<int64_t, Op<int64_t>>();
        registry.template register_function<uint8_t, Op<uint8_t>>();
        registry.template register_function<uint16_t, Op<uint16_t>>();
        registry.template register_function<uint32_t, Op<uint32_t>>();
        registry.template register_function<uint64_t, Op<uint64_t>>();

        registry.template register_function<bool, Op<bool>>();

        registry.template register_function<std::complex<float>,
                                            Op<std::complex<float>>>();
        registry.template register_function<std::complex<double>,
                                            Op<std::complex<double>>>();
    }

    /**
     * @brief Registers binary operations for standard types in the given
     * registry.
     *
     * This function registers binary operations for a variety of standard types
     * including floating point types, integer types, boolean, and complex
     * types. The operations are registered in the provided
     * BinaryTypeDispatchRegistry.
     *
     * @tparam Op The binary operation template to be registered.
     * @tparam Ret The return type of the binary operation.
     * @tparam Args The argument types of the binary operation.
     * @param registry The BinaryTypeDispatchRegistry where the functions will
     * be registered.
     *
     * The following type combinations are registered:
     * - float, float
     * - float, double
     * - float, int32_t
     * - float, int64_t
     * - double, float
     * - double, double
     * - double, int32_t
     * - double, int64_t
     * - int32_t, float
     * - int32_t, double
     * - int32_t, int32_t
     * - int32_t, int64_t
     * - int64_t, float
     * - int64_t, double
     * - int64_t, int32_t
     * - int64_t, int64_t
     * - int8_t, int8_t
     * - int16_t, int16_t
     * - uint8_t, uint8_t
     * - uint16_t, uint16_t
     * - uint32_t, uint32_t
     * - uint64_t, uint64_t
     * - bool, bool
     * - std::complex<float>, std::complex<float>
     * - std::complex<double>, std::complex<double>
     * - std::complex<float>, float
     * - float, std::complex<float>
     * - std::complex<double>, double
     * - double, std::complex<double>
     * - std::complex<float>, std::complex<double>
     * - std::complex<double>, std::complex<float>
     */
    template <template <typename, typename> class Op, typename Ret,
              typename... Args>
    void register_binary_standard_types(
        BinaryTypeDispatchRegistry<Ret, Args...>& registry) {
        registry.template register_function<float, float, Op<float, float>>();
        registry.template register_function<float, double, Op<float, double>>();
        registry
            .template register_function<float, int32_t, Op<float, int32_t>>();
        registry
            .template register_function<float, int64_t, Op<float, int64_t>>();

        registry.template register_function<double, float, Op<double, float>>();
        registry
            .template register_function<double, double, Op<double, double>>();
        registry
            .template register_function<double, int32_t, Op<double, int32_t>>();
        registry
            .template register_function<double, int64_t, Op<double, int64_t>>();

        registry
            .template register_function<int32_t, float, Op<int32_t, float>>();
        registry
            .template register_function<int32_t, double, Op<int32_t, double>>();
        registry.template register_function<int32_t, int32_t,
                                            Op<int32_t, int32_t>>();
        registry.template register_function<int32_t, int64_t,
                                            Op<int32_t, int64_t>>();

        registry
            .template register_function<int64_t, float, Op<int64_t, float>>();
        registry
            .template register_function<int64_t, double, Op<int64_t, double>>();
        registry.template register_function<int64_t, int32_t,
                                            Op<int64_t, int32_t>>();
        registry.template register_function<int64_t, int64_t,
                                            Op<int64_t, int64_t>>();

        registry
            .template register_function<int8_t, int8_t, Op<int8_t, int8_t>>();
        registry.template register_function<int16_t, int16_t,
                                            Op<int16_t, int16_t>>();
        registry.template register_function<uint8_t, uint8_t,
                                            Op<uint8_t, uint8_t>>();
        registry.template register_function<uint16_t, uint16_t,
                                            Op<uint16_t, uint16_t>>();
        registry.template register_function<uint32_t, uint32_t,
                                            Op<uint32_t, uint32_t>>();
        registry.template register_function<uint64_t, uint64_t,
                                            Op<uint64_t, uint64_t>>();

        registry.template register_function<bool, bool, Op<bool, bool>>();

        registry.template register_function<
            std::complex<float>, std::complex<float>,
            Op<std::complex<float>, std::complex<float>>>();
        registry.template register_function<
            std::complex<double>, std::complex<double>,
            Op<std::complex<double>, std::complex<double>>>();
        registry.template register_function<std::complex<float>, float,
                                            Op<std::complex<float>, float>>();
        registry.template register_function<float, std::complex<float>,
                                            Op<float, std::complex<float>>>();
        registry.template register_function<std::complex<double>, double,
                                            Op<std::complex<double>, double>>();
        registry.template register_function<double, std::complex<double>,
                                            Op<double, std::complex<double>>>();

        registry.template register_function<
            std::complex<float>, std::complex<double>,
            Op<std::complex<float>, std::complex<double>>>();
        registry.template register_function<
            std::complex<double>, std::complex<float>,
            Op<std::complex<double>, std::complex<float>>>();
    }

private:
    Dispatcher() = default;
    ~Dispatcher() = default;

    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;
    Dispatcher(Dispatcher&&) = delete;
    Dispatcher& operator=(Dispatcher&&) = delete;
};

/**
 * @brief Dispatches a function call based on the provided data type.
 *
 * This function uses a registry to retrieve and call a function corresponding
 * to the given data type. It supports various data types including floating
 * point, integer, boolean, and complex types.
 *
 * @tparam Ret The return type of the function to be dispatched.
 * @tparam Args The argument types of the function to be dispatched.
 * @param registry The registry containing the functions for different data
 * types.
 * @param dtype The data type based on which the function is dispatched.
 * @param args The arguments to be forwarded to the dispatched function.
 * @return The result of the dispatched function call.
 * @throws core::error::LogicError If no function is registered for the given
 * data type.
 */
template <typename Ret, typename... Args>
Ret dispatch_function(const TypeDispatchRegistry<Ret, Args...>& registry,
                      DataType dtype, Args... args) {
    switch (dtype) {
        case DataType::Float32:
            return registry.template get_function<float>()(
                std::forward<Args>(args)...);
        case DataType::Float64:
            return registry.template get_function<double>()(
                std::forward<Args>(args)...);
        case DataType::Int8:
            return registry.template get_function<int8_t>()(
                std::forward<Args>(args)...);
        case DataType::Int16:
            return registry.template get_function<int16_t>()(
                std::forward<Args>(args)...);
        case DataType::Int32:
            return registry.template get_function<int32_t>()(
                std::forward<Args>(args)...);
        case DataType::Int64:
            return registry.template get_function<int64_t>()(
                std::forward<Args>(args)...);
        case DataType::UInt8:
            return registry.template get_function<uint8_t>()(
                std::forward<Args>(args)...);
        case DataType::UInt16:
            return registry.template get_function<uint16_t>()(
                std::forward<Args>(args)...);
        case DataType::UInt32:
            return registry.template get_function<uint32_t>()(
                std::forward<Args>(args)...);
        case DataType::UInt64:
            return registry.template get_function<uint64_t>()(
                std::forward<Args>(args)...);
        case DataType::Bool:
            return registry.template get_function<bool>()(
                std::forward<Args>(args)...);
        case DataType::Complex64:
            return registry.template get_function<std::complex<float>>()(
                std::forward<Args>(args)...);
        case DataType::Complex128:
            return registry.template get_function<std::complex<double>>()(
                std::forward<Args>(args)...);
        default:
            throw core::error::LogicError("No function registered for type {}",
                                          data_type_name(dtype));
    }
}

/**
 * @brief Dispatches a binary function based on the data types of the arguments.
 *
 * This function uses a registry to look up and call a binary function that
 * matches the provided data types of the arguments. It supports various
 * combinations of data types including floating point, integer, and complex
 * types.
 *
 * @tparam Ret The return type of the binary function.
 * @tparam Arg1 The type of the first argument.
 * @tparam Arg2 The type of the second argument.
 * @tparam Args The types of additional arguments.
 * @param registry The registry containing the binary functions.
 * @param dtype1 The data type of the first argument.
 * @param dtype2 The data type of the second argument.
 * @param arg1 The first argument.
 * @param arg2 The second argument.
 * @param args Additional arguments.
 * @return The result of the dispatched binary function.
 * @throws core::error::LogicError If no function is registered for the given
 * data types.
 */
template <typename Ret, typename Arg1, typename Arg2, typename... Args>
Ret dispatch_binary_function(
    const BinaryTypeDispatchRegistry<Ret, Arg1, Arg2, Args...>& registry,
    DataType dtype1, DataType dtype2, Arg1 arg1, Arg2 arg2, Args... args) {
#define DISPATCH_BINARY(T1, T2)                      \
    return registry.template get_function<T1, T2>()( \
        arg1, arg2, std::forward<Args>(args)...)

    if (dtype1 == DataType::Float32 && dtype2 == DataType::Float32) {
        DISPATCH_BINARY(float, float);
    } else if (dtype1 == DataType::Float64 && dtype2 == DataType::Float64) {
        DISPATCH_BINARY(double, double);
    } else if (dtype1 == DataType::Int32 && dtype2 == DataType::Int32) {
        DISPATCH_BINARY(int32_t, int32_t);
    } else if (dtype1 == DataType::Int64 && dtype2 == DataType::Int64) {
        DISPATCH_BINARY(int64_t, int64_t);
    }

    switch (dtype1) {
        case DataType::Float32:
            switch (dtype2) {
                case DataType::Float32:
                    DISPATCH_BINARY(float, float);
                case DataType::Float64:
                    DISPATCH_BINARY(float, double);
                case DataType::Int32:
                    DISPATCH_BINARY(float, int32_t);
                case DataType::Int64:
                    DISPATCH_BINARY(float, int64_t);
                case DataType::Complex64:
                    DISPATCH_BINARY(float, std::complex<float>);
                default:
                    break;
            }
            break;

        case DataType::Float64:
            switch (dtype2) {
                case DataType::Float32:
                    DISPATCH_BINARY(double, float);
                case DataType::Float64:
                    DISPATCH_BINARY(double, double);
                case DataType::Int32:
                    DISPATCH_BINARY(double, int32_t);
                case DataType::Int64:
                    DISPATCH_BINARY(double, int64_t);
                case DataType::Complex128:
                    DISPATCH_BINARY(double, std::complex<double>);
                default:
                    break;
            }
            break;

        case DataType::Int32:
            switch (dtype2) {
                case DataType::Float32:
                    DISPATCH_BINARY(int32_t, float);
                case DataType::Float64:
                    DISPATCH_BINARY(int32_t, double);
                case DataType::Int32:
                    DISPATCH_BINARY(int32_t, int32_t);
                case DataType::Int64:
                    DISPATCH_BINARY(int32_t, int64_t);
                default:
                    break;
            }
            break;

        case DataType::Int64:
            switch (dtype2) {
                case DataType::Float32:
                    DISPATCH_BINARY(int64_t, float);
                case DataType::Float64:
                    DISPATCH_BINARY(int64_t, double);
                case DataType::Int32:
                    DISPATCH_BINARY(int64_t, int32_t);
                case DataType::Int64:
                    DISPATCH_BINARY(int64_t, int64_t);
                default:
                    break;
            }
            break;

        case DataType::Complex64:
            switch (dtype2) {
                case DataType::Float32:
                    DISPATCH_BINARY(std::complex<float>, float);
                case DataType::Complex64:
                    DISPATCH_BINARY(std::complex<float>, std::complex<float>);
                case DataType::Complex128:
                    DISPATCH_BINARY(std::complex<float>, std::complex<double>);
                default:
                    break;
            }
            break;

        case DataType::Complex128:
            switch (dtype2) {
                case DataType::Float64:
                    DISPATCH_BINARY(std::complex<double>, double);
                case DataType::Complex64:
                    DISPATCH_BINARY(std::complex<double>, std::complex<float>);
                case DataType::Complex128:
                    DISPATCH_BINARY(std::complex<double>, std::complex<double>);
                default:
                    break;
            }
            break;

        case DataType::Bool:
            if (dtype2 == DataType::Bool)
                DISPATCH_BINARY(bool, bool);
            break;

        case DataType::Int8:
            if (dtype2 == DataType::Int8)
                DISPATCH_BINARY(int8_t, int8_t);
            break;

        case DataType::Int16:
            if (dtype2 == DataType::Int16)
                DISPATCH_BINARY(int16_t, int16_t);
            break;

        case DataType::UInt8:
            if (dtype2 == DataType::UInt8)
                DISPATCH_BINARY(uint8_t, uint8_t);
            break;

        case DataType::UInt16:
            if (dtype2 == DataType::UInt16)
                DISPATCH_BINARY(uint16_t, uint16_t);
            break;

        case DataType::UInt32:
            if (dtype2 == DataType::UInt32)
                DISPATCH_BINARY(uint32_t, uint32_t);
            break;

        case DataType::UInt64:
            if (dtype2 == DataType::UInt64)
                DISPATCH_BINARY(uint64_t, uint64_t);
            break;

        default:
            break;

#undef DISPATCH_BINARY

            throw core::error::LogicError(
                "No function registered for types {} and {}",
                data_type_name(dtype1), data_type_name(dtype2));
    }

    throw core::error::LogicError("No function registered for types {} and {}",
                                  data_type_name(dtype1),
                                  data_type_name(dtype2));
}
}  // namespace dispatch
}  // namespace brezel::tensor::utils