#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/shape/shape.hpp>
#include <brezel/tensor/storage/storage.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

namespace brezel::tensor::io {

/**
 * @brief Supported file formats for tensor I/O
 */
enum class FileFormat {
    /// Automatic detection based on file extension
    Auto,

    /// Binary format (raw data)
    Binary,

    /// NumPy .npy format
    NPY,

    /// CSV format
    CSV,

    /// JSON format
    JSON
};

/**
 * @brief Options for tensor I/O operations
 */
struct BREZEL_API IOOptions {
    /// File format
    FileFormat format = FileFormat::Auto;

    /// Delimiter for CSV format
    char delimiter = ',';

    /// Whether to include header in CSV
    bool include_header = true;

    /// Whether to use scientific notation for floating-point values
    bool scientific = false;

    /// Floating-point precision
    int precision = 6;

    /// Whether to convert endianness if needed
    bool convert_endian = true;

    /// Whether to transpose 2D tensors (swap rows and columns)
    bool transpose = false;

    /// Row major (C-style) or column major (Fortran-style) ordering
    bool row_major = true;
};

/**
 * @brief Detect file format from extension
 *
 * @param path File path
 * @return FileFormat Detected format
 */
BREZEL_NODISCARD inline FileFormat detect_format(
    const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".npy") {
        return FileFormat::NPY;
    } else if (ext == ".csv") {
        return FileFormat::CSV;
    } else if (ext == ".json") {
        return FileFormat::JSON;
    } else {
        return FileFormat::Binary;
    }
}

/**
 * @brief Get file format name
 *
 * @param format File format
 * @return std::string Format name
 */
BREZEL_NODISCARD inline std::string format_to_string(FileFormat format) {
    switch (format) {
        case FileFormat::Auto:
            return "Auto";
        case FileFormat::Binary:
            return "Binary";
        case FileFormat::NPY:
            return "NPY";
        case FileFormat::CSV:
            return "CSV";
        case FileFormat::JSON:
            return "JSON";
        default:
            return "Unknown";
    }
}

/**
 * @brief Parse file format from string
 *
 * @param format_str Format string
 * @return FileFormat Parsed format
 * @throws LogicError if format is not recognized
 */
BREZEL_NODISCARD inline FileFormat parse_format(const std::string& format_str) {
    std::string lower = format_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "auto") {
        return FileFormat::Auto;
    } else if (lower == "binary" || lower == "bin" || lower == "raw") {
        return FileFormat::Binary;
    } else if (lower == "npy" || lower == "numpy") {
        return FileFormat::NPY;
    } else if (lower == "csv") {
        return FileFormat::CSV;
    } else if (lower == "json") {
        return FileFormat::JSON;
    } else {
        throw core::error::LogicError("Unknown file format: {}", format_str);
    }
}

/**
 * @brief Detect endianness of the system
 *
 * @return bool True if little endian
 */
BREZEL_NODISCARD inline bool is_little_endian() {
    const uint32_t value = 0x01020304;
    const auto ptr = reinterpret_cast<const uint8_t*>(&value);
    return ptr[0] == 0x04;
}

/**
 * @brief Swap bytes to change endianness
 *
 * @tparam T Data type
 * @param value Value to swap
 * @return T Value with swapped bytes
 */
template <typename T>
BREZEL_NODISCARD T swap_endian(T value) {
    static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

    union {
        T value;
        uint8_t bytes[sizeof(T)];
    } source, dest;

    source.value = value;

    for (size_t i = 0; i < sizeof(T); ++i) {
        dest.bytes[i] = source.bytes[sizeof(T) - i - 1];
    }

    return dest.value;
}

/**
 * @brief Get dtype string for NPY format
 *
 * @tparam T Data type
 * @param little_endian Whether system is little endian
 * @return std::string NPY dtype string
 */
template <typename T>
BREZEL_NODISCARD std::string get_npy_dtype(bool little_endian) {
    std::string endian_char = little_endian ? "<" : ">";

    if constexpr (std::is_same_v<T, float>) {
        return endian_char + "f4";
    } else if constexpr (std::is_same_v<T, double>) {
        return endian_char + "f8";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "|i1";
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return endian_char + "i2";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return endian_char + "i4";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return endian_char + "i8";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return "|u1";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return endian_char + "u2";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return endian_char + "u4";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return endian_char + "u8";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "|b1";
    } else {
        throw core::error::LogicError("Unsupported type for NPY format");
    }
}

/**
 * @brief Parse NPY dtype string
 *
 * @param dtype_str NPY dtype string
 * @return std::pair<std::type_index, bool> Type index and endianness
 * @throws LogicError if dtype is not recognized
 */
BREZEL_NODISCARD inline std::pair<std::type_index, bool> parse_npy_dtype(
    const std::string& dtype_str) {
    if (dtype_str.empty()) {
        throw core::error::LogicError("Empty dtype string");
    }

    bool little_endian = true;
    if (dtype_str[0] == '<') {
        little_endian = true;
    } else if (dtype_str[0] == '>') {
        little_endian = false;
    } else if (dtype_str[0] == '|') {
        little_endian = is_little_endian();
    } else {
        throw core::error::LogicError("Invalid dtype string: {}", dtype_str);
    }

    std::string type_part = dtype_str.substr(1);
    char type_char = type_part[0];
    int size = 0;

    try {
        size = std::stoi(type_part.substr(1));
    } catch (const std::exception& e) {
        throw core::error::LogicError("Invalid dtype size: {}",
                                      type_part.substr(1));
    }

    if (type_char == 'f' && size == 4) {
        return {std::type_index(typeid(float)), little_endian};
    } else if (type_char == 'f' && size == 8) {
        return {std::type_index(typeid(double)), little_endian};
    } else if (type_char == 'i' && size == 1) {
        return {std::type_index(typeid(int8_t)), little_endian};
    } else if (type_char == 'i' && size == 2) {
        return {std::type_index(typeid(int16_t)), little_endian};
    } else if (type_char == 'i' && size == 4) {
        return {std::type_index(typeid(int32_t)), little_endian};
    } else if (type_char == 'i' && size == 8) {
        return {std::type_index(typeid(int64_t)), little_endian};
    } else if (type_char == 'u' && size == 1) {
        return {std::type_index(typeid(uint8_t)), little_endian};
    } else if (type_char == 'u' && size == 2) {
        return {std::type_index(typeid(uint16_t)), little_endian};
    } else if (type_char == 'u' && size == 4) {
        return {std::type_index(typeid(uint32_t)), little_endian};
    } else if (type_char == 'u' && size == 8) {
        return {std::type_index(typeid(uint64_t)), little_endian};
    } else if (type_char == 'b' && size == 1) {
        return {std::type_index(typeid(bool)), little_endian};
    } else {
        throw core::error::LogicError("Unsupported dtype: {}", dtype_str);
    }
}

/**
 * @brief Calculate header size for NPY format
 *
 * @param header_dict Header dictionary
 * @return size_t Header size (padded to 64-byte boundary)
 */
BREZEL_NODISCARD inline size_t calculate_npy_header_size(
    const std::string& header_dict) {
    const size_t base_size = 10;
    const size_t header_len = header_dict.size();

    size_t total_len = base_size + header_len;
    size_t padding = (64 - (total_len % 64)) % 64;

    return total_len + padding;
}

/**
 * @brief Write NPY header
 *
 * @param stream Output stream
 * @param shape Shape of the tensor
 * @param dtype_str NPY dtype string
 * @param fortran_order Whether data is in Fortran order
 */
inline void write_npy_header(std::ostream& stream, const shape::Shape& shape,
                             const std::string& dtype_str, bool fortran_order) {
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t major_version = 1;
    const uint8_t minor_version = 0;

    stream.write(magic, sizeof(magic));
    stream.write(reinterpret_cast<const char*>(&major_version), 1);
    stream.write(reinterpret_cast<const char*>(&minor_version), 1);

    std::ostringstream header_stream;
    header_stream << "{'descr': '" << dtype_str << "', 'fortran_order': ";
    header_stream << (fortran_order ? "True" : "False") << ", 'shape': (";

    for (size_t i = 0; i < shape.size(); ++i) {
        header_stream << shape[i];
        if (i < shape.size() - 1) {
            header_stream << ", ";
        }
    }

    if (shape.size() == 1) {
        header_stream << ",";
    }

    header_stream << "), }";

    std::string header_dict = header_stream.str();
    size_t header_len = header_dict.size();
    uint16_t header_len_le = static_cast<uint16_t>(header_len);

    if (!is_little_endian()) {
        header_len_le = swap_endian(header_len_le);
    }

    stream.write(reinterpret_cast<const char*>(&header_len_le), 2);

    stream << header_dict;

    size_t current_pos = sizeof(magic) + 2 + 2 + header_len;
    size_t padding = (64 - (current_pos % 64)) % 64;

    for (size_t i = 0; i < padding; ++i) {
        stream.put(' ');
    }
}

/**
 * @brief Parse NPY header
 *
 * @param stream Input stream
 * @return std::tuple<shape::Shape, std::string, bool> Shape, dtype string, and
 * fortran_order flag
 * @throws LogicError if header is invalid
 */
BREZEL_NODISCARD inline std::tuple<shape::Shape, std::string, bool>
parse_npy_header(std::istream& stream) {
    char magic[6];
    stream.read(magic, sizeof(magic));

    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw core::error::LogicError("Invalid NPY file (wrong magic string)");
    }

    uint8_t major, minor;
    stream.read(reinterpret_cast<char*>(&major), 1);
    stream.read(reinterpret_cast<char*>(&minor), 1);

    if (major != 1 || minor != 0) {
        throw core::error::LogicError("Unsupported NPY version: {}.{}", major,
                                      minor);
    }

    uint16_t header_len;
    stream.read(reinterpret_cast<char*>(&header_len), 2);

    if (!is_little_endian()) {
        header_len = swap_endian(header_len);
    }

    std::vector<char> header_data(header_len);
    stream.read(header_data.data(), header_len);

    std::string header_str(header_data.begin(), header_data.end());
    size_t descr_pos = header_str.find("'descr'");
    if (descr_pos == std::string::npos) {
        throw core::error::LogicError("Invalid NPY header: missing 'descr'");
    }

    size_t descr_start = header_str.find("'", descr_pos + 7) + 1;
    size_t descr_end = header_str.find("'", descr_start);
    std::string dtype_str =
        header_str.substr(descr_start, descr_end - descr_start);

    size_t order_pos = header_str.find("'fortran_order'");
    if (order_pos == std::string::npos) {
        throw core::error::LogicError(
            "Invalid NPY header: missing 'fortran_order'");
    }

    size_t order_start = header_str.find(":", order_pos) + 1;
    while (order_start < header_str.size() &&
           isspace(header_str[order_start])) {
        order_start++;
    }

    bool fortran_order = false;
    if (header_str.substr(order_start, 4) == "True") {
        fortran_order = true;
    }

    size_t shape_pos = header_str.find("'shape'");
    if (shape_pos == std::string::npos) {
        throw core::error::LogicError("Invalid NPY header: missing 'shape'");
    }

    size_t shape_start = header_str.find("(", shape_pos) + 1;
    size_t shape_end = header_str.find(")", shape_start);
    std::string shape_str =
        header_str.substr(shape_start, shape_end - shape_start);

    std::vector<int64_t> dims;
    size_t pos = 0;

    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (isspace(shape_str[pos]) || shape_str[pos] == ',')) {
            pos++;
        }

        if (pos >= shape_str.size()) {
            break;
        }

        size_t num_end = pos;
        while (num_end < shape_str.size() &&
               (isdigit(shape_str[num_end]) || shape_str[num_end] == '-')) {
            num_end++;
        }

        if (num_end > pos) {
            int64_t dim = std::stoll(shape_str.substr(pos, num_end - pos));
            dims.push_back(dim);
            pos = num_end;
        } else {
            pos++;
        }
    }

    shape::Shape shape(dims);
    size_t header_size = 6 + 2 + 2 + header_len;
    size_t padding = (64 - (header_size % 64)) % 64;

    if (padding > 0) {
        stream.ignore(padding);
    }

    return {shape, dtype_str, fortran_order};
}

/**
 * @brief Format a single element for text output
 *
 * @tparam T Element type
 * @param value Element value
 * @param options I/O options
 * @return std::string Formatted value
 */
template <typename T>
BREZEL_NODISCARD std::string format_element(const T& value,
                                            const IOOptions& options) {
    std::ostringstream ss;

    if constexpr (std::is_floating_point_v<T>) {
        if (options.scientific) {
            ss << std::scientific;
        } else {
            ss << std::fixed;
        }

        ss.precision(options.precision);
    }

    ss << value;
    return ss.str();
}

/**
 * @brief Write CSV header row
 *
 * @param stream Output stream
 * @param shape Shape of the tensor
 * @param options I/O options
 */
inline void write_csv_header(std::ostream& stream, const shape::Shape& shape,
                             const IOOptions& options) {
    if (shape.size() < 1 || shape.size() > 2) {
        throw core::error::LogicError(
            "CSV format supports only 1D or 2D tensors, got {}D", shape.size());
    }

    if (shape.size() == 1) {
        stream << "value";
        return;
    }

    for (int64_t j = 0; j < shape[1]; ++j) {
        if (j > 0) {
            stream << options.delimiter;
        }

        stream << "col_" << j;
    }

    stream << "\n";
}

}  // namespace brezel::tensor::io