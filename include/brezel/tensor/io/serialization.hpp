#pragma once

#pragma once

#include <brezel/core/error/error.hpp>
#include <brezel/core/macros.hpp>
#include <brezel/tensor/io/format.hpp>
#include <brezel/tensor/shape/shape.hpp>
#include <brezel/tensor/storage/shared_storage.hpp>
#include <brezel/tensor/storage/storage.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace brezel::tensor::io {

/**
 * @brief Save tensor data to a binary file
 *
 * @tparam T Element type
 * @param storage Storage to save
 * @param shape Shape of the tensor
 * @param path File path
 * @param options I/O options
 * @throws core::error::RuntimeError if file cannot be opened
 */
template <typename T>
void save_binary(const storage::Storage<T>& storage, const shape::Shape& shape,
                 const std::filesystem::path& path,
                 const IOOptions& options = IOOptions{}) {
    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw core::error::RuntimeError("Failed to open file for writing: {}",
                                        path.string());
    }

    uint32_t ndim = shape.size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    for (size_t i = 0; i < shape.size(); ++i) {
        int64_t dim = shape[i];
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    const T* data = storage.data_ptr();
    size_t num_elements = storage.size();

    if (options.convert_endian && is_little_endian() != is_little_endian()) {
        std::vector<T> converted_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            converted_data[i] = swap_endian(data[i]);
        }

        file.write(reinterpret_cast<const char*>(converted_data.data()),
                   num_elements * sizeof(T));
    } else {
        file.write(reinterpret_cast<const char*>(data),
                   num_elements * sizeof(T));
    }
}

/**
 * @brief Save tensor data to a file in specified format
 *
 * @tparam T Element type
 * @param storage Storage to save
 * @param shape Shape of the tensor
 * @param path File path
 * @param options I/O options
 * @throws core::error::RuntimeError if file cannot be opened
 * @throws core::error::LogicError if format is not supported
 */
template <typename T>
void save(const storage::Storage<T>& storage, const shape::Shape& shape,
          const std::filesystem::path& path,
          const IOOptions& options = IOOptions{}) {
    FileFormat format = options.format;

    if (format == FileFormat::Auto) {
        format = detect_format(path);
    }

    switch (format) {
        case FileFormat::Binary:
            save_binary(storage, shape, path, options);
            break;

        case FileFormat::NPY: {
            std::ofstream file(path, std::ios::binary);

            if (!file) {
                throw core::error::RuntimeError(
                    "Failed to open file for writing: {}", path.string());
            }

            std::string dtype_str = get_npy_dtype<T>(is_little_endian());
            write_npy_header(file, shape, dtype_str, !options.row_major);

            const T* data = storage.data_ptr();
            size_t num_elements = storage.size();

            file.write(reinterpret_cast<const char*>(data),
                       num_elements * sizeof(T));
            break;
        }

        case FileFormat::CSV: {
            if (shape.size() < 1 || shape.size() > 2) {
                throw core::error::LogicError(
                    "CSV format supports only 1D or 2D tensors, got {}D",
                    shape.size());
            }

            std::ofstream file(path);

            if (!file) {
                throw core::error::RuntimeError(
                    "Failed to open file for writing: {}", path.string());
            }

            if (options.include_header) {
                write_csv_header(file, shape, options);
            }

            const T* data = storage.data_ptr();
            if (shape.size() == 1) {
                for (int64_t i = 0; i < shape[0]; ++i) {
                    file << format_element(data[i], options) << "\n";
                }
            } else {
                for (int64_t i = 0; i < shape[0]; ++i) {
                    for (int64_t j = 0; j < shape[1]; ++j) {
                        if (j > 0) {
                            file << options.delimiter;
                        }

                        size_t index = options.transpose ? j * shape[0] + i
                                                         : i * shape[1] + j;

                        file << format_element(data[index], options);
                    }

                    file << "\n";
                }
            }

            break;
        }

        case FileFormat::JSON: {
            std::ofstream file(path);

            if (!file) {
                throw core::error::RuntimeError(
                    "Failed to open file for writing: {}", path.string());
            }

            file << "{\n";
            file << "  \"dtype\": \"" << get_npy_dtype<T>(is_little_endian())
                 << "\",\n";
            file << "  \"shape\": [";

            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) {
                    file << ", ";
                }

                file << shape[i];
            }

            file << "],\n";
            file << "  \"data\": ";

            const T* data = storage.data_ptr();
            size_t num_elements = shape.numel();

            if (shape.size() <= 1) {
                file << "[";
                for (size_t i = 0; i < num_elements; ++i) {
                    if (i > 0) {
                        file << ", ";
                    }

                    file << format_element(data[i], options);
                }

                file << "]\n";
            } else if (shape.size() == 2) {
                file << "[\n";

                for (int64_t i = 0; i < shape[0]; ++i) {
                    file << "    [";

                    for (int64_t j = 0; j < shape[1]; ++j) {
                        if (j > 0) {
                            file << ", ";
                        }

                        size_t index = options.transpose ? j * shape[0] + i
                                                         : i * shape[1] + j;

                        file << format_element(data[index], options);
                    }

                    if (i < shape[0] - 1) {
                        file << "],\n";
                    } else {
                        file << "]\n";
                    }
                }

                file << "  ]\n";
            } else {
                file << "[";

                for (size_t i = 0; i < num_elements; ++i) {
                    if (i > 0) {
                        file << ", ";
                    }

                    file << format_element(data[i], options);
                }

                file << "]\n";
            }

            file << "}\n";
            break;
        }

        default:
            throw core::error::LogicError("Unsupported file format: {}",
                                          format_to_string(format));
    }
}

/**
 * @brief Save tensor data from shared storage
 *
 * @tparam T Element type
 * @param storage Shared storage to save
 * @param shape Shape of the tensor
 * @param path File path
 * @param options I/O options
 */
template <typename T>
void save(const storage::SharedStorage<T>& storage, const shape::Shape& shape,
          const std::filesystem::path& path,
          const IOOptions& options = IOOptions{}) {
    save(storage.storage(), shape, path, options);
}

/**
 * @brief Load tensor data from a binary file
 *
 * @tparam T Element type
 * @param path File path
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Loaded storage and shape
 * @throws core::error::RuntimeError if file cannot be opened
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> load_binary(
    const std::filesystem::path& path, const IOOptions& options = IOOptions{}) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw core::error::RuntimeError("Failed to open file for reading: {}",
                                        path.string());
    }

    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    std::vector<int64_t> dims(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&dims[i]), sizeof(int64_t));
    }

    shape::Shape shape(dims);
    size_t num_elements = shape.numel();
    storage::Storage<T> storage(num_elements);
    T* data = storage.data_ptr();
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));

    if (options.convert_endian && is_little_endian() != is_little_endian()) {
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = swap_endian(data[i]);
        }
    }

    return {storage, shape};
}

/**
 * @brief Load tensor data from a NPY file
 *
 * @tparam T Element type
 * @param path File path
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Loaded storage and shape
 * @throws core::error::RuntimeError if file cannot be opened
 * @throws core::error::LogicError if format is invalid
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> load_npy(
    const std::filesystem::path& path, const IOOptions& options = IOOptions{}) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw core::error::RuntimeError("Failed to open file for reading: {}",
                                        path.string());
    }

    auto [shape, dtype_str, fortran_order] = parse_npy_header(file);
    auto [type_idx, file_little_endian] = parse_npy_dtype(dtype_str);

    if (type_idx != std::type_index(typeid(T))) {
        throw core::error::LogicError(
            "Type mismatch: file has type {}, but requested type is {}",
            dtype_str, get_npy_dtype<T>(is_little_endian()));
    }

    size_t num_elements = shape.numel();
    storage::Storage<T> storage(num_elements);
    T* data = storage.data_ptr();
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));

    if (options.convert_endian && is_little_endian() != file_little_endian) {
        for (size_t i = 0; i < num_elements; ++i) {
            data[i] = swap_endian(data[i]);
        }
    }

    if (fortran_order && options.row_major) {
        // TODO: Implement transpose for N-D tensors
        if (shape.size() == 2) {
            int64_t rows = shape[0];
            int64_t cols = shape[1];

            storage::Storage<T> transposed_storage(num_elements);
            T* transposed_data = transposed_storage.data_ptr();

            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t j = 0; j < cols; ++j) {
                    transposed_data[i * cols + j] = data[j * rows + i];
                }
            }

            storage = transposed_storage;
        } else {
            throw core::error::LogicError(
                "Transposing {}D tensors is not implemented", shape.size());
        }
    }

    return {storage, shape};
}

/**
 * @brief Parse CSV data into a tensor
 *
 * @tparam T Element type
 * @param data CSV data as string
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Loaded storage and shape
 * @throws core::error::LogicError if CSV format is invalid
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> parse_csv(
    const std::string& data, const IOOptions& options = IOOptions{}) {
    std::istringstream stream(data);
    std::string line;
    std::vector<std::vector<T>> rows;

    if (options.include_header) {
        std::getline(stream, line);
    }

    size_t max_cols = 0;
    while (std::getline(stream, line)) {
        std::vector<T> row;
        std::istringstream line_stream(line);
        std::string cell;

        while (std::getline(line_stream, cell, options.delimiter)) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

            if (cell.empty()) {
                continue;
            }

            T value;

            try {
                if constexpr (std::is_same_v<T, float> ||
                              std::is_same_v<T, double>) {
                    value = static_cast<T>(std::stod(cell));
                } else if constexpr (std::is_integral_v<T>) {
                    value = static_cast<T>(std::stoll(cell));
                } else if constexpr (std::is_same_v<T, bool>) {
                    value = (cell == "true" || cell == "True" || cell == "1");
                } else {
                    throw core::error::LogicError(
                        "Unsupported type for CSV parsing");
                }
            } catch (const std::exception& e) {
                throw core::error::LogicError("Failed to parse CSV cell: {}",
                                              cell);
            }

            row.push_back(value);
        }

        if (!row.empty()) {
            max_cols = std::max(max_cols, row.size());
            rows.push_back(row);
        }
    }

    shape::Shape shape;
    storage::Storage<T> storage;

    if (rows.empty()) {
        shape = shape::Shape({0});
        storage = storage::Storage<T>(0);
    } else if (rows.size() == 1 && max_cols == 1) {
        shape = shape::Shape({1});
        storage = storage::Storage<T>(1);
        storage[0] = rows[0][0];
    } else if (rows.size() == 1) {
        shape = shape::Shape({static_cast<int64_t>(max_cols)});
        storage = storage::Storage<T>(max_cols);

        for (size_t i = 0; i < max_cols; ++i) {
            storage[i] = i < rows[0].size() ? rows[0][i] : T(0);
        }
    } else if (max_cols == 1) {
        shape = shape::Shape({static_cast<int64_t>(rows.size())});
        storage = storage::Storage<T>(rows.size());

        for (size_t i = 0; i < rows.size(); ++i) {
            storage[i] = rows[i][0];
        }
    } else {
        shape = shape::Shape({static_cast<int64_t>(rows.size()),
                              static_cast<int64_t>(max_cols)});
        storage = storage::Storage<T>(rows.size() * max_cols);

        for (size_t i = 0; i < rows.size(); ++i) {
            for (size_t j = 0; j < max_cols; ++j) {
                size_t index =
                    options.transpose ? j * rows.size() + i : i * max_cols + j;

                storage[index] = j < rows[i].size() ? rows[i][j] : T(0);
            }
        }
    }

    return {storage, shape};
}

/**
 * @brief Load tensor data from a CSV file
 *
 * @tparam T Element type
 * @param path File path
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Loaded storage and shape
 * @throws core::error::RuntimeError if file cannot be opened
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> load_csv(
    const std::filesystem::path& path, const IOOptions& options = IOOptions{}) {
    std::ifstream file(path);

    if (!file) {
        throw core::error::RuntimeError("Failed to open file for reading: {}",
                                        path.string());
    }

    std::ostringstream ss;
    ss << file.rdbuf();

    return parse_csv<T>(ss.str(), options);
}

/**
 * @brief Load tensor data from a file in specified format
 *
 * @tparam T Element type
 * @param path File path
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Loaded storage and shape
 * @throws core::error::RuntimeError if file cannot be opened
 * @throws core::error::LogicError if format is not supported
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> load(
    const std::filesystem::path& path, const IOOptions& options = IOOptions{}) {
    FileFormat format = options.format;

    if (format == FileFormat::Auto) {
        format = detect_format(path);
    }

    switch (format) {
        case FileFormat::Binary:
            return load_binary<T>(path, options);

        case FileFormat::NPY:
            return load_npy<T>(path, options);

        case FileFormat::CSV:
            return load_csv<T>(path, options);

        case FileFormat::JSON:
            // TODO: Implement JSON loader
            throw core::error::LogicError("JSON loading not implemented yet");

        default:
            throw core::error::LogicError("Unsupported file format: {}",
                                          format_to_string(format));
    }
}

/**
 * @brief Load tensor data into shared storage
 *
 * @tparam T Element type
 * @param path File path
 * @param options I/O options
 * @return std::pair<storage::SharedStorage<T>, shape::Shape> Loaded shared
 * storage and shape
 */
template <typename T>
std::pair<storage::SharedStorage<T>, shape::Shape> load_shared(
    const std::filesystem::path& path, const IOOptions& options = IOOptions{}) {
    auto [storage, shape] = load<T>(path, options);
    return {storage::SharedStorage<T>(std::move(storage)), shape};
}

/**
 * @brief Save a tensor to a string in CSV format
 *
 * @tparam T Element type
 * @param storage Storage to save
 * @param shape Shape of the tensor
 * @param options I/O options
 * @return std::string CSV representation
 * @throws core::error::LogicError if tensor shape is not supported
 */
template <typename T>
std::string to_csv_string(const storage::Storage<T>& storage,
                          const shape::Shape& shape,
                          const IOOptions& options = IOOptions{}) {
    if (shape.size() < 1 || shape.size() > 2) {
        throw core::error::LogicError(
            "CSV format supports only 1D or 2D tensors, got {}D", shape.size());
    }

    std::ostringstream ss;
    if (options.include_header) {
        write_csv_header(ss, shape, options);
    }

    const T* data = storage.data_ptr();
    if (shape.size() == 1) {
        for (int64_t i = 0; i < shape[0]; ++i) {
            ss << format_element(data[i], options) << "\n";
        }
    } else {
        for (int64_t i = 0; i < shape[0]; ++i) {
            for (int64_t j = 0; j < shape[1]; ++j) {
                if (j > 0) {
                    ss << options.delimiter;
                }

                size_t index =
                    options.transpose ? j * shape[0] + i : i * shape[1] + j;

                ss << format_element(data[index], options);
            }

            ss << "\n";
        }
    }

    return ss.str();
}

/**
 * @brief Parse tensor data from a string in specified format
 *
 * @tparam T Element type
 * @param data String data
 * @param format Format of the data
 * @param options I/O options
 * @return std::pair<storage::Storage<T>, shape::Shape> Parsed storage and shape
 * @throws core::error::LogicError if format is not supported
 */
template <typename T>
std::pair<storage::Storage<T>, shape::Shape> parse_string(
    const std::string& data, FileFormat format,
    const IOOptions& options = IOOptions{}) {
    switch (format) {
        case FileFormat::CSV:
            return parse_csv<T>(data, options);

        default:
            throw core::error::LogicError(
                "Unsupported format for string parsing: {}",
                format_to_string(format));
    }
}

}  // namespace brezel::tensor::io