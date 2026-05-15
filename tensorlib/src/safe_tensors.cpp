#include "safe_tensors.h"
#include "stream_descriptor.h"
#include "tensorlib.h"

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

static uint64_t le_to_host(const uint64_t value)
{
    static constexpr uint16_t test = 1;
    static const bool is_little_endian = *reinterpret_cast<const uint8_t *>(&test) == 1;

    if (is_little_endian)
    {
        return value; // Already little endian
    }

    // Convert from little endian to big endian (host)
    return ((value & 0xFF00000000000000ULL) >> 56) | ((value & 0x00FF000000000000ULL) >> 40) |
           ((value & 0x0000FF0000000000ULL) >> 24) | ((value & 0x000000FF00000000ULL) >> 8) |
           ((value & 0x00000000FF000000ULL) << 8) | ((value & 0x0000000000FF0000ULL) << 24) |
           ((value & 0x000000000000FF00ULL) << 40) | ((value & 0x00000000000000FFULL) << 56);
}

static uint64_t host_to_le(const uint64_t value)
{
    // Same implementation as le_to_host since conversion is symmetric
    return le_to_host(value);
}

// Static data type conversion functions
static std::string DataTypeToSafeTensorsName(const pi::tensorlib::DataType data_type)
{
    switch (data_type)
    {
        case pi::tensorlib::DataType::UINT32:
            return "U32";
        case pi::tensorlib::DataType::UINT64:
            return "U64";
        case pi::tensorlib::DataType::BFLOAT16:
            return "BF16";
        case pi::tensorlib::DataType::FLOAT16:
            return "F16";
        case pi::tensorlib::DataType::FLOAT32:
            return "F32";
        default:
            throw std::runtime_error("Unsupported data type for safetensors");
    }
}

static pi::tensorlib::DataType DataTypeFromSafeTensorsName(const std::string &dtype_name)
{
    if (dtype_name == "U32")
    {
        return pi::tensorlib::DataType::UINT32;
    }
    if (dtype_name == "U64")
    {
        return pi::tensorlib::DataType::UINT64;
    }
    if (dtype_name == "BF16")
    {
        return pi::tensorlib::DataType::BFLOAT16;
    }
    if (dtype_name == "F16")
    {
        return pi::tensorlib::DataType::FLOAT16;
    }
    if (dtype_name == "F32")
    {
        return pi::tensorlib::DataType::FLOAT32;
    }
    throw std::runtime_error("Unsupported safetensors data type: " + dtype_name);
}

static uint64_t read_le_u64(std::ifstream &file)
{
    uint64_t v{};
    file.read(reinterpret_cast<char *>(&v), sizeof(v));
    v = le_to_host(v);
    return v;
}

namespace pi::tensorlib::safetensors
{
    void LoadInplace(const std::string &file_path,
                     const std::map<std::string, std::shared_ptr<RealTensor>> &tensors_out)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        size_t file_size = 0;
        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        if (file_size < 0)
            throw std::runtime_error("Failed to determine file size");
        file.seekg(0, std::ios::beg);

        if (file_size < sizeof(uint64_t))
        {
            throw std::runtime_error("File is too small to contain a valid safetensors file");
        }

        uint64_t header_size = read_le_u64(file);

        if (header_size > file_size)
        {
            throw std::runtime_error("Invalid header size in safetensors file");
        }

        // Read the header
        std::string header(header_size, '\0');
        file.read(header.data(), static_cast<std::streamsize>(header_size));

        if (file.gcount() != static_cast<std::streamsize>(header_size))
        {
            throw std::runtime_error("Failed to read complete header");
        }

        // Calculate start of data section
        uint64_t data_start = sizeof(uint64_t) + header_size;
        // Determine remaining bytes
        size_t byte_buffer_size = file_size - data_start;

        nlohmann::json json_header;
        try
        {
            json_header = nlohmann::json::parse(header);
        }
        catch (const nlohmann::json::parse_error &e)
        {
            throw std::runtime_error("Invalid safetensors file: \"" + file_path +
                                     "\", failed to parse header: " + e.what());
        }

        if (!json_header.is_object())
        {
            throw std::runtime_error("Invalid safetensors file: header JSON is not an object");
        }

        // Iterate over all tensors in the header
        for (const auto &[name, tensor_info] : json_header.items())
        {
            if (name == "__metadata__")
            {
                continue;
            }

            // Validation
            if (!tensors_out.contains(name))
            {
                throw std::runtime_error("Tensor " + name + " not found in output map");
            }
            auto &tensor = tensors_out.at(name);

            if (!tensor_info.contains("shape"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain a shape");
            }
            if (!tensor_info.contains("dtype"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain a dtype");
            }
            if (!tensor_info.contains("data_offsets"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain data offsets");
            }

            auto shape = tensor_info["shape"].get<std::vector<uint64_t>>();
            auto dtype = tensor_info["dtype"].get<std::string>();
            auto offsets = tensor_info["data_offsets"].get<std::vector<uint64_t>>();

            if (offsets.size() != 2)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; expected 2, got " + std::to_string(offsets.size()));
            }

            size_t begin = offsets[0];
            size_t end = offsets[1];

            if (end <= begin)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; end offset must be greater than begin offset");
            }
            if (begin > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; begin offset must be less than byte buffer size");
            }
            if (end > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; end offset must be less than byte buffer size");
            }

            // Validate tensor properties
            const Shape &tensor_shape = tensor->shape();
            const std::vector<uint64_t> &tensor_dims = tensor_shape.dims();

            if (tensor_dims.size() != shape.size())
            {
                throw std::runtime_error("Shape mismatch for tensor " + name + ": expected " +
                                         std::to_string(shape.size()) + " dimensions, got " +
                                         std::to_string(tensor_dims.size()));
            }

            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (tensor_dims[i] != shape[i])
                {
                    throw std::runtime_error("Shape mismatch for tensor " + name + " at dimension " +
                                             std::to_string(i));
                }
            }

            DataType tensor_dtype = tensor->dtype();
            DataType safe_tensors_dtype = DataTypeFromSafeTensorsName(dtype);

            if (tensor_dtype != safe_tensors_dtype)
            {
                throw std::runtime_error(
                    "Cannot load safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                    "\" has a different dtype than supplied by the user as the destination tensor; user supplied " +
                    GetDataTypeName(tensor_dtype) + ", but found " + GetDataTypeName(safe_tensors_dtype) + " in file");
            }

            // Read contents
            const size_t tensor_size = end - begin;

            if (tensor_size > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid size; expected " + std::to_string(tensor_size) + ", got " +
                                         std::to_string(byte_buffer_size));
            }

            const auto &storage = tensor->storage();
            size_t expected_size = tensor_shape.numel() * GetDataTypeSize(tensor_dtype);

            if (tensor_size != expected_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid size; expected " + std::to_string(expected_size) + ", got " +
                                         std::to_string(tensor_size));
            }

            if (const Device &device = tensor->device(); device.device_type != DeviceType::CPU)
            {
                throw std::runtime_error("Cannot read tensor " + name + " from file: " + file_path +
                                         "; tensor is not on CPU device");
            }

            allocator::DefaultAllocatorRegistry &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
            auto &cpu_allocator = allocator_registry.getAllocator(DeviceType::CPU);

            // Create temporary storage for file data
            auto temp_storage = Storage::CreateFor(tensor_dtype, tensor_shape, Device{DeviceType::CPU, 0}, false);
            void *temp_data = cpu_allocator.allocate(tensor_size, 0, false, 0, false);
            if (!temp_data)
            {
                throw std::runtime_error("Failed to allocate temporary memory for tensor " + name);
            }
            temp_storage->initialize(temp_data, &cpu_allocator);

            // Read data from file
            file.seekg(static_cast<std::streamsize>(begin + data_start), std::ios::beg);
            file.read(static_cast<char *>(temp_data), static_cast<std::streamsize>(tensor_size));

            if (!file)
            {
                std::free(temp_data);
                throw std::runtime_error("Failed to read tensor " + name + " from file: " + file_path);
            }

            // Copy data to the tensor's storage using CopyFrom
            storage->copyFrom(*temp_storage, pi::tensorlib::GpuStreamDescriptors::Main);

            // Clean up temporary storage
            std::free(temp_data);
        }
    }

    std::map<std::string, std::shared_ptr<RealTensor>> Load(const std::string &file_path, const bool pinned)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        size_t file_size = 0;
        file.seekg(0, std::ios::end);
        file_size = file.tellg();
        if (file_size < 0)
            throw std::runtime_error("Failed to determine file size");
        file.seekg(0, std::ios::beg);

        if (file_size < sizeof(uint64_t))
        {
            throw std::runtime_error("File is too small to contain a valid safetensors file");
        }

        uint64_t header_size = read_le_u64(file);

        if (header_size > file_size)
        {
            throw std::runtime_error("Invalid header size in safetensors file");
        }

        // Read the header
        std::string header(header_size, '\0');
        file.read(header.data(), static_cast<std::streamsize>(header_size));

        if (file.gcount() != static_cast<std::streamsize>(header_size))
        {
            throw std::runtime_error("Failed to read complete header");
        }

        // Calculate start of data section
        uint64_t data_start = sizeof(uint64_t) + header_size;
        // Determine remaining bytes
        size_t byte_buffer_size = file_size - data_start;

        nlohmann::json json_header;
        try
        {
            json_header = nlohmann::json::parse(header);
        }
        catch (const nlohmann::json::parse_error &e)
        {
            throw std::runtime_error("Invalid safetensors file: \"" + file_path +
                                     "\", failed to parse header: " + e.what());
        }

        if (!json_header.is_object())
        {
            throw std::runtime_error("Invalid safetensors file: header JSON is not an object");
        }

        // Iterate over all tensors in the header
        std::map<std::string, std::shared_ptr<RealTensor>> tensors_out;

        for (const auto &[name, tensor_info] : json_header.items())
        {
            if (name == "__metadata__")
            {
                continue;
            }

            if (!tensor_info.contains("shape"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain a shape");
            }
            if (!tensor_info.contains("dtype"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain a dtype");
            }
            if (!tensor_info.contains("data_offsets"))
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" does not contain data offsets");
            }

            auto shape_vec = tensor_info["shape"].get<std::vector<uint64_t>>();
            auto dtype = tensor_info["dtype"].get<std::string>();
            auto offsets = tensor_info["data_offsets"].get<std::vector<uint64_t>>();

            if (offsets.size() != 2)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; expected 2, got " + std::to_string(offsets.size()));
            }

            size_t begin = offsets[0];
            size_t end = offsets[1];

            if (end <= begin)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; end offset must be greater than begin offset");
            }
            if (begin > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; begin offset must be less than byte buffer size");
            }
            if (end > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid offsets; end offset must be less than byte buffer size");
            }

            // Create tensor
            const size_t tensor_size = end - begin;

            if (tensor_size > byte_buffer_size)
            {
                throw std::runtime_error("Invalid safetensors file: \"" + file_path + "\", tensor with key \"" + name +
                                         "\" has invalid size; expected " + std::to_string(tensor_size) + ", got " +
                                         std::to_string(byte_buffer_size));
            }

            DataType data_type = DataTypeFromSafeTensorsName(dtype);
            Shape shape(shape_vec);
            Strides strides(shape);
            Device device{DeviceType::CPU, 0};

            if (const size_t expected_size = shape.numel() * GetDataTypeSize(data_type); tensor_size != expected_size)
            {
                throw std::runtime_error("Internal error; tensor with key \"" + name +
                                         "\" does not have the expected size; expected " +
                                         std::to_string(expected_size) + ", got " + std::to_string(tensor_size));
            }

            // Create RealTensor
            auto tensor =
                std::make_shared<RealTensor>(shape, strides, data_type, device, false, 0, pinned);

            allocator::DefaultAllocatorRegistry &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
            auto &cpu_allocator = allocator_registry.getAllocator(DeviceType::CPU);

            // Allocate and initialize storage
            void *data_ptr = cpu_allocator.allocate(tensor_size, 0, pinned, 0, false);
            if (!data_ptr)
            {
                throw std::runtime_error("Failed to allocate memory for tensor " + name);
            }
            tensor->storage()->initialize(data_ptr, &cpu_allocator);

            // Read data from file
            file.seekg(static_cast<std::streamsize>(begin + data_start), std::ios::beg);
            file.read(static_cast<char *>(data_ptr), static_cast<std::streamsize>(tensor_size));

            if (!file)
            {
                std::free(data_ptr);
                throw std::runtime_error("Failed to read tensor " + name + " from file: " + file_path);
            }

            tensors_out.emplace(name, tensor);
        }

        return tensors_out;
    }

    void SaveToFile(const std::string &file_path, const std::map<std::string, std::shared_ptr<RealTensor>> &tensors_in)
    {
        nlohmann::json json_header;
        json_header["__metadata__"] = nlohmann::json::object();

        uint64_t data_offset = 0;
        for (const auto &[name, tensor] : tensors_in)
        {
            nlohmann::json entry;
            const Shape &shape = tensor->shape();
            entry["shape"] = shape.dims();
            entry["dtype"] = DataTypeToSafeTensorsName(tensor->dtype());

            // Calculate size in bytes
            uint64_t size_bytes = shape.numel() * GetDataTypeSize(tensor->dtype());
            entry["data_offsets"] = {data_offset, data_offset + size_bytes};
            json_header[name] = std::move(entry);
            data_offset += size_bytes;
        }

        std::string header_str = json_header.dump(); // compact by default
        uint64_t header_len = header_str.size();

        std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + file_path);
        }

        uint64_t header_len_le = ::host_to_le(header_len);
        out.write(reinterpret_cast<const char *>(&header_len_le), sizeof(header_len_le));
        out.write(header_str.data(), static_cast<std::streamsize>(header_len));

        for (const auto &[name, tensor] : tensors_in)
        {
            const auto &storage = tensor->storage();
            const Device &device = tensor->device();

            // If tensor is on GPU, copy to CPU first
            std::shared_ptr<Storage> cpu_storage = storage;
            if (device.device_type != DeviceType::CPU)
            {
                cpu_storage = storage->toCPU();
            }

            size_t size_bytes = tensor->shape().numel() * GetDataTypeSize(tensor->dtype());
            if (!cpu_storage)
            {
                throw std::runtime_error("Safetensors write failed: null storage for tensor '" + name +
                                         "' in file " + file_path);
            }
            if (cpu_storage->isFreed() || cpu_storage->dataptr() == nullptr)
            {
                throw std::runtime_error("Safetensors write failed: storage freed or null for tensor '" + name +
                                         "' in file " + file_path);
            }
            const size_t element_size = GetDataTypeSize(tensor->dtype());
            const uint64_t offset_bytes = tensor->storageOffset() * element_size;
            if (offset_bytes + size_bytes > cpu_storage->sizeBytes())
            {
                throw std::runtime_error("Safetensors write failed: tensor '" + name +
                                         "' exceeds storage bounds in file " + file_path);
            }
            out.write(static_cast<const char *>(cpu_storage->dataptr()) + offset_bytes,
                      static_cast<std::streamsize>(size_bytes));
            if (!out.good())
            {
                throw std::runtime_error("Failed while writing tensor '" + name + "' to safetensors file: " +
                                         file_path);
            }
        }
    }
} // namespace pi::tensorlib::safetensors
