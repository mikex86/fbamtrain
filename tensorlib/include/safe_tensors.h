#pragma once

#include <map>
#include <string>
#include <memory>

namespace pi::tensorlib
{
    class RealTensor;
    class Storage;
}

namespace pi::tensorlib::safetensors
{
    /**
     * Reads a safetensors file and writes the results into tensors_out.
     * Each key must be found in the file, else the function will throw an exception.
     * @param file_path the path to the safetensors file
     * @param tensors_out a map of tensor names to RealTensors to be filled
     * @throws std::runtime_error if the file cannot be read, or if any of the tensors are not found
     */
    void LoadInplace(const std::string &file_path, const std::map<std::string, std::shared_ptr<RealTensor>> &tensors_out);

    /**
     * Reads a safetensors file and returns a map of tensor names to RealTensors.
     * @param file_path the path to the safetensors file
     * @return a map of tensor names to RealTensors
     */
    std::map<std::string, std::shared_ptr<RealTensor>> Load(const std::string &file_path, bool pinned = false);

    /**
     * Writes a safetensors file from the tensors in tensors_in.
     * Each tensor will be written to the file with the name as the key in the map.
     * @param file_path the path to the safetensors file
     * @param tensors_in a map of tensor names to RealTensors to be written
     * @throws std::runtime_error if the file cannot be written, or if any of the tensors are not found
     */
    void SaveToFile(const std::string &file_path, const std::map<std::string, std::shared_ptr<RealTensor>> &tensors_in);

}
