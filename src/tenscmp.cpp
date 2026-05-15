#include <argparse/argparse.hpp>
#include <safe_tensors.h>
#include <testing.h>
#include <tensorlib.h>

#include <iostream>
#include <set>
#include <string>

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("tenscmp");
    program.add_argument("lhs").help("Path to first safetensors file");
    program.add_argument("rhs").help("Path to second safetensors file");
    program.add_argument("-t", "--threshold")
        .default_value(1e-3)
        .scan<'g', double>()
        .help("Maximum allowed absolute difference");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto lhs_path = program.get<std::string>("lhs");
    const auto rhs_path = program.get<std::string>("rhs");
    const auto threshold = program.get<double>("--threshold");

    const auto lhs_tensors = pi::tensorlib::safetensors::Load(lhs_path);
    const auto rhs_tensors = pi::tensorlib::safetensors::Load(rhs_path);

    std::set<std::string> tensor_names{};
    for (const auto &entry : lhs_tensors)
    {
        tensor_names.insert(entry.first);
    }
    for (const auto &entry : rhs_tensors)
    {
        tensor_names.insert(entry.first);
    }

    bool success = true;
    for (const auto &name : tensor_names)
    {
        const auto lhs_it = lhs_tensors.find(name);
        const auto rhs_it = rhs_tensors.find(name);
        if (lhs_it == lhs_tensors.end() || rhs_it == rhs_tensors.end())
        {
            std::cerr << "Tensor \"" << name << "\" missing in "
                      << (lhs_it == lhs_tensors.end() ? "lhs" : "rhs") << " file" << std::endl;
            success = false;
            continue;
        }

        try
        {
            pi::tensorlib::testing::AssertSimilar(lhs_it->second, rhs_it->second, threshold);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Tensor \"" << name << "\" differs: " << ex.what() << std::endl;
            success = false;
        }
    }

    if (success)
    {
        std::cout << "PASS: tensors within threshold " << threshold << std::endl;
        return 0;
    }

    std::cerr << "FAIL: tensors differ beyond threshold " << threshold << std::endl;
    return 1;
}
