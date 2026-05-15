#include "module.h"

#include <ranges>

using namespace pi::tensorlib;

void BackwardContext::saveForBackward(const std::string &name, TraceTensor &tensor)
{
    saved.insert_or_assign(name, tensor);
}

void BackwardContext::release(OpGraph &graph)
{
    for (auto &tensor : saved | std::views::values)
    {
        if (tensor.isView())
        {
            continue;
        }
        if (tensor.retained())
        {
            continue;
        }
        if (!graph.hasTensor(tensor.id()))
        {
            continue;
        }
        graph.deleteTensor(tensor);
    }
    saved.clear();
}

bool BackwardContext::hasSaved(const std::string &name) const { return saved.contains(name); }

const TraceTensor &BackwardContext::getSaved(const std::string &name) const
{
    return saved.at(name);
}
