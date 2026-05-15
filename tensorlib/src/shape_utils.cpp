#include "shape_utils.h"

#include "tensorlib.h"

std::optional<pi::tensorlib::Strides>
pi::tensorlib::shape_utils::ComputeViewStrides(const Shape &oldShape, const Strides &oldStrides, const Shape &newShape)
{
    // rank
    const size_t oldRank = oldShape.dims().size();
    const size_t newRank = newShape.dims().size();

    // zero-d tensors: just make all strides = 1
    if (oldRank == 0)
    {
        const std::vector<uint64_t> s(newRank, 1);
        return Strides{s};
    }

    // total # elements
    const size_t numel = oldShape.numel();

    // grab raw arrays
    const auto &oldDims = oldShape.dims();
    const auto &oldStridesVec = oldStrides.strides();
    const auto &newDims = newShape.dims();

    // zero-element fast paths
    if (numel == 0)
    {
        // if shapes identical, keep old strides
        if (oldShape.dims() == newShape.dims())
        {
            return oldStrides;
        }
        // otherwise, build a new contiguous stride
        std::vector<uint64_t> out(newRank);
        uint64_t st = 1;
        for (int i = static_cast<int>(newRank) - 1; i >= 0; --i)
        {
            out[i] = st;
            st *= newDims[i];
        }
        return Strides{out};
    }

    std::vector<uint64_t> newStrides(newRank);
    int viewD = static_cast<int>(newRank) - 1;
    uint64_t chunkBaseStride = oldStridesVec[oldRank - 1];
    uint64_t tensorNumel = 1, viewNumel = 1;

    for (int tensorD = static_cast<int>(oldRank) - 1; tensorD >= 0; --tensorD)
    {
        tensorNumel *= oldDims[tensorD];
        const bool isChunkEnd = (tensorD == 0) || (oldDims[tensorD - 1] != 1 &&
                                                   oldStridesVec[tensorD - 1] != chunkBaseStride * tensorNumel);
        if (isChunkEnd)
        {
            // carve off as many new dims as belong to this chunk
            while (viewD >= 0 && (viewNumel < tensorNumel || newDims[viewD] == 1))
            {
                newStrides[viewD] = chunkBaseStride * viewNumel;
                viewNumel *= newDims[viewD];
                --viewD;
            }
            if (viewNumel != tensorNumel)
            {
                return std::nullopt; // incompatible
            }
            // reset for next chunk
            if (tensorD > 0)
            {
                chunkBaseStride = oldStridesVec[tensorD - 1];
                tensorNumel = viewNumel = 1;
            }
        }
    }

    if (viewD != -1) // didn't consume all new dims
        return std::nullopt;

    return Strides{newStrides};
}

bool pi::tensorlib::shape_utils::IsBroadcastable(const Shape &shapeA, const Shape &shapeB)
{
    const auto &dimsA = shapeA.dims();
    const auto &dimsB = shapeB.dims();
    const size_t rankA = dimsA.size();
    const size_t rankB = dimsB.size();
    const size_t maxRank = std::max(rankA, rankB);

    for (size_t i = 0; i < maxRank; ++i)
    {
        const uint64_t dimA = (i < maxRank - rankA) ? 1 : dimsA[i - (maxRank - rankA)];
        const uint64_t dimB = (i < maxRank - rankB) ? 1 : dimsB[i - (maxRank - rankB)];
        if (dimA != dimB && dimA != 1 && dimB != 1)
        {
            return false;
        }
    }
    return true;
}

bool pi::tensorlib::shape_utils::IsRowMajorContiguous(const Shape &shape, const Strides &strides)
{
    const auto ndims = shape.ndims();
    if (ndims == 0)
    {
        return true;
    }

    uint64_t expected_stride = 1;
    for (int64_t dim = static_cast<int64_t>(ndims) - 1; dim >= 0; --dim)
    {
        if (strides[dim] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape[dim];
    }
    return true;
}

bool pi::tensorlib::shape_utils::IsRowMajorContiguous(const std::shared_ptr<RealTensor> &tensor)
{
    return IsRowMajorContiguous(tensor->shape(), tensor->strides());
}
