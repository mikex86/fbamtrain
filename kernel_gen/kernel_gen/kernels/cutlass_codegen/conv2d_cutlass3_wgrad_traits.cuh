#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

#include <cutlass/kernel_hardware_info.h>
#include <cutlass/conv/collective/collective_builder.hpp>
#include <cutlass/conv/convnd_problem_shape.hpp>
#include <cutlass/conv/kernel/conv_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

namespace fbamtrain::kernel_gen::cutlass3_conv2d
{
template <cutlass::conv::Operator ConvOp_, typename ArchTag_, typename Element_, typename ElementAccumulator_,
          int TileM, int TileN, int TileK, int ClusterM, int ClusterN, int ClusterK, int AlignmentA, int AlignmentB>
struct Conv2dTraits
{
    static constexpr cutlass::conv::Operator ConvOp = ConvOp_;
    using ArchTag = ArchTag_;
    using Element = Element_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementAccumulator_;

    using TileShapeMNK = cute::conditional_t<
        ConvOp == cutlass::conv::Operator::kWgrad,
        cute::Shape<cute::Int<TileM>, cute::Shape<cute::Int<TileN>>, cute::Shape<cute::Int<TileK>>>,
        cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Shape<cute::Int<TileK>>>>;
    using ClusterShapeMNK = cute::Shape<cute::Int<ClusterM>, cute::Int<ClusterN>, cute::Int<ClusterK>>;
    using OutputLayout = cute::conditional_t<ConvOp == cutlass::conv::Operator::kWgrad,
                                             cutlass::layout::TensorKCSR, cutlass::layout::TensorNHWC>;
    using EpilogueSchedule = cute::conditional_t<ConvOp == cutlass::conv::Operator::kWgrad,
                                                 cutlass::epilogue::NoSmemWarpSpecialized,
                                                 cutlass::epilogue::TmaWarpSpecialized>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp, TileShapeMNK, ClusterShapeMNK,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, Element,
        OutputLayout, AlignmentA, Element, OutputLayout, AlignmentA, EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp, ConvOp, Element,
        cutlass::layout::TensorNHWC, AlignmentA, Element, cutlass::layout::TensorNHWC, AlignmentB,
        ElementAccumulator, TileShapeMNK, ClusterShapeMNK,
        cutlass::conv::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;

    using ProblemShape = cutlass::conv::ConvProblemShape<CollectiveMainloop::DispatchPolicy::ConvOp,
                                                         CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;

    using ConvKernel = cutlass::conv::kernel::ConvUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Params = typename ConvKernel::Params;
    using Arguments = typename ConvKernel::Arguments;

    static ProblemShape make_problem_shape(const int32_t n, const int32_t h, const int32_t w, const int32_t c,
                                           const int32_t k, const int32_t r, const int32_t s, const int32_t pad_h,
                                           const int32_t pad_w, const int32_t stride_h, const int32_t stride_w,
                                           const int32_t dilation_h, const int32_t dilation_w, const int32_t groups)
    {
        const int32_t group_count = groups > 0 ? groups : 1;
        return ProblemShape(cutlass::conv::Mode::kCrossCorrelation, {n, h, w, c}, {k, r, s, c / group_count},
                            {pad_h, pad_w}, {pad_h, pad_w}, {stride_h, stride_w}, {dilation_h, dilation_w},
                            group_count);
    }

    static Arguments make_arguments(ProblemShape const &problem_shape, Element const *ptr_a, Element const *ptr_b,
                                    Element *ptr_c, Element *ptr_d, const ElementCompute alpha,
                                    const ElementCompute beta, cutlass::KernelHardwareInfo const &hw_info)
    {
        auto stride_c = typename ConvKernel::StrideC{};
        auto stride_d = typename ConvKernel::StrideD{};
        if constexpr (ConvOp == cutlass::conv::Operator::kWgrad)
        {
            stride_c = cutlass::make_cute_packed_stride(typename ConvKernel::StrideC{}, problem_shape.shape_C,
                                                        problem_shape.stride_C, ConvOp);
            stride_d = cutlass::make_cute_packed_stride(typename ConvKernel::StrideD{}, problem_shape.shape_C,
                                                        problem_shape.stride_C, ConvOp);
        }
        else
        {
            cute::for_each(cute::make_seq<cute::rank<0>(typename ConvKernel::StrideC{})>{}, [&](auto i)
                           { cute::get<0, i>(stride_c) = problem_shape.stride_C[ProblemShape::RankT - 2 - i]; });
            cute::for_each(cute::make_seq<cute::rank<0>(typename ConvKernel::StrideD{})>{}, [&](auto i)
                           { cute::get<0, i>(stride_d) = problem_shape.stride_C[ProblemShape::RankT - 2 - i]; });
        }

        typename ConvKernel::MainloopArguments mainloop_args{ptr_a, ptr_b};
        typename ConvKernel::EpilogueArguments epilogue_args{{}, ptr_c, stride_c, ptr_d, stride_d};
        typename ConvKernel::TileScheduler::Arguments scheduler_args{};
        typename ConvKernel::Arguments args{problem_shape, mainloop_args, epilogue_args, hw_info, scheduler_args};
        args.epilogue.thread.alpha = alpha;
        args.epilogue.thread.beta = beta;
        return args;
    }
};

template <typename ArchTag, typename Element, typename ElementAccumulator, int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK, int AlignmentA, int AlignmentB>
using FpropTraits = Conv2dTraits<cutlass::conv::Operator::kFprop, ArchTag, Element, ElementAccumulator, TileM,
                                TileN, TileK, ClusterM, ClusterN, ClusterK, AlignmentA, AlignmentB>;

template <typename ArchTag, typename Element, typename ElementAccumulator, int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK, int AlignmentA, int AlignmentB>
using DgradTraits = Conv2dTraits<cutlass::conv::Operator::kDgrad, ArchTag, Element, ElementAccumulator, TileM,
                                TileN, TileK, ClusterM, ClusterN, ClusterK, AlignmentA, AlignmentB>;

template <typename ArchTag, typename Element, typename ElementAccumulator, int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK, int AlignmentA, int AlignmentB>
using WgradTraits = Conv2dTraits<cutlass::conv::Operator::kWgrad, ArchTag, Element, ElementAccumulator, TileM,
                                TileN, TileK, ClusterM, ClusterN, ClusterK, AlignmentA, AlignmentB>;
} // namespace fbamtrain::kernel_gen::cutlass3_conv2d
