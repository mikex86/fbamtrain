#pragma once

#include "execution_plan.h"
#include "op_graph.h"

class MatmulFusePass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class DivAddFusePass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class MatmulImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class FillZerosImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class FillConstantImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class FillUniformImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class FillNormalImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class LayerNormImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class RmsNormImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class AvgPool1dImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class AvgPool2dImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class AvgPool2dBwdImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class Conv2dImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class MeanImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class ReduceSumImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class FuseMulReducePass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class CrossEntropyOnTargetsImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class BuildCellEmbedPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class BuildCellEmbedBwdPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class MhaAttentionImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class GatherImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class CastImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class ContiguousImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class TrailingBroadcastAddImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class TrailingBroadcastMulImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class DivImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class SqrtImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class ActImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class LstmCellImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class LstmCellBwdImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class ReduceSumPartialImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};

class OptimizerImplPass final : public pi::tensorlib::passes::CompilerPass
{
  public:
    void transform(pi::tensorlib::ExecutionPlan &execution_plan) override;
};
