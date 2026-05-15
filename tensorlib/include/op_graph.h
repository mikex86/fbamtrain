#pragma once

#include "allocator.h"

#include <any>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stream_descriptor.h"
#include "tensorlib.h"

#include <optional>

namespace pi::tensorlib
{
    enum class OpType
    {
        CREATE_TENSOR,
        DELETE_TENSOR,
        DEVICE_COPY,

        CAST,

        FILL_ZEROS,
        FILL_UNIFORM,
        FILL_NORMAL,
        FILL_CONSTANT,

        MATMUL,
        PLUS,
        MUL,
        DIV,
        SQRT,
        ACT_FN,
        ACT_FN_BWD,

        MHA_ATTN_FWD,
        MHA_ATTN_BWD_PRE,
        MHA_ATTN_BWD,
        LAYER_NORM_FWD,
        RMS_NORM_FWD,
        RMS_NORM_BWD,
        AVG_POOL1D,
        AVG_POOL2D,
        AVG_POOL2D_BWD,
        CONV2D,
        CONV2D_DGRAD,
        CONV2D_WGRAD,
        MEAN,
        REDUCE_SUM,
        GATHER,
        CROSS_ENTROPY_ON_TARGETS,
        CROSS_ENTROPY_ON_TARGETS_BWD,
        REDUCE_SUM_PARTIAL,
        OPTIMIZER_SGD,
        OPTIMIZER_ADAMW,

        VIEW,
        TRANSPOSE,
        CONTIGUOUS,
        SPLIT,
        AT,

        // misc ops
        LSTM_CELL_FWD,
        LSTM_CELL_RECOMPUTE,
        LSTM_CELL_BWD,

        // stream sync ops
        RECORD_EVENT,
        AWAIT_EVENT,

        // profiler ops
        BEGIN_GPUTX_RANGE,
        END_GPUTX_RANGE,

        // Custom op type for user-defined operations
        // should be paired with attributes the user-defined passes can react
        // to then correctly employ kernels to replace the custom op.
        // "custom_op_name" should be used in attributes to differentiate between different types of custom operations.
        CUSTOM_OP
    };

    struct OperationEntry
    {
        OpType type;
        std::vector<TraceTensor> inputs;
        std::vector<TraceTensor> outputs;
        bool is_useful{true};
        std::unordered_map<std::string, std::any> attributes;
        
        /**
         * If set, specifies the GPU stream on which the operation is executed.
         * For cpu operations, we consider these to run on the "main" stream.
         */
        GpuStreamDescriptor gpu_stream_desc = GpuStreamDescriptors::Main;
    };

    struct GraphInputDescriptor
    {
        std::string name;
        TraceTensor tensor;
    };

    class OpGraphGpuTxRange
    {
        OpGraph &parent_graph;
        std::string range_name;

        OpGraphGpuTxRange(OpGraph &parent_graph, const std::string &range_name);

      public:
        [[nodiscard]] static OpGraphGpuTxRange Start(OpGraph &parent_graph, const std::string &range_name);

        ~OpGraphGpuTxRange();
    };

    class OpGraph
    {
      public:
        struct GpuEventHandle
        {
            size_t id{};
            int device_ordinal{};
        };

        explicit OpGraph(const std::vector<GraphInputDescriptor> &input_descriptors,
                         const std::vector<GraphInputDescriptor> &parameter_descriptors);

        void recordOperation(const OperationEntry &entry);

        [[nodiscard]] TraceTensor createTensor(const std::vector<uint64_t> &dims, DataType data_type, Device device,
                                               const GpuStreamDescriptor &stream_desc, bool pinned);

        void createTensor(TraceTensor &tensor);

        void deleteTensor(TraceTensor &tensor);

        [[nodiscard]] bool hasTensor(uint64_t id) const;

        [[nodiscard]] OpGraphGpuTxRange createGpuTxRange(const std::string &range_name);

        void finalize();

        [[nodiscard]] GpuEventHandle createGpuEvent(const Device &device);
        
        void deleteGpuEvent(const GpuEventHandle &handle);

        void recordGpuEvent(const GpuEventHandle &handle, const GpuStreamDescriptor &stream_desc);

        void awaitGpuEvent(const GpuEventHandle &handle, const GpuStreamDescriptor &stream_desc);

        [[nodiscard]] const std::vector<GraphInputDescriptor> &getInputDescriptors() const;

        [[nodiscard]] const std::vector<GraphInputDescriptor> &getParameterDescriptors() const;

        [[nodiscard]] const std::vector<OperationEntry> &getEntries() const;

        [[nodiscard]] size_t numEvents() const { return event_alive_.size(); }

      private:
        std::vector<GraphInputDescriptor> input_descriptors_;
        std::vector<GraphInputDescriptor> parameter_descriptors_;
        std::vector<OperationEntry> entries_;
        std::vector<bool> event_alive_{};
        std::unordered_set<uint64_t> created_tensor_ids_{};
        std::unordered_map<uint64_t, GpuStreamDescriptor> last_tensor_stream_descs_{};
    };

    [[nodiscard]] inline std::string GetOpTypeName(const OpType &type)
    {
        switch (type)
        {
            case OpType::CREATE_TENSOR:
                return "CREATE_TENSOR";
            case OpType::DELETE_TENSOR:
                return "DELETE_TENSOR";
            case OpType::FILL_ZEROS:
                return "FILL_ZEROS";
            case OpType::FILL_UNIFORM:
                return "FILL_UNIFORM";
            case OpType::FILL_NORMAL:
                return "FILL_NORMAL";
            case OpType::FILL_CONSTANT:
                return "FILL_CONSTANT";
            case OpType::MATMUL:
                return "MATMUL";
            case OpType::PLUS:
                return "PLUS";
            case OpType::MUL:
                return "MUL";
            case OpType::DIV:
                return "DIV";
            case OpType::SQRT:
                return "SQRT";
            case OpType::ACT_FN:
                return "ACT_FN";
            case OpType::ACT_FN_BWD:
                return "ACT_FN_BWD";
            case OpType::DEVICE_COPY:
                return "DEVICE_COPY";
            case OpType::CAST:
                return "CAST";
            case OpType::MHA_ATTN_FWD:
                return "MHA_ATTN_FWD";
            case OpType::MHA_ATTN_BWD_PRE:
                return "MHA_ATTN_BWD_PRE";
            case OpType::MHA_ATTN_BWD:
                return "MHA_ATTN_BWD";
            case OpType::LAYER_NORM_FWD:
                return "LAYER_NORM_FWD";
            case OpType::RMS_NORM_FWD:
                return "RMS_NORM_FWD";
            case OpType::RMS_NORM_BWD:
                return "RMS_NORM_BWD";
            case OpType::AVG_POOL1D:
                return "AVG_POOL1D";
            case OpType::AVG_POOL2D:
                return "AVG_POOL2D";
            case OpType::AVG_POOL2D_BWD:
                return "AVG_POOL2D_BWD";
            case OpType::CONV2D:
                return "CONV2D";
            case OpType::CONV2D_DGRAD:
                return "CONV2D_DGRAD";
            case OpType::CONV2D_WGRAD:
                return "CONV2D_WGRAD";
            case OpType::MEAN:
                return "MEAN";
            case OpType::REDUCE_SUM:
                return "REDUCE_SUM";
            case OpType::GATHER:
                return "GATHER";
            case OpType::CROSS_ENTROPY_ON_TARGETS:
                return "CROSS_ENTROPY_ON_TARGETS";
            case OpType::CROSS_ENTROPY_ON_TARGETS_BWD:
                return "CROSS_ENTROPY_ON_TARGETS_BWD";
            case OpType::REDUCE_SUM_PARTIAL:
                return "REDUCE_SUM_PARTIAL";
            case OpType::OPTIMIZER_SGD:
                return "OPTIMIZER_SGD";
            case OpType::OPTIMIZER_ADAMW:
                return "OPTIMIZER_ADAMW";
            case OpType::VIEW:
                return "VIEW";
            case OpType::TRANSPOSE:
                return "TRANSPOSE";
            case OpType::CONTIGUOUS:
                return "CONTIGUOUS";
            case OpType::SPLIT:
                return "SPLIT";
            case OpType::AT:
                return "AT";
            case OpType::LSTM_CELL_FWD:
                return "LSTM_CELL_FWD";
            case OpType::LSTM_CELL_RECOMPUTE:
                return "LSTM_CELL_RECOMPUTE";
            case OpType::LSTM_CELL_BWD:
                return "LSTM_CELL_BWD";
            case OpType::RECORD_EVENT:
                return "RECORD_EVENT";
            case OpType::AWAIT_EVENT:
                return "AWAIT_EVENT";
            case OpType::BEGIN_GPUTX_RANGE:
                return "BEGIN_GPUTX_RANGE";
            case OpType::END_GPUTX_RANGE:
                return "END_GPUTX_RANGE";
            case OpType::CUSTOM_OP:
                return "CUSTOM_OP";
            default:
                return "UNKNOWN_OP_TYPE";
        }
    }
} // namespace pi::tensorlib
