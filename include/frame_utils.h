#pragma once

#include <cstddef>
#include <cstdint>

/// Codepoint state + Foreground Color + Background Color = 3 uint32 values per cell
#define NUM_FRAME_CHANNELS 3

#define CODE_POINT_CHANNEL_IDX 0
#define FG_COLOR_CHANNEL_IDX 1
#define BG_COLOR_CHANNEL_IDX 2

struct TerminalFrame;

namespace fbamtrain::frameutils
{
    void PrepareCellStates(const TerminalFrame &frame, const size_t seq_idx, const size_t rows, const size_t cols,
                           uint32_t max_code_point, uint32_t *cell_states_data);
}
