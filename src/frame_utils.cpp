#include "frame_utils.h"

#include <termstreamxz/termstream.h>

void fbamtrain::frameutils::PrepareCellStates(const TerminalFrame &frame, const size_t seq_idx, const size_t rows,
                                              const size_t cols, const uint32_t max_code_point,
                                              uint32_t *cell_states_data)
{
    const size_t frame_size = rows * cols;
    const uint32_t max_code_point_index = (max_code_point > 0) ? (max_code_point - 1) : 0;
#pragma omp simd
    // prepare cell states on cpu
    for (size_t i = 0; i < frame_size; i++)
    {
        const auto &cell = frame.cells[i];

        uint32_t cp = cell.codepoint;
        if (cp > max_code_point_index)
        {
            cp = max_code_point_index;
        }
        const uint32_t fg = cell.fg_r << 16 | cell.fg_g << 8 | cell.fg_b;
        const uint32_t bg = cell.bg_r << 16 | cell.bg_g << 8 | cell.bg_b;

        cell_states_data[seq_idx * frame_size * NUM_FRAME_CHANNELS + i * NUM_FRAME_CHANNELS + CODE_POINT_CHANNEL_IDX] =
            cp;
        cell_states_data[seq_idx * frame_size * NUM_FRAME_CHANNELS + i * NUM_FRAME_CHANNELS + FG_COLOR_CHANNEL_IDX] =
            fg;
        cell_states_data[seq_idx * frame_size * NUM_FRAME_CHANNELS + i * NUM_FRAME_CHANNELS + BG_COLOR_CHANNEL_IDX] =
            bg;
    }
}
