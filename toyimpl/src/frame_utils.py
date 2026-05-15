import torch
import triton
from pytermstreamxz import TerminalFrame


def print_frame(frame: TerminalFrame):
    width = frame.width
    height = frame.height
    for y in range(height):
        for x in range(width):
            cell = frame.get_cell(x, y)
            char = chr(cell.codepoint)
            print(char, end="")
        print()
    print()