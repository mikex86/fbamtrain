import argparse
import collections
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import init_utils
from safetensors.torch import save_file, load_file

from rand import Random
from init_utils import init_lstm_streaming_weights

import numpy as np

from pytermstreamxz import FileInputStream, TermInflateStream, TerminalFrame
from model import FrameHeadModule
from tokenizer import Tokenizer, load_tokenizer, TokenizerHandle

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

NUM_FRAME_CHANNELS = 3
CODE_POINT_CHANNEL_IDX = 0
FG_COLOR_CHANNEL_IDX = 1
BG_COLOR_CHANNEL_IDX = 2


def _should_emit_metrics_json() -> bool:
    return os.getenv("FBAMTRAIN_METRICS_JSON", "") not in ("", "0")


def _emit_metrics_json(
    step: int,
    train_loss: Optional[float] = None,
    step_time_sec: Optional[float] = None,
    validation_loss: Optional[float] = None,
) -> None:
    if not _should_emit_metrics_json():
        return
    payload = {"step": int(step)}
    if train_loss is not None:
        payload["loss/train"] = float(train_loss)
    if step_time_sec is not None:
        payload["step_time_sec"] = float(step_time_sec)
    if validation_loss is not None:
        payload["loss/validation"] = float(validation_loss)
    print(json.dumps(payload, separators=(",", ":")), flush=True)


@dataclass
class TrainDataConfig:
    recordings_directory_path: str
    iteration_seed: int


@dataclass
class ValidationDataConfig:
    recordings_directory_path: str
    iteration_seed: int


@dataclass
class FbamModelConfiguration:
    frame_cols: int
    frame_rows: int

    n_embed: int
    n_layer: int
    n_head: int
    bias: bool
    downsample_blocks: int
    max_code_point: int
    rms_norm_eps: float

    model_init_seed: int = 42
    dtype: str = "fp16"
    downsample_conv_mode: str = "dilated"
    downsample_conv_dilation: int = 2
    use_fp16_accumulation: bool = True
    streaming_chunk_size: int = 1
    recompute_interval: int = 1
    frame_head_reduction_strategy: str = "last_pos"


@dataclass
class OptimizerConfiguration:
    type: str
    learning_rate: float
    muon_learning_rate: Optional[float]
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    momentum: float
    nesterov: bool


@dataclass
class CheckpointingConfiguration:
    checkpoint_interval: int
    checkpoint_directory: str
    resume_behavior: str


@dataclass
class RunConfiguration:
    micro_batch_size: int
    total_batch_size: int
    train_sequence_length: int
    validation_sequence_length: int
    validation_interval: int
    enable_validation: bool
    max_training_steps: int
    frame_cols: int
    frame_rows: int
    train_data_config: TrainDataConfig
    validation_data_config: ValidationDataConfig
    model_config: FbamModelConfiguration
    optimizer_config: OptimizerConfiguration
    checkpointing: CheckpointingConfiguration
    tokenizer_file_path: str

    @staticmethod
    def from_json_file(path: str) -> "RunConfiguration":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)

        train_dc = TrainDataConfig(
            recordings_directory_path=j["train_data_config"]["recordings_directory_path"],
            iteration_seed=int(j["train_data_config"].get("iteration_seed", 0)),
        )
        valid_dc = ValidationDataConfig(
            recordings_directory_path=j["validation_data_config"]["recordings_directory_path"],
            iteration_seed=int(j["validation_data_config"].get("iteration_seed", 0)),
        )
        model_c = FbamModelConfiguration(
            n_embed=int(j["model_config"]["n_embed"]),
            frame_cols=int(j["model_config"]["frame_cols"]),
            frame_rows=int(j["model_config"]["frame_rows"]),
            model_init_seed=int(j["model_config"].get("model_init_seed", 42)),
            n_layer=int(j["model_config"]["n_layer"]),
            n_head=int(j["model_config"]["n_head"]),
            bias=bool(j["model_config"]["bias"]),
            downsample_blocks=int(j["model_config"]["downsample_blocks"]),
            max_code_point=int(j["model_config"]["max_code_point"]),
            rms_norm_eps=float(j["model_config"]["rms_norm_eps"]),
            dtype=str(j["model_config"].get("dtype", "fp16")).lower(),
            downsample_conv_mode=str(j["model_config"].get("downsample_conv_mode", "dilated")),
            downsample_conv_dilation=int(j["model_config"].get("downsample_conv_dilation", 2)),
            use_fp16_accumulation=bool(j["model_config"].get("use_fp16_accumulation", True)),
            streaming_chunk_size=int(j["model_config"]["streaming_chunk_size"]),
            recompute_interval=int(j["model_config"]["recompute_interval"]),
            frame_head_reduction_strategy=str(
                j["model_config"].get("frame_head_reduction_strategy", "last_pos")
            ).lower(),
        )
        if model_c.dtype not in ("fp16", "bf16"):
            raise ValueError(f"Unsupported dtype: {model_c.dtype}. Expected 'fp16' or 'bf16'.")
        if model_c.dtype == "bf16" and model_c.use_fp16_accumulation:
            raise ValueError("use_fp16_accumulation cannot be true when dtype is bf16.")
        if model_c.frame_head_reduction_strategy not in ("mean", "last_pos"):
            raise ValueError(
                "frame_head_reduction_strategy must be 'mean' or 'last_pos'. "
                f"Got '{model_c.frame_head_reduction_strategy}'."
            )

        optim_j = j["optimizer"]
        optimizer_c = OptimizerConfiguration(
            type=str(optim_j["type"]),
            learning_rate=float(optim_j["learning_rate"]),
            muon_learning_rate=(
                None if optim_j.get("muon_learning_rate") is None else float(optim_j["muon_learning_rate"])
            ),
            weight_decay=float(optim_j["weight_decay"]),
            beta1=float(optim_j["beta1"]),
            beta2=float(optim_j["beta2"]),
            eps=float(optim_j["eps"]),
            momentum=float(optim_j["momentum"]),
            nesterov=bool(optim_j["nesterov"]),
        )

        checkpoint_j = j.get("checkpointing")
        if checkpoint_j is None:
            raise ValueError("checkpointing must be specified in the run configuration")

        checkpoint_interval = int(checkpoint_j.get("checkpoint_interval", 0))
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")

        checkpoint_directory = str(checkpoint_j.get("checkpoint_directory", "")).strip()
        if not checkpoint_directory:
            raise ValueError("checkpoint_directory must be provided")

        resume_behavior = str(checkpoint_j.get("resume_behavior", "")).lower()
        if resume_behavior not in ("load_latest", "none", "disabled"):
            raise ValueError("resume_behavior must be 'load_latest' or 'none'")

        if resume_behavior == "disabled":
            resume_behavior = "none"

        checkpoint_c = CheckpointingConfiguration(
            checkpoint_interval=checkpoint_interval,
            checkpoint_directory=checkpoint_directory,
            resume_behavior=resume_behavior,
        )

        micro_batch_size = int(j["micro_batch_size"])
        total_batch_size = int(j["total_batch_size"])
        if total_batch_size <= 0:
            raise RuntimeError("total_batch_size must be greater than zero")
        if total_batch_size % micro_batch_size != 0:
            raise RuntimeError("total_batch_size must be a multiple of micro_batch_size")

        return RunConfiguration(
            micro_batch_size=micro_batch_size,
            total_batch_size=total_batch_size,
            train_sequence_length=int(j["train_sequence_length"]),
            validation_sequence_length=int(j["validation_sequence_length"]),
            validation_interval=int(j.get("validation_interval", 0)),
            enable_validation=bool(j.get("enable_validation", True)),
            max_training_steps=int(j["max_training_steps"]),
            frame_cols=int(j["frame_cols"]),
            frame_rows=int(j["frame_rows"]),
            train_data_config=train_dc,
            validation_data_config=valid_dc,
            model_config=model_c,
            optimizer_config=optimizer_c,
            checkpointing=checkpoint_c,
            tokenizer_file_path=str(j["tokenizer_file_path"]),
        )


class BatchIterator:
    def __init__(self, streams: List[TermInflateStream]):
        self.streams = streams


class _PooledStream:
    def __init__(self, path: str):
        self.path = path
        self.file_stream = FileInputStream(path)
        self.inflate_stream = TermInflateStream(self.file_stream)


class RecordingDatasetIterator:

    def __init__(self, recordings_folder_path: str, batch_size: int, sequence_length: int, seed: int):
        if not os.path.exists(recordings_folder_path) or not os.path.isdir(recordings_folder_path):
            raise RuntimeError(f"Recordings folder path does not exist or is not a directory: {recordings_folder_path}")

        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)

        self._recording_paths: List[str] = []

        for entry in sorted(os.listdir(recordings_folder_path)):
            if entry.endswith(".texz"):
                full = os.path.join(recordings_folder_path, entry)
                self._recording_paths.append(full)

        if not self._recording_paths:
            raise RuntimeError(f"No .texz recordings found in folder: {recordings_folder_path}")

        self._pool: Dict[str, List[_PooledStream]] = {path: [] for path in self._recording_paths}
        self._in_use: List[_PooledStream] = []
        self._rng = Random(self.seed)
        self._restored_prefetch: Optional[List[Tuple[int, int]]] = None

    def _release_in_use(self) -> None:
        for handle in self._in_use:
            self._pool[handle.path].append(handle)
        self._in_use.clear()

    def _acquire(self, path: str) -> TermInflateStream:
        pool = self._pool[path]
        if pool:
            handle = pool.pop()
        else:
            handle = _PooledStream(path)
        self._in_use.append(handle)
        return handle.inflate_stream

    def next_batch(self) -> BatchIterator:
        self._release_in_use()
        streams = []
        if self._restored_prefetch is not None:
            for stream_idx, start_position in self._restored_prefetch:
                if stream_idx < 0 or stream_idx >= len(self._recording_paths):
                    raise RuntimeError(f"Checkpoint file index out of range for dataset iterator: {stream_idx}")
                stream = self._acquire(self._recording_paths[stream_idx])
                stream.seek(start_position)
                streams.append(stream)
            self._restored_prefetch = None
            return BatchIterator(streams)

        n = len(self._recording_paths)
        for _ in range(self.batch_size):
            idx = self._rng.randint(0, n)
            stream = self._acquire(self._recording_paths[idx])
            num_frames = stream.get_total_num_frames()
            if num_frames <= 0:
                start_position = 0
            else:
                max_start = max(0, num_frames - self.sequence_length)
                start_position = self._rng.randint(0, max_start + 1)
            stream.seek(start_position)
            streams.append(stream)
        return BatchIterator(streams)

    def checkpoint_state(self) -> int:
        return int(self._rng.get_state())

    def restore_state(self, rng_state: int) -> None:
        self._rng.set_state(int(rng_state))
        self._release_in_use()
        self._restored_prefetch = None

    def restore_state_from_checkpoint(
        self,
        rng_state: int,
        prepared_indices: Optional[torch.Tensor],
        prepared_start_positions: Optional[torch.Tensor],
        prefetch_buffer_index: Optional[int] = None,
    ) -> None:
        self.restore_state(rng_state)

        if prefetch_buffer_index is not None and prefetch_buffer_index not in (0, 1):
            raise RuntimeError("Checkpoint prefetch_buffer_index must be 0 or 1.")

        if prepared_indices is None and prepared_start_positions is None:
            return
        if prepared_indices is None or prepared_start_positions is None:
            raise RuntimeError("Checkpoint must provide both prepared indices and start positions.")

        indices = [int(v) for v in prepared_indices.reshape(-1).tolist()]
        starts = [int(v) for v in prepared_start_positions.reshape(-1).tolist()]
        if len(indices) != self.batch_size:
            raise RuntimeError("Checkpoint prepared_indices size mismatch.")
        if len(starts) != self.batch_size:
            raise RuntimeError("Checkpoint prepared_start_positions size mismatch.")

        self._restored_prefetch = list(zip(indices, starts))


def prepare_cell_states(
    frame, seq_idx: int, rows: int, cols: int, cell_states: np.ndarray, max_code_point: int
):
    if frame.height != rows or frame.width != cols:
        raise RuntimeError(
            f"Obtained frame with size different from configuration: expected {rows}x{cols}, "
            f"got {frame.height}x{frame.width}"
        )

    n = rows * cols

    cp_arr = np.empty(n, dtype=np.uint32)
    fg_r = np.empty(n, dtype=np.uint8)
    fg_g = np.empty(n, dtype=np.uint8)
    fg_b = np.empty(n, dtype=np.uint8)
    bg_r = np.empty(n, dtype=np.uint8)
    bg_g = np.empty(n, dtype=np.uint8)
    bg_b = np.empty(n, dtype=np.uint8)

    frame.read_components(
        codepoints_out=cp_arr.ctypes.data,
        fg_r_out=fg_r.ctypes.data,
        fg_g_out=fg_g.ctypes.data,
        fg_b_out=fg_b.ctypes.data,
        bg_r_out=bg_r.ctypes.data,
        bg_g_out=bg_g.ctypes.data,
        bg_b_out=bg_b.ctypes.data,
        num_elements=n,
    )

    # Match C++ data loader behavior: clamp codepoints to max_code_point - 1 (vocab size).
    max_code_point_index = max_code_point - 1 if max_code_point > 0 else 0
    np.minimum(cp_arr, np.uint32(max_code_point_index), out=cp_arr)

    fg_packed = (fg_r.astype(np.uint32) << 16) | (fg_g.astype(np.uint32) << 8) | fg_b.astype(np.uint32)
    bg_packed = (bg_r.astype(np.uint32) << 16) | (bg_g.astype(np.uint32) << 8) | bg_b.astype(np.uint32)

    cp_2d = cp_arr.reshape((rows, cols))
    fg_2d = fg_packed.reshape((rows, cols))
    bg_2d = bg_packed.reshape((rows, cols))

    cell_states[seq_idx, :, :, CODE_POINT_CHANNEL_IDX] = cp_2d
    cell_states[seq_idx, :, :, FG_COLOR_CHANNEL_IDX] = fg_2d
    cell_states[seq_idx, :, :, BG_COLOR_CHANNEL_IDX] = bg_2d


class StreamTokenizationContext:
    def __init__(self, stream: TermInflateStream, tokenizer: Tokenizer):
        self.stream = stream
        self.tokenizer = tokenizer
        self.tokenizer_handle = None
        self.exhausted = not stream.has_next_frame()
        self.start_frame = None if self.exhausted else stream.read_frame()

def print_frame(frame: TerminalFrame) -> None:
    width = frame.width
    height = frame.height
    for row in range(height):
        line = ''
        for col in range(width):
            cell = frame.get_cell(col, row)
            cp = cell.codepoint
            if cp == 5:
                cp = ord('|')  # visualize ENQ as |
            line += chr(cp) if cp != 0 and cp < 256 else '\uFFFD'
        print(line)


def consume_next_token(seq_idx, context: StreamTokenizationContext) -> Tuple[int, Optional[TerminalFrame]]:
    if context.exhausted or context.start_frame is None:
        context.exhausted = True
        return 0, None

    tokenizer_handle = context.tokenizer_handle
    if tokenizer_handle is None:
        tokenizer_handle = context.tokenizer.new_tokenization_handle()
        context.tokenizer_handle = tokenizer_handle
    else:
        tokenizer_handle.reset()

    start_frame = context.start_frame
    frame = start_frame
    frame_data = frame.get_user_data()

    last_token = None
    last_frame_index = None
    while True:
        token = tokenizer_handle.get_current_token()

        if not frame_data:
            raise RuntimeError("Frame is missing user_data for tokenization")

        user_data = list(frame_data)
        if user_data and user_data[-1] == 0:
            user_data = user_data[:-1]
        for ch in user_data:
            tokenizer_handle.add_char(ch)

        prev_token = token

        if token is not None:
            last_token = token
            last_frame_index = context.stream.get_last_frame_index()

        if tokenizer_handle.is_dead():
            if prev_token is None:
                if last_token is not None:
                    context.stream.seek(last_frame_index)
                    context.start_frame = context.stream.read_frame()
                    prev_token = last_token
                else:
                    raise RuntimeError("Tokenizer handle is dead but no token was produced")
            token_str, token_id = prev_token
            #if seq_idx == 0:
            #    print_frame(start_frame)
            #    print("Next token: ", token_str, ", token id: ", token_id)
            #    print("=====")
            return token_id, start_frame

        if not context.stream.has_next_frame():
            context.exhausted = True
            return 0, None

        frame = context.start_frame = context.stream.read_frame()
        frame_data = frame.get_user_data()

def _format_dump_path(base_path: str, step: int) -> str:
    if "." not in base_path:
        return f"{base_path}.step{step}"
    idx = base_path.rfind(".")
    return f"{base_path[:idx]}.step{step}{base_path[idx:]}"


def _checkpoint_step_width(max_steps: int) -> int:
    if max_steps <= 0:
        return 1
    return len(str(max_steps))


def _checkpoint_path(directory: str, step: int, width: int) -> str:
    filename = f"checkpoint_step_{step:0{width}d}.safetensors"
    return os.path.join(directory, filename)


def _parse_checkpoint_step(filename: str, width: int, max_steps: int) -> Optional[int]:
    prefix = "checkpoint_step_"
    suffix = ".safetensors"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    num_str = filename[len(prefix) : -len(suffix)]
    if width > 0 and len(num_str) != width:
        return None
    if not num_str.isdigit():
        return None
    step = int(num_str)
    if step > max_steps:
        return None
    return step


def _find_latest_checkpoint(directory: str, width: int, max_steps: int) -> Optional[Tuple[int, str]]:
    if not os.path.isdir(directory):
        return None
    best_step = None
    best_path = None
    for entry in os.listdir(directory):
        step = _parse_checkpoint_step(entry, width, max_steps)
        if step is None:
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_path = os.path.join(directory, entry)
    if best_step is None or best_path is None:
        return None
    return best_step, best_path


def _collect_checkpoint_params(
    model: FrameHeadModule,
    lstm: torch.nn.LSTM,
    head: torch.nn.Linear,
    h0: torch.nn.Parameter,
    c0: torch.nn.Parameter,
) -> Dict[str, torch.Tensor]:
    # Emit C++-compatible parameter keys/layouts for bidirectional checkpoint portability.
    tensors: Dict[str, torch.Tensor] = {}
    cpp_named: Dict[str, torch.Tensor] = {}
    cpp_named.update(_collect_frame_head_params(model))
    cpp_named.update(_collect_action_model_params(lstm, head, h0, c0))
    for key, tensor in cpp_named.items():
        if not key.startswith("param_"):
            raise RuntimeError(f"Unexpected parameter key format: {key}")
        tensors[f"param/{key[len('param_'):]}"] = tensor
    return tensors


def _load_checkpoint_params(
    model: FrameHeadModule,
    lstm: torch.nn.LSTM,
    head: torch.nn.Linear,
    h0: torch.nn.Parameter,
    c0: torch.nn.Parameter,
    tensors: Dict[str, torch.Tensor],
) -> None:
    model_state = model.state_dict()
    model_load: Dict[str, torch.Tensor] = {}
    for name, target in model_state.items():
        key = f"param/model/{name}"
        if key not in tensors:
            raise RuntimeError(f"Missing checkpoint tensor: {key}")
        model_load[name] = tensors[key].to(device=target.device, dtype=target.dtype)
    model.load_state_dict(model_load)

    lstm_state = lstm.state_dict()
    lstm_load: Dict[str, torch.Tensor] = {}
    for name, target in lstm_state.items():
        key = f"param/lstm/{name}"
        if key not in tensors:
            raise RuntimeError(f"Missing checkpoint tensor: {key}")
        lstm_load[name] = tensors[key].to(device=target.device, dtype=target.dtype)
    lstm.load_state_dict(lstm_load)

    head_state = head.state_dict()
    head_load: Dict[str, torch.Tensor] = {}
    for name, target in head_state.items():
        key = f"param/head/{name}"
        if key not in tensors:
            raise RuntimeError(f"Missing checkpoint tensor: {key}")
        head_load[name] = tensors[key].to(device=target.device, dtype=target.dtype)
    head.load_state_dict(head_load)

    if "param/h0" not in tensors or "param/c0" not in tensors:
        raise RuntimeError("Missing checkpoint tensor: param/h0 or param/c0")
    h0.data.copy_(tensors["param/h0"].to(device=h0.device, dtype=h0.dtype))
    c0.data.copy_(tensors["param/c0"].to(device=c0.device, dtype=c0.dtype))


def _identity_layout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _transpose_2d_layout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(0, 1).contiguous()


def _conv_ohwc_to_oihw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 3, 1, 2).contiguous()


def _build_cpp_param_bindings(
    model: FrameHeadModule,
    lstm: torch.nn.LSTM,
    head: torch.nn.Linear,
    h0: torch.nn.Parameter,
    c0: torch.nn.Parameter,
) -> Dict[str, Tuple[torch.nn.Parameter, Callable[[torch.Tensor], torch.Tensor]]]:
    bindings: Dict[str, Tuple[torch.nn.Parameter, Callable[[torch.Tensor], torch.Tensor]]] = {}

    def add(name: str, param: torch.nn.Parameter, from_cpp: Callable[[torch.Tensor], torch.Tensor]) -> None:
        if name in bindings:
            raise RuntimeError(f"Duplicate C++ parameter binding: {name}")
        bindings[name] = (param, from_cpp)

    add("frame_head.codepoint_embedding", model.codepoint_embed, _identity_layout)
    add("frame_head.fg_r_embed", model.fg_r_embed, _identity_layout)
    add("frame_head.fg_g_embed", model.fg_g_embed, _identity_layout)
    add("frame_head.fg_b_embed", model.fg_b_embed, _identity_layout)
    add("frame_head.bg_r_embed", model.bg_r_embed, _identity_layout)
    add("frame_head.bg_g_embed", model.bg_g_embed, _identity_layout)
    add("frame_head.bg_b_embed", model.bg_b_embed, _identity_layout)
    add("frame_head.position_embedding.weight", model.position_embed, _identity_layout)

    for idx, block in enumerate(model.downsample_blocks):
        add(f"frame_head.downsample_block.{idx}.ln.weight", block.ln_1.weight, _identity_layout)
        add(f"frame_head.downsample_block.{idx}.conv.weight", block.conv.weight, _conv_ohwc_to_oihw)

    for idx, block in enumerate(model.blocks):
        add(f"frame_head.block.{idx}.ln1.weight", block.ln_1.weight, _identity_layout)
        add(f"frame_head.block.{idx}.ln2.weight", block.ln_2.weight, _identity_layout)

        attn = block.attn
        add(f"frame_head.block.{idx}.attn.w_qkv", attn.c_attn.weight, _transpose_2d_layout)
        if attn.c_attn.bias is not None:
            add(f"frame_head.block.{idx}.attn.b_qkv", attn.c_attn.bias, _identity_layout)

        add(f"frame_head.block.{idx}.attn.w_proj", attn.c_proj.weight, _transpose_2d_layout)
        if attn.c_proj.bias is not None:
            add(f"frame_head.block.{idx}.attn.b_proj", attn.c_proj.bias, _identity_layout)

        mlp = block.mlp
        add(f"frame_head.block.{idx}.mlp.fc1.weight", mlp.c_fc.weight, _transpose_2d_layout)
        if mlp.c_fc.bias is not None:
            add(f"frame_head.block.{idx}.mlp.fc1.bias", mlp.c_fc.bias, _identity_layout)
        add(f"frame_head.block.{idx}.mlp.fc2.weight", mlp.c_proj.weight, _transpose_2d_layout)
        if mlp.c_proj.bias is not None:
            add(f"frame_head.block.{idx}.mlp.fc2.bias", mlp.c_proj.bias, _identity_layout)

    add("action_model.lstm.weights_ih", lstm.weight_ih_l0, _transpose_2d_layout)
    add("action_model.lstm.weights_hh", lstm.weight_hh_l0, _transpose_2d_layout)
    add("action_model.lstm.bias_ih", lstm.bias_ih_l0, _identity_layout)
    add("action_model.lstm.bias_hh", lstm.bias_hh_l0, _identity_layout)
    add("action_model.lstm.h0", h0, _identity_layout)
    add("action_model.lstm.c0", c0, _identity_layout)
    add("action_model.head.weight", head.weight, _transpose_2d_layout)

    return bindings


def _is_cpp_checkpoint_format(tensors: Dict[str, torch.Tensor]) -> bool:
    return "param/frame_head.codepoint_embedding" in tensors


def _read_scalar_int(tensors: Dict[str, torch.Tensor], key: str) -> int:
    tensor = tensors.get(key)
    if tensor is None:
        raise RuntimeError(f"Missing checkpoint tensor: {key}")
    if tensor.numel() != 1:
        raise RuntimeError(f"Checkpoint tensor must be scalar: {key}")
    return int(tensor.reshape(-1)[0].item())


def _load_cpp_checkpoint_params(
    tensors: Dict[str, torch.Tensor],
    cpp_param_bindings: Dict[str, Tuple[torch.nn.Parameter, Callable[[torch.Tensor], torch.Tensor]]],
) -> None:
    for name, (param, from_cpp) in cpp_param_bindings.items():
        key = f"param/{name}"
        if key not in tensors:
            raise RuntimeError(f"Missing checkpoint tensor: {key}")
        source = from_cpp(tensors[key])
        if tuple(source.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Checkpoint shape mismatch for '{key}': expected {tuple(param.shape)}, got {tuple(source.shape)}"
            )
        param.data.copy_(source.to(device=param.device, dtype=param.dtype))


def _restore_iterator_from_checkpoint(
    iterator: RecordingDatasetIterator,
    tensors: Dict[str, torch.Tensor],
) -> None:
    rng_state = _read_scalar_int(tensors, "dataset_rng_state")
    if "dataset_prefetch_indices" in tensors and "dataset_prefetch_starts" in tensors:
        prefetch_buffer_index = (
            _read_scalar_int(tensors, "dataset_prefetch_buffer_index")
            if "dataset_prefetch_buffer_index" in tensors
            else None
        )
        iterator.restore_state_from_checkpoint(
            rng_state,
            tensors["dataset_prefetch_indices"],
            tensors["dataset_prefetch_starts"],
            prefetch_buffer_index=prefetch_buffer_index,
        )
        return
    iterator.restore_state(rng_state)


def _optimizer_scope_from_param_name(param_name: str) -> str:
    if param_name.startswith("frame_head."):
        return "frame_head"
    if param_name.startswith("action_model."):
        return "action_model"
    raise RuntimeError(f"Unsupported optimizer parameter scope: {param_name}")


def _to_cpp_param_layout(param_name: str, tensor: torch.Tensor) -> torch.Tensor:
    if param_name.endswith(".conv.weight"):
        return tensor.permute(0, 2, 3, 1).contiguous()
    transpose_suffixes = (
        ".attn.w_qkv",
        ".attn.w_proj",
        ".mlp.fc1.weight",
        ".mlp.fc2.weight",
        ".lstm.weights_ih",
        ".lstm.weights_hh",
        ".head.weight",
    )
    if any(param_name.endswith(suffix) for suffix in transpose_suffixes):
        return tensor.transpose(0, 1).contiguous()
    return tensor


def _require_grad(param: torch.nn.Parameter, name: str) -> torch.Tensor:
    if param.grad is None:
        raise RuntimeError(f"Missing frame head gradient for {name}")
    return param.grad


def _build_optimizer(cfg: OptimizerConfiguration, params):
    opt_type = cfg.type.lower()
    if opt_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    if opt_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    if opt_type == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam
        except ImportError as exc:
            raise RuntimeError(
                "Muon optimizer requested but not installed. "
                "Install muon-optimizer from git+https://github.com/KellerJordan/Muon."
            ) from exc

        muon_params = [p for p in params if p.ndim >= 2]
        adam_params = [p for p in params if p.ndim < 2]
        if not muon_params and not adam_params:
            raise RuntimeError("Muon optimizer requires at least one parameter.")

        muon_lr = cfg.learning_rate if cfg.muon_learning_rate is None else cfg.muon_learning_rate
        param_groups = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=muon_lr,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                )
            )
        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=cfg.learning_rate,
                    betas=(cfg.beta1, cfg.beta2),
                    eps=cfg.eps,
                    weight_decay=cfg.weight_decay,
                )
            )
        return SingleDeviceMuonWithAuxAdam(param_groups)
    raise RuntimeError(f"Unsupported optimizer type: {cfg.type}")


class Fp32MasterOptimizer:
    def __init__(self, cfg: OptimizerConfiguration, params, param_names: Optional[List[str]] = None):
        self._params = list(params)
        if param_names is None:
            self._param_names = [str(i) for i in range(len(self._params))]
        else:
            if len(param_names) != len(self._params):
                raise RuntimeError("param_names length must match params length")
            self._param_names = list(param_names)
        self._master_params = [p.detach().float().clone().requires_grad_(True) for p in self._params]
        self._optimizer = _build_optimizer(cfg, self._master_params)
        self._sync_model_from_master()

    def _sync_model_from_master(self) -> None:
        for param, master in zip(self._params, self._master_params):
            param.data.copy_(master.data.to(dtype=param.dtype))

    def sync_master_from_model(self) -> None:
        for param, master in zip(self._params, self._master_params):
            master.data.copy_(param.data.to(dtype=master.dtype))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self._params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for param, master in zip(self._params, self._master_params):
            if param.grad is None:
                master.grad = None
                continue
            grad = param.grad.detach().float()
            if master.grad is None:
                master.grad = grad
            else:
                master.grad.copy_(grad)
        self._optimizer.step()
        self._sync_model_from_master()

    def state_tensors(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for idx, master in enumerate(self._master_params):
            tensors[f"optim/master/{idx}"] = master.detach().cpu()
            state = self._optimizer.state.get(master, {})
            for key, value in state.items():
                name = f"optim/state/{idx}/{key}"
                if torch.is_tensor(value):
                    tensors[name] = value.detach().cpu()
                elif isinstance(value, bool):
                    tensors[name] = torch.tensor(int(value), dtype=torch.uint64)
                elif isinstance(value, int):
                    tensors[name] = torch.tensor(value, dtype=torch.uint64)
                elif isinstance(value, float):
                    tensors[name] = torch.tensor(value, dtype=torch.float32)
                else:
                    raise RuntimeError(f"Unsupported optimizer state type for key {name}: {type(value)}")
        return tensors

    def state_tensors_cpp(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for name, param, master in zip(self._param_names, self._params, self._master_params):
            scope = _optimizer_scope_from_param_name(name)
            family = "muon" if master.ndim >= 2 else "adam"
            prefix = f"optim/{scope}/{family}/{name}"
            state = self._optimizer.state.get(master, {})
            use_master = param.dtype != torch.float32

            if family == "muon":
                momentum = state.get("momentum_buffer")
                if momentum is None:
                    momentum = torch.zeros_like(master)
                tensors[f"{prefix}/momentum"] = _to_cpp_param_layout(name, momentum.detach()).float().cpu()
                if use_master:
                    tensors[f"{prefix}/master"] = _to_cpp_param_layout(name, master.detach()).float().cpu()
            else:
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(master)
                if exp_avg_sq is None:
                    exp_avg_sq = torch.zeros_like(master)
                tensors[f"{prefix}/m"] = _to_cpp_param_layout(name, exp_avg.detach()).float().cpu()
                tensors[f"{prefix}/v"] = _to_cpp_param_layout(name, exp_avg_sq.detach()).float().cpu()
                if use_master:
                    tensors[f"{prefix}/master"] = _to_cpp_param_layout(name, master.detach()).float().cpu()

        return tensors

    def load_state_tensors(self, tensors: Dict[str, torch.Tensor]) -> None:
        # Restore master params first.
        for idx, master in enumerate(self._master_params):
            key = f"optim/master/{idx}"
            if key not in tensors:
                raise RuntimeError(f"Missing checkpoint tensor: {key}")
            master.data.copy_(tensors[key].to(device=master.device, dtype=master.dtype))

        # Restore optimizer state entries.
        for idx, master in enumerate(self._master_params):
            state_prefix = f"optim/state/{idx}/"
            state: Dict[str, object] = {}
            for key, value in tensors.items():
                if not key.startswith(state_prefix):
                    continue
                state_key = key[len(state_prefix) :]
                if value.numel() == 1 and value.dtype in (torch.int64, torch.int32, torch.uint64, torch.uint32):
                    state[state_key] = int(value.item())
                elif value.numel() == 1 and value.dtype in (torch.float16, torch.float32, torch.float64):
                    state[state_key] = float(value.item())
                else:
                    state[state_key] = value.to(device=master.device)
            self._optimizer.state[master] = state

        self._sync_model_from_master()

    def load_cpp_state_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        cpp_param_bindings: Dict[str, Tuple[torch.nn.Parameter, Callable[[torch.Tensor], torch.Tensor]]],
        step_count: int,
    ) -> None:
        for name, master in zip(self._param_names, self._master_params):
            if name not in cpp_param_bindings:
                raise RuntimeError(f"Missing C++ parameter binding for optimizer state: {name}")
            _, from_cpp = cpp_param_bindings[name]

            scope = _optimizer_scope_from_param_name(name)
            family = "muon" if master.ndim >= 2 else "adam"
            key_prefix = f"optim/{scope}/{family}/{name}"

            state: Dict[str, object] = {}

            master_key = f"{key_prefix}/master"
            if master_key in tensors:
                master_tensor = from_cpp(tensors[master_key])
                if tuple(master_tensor.shape) != tuple(master.shape):
                    raise RuntimeError(
                        f"Optimizer master shape mismatch for '{master_key}': "
                        f"expected {tuple(master.shape)}, got {tuple(master_tensor.shape)}"
                    )
                master.data.copy_(master_tensor.to(device=master.device, dtype=master.dtype))

            if family == "muon":
                momentum_key = f"{key_prefix}/momentum"
                if momentum_key not in tensors:
                    raise RuntimeError(f"Missing checkpoint tensor: {momentum_key}")
                momentum = from_cpp(tensors[momentum_key])
                if tuple(momentum.shape) != tuple(master.shape):
                    raise RuntimeError(
                        f"Optimizer momentum shape mismatch for '{momentum_key}': "
                        f"expected {tuple(master.shape)}, got {tuple(momentum.shape)}"
                    )
                state["momentum_buffer"] = momentum.to(device=master.device, dtype=master.dtype)
            else:
                m_key = f"{key_prefix}/m"
                v_key = f"{key_prefix}/v"
                if m_key not in tensors or v_key not in tensors:
                    raise RuntimeError(f"Missing checkpoint tensor: {m_key} or {v_key}")
                exp_avg = from_cpp(tensors[m_key])
                exp_avg_sq = from_cpp(tensors[v_key])
                if tuple(exp_avg.shape) != tuple(master.shape) or tuple(exp_avg_sq.shape) != tuple(master.shape):
                    raise RuntimeError(f"Optimizer Adam state shape mismatch for parameter: {name}")
                state["exp_avg"] = exp_avg.to(device=master.device, dtype=master.dtype)
                state["exp_avg_sq"] = exp_avg_sq.to(device=master.device, dtype=master.dtype)
                state["step"] = int(step_count)

            self._optimizer.state[master] = state

        self._sync_model_from_master()


def _collect_frame_head_grads(model: FrameHeadModule) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    grad_dtype = model.dtype

    grads["grad_frame_head.codepoint_embedding"] = _require_grad(
        model.codepoint_embed, "frame_head.codepoint_embedding"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.position_embedding.weight"] = _require_grad(
        model.position_embed, "frame_head.position_embedding.weight"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.fg_r_embed"] = _require_grad(
        model.fg_r_embed, "frame_head.fg_r_embed"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.fg_g_embed"] = _require_grad(
        model.fg_g_embed, "frame_head.fg_g_embed"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.fg_b_embed"] = _require_grad(
        model.fg_b_embed, "frame_head.fg_b_embed"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.bg_r_embed"] = _require_grad(
        model.bg_r_embed, "frame_head.bg_r_embed"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.bg_g_embed"] = _require_grad(
        model.bg_g_embed, "frame_head.bg_g_embed"
    ).to(dtype=grad_dtype).cpu()
    grads["grad_frame_head.bg_b_embed"] = _require_grad(
        model.bg_b_embed, "frame_head.bg_b_embed"
    ).to(dtype=grad_dtype).cpu()

    for idx, block in enumerate(model.downsample_blocks):
        grads[f"grad_frame_head.downsample_block.{idx}.ln.weight"] = _require_grad(
            block.ln_1.weight, f"frame_head.downsample_block.{idx}.ln.weight"
        ).to(dtype=grad_dtype).cpu()
        conv_grad = _require_grad(block.conv.weight, f"frame_head.downsample_block.{idx}.conv.weight")
        grads[f"grad_frame_head.downsample_block.{idx}.conv.weight"] = (
            conv_grad.permute(0, 2, 3, 1).contiguous().to(dtype=grad_dtype).cpu()
        )

    for idx, block in enumerate(model.blocks):
        grads[f"grad_frame_head.block.{idx}.ln1.weight"] = _require_grad(
            block.ln_1.weight, f"frame_head.block.{idx}.ln1.weight"
        ).to(dtype=grad_dtype).cpu()
        grads[f"grad_frame_head.block.{idx}.ln2.weight"] = _require_grad(
            block.ln_2.weight, f"frame_head.block.{idx}.ln2.weight"
        ).to(dtype=grad_dtype).cpu()

        attn = block.attn
        qkv_grad = _require_grad(attn.c_attn.weight, f"frame_head.block.{idx}.attn.w_qkv")
        grads[f"grad_frame_head.block.{idx}.attn.w_qkv"] = (
            qkv_grad.transpose(0, 1).contiguous().to(dtype=grad_dtype).cpu()
        )
        if attn.c_attn.bias is not None:
            grads[f"grad_frame_head.block.{idx}.attn.b_qkv"] = _require_grad(
                attn.c_attn.bias, f"frame_head.block.{idx}.attn.b_qkv"
            ).to(dtype=grad_dtype).cpu()

        proj_grad = _require_grad(attn.c_proj.weight, f"frame_head.block.{idx}.attn.w_proj")
        grads[f"grad_frame_head.block.{idx}.attn.w_proj"] = (
            proj_grad.transpose(0, 1).contiguous().to(dtype=grad_dtype).cpu()
        )
        if attn.c_proj.bias is not None:
            grads[f"grad_frame_head.block.{idx}.attn.b_proj"] = _require_grad(
                attn.c_proj.bias, f"frame_head.block.{idx}.attn.b_proj"
            ).to(dtype=grad_dtype).cpu()

        mlp = block.mlp
        fc1_grad = _require_grad(mlp.c_fc.weight, f"frame_head.block.{idx}.mlp.fc1.weight")
        grads[f"grad_frame_head.block.{idx}.mlp.fc1.weight"] = (
            fc1_grad.transpose(0, 1).contiguous().to(dtype=grad_dtype).cpu()
        )
        if mlp.c_fc.bias is not None:
            grads[f"grad_frame_head.block.{idx}.mlp.fc1.bias"] = _require_grad(
                mlp.c_fc.bias, f"frame_head.block.{idx}.mlp.fc1.bias"
            ).to(dtype=grad_dtype).cpu()

        fc2_grad = _require_grad(mlp.c_proj.weight, f"frame_head.block.{idx}.mlp.fc2.weight")
        grads[f"grad_frame_head.block.{idx}.mlp.fc2.weight"] = (
            fc2_grad.transpose(0, 1).contiguous().to(dtype=grad_dtype).cpu()
        )
        if mlp.c_proj.bias is not None:
            grads[f"grad_frame_head.block.{idx}.mlp.fc2.bias"] = _require_grad(
                mlp.c_proj.bias, f"frame_head.block.{idx}.mlp.fc2.bias"
            ).to(dtype=grad_dtype).cpu()

    return grads


def _collect_frame_head_params(model: FrameHeadModule) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    param_dtype = model.dtype

    params["param_frame_head.codepoint_embedding"] = model.codepoint_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.position_embedding.weight"] = model.position_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.fg_r_embed"] = model.fg_r_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.fg_g_embed"] = model.fg_g_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.fg_b_embed"] = model.fg_b_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.bg_r_embed"] = model.bg_r_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.bg_g_embed"] = model.bg_g_embed.detach().to(dtype=param_dtype).cpu()
    params["param_frame_head.bg_b_embed"] = model.bg_b_embed.detach().to(dtype=param_dtype).cpu()

    for idx, block in enumerate(model.downsample_blocks):
        params[f"param_frame_head.downsample_block.{idx}.ln.weight"] = (
            block.ln_1.weight.detach().to(dtype=param_dtype).cpu()
        )
        conv_weight = block.conv.weight.detach()
        params[f"param_frame_head.downsample_block.{idx}.conv.weight"] = (
            conv_weight.permute(0, 2, 3, 1).contiguous().to(dtype=param_dtype).cpu()
        )

    for idx, block in enumerate(model.blocks):
        params[f"param_frame_head.block.{idx}.ln1.weight"] = (
            block.ln_1.weight.detach().to(dtype=param_dtype).cpu()
        )
        params[f"param_frame_head.block.{idx}.ln2.weight"] = (
            block.ln_2.weight.detach().to(dtype=param_dtype).cpu()
        )

        attn = block.attn
        qkv_weight = attn.c_attn.weight.detach()
        params[f"param_frame_head.block.{idx}.attn.w_qkv"] = (
            qkv_weight.transpose(0, 1).contiguous().to(dtype=param_dtype).cpu()
        )
        if attn.c_attn.bias is not None:
            params[f"param_frame_head.block.{idx}.attn.b_qkv"] = (
                attn.c_attn.bias.detach().to(dtype=param_dtype).cpu()
            )

        proj_weight = attn.c_proj.weight.detach()
        params[f"param_frame_head.block.{idx}.attn.w_proj"] = (
            proj_weight.transpose(0, 1).contiguous().to(dtype=param_dtype).cpu()
        )
        if attn.c_proj.bias is not None:
            params[f"param_frame_head.block.{idx}.attn.b_proj"] = (
                attn.c_proj.bias.detach().to(dtype=param_dtype).cpu()
            )

        mlp = block.mlp
        fc1_weight = mlp.c_fc.weight.detach()
        params[f"param_frame_head.block.{idx}.mlp.fc1.weight"] = (
            fc1_weight.transpose(0, 1).contiguous().to(dtype=param_dtype).cpu()
        )
        if mlp.c_fc.bias is not None:
            params[f"param_frame_head.block.{idx}.mlp.fc1.bias"] = (
                mlp.c_fc.bias.detach().to(dtype=param_dtype).cpu()
            )

        fc2_weight = mlp.c_proj.weight.detach()
        params[f"param_frame_head.block.{idx}.mlp.fc2.weight"] = (
            fc2_weight.transpose(0, 1).contiguous().to(dtype=param_dtype).cpu()
        )
        if mlp.c_proj.bias is not None:
            params[f"param_frame_head.block.{idx}.mlp.fc2.bias"] = (
                mlp.c_proj.bias.detach().to(dtype=param_dtype).cpu()
            )

    return params


def _collect_action_model_params(
    lstm: torch.nn.LSTM,
    head: torch.nn.Linear,
    h0: torch.Tensor,
    c0: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    params["param_action_model.lstm.weights_ih"] = (
        lstm.weight_ih_l0.detach().transpose(0, 1).contiguous().float().cpu()
    )
    params["param_action_model.lstm.weights_hh"] = (
        lstm.weight_hh_l0.detach().transpose(0, 1).contiguous().float().cpu()
    )
    params["param_action_model.lstm.bias_ih"] = lstm.bias_ih_l0.detach().float().cpu()
    params["param_action_model.lstm.bias_hh"] = lstm.bias_hh_l0.detach().float().cpu()
    params["param_action_model.lstm.h0"] = h0.detach().float().cpu()
    params["param_action_model.lstm.c0"] = c0.detach().float().cpu()
    params["param_action_model.head.weight"] = (
        head.weight.detach().transpose(0, 1).contiguous().to(dtype=head.weight.dtype).cpu()
    )
    return params


def run_validation(
    run_config: RunConfiguration,
    tokenizer: Tokenizer,
    model: FrameHeadModule,
    lstm: torch.nn.LSTM,
    head: torch.nn.Linear,
    h0: torch.Tensor,
    c0: torch.Tensor,
    validation_iterator: RecordingDatasetIterator,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size_raw: int,
) -> float:
    batch_size = run_config.micro_batch_size
    total_batch_size = run_config.total_batch_size
    validation_seq_len = run_config.validation_sequence_length
    grad_accum_steps = total_batch_size // batch_size
    rows = run_config.frame_rows
    cols = run_config.frame_cols
    lstm_hidden_size = int(run_config.model_config.n_embed)

    cell_states = np.zeros((batch_size, rows, cols, NUM_FRAME_CHANNELS), dtype=np.uint32)
    validation_loss = 0.0

    model.train(False)
    lstm.train(False)
    head.train(False)
    with torch.inference_mode():
        for _ in range(grad_accum_steps):
            batch_iterator = validation_iterator.next_batch()

            frame_embeddings_seq = []
            targets = torch.empty((validation_seq_len, batch_size), device=device, dtype=torch.long)

            stream_contexts: List[StreamTokenizationContext] = []
            for inflate_stream in batch_iterator.streams:
                stream_contexts.append(StreamTokenizationContext(inflate_stream, tokenizer))

            for pos in range(validation_seq_len):
                for seq_idx, context in enumerate(stream_contexts):
                    if context.exhausted:
                        targets[pos, seq_idx] = 0
                        cell_states[seq_idx].fill(0)
                        continue

                    token_id, input_frame = consume_next_token(seq_idx, context)
                    if input_frame is None:
                        targets[pos, seq_idx] = 0
                        cell_states[seq_idx].fill(0)
                        continue

                    prepare_cell_states(
                        input_frame, seq_idx, rows, cols, cell_states, run_config.model_config.max_code_point
                    )
                    targets[pos, seq_idx] = int(token_id % vocab_size_raw)

                cell_states_torch = torch.from_numpy(cell_states).to(device=device, dtype=torch.int32)
                frame_embeddings = model(cell_states_torch)
                frame_embeddings_seq.append(frame_embeddings.to(device=device, dtype=dtype))

            lstm_input = torch.stack(frame_embeddings_seq, dim=0)
            h0_lstm = h0.view(1, 1, lstm_hidden_size).expand(1, batch_size, lstm_hidden_size).to(dtype=dtype)
            c0_lstm = c0.view(1, 1, lstm_hidden_size).expand(1, batch_size, lstm_hidden_size).to(dtype=dtype)

            lstm_out, _ = lstm(lstm_input, (h0_lstm, c0_lstm))
            logits = head(lstm_out.reshape(-1, lstm_hidden_size)).reshape(validation_seq_len, batch_size, -1).float()
            loss_sum = F.cross_entropy(
                logits.reshape(validation_seq_len * batch_size, -1),
                targets.reshape(-1),
                reduction="sum",
            )
            validation_loss += float((loss_sum / float(validation_seq_len * total_batch_size)).item())

    model.train(True)
    lstm.train(True)
    head.train(True)
    return validation_loss


def run(
        run_config: RunConfiguration,
        validation_dump: bool = False,
        validation_dump_file: str = "validation_dump.safetensors",
        validation_dump_steps: int = 1,
):
    mode = "validation-dump" if validation_dump else "training"
    print(f"Starting {mode} run with configuration:", run_config)

    dtype_str = run_config.model_config.dtype
    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Expected 'fp16' or 'bf16'.")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        dtype == torch.float16 and run_config.model_config.use_fp16_accumulation
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (dtype == torch.bfloat16)

    tokens = load_tokenizer(run_config.tokenizer_file_path)
    tokenizer = Tokenizer(tokens)

    train_iterator = RecordingDatasetIterator(
        run_config.train_data_config.recordings_directory_path,
        run_config.micro_batch_size,
        run_config.train_sequence_length,
        run_config.train_data_config.iteration_seed,
    )

    validation_iterator = RecordingDatasetIterator(
        run_config.validation_data_config.recordings_directory_path,
        run_config.micro_batch_size,
        run_config.validation_sequence_length,
        run_config.validation_data_config.iteration_seed,
    )

    batch_size = run_config.micro_batch_size
    total_batch_size = run_config.total_batch_size

    if total_batch_size <= 0:
        raise RuntimeError("total_batch_size must be greater than zero")
    if total_batch_size % batch_size != 0:
        raise RuntimeError("total_batch_size must be a multiple of micro_batch_size")

    # Gradient accumulation: run multiple micro batches per optimizer step.
    grad_accum_steps = total_batch_size // batch_size
    seq_len = run_config.validation_sequence_length if validation_dump else run_config.train_sequence_length

    rows = run_config.frame_rows
    cols = run_config.frame_cols

    if rows != run_config.model_config.frame_rows or cols != run_config.model_config.frame_cols:
        raise RuntimeError(
            f"Model frame size does not match training configuration: Model is {run_config.model_config.frame_rows}x{run_config.model_config.frame_cols}, "
            f"training configuration is {rows}x{cols}"
        )

    cell_states = np.zeros((batch_size, rows, cols, NUM_FRAME_CHANNELS), dtype=np.uint32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrameHeadModule(run_config.model_config, dtype, device, use_autograd_cell_embeds=True)

    lstm_hidden_size = int(run_config.model_config.n_embed)
    vocab_size_raw = len(tokens)
    vocab_alignment = 8
    vocab_size = ((vocab_size_raw + vocab_alignment - 1) // vocab_alignment) * vocab_alignment
    if vocab_size != vocab_size_raw:
        print(f"[INFO] Padding vocab size from {vocab_size_raw} to {vocab_size} for aligned GEMM")

    lstm = torch.nn.LSTM(
        input_size=lstm_hidden_size,
        hidden_size=lstm_hidden_size,
        batch_first=False,
        bias=True,
        device=device,
        dtype=dtype,
    )

    seed = int(run_config.model_config.model_init_seed)
    w_ih_stream, w_hh_stream, b_ih_stream, b_hh_stream = init_lstm_streaming_weights(
        lstm_hidden_size, lstm_hidden_size, seed, device=device, dtype=torch.float32
    )
    seed += 4

    head = torch.nn.Linear(
        lstm_hidden_size, vocab_size, bias=False, device=device, dtype=dtype
    )
    tmp_head_weight = torch.empty(
        (lstm_hidden_size, vocab_size), device=device, dtype=dtype
    )

    with torch.no_grad():
        lstm.weight_ih_l0.copy_(w_ih_stream.transpose(0, 1))
        lstm.weight_hh_l0.copy_(w_hh_stream.transpose(0, 1))
        lstm.bias_ih_l0.copy_(b_ih_stream)
        lstm.bias_hh_l0.copy_(b_hh_stream)
        init_utils.init_kaiming_uniform(tmp_head_weight, lstm_hidden_size, seed)
        head.weight.copy_(tmp_head_weight.transpose(0, 1))

    h0 = torch.nn.Parameter(torch.zeros((lstm_hidden_size,), device=device, dtype=torch.float32))
    c0 = torch.nn.Parameter(torch.zeros((lstm_hidden_size,), device=device, dtype=torch.float32))

    cpp_param_bindings = _build_cpp_param_bindings(model, lstm, head, h0, c0)
    trainable_params = list(model.parameters()) + list(lstm.parameters()) + list(head.parameters()) + [h0, c0]
    cpp_name_by_param_id = {id(param): name for name, (param, _) in cpp_param_bindings.items()}
    optimizer_param_names: List[str] = []
    for param in trainable_params:
        name = cpp_name_by_param_id.get(id(param))
        if name is None:
            raise RuntimeError("Missing C++ name mapping for optimizer parameter.")
        optimizer_param_names.append(name)

    optimizer = Fp32MasterOptimizer(
        run_config.optimizer_config,
        trainable_params,
        optimizer_param_names,
    )

    max_steps = validation_dump_steps if validation_dump else run_config.max_training_steps

    step = 0
    active_iterator = validation_iterator if validation_dump else train_iterator

    checkpoint_cfg = run_config.checkpointing
    checkpoint_enabled = not validation_dump
    checkpoint_width = _checkpoint_step_width(run_config.max_training_steps)
    if checkpoint_enabled and checkpoint_cfg.resume_behavior == "load_latest":
        latest = _find_latest_checkpoint(
            checkpoint_cfg.checkpoint_directory, checkpoint_width, run_config.max_training_steps
        )
        if latest is not None:
            step_from_name, checkpoint_path = latest
            print(f"[INFO] Loading checkpoint from {checkpoint_path}")
            tensors = load_file(checkpoint_path)
            if "step_count" not in tensors:
                raise RuntimeError("Missing checkpoint tensor: step_count")
            step_count = _read_scalar_int(tensors, "step_count")
            if step_count != step_from_name:
                raise RuntimeError(
                    f"Checkpoint step_count mismatch: filename says {step_from_name}, tensor has {step_count}"
                )
            if _is_cpp_checkpoint_format(tensors):
                _load_cpp_checkpoint_params(tensors, cpp_param_bindings)
                optimizer.sync_master_from_model()
                optimizer.load_cpp_state_tensors(tensors, cpp_param_bindings, step_count)
                _restore_iterator_from_checkpoint(active_iterator, tensors)
            else:
                if "dataset_rng_state" not in tensors:
                    raise RuntimeError("Missing checkpoint tensor: dataset_rng_state")
                active_iterator.restore_state(int(tensors["dataset_rng_state"].item()))
                _load_checkpoint_params(model, lstm, head, h0, c0, tensors)
                optimizer.load_state_tensors(tensors)
            step = step_count
            print(f"[INFO] Resuming from checkpoint at step_count={step_count}")

    while step < max_steps:
        start_time = time.time()
        print(f"[INFO] {mode.capitalize()} step {step}")
        loss_value = 0.0

        # Zero grads once per optimizer step; micro batches accumulate into these buffers.
        optimizer.zero_grad(set_to_none=True)

        for accum_idx in range(grad_accum_steps):
            if grad_accum_steps > 1:
                print(f"[INFO] Micro batch {accum_idx + 1}/{grad_accum_steps}")

            batch_iterator = active_iterator.next_batch()

            frame_embeddings_seq = []
            cell_states_seq = []
            targets = torch.empty((seq_len, batch_size), device=device, dtype=torch.long)

            # initialize tokenization contexts
            stream_contexts: List[StreamTokenizationContext] = []
            for inflate_stream in batch_iterator.streams:
                ctx = StreamTokenizationContext(inflate_stream, tokenizer)
                stream_contexts.append(ctx)

            for pos in range(seq_len):
                for seq_idx, context in enumerate(stream_contexts):
                    if context.exhausted:
                        targets[pos, seq_idx] = 0
                        cell_states[seq_idx].fill(0)
                        continue

                    token_id, input_frame = consume_next_token(seq_idx, context)
                    if input_frame is None:
                        targets[pos, seq_idx] = 0
                        cell_states[seq_idx].fill(0)
                        continue

                    prepare_cell_states(
                        input_frame, seq_idx, rows, cols, cell_states, run_config.model_config.max_code_point
                    )
                    targets[pos, seq_idx] = int(token_id % vocab_size_raw)

                cell_states_seq.append(cell_states.copy())

                with torch.no_grad():
                    cell_states_torch = torch.from_numpy(cell_states).to(device=device, dtype=torch.int32)
                    frame_embeddings = model(cell_states_torch)
                    frame_embeddings_seq.append(frame_embeddings.to(device=device, dtype=dtype))

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_step = time.time() - start_time
                frames_processed = (pos + 1) * batch_size
                fps = frames_processed / elapsed_step if elapsed_step > 0 else 0.0
                print(f"[DEBUG] Sequence position {pos}/{seq_len} processed | FPS: {fps:.2f}")

            lstm_input = torch.stack(frame_embeddings_seq, dim=0)
            lstm_input.requires_grad_(True)
            lstm_input.retain_grad()

            h0_lstm = h0.view(1, 1, lstm_hidden_size).expand(1, batch_size, lstm_hidden_size).to(dtype=dtype)
            c0_lstm = c0.view(1, 1, lstm_hidden_size).expand(1, batch_size, lstm_hidden_size).to(dtype=dtype)

            with torch.set_grad_enabled(True):
                lstm_out, (h_n, c_n) = lstm(lstm_input, (h0_lstm, c0_lstm))
                logits = head(lstm_out.reshape(-1, lstm_hidden_size)).reshape(
                    seq_len, batch_size, vocab_size
                )
                logits = logits.float()
                loss_sum = F.cross_entropy(
                    logits.reshape(seq_len * batch_size, vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )
                loss = loss_sum / float(seq_len * total_batch_size)

            loss_value += float(loss.item())
            loss.backward()

            if lstm_input.grad is None:
                raise RuntimeError("Missing LSTM input gradient")

            grad_x = lstm_input.grad.detach()
            with torch.set_grad_enabled(True):
                for pos in range(seq_len):
                    cell_states_torch = torch.from_numpy(cell_states_seq[pos]).to(device=device, dtype=torch.int32)
                    frame_embeddings = model(cell_states_torch)
                    frame_embeddings.backward(grad_x[pos], retain_graph=False)
                    print(f"[DEBUG] Backward through {pos}/{seq_len}")

        if device.type == "cuda":
            torch.cuda.synchronize()

        print(f"[INFO] loss/train: {loss_value}")
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[INFO] Step {step} completed in {elapsed:.3f} seconds")

        if validation_dump:
            cell_states_tensor = torch.from_numpy(np.stack(cell_states_seq, axis=0)).to(dtype=torch.float32)
            if lstm.weight_ih_l0.grad is None:
                raise RuntimeError("Missing LSTM weight_ih_l0 gradient")
            if lstm.weight_hh_l0.grad is None:
                raise RuntimeError("Missing LSTM weight_hh_l0 gradient")
            if lstm.bias_ih_l0.grad is None:
                raise RuntimeError("Missing LSTM bias_ih_l0 gradient")
            if lstm.bias_hh_l0.grad is None:
                raise RuntimeError("Missing LSTM bias_hh_l0 gradient")
            if lstm_input.grad is None:
                raise RuntimeError("Missing LSTM input gradient")
            if h0.grad is None:
                raise RuntimeError("Missing h0 gradient")
            if c0.grad is None:
                raise RuntimeError("Missing c0 gradient")
            grad_w_ih = lstm.weight_ih_l0.grad.transpose(0, 1).contiguous()
            grad_w_hh = lstm.weight_hh_l0.grad.transpose(0, 1).contiguous()
            grad_b_ih = lstm.bias_ih_l0.grad
            grad_b_hh = lstm.bias_hh_l0.grad
            grad_x = lstm_input.grad
            grad_h0 = h0.grad
            grad_c0 = c0.grad
            tensors = {
                "output": lstm_out.to(dtype=dtype).cpu(),
                "logits": logits.to(dtype=dtype).cpu(),
                "h_n": h_n.squeeze(0).to(dtype=dtype).cpu(),
                "c_n": c_n.squeeze(0).to(dtype=dtype).cpu(),
                "loss": loss.float().unsqueeze(0).cpu(),
                "frame_embeddings": lstm_input.detach().to(dtype=dtype).cpu(),
                "cell_states": cell_states_tensor.cpu(),
                "grad_x": grad_x.float().cpu(),
                "grad_w_ih": grad_w_ih.float().cpu(),
                "grad_w_hh": grad_w_hh.float().cpu(),
                "grad_b_ih": grad_b_ih.float().cpu(),
                "grad_b_hh": grad_b_hh.float().cpu(),
                "grad_h0": grad_h0.float().cpu(),
                "grad_c0": grad_c0.float().cpu(),
            }
            tensors.update(_collect_frame_head_grads(model))
            tensors.update(_collect_frame_head_params(model))
            tensors.update(_collect_action_model_params(lstm, head, h0, c0))
            dump_path = _format_dump_path(validation_dump_file, step)
            save_file(tensors, dump_path)
            print(f"[INFO] Validation dump saved to {dump_path}")

        optimizer.step()

        _emit_metrics_json(step, train_loss=loss_value, step_time_sec=elapsed)

        if (
            not validation_dump
            and run_config.enable_validation
            and run_config.validation_interval > 0
            and (step == 0 or (step + 1) % run_config.validation_interval == 0)
        ):
            validation_loss = run_validation(
                run_config=run_config,
                tokenizer=tokenizer,
                model=model,
                lstm=lstm,
                head=head,
                h0=h0,
                c0=c0,
                validation_iterator=validation_iterator,
                device=device,
                dtype=dtype,
                vocab_size_raw=vocab_size_raw,
            )
            print(f"[INFO] loss/validation: {validation_loss}")
            _emit_metrics_json(step, validation_loss=validation_loss)

        step += 1

        if checkpoint_enabled and step % checkpoint_cfg.checkpoint_interval == 0:
            os.makedirs(checkpoint_cfg.checkpoint_directory, exist_ok=True)
            tensors: Dict[str, torch.Tensor] = {}
            tensors.update(_collect_checkpoint_params(model, lstm, head, h0, c0))
            tensors.update(optimizer.state_tensors_cpp())
            tensors["step_count"] = torch.tensor(step, dtype=torch.uint64)
            tensors["dataset_rng_state"] = torch.tensor(
                active_iterator.checkpoint_state(), dtype=torch.uint64
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            checkpoint_path = _checkpoint_path(
                checkpoint_cfg.checkpoint_directory, step, checkpoint_width
            )
            save_file(tensors, checkpoint_path)
            print(f"[INFO] Saved checkpoint at step_count={step} to {checkpoint_path}")

    print("[INFO] Done preparing cell states for all steps.")


def main():
    parser = argparse.ArgumentParser(prog="fbamtrain_py")
    parser.add_argument("-c", "--config", required=True, help="Path to the training run configuration json file")
    parser.add_argument(
        "--validation-dump",
        action="store_true",
        help="Run a single validation sequence and dump outputs",
    )
    parser.add_argument(
        "--validation-dump-file",
        default="validation_dump.safetensors",
        help="Path to write validation safetensors (default: validation_dump.safetensors)",
    )
    parser.add_argument(
        "--validation-dump-steps",
        type=int,
        default=1,
        help="Number of validation steps to dump when --validation-dump is set (default: 1)",
    )
    args = parser.parse_args()

    config_path = args.config
    cfg = RunConfiguration.from_json_file(config_path)
    run(
        cfg,
        validation_dump=args.validation_dump,
        validation_dump_file=args.validation_dump_file,
        validation_dump_steps=args.validation_dump_steps,
    )


if __name__ == "__main__":
    main()
