import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import triton.testing
import triton.runtime.driver as triton_driver

ROOT = Path(__file__).resolve().parents[1]

from kernels import conv2d_bwd  # noqa: E402
from .triton_aot_compiler import CompileArgs, compile_kernel  # noqa: E402


DTYPES = [
    {
        "name": "bf16",
        "ptr": "bf16",
        "torch": torch.bfloat16,
        "code": 0,
    },
    {
        "name": "fp16",
        "ptr": "fp16",
        "torch": torch.float16,
        "code": 1,
    },
]

SUPPORTED_CONFIGS = [
    {
        "name": "ic3_oc8_k3",
        "batch": 4,
        "in_channels": 3,
        "out_channels": 8,
        "height": 16,
        "width": 16,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
    },
    {
        "name": "ic3_oc8_k3_dil2",
        "batch": 4,
        "in_channels": 3,
        "out_channels": 8,
        "height": 16,
        "width": 16,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 2,
        "dilation_w": 2,
    },
    {
        "name": "ic32_oc64_k3",
        "batch": 16,
        "in_channels": 32,
        "out_channels": 64,
        "height": 64,
        "width": 64,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
    },
    {
        "name": "ic768_oc768_k3_dil2",
        "batch": 4,
        "in_channels": 768,
        "out_channels": 768,
        "height": 32,
        "width": 32,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 2,
        "pad_w": 2,
        "dilation_h": 2,
        "dilation_w": 2,
    },
    {
        "name": "ic768_oc1536_k3",
        "batch": 4,
        "in_channels": 768,
        "out_channels": 1536,
        "height": 32,
        "width": 32,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
    },
    {
        "name": "ic1024_oc1024_k3_dil2",
        "batch": 4,
        "in_channels": 1024,
        "out_channels": 1024,
        "height": 32,
        "width": 32,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 2,
        "pad_w": 2,
        "dilation_h": 2,
        "dilation_w": 2,
    },
    {
        "name": "ic1024_oc2048_k3",
        "batch": 4,
        "in_channels": 1024,
        "out_channels": 2048,
        "height": 32,
        "width": 32,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "dilation_h": 1,
        "dilation_w": 1,
    },
]

TARGETS = [
    {"target": "cuda:80:32", "suffix": "sm80"},
    {"target": "cuda:89:32", "suffix": "sm89"},
    {"target": "cuda:90:32", "suffix": "sm90"},
    {"target": "cuda:100:32", "suffix": "sm100"},
    {"target": "hip:gfx942:64", "suffix": "gfx942"},
    {"target": "hip:gfx1101:32", "suffix": "gfx1101"},
]

AUTOTUNE_WARMUP = 10
AUTOTUNE_REP = 40

DGRAD_CONFIGS = [
    {"BLOCK_W": 64, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 64, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 32, "BLOCK_K": 16, "num_warps": 2, "num_stages": 1},
]

WGRAD_CONFIGS = [
    {"BLOCK_W": 128, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 64, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 64, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 32, "BLOCK_K": 32, "num_warps": 2, "num_stages": 1},
    {"BLOCK_W": 32, "BLOCK_OC": 32, "BLOCK_K": 16, "num_warps": 2, "num_stages": 1},
]


def _compute_out_dim(in_size: int, kernel: int, pad: int, stride: int, dilation: int) -> int:
    effective = 1 + dilation * (kernel - 1)
    return (in_size + 2 * pad - effective) // stride + 1


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _parse_target(target: str | None) -> tuple[str | None, str | None, int | None]:
    if target is None:
        return None, None, None
    parts = target.split(":")
    backend = parts[0] if parts else None
    arch = parts[1] if len(parts) > 1 else None
    warp = int(parts[2]) if len(parts) > 2 else None
    return backend, arch, warp


def _configure_autotune_target(target: str | None) -> None:
    backend, arch, _ = _parse_target(target)
    if backend is None:
        return
    if backend == "hip":
        if torch.version.hip is None:
            raise RuntimeError(
                "HIP autotune requested but this environment is CUDA-only. "
                "Use the ROCm venv to tune HIP targets."
            )
        current = None
        try:
            from triton.backends.amd.driver import HIPDriver  # type: ignore
        except Exception:
            # Some ROCm Triton builds don't expose triton.backends; rely on defaults.
            pass
        else:
            if hasattr(triton_driver, "set_active"):
                triton_driver.set_active(HIPDriver())
        # Best-effort target sanity check when available.
        try:
            active = getattr(triton_driver, "active", None)
            if active is not None and hasattr(active, "get_current_target"):
                current = active.get_current_target()
        except Exception:
            current = None
        if current is not None and getattr(current, "backend", None) != "hip":
            raise RuntimeError(f"Failed to activate HIP driver; current target is {current}")
        if arch is not None:
            # Try to pick a matching GPU when multiple HIP devices exist.
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gcn = getattr(props, "gcnArchName", None)
                if gcn and arch in gcn:
                    torch.cuda.set_device(idx)
                    break
    elif backend == "cuda":
        try:
            from triton.backends.nvidia.driver import CudaDriver  # type: ignore
        except Exception:
            return
        triton_driver.set_active(CudaDriver())


def _filter_targets(targets: list[Dict[str, str]]) -> list[Dict[str, str]]:
    backends = os.getenv("FBAMTRAIN_TRITON_TARGET_BACKENDS")
    if not backends:
        return targets
    allowed = {item.strip() for item in backends.split(",") if item.strip()}
    if not allowed:
        return targets
    filtered = [target for target in targets if target["target"].split(":", 1)[0] in allowed]
    if not filtered:
        raise RuntimeError(f"FBAMTRAIN_TRITON_TARGET_BACKENDS={backends} filtered out all targets.")
    return filtered


def _load_autotune_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"Failed to read autotune config '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Autotune config '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Autotune config '{path}' must contain a JSON object at the top level.")
    return payload


def _resolve_autotune_config_paths(arg_paths: list[str] | None) -> list[Path]:
    if arg_paths:
        return [Path(p) for p in arg_paths]
    env_path = os.getenv("TRITON_CONV2D_AUTOTUNE_CONFIG")
    if env_path:
        parts = [p.strip() for p in env_path.split(",") if p.strip()]
        return [Path(p) for p in parts]
    return [ROOT / "configs" / "triton_conv2d_autotune.json"]


def _merge_payloads(paths: list[Path]) -> dict:
    def merge_dicts(dst: dict, src: dict) -> dict:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                dst[key] = merge_dicts(dst[key], value)
            else:
                dst[key] = value
        return dst

    merged: dict = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(
                f"Autotune config '{path}' not found. Run the Conv2d autotuner to generate it."
            )
        payload = _load_autotune_payload(path)
        merged = merge_dicts(merged, payload)
    return merged


def _lookup_blocks(
    payload: dict,
    section_name: str,
    target_suffix: str,
    target_name: str,
    cfg_name: str,
    dtype_name: str,
) -> Dict[str, int]:
    section = payload.get(section_name)
    if not isinstance(section, dict):
        raise SystemExit(f"Autotune config missing top-level '{section_name}' section.")
    candidates = [target_suffix, target_name]
    if target_suffix.startswith("sm") and not target_suffix.startswith("sm_"):
        candidates.append(f"sm_{target_suffix[2:]}")
    if target_suffix.startswith("sm_"):
        candidates.append(target_suffix.replace("sm_", "sm"))
    target_payload = None
    for key in candidates:
        entry = section.get(key)
        if isinstance(entry, dict):
            target_payload = entry
            break
    if target_payload is None:
        raise SystemExit(
            f"Autotune config missing {section_name} entry for target '{target_suffix}' "
            f"(tried keys: {', '.join(candidates)})."
        )
    cfg_payload = target_payload.get(cfg_name)
    if not isinstance(cfg_payload, dict):
        raise SystemExit(
            f"Autotune config missing {section_name} entry for config '{cfg_name}' under target '{target_suffix}'."
        )
    dtype_payload = cfg_payload.get(dtype_name)
    if isinstance(dtype_payload, list):
        if not dtype_payload:
            raise SystemExit(
                f"Autotune config for {section_name} {cfg_name}/{dtype_name}/{target_suffix} is empty."
            )
        dtype_payload = dtype_payload[0]
    if not isinstance(dtype_payload, dict):
        raise SystemExit(
            f"Autotune config missing {section_name} entry for dtype '{dtype_name}' "
            f"under config '{cfg_name}' and target '{target_suffix}'."
        )
    required = ("BLOCK_W", "BLOCK_OC", "BLOCK_K", "num_warps", "num_stages")
    try:
        return {key: int(dtype_payload[key]) for key in required}
    except KeyError as exc:
        raise SystemExit(
            f"Autotune config for {section_name} {cfg_name}/{dtype_name}/{target_suffix} "
            f"is missing '{exc.args[0]}'"
        ) from exc


def _autotune_kernel(label: str, configs: list[Dict[str, int]], run_kernel) -> list[Dict[str, int]]:
    timings: list[tuple[Dict[str, int], float]] = []
    for cfg in configs:
        try:
            run_kernel(cfg)
            ms = triton.testing.do_bench(
                lambda: run_kernel(cfg),
                warmup=AUTOTUNE_WARMUP,
                rep=AUTOTUNE_REP,
            )
        except Exception as exc:  # pragma: no cover - best-effort tuning
            print(f"Skipping {label} config {cfg} due to error: {exc}")
            continue
        timings.append((cfg, float(ms)))
    if not timings:
        raise RuntimeError(f"Failed to autotune {label} conv2d bwd kernel; all configs failed")
    timings.sort(key=lambda item: item[1])
    best_cfg, best_ms = timings[0]
    print(f"Autotune {label} best config {best_cfg} ({best_ms:.3f} ms)")
    return [cfg for cfg, _ in timings]


def autotune_bwd_configs(
    cfg: Dict[str, int],
    dtype: torch.dtype,
    target: str | None = None,
) -> tuple[list[Dict[str, int]], list[Dict[str, int]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required to autotune Conv2d backward kernels")

    _configure_autotune_target(target)

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)

    batch = cfg["batch"]
    in_channels = cfg["in_channels"]
    out_channels = cfg["out_channels"]
    in_h = cfg["height"]
    in_w = cfg["width"]
    kernel_h = cfg["kernel_h"]
    kernel_w = cfg["kernel_w"]
    stride_h = cfg["stride_h"]
    stride_w = cfg["stride_w"]
    pad_h = cfg["pad_h"]
    pad_w = cfg["pad_w"]
    dilation_h = cfg["dilation_h"]
    dilation_w = cfg["dilation_w"]

    out_h = _compute_out_dim(in_h, kernel_h, pad_h, stride_h, dilation_h)
    out_w = _compute_out_dim(in_w, kernel_w, pad_w, stride_w, dilation_w)

    input_tensor = torch.randn((batch, in_h, in_w, in_channels), device=device, dtype=dtype)
    weight_tensor = torch.randn((out_channels, kernel_h, kernel_w, in_channels), device=device, dtype=dtype)
    dout_tensor = torch.randn((batch, out_h, out_w, out_channels), device=device, dtype=dtype)
    dinput_tensor = torch.empty_like(input_tensor)
    dweight_tensor = torch.empty_like(weight_tensor)

    input_strides = input_tensor.stride()
    weight_strides = weight_tensor.stride()
    dout_strides = dout_tensor.stride()
    dinput_strides = dinput_tensor.stride()
    dweight_strides = dweight_tensor.stride()

    dtype_code = 0 if dtype == torch.bfloat16 else 1
    accumulate_in_fp32 = 1

    def run_dgrad(blocks: Dict[str, int]) -> None:
        grid = (
            _cdiv(in_channels, blocks["BLOCK_K"]),
            batch * in_h,
            _cdiv(in_w, blocks["BLOCK_W"]),
        )
        conv2d_bwd.conv2d_dgrad_kernel[grid](
            dout_tensor,
            weight_tensor,
            dinput_tensor,
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            dout_strides[0],
            dout_strides[1],
            dout_strides[2],
            dout_strides[3],
            weight_strides[0],
            weight_strides[1],
            weight_strides[2],
            weight_strides[3],
            dinput_strides[0],
            dinput_strides[1],
            dinput_strides[2],
            dinput_strides[3],
            BLOCK_W=blocks["BLOCK_W"],
            BLOCK_OC=blocks["BLOCK_OC"],
            BLOCK_K=blocks["BLOCK_K"],
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            IN_CHANNELS=in_channels,
            ACCUMULATE_IN_FP32=accumulate_in_fp32,
            OUT_DTYPE_CODE=dtype_code,
            num_warps=blocks["num_warps"],
            num_stages=blocks["num_stages"],
        )

    def run_wgrad(blocks: Dict[str, int]) -> None:
        grid = (
            _cdiv(out_channels, blocks["BLOCK_OC"]),
            _cdiv(in_channels, blocks["BLOCK_K"]),
            kernel_h * kernel_w,
        )
        conv2d_bwd.conv2d_wgrad_kernel[grid](
            input_tensor,
            dout_tensor,
            dweight_tensor,
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            input_strides[3],
            dout_strides[0],
            dout_strides[1],
            dout_strides[2],
            dout_strides[3],
            dweight_strides[0],
            dweight_strides[1],
            dweight_strides[2],
            dweight_strides[3],
            0,
            BLOCK_W=blocks["BLOCK_W"],
            BLOCK_OC=blocks["BLOCK_OC"],
            BLOCK_K=blocks["BLOCK_K"],
            KERNEL_H=kernel_h,
            KERNEL_W=kernel_w,
            IN_CHANNELS=in_channels,
            ACCUMULATE_IN_FP32=accumulate_in_fp32,
            OUT_DTYPE_CODE=dtype_code,
            num_warps=blocks["num_warps"],
            num_stages=blocks["num_stages"],
        )

    best_dgrad = _autotune_kernel("dgrad", DGRAD_CONFIGS, run_dgrad)
    best_wgrad = _autotune_kernel("wgrad", WGRAD_CONFIGS, run_wgrad)
    return best_dgrad, best_wgrad


def _compile_with_fallback(label: str, configs: list[Dict[str, int]], compile_fn) -> Dict[str, int]:
    last_exc: Exception | None = None
    for cfg in configs:
        try:
            compile_fn(cfg)
        except Exception as exc:  # pragma: no cover - best-effort compile
            print(f"Skipping {label} compile for config {cfg} due to error: {exc}")
            last_exc = exc
            continue
        return cfg
    raise RuntimeError(f"Failed to compile {label} conv2d bwd kernel") from last_exc


def _build_common_constants(cfg: Dict[str, int], dtype_code: int, blocks: Dict[str, int],
                            accumulate_in_fp32: int) -> list[str]:
    constants = [
        str(blocks["BLOCK_W"]),
        str(blocks["BLOCK_OC"]),
        str(blocks["BLOCK_K"]),
        str(cfg["kernel_h"]),
        str(cfg["kernel_w"]),
        str(cfg["in_channels"]),
        str(accumulate_in_fp32),
        str(dtype_code),
    ]
    return constants


def _build_signature(pointer_parts: list[str], constants: list[str], int_count: int = 27) -> str:
    parts = pointer_parts + ["i32"] * int_count + constants
    return ", ".join(parts)


def compile_kernels(autotune_config_paths: list[Path]) -> None:
    (ROOT / "output").mkdir(exist_ok=True)
    targets = _filter_targets(TARGETS)
    payload = _merge_payloads(autotune_config_paths)

    for cfg in SUPPORTED_CONFIGS:
        for dtype in DTYPES:
            accumulate_in_fp32 = 1

            tag = cfg["name"]

            for target in targets:
                dgrad_blocks = _lookup_blocks(
                    payload,
                    "conv2d_dgrad",
                    target["suffix"],
                    target["target"],
                    cfg["name"],
                    dtype["name"],
                )
                wgrad_blocks = _lookup_blocks(
                    payload,
                    "conv2d_wgrad",
                    target["suffix"],
                    target["target"],
                    cfg["name"],
                    dtype["name"],
                )
                dgrad_out = ROOT / "output" / f"conv2d_dgrad_{dtype['name']}_{tag}_{target['suffix']}"
                compile_kernel(
                    CompileArgs(
                        path=str(ROOT / "kernels/conv2d_bwd.py"),
                        kernel_name="conv2d_dgrad_kernel",
                        signature=_build_signature(
                            [f"*{dtype['ptr']}:16"] * 3,
                            _build_common_constants(cfg, dtype["code"], dgrad_blocks, accumulate_in_fp32),
                        ),
                        target=target["target"],
                        num_warps=dgrad_blocks["num_warps"],
                        num_stages=dgrad_blocks["num_stages"],
                        out_name=str(dgrad_out),
                    )
                )
                print(
                    f"Compiled conv2d dgrad {dtype['name']} config {cfg['name']} "
                    f"for {target['suffix']} with blocks ({dgrad_blocks['BLOCK_W']}, "
                    f"{dgrad_blocks['BLOCK_OC']}, {dgrad_blocks['BLOCK_K']})"
                )

                wgrad_out = ROOT / "output" / f"conv2d_wgrad_{dtype['name']}_{tag}_{target['suffix']}"
                compile_kernel(
                    CompileArgs(
                        path=str(ROOT / "kernels/conv2d_bwd.py"),
                        kernel_name="conv2d_wgrad_kernel",
                        signature=_build_signature(
                            [f"*{dtype['ptr']}:16"] * 3,
                            _build_common_constants(cfg, dtype["code"], wgrad_blocks, accumulate_in_fp32),
                            28,
                        ),
                        target=target["target"],
                        num_warps=wgrad_blocks["num_warps"],
                        num_stages=wgrad_blocks["num_stages"],
                        out_name=str(wgrad_out),
                    )
                )
                print(
                    f"Compiled conv2d wgrad {dtype['name']} config {cfg['name']} "
                    f"for {target['suffix']} with blocks ({wgrad_blocks['BLOCK_W']}, "
                    f"{wgrad_blocks['BLOCK_OC']}, {wgrad_blocks['BLOCK_K']})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton Conv2d backward kernels using pre-tuned configs.")
    parser.add_argument(
        "--autotune-config",
        action="append",
        dest="autotune_configs",
        default=[],
        help=(
            "Path to the Triton Conv2d autotune JSON. "
            "May be passed multiple times to merge configs. "
            "Defaults to TRITON_CONV2D_AUTOTUNE_CONFIG or configs/triton_conv2d_autotune.json."
        ),
    )
    args = parser.parse_args()
    config_paths = _resolve_autotune_config_paths(args.autotune_configs)
    compile_kernels(config_paths)


if __name__ == "__main__":
    main()
