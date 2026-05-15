import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import triton.runtime.driver as triton_driver

ROOT = Path(__file__).resolve().parents[1]

from kernels import conv2d  # noqa: E402
from .triton_aot_compiler import CompileArgs, compile_kernel  # noqa: E402


POINTER_COUNT = 4
SIGNATURE_TAIL = ["i32"] * 28

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
    target_suffix: str,
    target_name: str,
    cfg_name: str,
    dtype_name: str,
) -> Dict[str, int]:
    conv_section = payload.get("conv2d")
    if not isinstance(conv_section, dict):
        raise SystemExit("Autotune config missing top-level 'conv2d' section.")
    candidates = [target_suffix, target_name]
    if target_suffix.startswith("sm") and not target_suffix.startswith("sm_"):
        candidates.append(f"sm_{target_suffix[2:]}")
    if target_suffix.startswith("sm_"):
        candidates.append(target_suffix.replace("sm_", "sm"))
    target_payload = None
    for key in candidates:
        entry = conv_section.get(key)
        if isinstance(entry, dict):
            target_payload = entry
            break
    if target_payload is None:
        raise SystemExit(
            f"Autotune config missing conv2d entry for target '{target_suffix}' "
            f"(tried keys: {', '.join(candidates)})."
        )
    cfg_payload = target_payload.get(cfg_name)
    if not isinstance(cfg_payload, dict):
        raise SystemExit(
            f"Autotune config missing conv2d entry for config '{cfg_name}' under target '{target_suffix}'."
        )
    dtype_payload = cfg_payload.get(dtype_name)
    if not isinstance(dtype_payload, dict):
        raise SystemExit(
            f"Autotune config missing conv2d entry for dtype '{dtype_name}' "
            f"under config '{cfg_name}' and target '{target_suffix}'."
        )
    required = ("BLOCK_W", "BLOCK_OC", "BLOCK_K", "num_warps", "num_stages")
    try:
        return {key: int(dtype_payload[key]) for key in required}
    except KeyError as exc:
        raise SystemExit(
            f"Autotune config for conv2d {cfg_name}/{dtype_name}/{target_suffix} "
            f"is missing '{exc.args[0]}'"
        ) from exc


def build_signature(constants: Dict[str, int], cfg: Dict[str, int], dtype_token: str, dtype_code: int) -> str:
    pointer_parts = [f"*{dtype_token}:16"] * POINTER_COUNT
    parts = pointer_parts + SIGNATURE_TAIL + [
        str(constants["BLOCK_W"]),
        str(constants["BLOCK_OC"]),
        str(constants["BLOCK_K"]),
        str(cfg["kernel_h"]),
        str(cfg["kernel_w"]),
        str(cfg["in_channels"]),
        str(dtype_code),
    ]
    return ", ".join(parts)


def compile_kernels(autotune_config_paths: list[Path]) -> None:
    (ROOT / "output").mkdir(exist_ok=True)
    targets = _filter_targets(TARGETS)
    payload = _merge_payloads(autotune_config_paths)

    for cfg in SUPPORTED_CONFIGS:
        for dtype in DTYPES:
            tag = cfg["name"]

            for target in targets:
                best = _lookup_blocks(payload, target["suffix"], target["target"], cfg["name"], dtype["name"])

                signature = build_signature(best, cfg, dtype_token=dtype["ptr"], dtype_code=dtype["code"])
                out_name = ROOT / "output" / f"conv2d_{dtype['name']}_{tag}_{target['suffix']}"
                compile_kernel(
                    CompileArgs(
                        path=str(ROOT / "kernels/conv2d.py"),
                        kernel_name="conv2d_kernel",
                        signature=signature,
                        target=target["target"],
                        num_warps=best["num_warps"],
                        num_stages=best["num_stages"],
                        out_name=str(out_name),
                    )
                )
                print(
                    f"Compiled conv2d {dtype['name']} config {cfg['name']} with blocks ("
                    f"{best['BLOCK_W']}, {best['BLOCK_OC']}, {best['BLOCK_K']}) "
                    f"for {target['suffix']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton Conv2d kernels using pre-tuned configs.")
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
