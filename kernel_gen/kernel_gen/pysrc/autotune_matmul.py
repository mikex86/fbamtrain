from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import triton
import triton.runtime.driver as triton_driver
import triton.testing

ROOT = Path(__file__).resolve().parents[1]

from kernels import addmm_act  # noqa: E402
from .matmul_variants import MatmulVariant, get_variant_sets  # noqa: E402
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
            pass
        else:
            if hasattr(triton_driver, "set_active"):
                triton_driver.set_active(HIPDriver())
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


def _match_targets(targets: list[Dict[str, str]], filters: list[str]) -> list[Dict[str, str]]:
    if not filters:
        return targets
    matched: list[Dict[str, str]] = []
    for target in targets:
        suffix = target["suffix"]
        name = target["target"]
        arch = name.split(":", 2)[1] if ":" in name else name
        candidates = {suffix, name, arch}
        if suffix.startswith("sm") and not suffix.startswith("sm_"):
            candidates.add(f"sm_{suffix[2:]}")
        if suffix.startswith("sm_"):
            candidates.add(suffix.replace("sm_", "sm"))
        if any(f in candidates for f in filters):
            matched.append(target)
    return matched


def _load_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"Autotune config '{path}' must contain a JSON object at the top level.")
    return payload


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype}'")


def _shape(aligned: bool) -> tuple[int, int, int]:
    base = int(os.getenv("FBAMTRAIN_MATMUL_AUTOTUNE_SIZE", "8192"))
    offset = int(os.getenv("FBAMTRAIN_MATMUL_AUTOTUNE_OFFSET", "7"))
    if base <= 0:
        raise ValueError("FBAMTRAIN_MATMUL_AUTOTUNE_SIZE must be positive")
    if aligned:
        return base, base, base
    return base + offset, base + offset, base + offset


def _kernel_kwargs(config: triton.Config) -> Dict[str, int]:
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")
    return {key: int(config.kwargs[key]) for key in keys}


def _out_of_resources_types() -> tuple[type, ...]:
    types: list[type] = []
    try:
        from triton.runtime.autotuner import OutOfResources as AutotunerOutOfResources  # type: ignore
    except Exception:
        AutotunerOutOfResources = None
    if AutotunerOutOfResources is not None:
        types.append(AutotunerOutOfResources)
    try:
        from triton.runtime.errors import OutOfResources as RuntimeOutOfResources  # type: ignore
    except Exception:
        RuntimeOutOfResources = None
    if RuntimeOutOfResources is not None:
        types.append(RuntimeOutOfResources)
    return tuple(types)


def _bench_variant(variant: MatmulVariant) -> Dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to autotune Triton matmul kernels")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype_ab = _parse_dtype(variant.dtype_ab)
    dtype_cd = _parse_dtype(variant.dtype_cd)

    m, n, k = _shape(variant.aligned)

    torch.manual_seed(0)

    if variant.transpose in {"ta", "tab"}:
        a_base = torch.randn((k, m), device=device, dtype=dtype_ab)
        a = a_base.t()
    else:
        a = torch.randn((m, k), device=device, dtype=dtype_ab)

    if variant.transpose in {"tb", "tab"}:
        b_base = torch.randn((n, k), device=device, dtype=dtype_ab)
        b = b_base.t()
    else:
        b = torch.randn((k, n), device=device, dtype=dtype_ab)

    c = torch.randn((m, n), device=device, dtype=dtype_cd)
    d = torch.empty((m, n), device=device, dtype=dtype_cd)
    e = None
    if variant.kind == "addmm_preact":
        e = torch.empty((m, n), device=device, dtype=dtype_cd)

    configs = addmm_act.get_autotune_config()
    if not configs:
        raise RuntimeError("No Triton matmul autotune configs available")

    best_ms = None
    best_cfg = None

    out_of_resources = _out_of_resources_types()

    for cfg in configs:
        kwargs = _kernel_kwargs(cfg)
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

        if variant.kind == "addmm_preact":
            assert e is not None

            def run():
                addmm_act.addmm_act_preact[grid](
                    a,
                    b,
                    c,
                    d,
                    e,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c.stride(0),
                    c.stride(1),
                    d.stride(0),
                    d.stride(1),
                    e.stride(0),
                    e.stride(1),
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                    ACTIVATION=variant.activation,
                    ACCUMULATE_IN_FP16=int(variant.accum_fp16),
                    HAS_BIAS=int(variant.has_bias),
                    ACCUMULATE_OUTPUT=0,
                    **kwargs,
                )
        else:

            def run():
                addmm_act.addmm_act[grid](
                    a,
                    b,
                    c,
                    d,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c.stride(0),
                    c.stride(1),
                    d.stride(0),
                    d.stride(1),
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                    ACTIVATION=variant.activation,
                    ACCUMULATE_IN_FP16=int(variant.accum_fp16),
                    HAS_BIAS=int(variant.has_bias),
                    ACCUMULATE_OUTPUT=0,
                    **kwargs,
                )

        warmup = int(os.getenv("FBAMTRAIN_MATMUL_AUTOTUNE_WARMUP", "10"))
        rep = int(os.getenv("FBAMTRAIN_MATMUL_AUTOTUNE_REP", "20"))
        if out_of_resources:
            try:
                ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
            except out_of_resources:
                continue
        else:
            ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    if best_cfg is None:
        raise RuntimeError(f"Failed to autotune matmul variant {variant.name}")

    best_kwargs = _kernel_kwargs(best_cfg)
    return {
        **best_kwargs,
        "num_warps": int(best_cfg.num_warps),
        "num_stages": int(best_cfg.num_stages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune Triton matmul kernels.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON for tuned constants.")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        default=[],
        help="Target suffix or target string to tune (e.g. sm89, sm_89, cuda:89:32).",
    )
    parser.add_argument(
        "--variant-set",
        action="append",
        dest="variant_sets",
        default=[],
        help="Which matmul variant set(s) to tune: base, gelu, or all.",
    )
    args = parser.parse_args()

    targets = _match_targets(_filter_targets(TARGETS), args.targets)
    if not targets:
        raise SystemExit("No targets matched for matmul autotuning.")

    variant_sets = args.variant_sets or ["all"]
    variants = get_variant_sets(variant_sets)

    payload = _load_payload(args.output)
    matmul_section = payload.setdefault("matmul", {})

    for target in targets:
        _configure_autotune_target(target["target"])
        target_payload = matmul_section.setdefault(target["suffix"], {})
        for variant in variants:
            if variant.cuda_only and not target["target"].startswith("cuda"):
                continue
            print(f"Tuning {variant.name} for {target['suffix']}")
            best = _bench_variant(variant)
            target_payload[variant.name] = best

    _write_payload(args.output, payload)


if __name__ == "__main__":
    main()
