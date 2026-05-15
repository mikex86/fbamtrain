from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]

from .matmul_variants import MatmulVariant, get_variant_sets  # noqa: E402
from .triton_aot_compiler import CompileArgs, compile_kernel  # noqa: E402


TARGETS = [
    {"target": "cuda:80:32", "suffix": "sm80"},
    {"target": "cuda:89:32", "suffix": "sm89"},
    {"target": "cuda:90:32", "suffix": "sm90"},
    {"target": "cuda:100:32", "suffix": "sm100"},
    {"target": "hip:gfx942:64", "suffix": "gfx942"},
    {"target": "hip:gfx1101:32", "suffix": "gfx1101"},
]

PTR_TOKENS = {
    "bf16": "*bf16:16",
    "fp16": "*fp16:16",
    "fp32": "*fp32:16",
}


def _match_targets(targets: list[dict[str, str]], filters: list[str]) -> list[dict[str, str]]:
    if not filters:
        return targets
    matched: list[dict[str, str]] = []
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
    env_path = os.getenv("TRITON_MATMUL_AUTOTUNE_CONFIG")
    if env_path:
        parts = [p.strip() for p in env_path.split(",") if p.strip()]
        return [Path(p) for p in parts]
    return [ROOT / "configs" / "triton_matmul_autotune.json"]


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
                f"Autotune config '{path}' not found. Run the matmul autotuner to generate it."
            )
        payload = _load_autotune_payload(path)
        merged = merge_dicts(merged, payload)
    return merged


def _lookup_blocks(
    payload: dict,
    target_suffix: str,
    target_name: str,
    variant_name: str,
) -> Dict[str, int]:
    section = payload.get("matmul")
    if not isinstance(section, dict):
        raise SystemExit("Autotune config missing top-level 'matmul' section.")
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
            f"Autotune config missing matmul entry for target '{target_suffix}' "
            f"(tried keys: {', '.join(candidates)})."
        )
    variant_payload = target_payload.get(variant_name)
    if not isinstance(variant_payload, dict):
        raise SystemExit(
            f"Autotune config missing matmul entry for variant '{variant_name}' under target '{target_suffix}'."
        )
    required = (
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    )
    try:
        return {key: int(variant_payload[key]) for key in required}
    except KeyError as exc:
        raise SystemExit(
            f"Autotune config for matmul {variant_name}/{target_suffix} is missing '{exc.args[0]}'"
        ) from exc


def _join_signature(parts: list[str | int]) -> str:
    return ", ".join(str(p) for p in parts)


def _addmm_signature(
    *,
    variant: MatmulVariant,
    blocks: Dict[str, int],
) -> str:
    aligned = variant.aligned
    a_ptr = PTR_TOKENS[variant.dtype_ab]
    b_ptr = PTR_TOKENS[variant.dtype_ab]
    c_ptr = PTR_TOKENS[variant.dtype_cd]
    d_ptr = PTR_TOKENS[variant.dtype_cd]

    if aligned:
        m_sig = n_sig = k_sig = "i32:16"
        stride_am = "i32:16"
        stride_bk = "i32:16"
        stride_cm = "i32"
        stride_dm = "i32:16"
        stride_ak = "1"
        stride_bn = "1"
        stride_cn = "1"
        stride_dn = "1"
    else:
        m_sig = n_sig = k_sig = "i32"
        stride_am = stride_ak = stride_bk = stride_bn = "i32"
        stride_cm = stride_cn = stride_dm = stride_dn = "i32"

    parts: list[str | int] = [
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        m_sig,
        n_sig,
        k_sig,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        blocks["BLOCK_SIZE_M"],
        blocks["BLOCK_SIZE_N"],
        blocks["BLOCK_SIZE_K"],
        blocks["GROUP_SIZE_M"],
        variant.activation,
        int(variant.accum_fp16),
        int(variant.has_bias),
    ]
    return _join_signature(parts)


def _addmm_preact_signature(
    *,
    variant: MatmulVariant,
    blocks: Dict[str, int],
) -> str:
    aligned = variant.aligned
    a_ptr = PTR_TOKENS[variant.dtype_ab]
    b_ptr = PTR_TOKENS[variant.dtype_ab]
    c_ptr = PTR_TOKENS[variant.dtype_cd]
    d_ptr = PTR_TOKENS[variant.dtype_cd]
    e_ptr = PTR_TOKENS[variant.dtype_e or variant.dtype_cd]

    if aligned:
        m_sig = n_sig = k_sig = "i32:16"
        stride_am = "i32:16"
        stride_bk = "i32:16"
        stride_cm = "i32"
        stride_dm = "i32:16"
        stride_em = "i32:16"
        stride_ak = "1"
        stride_bn = "1"
        stride_cn = "1"
        stride_dn = "1"
        stride_en = "1"
    else:
        m_sig = n_sig = k_sig = "i32"
        stride_am = stride_ak = stride_bk = stride_bn = "i32"
        stride_cm = stride_cn = stride_dm = stride_dn = stride_em = stride_en = "i32"

    parts: list[str | int] = [
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        e_ptr,
        m_sig,
        n_sig,
        k_sig,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        stride_em,
        stride_en,
        blocks["BLOCK_SIZE_M"],
        blocks["BLOCK_SIZE_N"],
        blocks["BLOCK_SIZE_K"],
        blocks["GROUP_SIZE_M"],
        variant.activation,
        int(variant.accum_fp16),
        int(variant.has_bias),
    ]
    return _join_signature(parts)


def _matmul_signature(
    *,
    variant: MatmulVariant,
    blocks: Dict[str, int],
) -> str:
    aligned = variant.aligned
    transpose = variant.transpose
    a_ptr = PTR_TOKENS[variant.dtype_ab]
    b_ptr = PTR_TOKENS[variant.dtype_ab]
    c_ptr = PTR_TOKENS[variant.dtype_cd]
    d_ptr = PTR_TOKENS[variant.dtype_cd]

    if aligned:
        m_sig = n_sig = k_sig = "i32:16"
        if transpose == "ta":
            stride_am = "1"
            stride_ak = "i32:16"
            stride_bk = "i32:16"
            stride_bn = "1"
        elif transpose == "tb":
            stride_am = "i32:16"
            stride_ak = "1"
            stride_bk = "1"
            stride_bn = "i32:16"
        elif transpose == "tab":
            stride_am = "1"
            stride_ak = "i32:16"
            stride_bk = "1"
            stride_bn = "i32:16"
        else:
            stride_am = "i32:16"
            stride_ak = "1"
            stride_bk = "i32:16"
            stride_bn = "1"
        stride_cm = "i32"
        stride_cn = "1"
        stride_dm = "i32:16"
        stride_dn = "1"
    else:
        m_sig = n_sig = k_sig = "i32"
        stride_am = stride_ak = stride_bk = stride_bn = "i32"
        stride_cm = stride_cn = stride_dm = stride_dn = "i32"

    parts: list[str | int] = [
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        m_sig,
        n_sig,
        k_sig,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        blocks["BLOCK_SIZE_M"],
        blocks["BLOCK_SIZE_N"],
        blocks["BLOCK_SIZE_K"],
        blocks["GROUP_SIZE_M"],
        variant.activation,
        int(variant.accum_fp16),
        int(variant.has_bias),
    ]
    return _join_signature(parts)


def build_signature_base(variant: MatmulVariant, blocks: Dict[str, int]) -> str:
    if variant.kind == "addmm":
        return _addmm_signature(variant=variant, blocks=blocks)
    if variant.kind == "addmm_preact":
        return _addmm_preact_signature(variant=variant, blocks=blocks)
    if variant.kind == "matmul":
        return _matmul_signature(variant=variant, blocks=blocks)
    raise ValueError(f"Unknown matmul variant kind '{variant.kind}'.")


def compile_kernels(
    autotune_config_paths: list[Path],
    variant_sets: list[str],
    target_filters: list[str],
    jobs: int | None = None,
) -> None:
    (ROOT / "output").mkdir(exist_ok=True)
    targets = _filter_targets(TARGETS)
    if target_filters:
        targets = _match_targets(targets, target_filters)
    if not targets:
        raise SystemExit("No targets matched for matmul compilation.")
    payload = _merge_payloads(autotune_config_paths)
    variants = get_variant_sets(variant_sets)

    tasks: list[tuple[CompileArgs, str]] = []
    for variant in variants:
        for target in targets:
            backend = target["target"].split(":", 1)[0]
            if variant.cuda_only and backend != "cuda":
                continue
            blocks = _lookup_blocks(payload, target["suffix"], target["target"], variant.name)
            signature_base = build_signature_base(variant, blocks)
            for accum_output, suffix in ((0, ""), (1, "_cacc")):
                signature = f"{signature_base}, {accum_output}"
                out_name = ROOT / "output" / f"{variant.name}{suffix}_{target['suffix']}"
                args = CompileArgs(
                    path=str(ROOT / "kernels/addmm_act.py"),
                    kernel_name=variant.kernel,
                    signature=signature,
                    target=target["target"],
                    num_warps=blocks["num_warps"],
                    num_stages=blocks["num_stages"],
                    out_name=str(out_name),
                )
                msg = (
                    f"Compiled {variant.name}{suffix} for {target['suffix']} "
                    f"(BS=({blocks['BLOCK_SIZE_M']}, {blocks['BLOCK_SIZE_N']}, {blocks['BLOCK_SIZE_K']}), "
                    f"G={blocks['GROUP_SIZE_M']}, warps={blocks['num_warps']}, stages={blocks['num_stages']})"
                )
                tasks.append((args, msg))

    if jobs is None:
        env_jobs = os.getenv("TRITON_JOBS")
        jobs = int(env_jobs) if env_jobs else (os.cpu_count() or 1)
    jobs = max(1, jobs)

    if jobs <= 1 or len(tasks) <= 1:
        for args, msg in tasks:
            compile_kernel(args)
            print(msg)
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
        futures = {executor.submit(_compile_task, args, msg): msg for args, msg in tasks}
        for future in as_completed(futures):
            msg = future.result()
            print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton matmul kernels using pre-tuned configs.")
    parser.add_argument(
        "--autotune-config",
        action="append",
        dest="autotune_configs",
        default=[],
        help=(
            "Path to the Triton matmul autotune JSON. "
            "May be passed multiple times to merge configs. "
            "Defaults to TRITON_MATMUL_AUTOTUNE_CONFIG or configs/triton_matmul_autotune.json."
        ),
    )
    parser.add_argument(
        "--variant-set",
        action="append",
        dest="variant_sets",
        default=[],
        help="Which matmul variant set(s) to compile: base, gelu, or all.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        default=[],
        help="Target suffix or target string to compile (e.g. sm89, sm_89, cuda:89:32).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel compile jobs (default: TRITON_JOBS or CPU count).",
    )
    args = parser.parse_args()
    variant_sets = args.variant_sets or ["all"]
    config_paths = _resolve_autotune_config_paths(args.autotune_configs)
    extra_targets = args.targets
    env_target = os.getenv("FBAMTRAIN_TRITON_TARGET_SUFFIXES")
    if env_target:
        extra_targets.extend([t.strip() for t in env_target.split(",") if t.strip()])
    compile_kernels(config_paths, variant_sets, extra_targets, jobs=args.jobs)


def _compile_task(args: CompileArgs, msg: str) -> str:
    compile_kernel(args)
    return msg


if __name__ == "__main__":
    main()
