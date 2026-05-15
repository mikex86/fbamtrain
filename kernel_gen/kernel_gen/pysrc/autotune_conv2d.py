import argparse
import json
from pathlib import Path
from typing import Dict

from .compile_conv2d import (
    DTYPES,
    SUPPORTED_CONFIGS,
    TARGETS,
    _configure_autotune_target,
    _filter_targets,
)
from kernels import conv2d


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune Triton Conv2d forward kernels.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON for tuned constants.")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        default=[],
        help="Target suffix or target string to tune (e.g. sm89, sm_89, cuda:89:32).",
    )
    args = parser.parse_args()

    targets = _match_targets(_filter_targets(TARGETS), args.targets)
    if not targets:
        raise SystemExit("No targets matched for Conv2d autotuning.")

    payload = _load_payload(args.output)
    conv_section = payload.setdefault("conv2d", {})

    for target in targets:
        _configure_autotune_target(target["target"])
        target_payload = conv_section.setdefault(target["suffix"], {})
        for cfg in SUPPORTED_CONFIGS:
            cfg_payload = target_payload.setdefault(cfg["name"], {})
            for dtype in DTYPES:
                best = conv2d.autotune_conv2d_config(
                    batch=cfg["batch"],
                    in_channels=cfg["in_channels"],
                    out_channels=cfg["out_channels"],
                    height=cfg["height"],
                    width=cfg["width"],
                    kernel_h=cfg["kernel_h"],
                    kernel_w=cfg["kernel_w"],
                    stride_h=cfg["stride_h"],
                    stride_w=cfg["stride_w"],
                    pad_h=cfg["pad_h"],
                    pad_w=cfg["pad_w"],
                    dilation_h=cfg["dilation_h"],
                    dilation_w=cfg["dilation_w"],
                    dtype=dtype["torch"],
                )
                cfg_payload[dtype["name"]] = best

    _write_payload(args.output, payload)


if __name__ == "__main__":
    main()
