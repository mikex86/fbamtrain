import argparse
import json
from pathlib import Path
from typing import Dict

from .compile_conv2d_bwd import (
    DTYPES,
    SUPPORTED_CONFIGS,
    TARGETS,
    _configure_autotune_target,
    _filter_targets,
    autotune_bwd_configs,
)


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
    parser = argparse.ArgumentParser(description="Autotune Triton Conv2d backward kernels.")
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
        raise SystemExit("No targets matched for Conv2d backward autotuning.")

    payload = _load_payload(args.output)
    dgrad_section = payload.setdefault("conv2d_dgrad", {})
    wgrad_section = payload.setdefault("conv2d_wgrad", {})

    for target in targets:
        _configure_autotune_target(target["target"])
        dgrad_target = dgrad_section.setdefault(target["suffix"], {})
        wgrad_target = wgrad_section.setdefault(target["suffix"], {})
        for cfg in SUPPORTED_CONFIGS:
            dgrad_cfg = dgrad_target.setdefault(cfg["name"], {})
            wgrad_cfg = wgrad_target.setdefault(cfg["name"], {})
            for dtype in DTYPES:
                dgrad_best, wgrad_best = autotune_bwd_configs(
                    cfg,
                    dtype["torch"],
                    target["target"],
                )
                dgrad_cfg[dtype["name"]] = dgrad_best
                wgrad_cfg[dtype["name"]] = wgrad_best

    _write_payload(args.output, payload)


if __name__ == "__main__":
    main()
