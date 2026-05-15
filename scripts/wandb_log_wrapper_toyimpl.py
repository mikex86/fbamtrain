#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toyimpl with W&B logging from JSON metric lines.")
    parser.add_argument("--script", required=True, help="Path to toyimpl main.py")
    parser.add_argument("--config", required=True, help="Run configuration JSON path")
    parser.add_argument("--workdir", required=True, help="Working directory for the run")
    parser.add_argument("--wandb-project", required=True, help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity/team name")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-group", default=None, help="W&B group name")
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated list of W&B tags")
    parser.add_argument("--", dest="extra_args", nargs=argparse.REMAINDER, help="Extra args for toyimpl")
    return parser.parse_args()


def maybe_parse_metrics(line: str) -> Dict[str, Any] | None:
    stripped = line.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "step" not in payload:
        return None
    return payload


def main() -> int:
    args = parse_args()

    tags = None
    if args.wandb_tags:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=tags,
        config={
            "script": args.script,
            "config": args.config,
            "workdir": args.workdir,
        },
    )

    cmd = [sys.executable, "-u", args.script, "--config", args.config]
    if args.extra_args:
        cmd.extend(args.extra_args)

    env = os.environ.copy()
    env["FBAMTRAIN_METRICS_JSON"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=args.workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        metrics = maybe_parse_metrics(line)
        if metrics is None:
            continue

        step = metrics.get("step")
        if isinstance(step, (int, float)):
            step_val = int(step)
        else:
            step_val = None

        log_metrics = {k: v for k, v in metrics.items() if k != "step"}
        wandb.log(log_metrics, step=step_val)

    return_code = process.wait()
    wandb.finish()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
