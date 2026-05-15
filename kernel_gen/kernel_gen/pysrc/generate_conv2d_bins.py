#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def _require(data: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise SystemExit(f"Missing '{key}' in {path}")
    return data[key]


def _conv_mode(model: Mapping[str, Any], path: Path) -> str:
    mode = _require(model, "downsample_conv_mode", path)
    if mode not in ("dilated", "geglu"):
        raise SystemExit(f"{path}: unsupported downsample_conv_mode '{mode}'")
    return mode


def _collect_shapes(config_paths: List[Path]) -> List[Dict[str, Any]]:
    shapes: List[Dict[str, Any]] = []
    for path in config_paths:
        cfg = _load_json(path)
        model = _require(cfg, "model_config", path)

        batch = int(_require(cfg, "micro_batch_size", path))
        rows = int(_require(cfg, "frame_rows", path))
        cols = int(_require(cfg, "frame_cols", path))
        downsample_blocks = int(_require(model, "downsample_blocks", path))
        embed_dim = int(_require(model, "n_embed", path))
        dilation = int(_require(model, "downsample_conv_dilation", path))
        mode = _conv_mode(model, path)

        if downsample_blocks <= 0:
            continue

        if mode == "dilated":
            out_channels = embed_dim
            dilation_h = dilation
            dilation_w = dilation
            padding_h = dilation
            padding_w = dilation
        else:
            out_channels = 2 * embed_dim
            dilation_h = 1
            dilation_w = 1
            padding_h = 1
            padding_w = 1

        current_rows = rows
        current_cols = cols
        for block_idx in range(downsample_blocks):
            shapes.append(
                {
                    "config": str(path),
                    "block_index": block_idx,
                    "batch": batch,
                    "height": current_rows,
                    "width": current_cols,
                    "in_channels": embed_dim,
                    "out_channels": out_channels,
                    "kernel_h": 3,
                    "kernel_w": 3,
                    "stride": 1,
                    "padding": padding_h,
                    "dilation": dilation_h,
                    "groups": 1,
                    "conv_mode": mode,
                }
            )
            current_rows //= 2
            current_cols //= 2
    return shapes


def _dedupe_bins(shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: dict[Tuple[int, ...], Dict[str, Any]] = {}
    for shape in shapes:
        key = (
            int(shape["batch"]),
            int(shape["height"]),
            int(shape["width"]),
            int(shape["in_channels"]),
            int(shape["out_channels"]),
            int(shape["kernel_h"]),
            int(shape["kernel_w"]),
            int(shape["stride"]),
            int(shape["padding"]),
            int(shape["dilation"]),
            int(shape["groups"]),
        )
        if key not in seen:
            seen[key] = dict(shape)
    return list(seen.values())


def _signature(shape: Mapping[str, Any]) -> str:
    return (
        f"ic{shape['in_channels']}_oc{shape['out_channels']}"
        f"_k{shape['kernel_h']}x{shape['kernel_w']}"
        f"_s{shape['stride']}_p{shape['padding']}_d{shape['dilation']}_g{shape['groups']}"
    )


def _forward_variant_specs() -> Iterable[Tuple[str, str, str, bool]]:
    return (
        ("bf16", "", "bf16", False),
        ("fp16", "_fp16", "fp16", False),
        ("fp16_acc_fp16", "_fp16_acc_fp16", "fp16_acc_fp16", True),
    )


def _bwd_variant_specs() -> Iterable[Tuple[str, str, str]]:
    return (
        ("bf16", "", "bf16"),
        ("fp16", "_fp16", "fp16"),
    )


def _kernel_alias(shape: Mapping[str, Any], variant_suffix: str, *, op: Literal["fwd", "dgrad", "wgrad"]) -> str:
    if op == "fwd":
        prefix = "conv2d_cutlass"
    elif op == "dgrad":
        prefix = "conv2d_dgrad_cutlass"
    else:
        prefix = "conv2d_wgrad_cutlass"
    return (
        f"{prefix}_{variant_suffix}_stride{shape['stride']}_pad{shape['padding']}_"
        f"dil{shape['dilation']}_ic{shape['in_channels']}_oc{shape['out_channels']}_"
        f"b{shape['batch']}_h{shape['height']}_w{shape['width']}"
    )


def _kernel_name(shape: Mapping[str, Any], variant_suffix: str, *, op: Literal["fwd", "dgrad", "wgrad"]) -> str:
    if op == "fwd":
        prefix = "cutlass_conv2d"
    elif op == "dgrad":
        prefix = "cutlass_conv2d_dgrad"
    else:
        prefix = "cutlass_conv2d_wgrad"
    return (
        f"{prefix}_{variant_suffix}_stride{shape['stride']}_pad{shape['padding']}_"
        f"dil{shape['dilation']}_ic{shape['in_channels']}_oc{shape['out_channels']}_"
        f"b{shape['batch']}_h{shape['height']}_w{shape['width']}_kernel"
    )


def _base_defines(shape: Mapping[str, Any], *, use_fp16: bool, use_fp16_acc: bool) -> Dict[str, str]:
    defines: Dict[str, str] = {
        "CUTLASS_CONV_EXPECT_DILATION_H": str(shape["dilation"]),
        "CUTLASS_CONV_EXPECT_DILATION_W": str(shape["dilation"]),
        "CUTLASS_CONV_EXPECT_GROUPS": str(shape["groups"]),
        "CUTLASS_CONV_EXPECT_KERNEL_H": str(shape["kernel_h"]),
        "CUTLASS_CONV_EXPECT_KERNEL_W": str(shape["kernel_w"]),
        "CUTLASS_CONV_EXPECT_PADDING_H": str(shape["padding"]),
        "CUTLASS_CONV_EXPECT_PADDING_W": str(shape["padding"]),
        "CUTLASS_CONV_EXPECT_STRIDE_H": str(shape["stride"]),
        "CUTLASS_CONV_EXPECT_STRIDE_W": str(shape["stride"]),
        "CUTLASS_CONV_ITERATOR_ALGO": "cutlass::conv::IteratorAlgorithm::kOptimized",
        "CUTLASS_CONV_META_IN_CHANNELS": str(shape["in_channels"]),
        "CUTLASS_CONV_NUM_STAGES": "3",
        "CUTLASS_CONV_STRIDE_SUPPORT": "cutlass::conv::StrideSupport::kUnity",
    }
    if use_fp16:
        defines["CUTLASS_CONV_FP16"] = "1"
    if use_fp16_acc:
        defines["CUTLASS_CONV_FP16_ACCUM"] = "1"
    return defines


def _write_kernel_header(path: Path, aliases: List[str]) -> None:
    lines = [
        "// Auto-generated by generate_conv2d_bins.py. Do not edit.",
        "",
        "#pragma once",
        "",
        "#define DECLARE_CONV2D_BIN_KERNELS(nv_arch) \\",
    ]
    if aliases:
        for idx, alias in enumerate(aliases):
            suffix = " \\" if idx + 1 < len(aliases) else ""
            lines.append(f"    DECLARE_CONV2D_CUTLASS_KERNEL_BINARY({alias}_##nv_arch);{suffix}")
    else:
        lines.append("    /* no conv2d bin kernels */")
    lines.append("")
    lines.append("#define DECLARE_CONV2D_BIN_KERNEL_ALIASES(nv_arch) \\")
    if aliases:
        for idx, alias in enumerate(aliases):
            suffix = " \\" if idx + 1 < len(aliases) else ""
            lines.append(f"    DECLARE_KERNEL_ALIAS({alias}_##nv_arch, {alias});{suffix}")
    else:
        lines.append("    /* no conv2d bin kernel aliases */")
    lines.append("")
    path.write_text("\n".join(lines))


def _write_embed_include(path: Path, aliases: List[str], sm_suffix: str) -> None:
    lines = ["// Auto-generated by generate_conv2d_bins.py. Do not edit."]
    if not aliases:
        lines.append("// (no conv2d bin kernels)")
        path.write_text("\n".join(lines) + "\n")
        return
    for alias in aliases:
        lines.append(f"EMBED_ASSET {alias}_{sm_suffix}, {alias}_{sm_suffix}.cubin, 4")
    path.write_text("\n".join(lines) + "\n")


def _write_kinfo_include(path: Path, aliases: List[str], sm_suffix: str) -> None:
    lines = ["// Auto-generated by generate_conv2d_bins.py. Do not edit."]
    if not aliases:
        lines.append("// (no conv2d bin kernels)")
        path.write_text("\n".join(lines) + "\n")
        return
    for alias in aliases:
        lines.append(f"#include \"../kernel_gen/output/{alias}_{sm_suffix}.kinfo.inc.asm\"")
        lines.append(f"DECLARE_KINFO {alias}_{sm_suffix}")
    path.write_text("\n".join(lines) + "\n")


def _write_map_header(
    path: Path,
    forward_entries: List[Dict[str, Any]],
    dgrad_entries: List[Dict[str, Any]],
    wgrad_entries: List[Dict[str, Any]],
) -> None:
    cutlass3_sms = sorted(
        {
            int(str(sm).removeprefix("sm"))
            for entries in (forward_entries, dgrad_entries, wgrad_entries)
            for entry in entries
            for sm in entry.get("cutlass3_sms", [])
        }
    )
    lines = [
        "// Auto-generated by generate_conv2d_bins.py. Do not edit.",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "#include <kernels/kernel_binaries.h>",
        "",
        "struct Conv2dBinKey",
        "{",
        "    uint32_t batch;",
        "    uint32_t height;",
        "    uint32_t width;",
        "};",
        "",
        "struct Conv2dBinKernelKey",
        "{",
        "    uint32_t in_channels;",
        "    uint32_t out_channels;",
        "    uint32_t kernel_h;",
        "    uint32_t kernel_w;",
        "    uint32_t stride_h;",
        "    uint32_t stride_w;",
        "    uint32_t dilation_h;",
        "    uint32_t dilation_w;",
        "};",
        "",
        "enum class CutlassConv2dBinImplementation : uint8_t",
        "{",
        "    Cutlass2,",
        "    Cutlass3,",
        "};",
        "",
        "struct CutlassConv2dBinKernelEntry",
        "{",
        "    Conv2dBinKernelKey key;",
        "    std::array<uint32_t, 2> padding;",
        "    Conv2dBinKey bin;",
        "    CutlassConv2dBinImplementation implementation;",
        "    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *bf16_kernel;",
        "    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_kernel;",
        "    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_acc_fp16_kernel;",
        "};",
        "",
        "struct CutlassConv2dBwdBinKernelEntry",
        "{",
        "    Conv2dBinKernelKey key;",
        "    std::array<uint32_t, 2> padding;",
        "    Conv2dBinKey bin;",
        "    CutlassConv2dBinImplementation implementation;",
        "    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *bf16_kernel;",
        "    const kernel_bin_t<kernel_meta_conv2d_cutlass_t> *fp16_kernel;",
        "};",
        "",
        "#if PI_TENSORLIB_ENABLE_CUDA",
        (
            "inline constexpr std::array<uint32_t, "
            f"{len(cutlass3_sms)}> kCutlassConv2dCutlass3Architectures{{{{"
            + ", ".join(str(sm) for sm in cutlass3_sms)
            + "}};"
        ),
        "constexpr bool IsCutlassConv2dCutlass3Architecture(const uint32_t architecture)",
        "{",
        "    for (const uint32_t supported : kCutlassConv2dCutlass3Architectures)",
        "    {",
        "        if (supported == architecture)",
        "        {",
        "            return true;",
        "        }",
        "    }",
        "    return false;",
        "}",
        "",
    ]

    lines.append(
        "inline constexpr std::array<CutlassConv2dBinKernelEntry, "
        f"{len(forward_entries)}> kCutlassConv2dBinKernels{{{{"
    )
    for entry in forward_entries:
        implementation = _cutlass3_implementation_expr(entry)
        lines.append(
            "    CutlassConv2dBinKernelEntry{"
            f"{{{entry['in_channels']}, {entry['out_channels']}, {entry['kernel_h']}, {entry['kernel_w']}, "
            f"{entry['stride']}, {entry['stride']}, {entry['dilation']}, {entry['dilation']}}}, "
            f"{{{entry['padding']}, {entry['padding']}}}, "
            f"{{{entry['batch']}, {entry['height']}, {entry['width']}}}, "
            f"{implementation}, "
            f"&k{entry['bf16_alias']}, &k{entry['fp16_alias']}, &k{entry['fp16_acc_alias']}"
            "},"
        )
    lines.append("}};")
    lines.append("")

    lines.append(
        "inline constexpr std::array<CutlassConv2dBwdBinKernelEntry, "
        f"{len(dgrad_entries)}> kCutlassConv2dDgradBinKernels{{{{"
    )
    for entry in dgrad_entries:
        implementation = _cutlass3_implementation_expr(entry)
        lines.append(
            "    CutlassConv2dBwdBinKernelEntry{"
            f"{{{entry['in_channels']}, {entry['out_channels']}, {entry['kernel_h']}, {entry['kernel_w']}, "
            f"{entry['stride']}, {entry['stride']}, {entry['dilation']}, {entry['dilation']}}}, "
            f"{{{entry['padding']}, {entry['padding']}}}, "
            f"{{{entry['batch']}, {entry['height']}, {entry['width']}}}, "
            f"{implementation}, "
            f"&k{entry['bf16_alias']}, &k{entry['fp16_alias']}"
            "},"
        )
    lines.append("}};")
    lines.append("")

    lines.append(
        "inline constexpr std::array<CutlassConv2dBwdBinKernelEntry, "
        f"{len(wgrad_entries)}> kCutlassConv2dWgradBinKernels{{{{"
    )
    for entry in wgrad_entries:
        implementation = _cutlass3_implementation_expr(entry)
        lines.append(
            "    CutlassConv2dBwdBinKernelEntry{"
            f"{{{entry['in_channels']}, {entry['out_channels']}, {entry['kernel_h']}, {entry['kernel_w']}, "
            f"{entry['stride']}, {entry['stride']}, {entry['dilation']}, {entry['dilation']}}}, "
            f"{{{entry['padding']}, {entry['padding']}}}, "
            f"{{{entry['batch']}, {entry['height']}, {entry['width']}}}, "
            f"{implementation}, "
            f"&k{entry['bf16_alias']}, &k{entry['fp16_alias']}"
            "},"
        )
    lines.append("}};")
    lines.append("#else")
    lines.append("inline constexpr std::array<CutlassConv2dBinKernelEntry, 0> kCutlassConv2dBinKernels{};")
    lines.append("inline constexpr std::array<CutlassConv2dBwdBinKernelEntry, 0> kCutlassConv2dDgradBinKernels{};")
    lines.append("inline constexpr std::array<CutlassConv2dBwdBinKernelEntry, 0> kCutlassConv2dWgradBinKernels{};")
    lines.append("#endif")
    lines.append("")
    path.write_text("\n".join(lines))


def _cutlass3_implementation_expr(entry: Mapping[str, Any]) -> str:
    if not entry.get("cutlass3_sms", []):
        return "CutlassConv2dBinImplementation::Cutlass2"
    return (
        "(IsCutlassConv2dCutlass3Architecture(NV_KERNEL_ARCH) "
        "? CutlassConv2dBinImplementation::Cutlass3 "
        ": CutlassConv2dBinImplementation::Cutlass2)"
    )


def _make_config_entry(
    shape: Mapping[str, Any],
    *,
    kernel_alias: str,
    kernel_name: str,
    use_fp16: bool,
    use_fp16_acc: bool,
    source_file: str,
    op: Literal["fwd", "dgrad", "wgrad"],
) -> Dict[str, Any]:
    defines = _base_defines(shape, use_fp16=use_fp16, use_fp16_acc=use_fp16_acc)
    defines["CUTLASS_CONV_KERNEL_NAME"] = kernel_name
    defines["CUTLASS_CONV_OP"] = {
        "fwd": "cutlass::conv::Operator::kFprop",
        "dgrad": "cutlass::conv::Operator::kDgrad",
        "wgrad": "cutlass::conv::Operator::kWgrad",
    }[op]
    return {
        "basename": f"{kernel_alias}_sm80",
        "kernel_name": kernel_name,
        "defines": defines,
        "batch": shape["batch"],
        "height": shape["height"],
        "width": shape["width"],
        "in_channels": shape["in_channels"],
        "out_channels": shape["out_channels"],
        "stride": shape["stride"],
        "padding": shape["padding"],
        "dilation": shape["dilation"],
        "kernel_h": shape["kernel_h"],
        "kernel_w": shape["kernel_w"],
        "groups": shape["groups"],
        "accumulate_in_fp16": use_fp16_acc,
        "enabled": True,
        "source_file": source_file,
    }


def _maybe_add_cutlass3_override(entry: Dict[str, Any], shape: Mapping[str, Any]) -> List[str]:
    if int(shape["groups"]) == 1:
        # CUTLASS 3's SM90 TMA/GMMA conv path is the supported generated path
        # for these binned downsample shapes.  SM80/SM89 stay on the CUTLASS 2
        # generator, and SM100 remains on CUTLASS 2 until this CUTLASS tree has
        # matching conv collectives for every generated variant.
        cutlass3_sms = ["sm90"]
        entry["source_file_by_sm"] = {"sm90": "conv2d_cutlass3_kernel.cu"}
        entry["defines_by_sm"] = {
            "sm90": {
                "CUTLASS_CONV_THREADBLOCK_M": "128",
                "CUTLASS_CONV_THREADBLOCK_N": "128",
                "CUTLASS_CONV_THREADBLOCK_K": "64",
                "CUTLASS_CONV_CLUSTER_M": "2",
                "CUTLASS_CONV_CLUSTER_N": "4",
                "CUTLASS_CONV_CLUSTER_K": "1",
                "CUTLASS_CONV_ALIGNMENT_A": "8",
                "CUTLASS_CONV_ALIGNMENT_B": "8",
            }
        }
        return cutlass3_sms
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bin-specific conv2d configs and headers.")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        required=True,
        help="Path to a run-config JSON file (repeatable).",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional output path for the collected conv2d shape manifest JSON.",
    )
    parser.add_argument("--config-out", type=Path, required=True, help="Output JSON for bin configs.")
    parser.add_argument("--kernel-header-out", type=Path, required=True, help="Output header for bin kernel declarations.")
    parser.add_argument("--map-header-out", type=Path, required=True, help="Output header for bin kernel mapping.")
    parser.add_argument("--embed-sm89-out", type=Path, required=True, help="Output embed include for sm89.")
    parser.add_argument("--embed-sm90-out", type=Path, required=True, help="Output embed include for sm90.")
    parser.add_argument("--embed-sm100-out", type=Path, required=True, help="Output embed include for sm100.")
    parser.add_argument("--kinfo-sm89-out", type=Path, required=True, help="Output kinfo include for sm89.")
    parser.add_argument("--kinfo-sm90-out", type=Path, required=True, help="Output kinfo include for sm90.")
    parser.add_argument("--kinfo-sm100-out", type=Path, required=True, help="Output kinfo include for sm100.")
    args = parser.parse_args()

    config_paths = [Path(cfg) for cfg in args.configs]
    shapes = _collect_shapes(config_paths)
    bins = _dedupe_bins(shapes)
    for item in bins:
        item["signature"] = _signature(item)

    if args.manifest_out is not None:
        manifest = {
            "source_configs": [str(path) for path in config_paths],
            "conv2d_shapes": shapes,
            "bins": sorted(bins, key=lambda entry: (entry["signature"], entry["batch"], entry["height"], entry["width"])),
        }
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    conv2d_configs: Dict[str, Dict[str, Any]] = {}
    conv2d_dgrad_configs: Dict[str, Dict[str, Any]] = {}
    conv2d_wgrad_configs: Dict[str, Dict[str, Any]] = {}
    forward_map_entries: List[Dict[str, Any]] = []
    dgrad_map_entries: List[Dict[str, Any]] = []
    wgrad_map_entries: List[Dict[str, Any]] = []
    kernel_aliases: List[str] = []

    for shape in bins:
        base_name = (
            f"ic{shape['in_channels']}_oc{shape['out_channels']}"
            f"_stride{shape['stride']}_pad{shape['padding']}_dil{shape['dilation']}"
        )

        fwd_entry_record = {
            "batch": shape["batch"],
            "height": shape["height"],
            "width": shape["width"],
            "in_channels": shape["in_channels"],
            "out_channels": shape["out_channels"],
            "kernel_h": shape["kernel_h"],
            "kernel_w": shape["kernel_w"],
            "stride": shape["stride"],
            "padding": shape["padding"],
            "dilation": shape["dilation"],
            "groups": shape["groups"],
        }
        dgrad_entry_record = dict(fwd_entry_record)
        wgrad_entry_record = dict(fwd_entry_record)

        for variant_label, config_suffix, variant_suffix, use_fp16_acc in _forward_variant_specs():
            config_name = f"{base_name}_b{shape['batch']}_h{shape['height']}_w{shape['width']}{config_suffix}"
            kernel_alias = _kernel_alias(shape, variant_suffix, op="fwd")
            kernel_name = _kernel_name(shape, variant_suffix, op="fwd")

            conv2d_configs[config_name] = _make_config_entry(
                shape,
                kernel_alias=kernel_alias,
                kernel_name=kernel_name,
                use_fp16=(variant_label != "bf16"),
                use_fp16_acc=use_fp16_acc,
                source_file="conv2d_cutlass_kernel.cu",
                op="fwd",
            )
            cutlass3_sms = _maybe_add_cutlass3_override(conv2d_configs[config_name], shape)
            if cutlass3_sms:
                fwd_entry_record["cutlass3_sms"] = sorted(
                    set(fwd_entry_record.get("cutlass3_sms", [])) | set(cutlass3_sms)
                )
            kernel_aliases.append(kernel_alias)

            if variant_label == "bf16":
                fwd_entry_record["bf16_alias"] = kernel_alias
            elif variant_label == "fp16":
                fwd_entry_record["fp16_alias"] = kernel_alias
            else:
                fwd_entry_record["fp16_acc_alias"] = kernel_alias

        for variant_label, config_suffix, variant_suffix in _bwd_variant_specs():
            config_name = f"{base_name}_b{shape['batch']}_h{shape['height']}_w{shape['width']}{config_suffix}"

            dgrad_alias = _kernel_alias(shape, variant_suffix, op="dgrad")
            dgrad_name = _kernel_name(shape, variant_suffix, op="dgrad")
            conv2d_dgrad_configs[config_name] = _make_config_entry(
                shape,
                kernel_alias=dgrad_alias,
                kernel_name=dgrad_name,
                use_fp16=(variant_label != "bf16"),
                use_fp16_acc=False,
                source_file="conv2d_cutlass_dgrad_kernel.cu",
                op="dgrad",
            )
            cutlass3_sms = _maybe_add_cutlass3_override(conv2d_dgrad_configs[config_name], shape)
            if cutlass3_sms:
                dgrad_entry_record["cutlass3_sms"] = sorted(
                    set(dgrad_entry_record.get("cutlass3_sms", [])) | set(cutlass3_sms)
                )
            kernel_aliases.append(dgrad_alias)

            if variant_label == "bf16":
                dgrad_entry_record["bf16_alias"] = dgrad_alias
            else:
                dgrad_entry_record["fp16_alias"] = dgrad_alias

            wgrad_alias = _kernel_alias(shape, variant_suffix, op="wgrad")
            wgrad_name = _kernel_name(shape, variant_suffix, op="wgrad")
            wgrad_config_entry = _make_config_entry(
                shape,
                kernel_alias=wgrad_alias,
                kernel_name=wgrad_name,
                use_fp16=(variant_label != "bf16"),
                use_fp16_acc=False,
                source_file="conv2d_cutlass_wgrad_kernel.cu",
                op="wgrad",
            )
            cutlass3_sms = _maybe_add_cutlass3_override(wgrad_config_entry, shape)
            if cutlass3_sms:
                wgrad_entry_record["cutlass3_sms"] = sorted(
                    set(wgrad_entry_record.get("cutlass3_sms", [])) | set(cutlass3_sms)
                )
            conv2d_wgrad_configs[config_name] = wgrad_config_entry
            kernel_aliases.append(wgrad_alias)

            if variant_label == "bf16":
                wgrad_entry_record["bf16_alias"] = wgrad_alias
            else:
                wgrad_entry_record["fp16_alias"] = wgrad_alias

        forward_map_entries.append(fwd_entry_record)
        dgrad_map_entries.append(dgrad_entry_record)
        wgrad_map_entries.append(wgrad_entry_record)

    kernel_aliases = sorted(set(kernel_aliases))
    forward_map_entries = sorted(
        forward_map_entries,
        key=lambda entry: (
            entry["in_channels"],
            entry["out_channels"],
            entry["stride"],
            entry["padding"],
            entry["dilation"],
            entry["batch"],
            entry["height"],
            entry["width"],
        ),
    )
    dgrad_map_entries = sorted(
        dgrad_map_entries,
        key=lambda entry: (
            entry["in_channels"],
            entry["out_channels"],
            entry["stride"],
            entry["padding"],
            entry["dilation"],
            entry["batch"],
            entry["height"],
            entry["width"],
        ),
    )
    wgrad_map_entries = sorted(
        wgrad_map_entries,
        key=lambda entry: (
            entry["in_channels"],
            entry["out_channels"],
            entry["stride"],
            entry["padding"],
            entry["dilation"],
            entry["batch"],
            entry["height"],
            entry["width"],
        ),
    )
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(
            {
                "conv2d": conv2d_configs,
                "conv2d_dgrad": conv2d_dgrad_configs,
                "conv2d_wgrad": conv2d_wgrad_configs,
            },
            indent=2,
            sort_keys=True,
        )
    )

    args.kernel_header_out.parent.mkdir(parents=True, exist_ok=True)
    _write_kernel_header(args.kernel_header_out, kernel_aliases)

    args.map_header_out.parent.mkdir(parents=True, exist_ok=True)
    _write_map_header(args.map_header_out, forward_map_entries, dgrad_map_entries, wgrad_map_entries)

    args.embed_sm89_out.parent.mkdir(parents=True, exist_ok=True)
    _write_embed_include(args.embed_sm89_out, kernel_aliases, "sm89")

    args.embed_sm90_out.parent.mkdir(parents=True, exist_ok=True)
    _write_embed_include(args.embed_sm90_out, kernel_aliases, "sm90")

    args.embed_sm100_out.parent.mkdir(parents=True, exist_ok=True)
    _write_embed_include(args.embed_sm100_out, kernel_aliases, "sm100")

    args.kinfo_sm89_out.parent.mkdir(parents=True, exist_ok=True)
    _write_kinfo_include(args.kinfo_sm89_out, kernel_aliases, "sm89")

    args.kinfo_sm90_out.parent.mkdir(parents=True, exist_ok=True)
    _write_kinfo_include(args.kinfo_sm90_out, kernel_aliases, "sm90")

    args.kinfo_sm100_out.parent.mkdir(parents=True, exist_ok=True)
    _write_kinfo_include(args.kinfo_sm100_out, kernel_aliases, "sm100")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
