import importlib
import inspect
import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import triton

from .kinfo_asm_gen import write_kinfo_and_macro


@dataclass
class CompileArgs:
    path: str = ''
    kernel_name: str = ''
    signature: str = ''
    target: str | None = None
    num_warps: int = 1
    num_stages: int = 3
    out_name: str | None = None
    out_path: Path | None = None


desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""


def main():
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="The target to compile towards, in format of '<backend>:<arch>:<warp-size>'; "
             "e.g., 'cuda:80:32', 'hip:gfx942:64'. Default to None, which means using current machine's GPU target")
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    cli_args = parser.parse_args()
    args = CompileArgs(**vars(cli_args))  # A sanity check to ensure class CompileArgs is updated as well.
    compile_kernel(args)


def compile_kernel(args: CompileArgs):
    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

    def _validate_ptxas_override_for_cuda() -> None:
        if not os.environ.get("TRITON_PTXAS_PATH"):
            return
        try:
            from triton import knobs
            ptxas_version = knobs.nvidia.ptxas.version
        except Exception:
            return

        version_match = re.match(r"^(\d+)(?:\.(\d+))?", str(ptxas_version))
        if not version_match:
            return
        major = int(version_match.group(1))
        if triton.__version__.startswith("3.4.") and major >= 13:
            raise RuntimeError(
                "TRITON_PTXAS_PATH points to a CUDA "
                f"{ptxas_version} ptxas, but Triton {triton.__version__} cannot map CUDA 13+ "
                "toolkit versions to PTX ISA versions. Unset TRITON_PTXAS_PATH to use Triton's "
                "bundled ptxas, or use a CUDA 12.x ptxas override."
            )

    def _parse_target_spec(spec: str | None) -> tuple[str | None, str | None, int | None]:
        if not spec:
            return None, None, None
        parts = spec.split(":")
        backend = parts[0] if parts else None
        arch = parts[1] if len(parts) > 1 else None
        warp = int(parts[2]) if len(parts) > 2 else None
        return backend, arch, warp

    def _expected_hip_warp_size(arch: str | int | None) -> int | None:
        if arch is None:
            return None
        if isinstance(arch, int):
            arch_num = arch
        else:
            arch_str = str(arch)
            if arch_str.startswith("gfx"):
                arch_str = arch_str[3:]
            if not arch_str.isdigit():
                return None
            arch_num = int(arch_str)
        major = arch_num // 100
        return 64 if major < 10 else 32

    backend, arch, warp_size = _parse_target_spec(args.target)
    if backend == "hip":
        expected = _expected_hip_warp_size(arch)
        if expected is not None and warp_size is not None and warp_size != expected:
            raise ValueError(
                f"Invalid HIP warp size {warp_size} for target '{args.target}'; "
                f"expected {expected} based on arch {arch}."
            )
    elif backend in (None, "cuda"):
        _validate_ptxas_override_for_cuda()

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path).resolve()
    root = Path(__file__).resolve().parents[1]
    try:
        rel = arg_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Kernel path {arg_path} must live under {root} when using the AOT compiler."
        ) from exc
    module_name = ".".join(rel.with_suffix("").parts)
    mod = importlib.import_module(module_name)
    kernel = getattr(mod, args.kernel_name)
    try:
        from triton.runtime.autotuner import Autotuner  # type: ignore
    except ImportError:
        Autotuner = None  # type: ignore

    if Autotuner is not None and isinstance(kernel, Autotuner):  # type: ignore[arg-type]
        kernel = kernel.fn

    # validate and parse signature
    signature_parts = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def _sanitize_ident(s: str) -> str:
        s = re.sub(r'[^0-9A-Za-z_]', '_', s)
        if re.match(r'^[0-9]', s):
            s = '_' + s
        return s

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i,): constexpr(s.split(":")[1]) for i, s in enumerate(signature_parts) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature_parts)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature_by_name = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature_parts)}
    for key in constants:
        signature_by_name[key] = 'constexpr'

    # Triton expects signature mapping from argument name -> dtype string.
    ast_signature = {name: signature_by_name[name] for name in kernel.arg_names}
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    ast_source_signature = inspect.signature(triton.compiler.ASTSource)
    if "constexprs" in ast_source_signature.parameters:
        src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=ast_signature, attrs=attrs)
    else:
        src = triton.compiler.ASTSource(fn=kernel, signature=ast_signature, constants=constants, attrs=attrs)

    target = None
    if args.target is not None:
        # Accept targets of the form "<backend>:<arch>:<warp-size>" (warp size currently unused but parsed for validation).
        parts = [p.strip() for p in args.target.split(":")]
        if len(parts) != 3 or any(not p for p in parts):
            raise ValueError(
                f"Invalid target specification '{args.target}'. Expected format '<backend>:<arch>:<warp-size>'.")
        backend, arch_str, warp_str = parts
        try:
            arch = int(arch_str)
        except ValueError:
            arch = arch_str
        try:
            warp_size = int(warp_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid warp size '{warp_str}' in target '{args.target}'. Must be an integer.") from exc
        try:
            from triton.backends.compiler import GPUTarget  # type: ignore
            target = GPUTarget(backend=backend, arch=arch, warp_size=warp_size)
        except Exception:
            target = (backend, arch)
    kwargs = {"num_warps": args.num_warps, "num_stages": args.num_stages}

    # Use the provided kernel name as the kernel symbol name.
    safe_base = _sanitize_ident(args.kernel_name)
    func_name = safe_base
    original_repr = getattr(kernel, "_repr", None)
    kernel._repr = lambda _: func_name  # type: ignore[assignment]
    try:
        ccinfo = triton.compile(src, target=target, options=kwargs)
    finally:
        kernel._repr = original_repr  # type: ignore[assignment]

    entry_name = func_name
    ptx_src = ccinfo.asm.get("ptx")
    
    if isinstance(ptx_src, (bytes, bytearray)):
        ptx_src = ptx_src.decode("utf-8", errors="ignore")

    if ptx_src:
        match = re.search(r"\.entry\s+([^(]+)", ptx_src)
        if match:
            entry_name = match.group(1).strip()
    else:
        amdgcn_src = ccinfo.asm.get("amdgcn")
        if isinstance(amdgcn_src, (bytes, bytearray)):
            amdgcn_src = amdgcn_src.decode("utf-8", errors="ignore")
        if amdgcn_src:
            match = re.search(r"\.amdhsa_kernel\s+([^\s;]+)", amdgcn_src)
            if not match:
                match = re.search(r"\.globl\s+([^\s;]+)", amdgcn_src)
            if match:
                entry_name = match.group(1).strip()

    def _extract_arglist(ir: str, name: str) -> str | None:
        match = re.search(rf"define\s+.*@{re.escape(name)}\s*\(", ir)
        if not match:
            return None
        start = match.end() - 1
        depth = 0
        for idx in range(start, len(ir)):
            ch = ir[idx]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return ir[start + 1:idx]
        return None

    def _count_arglist_args(arglist: str) -> int:
        args = []
        depth = 0
        current = []
        for ch in arglist:
            if ch == ',' and depth == 0:
                token = ''.join(current).strip()
                if token:
                    args.append(token)
                current = []
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(depth - 1, 0)
            current.append(ch)
        token = ''.join(current).strip()
        if token:
            args.append(token)
        return len(args)

    def _count_entry_params_from_llir(ir: str, name: str) -> int:
        arglist = _extract_arglist(ir, name)
        if not arglist:
            return 0
        return _count_arglist_args(arglist)

    def _count_first_entry_params_from_llir(ir: str) -> int:
        match = re.search(r"define\s+.*@([^\s(]+)\s*\(", ir)
        if not match:
            return 0
        return _count_entry_params_from_llir(ir, match.group(1))

    metadata = ccinfo.metadata
    if isinstance(metadata, dict):
        metadata_shared_mem = int(metadata.get("shared", 0))
        metadata_num_warps = int(metadata.get("num_warps", kwargs.get("num_warps", 1)))
        metadata_global_scratch = int(metadata.get("global_scratch_size", 0))
    else:
        metadata_shared_mem = metadata.shared
        metadata_num_warps = metadata.num_warps
        metadata_global_scratch = getattr(metadata, "global_scratch_size", 0)

    arg_names = []
    arg_types = []
    arg_names_not_1 = []
    arg_types_not_1 = []
    for i, arg_name in enumerate(kernel.arg_names):
        if arg_name not in constants:
            arg_names.append(arg_name)
            arg_types.append(signature_by_name[arg_name])
            arg_names_not_1.append(arg_name)
            arg_types_not_1.append(signature_by_name[arg_name])
        elif hints.get((i,), None) == 1:
            arg_names.append(arg_name)
            arg_types.append("i32")

    if 'ptx' in ccinfo.asm:
        extensions = [dict(name='ptx', is_binary=False), dict(name='cubin', is_binary=True)]
    elif 'amdgcn' in ccinfo.asm:
        extensions = [dict(name='amdgcn', is_binary=False), dict(name='hsaco', is_binary=True)]
    else:
        raise ValueError("Unsupported backend in compiled artifact")

    llir_src = ccinfo.asm.get("llir")
    if isinstance(llir_src, (bytes, bytearray)):
        llir_src = llir_src.decode("utf-8", errors="ignore")
    
    arg_count = 0
    if llir_src:
        arg_count = _count_entry_params_from_llir(llir_src, entry_name)
        if arg_count == 0:
            arg_count = _count_first_entry_params_from_llir(llir_src)
    
    if arg_count == 0:
        raise ValueError(f"Could not determine argument count for kernel {args.kernel_name} from llir")

    output_files = []
    for extension_info in extensions:
        extension = extension_info['name']
        is_binary = extension_info['is_binary']
        asm = ccinfo.asm[extension]

        with open(out_path.with_suffix('.' + extension), 'w' + ('b' if is_binary else '')) as f:
            f.write(asm)
            output_files.append(f.name)
        kv = {
            "function_name": entry_name,
            "num_warps": metadata_num_warps,
            "smem_bytes": metadata_shared_mem,
            "swizzle_size": 1,
            "global_scratch_size": metadata_global_scratch,
        }
        # For elementwise-style kernels we expect BLOCK_SIZE meta to be present.
        if "BLOCK_SIZE" in constants:
            kv["block_size"] = constants["BLOCK_SIZE"]
        # add meta arguments
        for k, v in constants.items():
            kv[k.lower()] = v
            if k.upper() == "BLOCK_W":
                kv["block_pixels"] = v
        kv["arg_count"] = arg_count

        write_kinfo_and_macro(out_path, func_name, kv)

    return func_name, output_files


if __name__ == "__main__":
    main()
