import re
from pathlib import Path
from typing import Union, Mapping


def _sanitize_ident(s: str) -> str:
    """Turn arbitrary keys into safe assembler identifiers."""
    s = re.sub(r'[^0-9A-Za-z_]', '_', str(s))
    if re.match(r'^[0-9]', s):
        s = '_' + s
    return s


def _is_int_like(v: Union[int, str]) -> bool:
    """True for ints or strings that parse as int (supports 0x hex)."""
    if isinstance(v, int):
        return True
    if not isinstance(v, str):
        return False
    try:
        int(v, 0)
        return True
    except ValueError:
        return False


def _escape_asciz(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"')


def write_kinfo_and_macro(
        out_path: Path,
        kernel_name: str,
        kv: Mapping[str, Union[int, str]],
        *,
        align: int = 3
):
    macro_asm = out_path.with_suffix(f".kinfo.inc.asm")
    with open(macro_asm, 'w', encoding='utf-8') as f:
        f.write(f"/* auto-generated: DO NOT EDIT */\n")

        # undef DECLARE_KINFO to allow multiple includes
        f.write(f"#ifndef DECLARE_KINFO_SEEN\n")
        f.write(f"#define DECLARE_KINFO_SEEN 1\n\n")
        f.write(f"#else\n")
        f.write(f".purgem DECLARE_KINFO\n")
        f.write(f"#endif\n\n")

        f.write(f".macro DECLARE_KINFO prefix, align={align}\n\n")

        f.write("#if defined(__ELF__)\n")
        f.write("\t.section .rodata.kernels,\"a\",@progbits\n")
        f.write("#elif defined(__MACH__)\n")
        f.write("\t.section __TEXT,__const\n")
        f.write("#else\n")
        f.write("\t.section .rdata\n")
        f.write("#endif\n\n")

        f.write("\t.p2align \\align\n\n")

        for key in kv.keys():
            sym = _sanitize_ident(key)
            f.write(f"\t.globl \\prefix\\()_{sym}\n")
        f.write("\n")

        for key, val in kv.items():
            sym = _sanitize_ident(key)
            if _is_int_like(val):
                intval = int(val, 0) if not isinstance(val, int) else val
                f.write("\t.p2align 3\n")  # align for .quad
                f.write("#if defined(__ELF__)\n")
                f.write(f"\t.type \\prefix\\()_{sym}, @object\n")
                f.write("#endif\n")
                f.write(f"\\prefix\\()_{sym}:\n")
                f.write(f"\t.quad {intval}\n")
                f.write("#if defined(__ELF__)\n")
                f.write(f"\t.size \\prefix\\()_{sym}, .-\\prefix\\()_{sym}\n")
                f.write("#endif\n\n")
            else:
                s = _escape_asciz(str(val))
                f.write("\t.p2align 0\n")
                f.write("#if defined(__ELF__)\n")
                f.write(f"\t.type \\prefix\\()_{sym}, @object\n")
                f.write("#endif\n")
                f.write(f"\\prefix\\()_{sym}:\n")
                f.write(f'\t.asciz "{s}"\n')
                f.write("#if defined(__ELF__)\n")
                f.write(f"\t.size \\prefix\\()_{sym}, .-\\prefix\\()_{sym}\n")
                f.write("#endif\n\n")

        f.write(".endm\n")
