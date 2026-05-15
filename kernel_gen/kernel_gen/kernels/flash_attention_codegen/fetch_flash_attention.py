from __future__ import annotations

from pathlib import Path

from .ensure_flash_attention import ensure_flash_attention_sources


def main() -> None:
    root = Path(__file__).resolve().parent
    flash_root = ensure_flash_attention_sources(root)
    print(f"FlashAttention sources ready at {flash_root}")


if __name__ == "__main__":
    main()
