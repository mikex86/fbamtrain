from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


_FLASH_ATTENTION_COMMIT = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
_FLASH_ATTENTION_ARCHIVE = f"flash-attention-{_FLASH_ATTENTION_COMMIT}.tar.gz"
_FLASH_ATTENTION_URL = (
    f"https://github.com/Dao-AILab/flash-attention/archive/{_FLASH_ATTENTION_COMMIT}.tar.gz"
)
_PATCH_DIR = Path(__file__).resolve().parent / "patches"
_PATCHES = ("flash-attention-fp16-accum.patch",)


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = destination / member.name
        if not str(member_path.resolve()).startswith(str(destination)):
            raise RuntimeError("Refusing to extract tarball with unsafe paths")
    tar.extractall(path=destination)


def ensure_flash_attention_sources(root: Path) -> Path:
    third_party = root / "third_party"
    flash_root = third_party / "flash-attention"
    marker_path = flash_root / "flash_attn" / "__init__.py"
    patch_marker = flash_root / ".fbamtrain_patches_applied"
    expected_marker = json.dumps({"commit": _FLASH_ATTENTION_COMMIT, "patches": list(_PATCHES)})

    if marker_path.exists() and patch_marker.exists():
        if patch_marker.read_text().strip() == expected_marker:
            return flash_root
        shutil.rmtree(flash_root)
    elif marker_path.exists():
        shutil.rmtree(flash_root)

    third_party.mkdir(parents=True, exist_ok=True)

    archive_path = third_party / _FLASH_ATTENTION_ARCHIVE
    if not archive_path.exists():
        with urllib.request.urlopen(_FLASH_ATTENTION_URL) as response, open(archive_path, "wb") as output:
            shutil.copyfileobj(response, output)

    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract(tar, third_party)

    extracted_root = third_party / f"flash-attention-{_FLASH_ATTENTION_COMMIT}"
    if flash_root.exists():
        shutil.rmtree(flash_root)
    extracted_root.rename(flash_root)

    try:
        archive_path.unlink()
    except FileNotFoundError:
        pass

    for patch_name in _PATCHES:
        patch_path = _PATCH_DIR / patch_name
        if not patch_path.exists():
            raise RuntimeError(f"Missing FlashAttention patch file: {patch_path}")
        subprocess.run(["patch", "-p1", "-N", "-i", str(patch_path)], cwd=flash_root, check=True)

    patch_marker.write_text(expected_marker)
    return flash_root
