from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path


_CUTLASS_VERSION = "4.3.5"
_CUTLASS_ARCHIVE = f"cutlass-{_CUTLASS_VERSION}.tar.gz"
_CUTLASS_URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/v{_CUTLASS_VERSION}.tar.gz"


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = destination / member.name
        if not str(member_path.resolve()).startswith(str(destination)):
            raise RuntimeError("Refusing to extract tarball with unsafe paths")
    tar.extractall(path=destination)

def fetch_cutlass_sources(root: Path) -> Path:
    third_party = root / "third_party"
    cutlass_root = third_party / "cutlass"
    header_path = cutlass_root / "include" / "cutlass" / "cutlass.h"
    third_party.mkdir(parents=True, exist_ok=True)

    for stale_dir in third_party.glob("cutlass_*"):
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)

    if header_path.exists():
        version_header = cutlass_root / "include" / "cutlass" / "version.h"
        if version_header.exists():
            text = version_header.read_text()
            major = _CUTLASS_VERSION.split(".")[0]
            minor = _CUTLASS_VERSION.split(".")[1]
            patch = _CUTLASS_VERSION.split(".")[2]
            if (
                f"#define CUTLASS_MAJOR {major}" in text
                and f"#define CUTLASS_MINOR {minor}" in text
                and f"#define CUTLASS_PATCH {patch}" in text
            ):
                return cutlass_root
        shutil.rmtree(cutlass_root)

    archive_path = third_party / _CUTLASS_ARCHIVE
    if not archive_path.exists():
        with urllib.request.urlopen(_CUTLASS_URL) as response, open(archive_path, "wb") as output:
            shutil.copyfileobj(response, output)

    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract(tar, third_party)

    extracted_root = third_party / f"cutlass-{_CUTLASS_VERSION}"
    if cutlass_root.exists():
        shutil.rmtree(cutlass_root)
    extracted_root.rename(cutlass_root)

    try:
        archive_path.unlink()
    except FileNotFoundError:
        pass

    return cutlass_root


def main() -> None:
    root = Path(__file__).resolve().parent
    cutlass_root = fetch_cutlass_sources(root)
    print(f"CUTLASS sources ready at {cutlass_root}")


if __name__ == "__main__":
    main()
