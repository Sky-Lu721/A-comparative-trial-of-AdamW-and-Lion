"""Download the official CIFAR-10 archive in verified, resumable byte ranges.

An optional alternative when a long torchvision download is interrupted.
Training still validates and extracts the archive through torchvision.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import _common  # Make src importable when launched as a direct script.
from src.utils import load_config, project_path
from torchvision.datasets import CIFAR10


def prepare_data(config_path: Path | None, workers: int, chunk_bytes: int) -> None:
    """Fetch official ranges and require torchvision's published MD5."""
    if workers < 1 or chunk_bytes < 1:
        raise ValueError("workers and chunk_bytes must be positive")
    config = load_config(config_path)
    root = project_path(config["dataset"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    archive = root / CIFAR10.filename
    if archive.exists():
        checksum = hashlib.md5()
        with archive.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                checksum.update(block)
        if checksum.hexdigest() == CIFAR10.tgz_md5:
            print("Official archive checksum already verified.")
            return
    # Follow the dataset library's official URL and validate TLS normally.
    curl = shutil.which("curl.exe") if os.name == "nt" else None
    if curl:
        response = subprocess.run(
            [curl, "--ssl-revoke-best-effort", "--silent", "--show-error", "--fail",
             "--head", "--location", "--retry", "3", "--max-time", "30",
             "--write-out", "\nEffective-URL: %{url_effective}\n", CIFAR10.url],
            capture_output=True, text=True, check=True,
        )
        metadata = dict(line.split(":", 1) for line in response.stdout.splitlines()
                        if ":" in line)
        url = metadata["Effective-URL"].strip()
        size = int(metadata["Content-Length"].strip())
        if metadata.get("Accept-Ranges", "").strip() != "bytes":
            raise RuntimeError("Official server does not support byte ranges")
    else:
        request = urllib.request.Request(CIFAR10.url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            url = response.url
            size = int(response.headers["Content-Length"])
            if response.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("Official server does not support byte ranges")
    parts = root / "download_parts"
    parts.mkdir(exist_ok=True)
    ranges = [(start, min(start + chunk_bytes, size) - 1)
              for start in range(0, size, chunk_bytes)]

    def fetch(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        path = parts / f"{start}_{end}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        temporary = path.with_suffix(".tmp")
        for attempt in range(3):
            try:
                expected = f"bytes {start}-{end}/{size}"
                if curl:
                    # Windows curl uses the native certificate store. TLS
                    # validation stays enabled when revocation is offline.
                    headers = path.with_suffix(".headers")
                    completed = subprocess.run(
                        [curl, "--ssl-revoke-best-effort", "--silent", "--show-error",
                         "--fail", "--range", f"{start}-{end}", "--max-time", "120",
                         "--dump-header", str(headers), "--output", str(temporary), url],
                        capture_output=True, text=True, check=False,
                    )
                    if completed.returncode:
                        raise RuntimeError(completed.stderr.strip())
                    values = [line.split(":", 1)[1].strip()
                              for line in headers.read_text().splitlines()
                              if line.lower().startswith("content-range:")]
                    if not values or values[-1] != expected:
                        raise RuntimeError("Server returned an unexpected byte range")
                else:
                    request = urllib.request.Request(
                        url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if (response.status != 206
                                or response.headers.get("Content-Range") != expected):
                            raise RuntimeError("Server returned an unexpected byte range")
                        with temporary.open("wb") as stream:
                            while block := response.read(64 * 1024):
                                stream.write(block)
                if temporary.stat().st_size != end - start + 1:
                    raise RuntimeError(f"Incomplete download range: {start}-{end}")
                temporary.replace(path)
                return path
            except (OSError, RuntimeError) as error:
                if attempt == 2:
                    raise
                print(f"Retry range {start}-{end}: {error}", flush=True)
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable download state")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, bounds): bounds for bounds in ranges}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"Verified range lengths: {count}/{len(ranges)}", flush=True)
    temporary = archive.with_suffix(".assembling")
    digest = hashlib.md5()
    with temporary.open("wb") as stream:
        for start, end in ranges:
            with (parts / f"{start}_{end}.part").open("rb") as part:
                while block := part.read(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
    if digest.hexdigest() != CIFAR10.tgz_md5:
        raise RuntimeError("Assembled archive failed the official CIFAR-10 MD5 check")
    temporary.replace(archive)
    print(f"Official CIFAR-10 verified: {archive} ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    prepare_data(args.config, args.workers, args.chunk_bytes)
