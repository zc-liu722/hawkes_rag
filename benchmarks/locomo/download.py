from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


LOCOMO10_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official LoCoMo 10-conversation JSON.")
    parser.add_argument("--url", default=LOCOMO10_URL)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/locomo/cache/locomo10.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"exists={args.output}")
        print(f"sha256={sha256(args.output)}")
        return

    with urllib.request.urlopen(args.url, timeout=60) as response:
        payload = response.read()
    args.output.write_bytes(payload)
    print(f"downloaded={args.output}")
    print(f"bytes={len(payload)}")
    print(f"sha256={sha256(args.output)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
