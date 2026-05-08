from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LongMemEval-S into the local benchmark cache.")
    parser.add_argument("--url", default=LONGMEMEVAL_S_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/longmemeval/cache/longmemeval_s.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        print(f"exists={args.output}")
        print(f"sha256={sha256(args.output)}")
        return

    request = urllib.request.Request(args.url, headers={"User-Agent": "hawkes-rag-benchmark"})
    with urllib.request.urlopen(request, timeout=120) as response:
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
