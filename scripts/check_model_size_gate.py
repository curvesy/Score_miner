#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from score_miner_core.public_detect.size_gate import assert_size_under


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-mb", type=float, default=30.0)
    args = parser.parse_args()

    size = assert_size_under(args.path, args.max_mb)
    print(f"PASS {args.path} size={size:.2f}MB max={args.max_mb:.2f}MB")


if __name__ == "__main__":
    main()
