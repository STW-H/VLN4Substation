#!/usr/bin/env python3
"""Convert LAS/LAZ integer coordinates to real-world-coordinate binary PLY."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.las import write_las_real_ply  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert LAS/LAZ to real-coordinate PLY.")
    parser.add_argument("input", type=Path, help="Input LAS/LAZ file")
    parser.add_argument("--output", type=Path, required=True, help="Output binary PLY file")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Points per streaming chunk")
    parser.add_argument("--metadata", type=Path, help="Optional JSON metadata output")
    args = parser.parse_args()

    write_las_real_ply(args.input, args.output, args.chunk_size, args.metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
