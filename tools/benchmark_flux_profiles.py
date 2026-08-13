#!/usr/bin/env python3
"""Create a repeatable FLUX.2 Klein fixed-seed comparison manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt_workbench import build_flux_fixed_seed_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the four-case FLUX.2 Klein matrix: distilled/base crossed "
            "with official/abliterated Qwen3, all using one prompt and seed."
        )
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Corrected FLUX prompt text.")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file containing the corrected FLUX prompt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flux_benchmark_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else args.prompt
    )
    manifest = build_flux_fixed_seed_benchmark(prompt, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest['cases'])} benchmark cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
