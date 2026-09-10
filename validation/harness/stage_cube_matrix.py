"""Expand the frozen cube matrix into reviewable per-case CASCADE settings."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


TREE_ROOT = Path(
    "/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/trees_cache"
)
TREE_FAMILY = "a50c006ac491d07e399f94f1be37cfe0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-settings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    args = parser.parse_args()

    base = json.loads(args.base_settings.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        for fluid in ("blood", "water"):
            raw = copy.deepcopy(base)
            raw["network"]["input_path"] = str(TREE_ROOT / f"tree_{TREE_FAMILY}_t{target}.tree.npz")
            raw["network"]["target_terminal_count"] = int(target)
            raw["simulation"]["fluid"] = fluid
            raw["simulation"]["tissue_accel"] = "gpu" if target >= 1_000_000 else "cpu"
            raw["outputs"]["out_dir"] = (
                f"../../runs/{args.campaign_id}/cube-{target}-{fluid}/cascade-output"
            )
            raw["outputs"]["prefix"] = f"cube_{target}_{fluid}"
            path = args.output_dir / f"cube-{target}-{fluid}-settings.json"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
