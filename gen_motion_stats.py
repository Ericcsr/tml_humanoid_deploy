#!/usr/bin/env python3
"""Summarize total motion duration by name-prefix category.

Uses the same motion discovery rules as ``run_mujoco_eval`` (subdir, flat terrain, or
free-space layouts). Frame counts come from deploy ``.npz`` files (``joint_pos`` length).
Duration assumes a fixed FPS (default 50).

Example::

    python run_motion_statistics.py \\
        --motion-dir /path/to/merged_motions \\
        --category terrain:obstacles1,lafan \\
        --category object:omomo,hand \\
        --category free:amass
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from run_mujoco_eval import _batch_discovery_hint, _discover_batch_jobs

_FRAME_KEYS = ("joint_pos", "qpos", "dof_pos")


@dataclass(frozen=True)
class MotionCategory:
    name: str
    prefixes: tuple[str, ...]


def _parse_category(spec: str) -> MotionCategory:
    """Parse ``NAME:prefix1,prefix2`` or ``NAME=prefix1,prefix2``."""
    for sep in (":", "="):
        if sep in spec:
            name, rest = spec.split(sep, 1)
            break
    else:
        raise argparse.ArgumentTypeError(
            f"Category {spec!r} must be NAME:prefix1,prefix2 (or NAME=prefix1,prefix2)"
        )
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Empty category name in {spec!r}")
    prefixes = tuple(p.strip() for p in rest.split(",") if p.strip())
    if not prefixes:
        raise argparse.ArgumentTypeError(f"No prefixes for category {name!r} in {spec!r}")
    return MotionCategory(name=name, prefixes=prefixes)


def _motion_num_frames(npz_path: Path) -> int:
    data = np.load(npz_path)
    for key in _FRAME_KEYS:
        if key in data:
            n = int(data[key].shape[0])
            if n <= 0:
                raise ValueError(f"{npz_path}: {key} has zero frames")
            return n
    keys = ", ".join(sorted(data.files))
    raise KeyError(f"{npz_path}: expected one of {_FRAME_KEYS}; found keys: {keys}")


def _classify_motion(label: str, categories: list[MotionCategory]) -> str | None:
    for cat in categories:
        if any(label.startswith(p) for p in cat.prefixes):
            return cat.name
    return None


def _collect_motions(motion_dir: Path) -> list[tuple[str, Path]]:
    jobs = _discover_batch_jobs(motion_dir)
    if not jobs:
        raise FileNotFoundError(
            f"No motions under {motion_dir}. {_batch_discovery_hint(motion_dir)}"
        )
    return [(name, npz) for name, npz, *_ in jobs]


def run_statistics(
    motion_dir: Path,
    categories: list[MotionCategory],
    *,
    fps: float = 50.0,
    verbose: bool = False,
) -> dict[str, dict[str, float | int]]:
    motions = _collect_motions(motion_dir)
    per_cat_frames: dict[str, int] = defaultdict(int)
    per_cat_count: dict[str, int] = defaultdict(int)
    uncategorized: list[tuple[str, int, Path]] = []

    for label, npz_path in motions:
        n_frames = _motion_num_frames(npz_path)
        cat = _classify_motion(label, categories)
        if cat is None:
            uncategorized.append((label, n_frames, npz_path))
            continue
        per_cat_frames[cat] += n_frames
        per_cat_count[cat] += 1
        if verbose:
            print(
                f"  {label:<48} {cat:<12} {n_frames:>6} frames  "
                f"{n_frames / fps:>8.2f}s  {npz_path.name}",
                flush=True,
            )

    total_frames = sum(per_cat_frames.values()) + sum(n for _, n, _ in uncategorized)
    rows: dict[str, dict[str, float | int]] = {}
    for cat in categories:
        frames = per_cat_frames.get(cat.name, 0)
        rows[cat.name] = {
            "count": per_cat_count.get(cat.name, 0),
            "frames": frames,
            "seconds": frames / fps,
        }

    rows["_uncategorized"] = {
        "count": len(uncategorized),
        "frames": sum(n for _, n, _ in uncategorized),
        "seconds": sum(n for _, n, _ in uncategorized) / fps,
    }
    rows["_total_discovered"] = {
        "count": len(motions),
        "frames": total_frames,
        "seconds": total_frames / fps,
    }
    rows["_total_classified"] = {
        "count": sum(per_cat_count.values()),
        "frames": sum(per_cat_frames.values()),
        "seconds": sum(per_cat_frames.values()) / fps,
    }

    if uncategorized and verbose:
        print("\nUncategorized:", flush=True)
        for label, n_frames, npz_path in uncategorized:
            print(
                f"  {label:<48} {'—':<12} {n_frames:>6} frames  "
                f"{n_frames / fps:>8.2f}s  {npz_path.name}",
                flush=True,
            )

    return rows


def _print_summary(
    motion_dir: Path,
    categories: list[MotionCategory],
    rows: dict[str, dict[str, float | int]],
    *,
    fps: float,
) -> None:
    print(f"Motion directory: {motion_dir}", flush=True)
    print(f"FPS assumption:   {fps:g}", flush=True)
    print(f"Categories ({len(categories)}):", flush=True)
    for cat in categories:
        pref = ", ".join(cat.prefixes)
        print(f"  {cat.name}: starts with [{pref}]", flush=True)

    print(f"\n{'category':<20} {'motions':>8} {'frames':>10} {'seconds':>12}", flush=True)
    for cat in categories:
        r = rows[cat.name]
        print(
            f"{cat.name:<20} {int(r['count']):>8} {int(r['frames']):>10} {float(r['seconds']):>12.2f}",
            flush=True,
        )

    uncat = rows["_uncategorized"]
    if int(uncat["count"]) > 0:
        print(
            f"{'(uncategorized)':<20} {int(uncat['count']):>8} "
            f"{int(uncat['frames']):>10} {float(uncat['seconds']):>12.2f}",
            flush=True,
        )

    classified = rows["_total_classified"]
    discovered = rows["_total_discovered"]
    print(
        f"\n{'TOTAL (classified)':<20} {int(classified['count']):>8} "
        f"{int(classified['frames']):>10} {float(classified['seconds']):>12.2f}",
        flush=True,
    )
    if int(discovered["count"]) != int(classified["count"]):
        print(
            f"{'TOTAL (discovered)':<20} {int(discovered['count']):>8} "
            f"{int(discovered['frames']):>10} {float(discovered['seconds']):>12.2f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute total motion duration per name-prefix category (deploy .npz, fixed FPS)."
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        required=True,
        help="Root folder with motions (same layouts as run_mujoco_eval batch mode).",
    )
    parser.add_argument(
        "--category",
        action="append",
        type=_parse_category,
        required=True,
        metavar="NAME:prefix1,prefix2",
        help="Category name and comma-separated motion-name prefixes. Repeat for each category.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Assumed motion frame rate for converting frames to seconds (default: 50).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print one line per classified / uncategorized motion.",
    )
    args = parser.parse_args()

    motion_dir = args.motion_dir.expanduser().resolve()
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"--motion-dir is not a directory: {motion_dir}")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    categories: list[MotionCategory] = list(args.category)
    if args.verbose:
        print(f"Per-motion breakdown ({motion_dir}):", flush=True)
    rows = run_statistics(motion_dir, categories, fps=args.fps, verbose=args.verbose)
    _print_summary(motion_dir, categories, rows, fps=args.fps)

    if int(rows["_uncategorized"]["count"]) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()