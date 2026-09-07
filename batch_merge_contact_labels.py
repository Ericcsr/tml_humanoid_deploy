#!/usr/bin/env python3
"""
Batch: foot contact from extract_contact_labels_from_motion.py + wrist from existing
*_contact_nodes.npy -> {motion_stem}_merged_contact_labels.npy

Expected layout (e.g. exported_policies/for_sirui_box_lift_tracking_0501):
  <...>/leaf_dir/<stem>.npz                    (not *_object_fitted.npz)
  <...>/leaf_dir/<stem>_contact_nodes.npy      wrist/box labels; dict with contact_mask (T,4)

Merge rule (same column order as RL: left_foot, right_foot, left_wrist, right_wrist):
  merged[:, 0:2] from extractor output
  merged[:, 2:4] from wrist file's contact_mask

Run from repo root (or pass --repo-root). Requires the same Python env as the extract script
(mujoco, numpy, scipy, etc.).

Example:
  python batch_merge_contact_labels.py \\
    --motion-root exported_policies/for_sirui_box_lift_tracking_0501 \\
    --fps 50 --no-print-foot-heights
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _iter_motion_wrist_pairs(motion_root: Path):
    """
    Yield (motion_npz, wrist_npy, stem) for each valid pair under motion_root.
    """
    motion_root = motion_root.expanduser().resolve()
    for npz in sorted(motion_root.rglob("*.npz")):
        if npz.name.endswith("_object_fitted.npz"):
            continue
        stem = npz.stem
        wrist = npz.parent / f"{stem}_contact_nodes.npy"
        if wrist.is_file():
            yield npz, wrist, stem


def _load_contact_mask(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    d = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
    if not isinstance(d, dict) or "contact_mask" not in d:
        raise ValueError(f"Expected dict with 'contact_mask' in {path}")
    m = np.asarray(d["contact_mask"], dtype=np.float32)
    if m.ndim != 2 or m.shape[1] < 4:
        raise ValueError(f"contact_mask must be (T, >=4), got {m.shape} in {path}")
    return m


def _fps_from_npz(npz_path: Path, default_fps: float | None) -> float:
    z = np.load(npz_path, allow_pickle=True)
    if "fps" in z.files:
        return float(np.asarray(z["fps"]).reshape(-1)[0])
    if "control_dt" in z.files:
        return 1.0 / float(np.asarray(z["control_dt"]).reshape(-1)[0])
    if default_fps is not None:
        return float(default_fps)
    raise ValueError(f"No fps/control_dt in {npz_path}; pass --fps")


def main() -> None:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-root",
        type=Path,
        required=True,
        help="Root folder to scan for .npz + *_contact_nodes.npy pairs",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo,
        help="Deploy repo root (default: directory of this script)",
    )
    parser.add_argument(
        "--extract-script",
        type=Path,
        default=repo / "extract_contact_labels_from_motion.py",
        help="Path to extract_contact_labels_from_motion.py",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=Path("assets/g1/scene_29dof.xml"),
        help="MJCF path relative to repo root or absolute",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override motion Hz when ref .npz has no fps/control_dt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pairs only; do not run extraction or write outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_merged_contact_labels.npy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N motions (for testing)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable for extract script (default: sys.executable)",
    )
    args = parser.parse_args()

    extract_script = args.extract_script.expanduser().resolve()
    if not extract_script.is_file():
        parser.error(f"extract script not found: {extract_script}")

    py = args.python or sys.executable
    motion_root = args.motion_root.expanduser().resolve()
    if not motion_root.is_dir():
        parser.error(f"not a directory: {motion_root}")

    pairs = list(_iter_motion_wrist_pairs(motion_root))
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit)]

    print(f"[batch] found {len(pairs)} motion + wrist pairs under {motion_root}", flush=True)
    if args.dry_run:
        for npz, wrist, stem in pairs:
            print(f"  {stem}\n    npz={npz}\n    wrist={wrist}", flush=True)
        return

    repo_root = args.repo_root.expanduser().resolve()
    ok, fail = 0, 0
    for npz, wrist, stem in pairs:
        out_path = npz.parent / f"{stem}_merged_contact_labels.npy"
        if out_path.is_file() and not args.overwrite:
            print(f"[skip] exists {out_path.name}", flush=True)
            continue
        try:
            fps = _fps_from_npz(npz, args.fps)
        except ValueError as e:
            print(f"[fail] {stem}: {e}", flush=True)
            fail += 1
            continue

        with tempfile.NamedTemporaryFile(suffix="_foot_contact.npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            py,
            str(extract_script),
            "--ref-motion",
            str(npz),
            "--output",
            str(tmp_path),
            "--fps",
            str(fps),
            "--robot-xml",
            str(args.robot_xml),
            "--no-print-foot-heights",
        ]
        try:
            r = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode != 0:
                print(
                    f"[fail] {stem}: extract exited {r.returncode}\n{r.stderr or r.stdout}",
                    flush=True,
                )
                fail += 1
                continue

            foot_mask = _load_contact_mask(tmp_path)
            wrist_mask = _load_contact_mask(wrist)
            tf, tw = foot_mask.shape[0], wrist_mask.shape[0]
            if tf != tw:
                print(
                    f"[warn] {stem}: T mismatch foot={tf} wrist={tw}; trimming to min",
                    flush=True,
                )
            t = min(tf, tw)
            merged = np.zeros((t, 4), dtype=np.float32)
            merged[:, 0:2] = foot_mask[:t, 0:2]
            merged[:, 2:4] = wrist_mask[:t, 2:4]

            np.save(out_path, {"contact_mask": merged}, allow_pickle=True)
            print(
                f"[ok] {out_path.relative_to(motion_root)}  T={t}  fps={fps:g}",
                flush=True,
            )
            ok += 1
        except Exception as e:
            print(f"[fail] {stem}: {e}", flush=True)
            fail += 1
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    print(f"[batch] done  ok={ok}  fail={fail}", flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
