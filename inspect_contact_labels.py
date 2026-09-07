#!/usr/bin/env python3
"""
Load deploy contact label .npy files and visualize per-channel contact as a Gantt chart.

Expected file format (same as extract_contact_labels_from_motion / RLContactPolicy):
  data = np.load(path, allow_pickle=True).item()
  data["contact_mask"]  -> float array, shape (T, C)

Default channel names follow rl_policy conventions:
  C=4: left_foot, right_foot, left_wrist, right_wrist
  C=5: + pelvis_seat
  C=8: L/R foot and wrist split into _env / _obj
  C=10: 8-way limbs + pelvis_env, pelvis_obj

Usage:
  python inspect_contact_labels.py path/to/contact_labels.npy
  python inspect_contact_labels.py contacts.npy --fps 50 --save out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for inspect_contact_labels.py. "
        "Install with: pip install matplotlib"
    ) from e


def default_channel_names(num_channels: int) -> list[str]:
    if num_channels == 4:
        return ["left_foot", "right_foot", "left_wrist", "right_wrist"]
    if num_channels == 5:
        return [
            "left_foot",
            "right_foot",
            "left_wrist",
            "right_wrist",
            "pelvis_seat",
        ]
    if num_channels == 8:
        return [
            "left_foot_env",
            "left_foot_obj",
            "right_foot_env",
            "right_foot_obj",
            "left_wrist_env",
            "left_wrist_obj",
            "right_wrist_env",
            "right_wrist_obj",
        ]
    if num_channels == 10:
        return [
            "left_foot_env",
            "left_foot_obj",
            "right_foot_env",
            "right_foot_obj",
            "left_wrist_env",
            "left_wrist_obj",
            "right_wrist_env",
            "right_wrist_obj",
            "pelvis_env",
            "pelvis_obj",
        ]
    return [f"ch{i}" for i in range(num_channels)]


def load_contact_labels(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    if raw.ndim != 0 or raw.dtype != object:
        raise ValueError(
            f"Expected object array .npy (dict from np.save(..., allow_pickle=True)), "
            f"got shape={raw.shape} dtype={raw.dtype}"
        )
    data = raw.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in .npy file, got {type(data)}")
    if "contact_mask" not in data:
        raise KeyError(
            f"Missing 'contact_mask' key. Keys present: {sorted(data.keys())}"
        )
    mask = np.asarray(data["contact_mask"], dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"contact_mask must be 2D (T, C), got shape {mask.shape}")
    return data, mask


def binary_contact_runs(active: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True segments as (start_index, length) for broken_barh."""
    m = np.asarray(active, dtype=bool)
    if m.size == 0:
        return []
    padded = np.concatenate((np.array([False]), m, np.array([False])))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(float(s), float(e - s)) for s, e in zip(starts, ends)]


def print_summary(
    path: Path,
    data: dict[str, Any],
    mask: np.ndarray,
    names: list[str],
    threshold: float,
) -> None:
    t, c = mask.shape
    print(f"File: {path}")
    print(f"contact_mask shape: ({t}, {c})  dtype={mask.dtype}")
    other = sorted(k for k in data if k != "contact_mask")
    if other:
        print(f"Other keys: {other}")
    on = mask >= threshold
    print(f"Active threshold: >= {threshold}")
    for i, name in enumerate(names):
        frac = float(on[:, i].mean()) if t else 0.0
        print(f"  {name}: {100.0 * frac:.1f}% frames active")


def plot_contact_gantt(
    mask: np.ndarray,
    names: list[str],
    *,
    fps: float | None,
    t0: int,
    t1: int,
    threshold: float,
    title: str | None,
) -> tuple[plt.Figure, Axes]:
    mask = np.asarray(mask, dtype=np.float32)
    t_total, c = mask.shape
    t0 = max(0, int(t0))
    t1 = min(t_total, int(t1))
    if t1 <= t0:
        raise ValueError(f"Empty time range after clip: t0={t0} t1={t1} (T={t_total})")

    sl = mask[t0:t1]
    on = sl >= threshold
    n = sl.shape[0]

    use_time_axis = fps is not None and fps > 0
    if use_time_axis:
        x = (t0 + np.arange(n, dtype=np.float64)) / float(fps)
        xlabel = "Time (s)"
    else:
        x = t0 + np.arange(n, dtype=np.float64)
        xlabel = "Frame index"

    fig_h = max(3.0, 0.35 * c + 1.5)
    if use_time_axis:
        fig_w = min(24.0, max(8.0, n / float(fps) * 2.0))
    else:
        fig_w = min(24.0, max(8.0, n * 0.02))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = plt.cm.tab20(np.linspace(0, 1, max(c, 1), endpoint=False))

    bar_height = 0.7
    for row, name in enumerate(names):
        y_base = row
        runs = binary_contact_runs(on[:, row])
        if use_time_axis:
            xranges = [((t0 + s) / float(fps), w / float(fps)) for s, w in runs]
        else:
            xranges = [(t0 + s, w) for s, w in runs]
        for xr in xranges:
            ax.broken_barh([xr], (y_base, bar_height), facecolors=colors[row % len(colors)])

    ax.set_yticks(np.arange(c) + bar_height / 2)
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    ax.set_xlim(x[0], x[-1] if n > 1 else x[0] + 1)
    ax.set_ylim(-0.5, c - 0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ttl = title or "Contact labels (Gantt)"
    ax.set_title(ttl)
    fig.tight_layout()
    return fig, ax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "contact_labels",
        type=Path,
        help="Path to .npy dict with contact_mask (T, C)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="If set, horizontal axis is time in seconds (frame / fps)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Values >= threshold count as contact (default 0.5)",
    )
    p.add_argument(
        "--t-start",
        type=int,
        default=0,
        help="First frame index (inclusive)",
    )
    p.add_argument(
        "--t-end",
        type=int,
        default=None,
        help="Last frame index (exclusive); default = full length",
    )
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated y-axis labels (must match number of channels)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path (png/pdf/svg/...) instead of or in addition to showing",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (use with --save)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.contact_labels.expanduser().resolve()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    data, mask = load_contact_labels(path)
    c = mask.shape[1]
    if args.names:
        names = [s.strip() for s in args.names.split(",") if s.strip()]
        if len(names) != c:
            print(
                f"--names: expected {c} comma-separated labels, got {len(names)}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        names = default_channel_names(c)

    print_summary(path, data, mask, names, args.threshold)

    t_end = args.t_end if args.t_end is not None else mask.shape[0]
    fig, _ = plot_contact_gantt(
        mask,
        names,
        fps=args.fps,
        t0=args.t_start,
        t1=t_end,
        threshold=args.threshold,
        title=args.title,
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
