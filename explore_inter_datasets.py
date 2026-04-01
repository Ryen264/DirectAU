#!/usr/bin/env python3
"""Explore RecBole .inter datasets (Beauty, Gowalla, Yelp).

This script reads .inter files (tab-separated) and reports:
- total interactions
- unique users/items
- interactions per user/item distribution
- duplicate user-item pair count
- timestamp range (if a timestamp column exists)

Usage:
    python explore_inter_datasets.py
    python explore_inter_datasets.py --top-k 5
    python explore_inter_datasets.py --files dataset/Beauty/Beauty.inter dataset/Gowalla/Gowalla.inter
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_FILES = [
    "dataset/Beauty/Beauty.inter",
    "dataset/Gowalla/Gowalla.inter",
    "dataset/Yelp/Yelp.inter",
]


def find_column(columns: Iterable[str], prefix: str) -> Optional[str]:
    """Return the first column whose name starts with `prefix:` or equals prefix."""
    for col in columns:
        if col == prefix or col.startswith(prefix + ":"):
            return col
    return None


def percentile(values: List[int], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    if p <= 0:
        return float(values[0])
    if p >= 100:
        return float(values[-1])

    rank = (len(values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return values[low] * (1 - frac) + values[high] * frac


def format_ts(ts: float) -> str:
    """Convert unix timestamp to readable UTC string."""
    try:
        return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OverflowError, OSError, ValueError):
        return f"invalid timestamp ({ts})"


def summarize_file(file_path: Path, top_k: int) -> Dict[str, object]:
    """Read one .inter file and return summary statistics."""
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()

    interaction_count = 0
    ts_min = None
    ts_max = None

    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Missing header row")

        user_col = find_column(reader.fieldnames, "user_id")
        item_col = find_column(reader.fieldnames, "item_id")
        ts_col = find_column(reader.fieldnames, "timestamp")

        if user_col is None or item_col is None:
            raise ValueError("Required columns user_id/item_id not found")

        for row in reader:
            user = row[user_col]
            item = row[item_col]
            if user is None or item is None:
                continue

            interaction_count += 1
            user_counts[user] += 1
            item_counts[item] += 1
            pair_counts[(user, item)] += 1

            if ts_col:
                ts_value = row.get(ts_col)
                if ts_value not in (None, ""):
                    try:
                        ts_num = float(ts_value)
                        ts_min = ts_num if ts_min is None else min(ts_min, ts_num)
                        ts_max = ts_num if ts_max is None else max(ts_max, ts_num)
                    except ValueError:
                        pass

    n_users = len(user_counts)
    n_items = len(item_counts)
    density = 0.0
    if n_users > 0 and n_items > 0:
        density = interaction_count / float(n_users * n_items)

    user_dist = sorted(user_counts.values())
    item_dist = sorted(item_counts.values())

    duplicate_pairs = sum(v - 1 for v in pair_counts.values() if v > 1)

    summary: Dict[str, object] = {
        "file": str(file_path),
        "interactions": interaction_count,
        "users": n_users,
        "items": n_items,
        "density": density,
        "duplicate_pairs": duplicate_pairs,
        "top_users": user_counts.most_common(top_k),
        "top_items": item_counts.most_common(top_k),
        "user_dist": {
            "min": user_dist[0] if user_dist else 0,
            "p50": percentile(user_dist, 50),
            "p90": percentile(user_dist, 90),
            "p99": percentile(user_dist, 99),
            "max": user_dist[-1] if user_dist else 0,
        },
        "item_dist": {
            "min": item_dist[0] if item_dist else 0,
            "p50": percentile(item_dist, 50),
            "p90": percentile(item_dist, 90),
            "p99": percentile(item_dist, 99),
            "max": item_dist[-1] if item_dist else 0,
        },
        "timestamp": {
            "has_timestamp": ts_min is not None and ts_max is not None,
            "min": ts_min,
            "max": ts_max,
        },
    }
    return summary


def print_summary(summary: Dict[str, object], top_k: int) -> None:
    """Pretty-print one dataset summary."""
    print("=" * 80)
    print(f"Dataset: {summary['file']}")
    print("-" * 80)
    print(f"Interactions            : {summary['interactions']:,}")
    print(f"Unique users            : {summary['users']:,}")
    print(f"Unique items            : {summary['items']:,}")
    print(f"Density                 : {summary['density']:.8f}")
    print(f"Duplicate user-item rows: {summary['duplicate_pairs']:,}")

    user_dist = summary["user_dist"]
    item_dist = summary["item_dist"]
    print(
        "User interactions dist  : "
        f"min={user_dist['min']}, p50={user_dist['p50']:.2f}, "
        f"p90={user_dist['p90']:.2f}, p99={user_dist['p99']:.2f}, max={user_dist['max']}"
    )
    print(
        "Item interactions dist  : "
        f"min={item_dist['min']}, p50={item_dist['p50']:.2f}, "
        f"p90={item_dist['p90']:.2f}, p99={item_dist['p99']:.2f}, max={item_dist['max']}"
    )

    ts_info = summary["timestamp"]
    if ts_info["has_timestamp"]:
        print(
            "Timestamp range         : "
            f"{format_ts(ts_info['min'])} -> {format_ts(ts_info['max'])}"
        )
    else:
        print("Timestamp range         : not available")

    print(f"Top {top_k} users by interactions:")
    for user, cnt in summary["top_users"]:
        print(f"  user={user:<12} count={cnt}")

    print(f"Top {top_k} items by interactions:")
    for item, cnt in summary["top_items"]:
        print(f"  item={item:<12} count={cnt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore RecBole .inter datasets")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Paths to .inter files (default: Beauty/Gowalla/Yelp)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top users/items to show",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = max(1, args.top_k)

    for file_name in args.files:
        path = Path(file_name)
        if not path.exists():
            print("=" * 80)
            print(f"Dataset: {path}")
            print("Error: file does not exist")
            continue

        try:
            summary = summarize_file(path, top_k=top_k)
            print_summary(summary, top_k=top_k)
        except Exception as exc:  # pylint: disable=broad-except
            print("=" * 80)
            print(f"Dataset: {path}")
            print(f"Error while parsing file: {exc}")


if __name__ == "__main__":
    main()
