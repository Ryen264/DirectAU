#!/usr/bin/env python3
"""Generic explorer for implicit-feedback NPZ datasets.

Supported schema (as used by Pinterest/CiteULike in this workspace):
- train_data: ndarray of shape (n_interactions, 2), columns are [user_id, item_id]
- test_data : object array containing dict {user_id: (positive_item, negative_items)}

Usage:
    python explore_npz_datasets.py
    python explore_npz_datasets.py --top-k 5
    python explore_npz_datasets.py --files dataset/pinterest.npz/pinterest.npz
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DEFAULT_FILES = [
    "dataset/pinterest.npz/pinterest.npz",
    "dataset/citeulike-a.npz/citeulike-a.npz",
]


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


def _to_python_int(value: Any) -> int:
    """Convert numpy scalar/int-like values to plain Python int."""
    return int(value)


def explore_npz_dataset(file_path: Path, top_k: int = 10) -> Dict[str, Any]:
    """General function to read and explore one implicit NPZ dataset.

    Args:
        file_path: Path to .npz dataset.
        top_k: Number of top users/items to report.

    Returns:
        A dictionary of summary statistics for train and test parts.
    """
    data = np.load(file_path, allow_pickle=True)

    if "train_data" not in data.files:
        raise ValueError("Missing key 'train_data' in NPZ file")

    train_data = data["train_data"]
    if not isinstance(train_data, np.ndarray) or train_data.ndim != 2 or train_data.shape[1] < 2:
        raise ValueError("'train_data' must be a 2D ndarray with at least 2 columns [user_id, item_id]")

    users = train_data[:, 0]
    items = train_data[:, 1]

    user_counts: Counter[int] = Counter(_to_python_int(u) for u in users)
    item_counts: Counter[int] = Counter(_to_python_int(i) for i in items)
    pair_counts: Counter[Tuple[int, int]] = Counter(
        (_to_python_int(u), _to_python_int(i)) for u, i in zip(users, items)
    )

    interaction_count = int(train_data.shape[0])
    n_users = len(user_counts)
    n_items = len(item_counts)
    density = interaction_count / float(n_users * n_items) if n_users and n_items else 0.0
    duplicate_pairs = sum(v - 1 for v in pair_counts.values() if v > 1)

    user_dist = sorted(user_counts.values())
    item_dist = sorted(item_counts.values())

    test_users = 0
    test_avg_negs = 0.0
    test_min_negs = 0
    test_max_negs = 0
    test_pos_overlap_train = 0

    if "test_data" in data.files:
        raw_test = data["test_data"]
        test_dict: Dict[Any, Any]
        if isinstance(raw_test, np.ndarray) and raw_test.shape == ():
            test_dict = raw_test.item()
        elif isinstance(raw_test, dict):
            test_dict = raw_test
        else:
            raise ValueError("Unsupported 'test_data' format, expected dict or 0-d object ndarray")

        test_users = len(test_dict)
        neg_counts: List[int] = []

        for user_id, payload in test_dict.items():
            _ = user_id
            if not isinstance(payload, (tuple, list)) or len(payload) < 2:
                continue
            pos_item = _to_python_int(payload[0])
            neg_items = payload[1]
            neg_count = len(neg_items) if hasattr(neg_items, "__len__") else 0
            neg_counts.append(neg_count)

            if (_to_python_int(user_id), pos_item) in pair_counts:
                test_pos_overlap_train += 1

        if neg_counts:
            test_avg_negs = float(sum(neg_counts)) / len(neg_counts)
            test_min_negs = min(neg_counts)
            test_max_negs = max(neg_counts)

    summary: Dict[str, Any] = {
        "file": str(file_path),
        "train_interactions": interaction_count,
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
        "test": {
            "users": test_users,
            "avg_neg_items": test_avg_negs,
            "min_neg_items": test_min_negs,
            "max_neg_items": test_max_negs,
            "pos_overlap_with_train": test_pos_overlap_train,
        },
    }
    return summary


def print_summary(summary: Dict[str, Any], top_k: int) -> None:
    print("=" * 80)
    print(f"Dataset: {summary['file']}")
    print("-" * 80)
    print(f"Train interactions       : {summary['train_interactions']:,}")
    print(f"Unique users             : {summary['users']:,}")
    print(f"Unique items             : {summary['items']:,}")
    print(f"Density                  : {summary['density']:.8f}")
    print(f"Duplicate user-item rows : {summary['duplicate_pairs']:,}")

    user_dist = summary["user_dist"]
    item_dist = summary["item_dist"]
    print(
        "User interactions dist   : "
        f"min={user_dist['min']}, p50={user_dist['p50']:.2f}, "
        f"p90={user_dist['p90']:.2f}, p99={user_dist['p99']:.2f}, max={user_dist['max']}"
    )
    print(
        "Item interactions dist   : "
        f"min={item_dist['min']}, p50={item_dist['p50']:.2f}, "
        f"p90={item_dist['p90']:.2f}, p99={item_dist['p99']:.2f}, max={item_dist['max']}"
    )

    test = summary["test"]
    print(f"Test users               : {test['users']:,}")
    print(
        "Test negatives per user  : "
        f"avg={test['avg_neg_items']:.2f}, min={test['min_neg_items']}, max={test['max_neg_items']}"
    )
    print(f"Test positives in train  : {test['pos_overlap_with_train']:,}")

    print(f"Top {top_k} users by interactions:")
    for user, cnt in summary["top_users"]:
        print(f"  user={user:<12} count={cnt}")

    print(f"Top {top_k} items by interactions:")
    for item, cnt in summary["top_items"]:
        print(f"  item={item:<12} count={cnt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore Pinterest/CiteULike NPZ datasets")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Paths to .npz files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top users/items to display",
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
            summary = explore_npz_dataset(path, top_k=top_k)
            print_summary(summary, top_k=top_k)
        except Exception as exc:  # pylint: disable=broad-except
            print("=" * 80)
            print(f"Dataset: {path}")
            print(f"Error while parsing file: {exc}")


if __name__ == "__main__":
    main()
