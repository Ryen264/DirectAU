#!/usr/bin/env python3
"""Convert Pinterest/CiteULike NPZ files into RecBole benchmark .inter splits.

Output format per dataset folder:
- <DatasetName>.train.inter
- <DatasetName>.valid.inter
- <DatasetName>.test.inter

All files contain two columns:
user_id:token\titem_id:token
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Pair = Tuple[int, int]


DEFAULT_SOURCES = {
    "Pinterest": "dataset/pinterest.npz/pinterest.npz",
    "CiteULikeA": "dataset/citeulike-a.npz/citeulike-a.npz",
}


def load_npz_pairs(npz_path: Path) -> Tuple[List[Pair], Dict[int, int]]:
    data = np.load(npz_path, allow_pickle=True)
    train = data["train_data"]
    train_pairs: List[Pair] = [(int(u), int(i)) for u, i in train[:, :2]]

    test_pos: Dict[int, int] = {}
    if "test_data" in data.files:
        raw_test = data["test_data"]
        test_dict = raw_test.item() if isinstance(raw_test, np.ndarray) and raw_test.shape == () else raw_test
        for u, payload in test_dict.items():
            if isinstance(payload, (tuple, list)) and len(payload) >= 1:
                test_pos[int(u)] = int(payload[0])
    return train_pairs, test_pos


def split_train_valid(train_pairs: Sequence[Pair], test_pos: Dict[int, int], seed: int) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Create train/valid/test positives for RecBole benchmark split.

    - test positives are taken from NPZ test_data[user][0]
    - valid positives are sampled one per user from remaining train pairs (if possible)
    - overlapping (user, test_pos) pairs are removed from training pool
    """
    rng = np.random.default_rng(seed)

    dedup_pairs = list(dict.fromkeys(train_pairs))
    train_pool: List[Pair] = [p for p in dedup_pairs if test_pos.get(p[0]) != p[1]]

    by_user: Dict[int, List[int]] = defaultdict(list)
    for u, i in train_pool:
        by_user[u].append(i)

    valid_pairs: List[Pair] = []
    train_final: List[Pair] = []
    for u, items in by_user.items():
        if len(items) >= 2:
            pick_idx = int(rng.integers(0, len(items)))
            valid_item = items[pick_idx]
            valid_pairs.append((u, valid_item))
            for idx, item in enumerate(items):
                if idx != pick_idx:
                    train_final.append((u, item))
        else:
            train_final.append((u, items[0]))

    test_pairs: List[Pair] = sorted((u, i) for u, i in test_pos.items())
    return train_final, valid_pairs, test_pairs


def write_inter(file_path: Path, pairs: Iterable[Pair]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as f:
        f.write("user_id:token\titem_id:token\n")
        for u, i in pairs:
            f.write(f"{u}\t{i}\n")


def convert_one(name: str, source_npz: Path, output_root: Path, seed: int) -> None:
    if not source_npz.exists():
        raise FileNotFoundError(f"NPZ file not found: {source_npz}")

    train_pairs, test_pos = load_npz_pairs(source_npz)
    train_split, valid_split, test_split = split_train_valid(train_pairs, test_pos, seed=seed)

    dataset_dir = output_root / name
    write_inter(dataset_dir / f"{name}.train.inter", train_split)
    write_inter(dataset_dir / f"{name}.valid.inter", valid_split)
    write_inter(dataset_dir / f"{name}.test.inter", test_split)

    print(f"[{name}] source={source_npz}")
    print(f"[{name}] train={len(train_split):,} valid={len(valid_split):,} test={len(test_split):,}")
    if not valid_split:
        print(f"[{name}] WARNING: valid split is empty; metrics may be unstable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NPZ datasets for DirectAU in RecBole")
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=["Pinterest", "CiteULikeA"],
        help="Dataset names to convert (Pinterest, CiteULikeA)",
    )
    parser.add_argument("--output-root", default="dataset", help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for valid split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    for name in args.dataset:
        if name not in DEFAULT_SOURCES:
            raise ValueError(f"Unsupported dataset [{name}]. Use one of: {list(DEFAULT_SOURCES)}")
        source = Path(DEFAULT_SOURCES[name])
        convert_one(name=name, source_npz=source, output_root=output_root, seed=args.seed)


if __name__ == "__main__":
    main()
