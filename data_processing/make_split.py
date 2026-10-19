#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_split.py — build a reproducible user-wise 80/10/10 train/validation/test split

Background:
    Some datasets provide only train.txt + test.txt (summary.json may report test_ratio=0.2, i.e. 80/20),
    whereas the paper uses a random 80/10/10 split and validation-based early stopping.
    This script rebuilds a strict 80/10/10 split from all interactions and writes it to data/<dataset>/;
    all other files (kg_final, relation_list, llm_embedding, ...) are reused from the original dataset.

Usage:
    python data_processing/make_split.py
    python data_processing/make_split.py --datasets amazonbook_200k yelp_20-core goodreads

Output:
    data/<dataset>/{train.txt,val.txt,test.txt}   (new split)
    data/<dataset>/<other files>                 (symlink to data/<dataset>/<same file>)
"""
import os
import sys
import json
import argparse
import numpy as np
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DATA = os.path.join(REPO, "data")
DST_DATA = os.path.join(REPO, "data")  # overridden by --out-root
SPLIT_SEED = 2026  # fixed for reproducibility

# files/directories linked from the original dataset (everything except train/val/test)
LINK_NAMES = [
    "kg_final.txt", "relation_list.txt", "item_list.txt", "entity_list.txt",
    "user_list.txt", "all_positive_interactions.txt", "summary.json", "llm_embedding",
]


def _read_full_interactions(ds_dir):
    """Return uid -> [items] (deduplicated and sorted for determinism). Prefer all_positive, otherwise merge train+test."""
    def parse(path):
        d = OrderedDict()
        if not os.path.exists(path):
            return d
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                items = [int(x) for x in parts[1:]]
                d.setdefault(uid, set()).update(items)
        return d

    allpos = parse(os.path.join(ds_dir, "all_positive_interactions.txt"))
    train = parse(os.path.join(ds_dir, "train.txt"))
    test = parse(os.path.join(ds_dir, "test.txt"))

    full = OrderedDict()
    if allpos:
        for u, s in allpos.items():
            full[u] = set(s)
        # fallback: add interactions present in train/test but missing from all_positive
        for src in (train, test):
            for u, s in src.items():
                full.setdefault(u, set()).update(s)
    else:
        for src in (train, test):
            for u, s in src.items():
                full.setdefault(u, set()).update(s)
    return full


def _split_user(items, rng):
    """Split a user's items into train/val/test (80/10/10; small users keep train>=1 and test>=1)."""
    arr = np.array(sorted(items), dtype=np.int64)
    n = int(arr.shape[0])
    rng.shuffle(arr)
    if n >= 10:
        n_test = max(1, int(round(0.10 * n)))
        n_val = max(1, int(round(0.10 * n)))
    elif n >= 5:
        # small users keep one test and one validation item (actual ratios are reported in split_info.json)
        n_test = 1
        n_val = 1
    else:  # n in {1,2,3,4}
        n_test = 1
        n_val = 0
    # keep at least one training item
    if n - n_val - n_test < 1:
        n_val = max(0, n - n_test - 1)
    if n - n_val - n_test < 1:
        n_test = max(1, n - 1)
        n_val = 0
    n_train = n - n_val - n_test
    train = arr[:n_train]
    val = arr[n_train:n_train + n_val]
    test = arr[n_train + n_val:]
    return train, val, test


def _ensure_links(ds, dst_dir):
    src_dir = os.path.join(SRC_DATA, ds)
    for name in LINK_NAMES:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src):
            continue
        if os.path.islink(dst) or os.path.exists(dst):
            if os.path.islink(dst):
                os.remove(dst)
            else:
                continue
        os.symlink(os.path.relpath(src, dst_dir), dst)


def _write_lines(path, uid_to_items):
    with open(path, "w", encoding="utf-8") as f:
        for uid, items in uid_to_items.items():
            if len(items) == 0:
                continue
            f.write(" ".join([str(uid)] + [str(int(i)) for i in items]) + "\n")


def make_split(ds):
    src_dir = os.path.join(SRC_DATA, ds)
    dst_dir = os.path.join(DST_DATA, ds)
    if not os.path.isdir(src_dir):
        print("[skip] dataset dir not found:", src_dir)
        return None
    os.makedirs(dst_dir, exist_ok=True)
    _ensure_links(ds, dst_dir)

    full = _read_full_interactions(src_dir)
    rng = np.random.default_rng(SPLIT_SEED)

    train_d, val_d, test_d = OrderedDict(), OrderedDict(), OrderedDict()
    n_total = n_tr = n_va = n_te = 0
    for uid in sorted(full.keys()):
        items = full[uid]
        n_total += len(items)
        tr, va, te = _split_user(items, rng)
        train_d[uid] = tr
        val_d[uid] = va
        test_d[uid] = te
        n_tr += len(tr)
        n_va += len(va)
        n_te += len(te)

    # remove any existing train/val/test first (they may be symlinks)
    for name in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(dst_dir, name)
        if os.path.islink(p) or os.path.exists(p):
            os.remove(p)

    _write_lines(os.path.join(dst_dir, "train.txt"), train_d)
    _write_lines(os.path.join(dst_dir, "val.txt"), val_d)
    _write_lines(os.path.join(dst_dir, "test.txt"), test_d)

    info = {
        "dataset": ds, "split_seed": SPLIT_SEED,
        "users": len(full),
        "interactions_total": n_total,
        "train_interactions": n_tr, "val_interactions": n_va, "test_interactions": n_te,
        "train_ratio": round(n_tr / n_total, 4),
        "val_ratio": round(n_va / n_total, 4),
        "test_ratio": round(n_te / n_total, 4),
        "users_with_val": sum(1 for u in val_d if len(val_d[u]) > 0),
        "users_with_test": sum(1 for u in test_d if len(test_d[u]) > 0),
    }
    with open(os.path.join(dst_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[split] {}: total={} train={}({:.3f}) val={}({:.3f}) test={}({:.3f}) users={}".format(
        ds, n_total, n_tr, info["train_ratio"], n_va, info["val_ratio"], n_te, info["test_ratio"], len(full)))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data", help="Dataset root (relative to the repository root)")
    ap.add_argument("--datasets", nargs="+",
                    default=["amazonbook_200k", "yelp_20-core", "goodreads"])
    args = ap.parse_args()
    global DST_DATA
    DST_DATA = os.path.join(REPO, args.out_root)
    os.makedirs(DST_DATA, exist_ok=True)
    for ds in args.datasets:
        make_split(ds)


if __name__ == "__main__":
    main()
