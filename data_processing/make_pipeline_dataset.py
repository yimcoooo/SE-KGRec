#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_pipeline_dataset.py

Convert the aligned build under `aligned_build/<x>_full/` into a directory that the training pipeline can read directly:
`data/<dataset>/` (same format as `data/yelp_20-core/`).

File mapping (same format as `build_yelp_subset_dataset.py::write_mapping_files`):

    data/<dataset>/
        entity_list.txt                 <- mappings/entity_map.tsv   org_id = {entity_type}::{raw_value}
        item_list.txt                   <- mappings/item_map.tsv     org_id = raw_business_id
        user_list.txt                   <- mappings/user_map.tsv     org_id = raw_user_id
        relation_list.txt               <- mappings/relation_map.tsv org_id = relation_name
        train.txt                       <- interactions/train.txt
        test.txt                        <- interactions/test.txt
        all_positive_interactions.txt   <- interactions/all_positive.txt
        kg_final.txt                    <- kg/kg_final.txt
        summary.json                    <- combined statistics + provenance

Design principles (important):
  * Formatting and copying only: never re-encode any ID. The build script already emits contiguous integer IDs (0..n-1),
    `item_map.item_idx`, the `item_idx` emitted by the LLM semantic extraction, and
    the vocabulary alignment used by `load_max_item_id(item_list.txt)` in
    `preprocess_yelp_relation_embeddings.py` must remain identical.
  * If an output file already exists, exit with an error unless `--force` is given (to avoid accidental overwrites).

Usage:
    python data_processing/make_pipeline_dataset.py \
        --build-dir aligned_build/yelp_20core_full \
        --name yelp_20-core

Afterwards (llm_embedding must exist first):
    python data_processing/make_split.py --datasets yelp_20-core      # generate the 80/10/10 split -> data/
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def log(msg):
    import time
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_list(path: Path, rows):
    """rows: iterable of (org_id, remap_id)

    Note: existing list files under `data/*/` use a UTF-8 BOM (utf-8-sig); we keep the same encoding.
    Consumers (`load_max_item_id` / `load_relation_order` / `_load_relation_name_to_actual_id`)
    skip the header line, so the BOM does not affect parsing; byte-identical output is nevertheless preferable for provenance.
    """
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["org_id", "remap_id"])
        for org_id, remap_id in rows:
            w.writerow([org_id, remap_id])


def copy_or_link(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.relpath(str(src), str(dst.parent)), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, type=Path,
                    help="Aligned build directory, e.g. aligned_build/yelp_20core_full")
    ap.add_argument("--name", required=True, type=str,
                    help="Target dataset name (determines data/<name>/), e.g. yelp_20-core")
    ap.add_argument("--data-root", type=Path, default=REPO / "data",
                    help="Output root (default: <repo>/data)")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy",
                    help="How to move large files (train/test/kg/all_positive); default: copy")
    ap.add_argument("--force", action="store_true", help="Allow overwriting existing output files")
    args = ap.parse_args()

    build_dir = args.build_dir.resolve()
    out_dir = (args.data_root / args.name).resolve()
    if not build_dir.is_dir():
        raise SystemExit("build-dir not found: %s" % build_dir)
    if out_dir.exists() and not args.force:
        # the directory may already exist (e.g. with llm_embedding); file conflicts are checked individually below
        pass
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = build_dir / "mappings"
    inter = build_dir / "interactions"
    kd = build_dir / "kg"

    for p in [maps / "entity_map.tsv", maps / "item_map.tsv",
              maps / "user_map.tsv", maps / "relation_map.tsv",
              inter / "train.txt", inter / "test.txt",
              inter / "all_positive.txt", kd / "kg_final.txt"]:
        if not p.exists():
            raise SystemExit("missing required input: %s" % p)

    log("loading mappings ...")
    ent_rows = read_tsv(maps / "entity_map.tsv")
    item_rows = read_tsv(maps / "item_map.tsv")
    user_rows = read_tsv(maps / "user_map.tsv")
    rel_rows = read_tsv(maps / "relation_map.tsv")
    log("entities=%d items=%d users=%d relations=%d"
        % (len(ent_rows), len(item_rows), len(user_rows), len(rel_rows)))

    # ---- check ID contiguity (guards against silent re-encoding by the build script) ----
    def check_contiguous(rows, key, label):
        ids = sorted(int(r[key]) for r in rows)
        expect = list(range(len(rows)))
        if ids != expect:
            raise SystemExit("[FATAL] %s ids not 0..n-1 contiguous (got min=%d max=%d n=%d)"
                             % (label, ids[0], ids[-1], len(ids)))
        log("  %s id range OK: 0..%d" % (label, len(rows) - 1))

    check_contiguous(ent_rows, "entity_id", "entity")
    check_contiguous(item_rows, "item_idx", "item")
    check_contiguous(user_rows, "user_idx", "user")
    check_contiguous(rel_rows, "relation_id", "relation")

    # ---- write list files ----
    log("writing entity_list.txt ...")
    write_list(out_dir / "entity_list.txt",
               (("%s::%s" % (r["entity_type"], r["raw_value"]), int(r["entity_id"]))
                for r in ent_rows))
    log("writing item_list.txt ...")
    write_list(out_dir / "item_list.txt",
               ((r["raw_business_id"], int(r["item_idx"])) for r in item_rows))
    log("writing user_list.txt ...")
    write_list(out_dir / "user_list.txt",
               ((r["raw_user_id"], int(r["user_idx"])) for r in user_rows))
    log("writing relation_list.txt ...")
    write_list(out_dir / "relation_list.txt",
               ((r["relation_name"], int(r["relation_id"])) for r in rel_rows))

    # ---- move large files ----
    pairs = [
        (inter / "train.txt", out_dir / "train.txt"),
        (inter / "test.txt", out_dir / "test.txt"),
        (inter / "all_positive.txt", out_dir / "all_positive_interactions.txt"),
        (kd / "kg_final.txt", out_dir / "kg_final.txt"),
    ]
    for src, dst in pairs:
        log("place %s <- %s (%s)" % (dst.name, src.name, args.mode))
        copy_or_link(src, dst, args.mode)

    # ---- statistics ----
    n_train = sum(1 for _ in open(out_dir / "train.txt", encoding="utf-8"))
    n_test = sum(1 for _ in open(out_dir / "test.txt", encoding="utf-8"))
    n_kg = sum(1 for _ in open(out_dir / "kg_final.txt", encoding="utf-8"))
    train_int = sum(len(l.split()) - 1 for l in open(out_dir / "train.txt", encoding="utf-8") if l.strip())
    test_int = sum(len(l.split()) - 1 for l in open(out_dir / "test.txt", encoding="utf-8") if l.strip())

    summary = {
        "dataset": args.name,
        "provenance": {
            "generator": "data_processing/make_pipeline_dataset.py",
            "source_build_dir": str(build_dir),
            "note": "format conversion only; no ID remapping",
        },
        "counts": {
            "users": len(user_rows),
            "items": len(item_rows),
            "entities": len(ent_rows),
            "relations": len(rel_rows),
            "train_lines": n_train,
            "test_lines": n_test,
            "train_interactions": train_int,
            "test_interactions": test_int,
            "kg_triplets": n_kg,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("done -> %s" % out_dir)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
