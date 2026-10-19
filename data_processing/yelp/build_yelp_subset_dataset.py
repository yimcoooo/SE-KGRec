from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class UserInteractions:
    old_user_idx: int
    items: tuple[int, ...]


@dataclass
class CandidateSummary:
    prefix_users: int
    final_users: int
    final_items: int
    final_interactions: int
    kcore_iterations: int
    error: int


class JsonlShardWriter:
    def __init__(self, output_dir: Path, prefix: str, shard_size: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.part_index = -1
        self.part_record_count = 0
        self.total_records = 0
        self.total_parts = 0
        self.current_handle = None
        self._open_next_part()

    def _open_next_part(self) -> None:
        if self.current_handle is not None:
            self.current_handle.close()
        self.part_index += 1
        self.total_parts += 1
        self.part_record_count = 0
        part_path = self.output_dir / f"{self.prefix}_part_{self.part_index:05d}.jsonl"
        self.current_handle = part_path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, object]) -> None:
        if self.shard_size > 0 and self.part_record_count >= self.shard_size:
            self._open_next_part()
        assert self.current_handle is not None
        self.current_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.part_record_count += 1
        self.total_records += 1

    def close(self) -> None:
        if self.current_handle is not None:
            self.current_handle.close()
            self.current_handle = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smaller, internally consistent KGAT-ready Yelp subset from an already processed dataset directory."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "02_processed_data" / "full_datasets" / "aligned_full_pos_5core_ratio",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "02_processed_data" / "subset_datasets" / "aligned_500k_pos_5core_ratio",
    )
    parser.add_argument("--target-interactions", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--k-core", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--export-shard-size", type=int, default=250_000)
    parser.add_argument(
        "--tip-filter-mode",
        choices=("strict_user_item", "item_only"),
        default="strict_user_item",
    )
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}")
    sys.stdout.flush()


def ensure_clean_output_dir(path: Path, keep_output: bool) -> None:
    if path.exists() and not keep_output:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def write_tsv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def load_all_positive(path: Path) -> list[UserInteractions]:
    result: list[UserInteractions] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_user_idx = int(parts[0])
                items = tuple(int(token) for token in parts[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid all_positive line at {path}:{line_no}") from exc
            if items:
                result.append(UserInteractions(old_user_idx=old_user_idx, items=items))
    return result


def run_bipartite_k_core(
    user_records: Iterable[UserInteractions],
    *,
    k: int,
) -> tuple[dict[int, list[int]], int]:
    user_items: dict[int, set[int]] = {record.old_user_idx: set(record.items) for record in user_records}
    iterations = 0

    while True:
        iterations += 1
        item_degree: Counter[int] = Counter()
        for items in user_items.values():
            item_degree.update(items)

        removed_any = False
        bad_items = {item for item, degree in item_degree.items() if degree < k}
        if bad_items:
            removed_any = True
            for user_id in list(user_items):
                kept_items = user_items[user_id] - bad_items
                if kept_items:
                    user_items[user_id] = kept_items
                else:
                    del user_items[user_id]

        bad_users = {user_id for user_id, items in user_items.items() if len(items) < k}
        if bad_users:
            removed_any = True
            for user_id in bad_users:
                user_items.pop(user_id, None)

        if not removed_any:
            break

    final_user_items = {user_id: sorted(items) for user_id, items in user_items.items() if items}
    return final_user_items, iterations


def summarize_user_items(
    prefix_users: int,
    user_items: dict[int, list[int]],
    *,
    target_interactions: int,
    iterations: int,
) -> CandidateSummary:
    item_ids = {item_id for items in user_items.values() for item_id in items}
    interaction_count = sum(len(items) for items in user_items.values())
    return CandidateSummary(
        prefix_users=prefix_users,
        final_users=len(user_items),
        final_items=len(item_ids),
        final_interactions=interaction_count,
        kcore_iterations=iterations,
        error=interaction_count - target_interactions,
    )


def evaluate_prefix(
    shuffled_users: list[UserInteractions],
    prefix_users: int,
    *,
    k_core: int,
    target_interactions: int,
    with_graph: bool = False,
) -> tuple[CandidateSummary, dict[int, list[int]] | None]:
    candidate_graph, iterations = run_bipartite_k_core(shuffled_users[:prefix_users], k=k_core)
    summary = summarize_user_items(
        prefix_users,
        candidate_graph,
        target_interactions=target_interactions,
        iterations=iterations,
    )
    return summary, candidate_graph if with_graph else None


def search_best_candidate(
    shuffled_users: list[UserInteractions],
    *,
    k_core: int,
    target_interactions: int,
) -> tuple[CandidateSummary, dict[int, list[int]], list[CandidateSummary]]:
    history: dict[int, CandidateSummary] = {}
    left = 1
    right = len(shuffled_users)

    while left <= right:
        mid = (left + right) // 2
        summary, _ = evaluate_prefix(
            shuffled_users,
            mid,
            k_core=k_core,
            target_interactions=target_interactions,
        )
        history[mid] = summary
        log(
            f"search prefix={mid:,} -> users={summary.final_users:,} items={summary.final_items:,} "
            f"interactions={summary.final_interactions:,} error={summary.error:+,}"
        )
        if summary.final_interactions < target_interactions:
            left = mid + 1
        elif summary.final_interactions > target_interactions:
            right = mid - 1
        else:
            break

    extra_points = {left, right, max(1, left - 1), min(len(shuffled_users), right + 1)}
    for prefix_users in sorted(point for point in extra_points if 1 <= point <= len(shuffled_users)):
        if prefix_users in history:
            continue
        summary, _ = evaluate_prefix(
            shuffled_users,
            prefix_users,
            k_core=k_core,
            target_interactions=target_interactions,
        )
        history[prefix_users] = summary
        log(
            f"refine prefix={prefix_users:,} -> users={summary.final_users:,} items={summary.final_items:,} "
            f"interactions={summary.final_interactions:,} error={summary.error:+,}"
        )

    best_summary = min(history.values(), key=lambda row: (abs(row.error), row.prefix_users))
    best_summary, best_graph = evaluate_prefix(
        shuffled_users,
        best_summary.prefix_users,
        k_core=k_core,
        target_interactions=target_interactions,
        with_graph=True,
    )
    assert best_graph is not None
    return best_summary, best_graph, sorted(history.values(), key=lambda row: row.prefix_users)


def build_split(
    user_items_old: dict[int, list[int]],
    *,
    seed: int,
    test_ratio: float,
) -> tuple[
    dict[int, list[int]],
    dict[int, list[int]],
    dict[tuple[int, int], str],
]:
    train_items_by_user: dict[int, list[int]] = {}
    test_items_by_user: dict[int, list[int]] = {}
    split_by_pair: dict[tuple[int, int], str] = {}

    for old_user_idx in sorted(user_items_old):
        items = list(user_items_old[old_user_idx])
        if len(items) <= 1:
            train_items = []
            test_items = sorted(items)
            train_items_by_user[old_user_idx] = train_items
            test_items_by_user[old_user_idx] = test_items
            for item_idx in test_items:
                split_by_pair[(old_user_idx, item_idx)] = "test"
            continue
        rng = random.Random(seed ^ (old_user_idx * 1_315_423_911))
        rng.shuffle(items)
        raw_test_count = max(1, int(round(len(items) * test_ratio)))
        test_count = min(raw_test_count, len(items) - 1)
        train_items = sorted(items[:-test_count])
        test_items = sorted(items[-test_count:])
        train_items_by_user[old_user_idx] = train_items
        test_items_by_user[old_user_idx] = test_items
        for item_idx in train_items:
            split_by_pair[(old_user_idx, item_idx)] = "train"
        for item_idx in test_items:
            split_by_pair[(old_user_idx, item_idx)] = "test"

    return train_items_by_user, test_items_by_user, split_by_pair


def load_user_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        return {int(row["user_idx"]): row for row in reader}


def load_item_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        return {int(row["item_idx"]): row for row in reader}


def load_relation_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        return {int(row["relation_id"]): row for row in reader}


def compute_entity_closure(
    kg_path: Path,
    kept_item_entity_ids: set[int],
) -> tuple[set[int], int]:
    reachable = set(kept_item_entity_ids)
    iterations = 0
    while True:
        iterations += 1
        new_entities: set[int] = set()
        with kg_path.open("r", encoding="utf-8") as file_obj:
            for line_no, line in enumerate(file_obj, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(f"Invalid KG line at {kg_path}:{line_no}")
                head_id, _, tail_id = (int(token) for token in parts)
                if head_id in reachable and tail_id not in reachable:
                    new_entities.add(tail_id)
        if not new_entities:
            break
        reachable.update(new_entities)
    return reachable, iterations


def load_entity_rows(
    path: Path,
    kept_entity_ids: set[int],
) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        for row in reader:
            entity_id = int(row["entity_id"])
            if entity_id in kept_entity_ids:
                rows[entity_id] = row
    return rows


def build_entity_and_relation_maps(
    kept_user_items: dict[int, list[int]],
    source_dir: Path,
) -> tuple[
    dict[int, dict[str, str]],
    dict[int, dict[str, str]],
    dict[int, int],
    dict[int, int],
    set[int],
    dict[int, int],
    int,
]:
    kept_old_item_ids = {item_id for items in kept_user_items.values() for item_id in items}
    kg_path = source_dir / "kg" / "kg_final.txt"
    reachable_old_entity_ids, closure_iterations = compute_entity_closure(kg_path, kept_old_item_ids)

    source_item_rows = load_item_rows(source_dir / "mappings" / "item_map.tsv")
    source_entity_rows = load_entity_rows(source_dir / "mappings" / "entity_map.tsv", reachable_old_entity_ids)
    source_relation_rows = load_relation_rows(source_dir / "mappings" / "relation_map.tsv")

    old_item_to_new_item = {old_item_id: new_item_id for new_item_id, old_item_id in enumerate(sorted(kept_old_item_ids))}

    old_entity_to_new_entity: dict[int, int] = {}
    for old_item_id, new_item_id in old_item_to_new_item.items():
        old_entity_to_new_entity[old_item_id] = new_item_id
    next_entity_id = len(old_item_to_new_item)
    for old_entity_id in sorted(reachable_old_entity_ids):
        if old_entity_id in old_entity_to_new_entity:
            continue
        old_entity_to_new_entity[old_entity_id] = next_entity_id
        next_entity_id += 1

    used_old_relations: set[int] = set()
    with kg_path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Invalid KG line at {kg_path}:{line_no}")
            head_id, relation_id, tail_id = (int(token) for token in parts)
            if head_id in reachable_old_entity_ids and tail_id in reachable_old_entity_ids:
                used_old_relations.add(relation_id)

    old_relation_to_new_relation = {
        old_relation_id: new_relation_id for new_relation_id, old_relation_id in enumerate(sorted(used_old_relations))
    }

    kept_relation_rows = {relation_id: source_relation_rows[relation_id] for relation_id in sorted(used_old_relations)}
    kept_entity_rows = {entity_id: source_entity_rows[entity_id] for entity_id in sorted(reachable_old_entity_ids)}

    missing_items = sorted(item for item in kept_old_item_ids if item not in source_item_rows)
    if missing_items:
        raise ValueError(f"Missing item rows for {len(missing_items)} kept items. Example: {missing_items[:5]}")

    return (
        kept_entity_rows,
        kept_relation_rows,
        old_item_to_new_item,
        old_entity_to_new_entity,
        reachable_old_entity_ids,
        old_relation_to_new_relation,
        closure_iterations,
    )


def write_interaction_files(
    output_dir: Path,
    kept_user_items: dict[int, list[int]],
    train_items_by_user: dict[int, list[int]],
    test_items_by_user: dict[int, list[int]],
    old_user_to_new_user: dict[int, int],
    old_item_to_new_item: dict[int, int],
) -> dict[str, int]:
    interaction_dir = output_dir / "interactions"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    all_positive_path = interaction_dir / "all_positive.txt"
    all_positive_alias_path = interaction_dir / "all_positive_interactions.txt"
    train_path = interaction_dir / "train.txt"
    test_path = interaction_dir / "test.txt"

    train_count = 0
    test_count = 0
    total_positive = 0

    with all_positive_path.open("w", encoding="utf-8") as all_positive_file, \
        all_positive_alias_path.open("w", encoding="utf-8") as all_positive_alias_file, \
        train_path.open("w", encoding="utf-8") as train_file, \
        test_path.open("w", encoding="utf-8") as test_file:
        for old_user_idx in sorted(kept_user_items):
            new_user_idx = old_user_to_new_user[old_user_idx]
            new_all_items = [old_item_to_new_item[item] for item in kept_user_items[old_user_idx]]
            new_train_items = [old_item_to_new_item[item] for item in train_items_by_user[old_user_idx]]
            new_test_items = [old_item_to_new_item[item] for item in test_items_by_user[old_user_idx]]

            total_positive += len(new_all_items)
            train_count += len(new_train_items)
            test_count += len(new_test_items)

            all_line = f"{new_user_idx} {' '.join(map(str, new_all_items))}\n"
            all_positive_file.write(all_line)
            all_positive_alias_file.write(all_line)
            if new_train_items:
                train_file.write(f"{new_user_idx} {' '.join(map(str, new_train_items))}\n")
            if new_test_items:
                test_file.write(f"{new_user_idx} {' '.join(map(str, new_test_items))}\n")

    return {
        "users": len(kept_user_items),
        "positive_interactions": total_positive,
        "train_interactions": train_count,
        "test_interactions": test_count,
    }


def write_mapping_files(
    output_dir: Path,
    source_dir: Path,
    kept_user_items: dict[int, list[int]],
    old_user_to_new_user: dict[int, int],
    old_item_to_new_item: dict[int, int],
    kept_entity_rows: dict[int, dict[str, str]],
    kept_relation_rows: dict[int, dict[str, str]],
    old_entity_to_new_entity: dict[int, int],
    old_relation_to_new_relation: dict[int, int],
) -> dict[str, int]:
    mapping_dir = output_dir / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    source_user_rows = load_user_rows(source_dir / "mappings" / "user_map.tsv")
    source_item_rows = load_item_rows(source_dir / "mappings" / "item_map.tsv")

    user_positive_count = {old_user_idx: len(items) for old_user_idx, items in kept_user_items.items()}
    item_positive_count: Counter[int] = Counter()
    for items in kept_user_items.values():
        item_positive_count.update(items)

    write_tsv(
        mapping_dir / "user_map.tsv",
        ["user_idx", "raw_user_id", "source_user_idx", "positive_count"],
        (
            [
                old_user_to_new_user[old_user_idx],
                source_user_rows[old_user_idx]["raw_user_id"],
                old_user_idx,
                user_positive_count[old_user_idx],
            ]
            for old_user_idx in sorted(old_user_to_new_user)
        ),
    )
    write_tsv(
        mapping_dir / "item_map.tsv",
        ["item_idx", "entity_id", "raw_business_id", "source_item_idx", "positive_count"],
        (
            [
                old_item_to_new_item[old_item_idx],
                old_item_to_new_item[old_item_idx],
                source_item_rows[old_item_idx]["raw_business_id"],
                old_item_idx,
                item_positive_count[old_item_idx],
            ]
            for old_item_idx in sorted(old_item_to_new_item)
        ),
    )
    write_tsv(
        mapping_dir / "entity_map.tsv",
        ["entity_id", "entity_type", "source_field", "raw_value", "label", "metadata_json", "source_entity_id"],
        (
            [
                old_entity_to_new_entity[old_entity_id],
                row["entity_type"],
                row["source_field"],
                row["raw_value"],
                row["label"],
                row["metadata_json"],
                old_entity_id,
            ]
            for old_entity_id, row in sorted(kept_entity_rows.items())
        ),
    )
    write_tsv(
        mapping_dir / "relation_map.tsv",
        ["relation_id", "relation_name", "head_type", "tail_type", "description", "source_relation_id"],
        (
            [
                old_relation_to_new_relation[old_relation_id],
                row["relation_name"],
                row["head_type"],
                row["tail_type"],
                row["description"],
                old_relation_id,
            ]
            for old_relation_id, row in sorted(kept_relation_rows.items())
        ),
    )

    with (output_dir / "user_list.txt").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["org_id", "remap_id"])
        for old_user_idx in sorted(old_user_to_new_user):
            writer.writerow([source_user_rows[old_user_idx]["raw_user_id"], old_user_to_new_user[old_user_idx]])

    with (output_dir / "item_list.txt").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["org_id", "remap_id"])
        for old_item_idx in sorted(old_item_to_new_item):
            writer.writerow([source_item_rows[old_item_idx]["raw_business_id"], old_item_to_new_item[old_item_idx]])

    with (output_dir / "entity_list.txt").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["org_id", "remap_id"])
        for old_entity_id, row in sorted(kept_entity_rows.items()):
            writer.writerow([f"{row['entity_type']}::{row['raw_value']}", old_entity_to_new_entity[old_entity_id]])

    with (output_dir / "relation_list.txt").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(["org_id", "remap_id"])
        for old_relation_id, row in sorted(kept_relation_rows.items()):
            writer.writerow([row["relation_name"], old_relation_to_new_relation[old_relation_id]])

    return {
        "users": len(old_user_to_new_user),
        "items": len(old_item_to_new_item),
        "entities": len(kept_entity_rows),
        "relations": len(kept_relation_rows),
    }


def write_kg_files(
    output_dir: Path,
    source_dir: Path,
    kept_entity_rows: dict[int, dict[str, str]],
    kept_relation_rows: dict[int, dict[str, str]],
    old_entity_to_new_entity: dict[int, int],
    old_relation_to_new_relation: dict[int, int],
    reachable_old_entity_ids: set[int],
) -> int:
    kg_dir = output_dir / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    kg_path = source_dir / "kg" / "kg_final.txt"
    out_kg_path = kg_dir / "kg_final.txt"
    out_edges_path = kg_dir / "kg_edges.tsv"

    triple_count = 0
    with out_kg_path.open("w", encoding="utf-8") as kg_file, out_edges_path.open(
        "w", encoding="utf-8", newline=""
    ) as edges_file:
        writer = csv.writer(edges_file, delimiter="\t")
        writer.writerow(
            [
                "head_entity_id",
                "head_type",
                "head_raw_value",
                "relation_id",
                "relation_name",
                "tail_entity_id",
                "tail_type",
                "tail_raw_value",
                "source",
            ]
        )
        with kg_path.open("r", encoding="utf-8") as file_obj:
            for line_no, line in enumerate(file_obj, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 3:
                    raise ValueError(f"Invalid KG line at {kg_path}:{line_no}")
                old_head, old_relation, old_tail = (int(token) for token in parts)
                if old_head not in reachable_old_entity_ids or old_tail not in reachable_old_entity_ids:
                    continue
                if old_relation not in old_relation_to_new_relation:
                    continue
                new_head = old_entity_to_new_entity[old_head]
                new_relation = old_relation_to_new_relation[old_relation]
                new_tail = old_entity_to_new_entity[old_tail]
                kg_file.write(f"{new_head} {new_relation} {new_tail}\n")
                head_row = kept_entity_rows[old_head]
                tail_row = kept_entity_rows[old_tail]
                relation_row = kept_relation_rows[old_relation]
                writer.writerow(
                    [
                        new_head,
                        head_row["entity_type"],
                        head_row["raw_value"],
                        new_relation,
                        relation_row["relation_name"],
                        new_tail,
                        tail_row["entity_type"],
                        tail_row["raw_value"],
                        f"source_entity_ids={old_head}->{old_tail};source_relation_id={old_relation}",
                    ]
                )
                triple_count += 1

    return triple_count


def stream_filter_positive_details(
    output_dir: Path,
    source_dir: Path,
    kept_pairs_old: set[tuple[int, int]],
    split_by_pair: dict[tuple[int, int], str],
    old_user_to_new_user: dict[int, int],
    old_item_to_new_item: dict[int, int],
) -> tuple[int, int]:
    interaction_dir = output_dir / "interactions"
    positive_detail_path = interaction_dir / "positive_interactions.tsv"
    split_path = interaction_dir / "review_split.tsv"
    source_path = source_dir / "interactions" / "positive_interactions.tsv"

    kept_rows = 0
    split_rows = 0
    with source_path.open("r", encoding="utf-8", newline="") as source_file, positive_detail_path.open(
        "w", encoding="utf-8", newline=""
    ) as positive_file, split_path.open("w", encoding="utf-8", newline="") as split_file:
        reader = csv.DictReader(source_file, delimiter="\t")
        positive_writer = csv.writer(positive_file, delimiter="\t")
        split_writer = csv.writer(split_file, delimiter="\t")

        positive_writer.writerow(
            [
                "review_id",
                "user_idx",
                "item_idx",
                "raw_user_id",
                "raw_business_id",
                "date",
                "stars",
                "source_user_idx",
                "source_item_idx",
            ]
        )
        split_writer.writerow(
            [
                "review_id",
                "split",
                "user_idx",
                "item_idx",
                "raw_user_id",
                "raw_business_id",
                "date",
                "stars",
                "source_user_idx",
                "source_item_idx",
            ]
        )

        for row in reader:
            old_user_idx = int(row["user_idx"])
            old_item_idx = int(row["item_idx"])
            pair = (old_user_idx, old_item_idx)
            if pair not in kept_pairs_old:
                continue
            new_user_idx = old_user_to_new_user[old_user_idx]
            new_item_idx = old_item_to_new_item[old_item_idx]
            positive_writer.writerow(
                [
                    row["review_id"],
                    new_user_idx,
                    new_item_idx,
                    row["raw_user_id"],
                    row["raw_business_id"],
                    row["date"],
                    row["stars"],
                    old_user_idx,
                    old_item_idx,
                ]
            )
            split_writer.writerow(
                [
                    row["review_id"],
                    split_by_pair[pair],
                    new_user_idx,
                    new_item_idx,
                    row["raw_user_id"],
                    row["raw_business_id"],
                    row["date"],
                    row["stars"],
                    old_user_idx,
                    old_item_idx,
                ]
            )
            kept_rows += 1
            split_rows += 1

    return kept_rows, split_rows


def stream_filter_jsonl_dir(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    *,
    keep_predicate,
    transform,
    export_shard_size: int,
) -> int:
    writer = JsonlShardWriter(output_dir, prefix, export_shard_size)
    kept = 0
    source_files = sorted(path for path in source_dir.glob("*.jsonl") if path.is_file())
    for source_file in source_files:
        for payload in iter_jsonl(source_file):
            if not keep_predicate(payload):
                continue
            writer.write(transform(payload))
            kept += 1
    writer.close()
    return kept


def filter_aligned_exports(
    output_dir: Path,
    source_dir: Path,
    kept_pairs_old: set[tuple[int, int]],
    old_user_to_new_user: dict[int, int],
    old_item_to_new_item: dict[int, int],
    *,
    export_shard_size: int,
    tip_filter_mode: str,
) -> dict[str, int]:
    kept_old_users = set(old_user_to_new_user)
    kept_old_items = set(old_item_to_new_item)

    aligned_reviews = stream_filter_jsonl_dir(
        source_dir / "aligned_reviews",
        output_dir / "aligned_reviews",
        "aligned_reviews",
        keep_predicate=lambda payload: (int(payload["user_idx"]), int(payload["item_idx"])) in kept_pairs_old,
        transform=lambda payload: {
            **payload,
            "user_idx": old_user_to_new_user[int(payload["user_idx"])],
            "item_idx": old_item_to_new_item[int(payload["item_idx"])],
            "entity_id": old_item_to_new_item[int(payload["item_idx"])],
        },
        export_shard_size=export_shard_size,
    )

    aligned_users = stream_filter_jsonl_dir(
        source_dir / "aligned_users",
        output_dir / "aligned_users",
        "aligned_users",
        keep_predicate=lambda payload: int(payload["user_idx"]) in kept_old_users,
        transform=lambda payload: {
            **payload,
            "user_idx": old_user_to_new_user[int(payload["user_idx"])],
        },
        export_shard_size=export_shard_size,
    )

    aligned_businesses = stream_filter_jsonl_dir(
        source_dir / "aligned_businesses",
        output_dir / "aligned_businesses",
        "aligned_businesses",
        keep_predicate=lambda payload: int(payload["item_idx"]) in kept_old_items,
        transform=lambda payload: {
            **payload,
            "item_idx": old_item_to_new_item[int(payload["item_idx"])],
            "entity_id": old_item_to_new_item[int(payload["item_idx"])],
        },
        export_shard_size=export_shard_size,
    )

    aligned_checkins = stream_filter_jsonl_dir(
        source_dir / "aligned_checkins",
        output_dir / "aligned_checkins",
        "aligned_checkins",
        keep_predicate=lambda payload: int(payload["item_idx"]) in kept_old_items,
        transform=lambda payload: {
            **payload,
            "item_idx": old_item_to_new_item[int(payload["item_idx"])],
            "entity_id": old_item_to_new_item[int(payload["item_idx"])],
        },
        export_shard_size=export_shard_size,
    )

    def tip_keep_predicate(payload: dict[str, object]) -> bool:
        old_item_idx = int(payload["item_idx"])
        if old_item_idx not in kept_old_items:
            return False
        old_user_idx = payload.get("user_idx")
        if tip_filter_mode == "item_only":
            return True
        if old_user_idx is None:
            return False
        return int(old_user_idx) in kept_old_users

    def tip_transform(payload: dict[str, object]) -> dict[str, object]:
        result = dict(payload)
        result["item_idx"] = old_item_to_new_item[int(payload["item_idx"])]
        result["entity_id"] = result["item_idx"]
        old_user_idx = payload.get("user_idx")
        if old_user_idx is not None and int(old_user_idx) in kept_old_users:
            result["user_idx"] = old_user_to_new_user[int(old_user_idx)]
        else:
            result["user_idx"] = None
        return result

    aligned_tips = stream_filter_jsonl_dir(
        source_dir / "aligned_tips",
        output_dir / "aligned_tips",
        "aligned_tips",
        keep_predicate=tip_keep_predicate,
        transform=tip_transform,
        export_shard_size=export_shard_size,
    )

    return {
        "aligned_reviews": aligned_reviews,
        "aligned_users": aligned_users,
        "aligned_businesses": aligned_businesses,
        "aligned_checkins": aligned_checkins,
        "aligned_tips": aligned_tips,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    ensure_clean_output_dir(output_dir, keep_output=args.keep_output)
    log(f"source_dir={source_dir}")
    log(f"output_dir={output_dir}")

    all_positive = load_all_positive(source_dir / "interactions" / "all_positive.txt")
    total_users = len(all_positive)
    total_interactions = sum(len(row.items) for row in all_positive)
    log(f"loaded {total_users:,} users and {total_interactions:,} positive interactions")

    shuffled_users = list(all_positive)
    random.Random(args.seed).shuffle(shuffled_users)

    best_summary, kept_user_items_old, search_history = search_best_candidate(
        shuffled_users,
        k_core=args.k_core,
        target_interactions=args.target_interactions,
    )
    log(
        f"selected prefix={best_summary.prefix_users:,}; final users={best_summary.final_users:,}, "
        f"items={best_summary.final_items:,}, interactions={best_summary.final_interactions:,}, "
        f"error={best_summary.error:+,}"
    )

    old_user_to_new_user = {
        old_user_idx: new_user_idx for new_user_idx, old_user_idx in enumerate(sorted(kept_user_items_old))
    }
    train_items_old, test_items_old, split_by_pair = build_split(
        kept_user_items_old,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    (
        kept_entity_rows,
        kept_relation_rows,
        old_item_to_new_item,
        old_entity_to_new_entity,
        reachable_old_entity_ids,
        old_relation_to_new_relation,
        closure_iterations,
    ) = build_entity_and_relation_maps(kept_user_items_old, source_dir)

    interaction_stats = write_interaction_files(
        output_dir,
        kept_user_items_old,
        train_items_old,
        test_items_old,
        old_user_to_new_user,
        old_item_to_new_item,
    )

    kept_pairs_old = {
        (old_user_idx, old_item_idx)
        for old_user_idx, items in kept_user_items_old.items()
        for old_item_idx in items
    }
    positive_rows, split_rows = stream_filter_positive_details(
        output_dir,
        source_dir,
        kept_pairs_old,
        split_by_pair,
        old_user_to_new_user,
        old_item_to_new_item,
    )
    log(f"wrote {positive_rows:,} positive_interactions rows and {split_rows:,} review_split rows")

    mapping_stats = write_mapping_files(
        output_dir,
        source_dir,
        kept_user_items_old,
        old_user_to_new_user,
        old_item_to_new_item,
        kept_entity_rows,
        kept_relation_rows,
        old_entity_to_new_entity,
        old_relation_to_new_relation,
    )

    kg_triples = write_kg_files(
        output_dir,
        source_dir,
        kept_entity_rows,
        kept_relation_rows,
        old_entity_to_new_entity,
        old_relation_to_new_relation,
        reachable_old_entity_ids,
    )
    log(f"wrote {kg_triples:,} KG triples")

    export_stats = filter_aligned_exports(
        output_dir,
        source_dir,
        kept_pairs_old,
        old_user_to_new_user,
        old_item_to_new_item,
        export_shard_size=args.export_shard_size,
        tip_filter_mode=args.tip_filter_mode,
    )
    log(
        "filtered aligned exports: "
        + ", ".join(f"{name}={count:,}" for name, count in sorted(export_stats.items()))
    )

    summary_payload = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_interactions": args.target_interactions,
        "random_seed": args.seed,
        "k_core": args.k_core,
        "test_ratio": args.test_ratio,
        "tip_filter_mode": args.tip_filter_mode,
        "source_dataset": {
            "users": total_users,
            "positive_interactions": total_interactions,
        },
        "selection": {
            "selected_prefix_users": best_summary.prefix_users,
            "final_users": best_summary.final_users,
            "final_items": best_summary.final_items,
            "final_positive_interactions": best_summary.final_interactions,
            "error_vs_target": best_summary.error,
            "absolute_error": abs(best_summary.error),
            "sampling_5core_iterations": best_summary.kcore_iterations,
            "kg_closure_iterations": closure_iterations,
            "search_history": [
                {
                    "prefix_users": row.prefix_users,
                    "final_users": row.final_users,
                    "final_items": row.final_items,
                    "final_interactions": row.final_interactions,
                    "error": row.error,
                    "sampling_5core_iterations": row.kcore_iterations,
                }
                for row in search_history
            ],
        },
        "outputs": {
            "users": mapping_stats["users"],
            "items": mapping_stats["items"],
            "entities": mapping_stats["entities"],
            "relations": mapping_stats["relations"],
            "positive_interactions": interaction_stats["positive_interactions"],
            "train_interactions": interaction_stats["train_interactions"],
            "test_interactions": interaction_stats["test_interactions"],
            "kg_triples": kg_triples,
            "positive_detail_rows": positive_rows,
            "review_split_rows": split_rows,
            "aligned_reviews": export_stats["aligned_reviews"],
            "aligned_users": export_stats["aligned_users"],
            "aligned_businesses": export_stats["aligned_businesses"],
            "aligned_checkins": export_stats["aligned_checkins"],
            "aligned_tips": export_stats["aligned_tips"],
        },
        "notes": {
            "sampling_unit": "user",
            "kg_rule": "keep only entities reachable from kept business entities in the processed KG",
            "split_rule": "per-user random 8/2 split with at least one test interaction",
            "all_positive_source": "interactions/all_positive.txt",
            "original_raw_files_scanned": False,
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, ensure_ascii=False, indent=2)

    log("done")


if __name__ == "__main__":
    main()

