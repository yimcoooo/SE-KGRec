import argparse
import bisect
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "processed_data" / "processed_full_analysis"
    parser = argparse.ArgumentParser(
        description="Build a smaller KGAT dataset from an existing kgat_dataset directory."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=processed_dir / "kgat_dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=processed_dir / "kgat_dataset_500k",
    )
    parser.add_argument("--target-interactions", type=int, default=500000)
    parser.add_argument("--user-min", type=int, default=5)
    parser.add_argument("--item-min", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def reset_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in sorted(output_dir.iterdir(), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            reset_output_dir(child)
            if child.exists():
                child.rmdir()


class ProgressBar:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(total, 1)
        self.start_time = time.time()
        self.last_update = 0.0
        self.last_len = 0

    def update(self, value: int, extra: str = "") -> None:
        now = time.time()
        if now - self.last_update < 0.2 and value < self.total:
            return
        self.last_update = now
        ratio = min(max(value / self.total, 0.0), 1.0)
        width = 28
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(now - self.start_time, 1e-6)
        speed = value / 1024 / 1024 / elapsed
        message = (
            f"\r{self.label:<18} [{bar}] {ratio * 100:6.2f}% "
            f"{value / 1024 / 1024:9.1f}/{self.total / 1024 / 1024:9.1f} MB "
            f"{speed:7.2f} MB/s"
        )
        if extra:
            message += f" | {extra}"
        padding = max(self.last_len - len(message), 0)
        sys.stdout.write(message + (" " * padding))
        sys.stdout.flush()
        self.last_len = len(message)

    def close(self) -> None:
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()


def load_positive_records(path: Path) -> List[Tuple[int, Tuple[int, ...]]]:
    records: List[Tuple[int, Tuple[int, ...]]] = []
    progress = ProgressBar("load_positive", path.stat().st_size)
    processed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            processed += len(line.encode("utf-8"))
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = tuple(int(value) for value in parts[1:])
            records.append((user_id, item_ids))
            if len(records) % 5000 == 0:
                progress.update(processed, extra=f"users={len(records):,}")
    progress.close()
    return records


def load_reverse_map(path: Path, expected_columns: int) -> Dict[int, str]:
    reverse_map: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < expected_columns:
                continue
            if expected_columns == 2:
                reverse_map[int(parts[-1])] = " ".join(parts[:-1])
            else:
                reverse_map[int(parts[1])] = parts[0]
    return reverse_map


def prune_to_kcore(
    selected_records: Sequence[Tuple[int, Tuple[int, ...]]],
    user_min: int,
    item_min: int,
) -> Tuple[Dict[int, List[int]], Counter, int]:
    user_items: Dict[int, List[int]] = {
        user_id: list(item_ids)
        for user_id, item_ids in selected_records
        if len(item_ids) >= user_min
    }
    iteration = 0

    while True:
        iteration += 1
        item_counts: Counter = Counter()
        for item_ids in user_items.values():
            item_counts.update(item_ids)

        valid_items = {item_id for item_id, count in item_counts.items() if count >= item_min}
        next_user_items: Dict[int, List[int]] = {}
        changed = False

        for user_id, item_ids in user_items.items():
            filtered_items = [item_id for item_id in item_ids if item_id in valid_items]
            if len(filtered_items) >= user_min:
                next_user_items[user_id] = filtered_items
                if filtered_items != item_ids:
                    changed = True
            else:
                changed = True

        if not changed and len(next_user_items) == len(user_items):
            final_counts: Counter = Counter()
            for item_ids in next_user_items.values():
                final_counts.update(item_ids)
            return next_user_items, final_counts, iteration

        user_items = next_user_items


def build_selected_records(
    records: Sequence[Tuple[int, Tuple[int, ...]]],
    order: Sequence[int],
    prefix_users: int,
) -> List[Tuple[int, Tuple[int, ...]]]:
    return [records[index] for index in order[:prefix_users]]


def evaluate_prefix(
    records: Sequence[Tuple[int, Tuple[int, ...]]],
    order: Sequence[int],
    prefix_users: int,
    user_min: int,
    item_min: int,
    cache: Dict[int, Dict],
    materialize: bool = False,
) -> Dict:
    if prefix_users in cache and (not materialize or "user_items" in cache[prefix_users]):
        return cache[prefix_users]

    selected_records = build_selected_records(records, order, prefix_users)
    user_items, item_counts, iterations = prune_to_kcore(selected_records, user_min, item_min)
    interactions = sum(len(item_ids) for item_ids in user_items.values())

    result = {
        "prefix_users": prefix_users,
        "users": len(user_items),
        "items": len(item_counts),
        "interactions": interactions,
        "iterations": iterations,
    }
    if materialize:
        result["user_items"] = user_items
        result["item_counts"] = item_counts

    cache[prefix_users] = result
    return result


def search_best_subset(
    records: Sequence[Tuple[int, Tuple[int, ...]]],
    order: Sequence[int],
    target_interactions: int,
    user_min: int,
    item_min: int,
) -> Dict:
    cache: Dict[int, Dict] = {}
    cumulative_raw: List[int] = []
    running = 0
    for index in order:
        running += len(records[index][1])
        cumulative_raw.append(running)

    high = bisect.bisect_left(cumulative_raw, max(target_interactions, user_min * 1000)) + 1
    high = min(max(high, user_min), len(order))
    low = user_min

    high_result = evaluate_prefix(records, order, high, user_min, item_min, cache)
    while high_result["interactions"] < target_interactions and high < len(order):
        low = high
        high = min(len(order), high * 2)
        high_result = evaluate_prefix(records, order, high, user_min, item_min, cache)

    best = high_result
    low_result = evaluate_prefix(records, order, low, user_min, item_min, cache)
    if abs(low_result["interactions"] - target_interactions) < abs(
        best["interactions"] - target_interactions
    ):
        best = low_result

    while low <= high:
        mid = (low + high) // 2
        mid_result = evaluate_prefix(records, order, mid, user_min, item_min, cache)
        if abs(mid_result["interactions"] - target_interactions) < abs(
            best["interactions"] - target_interactions
        ):
            best = mid_result
        if mid_result["interactions"] < target_interactions:
            low = mid + 1
        else:
            high = mid - 1

    candidate_ns = set()
    center = best["prefix_users"]
    for step in (1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000):
        for candidate in (center - step, center + step):
            if user_min <= candidate <= len(order):
                candidate_ns.add(candidate)
    candidate_ns.update({best["prefix_users"], low, high})

    for candidate in sorted(candidate_ns):
        candidate_result = evaluate_prefix(
            records, order, candidate, user_min, item_min, cache
        )
        if abs(candidate_result["interactions"] - target_interactions) < abs(
            best["interactions"] - target_interactions
        ):
            best = candidate_result

    return evaluate_prefix(
        records, order, best["prefix_users"], user_min, item_min, cache, materialize=True
    )


def iter_triples(path: Path) -> Iterable[Tuple[int, int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            head, relation, tail = line.strip().split()
            yield int(head), int(relation), int(tail)


def expand_entity_closure(path: Path, seed_items: Sequence[int]) -> Tuple[set, int]:
    retained = set(seed_items)
    iterations = 0
    while True:
        iterations += 1
        new_entities = set()
        for head, _relation, tail in iter_triples(path):
            if head in retained and tail not in retained:
                new_entities.add(tail)
        if not new_entities:
            break
        retained.update(new_entities)
    return retained, iterations


def collect_used_relations(path: Path, retained_entities: set) -> set:
    used_relations = set()
    for head, relation, tail in iter_triples(path):
        if head in retained_entities and tail in retained_entities:
            used_relations.add(relation)
    return used_relations


def write_reduced_outputs(
    input_dir: Path,
    output_dir: Path,
    final_user_items: Dict[int, List[int]],
    final_item_counts: Counter,
    entity_closure: set,
    relation_names: Dict[int, str],
    user_org_ids: Dict[int, str],
    item_org_ids: Dict[int, str],
    entity_org_ids: Dict[int, str],
    test_ratio: float,
    random_seed: int,
) -> Dict:
    all_positive_path = output_dir / "all_positive_interactions.txt"
    train_path = output_dir / "train.txt"
    test_path = output_dir / "test.txt"
    user_list_path = output_dir / "user_list.txt"
    item_list_path = output_dir / "item_list.txt"
    entity_list_path = output_dir / "entity_list.txt"
    relation_list_path = output_dir / "relation_list.txt"
    kg_final_path = output_dir / "kg_final.txt"

    selected_users = sorted(final_user_items)
    selected_items = sorted(final_item_counts)
    non_item_entities = sorted(entity_id for entity_id in entity_closure if entity_id not in final_item_counts)

    new_user_ids = {old_user_id: new_user_id for new_user_id, old_user_id in enumerate(selected_users)}
    new_item_ids = {old_item_id: new_item_id for new_item_id, old_item_id in enumerate(selected_items)}

    new_entity_ids: Dict[int, int] = {}
    next_entity_id = 0
    for old_item_id in selected_items:
        new_entity_ids[old_item_id] = next_entity_id
        next_entity_id += 1
    for old_entity_id in non_item_entities:
        new_entity_ids[old_entity_id] = next_entity_id
        next_entity_id += 1

    used_relations = collect_used_relations(input_dir / "kg_final.txt", entity_closure)
    new_relation_ids = {
        old_relation_id: new_relation_id
        for new_relation_id, old_relation_id in enumerate(sorted(used_relations))
    }

    with user_list_path.open("w", encoding="utf-8") as handle:
        handle.write("org_id remap_id\n")
        for old_user_id in selected_users:
            handle.write(f"{user_org_ids[old_user_id]} {new_user_ids[old_user_id]}\n")

    with item_list_path.open("w", encoding="utf-8") as handle:
        handle.write("org_id remap_id entity_id\n")
        for old_item_id in selected_items:
            new_item_id = new_item_ids[old_item_id]
            handle.write(f"{item_org_ids[old_item_id]} {new_item_id} {new_entity_ids[old_item_id]}\n")

    with entity_list_path.open("w", encoding="utf-8") as handle:
        handle.write("org_id remap_id\n")
        for old_entity_id, new_entity_id in sorted(new_entity_ids.items(), key=lambda pair: pair[1]):
            handle.write(f"{entity_org_ids[old_entity_id]} {new_entity_id}\n")

    with relation_list_path.open("w", encoding="utf-8") as handle:
        handle.write("org_id remap_id\n")
        for old_relation_id, new_relation_id in sorted(
            new_relation_ids.items(), key=lambda pair: pair[1]
        ):
            handle.write(f"{relation_names[old_relation_id]} {new_relation_id}\n")

    train_users = 0
    test_users = 0
    train_interactions = 0
    test_interactions = 0
    effective_test_ratio = min(max(test_ratio, 0.0), 0.5)
    rng = random.Random(random_seed)

    with all_positive_path.open("w", encoding="utf-8") as positive_handle, train_path.open(
        "w", encoding="utf-8"
    ) as train_handle, test_path.open("w", encoding="utf-8") as test_handle:
        for old_user_id in selected_users:
            new_user_id = new_user_ids[old_user_id]
            item_ids = sorted(new_item_ids[old_item_id] for old_item_id in final_user_items[old_user_id])
            positive_handle.write(" ".join([str(new_user_id)] + [str(item_id) for item_id in item_ids]) + "\n")

            test_count = max(1, int(len(item_ids) * effective_test_ratio))
            test_count = min(test_count, len(item_ids) - 1)
            shuffled = list(item_ids)
            rng.shuffle(shuffled)
            test_set = set(shuffled[:test_count])
            train_items = [item_id for item_id in item_ids if item_id not in test_set]
            test_items = [item_id for item_id in item_ids if item_id in test_set]

            train_handle.write(" ".join([str(new_user_id)] + [str(item_id) for item_id in train_items]) + "\n")
            test_handle.write(" ".join([str(new_user_id)] + [str(item_id) for item_id in test_items]) + "\n")

            train_users += 1
            test_users += 1
            train_interactions += len(train_items)
            test_interactions += len(test_items)

    with kg_final_path.open("w", encoding="utf-8") as handle:
        kg_triples = 0
        for head, relation, tail in iter_triples(input_dir / "kg_final.txt"):
            if head in entity_closure and tail in entity_closure and relation in new_relation_ids:
                handle.write(
                    f"{new_entity_ids[head]} {new_relation_ids[relation]} {new_entity_ids[tail]}\n"
                )
                kg_triples += 1

    return {
        "users": len(selected_users),
        "items": len(selected_items),
        "entities": len(new_entity_ids),
        "relations": len(new_relation_ids),
        "all_positive_interactions": sum(len(item_ids) for item_ids in final_user_items.values()),
        "train_users": train_users,
        "test_users": test_users,
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "kg_triples": kg_triples,
        "paths": {
            "all_positive_interactions": str(all_positive_path),
            "train": str(train_path),
            "test": str(test_path),
            "user_list": str(user_list_path),
            "item_list": str(item_list_path),
            "entity_list": str(entity_list_path),
            "relation_list": str(relation_list_path),
            "kg_final": str(kg_final_path),
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if args.force and output_dir.exists():
        reset_output_dir(output_dir)
        if output_dir.exists():
            output_dir.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading positive interactions from: {input_dir}")
    records = load_positive_records(input_dir / "all_positive_interactions.txt")

    print("Loading current mapping files")
    user_org_ids = load_reverse_map(input_dir / "user_list.txt", 2)
    item_org_ids = load_reverse_map(input_dir / "item_list.txt", 3)
    entity_org_ids = load_reverse_map(input_dir / "entity_list.txt", 2)
    relation_names = load_reverse_map(input_dir / "relation_list.txt", 2)

    rng = random.Random(args.random_seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    print("Searching for a subset close to the target interaction count")
    best = search_best_subset(
        records=records,
        order=order,
        target_interactions=args.target_interactions,
        user_min=args.user_min,
        item_min=args.item_min,
    )

    print(
        "Selected subset:",
        json.dumps(
            {
                "prefix_users": best["prefix_users"],
                "users": best["users"],
                "items": best["items"],
                "interactions": best["interactions"],
                "iterations": best["iterations"],
            },
            ensure_ascii=False,
        ),
    )

    print("Expanding KG entity closure from selected items")
    entity_closure, kg_iterations = expand_entity_closure(
        input_dir / "kg_final.txt", sorted(best["item_counts"])
    )

    print("Writing reduced dataset")
    stats = write_reduced_outputs(
        input_dir=input_dir,
        output_dir=output_dir,
        final_user_items=best["user_items"],
        final_item_counts=best["item_counts"],
        entity_closure=entity_closure,
        relation_names=relation_names,
        user_org_ids=user_org_ids,
        item_org_ids=item_org_ids,
        entity_org_ids=entity_org_ids,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    summary = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_interactions": args.target_interactions,
        "actual_interactions": stats["all_positive_interactions"],
        "interaction_error": stats["all_positive_interactions"] - args.target_interactions,
        "random_seed": args.random_seed,
        "test_ratio": args.test_ratio,
        "user_min": args.user_min,
        "item_min": args.item_min,
        "search_prefix_users": best["prefix_users"],
        "subset_users": best["users"],
        "subset_items": best["items"],
        "subset_interactions": best["interactions"],
        "subset_kcore_iterations": best["iterations"],
        "kg_closure_iterations": kg_iterations,
        **stats,
    }
    write_json(output_dir / "summary.json", summary)

    print("\nReduced dataset finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
