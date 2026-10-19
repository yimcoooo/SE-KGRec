from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from pathlib import Path


DEFAULT_EXCLUDED_SHELVES = {
    "to-read",
    "currently-reading",
    "owned",
    "books-i-own",
    "kindle",
    "ebook",
    "favourites",
    "favorites",
    "default",
    "library",
    "my-books",
    "wishlist",
    "wish-list",
    "did-not-finish",
    "dnf",
    "unread",
    "read",
}


RELATION_PROFILES = {
    "rec_core3": [
        "written_by",
        "in_series",
        "tagged_with_shelf",
    ],
    "rec_plus5": [
        "written_by",
        "in_series",
        "tagged_with_shelf",
        "similar_to",
        "published_by",
    ],
    "rec_plus6": [
        "written_by",
        "in_series",
        "tagged_with_shelf",
        "similar_to",
        "published_by",
        "in_language",
    ],
    "full9": [
        "written_by",
        "in_series",
        "tagged_with_shelf",
        "similar_to",
        "published_by",
        "in_language",
        "published_in_year",
        "from_country",
        "has_format",
    ],
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    base_default = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a recommender-oriented KGAT dataset from Goodreads Books/Reviews/Interactions JSONL files."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=base_default)
    parser.add_argument(
        "--books-file",
        type=Path,
        default=Path("goodreads_books_fantasy_paranormal.json")
        / "goodreads_books_fantasy_paranormal.json",
    )
    parser.add_argument(
        "--reviews-file",
        type=Path,
        default=Path("goodreads_reviews_fantasy_paranormal.json")
        / "goodreads_reviews_fantasy_paranormal.json",
    )
    parser.add_argument(
        "--interactions-file",
        type=Path,
        default=Path("goodreads_interactions_fantasy_paranormal.json")
        / "goodreads_interactions_fantasy_paranormal.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("02_processed_data") / "aligned_full_pos_5core_rec_plus6_ratio",
    )

    parser.add_argument(
        "--interaction-source",
        choices=("reviews", "interactions", "union"),
        default="reviews",
        help="reviews is recommended for explicit positive feedback.",
    )
    parser.add_argument(
        "--include-read-as-positive",
        action="store_true",
        help="Only used when reading interactions: treat is_read=true with rating==0 as positive.",
    )
    parser.add_argument("--min-stars", type=float, default=4.0)
    parser.add_argument("--min-user-ratings", type=int, default=5)
    parser.add_argument("--min-item-ratings", type=int, default=5)
    parser.add_argument("--filter-rounds", type=int, default=30)

    parser.add_argument(
        "--relation-profile",
        choices=tuple(RELATION_PROFILES.keys()),
        default="rec_plus6",
        help="rec_plus6 excludes format and keeps recommendation-relevant relations.",
    )
    parser.add_argument("--min-shelf-vote", type=int, default=3)
    parser.add_argument("--max-shelves-per-book", type=int, default=15)
    parser.add_argument("--disable-generic-shelf-filter", action="store_true")
    parser.add_argument(
        "--similar-mode",
        choices=("internal", "external"),
        default="internal",
        help="internal keeps similar_to edges only when tail is also a kept item.",
    )

    parser.add_argument("--split-strategy", choices=("ratio", "leave_one_out"), default="ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026)

    parser.add_argument("--progress-every", type=int, default=500_000)
    parser.add_argument("--commit-every", type=int, default=200_000)
    return parser.parse_args()


def resolve_path(base_dir: Path, child: Path) -> Path:
    return child if child.is_absolute() else base_dir / child


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split()) if text else ""


def normalize_shelf(value: object) -> str:
    text = normalize_text(value).lower()
    return text.replace("_", "-") if text else ""


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positive_interactions (
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            rating REAL NOT NULL,
            PRIMARY KEY (user_id, item_id)
        ) WITHOUT ROWID;
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_positive_item ON positive_interactions(item_id);")
    conn.commit()
    return conn


def count_rows(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM positive_interactions;").fetchone()[0])


def count_distinct_users(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(DISTINCT user_id) FROM positive_interactions;").fetchone()[0])


def count_distinct_items(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(DISTINCT item_id) FROM positive_interactions;").fetchone()[0])


def min_user_degree(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MIN(cnt) FROM (SELECT COUNT(*) AS cnt FROM positive_interactions GROUP BY user_id);"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def min_item_degree(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MIN(cnt) FROM (SELECT COUNT(*) AS cnt FROM positive_interactions GROUP BY item_id);"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def ingest_from_reviews(
    reviews_path: Path,
    conn: sqlite3.Connection,
    *,
    min_stars: float,
    progress_every: int,
    commit_every: int,
) -> dict[str, int]:
    insert_sql = (
        "INSERT INTO positive_interactions(user_id, item_id, rating) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, item_id) DO UPDATE SET rating = MAX(rating, excluded.rating);"
    )
    scanned = 0
    kept = 0
    malformed = 0
    started = time.time()
    cursor = conn.cursor()

    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue

            user_id = normalize_text(payload.get("user_id"))
            item_id = normalize_text(payload.get("book_id"))
            if not user_id or not item_id:
                malformed += 1
                continue

            try:
                rating = float(payload.get("rating"))
            except (TypeError, ValueError):
                malformed += 1
                continue

            if rating < min_stars:
                continue

            cursor.execute(insert_sql, (user_id, item_id, rating))
            kept += 1

            if kept % commit_every == 0:
                conn.commit()
            if scanned % progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                speed = int(scanned / elapsed)
                print(
                    f"[reviews] scanned={scanned:,} kept={kept:,} malformed={malformed:,} rows_per_sec={speed:,}"
                )

    conn.commit()
    return {
        "scanned_reviews": scanned,
        "kept_reviews": kept,
        "malformed_reviews": malformed,
        "deduped_positive_pairs": count_rows(conn),
    }


def ingest_from_interactions(
    interactions_path: Path,
    conn: sqlite3.Connection,
    *,
    min_stars: float,
    include_read_as_positive: bool,
    progress_every: int,
    commit_every: int,
) -> dict[str, int]:
    insert_sql = (
        "INSERT INTO positive_interactions(user_id, item_id, rating) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, item_id) DO UPDATE SET rating = MAX(rating, excluded.rating);"
    )
    scanned = 0
    kept = 0
    malformed = 0
    kept_by_read = 0
    started = time.time()
    cursor = conn.cursor()

    with interactions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue

            user_id = normalize_text(payload.get("user_id"))
            item_id = normalize_text(payload.get("book_id"))
            if not user_id or not item_id:
                malformed += 1
                continue

            rating_raw = payload.get("rating")
            rating = None
            if rating_raw is not None:
                try:
                    rating = float(rating_raw)
                except (TypeError, ValueError):
                    rating = None

            keep = False
            keep_rating = min_stars
            if rating is not None and rating >= min_stars:
                keep = True
                keep_rating = rating
            elif include_read_as_positive and bool(payload.get("is_read")):
                keep = True
                kept_by_read += 1

            if not keep:
                continue

            cursor.execute(insert_sql, (user_id, item_id, keep_rating))
            kept += 1

            if kept % commit_every == 0:
                conn.commit()
            if scanned % progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                speed = int(scanned / elapsed)
                print(
                    f"[interactions] scanned={scanned:,} kept={kept:,} by_read={kept_by_read:,} malformed={malformed:,} rows_per_sec={speed:,}"
                )

    conn.commit()
    return {
        "scanned_interactions": scanned,
        "kept_interactions": kept,
        "kept_by_read": kept_by_read,
        "malformed_interactions": malformed,
        "deduped_positive_pairs": count_rows(conn),
    }


def apply_k_core(
    conn: sqlite3.Connection,
    *,
    min_user_ratings: int,
    min_item_ratings: int,
    max_rounds: int,
) -> list[dict[str, int]]:
    rounds: list[dict[str, int]] = []
    for round_idx in range(1, max_rounds + 1):
        before = count_rows(conn)

        conn.execute(
            """
            DELETE FROM positive_interactions
            WHERE user_id IN (
                SELECT user_id FROM (
                    SELECT user_id
                    FROM positive_interactions
                    GROUP BY user_id
                    HAVING COUNT(*) < ?
                )
            );
            """,
            (min_user_ratings,),
        )
        conn.commit()
        after_user_delete = count_rows(conn)

        conn.execute(
            """
            DELETE FROM positive_interactions
            WHERE item_id IN (
                SELECT item_id FROM (
                    SELECT item_id
                    FROM positive_interactions
                    GROUP BY item_id
                    HAVING COUNT(*) < ?
                )
            );
            """,
            (min_item_ratings,),
        )
        conn.commit()
        after_item_delete = count_rows(conn)

        removed_user_side = before - after_user_delete
        removed_item_side = after_user_delete - after_item_delete

        payload = {
            "round": round_idx,
            "removed_user_side": removed_user_side,
            "removed_item_side": removed_item_side,
            "remaining_interactions": after_item_delete,
            "remaining_users": count_distinct_users(conn),
            "remaining_items": count_distinct_items(conn),
        }
        rounds.append(payload)
        print(
            f"[kcore] round={round_idx} removed_user_side={removed_user_side:,} "
            f"removed_item_side={removed_item_side:,} remaining={after_item_delete:,}"
        )

        if removed_user_side == 0 and removed_item_side == 0:
            break

    return rounds


def split_items(
    item_indices: list[int],
    *,
    strategy: str,
    test_ratio: float,
    split_seed: int,
    user_idx: int,
) -> tuple[list[int], list[int]]:
    unique_sorted = sorted(set(item_indices))
    if len(unique_sorted) <= 1:
        return unique_sorted, []

    if strategy == "leave_one_out":
        return unique_sorted[:-1], unique_sorted[-1:]

    shuffled = unique_sorted[:]
    rng = random.Random(split_seed + user_idx)
    rng.shuffle(shuffled)

    raw_test_count = max(1, int(round(len(shuffled) * test_ratio)))
    test_count = min(raw_test_count, len(shuffled) - 1)

    test = sorted(shuffled[:test_count])
    train = sorted(shuffled[test_count:])
    return train, test


class EntityCatalog:
    def __init__(self, item_map: dict[str, int]) -> None:
        self.key_to_id: dict[tuple[str, str], int] = {}
        self.id_to_org_id: dict[int, str] = {}
        self.next_id = 0

        for item_id, entity_id in sorted(item_map.items(), key=lambda pair: pair[1]):
            self.key_to_id[("item", item_id)] = entity_id
            self.id_to_org_id[entity_id] = f"item::{item_id}"
            self.next_id = max(self.next_id, entity_id + 1)

    def ensure(self, entity_type: str, raw_value: str) -> int:
        normalized = normalize_text(raw_value)
        if not normalized:
            raise ValueError("raw_value cannot be empty")
        key = (entity_type, normalized)
        if key in self.key_to_id:
            return self.key_to_id[key]
        entity_id = self.next_id
        self.next_id += 1
        self.key_to_id[key] = entity_id
        self.id_to_org_id[entity_id] = f"{entity_type}::{normalized}"
        return entity_id

    def write_entity_list(self, path: Path) -> int:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("org_id\tremap_id\n")
            for entity_id in sorted(self.id_to_org_id):
                handle.write(f"{self.id_to_org_id[entity_id]}\t{entity_id}\n")
        return len(self.id_to_org_id)


def export_interactions_and_mappings(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    strategy: str,
    test_ratio: float,
    split_seed: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    users = [
        row[0]
        for row in conn.execute("SELECT DISTINCT user_id FROM positive_interactions ORDER BY user_id;")
    ]
    items = [
        row[0]
        for row in conn.execute("SELECT DISTINCT item_id FROM positive_interactions ORDER BY item_id;")
    ]

    user_map = {user_id: idx for idx, user_id in enumerate(users)}
    item_map = {item_id: idx for idx, item_id in enumerate(items)}

    with (output_dir / "user_list.txt").open("w", encoding="utf-8") as handle:
        handle.write("org_id\tremap_id\n")
        for user_id, user_idx in user_map.items():
            handle.write(f"{user_id}\t{user_idx}\n")

    with (output_dir / "item_list.txt").open("w", encoding="utf-8") as handle:
        handle.write("org_id\tremap_id\n")
        for item_id, item_idx in item_map.items():
            handle.write(f"{item_id}\t{item_idx}\n")

    stats = {
        "users": len(user_map),
        "items": len(item_map),
        "all_positive_interactions": 0,
        "train_users": 0,
        "test_users": 0,
        "train_interactions": 0,
        "test_interactions": 0,
    }

    with (output_dir / "all_positive_interactions.txt").open("w", encoding="utf-8") as all_handle, (
        output_dir / "train.txt"
    ).open("w", encoding="utf-8") as train_handle, (output_dir / "test.txt").open(
        "w", encoding="utf-8"
    ) as test_handle:
        cursor = conn.execute(
            "SELECT user_id, item_id FROM positive_interactions ORDER BY user_id, item_id;"
        )
        current_user = None
        current_items: list[str] = []

        def flush_user() -> None:
            if current_user is None or not current_items:
                return
            user_idx = user_map[current_user]
            item_indices = [item_map[item] for item in current_items]
            train_items, test_items = split_items(
                item_indices,
                strategy=strategy,
                test_ratio=test_ratio,
                split_seed=split_seed,
                user_idx=user_idx,
            )

            all_items = sorted(set(train_items + test_items))
            all_handle.write(" ".join([str(user_idx)] + [str(i) for i in all_items]) + "\n")
            train_handle.write(" ".join([str(user_idx)] + [str(i) for i in train_items]) + "\n")
            if test_items:
                test_handle.write(" ".join([str(user_idx)] + [str(i) for i in test_items]) + "\n")

            stats["all_positive_interactions"] += len(all_items)
            stats["train_users"] += 1
            stats["train_interactions"] += len(train_items)
            if test_items:
                stats["test_users"] += 1
                stats["test_interactions"] += len(test_items)

        for user_id, item_id in cursor:
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                flush_user()
                current_user = user_id
                current_items = []
            current_items.append(item_id)

        flush_user()

    return user_map, item_map, stats


def build_goodreads_kg(
    output_dir: Path,
    *,
    books_path: Path,
    item_map: dict[str, int],
    profile_name: str,
    min_shelf_vote: int,
    max_shelves_per_book: int,
    filter_generic_shelves: bool,
    similar_mode: str,
    progress_every: int,
) -> tuple[int, int, int, dict[str, int]]:
    relation_names = RELATION_PROFILES[profile_name]
    relation_to_id = {name: idx for idx, name in enumerate(relation_names)}

    relation_counts = {name: 0 for name in relation_names}

    with (output_dir / "relation_list.txt").open("w", encoding="utf-8") as handle:
        handle.write("org_id\tremap_id\n")
        for name, rel_id in relation_to_id.items():
            handle.write(f"{name}\t{rel_id}\n")

    entity_catalog = EntityCatalog(item_map)
    kg_edges = 0
    scanned = 0
    kept_items = 0

    with (output_dir / "kg_final.txt").open("w", encoding="utf-8") as kg_handle, books_path.open(
        "r", encoding="utf-8"
    ) as books_handle:
        for line in books_handle:
            scanned += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            item_id = normalize_text(payload.get("book_id"))
            if not item_id or item_id not in item_map:
                continue

            kept_items += 1
            head_id = item_map[item_id]
            seen_edges: set[tuple[int, int, int]] = set()

            if "written_by" in relation_to_id:
                for author in payload.get("authors") or []:
                    if not isinstance(author, dict):
                        continue
                    author_id = normalize_text(author.get("author_id") or author.get("name"))
                    if not author_id:
                        continue
                    tail_id = entity_catalog.ensure("author", author_id)
                    edge = (head_id, relation_to_id["written_by"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["written_by"] += 1

            if "in_series" in relation_to_id:
                for series_id in payload.get("series") or []:
                    sid = normalize_text(series_id)
                    if not sid:
                        continue
                    tail_id = entity_catalog.ensure("series", sid)
                    edge = (head_id, relation_to_id["in_series"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["in_series"] += 1

            if "published_by" in relation_to_id:
                publisher = normalize_text(payload.get("publisher"))
                if publisher:
                    tail_id = entity_catalog.ensure("publisher", publisher)
                    edge = (head_id, relation_to_id["published_by"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["published_by"] += 1

            if "in_language" in relation_to_id:
                language = normalize_text(payload.get("language_code"))
                if language:
                    tail_id = entity_catalog.ensure("language", language)
                    edge = (head_id, relation_to_id["in_language"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["in_language"] += 1

            if "published_in_year" in relation_to_id:
                year = normalize_text(payload.get("publication_year"))
                if year:
                    tail_id = entity_catalog.ensure("year", year)
                    edge = (head_id, relation_to_id["published_in_year"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["published_in_year"] += 1

            if "from_country" in relation_to_id:
                country = normalize_text(payload.get("country_code"))
                if country:
                    tail_id = entity_catalog.ensure("country", country)
                    edge = (head_id, relation_to_id["from_country"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["from_country"] += 1

            if "has_format" in relation_to_id:
                fmt = normalize_text(payload.get("format"))
                if fmt:
                    tail_id = entity_catalog.ensure("format", fmt)
                    edge = (head_id, relation_to_id["has_format"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["has_format"] += 1

            if "tagged_with_shelf" in relation_to_id:
                shelves = []
                for shelf in payload.get("popular_shelves") or []:
                    if not isinstance(shelf, dict):
                        continue
                    shelf_name = normalize_shelf(shelf.get("name"))
                    if not shelf_name:
                        continue
                    if filter_generic_shelves and shelf_name in DEFAULT_EXCLUDED_SHELVES:
                        continue
                    shelf_count = safe_int(shelf.get("count"), default=0)
                    if shelf_count < min_shelf_vote:
                        continue
                    shelves.append((shelf_name, shelf_count))

                shelves.sort(key=lambda pair: (-pair[1], pair[0]))
                for shelf_name, _ in shelves[:max_shelves_per_book]:
                    tail_id = entity_catalog.ensure("shelf", shelf_name)
                    edge = (head_id, relation_to_id["tagged_with_shelf"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["tagged_with_shelf"] += 1

            if "similar_to" in relation_to_id:
                for sid in payload.get("similar_books") or []:
                    similar_id = normalize_text(sid)
                    if not similar_id:
                        continue

                    if similar_mode == "internal":
                        if similar_id not in item_map:
                            continue
                        tail_id = item_map[similar_id]
                    else:
                        tail_id = entity_catalog.ensure("book", similar_id)

                    edge = (head_id, relation_to_id["similar_to"], tail_id)
                    if edge not in seen_edges:
                        kg_handle.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
                        seen_edges.add(edge)
                        kg_edges += 1
                        relation_counts["similar_to"] += 1

            if scanned % progress_every == 0:
                print(
                    f"[books] scanned={scanned:,} kept_items={kept_items:,} entities={len(entity_catalog.id_to_org_id):,} kg_edges={kg_edges:,}"
                )

    entity_count = entity_catalog.write_entity_list(output_dir / "entity_list.txt")
    relation_count = len(relation_to_id)
    return entity_count, relation_count, kg_edges, relation_counts


def write_summary(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = resolve_path(base_dir, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    books_path = resolve_path(base_dir, args.books_file)
    reviews_path = resolve_path(base_dir, args.reviews_file)
    interactions_path = resolve_path(base_dir, args.interactions_file)

    required = {"books": books_path}
    if args.interaction_source in {"reviews", "union"}:
        required["reviews"] = reviews_path
    if args.interaction_source in {"interactions", "union"}:
        required["interactions"] = interactions_path

    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + "; ".join(missing))

    started = time.time()
    db_path = output_dir / "work" / "interactions.sqlite"
    conn = open_sqlite(db_path)

    ingest_stats: dict[str, int] = {}
    if args.interaction_source in {"reviews", "union"}:
        print("[1/5] ingest positives from reviews")
        ingest_stats.update(
            ingest_from_reviews(
                reviews_path,
                conn,
                min_stars=args.min_stars,
                progress_every=args.progress_every,
                commit_every=args.commit_every,
            )
        )

    if args.interaction_source in {"interactions", "union"}:
        print("[1/5] ingest positives from interactions")
        ingest_stats.update(
            ingest_from_interactions(
                interactions_path,
                conn,
                min_stars=args.min_stars,
                include_read_as_positive=args.include_read_as_positive,
                progress_every=args.progress_every,
                commit_every=args.commit_every,
            )
        )

    print("[2/5] apply bipartite n-core")
    kcore_rounds = apply_k_core(
        conn,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
        max_rounds=args.filter_rounds,
    )

    print("[3/5] export interactions and mapping tables")
    user_map, item_map, split_stats = export_interactions_and_mappings(
        conn,
        output_dir,
        strategy=args.split_strategy,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    print("[4/5] build recommendation-oriented KG")
    entity_count, relation_count, kg_triples, relation_edge_counts = build_goodreads_kg(
        output_dir,
        books_path=books_path,
        item_map=item_map,
        profile_name=args.relation_profile,
        min_shelf_vote=args.min_shelf_vote,
        max_shelves_per_book=args.max_shelves_per_book,
        filter_generic_shelves=not args.disable_generic_shelf_filter,
        similar_mode=args.similar_mode,
        progress_every=args.progress_every,
    )

    print("[5/5] write summary")
    summary = {
        "input": {
            "base_dir": str(base_dir),
            "books_path": str(books_path),
            "reviews_path": str(reviews_path),
            "interactions_path": str(interactions_path),
            "interaction_source": args.interaction_source,
        },
        "filtering": {
            "min_stars": args.min_stars,
            "min_user_ratings": args.min_user_ratings,
            "min_item_ratings": args.min_item_ratings,
            "kcore_rounds": kcore_rounds,
            "final_min_user_degree": min_user_degree(conn),
            "final_min_item_degree": min_item_degree(conn),
        },
        "relations": {
            "profile": args.relation_profile,
            "relation_names": RELATION_PROFILES[args.relation_profile],
            "relation_count": relation_count,
            "edge_count_by_relation": relation_edge_counts,
            "min_shelf_vote": args.min_shelf_vote,
            "max_shelves_per_book": args.max_shelves_per_book,
            "generic_shelf_filter_enabled": not args.disable_generic_shelf_filter,
            "similar_mode": args.similar_mode,
        },
        "ingest_stats": ingest_stats,
        "split_stats": split_stats,
        "kg_stats": {
            "entities": entity_count,
            "relations": relation_count,
            "kg_triples": kg_triples,
            "users": len(user_map),
            "items": len(item_map),
        },
        "paths": {
            "all_positive_interactions": str(output_dir / "all_positive_interactions.txt"),
            "train": str(output_dir / "train.txt"),
            "test": str(output_dir / "test.txt"),
            "user_list": str(output_dir / "user_list.txt"),
            "item_list": str(output_dir / "item_list.txt"),
            "entity_list": str(output_dir / "entity_list.txt"),
            "relation_list": str(output_dir / "relation_list.txt"),
            "kg_final": str(output_dir / "kg_final.txt"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_summary(output_dir / "summary.json", summary)

    print("Done")
    print(f"output_dir={output_dir}")
    print(f"users={len(user_map):,} items={len(item_map):,}")
    print(f"entities={entity_count:,} relations={relation_count:,} kg_triples={kg_triples:,}")
    print("relation profile:", args.relation_profile, "=>", ", ".join(RELATION_PROFILES[args.relation_profile]))


if __name__ == "__main__":
    main()
