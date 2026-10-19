from __future__ import annotations

import argparse
import ast
import csv
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_FILES = {
    "review": "yelp_academic_dataset_review.json",
    "business": "yelp_academic_dataset_business.json",
    "user": "yelp_academic_dataset_user.json",
    "tip": "yelp_academic_dataset_tip.json",
    "checkin": "yelp_academic_dataset_checkin.json",
}


@dataclass
class ReviewExportStats:
    kept_reviews: int = 0
    skipped_low_stars: int = 0
    skipped_missing_user: int = 0
    skipped_missing_business: int = 0


class ProgressBar:
    def __init__(self, label: str, path: Path, *, width: int = 28) -> None:
        self.label = label
        self.path = path
        self.width = width
        self.total_bytes = max(path.stat().st_size, 1)
        self.current_bytes = 0
        self.extra = ""
        self.start_time = time.monotonic()
        self.last_render_time = 0.0

    def set_extra(self, extra: str) -> None:
        self.extra = extra

    def advance(self, byte_count: int) -> None:
        self.current_bytes += byte_count
        self._render(force=False)

    def close(self, extra: str | None = None) -> None:
        if extra is not None:
            self.extra = extra
        self.current_bytes = self.total_bytes
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self.last_render_time < 0.2:
            return
        self.last_render_time = now

        ratio = min(1.0, self.current_bytes / self.total_bytes)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(now - self.start_time, 1e-6)
        mb_per_sec = (self.current_bytes / 1024 / 1024) / elapsed
        message = (
            f"\r[{self.label}] [{bar}] {ratio * 100:6.2f}% "
            f"{self.current_bytes / 1024 / 1024:9.1f}/{self.total_bytes / 1024 / 1024:9.1f} MB "
            f"{mb_per_sec:6.2f} MB/s"
        )
        if self.extra:
            message += f" | {self.extra}"
        sys.stdout.write(message)
        sys.stdout.flush()


class CountProgressBar:
    def __init__(self, label: str, total: int, *, width: int = 28) -> None:
        self.label = label
        self.total = max(total, 1)
        self.width = width
        self.current = 0
        self.extra = ""
        self.start_time = time.monotonic()
        self.last_render_time = 0.0

    def set_extra(self, extra: str) -> None:
        self.extra = extra

    def advance(self, count: int = 1) -> None:
        self.current += count
        self._render(force=False)

    def close(self, extra: str | None = None) -> None:
        if extra is not None:
            self.extra = extra
        self.current = self.total
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self.last_render_time < 0.2:
            return
        self.last_render_time = now
        ratio = min(1.0, self.current / self.total)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(now - self.start_time, 1e-6)
        rows_per_sec = self.current / elapsed
        message = f"\r[{self.label}] [{bar}] {ratio * 100:6.2f}% {self.current:,}/{self.total:,} rows {rows_per_sec:8.1f} rows/s"
        if self.extra:
            message += f" | {self.extra}"
        sys.stdout.write(message)
        sys.stdout.flush()


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


class RelationCatalog:
    def __init__(self) -> None:
        self.name_to_id: dict[str, int] = {}
        self.rows: list[dict[str, str | int]] = []

    def register(self, name: str, head_type: str, tail_type: str, description: str) -> int:
        if name in self.name_to_id:
            return self.name_to_id[name]
        relation_id = len(self.rows)
        self.name_to_id[name] = relation_id
        self.rows.append(
            {
                "relation_id": relation_id,
                "relation_name": name,
                "head_type": head_type,
                "tail_type": tail_type,
                "description": description,
            }
        )
        return relation_id


class EntityCatalog:
    def __init__(self, item_map: dict[str, int]) -> None:
        self.lookup: dict[tuple[str, str], int] = {}
        self.rows_by_id: dict[int, dict[str, str | int]] = {}
        self.next_id = 0
        for business_id, entity_id in sorted(item_map.items(), key=lambda pair: pair[1]):
            self.lookup[("business", business_id)] = entity_id
            self.rows_by_id[entity_id] = {
                "entity_id": entity_id,
                "entity_type": "business",
                "source_field": "business_id",
                "raw_value": business_id,
                "label": business_id,
                "metadata_json": "",
            }
            self.next_id = max(self.next_id, entity_id + 1)

    def ensure_entity(
        self,
        entity_type: str,
        raw_value: str,
        *,
        source_field: str,
        label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        key = (entity_type, raw_value)
        if key not in self.lookup:
            entity_id = self.next_id
            self.next_id += 1
            self.lookup[key] = entity_id
            self.rows_by_id[entity_id] = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "source_field": source_field,
                "raw_value": raw_value,
                "label": label or raw_value,
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=False) if metadata else "",
            }
        else:
            entity_id = self.lookup[key]
            if label:
                self.rows_by_id[entity_id]["label"] = label
            if metadata:
                self.rows_by_id[entity_id]["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        return entity_id

    def update_business_metadata(self, business_id: str, *, label: str, metadata: dict[str, object]) -> None:
        entity_id = self.lookup[("business", business_id)]
        self.rows_by_id[entity_id]["label"] = label
        self.rows_by_id[entity_id]["metadata_json"] = json.dumps(metadata, ensure_ascii=False)


class KgWriter:
    def __init__(self, kg_path: Path, readable_path: Path) -> None:
        kg_path.parent.mkdir(parents=True, exist_ok=True)
        self.kg_file = kg_path.open("w", encoding="utf-8", newline="")
        self.readable_file = readable_path.open("w", encoding="utf-8", newline="")
        self.readable_writer = csv.writer(self.readable_file, delimiter="\t")
        self.readable_writer.writerow(
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
        self.edge_count = 0

    def write(
        self,
        *,
        head_id: int,
        head_type: str,
        head_raw: str,
        relation_id: int,
        relation_name: str,
        tail_id: int,
        tail_type: str,
        tail_raw: str,
        source: str,
    ) -> None:
        self.kg_file.write(f"{head_id} {relation_id} {tail_id}\n")
        self.readable_writer.writerow(
            [head_id, head_type, head_raw, relation_id, relation_name, tail_id, tail_type, tail_raw, source]
        )
        self.edge_count += 1

    def close(self) -> None:
        self.kg_file.close()
        self.readable_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and align the Yelp JSON dataset for recommendation/KG/LLM experiments."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "01_raw_data" / "raw_json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "02_processed_data" / "full_datasets" / "aligned_yelp",
    )
    parser.add_argument("--review-file", type=Path, default=Path(DEFAULT_FILES["review"]))
    parser.add_argument("--business-file", type=Path, default=Path(DEFAULT_FILES["business"]))
    parser.add_argument("--user-file", type=Path, default=Path(DEFAULT_FILES["user"]))
    parser.add_argument("--tip-file", type=Path, default=Path(DEFAULT_FILES["tip"]))
    parser.add_argument("--checkin-file", type=Path, default=Path(DEFAULT_FILES["checkin"]))
    parser.add_argument(
        "--min-stars",
        type=float,
        default=4.0,
        help="Only keep reviews with stars >= this threshold. Default 4.0 means positive interactions only.",
    )
    parser.add_argument("--min-user-reviews", type=int, default=2)
    parser.add_argument("--min-business-reviews", type=int, default=2)
    parser.add_argument("--filter-rounds", type=int, default=2)
    parser.add_argument("--require-user-record", action="store_true")
    parser.add_argument(
        "--split-strategy",
        choices=("leave_one_out", "ratio"),
        default="ratio",
        help="Default is ratio, which creates an 8/2 split with at least one test interaction per user.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--split-shards", type=int, default=512)
    parser.add_argument(
        "--export-shard-size",
        type=int,
        default=250_000,
        help="Max records per exported JSONL shard. Use 0 to keep one file per export.",
    )
    parser.add_argument("--count-flush-size", type=int, default=200_000)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path, progress: ProgressBar | None = None) -> Iterable[dict[str, object]]:
    with path.open("rb") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            if progress is not None:
                progress.advance(len(raw_line))
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def configure_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")


def create_review_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS latest_reviews")
    conn.execute("DROP TABLE IF EXISTS user_counts")
    conn.execute("DROP TABLE IF EXISTS business_counts")
    conn.execute(
        """
        CREATE TABLE latest_reviews (
            user_id TEXT NOT NULL,
            business_id TEXT NOT NULL,
            review_id TEXT NOT NULL,
            review_date TEXT NOT NULL,
            stars REAL NOT NULL,
            offset INTEGER NOT NULL,
            PRIMARY KEY (user_id, business_id)
        )
        """
    )
    conn.execute("CREATE TABLE user_counts (raw_id TEXT PRIMARY KEY, cnt INTEGER NOT NULL)")
    conn.execute("CREATE TABLE business_counts (raw_id TEXT PRIMARY KEY, cnt INTEGER NOT NULL)")
    conn.commit()


def flush_counter(conn: sqlite3.Connection, table_name: str, counter: Counter[str]) -> None:
    if not counter:
        return
    conn.executemany(
        f"""
        INSERT INTO {table_name} (raw_id, cnt)
        VALUES (?, ?)
        ON CONFLICT(raw_id) DO UPDATE SET cnt = cnt + excluded.cnt
        """,
        counter.items(),
    )
    conn.commit()
    counter.clear()


def flush_latest_reviews(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, str, float, int]],
) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO latest_reviews (user_id, business_id, review_id, review_date, stars, offset)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, business_id) DO UPDATE SET
            review_id = excluded.review_id,
            review_date = excluded.review_date,
            stars = excluded.stars,
            offset = excluded.offset
        WHERE excluded.review_date > latest_reviews.review_date
           OR (
                excluded.review_date = latest_reviews.review_date
                AND excluded.review_id > latest_reviews.review_id
           )
        """,
        rows,
    )
    conn.commit()
    rows.clear()


def build_latest_review_table(
    review_path: Path,
    db_path: Path,
    *,
    flush_size: int,
    progress_every: int,
) -> tuple[int, int]:
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    configure_sqlite(conn)
    create_review_tables(conn)
    pending_rows: list[tuple[str, str, str, str, float, int]] = []
    scanned = 0
    progress = ProgressBar("latest", review_path)

    with review_path.open("rb") as file_obj:
        while True:
            offset = file_obj.tell()
            raw_line = file_obj.readline()
            if not raw_line:
                break
            progress.advance(len(raw_line))
            line = raw_line.strip()
            if not line:
                continue

            scanned += 1
            if scanned % progress_every == 0:
                progress.set_extra(f"scanned={scanned:,} pending={len(pending_rows):,}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {review_path}:{scanned}") from exc

            pending_rows.append(
                (
                    str(record["user_id"]),
                    str(record["business_id"]),
                    str(record.get("review_id", "")),
                    str(record.get("date", "")),
                    float(record.get("stars", 0.0)),
                    offset,
                )
            )
            if len(pending_rows) >= flush_size:
                flush_latest_reviews(conn, pending_rows)

    flush_latest_reviews(conn, pending_rows)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_latest_reviews_business ON latest_reviews (business_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_latest_reviews_offset ON latest_reviews (offset)")
    conn.commit()
    unique_pairs = conn.execute("SELECT COUNT(*) FROM latest_reviews").fetchone()[0]
    conn.close()
    progress.close(f"scanned={scanned:,} latest_pairs={unique_pairs:,}")
    return scanned, unique_pairs


def count_positive_latest_reviews(db_path: Path, min_stars: float) -> int:
    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM latest_reviews WHERE stars >= ?",
        (min_stars,),
    ).fetchone()[0]
    conn.close()
    return count


def load_active_count_map(
    db_path: Path,
    table_name: str,
    threshold: int,
    *,
    group_field: str | None = None,
    min_stars: float | None = None,
) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    if group_field is not None and min_stars is not None:
        conn.execute(f"DELETE FROM {table_name}")
        conn.execute(
            f"""
            INSERT INTO {table_name} (raw_id, cnt)
            SELECT {group_field} AS raw_id, COUNT(*) AS cnt
            FROM latest_reviews
            WHERE stars >= ?
            GROUP BY {group_field}
            """,
            (min_stars,),
        )
        conn.commit()
    cursor = conn.execute(
        f"SELECT raw_id, cnt FROM {table_name} WHERE cnt >= ? ORDER BY raw_id",
        (threshold,),
    )
    result = {raw_id: cnt for raw_id, cnt in cursor.fetchall()}
    conn.close()
    return result


def build_id_map(raw_ids: Iterable[str]) -> dict[str, int]:
    return {raw_id: idx for idx, raw_id in enumerate(sorted(raw_ids))}


def collect_present_ids(
    source_path: Path,
    key_name: str,
    candidate_ids: set[str],
    *,
    progress_every: int,
    label: str,
) -> set[str]:
    present_ids: set[str] = set()
    scanned = 0
    progress = ProgressBar(label, source_path)
    for record in iter_jsonl(source_path, progress):
        scanned += 1
        if scanned % progress_every == 0:
            progress.set_extra(f"scanned={scanned:,} matched={len(present_ids):,}")
        raw_id = str(record[key_name])
        if raw_id in candidate_ids:
            present_ids.add(raw_id)
    progress.close(f"scanned={scanned:,} matched={len(present_ids):,}")
    return present_ids


def refine_active_sets(
    db_path: Path,
    *,
    min_stars: float,
    min_user_reviews: int,
    min_business_reviews: int,
    filter_rounds: int,
    user_counts: dict[str, int],
    business_counts: dict[str, int],
    progress_every: int,
) -> tuple[dict[str, int], dict[str, int], list[dict[str, int]]]:
    conn = sqlite3.connect(db_path)
    positive_latest_total = count_positive_latest_reviews(db_path, min_stars)
    active_users = set(user_counts)
    active_businesses = set(business_counts)
    round_stats: list[dict[str, int]] = []
    final_user_counts = user_counts
    final_business_counts = business_counts

    for round_idx in range(filter_rounds):
        kept_reviews = 0
        next_user_counts: Counter[str] = Counter()
        next_business_counts: Counter[str] = Counter()
        scanned = 0
        progress = CountProgressBar(f"refine:{round_idx + 1}", positive_latest_total)

        cursor = conn.execute(
            """
            SELECT user_id, business_id
            FROM latest_reviews
            WHERE stars >= ?
            """,
            (min_stars,),
        )
        for user_id, business_id in cursor:
            scanned += 1
            progress.advance()
            if scanned % progress_every == 0:
                progress.set_extra(f"scanned={scanned:,} kept={kept_reviews:,}")
            if user_id not in active_users or business_id not in active_businesses:
                continue

            next_user_counts[user_id] += 1
            next_business_counts[business_id] += 1
            kept_reviews += 1

        filtered_user_counts = {
            user_id: cnt for user_id, cnt in next_user_counts.items() if cnt >= min_user_reviews
        }
        filtered_business_counts = {
            business_id: cnt
            for business_id, cnt in next_business_counts.items()
            if cnt >= min_business_reviews
        }

        round_stats.append(
            {
                "round": round_idx + 1,
                "kept_reviews": kept_reviews,
                "active_users": len(filtered_user_counts),
                "active_businesses": len(filtered_business_counts),
            }
        )

        same_users = set(filtered_user_counts) == active_users
        same_businesses = set(filtered_business_counts) == active_businesses
        final_user_counts = filtered_user_counts
        final_business_counts = filtered_business_counts
        active_users = set(filtered_user_counts)
        active_businesses = set(filtered_business_counts)
        progress.close(
            f"scanned={scanned:,} kept={kept_reviews:,} users={len(filtered_user_counts):,} businesses={len(filtered_business_counts):,}"
        )

        if same_users and same_businesses:
            break

    conn.close()
    return final_user_counts, final_business_counts, round_stats


def write_mapping_table(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def split_categories(value: object) -> list[str]:
    if not value or not isinstance(value, str):
        return []
    return sorted({part.strip() for part in value.split(",") if part.strip()})


def try_parse_nested(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value
    return value


def flatten_attributes(attributes: object) -> list[tuple[str, str]]:
    flattened: list[tuple[str, str]] = []

    def _walk(prefix: str, value: object) -> None:
        parsed = try_parse_nested(value)
        if isinstance(parsed, dict):
            for key, child in parsed.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(next_prefix, child)
            return
        if parsed in (None, "", "None"):
            return
        normalized = str(parsed).strip()
        if normalized:
            flattened.append((prefix, normalized))

    if isinstance(attributes, dict):
        for key, value in attributes.items():
            _walk(str(key), value)

    return sorted(set(flattened))


def hour_bucket(hour_value: int) -> str:
    if 0 <= hour_value <= 5:
        return "00_05"
    if 6 <= hour_value <= 11:
        return "06_11"
    if 12 <= hour_value <= 17:
        return "12_17"
    return "18_23"


def parse_checkin_features(date_value: object) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(date_value, str) or not date_value.strip():
        return [], [], []

    weekdays: set[str] = set()
    months: set[str] = set()
    hour_buckets: set[str] = set()

    for raw_timestamp in date_value.split(","):
        raw_timestamp = raw_timestamp.strip()
        if not raw_timestamp:
            continue
        try:
            dt = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        weekdays.add(dt.strftime("%A"))
        months.add(dt.strftime("%Y-%m"))
        hour_buckets.add(hour_bucket(dt.hour))

    return sorted(weekdays), sorted(months), sorted(hour_buckets)


def sort_review_key(record: dict[str, object]) -> tuple[str, str]:
    return (str(record.get("date", "")), str(record.get("review_id", "")))


def export_aligned_reviews(
    db_path: Path,
    review_path: Path,
    output_dir: Path,
    shard_dir: Path,
    *,
    user_map: dict[str, int],
    item_map: dict[str, int],
    min_stars: float,
    split_shards: int,
    export_shard_size: int,
    progress_every: int,
) -> ReviewExportStats:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_handles: list[object | None] = [None for _ in range(split_shards)]
    stats = ReviewExportStats()
    conn = sqlite3.connect(db_path)
    total_rows = count_positive_latest_reviews(db_path, min_stars)
    scanned = 0
    progress = CountProgressBar("reviews", total_rows)
    writer = JsonlShardWriter(output_dir, "aligned_reviews", export_shard_size)

    with review_path.open("rb") as review_file:
        cursor = conn.execute(
            """
            SELECT user_id, business_id, review_id, review_date, stars, offset
            FROM latest_reviews
            WHERE stars >= ?
            ORDER BY offset
            """,
            (min_stars,),
        )
        for user_id, business_id, review_id, review_date, stars, offset in cursor:
            scanned += 1
            progress.advance()
            if scanned % progress_every == 0:
                progress.set_extra(f"scanned={scanned:,} kept={stats.kept_reviews:,}")

            if user_id not in user_map:
                stats.skipped_missing_user += 1
                continue
            if business_id not in item_map:
                stats.skipped_missing_business += 1
                continue

            review_file.seek(offset)
            raw_line = review_file.readline()
            if not raw_line:
                raise ValueError(f"Missing review line at offset {offset} in {review_path}")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at offset {offset} in {review_path}") from exc

            user_idx = user_map[user_id]
            item_idx = item_map[business_id]
            enriched = {
                "review_id": record["review_id"],
                "raw_user_id": user_id,
                "user_idx": user_idx,
                "raw_business_id": business_id,
                "item_idx": item_idx,
                "entity_id": item_idx,
                "stars": stars,
                "useful": record.get("useful"),
                "funny": record.get("funny"),
                "cool": record.get("cool"),
                "date": review_date,
                "text": record.get("text", ""),
            }
            writer.write(enriched)

            shard_record = {
                "review_id": review_id,
                "user_idx": user_idx,
                "item_idx": item_idx,
                "raw_user_id": user_id,
                "raw_business_id": business_id,
                "date": review_date,
                "stars": stars,
            }
            shard_index = user_idx % split_shards
            if shard_handles[shard_index] is None:
                shard_handles[shard_index] = (
                    shard_dir / f"reviews_shard_{shard_index:03d}.jsonl"
                ).open("w", encoding="utf-8")
            shard_handles[shard_index].write(json.dumps(shard_record, ensure_ascii=False) + "\n")
            stats.kept_reviews += 1

    writer.close()
    progress.close(
        f"scanned={scanned:,} kept={stats.kept_reviews:,} parts={writer.total_parts:,}"
    )

    for handle in shard_handles:
        if handle is not None:
            handle.close()

    conn.close()
    return stats


def export_aligned_users(
    user_path: Path,
    output_dir: Path,
    *,
    user_map: dict[str, int],
    export_shard_size: int,
    progress_every: int,
) -> int:
    kept = 0
    scanned = 0
    progress = ProgressBar("users", user_path)
    writer = JsonlShardWriter(output_dir, "aligned_users", export_shard_size)
    for record in iter_jsonl(user_path, progress):
        scanned += 1
        if scanned % progress_every == 0:
            progress.set_extra(f"scanned={scanned:,} kept={kept:,}")

        user_id = str(record["user_id"])
        if user_id not in user_map:
            continue

        enriched = dict(record)
        enriched["raw_user_id"] = user_id
        enriched["user_idx"] = user_map[user_id]
        writer.write(enriched)
        kept += 1
    writer.close()
    progress.close(f"scanned={scanned:,} kept={kept:,} parts={writer.total_parts:,}")
    return kept


def export_aligned_tips(
    tip_path: Path,
    output_dir: Path,
    *,
    user_map: dict[str, int],
    item_map: dict[str, int],
    export_shard_size: int,
    progress_every: int,
) -> int:
    kept = 0
    scanned = 0
    progress = ProgressBar("tips", tip_path)
    writer = JsonlShardWriter(output_dir, "aligned_tips", export_shard_size)
    for record in iter_jsonl(tip_path, progress):
        scanned += 1
        if scanned % progress_every == 0:
            progress.set_extra(f"scanned={scanned:,} kept={kept:,}")

        business_id = str(record["business_id"])
        if business_id not in item_map:
            continue

        user_id = str(record["user_id"])
        enriched = {
            "tip_id": f"{user_id}::{business_id}::{record.get('date', '')}",
            "raw_user_id": user_id,
            "user_idx": user_map.get(user_id),
            "raw_business_id": business_id,
            "item_idx": item_map[business_id],
            "entity_id": item_map[business_id],
            "date": record.get("date"),
            "compliment_count": record.get("compliment_count"),
            "text": record.get("text", ""),
        }
        writer.write(enriched)
        kept += 1
    writer.close()
    progress.close(f"scanned={scanned:,} kept={kept:,} parts={writer.total_parts:,}")
    return kept


def export_businesses_and_kg(
    business_path: Path,
    output_dir: Path,
    *,
    item_map: dict[str, int],
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    kg_writer: KgWriter,
    export_shard_size: int,
    progress_every: int,
) -> int:
    kept = 0
    scanned = 0
    city_state_edges_seen: set[tuple[int, int, int]] = set()

    category_rel = relation_catalog.register(
        "has_category", "business", "category", "Business belongs to a Yelp category."
    )
    city_rel = relation_catalog.register("located_in_city", "business", "city", "Business is located in a city.")
    state_rel = relation_catalog.register(
        "located_in_state", "business", "state", "Business is located in a state or region."
    )
    postal_rel = relation_catalog.register(
        "located_in_postal_code", "business", "postal_code", "Business is located in a postal code."
    )
    open_status_rel = relation_catalog.register(
        "has_open_status", "business", "open_status", "Business current open/closed status from Yelp."
    )
    attr_rel = relation_catalog.register(
        "has_attribute", "business", "attribute", "Business has a structured Yelp attribute value."
    )
    open_day_rel = relation_catalog.register(
        "open_on_day", "business", "day_of_week", "Business publishes opening hours for a day."
    )
    hour_rel = relation_catalog.register(
        "has_business_hours", "business", "business_hours", "Business opening-hours slot for a day."
    )
    city_to_state_rel = relation_catalog.register(
        "city_in_state", "city", "state", "City belongs to a state according to business metadata."
    )

    progress = ProgressBar("business", business_path)
    writer = JsonlShardWriter(output_dir, "aligned_businesses", export_shard_size)
    for record in iter_jsonl(business_path, progress):
        scanned += 1
        if scanned % progress_every == 0:
            progress.set_extra(f"scanned={scanned:,} kept={kept:,}")

        business_id = str(record["business_id"])
        if business_id not in item_map:
            continue

        entity_id = item_map[business_id]
        metadata = {
            "name": record.get("name"),
            "city": record.get("city"),
            "state": record.get("state"),
            "postal_code": record.get("postal_code"),
            "stars": record.get("stars"),
            "review_count": record.get("review_count"),
        }
        entity_catalog.update_business_metadata(
            business_id,
            label=str(record.get("name") or business_id),
            metadata=metadata,
        )

        enriched = dict(record)
        enriched["raw_business_id"] = business_id
        enriched["item_idx"] = entity_id
        enriched["entity_id"] = entity_id
        writer.write(enriched)
        kept += 1

        for category in split_categories(record.get("categories")):
            tail_id = entity_catalog.ensure_entity(
                "category",
                category,
                source_field="categories",
                label=category,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=category_rel,
                relation_name="has_category",
                tail_id=tail_id,
                tail_type="category",
                tail_raw=category,
                source="business.categories",
            )

        city = str(record.get("city") or "").strip()
        state = str(record.get("state") or "").strip()
        postal_code = str(record.get("postal_code") or "").strip()
        city_id = None
        state_id = None

        if city:
            city_id = entity_catalog.ensure_entity("city", city, source_field="city", label=city)
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=city_rel,
                relation_name="located_in_city",
                tail_id=city_id,
                tail_type="city",
                tail_raw=city,
                source="business.city",
            )

        if state:
            state_id = entity_catalog.ensure_entity("state", state, source_field="state", label=state)
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=state_rel,
                relation_name="located_in_state",
                tail_id=state_id,
                tail_type="state",
                tail_raw=state,
                source="business.state",
            )

        if city_id is not None and state_id is not None:
            bridge_key = (city_id, city_to_state_rel, state_id)
            if bridge_key not in city_state_edges_seen:
                city_state_edges_seen.add(bridge_key)
                kg_writer.write(
                    head_id=city_id,
                    head_type="city",
                    head_raw=city,
                    relation_id=city_to_state_rel,
                    relation_name="city_in_state",
                    tail_id=state_id,
                    tail_type="state",
                    tail_raw=state,
                    source="business.city+state",
                )

        if postal_code:
            postal_id = entity_catalog.ensure_entity(
                "postal_code",
                postal_code,
                source_field="postal_code",
                label=postal_code,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=postal_rel,
                relation_name="located_in_postal_code",
                tail_id=postal_id,
                tail_type="postal_code",
                tail_raw=postal_code,
                source="business.postal_code",
            )

        open_status = "open" if int(record.get("is_open", 0)) == 1 else "closed"
        status_id = entity_catalog.ensure_entity(
            "open_status",
            open_status,
            source_field="is_open",
            label=open_status,
        )
        kg_writer.write(
            head_id=entity_id,
            head_type="business",
            head_raw=business_id,
            relation_id=open_status_rel,
            relation_name="has_open_status",
            tail_id=status_id,
            tail_type="open_status",
            tail_raw=open_status,
            source="business.is_open",
        )

        for attr_key, attr_value in flatten_attributes(record.get("attributes")):
            attr_raw = f"{attr_key}={attr_value}"
            attr_id = entity_catalog.ensure_entity(
                "attribute",
                attr_raw,
                source_field="attributes",
                label=attr_raw,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=attr_rel,
                relation_name="has_attribute",
                tail_id=attr_id,
                tail_type="attribute",
                tail_raw=attr_raw,
                source="business.attributes",
            )

        hours = record.get("hours")
        if isinstance(hours, dict):
            for day_name, hour_span in sorted(hours.items()):
                if not hour_span:
                    continue
                day_id = entity_catalog.ensure_entity(
                    "day_of_week",
                    str(day_name),
                    source_field="hours.day",
                    label=str(day_name),
                )
                kg_writer.write(
                    head_id=entity_id,
                    head_type="business",
                    head_raw=business_id,
                    relation_id=open_day_rel,
                    relation_name="open_on_day",
                    tail_id=day_id,
                    tail_type="day_of_week",
                    tail_raw=str(day_name),
                    source="business.hours",
                )

                span_raw = f"{day_name}={hour_span}"
                span_id = entity_catalog.ensure_entity(
                    "business_hours",
                    span_raw,
                    source_field="hours.span",
                    label=span_raw,
                )
                kg_writer.write(
                    head_id=entity_id,
                    head_type="business",
                    head_raw=business_id,
                    relation_id=hour_rel,
                    relation_name="has_business_hours",
                    tail_id=span_id,
                    tail_type="business_hours",
                    tail_raw=span_raw,
                    source="business.hours",
                )

    writer.close()
    progress.close(f"scanned={scanned:,} kept={kept:,} parts={writer.total_parts:,}")
    return kept


def export_checkins_and_kg(
    checkin_path: Path,
    output_dir: Path,
    *,
    item_map: dict[str, int],
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
    kg_writer: KgWriter,
    export_shard_size: int,
    progress_every: int,
) -> int:
    kept = 0
    scanned = 0

    weekday_rel = relation_catalog.register(
        "checkin_on_weekday", "business", "day_of_week", "Business has observed check-ins on a weekday."
    )
    month_rel = relation_catalog.register(
        "checkin_in_month", "business", "year_month", "Business has observed check-ins in a year-month."
    )
    hour_rel = relation_catalog.register(
        "checkin_in_hour_bucket",
        "business",
        "hour_bucket",
        "Business has observed check-ins in a coarse hour bucket.",
    )

    progress = ProgressBar("checkin", checkin_path)
    writer = JsonlShardWriter(output_dir, "aligned_checkins", export_shard_size)
    for record in iter_jsonl(checkin_path, progress):
        scanned += 1
        if scanned % progress_every == 0:
            progress.set_extra(f"scanned={scanned:,} kept={kept:,}")

        business_id = str(record["business_id"])
        if business_id not in item_map:
            continue

        weekdays, months, hour_buckets = parse_checkin_features(record.get("date"))
        enriched = {
            "raw_business_id": business_id,
            "item_idx": item_map[business_id],
            "entity_id": item_map[business_id],
            "date": record.get("date", ""),
            "weekday_buckets": weekdays,
            "month_buckets": months,
            "hour_buckets": hour_buckets,
        }
        writer.write(enriched)
        kept += 1

        entity_id = item_map[business_id]
        for weekday in weekdays:
            weekday_id = entity_catalog.ensure_entity(
                "day_of_week",
                weekday,
                source_field="checkin.weekday",
                label=weekday,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=weekday_rel,
                relation_name="checkin_on_weekday",
                tail_id=weekday_id,
                tail_type="day_of_week",
                tail_raw=weekday,
                source="checkin.date",
            )

        for month in months:
            month_id = entity_catalog.ensure_entity(
                "year_month",
                month,
                source_field="checkin.month",
                label=month,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=month_rel,
                relation_name="checkin_in_month",
                tail_id=month_id,
                tail_type="year_month",
                tail_raw=month,
                source="checkin.date",
            )

        for bucket in hour_buckets:
            bucket_id = entity_catalog.ensure_entity(
                "hour_bucket",
                bucket,
                source_field="checkin.hour_bucket",
                label=bucket,
            )
            kg_writer.write(
                head_id=entity_id,
                head_type="business",
                head_raw=business_id,
                relation_id=hour_rel,
                relation_name="checkin_in_hour_bucket",
                tail_id=bucket_id,
                tail_type="hour_bucket",
                tail_raw=bucket,
                source="checkin.date",
            )

    writer.close()
    progress.close(f"scanned={scanned:,} kept={kept:,} parts={writer.total_parts:,}")
    return kept


def dedupe_user_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    latest_by_item: dict[int, dict[str, object]] = {}
    for record in sorted(records, key=sort_review_key):
        latest_by_item[int(record["item_idx"])] = record
    return sorted(latest_by_item.values(), key=sort_review_key)


def split_records(records: list[dict[str, object]], strategy: str, test_ratio: float) -> tuple[list[int], list[int]]:
    if len(records) <= 1:
        return [int(record["item_idx"]) for record in records], []

    if strategy == "leave_one_out":
        train_records = records[:-1]
        test_records = records[-1:]
    else:
        raw_test_count = max(1, int(round(len(records) * test_ratio)))
        test_count = min(raw_test_count, len(records) - 1)
        train_records = records[:-test_count]
        test_records = records[-test_count:]

    return [int(record["item_idx"]) for record in train_records], [int(record["item_idx"]) for record in test_records]


def build_train_test_from_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    strategy: str,
    test_ratio: float,
) -> dict[str, int]:
    interaction_dir = output_dir / "interactions"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    all_positive_path = interaction_dir / "all_positive.txt"
    train_path = interaction_dir / "train.txt"
    test_path = interaction_dir / "test.txt"
    positive_detail_path = interaction_dir / "positive_interactions.tsv"
    split_path = interaction_dir / "review_split.tsv"

    stats = {
        "users_in_all_positive": 0,
        "all_positive_interactions": 0,
        "split_skipped_users_lt2_positive_items": 0,
        "users_in_train": 0,
        "users_in_test": 0,
        "train_interactions": 0,
        "test_interactions": 0,
    }

    with all_positive_path.open("w", encoding="utf-8") as all_positive_file, train_path.open(
        "w", encoding="utf-8"
    ) as train_file, test_path.open(
        "w", encoding="utf-8"
    ) as test_file, positive_detail_path.open(
        "w", encoding="utf-8", newline=""
    ) as positive_detail_file, split_path.open("w", encoding="utf-8", newline="") as split_file:
        positive_detail_writer = csv.writer(positive_detail_file, delimiter="\t")
        positive_detail_writer.writerow(
            [
                "review_id",
                "user_idx",
                "item_idx",
                "raw_user_id",
                "raw_business_id",
                "date",
                "stars",
            ]
        )
        split_writer = csv.writer(split_file, delimiter="\t")
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
            ]
        )

        shard_paths = [path for path in sorted(shard_dir.glob("reviews_shard_*.jsonl")) if path.stat().st_size > 0]
        for shard_path in shard_paths:
            progress = ProgressBar(f"split:{shard_path.stem}", shard_path)
            grouped_records: dict[int, list[dict[str, object]]] = defaultdict(list)
            loaded = 0
            for record in iter_jsonl(shard_path, progress):
                grouped_records[int(record["user_idx"])].append(record)
                loaded += 1
                if loaded % 100_000 == 0:
                    progress.set_extra(f"loaded={loaded:,} users={len(grouped_records):,}")

            for user_idx, raw_records in grouped_records.items():
                records = dedupe_user_records(raw_records)
                all_items = [int(record["item_idx"]) for record in records]
                if all_items:
                    all_positive_file.write(f"{user_idx} {' '.join(map(str, all_items))}\n")
                    stats["users_in_all_positive"] += 1
                    stats["all_positive_interactions"] += len(all_items)
                for record in records:
                    positive_detail_writer.writerow(
                        [
                            record.get("review_id", ""),
                            user_idx,
                            int(record["item_idx"]),
                            record.get("raw_user_id", ""),
                            record.get("raw_business_id", ""),
                            record.get("date", ""),
                            record.get("stars", ""),
                        ]
                    )
                if len(records) < 2:
                    stats["split_skipped_users_lt2_positive_items"] += 1
                    continue
                train_items, test_items = split_records(records, strategy, test_ratio)
                if train_items:
                    train_file.write(f"{user_idx} {' '.join(map(str, train_items))}\n")
                    stats["users_in_train"] += 1
                    stats["train_interactions"] += len(train_items)
                if test_items:
                    test_file.write(f"{user_idx} {' '.join(map(str, test_items))}\n")
                    stats["users_in_test"] += 1
                    stats["test_interactions"] += len(test_items)

                test_item_set = set(test_items)
                for record in records:
                    split_name = "test" if int(record["item_idx"]) in test_item_set else "train"
                    split_writer.writerow(
                        [
                            record.get("review_id", ""),
                            split_name,
                            user_idx,
                            int(record["item_idx"]),
                            record.get("raw_user_id", ""),
                            record.get("raw_business_id", ""),
                            record.get("date", ""),
                            record.get("stars", ""),
                        ]
                    )
            progress.close(f"loaded={loaded:,} users={len(grouped_records):,}")

    return stats


def write_entity_and_relation_tables(
    output_dir: Path,
    *,
    user_map: dict[str, int],
    user_counts: dict[str, int],
    item_map: dict[str, int],
    business_counts: dict[str, int],
    entity_catalog: EntityCatalog,
    relation_catalog: RelationCatalog,
) -> None:
    mapping_dir = output_dir / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    write_mapping_table(
        mapping_dir / "user_map.tsv",
        ["user_idx", "raw_user_id", "kept_review_count"],
        (
            [user_idx, raw_user_id, user_counts.get(raw_user_id, 0)]
            for raw_user_id, user_idx in sorted(user_map.items(), key=lambda pair: pair[1])
        ),
    )

    write_mapping_table(
        mapping_dir / "item_map.tsv",
        ["item_idx", "entity_id", "raw_business_id", "kept_review_count"],
        (
            [item_idx, item_idx, raw_business_id, business_counts.get(raw_business_id, 0)]
            for raw_business_id, item_idx in sorted(item_map.items(), key=lambda pair: pair[1])
        ),
    )

    write_mapping_table(
        mapping_dir / "entity_map.tsv",
        ["entity_id", "entity_type", "source_field", "raw_value", "label", "metadata_json"],
        (
            [
                row["entity_id"],
                row["entity_type"],
                row["source_field"],
                row["raw_value"],
                row["label"],
                row["metadata_json"],
            ]
            for _, row in sorted(entity_catalog.rows_by_id.items())
        ),
    )

    write_mapping_table(
        mapping_dir / "relation_map.tsv",
        ["relation_id", "relation_name", "head_type", "tail_type", "description"],
        (
            [
                row["relation_id"],
                row["relation_name"],
                row["head_type"],
                row["tail_type"],
                row["description"],
            ]
            for row in relation_catalog.rows
        ),
    )


def write_stats(path: Path, payload: dict[str, object]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = resolve_path(base_dir, args.review_file)
    business_path = resolve_path(base_dir, args.business_file)
    user_path = resolve_path(base_dir, args.user_file)
    tip_path = resolve_path(base_dir, args.tip_file)
    checkin_path = resolve_path(base_dir, args.checkin_file)

    db_path = output_dir / "work" / "review_counts.sqlite"
    shard_dir = output_dir / "work" / "review_shards"

    scanned_reviews, latest_review_pairs = build_latest_review_table(
        review_path,
        db_path,
        flush_size=args.count_flush_size,
        progress_every=args.progress_every,
    )
    kept_reviews_before_filter = count_positive_latest_reviews(db_path, args.min_stars)

    initial_user_counts = load_active_count_map(
        db_path,
        "user_counts",
        args.min_user_reviews,
        group_field="user_id",
        min_stars=args.min_stars,
    )
    initial_business_counts = load_active_count_map(
        db_path,
        "business_counts",
        args.min_business_reviews,
        group_field="business_id",
        min_stars=args.min_stars,
    )
    final_user_counts, final_business_counts, round_stats = refine_active_sets(
        db_path,
        min_stars=args.min_stars,
        min_user_reviews=args.min_user_reviews,
        min_business_reviews=args.min_business_reviews,
        filter_rounds=args.filter_rounds,
        user_counts=initial_user_counts,
        business_counts=initial_business_counts,
        progress_every=args.progress_every,
    )

    present_business_ids = collect_present_ids(
        business_path,
        "business_id",
        set(final_business_counts),
        progress_every=args.progress_every,
        label="business-match",
    )
    present_user_ids = collect_present_ids(
        user_path,
        "user_id",
        set(final_user_counts),
        progress_every=args.progress_every,
        label="user-match",
    )

    source_round_stats: list[dict[str, int]] = []
    business_aligned_counts = {
        business_id: final_business_counts[business_id] for business_id in present_business_ids
    }
    user_aligned_counts = final_user_counts
    if args.require_user_record:
        user_aligned_counts = {user_id: final_user_counts[user_id] for user_id in present_user_ids}

    if len(business_aligned_counts) != len(final_business_counts) or (
        args.require_user_record and len(user_aligned_counts) != len(final_user_counts)
    ):
        user_aligned_counts, business_aligned_counts, source_round_stats = refine_active_sets(
            db_path,
            min_stars=args.min_stars,
            min_user_reviews=args.min_user_reviews,
            min_business_reviews=args.min_business_reviews,
            filter_rounds=args.filter_rounds,
            user_counts=user_aligned_counts,
            business_counts=business_aligned_counts,
            progress_every=args.progress_every,
        )

    final_user_counts = user_aligned_counts
    final_business_counts = business_aligned_counts

    user_map = build_id_map(final_user_counts)
    item_map = build_id_map(final_business_counts)

    review_stats = export_aligned_reviews(
        db_path,
        review_path,
        output_dir / "aligned_reviews",
        shard_dir,
        user_map=user_map,
        item_map=item_map,
        min_stars=args.min_stars,
        split_shards=args.split_shards,
        export_shard_size=args.export_shard_size,
        progress_every=args.progress_every,
    )

    kept_users = export_aligned_users(
        user_path,
        output_dir / "aligned_users",
        user_map=user_map,
        export_shard_size=args.export_shard_size,
        progress_every=args.progress_every,
    )

    entity_catalog = EntityCatalog(item_map)
    relation_catalog = RelationCatalog()
    kg_writer = KgWriter(output_dir / "kg" / "kg_final.txt", output_dir / "kg" / "kg_edges.tsv")

    kept_businesses = export_businesses_and_kg(
        business_path,
        output_dir / "aligned_businesses",
        item_map=item_map,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        kg_writer=kg_writer,
        export_shard_size=args.export_shard_size,
        progress_every=args.progress_every,
    )

    kept_checkins = export_checkins_and_kg(
        checkin_path,
        output_dir / "aligned_checkins",
        item_map=item_map,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
        kg_writer=kg_writer,
        export_shard_size=args.export_shard_size,
        progress_every=args.progress_every,
    )

    kept_tips = export_aligned_tips(
        tip_path,
        output_dir / "aligned_tips",
        user_map=user_map,
        item_map=item_map,
        export_shard_size=args.export_shard_size,
        progress_every=args.progress_every,
    )
    kg_writer.close()

    split_stats = build_train_test_from_shards(
        shard_dir,
        output_dir,
        strategy=args.split_strategy,
        test_ratio=args.test_ratio,
    )

    write_entity_and_relation_tables(
        output_dir,
        user_map=user_map,
        user_counts=final_user_counts,
        item_map=item_map,
        business_counts=final_business_counts,
        entity_catalog=entity_catalog,
        relation_catalog=relation_catalog,
    )

    write_stats(
        output_dir / "stats.json",
        {
            "input_files": {
                "review": str(review_path),
                "business": str(business_path),
                "user": str(user_path),
                "tip": str(tip_path),
                "checkin": str(checkin_path),
            },
            "config": {
                "min_stars": args.min_stars,
                "min_user_reviews": args.min_user_reviews,
                "min_business_reviews": args.min_business_reviews,
                "filter_rounds": args.filter_rounds,
                "split_strategy": args.split_strategy,
                "test_ratio": args.test_ratio,
                "split_shards": args.split_shards,
                "export_shard_size": args.export_shard_size,
            },
            "count_pass": {
                "scanned_reviews": scanned_reviews,
                "deduped_user_business_pairs": latest_review_pairs,
                "positive_latest_interactions_after_min_stars": kept_reviews_before_filter,
                "reviews_after_min_stars": kept_reviews_before_filter,
                "initial_active_users": len(initial_user_counts),
                "initial_active_businesses": len(initial_business_counts),
            },
            "refine_rounds": round_stats,
            "source_alignment": {
                "business_records_found": len(present_business_ids),
                "user_records_found": len(present_user_ids),
                "require_user_record": args.require_user_record,
                "post_source_refine_rounds": source_round_stats,
            },
            "final_active_set": {
                "users": len(user_map),
                "businesses": len(item_map),
            },
            "exports": {
                "aligned_reviews": review_stats.kept_reviews,
                "aligned_users": kept_users,
                "aligned_businesses": kept_businesses,
                "aligned_checkins": kept_checkins,
                "aligned_tips": kept_tips,
                "kg_edges": kg_writer.edge_count,
                "entities": len(entity_catalog.rows_by_id),
                "relations": len(relation_catalog.rows),
            },
            "review_export_debug": {
                "skipped_low_stars": review_stats.skipped_low_stars,
                "skipped_missing_user": review_stats.skipped_missing_user,
                "skipped_missing_business": review_stats.skipped_missing_business,
            },
            "split": split_stats,
        },
    )

    print(f"[done] output saved to: {output_dir}")


if __name__ == "__main__":
    main()

