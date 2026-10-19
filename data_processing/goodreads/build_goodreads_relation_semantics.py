from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


BOOK_SYSTEM_PROMPT = """You are a strict knowledge-graph relation semantic enrichment model for books.

You will process each item independently.
For each item, you get:
1) item metadata,
2) a fixed candidate relation list,
3) exactly three reviews from three different users.

Task objective:
- Keep only candidate relations that are explicitly supported by review evidence.
- For each kept relation, output one concise semantic phrase in English.
- Return strict JSON only.

Rules:
1. Do not invent relation types outside the provided candidate list.
2. Use review text as primary evidence.
3. Metadata can be used only as context, not as evidence by itself.
4. If evidence is weak, vague, indirect, or speculative, do not keep that relation.
5. semantic_text should be short and content-only, ideally 2-12 words.
6. Avoid meta wording like "reviews mention", "users say", "comments indicate".
7. confidence must be a number in [0, 1].
8. evidence_review_ids must contain only supporting review IDs from the provided three reviews.
9. If no relation is clearly supported, return an empty kept_relations array for that item.

Relation guidance for common Goodreads profiles:
- written_by: keep if reviews discuss author writing style, pacing, voice, or characterization.
- in_series: keep if reviews discuss series order, continuity, or installment context.
- tagged_with_shelf: keep if reviews clearly indicate genre/theme/style tags.
- similar_to: keep if reviews explicitly compare this book to other books/authors.
- published_by: keep if reviews discuss publisher, edition, print, or release-specific publishing details.
- in_language: keep if reviews discuss language, translation quality, readability, or wording.
"""


SCHEMA = {
    "name": "goodreads_kg_relation_semantics_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "batch_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "item_idx": {"type": "integer"},
                        "kept_relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "relation_name": {"type": "string"},
                                    "semantic_text": {"type": "string"},
                                    "evidence_review_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {"type": "number"},
                                },
                                "required": [
                                    "relation_name",
                                    "semantic_text",
                                    "evidence_review_ids",
                                    "confidence",
                                ],
                            },
                        },
                    },
                    "required": ["item_idx", "kept_relations"],
                },
            }
        },
        "required": ["batch_results"],
    },
}


BANNED_PREFIXES = (
    "reviews mention",
    "reviews show",
    "reviews indicate",
    "users say",
    "comments indicate",
    "comments mention",
    "according to reviews",
    "this book is",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Goodreads relation-type semantics from 3 recent distinct-user reviews with resume support."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "02_processed_data" / "aligned_full_pos_10core_rec_plus6_ratio",
    )
    parser.add_argument(
        "--books-file",
        type=Path,
        default=ROOT
        / "goodreads_books_fantasy_paranormal.json"
        / "goodreads_books_fantasy_paranormal.json",
    )
    parser.add_argument(
        "--reviews-file",
        type=Path,
        default=ROOT
        / "goodreads_reviews_fantasy_paranormal.json"
        / "goodreads_reviews_fantasy_paranormal.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-mode", choices=("responses", "chat-completions"), default="chat-completions")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default="")

    parser.add_argument("--reviews-per-item", type=int, default=3)
    parser.add_argument("--review-candidate-limit", type=int, default=8)
    parser.add_argument("--min-review-characters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=200000)
    parser.add_argument("--relation-names", default="")

    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def resolve_existing_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate)
    if cwd_candidate.exists():
        return cwd_candidate

    root_candidate = (ROOT / candidate)
    if root_candidate.exists():
        return root_candidate

    return cwd_candidate


def resolve_output_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate)


def ensure_output_dir(output_dir: Path, reset_output: bool) -> None:
    if reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    return " ".join(text.split()) if text else ""


def compact_spaces(text: str) -> str:
    return " ".join(normalize_text(text).split())


def relation_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {piece.strip() for piece in raw.split(",") if piece.strip()}


def iter_jsonl_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_map(path: Path, key_field: str) -> Dict[str, Dict]:
    result: Dict[str, Dict] = {}
    if not path.exists():
        return result
    for payload in iter_jsonl_records(path):
        key = normalize_text(payload.get(key_field))
        if key:
            result[key] = payload
    return result


def parse_map_line(line: str) -> tuple[str, int, int]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError("mapping line has fewer than 2 columns")

    if len(parts) >= 3:
        try:
            remap_id = int(parts[-2])
            entity_id = int(parts[-1])
            org_id = " ".join(parts[:-2])
            return org_id, remap_id, entity_id
        except ValueError:
            pass

    remap_id = int(parts[-1])
    org_id = " ".join(parts[:-1])
    return org_id, remap_id, remap_id


def load_item_records(dataset_dir: Path, max_items: Optional[int]) -> List[Dict]:
    item_path = dataset_dir / "item_list.txt"
    rows: List[Dict] = []

    with item_path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            try:
                org_id, item_idx, entity_id = parse_map_line(line)
            except Exception:
                continue
            rows.append(
                {
                    "item_idx": item_idx,
                    "entity_id": entity_id,
                    "book_id": org_id,
                }
            )

    rows.sort(key=lambda row: row["item_idx"])
    if max_items is not None:
        rows = rows[:max_items]
    return rows


def load_relation_names(dataset_dir: Path, keep_relations: set[str]) -> List[str]:
    relation_path = dataset_dir / "relation_list.txt"
    rows: List[Tuple[int, str]] = []

    with relation_path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            try:
                org_id, remap_id, _ = parse_map_line(line)
            except Exception:
                continue
            if keep_relations and org_id not in keep_relations:
                continue
            rows.append((remap_id, org_id))

    rows.sort(key=lambda pair: pair[0])
    seen = set()
    relation_names: List[str] = []
    for _, relation_name in rows:
        if relation_name in seen:
            continue
        seen.add(relation_name)
        relation_names.append(relation_name)
    return relation_names


def parse_goodreads_datetime(raw: str) -> int:
    value = normalize_text(raw)
    if not value:
        return 0
    formats = (
        "%a %b %d %H:%M:%S %z %Y",
        "%a %b %d %H:%M:%S %Y",
    )
    for pattern in formats:
        try:
            parsed = datetime.strptime(value, pattern)
            return int(parsed.timestamp())
        except ValueError:
            continue
    return 0


def extract_review_timestamp(payload: Dict) -> int:
    for key in ("date_updated", "read_at", "date_added", "started_at"):
        timestamp = parse_goodreads_datetime(normalize_text(payload.get(key)))
        if timestamp > 0:
            return timestamp
    return 0


def synthesize_review_id(book_id: str, user_id: str, timestamp: int, text: str) -> str:
    digest = hashlib.sha1(f"{book_id}|{user_id}|{timestamp}|{text[:128]}".encode("utf-8")).hexdigest()
    return digest[:16]


def review_priority(review: Dict) -> Tuple[int, int, int]:
    return (
        int(review.get("timestamp") or 0),
        int(round(float(review.get("rating") or 0.0) * 10)),
        min(len(normalize_text(review.get("text"))), 5000),
    )


def review_user_key(review: Dict) -> str:
    return normalize_text(review.get("user_id")) or f"anonymous:{normalize_text(review.get('review_id'))}"


def insert_review_candidate(candidates: List[Dict], review: Dict, limit: int) -> None:
    user_key = review_user_key(review)
    for index, existing in enumerate(candidates):
        if review_user_key(existing) == user_key:
            if review_priority(review) > review_priority(existing):
                candidates[index] = review
            break
    else:
        candidates.append(review)

    candidates.sort(key=review_priority, reverse=True)
    del candidates[limit:]


def build_meta_cache(
    books_path: Path,
    target_book_ids: Sequence[str],
    cache_path: Path,
    progress_every: int,
) -> Dict[str, Dict]:
    cached = load_jsonl_map(cache_path, key_field="book_id")
    missing = set(target_book_ids) - set(cached)
    if not missing:
        log(f"meta cache hit | found={len(cached):,}")
        return cached

    discovered = dict(cached)
    scanned = 0
    started_at = time.time()
    progress_step = max(1, progress_every)
    log(
        f"scan meta start | target_items={len(target_book_ids):,} cached={len(cached):,} missing={len(missing):,}"
    )
    for payload in iter_jsonl_records(books_path):
        scanned += 1
        book_id = normalize_text(payload.get("book_id"))
        if not book_id or book_id not in missing:
            if scanned % progress_step == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                rate = scanned / elapsed
                found_for_target = len(target_book_ids) - len(missing)
                log(
                    f"scan meta | scanned={scanned:,} found={found_for_target:,}/{len(target_book_ids):,} "
                    f"missing={len(missing):,} rows_per_sec={rate:,.0f}"
                )
            continue

        author_ids: List[str] = []
        for author in payload.get("authors") or []:
            if not isinstance(author, dict):
                continue
            author_id = normalize_text(author.get("author_id"))
            if author_id:
                author_ids.append(author_id)

        discovered[book_id] = {
            "book_id": book_id,
            "title": normalize_text(payload.get("title_without_series") or payload.get("title")),
            "language_code": normalize_text(payload.get("language_code")),
            "publisher": normalize_text(payload.get("publisher")),
            "author_ids": ", ".join(author_ids[:4]),
        }
        missing.remove(book_id)
        if scanned % progress_step == 0:
            elapsed = max(time.time() - started_at, 1e-6)
            rate = scanned / elapsed
            found_for_target = len(target_book_ids) - len(missing)
            log(
                f"scan meta | scanned={scanned:,} found={found_for_target:,}/{len(target_book_ids):,} "
                f"missing={len(missing):,} rows_per_sec={rate:,.0f}"
            )
        if not missing:
            break

    write_jsonl(cache_path, (discovered[key] for key in sorted(discovered)))
    log(f"meta cache ready | found={len(discovered):,} scanned_rows={scanned:,}")
    return discovered


def build_review_cache(
    reviews_path: Path,
    target_book_ids: Sequence[str],
    cache_path: Path,
    top_reviews: int,
    candidate_limit: int,
    min_review_characters: int,
    progress_every: int,
) -> Dict[str, List[Dict]]:
    cached = load_jsonl_map(cache_path, key_field="book_id")
    selected_from_cache: Dict[str, List[Dict]] = {}
    for book_id, payload in cached.items():
        reviews = payload.get("reviews", [])
        if isinstance(reviews, list) and reviews:
            selected_from_cache[book_id] = reviews[:top_reviews]

    target_set = set(target_book_ids)
    missing = target_set - set(selected_from_cache)
    if not missing:
        log(f"review cache hit | items={len(selected_from_cache):,}")
        return selected_from_cache

    candidate_map: Dict[str, List[Dict]] = {
        book_id: payload.get("reviews", [])[: max(candidate_limit, top_reviews)]
        for book_id, payload in cached.items()
        if book_id in target_set
    }

    scanned = 0
    kept = 0
    started_at = time.time()
    progress_step = max(1, progress_every)
    ready_items = {book_id for book_id, rows in candidate_map.items() if len(rows) >= top_reviews}
    log(
        f"scan reviews start | target_items={len(target_set):,} cached_ready={len(selected_from_cache):,} "
        f"missing={len(missing):,}"
    )
    for payload in iter_jsonl_records(reviews_path):
        scanned += 1
        book_id = normalize_text(payload.get("book_id"))
        if not book_id or book_id not in target_set:
            if scanned % progress_step == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                rate = scanned / elapsed
                log(
                    f"scan reviews | scanned={scanned:,} ready_items={len(ready_items):,}/{len(target_set):,} "
                    f"kept_candidates={kept:,} rows_per_sec={rate:,.0f}"
                )
            continue

        text = normalize_text(payload.get("review_text"))
        if len(text) < min_review_characters:
            continue

        user_id = normalize_text(payload.get("user_id"))
        rating = float(payload.get("rating") or 0.0)
        timestamp = extract_review_timestamp(payload)
        review_id = normalize_text(payload.get("review_id")) or synthesize_review_id(book_id, user_id, timestamp, text)

        review = {
            "review_id": review_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "rating": rating,
            "text": text,
        }

        candidates = candidate_map.setdefault(book_id, [])
        before = len(candidates)
        insert_review_candidate(candidates, review, max(candidate_limit, top_reviews))
        if len(candidates) != before:
            kept += 1
        if len(candidates) >= top_reviews:
            ready_items.add(book_id)
        if scanned % progress_step == 0:
            elapsed = max(time.time() - started_at, 1e-6)
            rate = scanned / elapsed
            log(
                f"scan reviews | scanned={scanned:,} ready_items={len(ready_items):,}/{len(target_set):,} "
                f"kept_candidates={kept:,} rows_per_sec={rate:,.0f}"
            )

    selected: Dict[str, List[Dict]] = {}
    for book_id in target_set:
        rows = sorted(candidate_map.get(book_id, []), key=review_priority, reverse=True)[:top_reviews]
        if rows:
            selected[book_id] = rows

    write_jsonl(
        cache_path,
        ({"book_id": book_id, "reviews": selected[book_id]} for book_id in sorted(selected)),
    )
    log(f"review cache ready | items={len(selected):,} scanned_rows={scanned:,} kept_candidates={kept:,}")
    return selected


def build_item_contexts(
    items: Sequence[Dict],
    relation_names: Sequence[str],
    meta_lookup: Dict[str, Dict],
    review_lookup: Dict[str, List[Dict]],
) -> List[Dict]:
    contexts: List[Dict] = []
    candidates = list(relation_names)

    for item in items:
        book_id = item["book_id"]
        meta = meta_lookup.get(book_id, {})
        reviews = review_lookup.get(book_id, [])

        contexts.append(
            {
                "item_idx": item["item_idx"],
                "book_id": book_id,
                "title": meta.get("title") or book_id,
                "language_code": meta.get("language_code") or "",
                "publisher": meta.get("publisher") or "",
                "author_ids": meta.get("author_ids") or "",
                "candidate_relations": candidates,
                "reviews": reviews,
            }
        )

    return contexts


def batch_prompt(batch_contexts: Sequence[Dict]) -> str:
    example = {
        "batch_results": [
            {
                "item_idx": 123,
                "kept_relations": [
                    {
                        "relation_name": "written_by",
                        "semantic_text": "lyrical introspective writing style",
                        "evidence_review_ids": ["review_1", "review_3"],
                        "confidence": 0.93,
                    }
                ],
            },
            {"item_idx": 456, "kept_relations": []},
        ]
    }

    lines: List[str] = []
    lines.append("You will process multiple items in one request.")
    lines.append("Return one result object per item.")
    lines.append("")

    for index, context in enumerate(batch_contexts, start=1):
        lines.append(f"[Item {index}]")
        lines.append("Item metadata:")
        lines.append(f"- item_idx: {context['item_idx']}")
        lines.append(f"- book_id: {context['book_id']}")
        lines.append(f"- title: {context['title']}")
        lines.append(f"- author_ids: {context['author_ids']}")
        lines.append(f"- publisher: {context['publisher']}")
        lines.append(f"- language_code: {context['language_code']}")
        lines.append("")
        lines.append("Candidate relation names:")
        lines.append(json.dumps(context["candidate_relations"], ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("Three reviews:")
        for review_index, review in enumerate(context["reviews"], start=1):
            lines.append(f"[Review {review_index} | review_id={review['review_id']}]")
            lines.append(review["text"])
            lines.append("")

    lines.append("Instructions:")
    lines.append("1. Process each item independently.")
    lines.append("2. Keep only relations clearly supported by explicit review evidence.")
    lines.append("3. For each kept relation, output one concise semantic_text phrase.")
    lines.append("4. Return evidence_review_ids from the provided review IDs only.")
    lines.append("5. confidence must be in [0, 1].")
    lines.append("6. Output strict JSON only.")
    lines.append("")
    lines.append("Output format example:")
    lines.append(json.dumps(example, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def responses_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {base_url}")
    path = parsed.path.rstrip("/")
    if path.endswith("/responses"):
        final = path
    elif path.endswith("/v1"):
        final = path + "/responses"
    elif path:
        final = path + "/v1/responses"
    else:
        final = "/v1/responses"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, final, "", "", ""))


def chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {base_url}")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final = path
    elif path.endswith("/v1"):
        final = path + "/chat/completions"
    elif path:
        final = path + "/v1/chat/completions"
    else:
        final = "/v1/chat/completions"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, final, "", "", ""))


def payload_responses(model: str, effort: str, prompt: str, max_tokens: int) -> Dict:
    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": effort},
        "max_output_tokens": max_tokens,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": BOOK_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "text": {"format": {"type": "json_schema", **SCHEMA}},
    }


def payload_chat_completions(model: str, prompt: str, max_tokens: int) -> Dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": BOOK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }


def post_json(url: str, api_key: str, payload: Dict, timeout: int) -> Dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_with_retry(
    url: str,
    api_key: str,
    payload: Dict,
    timeout: int,
    retry_count: int,
    retry_sleep: float,
) -> Dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            return post_json(url, api_key, payload, timeout)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retryable = error.code in {408, 409, 429, 500, 502, 503, 504, 524}
            if attempt > retry_count or not retryable:
                raise RuntimeError(f"HTTP {error.code}: {body}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt > retry_count:
                raise RuntimeError(f"Request failed after retries: {error}") from error
        time.sleep(retry_sleep * attempt)


def output_text(response: Dict) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

    output_text_value = response.get("output_text")
    if isinstance(output_text_value, str) and output_text_value.strip():
        return output_text_value

    chunks: List[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                elif isinstance(text, dict) and isinstance(text.get("value"), str):
                    chunks.append(text["value"])
    return "\n".join(part for part in chunks if part).strip()


def parse_json_text(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        return json.loads(stripped)


def clean_semantic_text(text: str) -> str:
    value = compact_spaces(str(text)).strip('"\'')
    lower = value.lower()
    for prefix in BANNED_PREFIXES:
        if lower.startswith(prefix):
            value = value[len(prefix) :].lstrip(" :,-")
            lower = value.lower()
    return value


def sanitize_batch(parsed: Dict, contexts: Sequence[Dict]) -> Dict[int, Dict]:
    by_item = {context["item_idx"]: context for context in contexts}
    normalized: Dict[int, Dict] = {}

    raw_results = parsed.get("batch_results", [])
    if not isinstance(raw_results, list):
        return normalized

    for row in raw_results:
        if not isinstance(row, dict):
            continue

        try:
            item_idx = int(row.get("item_idx"))
        except (TypeError, ValueError):
            continue

        context = by_item.get(item_idx)
        if context is None:
            continue

        relation_order = {name: idx for idx, name in enumerate(context["candidate_relations"])}
        allowed_relations = set(context["candidate_relations"])
        review_ids = {review["review_id"] for review in context["reviews"]}

        dedup_by_relation: Dict[str, Dict] = {}
        kept_relations = row.get("kept_relations", [])
        if not isinstance(kept_relations, list):
            kept_relations = []

        for relation in kept_relations:
            if not isinstance(relation, dict):
                continue

            relation_name = compact_spaces(relation.get("relation_name", ""))
            semantic_text = clean_semantic_text(relation.get("semantic_text", ""))
            if not relation_name or not semantic_text or relation_name not in allowed_relations:
                continue

            evidence: List[str] = []
            for review_id in relation.get("evidence_review_ids", []) or []:
                rid = normalize_text(review_id)
                if rid and rid in review_ids and rid not in evidence:
                    evidence.append(rid)
            if not evidence:
                continue

            try:
                confidence = float(relation.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            payload = {
                "relation_name": relation_name,
                "semantic_text": semantic_text,
                "evidence_review_ids": evidence,
                "confidence": round(confidence, 4),
            }

            if (
                relation_name not in dedup_by_relation
                or payload["confidence"] > dedup_by_relation[relation_name]["confidence"]
            ):
                dedup_by_relation[relation_name] = payload

        normalized[item_idx] = {
            "item_idx": item_idx,
            "kept_relations": sorted(
                dedup_by_relation.values(),
                key=lambda value: relation_order[value["relation_name"]],
            ),
        }

    return normalized


def load_done_items(result_path: Path) -> set[int]:
    done: set[int] = set()
    if not result_path.exists():
        return done
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                done.add(int(json.loads(raw).get("item_idx")))
            except Exception:
                continue
    return done


def rebuild_tsv(result_jsonl: Path, tsv_path: Path) -> Dict[str, int]:
    items = 0
    semantic_relations = 0

    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("item_idx\tbook_id\trelation_name\tsemantic_text\tevidence_review_ids\tconfidence\n")
        if result_jsonl.exists():
            with result_jsonl.open("r", encoding="utf-8") as source:
                for line in source:
                    raw = line.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    items += 1
                    for relation in row.get("kept_relations", []):
                        handle.write(
                            f"{row['item_idx']}\t{row['book_id']}\t{relation['relation_name']}\t"
                            f"{relation['semantic_text']}\t{'|'.join(relation['evidence_review_ids'])}\t"
                            f"{float(relation['confidence']):.4f}\n"
                        )
                        semantic_relations += 1

    return {"items": items, "semantic_relations": semantic_relations}


def write_resume_state(path: Path, payload: Dict) -> None:
    state = dict(payload)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    write_json(path, state)


def chunked(rows: Sequence[Dict], size: int) -> Iterable[List[Dict]]:
    block_size = max(1, size)
    for start in range(0, len(rows), block_size):
        yield list(rows[start : start + block_size])


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_existing_path(args.dataset_dir).resolve()
    books_path = resolve_existing_path(args.books_file).resolve()
    reviews_path = resolve_existing_path(args.reviews_file).resolve()
    output_dir = (
        resolve_output_path(args.output_dir).resolve()
        if args.output_dir is not None
        else (dataset_dir / "llm_relation_type_semantics_recent3").resolve()
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    if not books_path.exists():
        raise FileNotFoundError(f"books file not found: {books_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"reviews file not found: {reviews_path}")

    ensure_output_dir(output_dir, reset_output=args.reset_output)

    relation_names = load_relation_names(dataset_dir, relation_filter(args.relation_names))
    item_records = load_item_records(dataset_dir, max_items=args.max_items)
    if not item_records:
        raise RuntimeError("No items found in item_list.txt")
    if not relation_names:
        raise RuntimeError("No relations found in relation_list.txt")

    log(f"loaded items={len(item_records):,} relations={len(relation_names):,}")

    target_book_ids = [row["book_id"] for row in item_records]

    meta_cache_path = output_dir / "meta_cache.jsonl"
    review_cache_path = output_dir / "review_cache_recent_distinct.jsonl"
    prepared_inputs_path = output_dir / "prepared_inputs.jsonl"
    result_jsonl = output_dir / "item_relation_semantics.jsonl"
    result_tsv = output_dir / "item_relation_semantics.tsv"
    batch_status_jsonl = output_dir / "batch_status.jsonl"
    run_summary_path = output_dir / "run_summary.json"
    run_config_path = output_dir / "run_config.json"
    resume_state_path = output_dir / "resume_state.json"
    dry_run_preview_path = output_dir / "dry_run_preview.json"

    write_json(
        run_config_path,
        {
            "dataset_dir": str(dataset_dir),
            "books_file": str(books_path),
            "reviews_file": str(reviews_path),
            "output_dir": str(output_dir),
            "base_url": args.base_url,
            "api_mode": args.api_mode,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "api_key_env": args.api_key_env,
            "reviews_per_item": args.reviews_per_item,
            "review_candidate_limit": args.review_candidate_limit,
            "min_review_characters": args.min_review_characters,
            "batch_size": args.batch_size,
            "max_items": args.max_items,
            "timeout": args.timeout,
            "max_output_tokens": args.max_output_tokens,
            "retry_count": args.retry_count,
            "retry_sleep": args.retry_sleep,
            "request_delay_seconds": args.request_delay_seconds,
            "progress_every": args.progress_every,
            "prepare_only": args.prepare_only,
            "dry_run": args.dry_run,
            "reset_output": args.reset_output,
            "relation_names": relation_names,
            "system_prompt": BOOK_SYSTEM_PROMPT,
        },
    )

    meta_lookup = build_meta_cache(
        books_path,
        target_book_ids,
        meta_cache_path,
        progress_every=args.progress_every,
    )
    review_lookup = build_review_cache(
        reviews_path,
        target_book_ids,
        review_cache_path,
        top_reviews=max(1, args.reviews_per_item),
        candidate_limit=max(args.review_candidate_limit, args.reviews_per_item),
        min_review_characters=max(1, args.min_review_characters),
        progress_every=args.progress_every,
    )

    contexts = build_item_contexts(item_records, relation_names, meta_lookup, review_lookup)
    valid_contexts = [
        context
        for context in contexts
        if context["candidate_relations"] and len(context["reviews"]) >= max(1, args.reviews_per_item)
    ]

    write_jsonl(prepared_inputs_path, valid_contexts)
    log(f"prepared items={len(valid_contexts):,} skipped={len(contexts) - len(valid_contexts):,}")

    if args.prepare_only:
        write_json(
            run_summary_path,
            {
                "status": "prepared_only",
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "total_items": len(contexts),
                "prepared_items": len(valid_contexts),
                "skipped_items": len(contexts) - len(valid_contexts),
                "prepared_inputs_path": str(prepared_inputs_path),
                "meta_cache_path": str(meta_cache_path),
                "review_cache_path": str(review_cache_path),
            },
        )
        log(f"prepare-only done | prepared_items={len(valid_contexts):,}")
        return

    api_key = "" if args.dry_run else (args.api_key or os.environ.get(args.api_key_env, "")).strip()
    if not args.dry_run and not api_key:
        raise RuntimeError(
            "Missing API key. Provide --api-key or set environment variable "
            f"{args.api_key_env}"
        )

    if args.api_mode == "responses":
        request_url = responses_url(args.base_url)
        build_payload = lambda user_prompt: payload_responses(
            args.model,
            args.reasoning_effort,
            user_prompt,
            args.max_output_tokens,
        )
    else:
        request_url = chat_completions_url(args.base_url)
        build_payload = lambda user_prompt: payload_chat_completions(
            args.model,
            user_prompt,
            args.max_output_tokens,
        )

    done_items = load_done_items(result_jsonl)
    if done_items:
        log(f"resume detected | already_done_items={len(done_items):,}")

    remaining = [context for context in valid_contexts if context["item_idx"] not in done_items]
    if not remaining:
        stats = rebuild_tsv(result_jsonl, result_tsv)
        write_json(
            run_summary_path,
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "request_url": request_url,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "total_items": len(contexts),
                "prepared_items": len(valid_contexts),
                "processed_now": 0,
                "error_count": 0,
                "items_with_outputs": stats["items"],
                "semantic_relations": stats["semantic_relations"],
                "resume_message": "No remaining items. Existing outputs were reused.",
            },
        )
        log("nothing left to process")
        return

    batches = list(chunked(remaining, max(1, args.batch_size)))
    log(f"pending items={len(remaining):,} batches={len(batches):,}")

    if args.dry_run:
        preview_batch = batches[0]
        preview_prompt = batch_prompt(preview_batch)
        preview_payload = build_payload(preview_prompt)
        write_json(
            dry_run_preview_path,
            {
                "request_url": request_url,
                "batch_size": len(preview_batch),
                "sample_item_idxs": [row["item_idx"] for row in preview_batch],
                "request_payload": preview_payload,
            },
        )
        log(f"dry-run preview written to {dry_run_preview_path}")
        return

    started_at = time.time()
    processed_now = 0
    error_count = 0
    kept_relation_items = 0
    kept_relations = 0
    processed_batches = 0

    write_resume_state(
        resume_state_path,
        {
            "status": "running",
            "request_url": request_url,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "prepared_items": len(valid_contexts),
            "already_done_items": len(done_items),
            "target_items_this_run": len(remaining),
            "processed_now": 0,
            "error_count": 0,
            "kept_relation_items": 0,
            "kept_relations": 0,
            "results_path": str(result_jsonl),
            "prepared_inputs_path": str(prepared_inputs_path),
        },
    )

    for batch_number, batch in enumerate(batches, start=1):
        user_prompt = batch_prompt(batch)
        payload = build_payload(user_prompt)
        batch_item_idxs = [context["item_idx"] for context in batch]

        try:
            response = request_with_retry(
                request_url,
                api_key,
                payload,
                args.timeout,
                args.retry_count,
                args.retry_sleep,
            )
            parsed = parse_json_text(output_text(response))
            normalized = sanitize_batch(parsed, batch)

            missing_item_idxs = [
                context["item_idx"]
                for context in batch
                if normalized.get(context["item_idx"]) is None
            ]
            if missing_item_idxs:
                raise RuntimeError(
                    "Missing item_idx in model output: "
                    + ",".join(str(item_idx) for item_idx in missing_item_idxs)
                )

            for context in batch:
                normalized_row = normalized[context["item_idx"]]

                append_jsonl(
                    result_jsonl,
                    {
                        "item_idx": context["item_idx"],
                        "book_id": context["book_id"],
                        "title": context["title"],
                        "candidate_relations": context["candidate_relations"],
                        "selected_review_ids": [review["review_id"] for review in context["reviews"]],
                        "kept_relations": normalized_row["kept_relations"],
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort,
                        "request_url": request_url,
                        "response_id": response.get("id"),
                        "batch_number": batch_number,
                        "batch_size": len(batch),
                    },
                )
                processed_now += 1
                kept_relations += len(normalized_row["kept_relations"])
                if normalized_row["kept_relations"]:
                    kept_relation_items += 1

            processed_batches += 1
            append_jsonl(
                batch_status_jsonl,
                {
                    "batch_number": batch_number,
                    "status": "success",
                    "batch_size": len(batch),
                    "item_idxs": batch_item_idxs,
                    "response_id": response.get("id"),
                },
            )
            write_resume_state(
                resume_state_path,
                {
                    "status": "running",
                    "request_url": request_url,
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "prepared_items": len(valid_contexts),
                    "already_done_items": len(done_items),
                    "target_items_this_run": len(remaining),
                    "processed_now": processed_now,
                    "error_count": error_count,
                    "kept_relation_items": kept_relation_items,
                    "kept_relations": kept_relations,
                    "processed_batches": processed_batches,
                    "last_batch_number": batch_number,
                    "last_batch_status": "success",
                    "results_path": str(result_jsonl),
                    "prepared_inputs_path": str(prepared_inputs_path),
                },
            )
        except Exception as error:
            error_count += 1
            append_jsonl(
                batch_status_jsonl,
                {
                    "batch_number": batch_number,
                    "status": "error",
                    "batch_size": len(batch),
                    "item_idxs": batch_item_idxs,
                    "error": str(error),
                },
            )
            write_resume_state(
                resume_state_path,
                {
                    "status": "running",
                    "request_url": request_url,
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "prepared_items": len(valid_contexts),
                    "already_done_items": len(done_items),
                    "target_items_this_run": len(remaining),
                    "processed_now": processed_now,
                    "error_count": error_count,
                    "kept_relation_items": kept_relation_items,
                    "kept_relations": kept_relations,
                    "processed_batches": processed_batches,
                    "last_batch_number": batch_number,
                    "last_batch_status": "error",
                    "results_path": str(result_jsonl),
                    "prepared_inputs_path": str(prepared_inputs_path),
                },
            )
            log(
                f"error at batch {batch_number:,}; continue to next batch."
            )

        log(
            f"progress batch {batch_number:,}/{len(batches):,} | "
            f"processed={processed_now:,} kept_relations={kept_relations:,} errors={error_count:,}"
        )

        if args.request_delay_seconds > 0:
            time.sleep(args.request_delay_seconds)

    stats = rebuild_tsv(result_jsonl, result_tsv)
    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "request_url": request_url,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "total_items": len(contexts),
        "prepared_items": len(valid_contexts),
        "already_done_items": len(done_items),
        "target_items_this_run": len(remaining),
        "batch_size": args.batch_size,
        "total_batches": len(batches),
        "processed_batches": processed_batches,
        "processed_now": processed_now,
        "error_count": error_count,
        "kept_relation_items": kept_relation_items,
        "kept_relations": kept_relations,
        "items_with_outputs": stats["items"],
        "semantic_relations": stats["semantic_relations"],
        "reviews_per_item": args.reviews_per_item,
        "relation_type_count": len(relation_names),
        "relation_names": relation_names,
        "runtime_seconds": round(time.time() - started_at, 2),
        "notes": "Output is relation-type semantics with model confidence and resume support.",
    }
    write_json(run_summary_path, summary)
    write_resume_state(
        resume_state_path,
        {
            **summary,
            "status": "completed",
            "results_path": str(result_jsonl),
            "prepared_inputs_path": str(prepared_inputs_path),
        },
    )
    log(
        f"finished | items_with_outputs={stats['items']:,} semantic_relations={stats['semantic_relations']:,} "
        f"errors={error_count:,}"
    )


if __name__ == "__main__":
    main()
