
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BOOK_SYSTEM_PROMPT = """You are a strict knowledge-graph relation semantic enrichment model for books.

Your task is to read:
1. one book item,
2. a fixed list of candidate relation names,
3. exactly three recent reviews written by three different users for that item,

and decide which relation names are explicitly discussed in the reviews.

Core objective:
- You are NOT summarizing the whole item.
- You are NOT generating new relation types.
- You are ONLY enriching candidate relation names with short review-grounded semantic text.

Task requirements:
1. You may only process relation names explicitly listed in "Candidate relation names for this item".
2. You must not add any new relation type.
3. You must judge each candidate relation name independently.
4. You must use only the three reviews as evidence.
5. Do not retrieve, validate against, or rely on existing KG triples for evidence.
6. If a relation is not clearly and explicitly discussed in the reviews, do not output it.
7. If evidence is weak, indirect, vague, ambiguous, speculative, or only loosely implied, do not output that relation.
8. Metadata such as title, author name, and main category may be used as context only; they are NOT evidence. Evidence must come from the reviews.
9. The semantic description must be a concise English phrase, ideally 2 to 12 words.
10. Do not use meta-language such as "reviews mention", "users say", "comments indicate", or "this book is described as".
11. Do not write full sentences, explanations, or any extra commentary.
12. If multiple reviews support the same relation name, merge them into one tighter semantic phrase.
13. Output strict JSON only. No markdown. No explanation. No extra text.
14. The field "confidence" must be a number between 0 and 1.
15. The field "evidence_review_ids" must include only the review IDs that directly support the kept relation.

Books relation judgment principles:
- has_main_category:
  Keep only if the reviews explicitly discuss the broad book form or content type, such as novel, memoir, workbook, coloring book, cookbook, children's book, or similar.

- published_by:
  Keep only if the reviews explicitly discuss the publisher, imprint, edition, printing edition, translation edition, or a clearly release-specific publishing trait.

- in_language:
  Keep only if the reviews explicitly discuss language, translation quality, bilingual format, English readability, wording, or language choice.

- belongs_to_category:
  Keep only if the reviews explicitly discuss genre, theme domain, narrative style, audience-facing content type, or reading style.

- subcategory_of:
  Keep only if the reviews explicitly discuss a finer-grained subgenre or category placement.

- written_by:
  Keep only if the reviews explicitly discuss the author's writing style, pacing, characterization, narrative voice, emotional tone, or comparisons with the same author's other books.

- has_reading_age:
  Keep only if the reviews explicitly discuss age suitability, maturity level, or whether the book is suitable for children or teens.

- has_grade_level:
  Keep only if the reviews explicitly discuss school grade, classroom suitability, curriculum fit, grade-specific reading level, or independent reading difficulty by grade.

Default rule:
- Be conservative.
- If a relation is not naturally and explicitly discussable from the reviews, do not output it."""



ITEM_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "item_idx": {"type": "integer"},
        "kept_relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation_name": {"type": "string"},
                    "semantic_text": {"type": "string"},
                    "evidence_review_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["relation_name", "semantic_text", "evidence_review_ids", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["item_idx", "kept_relations"],
    "additionalProperties": False,
}


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": ITEM_RESULT_SCHEMA,
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "processed_data" / "processed_full_analysis"
    parser = argparse.ArgumentParser(
        description="Enrich existing Books KG item relations with review-grounded semantic text via an OpenAI-compatible responses API."
    )
    parser.add_argument("--dataset-dir", type=Path, default=processed_dir / "kgat_dataset_200k")
    parser.add_argument("--reviews", type=Path, default=root / "raw_data" / "Books.jsonl")
    parser.add_argument("--meta", type=Path, default=processed_dir / "aligned_meta_items.jsonl")
    parser.add_argument("--output-dir", type=Path, default=processed_dir / "kgat_dataset_200k_relation_semantics_recent3")
    parser.add_argument("--base-url", type=str, default="https://ai.qaq.al")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--top-reviews", type=int, default=3)
    parser.add_argument("--review-candidate-limit", type=int, default=8)
    parser.add_argument("--min-review-characters", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--item-idx", type=int, default=None)
    parser.add_argument("--parent-asin", type=str, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def compact_spaces(value: str) -> str:
    return " ".join(normalize_text(value).split())


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def flush_and_fsync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def iter_jsonl_records(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
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
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


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


def chunked(values: Sequence[Dict], size: int) -> Iterable[List[Dict]]:
    size = max(size, 1)
    for start in range(0, len(values), size):
        yield list(values[start:start + size])

class ByteProgressBar:
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


class CountProgressBar:
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
        rate = value / elapsed
        message = (
            f"\r{self.label:<18} [{bar}] {ratio * 100:6.2f}% "
            f"{value:6d}/{self.total:6d} items {rate:7.2f}/s"
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


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


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
            remap_id = int(parts[-1])
            org_id = " ".join(parts[:-1])
            reverse_map[remap_id] = org_id
    return reverse_map


def load_relation_names(dataset_dir: Path) -> List[str]:
    relation_map = load_reverse_map(dataset_dir / "relation_list.txt", expected_columns=2)
    return [relation_map[idx] for idx in sorted(relation_map)]


def load_item_records(dataset_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    with (dataset_dir / "item_list.txt").open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            parent_asin = " ".join(parts[:-2])
            item_idx = int(parts[-2])
            entity_id = int(parts[-1])
            items.append(
                {
                    "item_idx": item_idx,
                    "entity_id": entity_id,
                    "parent_asin": parent_asin,
                }
            )
    items.sort(key=lambda record: record["item_idx"])
    return items


def maybe_filter_items(items: Sequence[Dict], item_idx: Optional[int], parent_asin: Optional[str]) -> List[Dict]:
    filtered = list(items)
    if item_idx is not None:
        filtered = [item for item in filtered if item["item_idx"] == item_idx]
    if parent_asin:
        filtered = [item for item in filtered if item["parent_asin"] == parent_asin]
    return filtered


def load_jsonl_map(path: Path, key_field: str) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for payload in iter_jsonl_records(path):
        key = payload.get(key_field)
        if key is None:
            continue
        results[key] = payload
    return results


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def build_meta_cache(meta_path: Path, target_asins: Sequence[str], cache_path: Path) -> Dict[str, Dict]:
    cached = load_jsonl_map(cache_path, key_field="parent_asin")
    missing = set(target_asins) - set(cached)
    if not missing:
        return cached

    print(f"Building item metadata cache from: {meta_path}")
    progress = ByteProgressBar("scan_meta", meta_path.stat().st_size)
    processed = 0
    discovered = dict(cached)
    with meta_path.open("rb") as handle:
        for raw_line in handle:
            processed += len(raw_line)
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            payload = json.loads(line)
            parent_asin = normalize_text(payload.get("item_id") or payload.get("parent_asin"))
            if not parent_asin or parent_asin not in missing:
                continue
            discovered[parent_asin] = {
                "parent_asin": parent_asin,
                "title": normalize_text(payload.get("title")),
                "main_category": normalize_text(payload.get("main_category")),
                "author_name": normalize_text(payload.get("author_name")),
            }
            missing.remove(parent_asin)
            progress.update(processed, extra=f"found={len(discovered):,}/{len(target_asins):,}")
            if not missing:
                break
    progress.close()
    write_jsonl(cache_path, (discovered[key] for key in sorted(discovered)))
    return discovered

def synthesize_review_id(parent_asin: str, user_id: str, timestamp: int) -> str:
    digest = hashlib.sha1(f"{parent_asin}|{user_id}|{timestamp}".encode("utf-8")).hexdigest()
    return digest[:16]


def review_priority(review: Dict) -> Tuple[int, int, int, int]:
    return (
        review["timestamp"],
        1 if review["verified_purchase"] else 0,
        min(review["helpful_vote"], 100),
        int(round(review["rating"] * 10)),
    )


def review_user_key(review: Dict) -> str:
    return review["user_id"] or f"anonymous:{review['review_id']}"


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


def build_review_cache(
    reviews_path: Path,
    target_asins: Sequence[str],
    cache_path: Path,
    top_reviews: int,
    candidate_limit: int,
    min_review_characters: int,
) -> Dict[str, List[Dict]]:
    cached = load_jsonl_map(cache_path, key_field="parent_asin")
    if cached:
        return {asin: payload["reviews"][:top_reviews] for asin, payload in cached.items()}

    print(f"Building review cache from: {reviews_path}")
    target_set = set(target_asins)
    candidate_map: Dict[str, List[Dict]] = {}
    progress = ByteProgressBar("scan_reviews", reviews_path.stat().st_size)
    processed = 0
    kept = 0
    with reviews_path.open("rb") as handle:
        for raw_line in handle:
            processed += len(raw_line)
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            payload = json.loads(line)
            parent_asin = normalize_text(payload.get("parent_asin") or payload.get("asin"))
            if not parent_asin or parent_asin not in target_set:
                continue
            title = normalize_text(payload.get("title"))
            text = normalize_text(payload.get("text"))
            combined_parts = []
            if title:
                combined_parts.append(f"Title: {title}")
            if text:
                combined_parts.append(f"Review: {text}")
            combined_text = "\n".join(combined_parts).strip()
            if len(combined_text) < min_review_characters:
                continue
            user_id = normalize_text(payload.get("user_id"))
            timestamp = int(payload.get("timestamp") or 0)
            review = {
                "review_id": synthesize_review_id(parent_asin, user_id, timestamp),
                "user_id": user_id,
                "timestamp": timestamp,
                "rating": float(payload.get("rating") or 0.0),
                "verified_purchase": bool(payload.get("verified_purchase")),
                "helpful_vote": int(payload.get("helpful_vote") or 0),
                "text": combined_text,
            }
            candidates = candidate_map.setdefault(parent_asin, [])
            before = len(candidates)
            insert_review_candidate(candidates, review, max(candidate_limit, top_reviews))
            if len(candidates) != before:
                kept += 1
            progress.update(processed, extra=f"items={len(candidate_map):,} candidates={kept:,}")
    progress.close()

    selected = {parent_asin: sorted(reviews, key=review_priority, reverse=True)[:top_reviews] for parent_asin, reviews in candidate_map.items()}
    write_jsonl(
        cache_path,
        ({"parent_asin": parent_asin, "reviews": selected[parent_asin]} for parent_asin in sorted(selected)),
    )
    return selected


def build_item_contexts(
    items: Sequence[Dict],
    relation_names: Sequence[str],
    meta_lookup: Dict[str, Dict],
    review_lookup: Dict[str, List[Dict]],
) -> List[Dict]:
    contexts: List[Dict] = []
    candidate_relations = list(relation_names)
    for item in items:
        parent_asin = item["parent_asin"]
        meta = meta_lookup.get(parent_asin, {})
        reviews = review_lookup.get(parent_asin, [])
        context = {
            "item_idx": item["item_idx"],
            "parent_asin": parent_asin,
            "title": meta.get("title") or parent_asin,
            "main_category": meta.get("main_category") or "",
            "author_name": meta.get("author_name") or "",
            "candidate_relations": candidate_relations,
            "reviews": reviews,
        }
        contexts.append(context)
    return contexts

def build_batch_user_prompt(contexts: Sequence[Dict]) -> str:
    lines: List[str] = []
    lines.append("Input data:")
    lines.append("")
    lines.append("You will process multiple items in one request.")
    lines.append("Process each item independently and return one JSON result per item.")
    lines.append("")
    for batch_index, context in enumerate(contexts, start=1):
        lines.append(f"[Item {batch_index}]")
        lines.append("Item information:")
        lines.append(f"- item_idx: {context['item_idx']}")
        lines.append(f"- parent_asin: {context['parent_asin']}")
        lines.append(f"- title: {context['title']}")
        lines.append(f"- main_category: {context['main_category']}")
        lines.append(f"- author_name: {context['author_name']}")
        lines.append("")
        lines.append("Candidate relation names for this item:")
        lines.append(json.dumps(context["candidate_relations"], ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("Three reviews:")
        for review_index, review in enumerate(context["reviews"], start=1):
            lines.append(f"[Review {review_index} | review_id={review['review_id']}]")
            lines.append(review["text"])
            lines.append("")
    lines.append("Instructions:")
    lines.append("1. Process every item independently.")
    lines.append("2. For each item, iterate through every candidate relation name.")
    lines.append("3. Keep only the relation names clearly supported by review evidence.")
    lines.append("4. For each kept relation, generate one concise English semantic phrase.")
    lines.append("5. You must return exactly one result object for every input item.")
    lines.append("6. If an item has no clearly supported relation, return that item with an empty kept_relations array.")
    lines.append("7. Output strict JSON only.")
    lines.append("")
    lines.append("Output format:")
    example = {
        "results": [
            {
                "item_idx": 123,
                "kept_relations": [
                    {
                        "relation_name": "belongs_to_category",
                                                "semantic_text": "slow-burn coming-of-age mystery",
                        "evidence_review_ids": ["review_a", "review_b"],
                        "confidence": 0.91,
                    }
                ],
            },
            {"item_idx": 456, "kept_relations": []},
        ]
    }
    lines.append(json.dumps(example, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Output JSON only. No other text.")
    return "\n".join(lines)


def call_responses_api(
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout_seconds: int,
    max_retries: int,
    contexts: Sequence[Dict],
) -> Dict:
    endpoint = f"{normalize_base_url(base_url)}/responses"
    payload = {
        "model": model,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": BOOK_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": build_batch_user_prompt(contexts)}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "book_kg_relation_semantic_enrichment_batch",
                "schema": OUTPUT_SCHEMA,
                "strict": True,
            }
        },
    }
    request_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_error = None
    batch_label = f"{contexts[0]['item_idx']}-{contexts[-1]['item_idx']}"
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(endpoint, data=request_bytes, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            response_payload = json.loads(body)
            if response_payload.get("error"):
                raise RuntimeError(json.dumps(response_payload["error"], ensure_ascii=False))
            return response_payload
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} for batch={batch_label}: {body}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(30.0, 1.5 * (2 ** (attempt - 1))))
    raise RuntimeError(str(last_error))


def extract_output_text(response_payload: Dict) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                return content["text"].strip()
    raise ValueError("No output_text found in responses payload.")


def sanitize_item_output(model_output: Dict, context: Dict) -> Dict:
    allowed_relations = set(context["candidate_relations"])
    review_ids = {review["review_id"] for review in context["reviews"]}
    relation_order = {relation_name: index for index, relation_name in enumerate(context["candidate_relations"])}

    kept_relations: List[Dict] = []
    seen = set()
    for relation in model_output.get("kept_relations", []):
        relation_name = compact_spaces(relation.get("relation_name", ""))
        if relation_name not in allowed_relations or relation_name in seen:
            continue
        semantic_text = compact_spaces(relation.get("semantic_text", ""))
        if not semantic_text:
            continue
        evidence = []
        for review_id in relation.get("evidence_review_ids", []):
            review_id = normalize_text(review_id)
            if review_id and review_id in review_ids and review_id not in evidence:
                evidence.append(review_id)
        if not evidence:
            continue
        try:
            confidence = float(relation.get("confidence"))
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, confidence))
        kept_relations.append(
            {
                "relation_name": relation_name,
                "semantic_text": semantic_text,
                "evidence_review_ids": evidence,
                "confidence": round(confidence, 4),
            }
        )
        seen.add(relation_name)

    kept_relations.sort(key=lambda relation: relation_order[relation["relation_name"]])
    return {"item_idx": context["item_idx"], "kept_relations": kept_relations}


def sanitize_batch_output(model_output: Dict, contexts: Sequence[Dict]) -> Dict[int, Dict]:
    context_by_item = {context["item_idx"]: context for context in contexts}
    sanitized: Dict[int, Dict] = {}
    seen = set()
    for item_output in model_output.get("results", []):
        try:
            item_idx = int(item_output.get("item_idx"))
        except (TypeError, ValueError):
            continue
        if item_idx in seen:
            continue
        context = context_by_item.get(item_idx)
        if context is None:
            continue
        sanitized[item_idx] = sanitize_item_output(item_output, context)
        seen.add(item_idx)
    return sanitized


def load_completed_items(path: Path) -> set:
    completed = set()
    for payload in iter_jsonl_records(path):
        item_idx = payload.get("item_idx")
        if item_idx is None:
            continue
        completed.add(int(item_idx))
    return completed


def write_resume_state(path: Path, payload: Dict) -> None:
    state = dict(payload)
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    write_json(path, state)


def main() -> None:
    args = parse_args()
    if args.force and args.output_dir.exists():
        print(f"Removing previous output directory: {args.output_dir}")
        reset_output_dir(args.output_dir)
        if args.output_dir.exists():
            args.output_dir.rmdir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset structure from: {args.dataset_dir}")
    relation_names = load_relation_names(args.dataset_dir)
    items = load_item_records(args.dataset_dir)
    items = maybe_filter_items(items, args.item_idx, args.parent_asin)
    if not items:
        raise SystemExit("No matching items found after applying filters.")

    target_asins = [item["parent_asin"] for item in items]
    meta_cache_path = args.output_dir / "meta_cache.jsonl"
    review_cache_path = args.output_dir / "review_cache_recent_distinct.jsonl"
    prepared_inputs_path = args.output_dir / "prepared_inputs.jsonl"
    results_path = args.output_dir / "relation_semantics.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    summary_path = args.output_dir / "run_summary.json"
    resume_state_path = args.output_dir / "resume_state.json"

    meta_lookup = build_meta_cache(args.meta, target_asins, meta_cache_path)
    review_lookup = build_review_cache(
        args.reviews,
        target_asins,
        review_cache_path,
        top_reviews=args.top_reviews,
        candidate_limit=args.review_candidate_limit,
        min_review_characters=args.min_review_characters,
    )

    contexts = build_item_contexts(items, relation_names, meta_lookup, review_lookup)
    valid_contexts = [
        context
        for context in contexts
        if context["candidate_relations"] and len(context["reviews"]) >= args.top_reviews
    ]
    write_jsonl(prepared_inputs_path, valid_contexts)

    if args.prepare_only:
        prepared_summary = {
            "status": "prepared_only",
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "total_items": len(items),
            "prepared_items": len(valid_contexts),
            "skipped_items": len(items) - len(valid_contexts),
            "prepared_inputs_path": str(prepared_inputs_path),
            "meta_cache_path": str(meta_cache_path),
            "review_cache_path": str(review_cache_path),
        }
        write_json(summary_path, prepared_summary)
        write_resume_state(resume_state_path, prepared_summary)
        print(f"Prepared inputs saved to: {prepared_inputs_path}")
        return

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"Environment variable '{args.api_key_env}' is not set. Refusing to embed secrets in the script."
        )

    completed_items = load_completed_items(results_path)
    initial_completed_count = len(completed_items)
    remaining_contexts = [context for context in valid_contexts if context["item_idx"] not in completed_items]
    if args.limit is not None:
        remaining_contexts = remaining_contexts[: args.limit]

    if not remaining_contexts:
        write_resume_state(
            resume_state_path,
            {
                "status": "completed",
                "dataset_dir": str(args.dataset_dir),
                "output_dir": str(args.output_dir),
                "prepared_items": len(valid_contexts),
                "already_completed_items": initial_completed_count,
                "processed_this_run": 0,
                "errors_this_run": 0,
                "message": "Nothing left to process. Existing outputs already cover the selected items.",
                "results_path": str(results_path),
                "errors_path": str(errors_path),
            },
        )
        print("Nothing left to process. Existing outputs already cover the selected items.")
        return

    write_resume_state(
        resume_state_path,
        {
            "status": "running",
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "batch_size": args.batch_size,
            "top_reviews": args.top_reviews,
            "prepared_items": len(valid_contexts),
            "already_completed_items": initial_completed_count,
            "target_items_this_run": len(remaining_contexts),
            "processed_this_run": 0,
            "errors_this_run": 0,
            "kept_relation_items_this_run": 0,
            "kept_relations_this_run": 0,
            "results_path": str(results_path),
            "errors_path": str(errors_path),
            "prepared_inputs_path": str(prepared_inputs_path),
        },
    )

    batches = list(chunked(remaining_contexts, args.batch_size))
    processed_count = 0
    error_count = 0
    kept_relation_items = 0
    kept_relation_total = 0
    progress = CountProgressBar("llm_enrichment", len(remaining_contexts))
    with results_path.open("a", encoding="utf-8") as results_handle, errors_path.open("a", encoding="utf-8") as errors_handle:
        for batch in batches:
            batch_item_idxs = [context["item_idx"] for context in batch]
            batch_status = "success"
            last_error_message = ""
            try:
                response_payload = call_responses_api(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                    contexts=batch,
                )
                model_text = extract_output_text(response_payload)
                model_output = json.loads(model_text)
                sanitized_batch = sanitize_batch_output(model_output, batch)
                for context in batch:
                    sanitized = sanitized_batch.get(context["item_idx"])
                    if sanitized is None:
                        error_count += 1
                        errors_handle.write(
                            json.dumps(
                                {
                                    "item_idx": context["item_idx"],
                                    "parent_asin": context["parent_asin"],
                                    "error": "Batch response missing this item_idx",
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        continue
                    results_handle.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
                    processed_count += 1
                    kept_relation_total += len(sanitized["kept_relations"])
                    if sanitized["kept_relations"]:
                        kept_relation_items += 1
                flush_and_fsync(results_handle)
                flush_and_fsync(errors_handle)
            except Exception as exc:  # noqa: BLE001
                batch_status = "error"
                last_error_message = str(exc)
                for context in batch:
                    error_count += 1
                    errors_handle.write(
                        json.dumps(
                            {
                                "item_idx": context["item_idx"],
                                "parent_asin": context["parent_asin"],
                                "error": last_error_message,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
                flush_and_fsync(errors_handle)
                flush_and_fsync(results_handle)

            write_resume_state(
                resume_state_path,
                {
                    "status": "running",
                    "dataset_dir": str(args.dataset_dir),
                    "output_dir": str(args.output_dir),
                    "model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "batch_size": args.batch_size,
                    "top_reviews": args.top_reviews,
                    "prepared_items": len(valid_contexts),
                    "already_completed_items": initial_completed_count,
                    "target_items_this_run": len(remaining_contexts),
                    "processed_this_run": processed_count,
                    "errors_this_run": error_count,
                    "completed_items_on_disk_estimate": initial_completed_count + processed_count,
                    "remaining_items_without_results_estimate": max(len(remaining_contexts) - processed_count, 0),
                    "kept_relation_items_this_run": kept_relation_items,
                    "kept_relations_this_run": kept_relation_total,
                    "last_batch_item_idxs": batch_item_idxs,
                    "last_batch_status": batch_status,
                    "last_error": last_error_message,
                    "results_path": str(results_path),
                    "errors_path": str(errors_path),
                    "prepared_inputs_path": str(prepared_inputs_path),
                },
            )
            progress.update(processed_count + error_count, extra=f"kept_items={kept_relation_items} errors={error_count}")
            if args.request_delay_seconds > 0:
                time.sleep(args.request_delay_seconds)
    progress.close()

    final_summary = {
        "dataset_dir": str(args.dataset_dir),
        "reviews_path": str(args.reviews),
        "meta_path": str(args.meta),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "batch_size": args.batch_size,
        "top_reviews": args.top_reviews,
        "total_items": len(items),
        "prepared_items": len(valid_contexts),
        "processed_this_run": processed_count,
        "errors_this_run": error_count,
        "kept_relation_items_this_run": kept_relation_items,
        "kept_relations_this_run": kept_relation_total,
        "prepared_inputs_path": str(prepared_inputs_path),
        "results_path": str(results_path),
        "errors_path": str(errors_path),
        "meta_cache_path": str(meta_cache_path),
        "review_cache_path": str(review_cache_path),
    }
    write_json(summary_path, final_summary)
    write_resume_state(
        resume_state_path,
        {
            **final_summary,
            "status": "completed",
            "already_completed_items": initial_completed_count,
            "completed_items_on_disk_estimate": initial_completed_count + processed_count,
            "remaining_items_without_results_estimate": max(len(remaining_contexts) - processed_count, 0),
        },
    )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
