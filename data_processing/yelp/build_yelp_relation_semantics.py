from __future__ import annotations
import argparse, json, os, shutil, sys, time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = """You generate relation-type semantic descriptions for each business item from exactly three recent reviews written by three different users. For each input item, look only at the provided relation types. Decide whether the reviews explicitly discuss each relation type. If yes, generate one concise English phrase describing how that relation is expressed for the current item. If not, omit that relation. Do not verify KG triples, do not use tail entities, and do not invent new relation types. Keep the semantic text short, content-only, ideally 2-12 words, with no meta wording like 'reviews mention' or 'users say'. Output exactly one result object per input item. If no relation type is clearly discussed, return that item with an empty kept_relations list."""
SCHEMA = {
    "name": "kg_relation_type_semantics_batch",
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
                                    "evidence_review_ids": {"type": "array", "items": {"type": "string"}},
                                    "confidence": {"type": "number"}
                                },
                                "required": ["relation_name", "semantic_text", "evidence_review_ids", "confidence"]
                            }
                        }
                    },
                    "required": ["item_idx", "kept_relations"]
                }
            }
        },
        "required": ["batch_results"]
    }
}
BANNED = (
    "reviews mention", "reviews show", "reviews indicate", "users say",
    "comments indicate", "comments mention", "according to reviews", "this place is"
)

@dataclass(frozen=True)
class Meta:
    item_idx: int
    raw_business_id: str
    name: str
    city: str
    state: str

@dataclass(frozen=True)
class Review:
    review_id: str
    user_key: str
    date: str
    text: str

@dataclass(frozen=True)
class RelationInfo:
    relation_name: str
    description: str


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def args():
    p = argparse.ArgumentParser(description='Generate relation-type semantic descriptions for Yelp items from 3 recent reviews by different users.')
    p.add_argument('--source-dir', type=Path, default=ROOT/'02_processed_data'/'subset_datasets'/'aligned_50k_pos_5core_ratio')
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--base-url', default='https://ai.qaq.al')
    p.add_argument('--api-mode', choices=('responses', 'chat-completions'), default='responses')
    p.add_argument('--model', default='gpt-5.4')
    p.add_argument('--reasoning-effort', default='medium')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--reviews-per-item', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--max-items', type=int, default=None)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--max-output-tokens', type=int, default=12000)
    p.add_argument('--retry-count', type=int, default=5)
    p.add_argument('--retry-sleep', type=float, default=3.0)
    p.add_argument('--relation-names', default='')
    p.add_argument('--keep-output', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reset-output', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--rebuild-cache', action='store_true')
    return p.parse_args()


def ensure_dir(path: Path, keep: bool, resume: bool = False, reset: bool = False):
    if path.exists():
        if reset:
            shutil.rmtree(path)
        elif not (keep or resume):
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_jsonl_dir(path: Path):
    for fp in sorted(path.glob('*.jsonl')):
        with fp.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSON at {fp}:{i}') from e


def rel_filter(raw: str):
    return {x.strip() for x in raw.split(',') if x.strip()} if raw.strip() else set()


def responses_url(base_url: str):
    p = urllib.parse.urlparse(base_url)
    if not p.scheme or not p.netloc:
        raise ValueError(f'Invalid base URL: {base_url}')
    path = p.path.rstrip('/')
    if path.endswith('/responses'):
        final = path
    elif path.endswith('/v1'):
        final = path + '/responses'
    elif path:
        final = path + '/v1/responses'
    else:
        final = '/v1/responses'
    return urllib.parse.urlunparse((p.scheme, p.netloc, final, '', '', ''))


def chat_completions_url(base_url: str):
    p = urllib.parse.urlparse(base_url)
    if not p.scheme or not p.netloc:
        raise ValueError(f'Invalid base URL: {base_url}')
    path = p.path.rstrip('/')
    if path.endswith('/chat/completions'):
        final = path
    elif path.endswith('/v1'):
        final = path + '/chat/completions'
    elif path:
        final = path + '/v1/chat/completions'
    else:
        final = '/v1/chat/completions'
    return urllib.parse.urlunparse((p.scheme, p.netloc, final, '', '', ''))


def chunked(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]
def load_meta(src: Path):
    out = {}
    for r in iter_jsonl_dir(src/'aligned_businesses'):
        idx = int(r['item_idx'])
        out[idx] = Meta(
            item_idx=idx,
            raw_business_id=str(r.get('raw_business_id') or r.get('business_id') or ''),
            name=str(r.get('name') or ''),
            city=str(r.get('city') or ''),
            state=str(r.get('state') or ''),
        )
    return out


def keep_latest_distinct_users(buf: list[Review], cand: Review, k: int):
    rows = buf + [cand]
    rows.sort(key=lambda x: (x.date, x.review_id), reverse=True)
    out, seen_users, seen_reviews = [], set(), set()
    for row in rows:
        if row.review_id in seen_reviews:
            continue
        seen_reviews.add(row.review_id)
        if row.user_key in seen_users:
            continue
        seen_users.add(row.user_key)
        out.append(row)
        if len(out) >= k:
            break
    return out


def load_reviews(src: Path, k: int):
    out = {}
    for r in iter_jsonl_dir(src/'aligned_reviews'):
        text = str(r.get('text') or '').strip()
        if not text:
            continue
        idx = int(r['item_idx'])
        user_key = str(r.get('raw_user_id') or r.get('user_idx') or '')
        row = Review(str(r['review_id']), user_key, str(r.get('date') or ''), text)
        out[idx] = keep_latest_distinct_users(out.get(idx, []), row, k)
    return {idx: rows for idx, rows in out.items() if len(rows) >= k}


def load_relation_catalog(src: Path, relation_filter: set[str]):
    relation_map = src/'mappings'/'relation_map.tsv'
    catalog = []
    if relation_map.exists():
        with relation_map.open('r', encoding='utf-8', newline='') as f:
            header = f.readline().rstrip('\n').split('\t')
            idx = {name:i for i, name in enumerate(header)}
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != len(header):
                    continue
                name = parts[idx['relation_name']]
                head_type = parts[idx['head_type']]
                desc = parts[idx['description']] if 'description' in idx else ''
                if relation_filter and name not in relation_filter:
                    continue
                if not relation_filter and head_type != 'business':
                    continue
                catalog.append(RelationInfo(name, desc))
    else:
        with (src/'relation_list.txt').open('r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                name = raw.split('\t')[0]
                if relation_filter and name not in relation_filter:
                    continue
                catalog.append(RelationInfo(name, ''))
    seen, result = set(), []
    for row in catalog:
        if row.relation_name in seen:
            continue
        seen.add(row.relation_name)
        result.append(row)
    return result


def relation_block(relations: list[RelationInfo]):
    return '\n'.join(f'- {r.relation_name}: {r.description}'.rstrip(': ') for r in relations)


def batch_prompt(batch_rows: list[tuple[Meta, list[Review]]], relations: list[RelationInfo]):
    example = {
        'batch_results': [
            {
                'item_idx': 123,
                'kept_relations': [
                    {
                        'relation_name': 'has_category',
                        'semantic_text': 'all-vegan casual restaurant',
                        'evidence_review_ids': ['review_1', 'review_2'],
                        'confidence': 0.96,
                    }
                ],
            },
            {'item_idx': 124, 'kept_relations': []},
        ]
    }
    blocks = []
    for pos, (meta, reviews) in enumerate(batch_rows, 1):
        rev_block = '\n\n'.join(f'[Review {i} | review_id={rv.review_id}]\n{rv.text}' for i, rv in enumerate(reviews, 1))
        blocks.append(
            f"### Item {pos}\n"
            f"- item_idx: {meta.item_idx}\n"
            f"- raw_business_id: {meta.raw_business_id}\n"
            f"- name: {meta.name}\n"
            f"- city: {meta.city}\n"
            f"- state: {meta.state}\n\n"
            f"Three reviews from different users:\n{rev_block}"
        )
    joined_blocks = "\n\n".join(blocks)
    return (
        "Available relation types for all items:\n"
        f"{relation_block(relations)}\n\n"
        "For each item, decide which relation types are explicitly discussed in the three reviews. "
        "Generate at most one concise semantic description per relation type. "
        "If a relation type is not clearly discussed, omit it.\n\n"
        f"{joined_blocks}\n\n"
        "Return strict JSON only. Example output:\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )


def payload_responses(model: str, effort: str, uprompt: str, max_tokens: int):
    return {
        'model': model,
        'store': False,
        'reasoning': {'effort': effort},
        'max_output_tokens': max_tokens,
        'input': [
            {'role': 'system', 'content': [{'type': 'input_text', 'text': PROMPT}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': uprompt}]},
        ],
        'text': {'format': {'type': 'json_schema', **SCHEMA}},
    }


def payload_chat_completions(model: str, uprompt: str, max_tokens: int):
    # Keep this payload minimal for broad OpenAI-compatible endpoint support.
    return {
        'model': model,
        'messages': [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': uprompt},
        ],
        'max_tokens': max_tokens,
        'temperature': 0.2,
    }


def post(url: str, api_key: str, body: dict, timeout: int):
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))
def output_text(resp: dict):
    # chat-completions compatible output
    if isinstance(resp.get('choices'), list) and resp['choices']:
        first = resp['choices'][0]
        if isinstance(first, dict):
            msg = first.get('message', {})
            if isinstance(msg, dict):
                content = msg.get('content')
                if isinstance(content, str) and content.strip():
                    return content

    if isinstance(resp.get('output_text'), str) and resp['output_text'].strip():
        return resp['output_text']
    chunks = []
    for item in resp.get('output', []) or []:
        if not isinstance(item, dict):
            continue
        for c in item.get('content', []) or []:
            if not isinstance(c, dict):
                continue
            if c.get('type') in {'output_text', 'text'}:
                t = c.get('text')
                if isinstance(t, str):
                    chunks.append(t)
                elif isinstance(t, dict) and isinstance(t.get('value'), str):
                    chunks.append(t['value'])
    return '\n'.join(x for x in chunks if x).strip()


def parse_json_text(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.strip()
        if s.startswith('```'):
            lines = [line for line in s.splitlines() if not line.strip().startswith('```')]
            s = '\n'.join(lines).strip()
        return json.loads(s)


def clean_semantic(text: str):
    s = ' '.join(str(text).replace('\n', ' ').split()).strip().strip('"\'')
    low = s.lower()
    for p in BANNED:
        if low.startswith(p):
            s = s[len(p):].lstrip(' :,-')
            low = s.lower()
    return s


def validate_batch(obj: dict, batch_rows: list[tuple[Meta, list[Review]]], relation_names: set[str]):
    raw = obj.get('batch_results', [])
    if not isinstance(raw, list):
        raise ValueError('batch_results is not a list')
    valid = {meta.item_idx: {r.review_id for r in reviews} for meta, reviews in batch_rows}
    out = {}
    for x in raw:
        if not isinstance(x, dict):
            continue
        try:
            item_idx = int(x.get('item_idx'))
        except (TypeError, ValueError):
            continue
        if item_idx not in valid:
            continue
        rows = x.get('kept_relations', [])
        if not isinstance(rows, list):
            rows = []
        dedup = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            rn = str(r.get('relation_name', '')).strip()
            st = clean_semantic(r.get('semantic_text', ''))
            if not rn or not st or rn not in relation_names:
                continue
            ids = sorted({str(rid) for rid in (r.get('evidence_review_ids', []) or []) if str(rid) in valid[item_idx]})
            if not ids:
                continue
            try:
                cf = float(r.get('confidence', 0.0))
            except (TypeError, ValueError):
                cf = 0.0
            cf = max(0.0, min(1.0, cf))
            row = {'relation_name': rn, 'semantic_text': st, 'evidence_review_ids': ids, 'confidence': cf}
            if rn not in dedup or cf > dedup[rn]['confidence']:
                dedup[rn] = row
        out[item_idx] = {'item_idx': item_idx, 'kept_relations': sorted(dedup.values(), key=lambda z: z['relation_name'])}
    for meta, _ in batch_rows:
        out.setdefault(meta.item_idx, {'item_idx': meta.item_idx, 'kept_relations': []})
    return out


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rebuild_tsv(jsonl_path: Path, tsv_path: Path):
    items = 0
    rels = 0
    with tsv_path.open('w', encoding='utf-8', newline='') as f:
        f.write('item_idx\trelation_name\tsemantic_text\tevidence_review_ids\tconfidence\n')
        if jsonl_path.exists():
            with jsonl_path.open('r', encoding='utf-8') as g:
                for line in g:
                    raw = line.strip()
                    if not raw:
                        continue
                    obj = json.loads(raw)
                    items += 1
                    for r in obj.get('kept_relations', []):
                        f.write(f"{obj['item_idx']}\t{r['relation_name']}\t{r['semantic_text']}\t{'|'.join(r['evidence_review_ids'])}\t{float(r['confidence']):.4f}\n")
                        rels += 1
    return {'items': items, 'semantic_relations': rels}


def request_with_retry(url: str, key: str, body: dict, timeout: int, retries: int, sleep_s: float):
    n = 0
    while True:
        n += 1
        try:
            return post(url, key, body, timeout)
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode('utf-8', errors='replace')
            if n > retries or e.code not in {408, 409, 429, 500, 502, 503, 504, 524}:
                raise RuntimeError(f'HTTP {e.code}: {body_txt}') from e
            time.sleep(sleep_s * n)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if n > retries:
                raise RuntimeError(f'Request failed after retries: {e}') from e
            time.sleep(sleep_s * n)


def load_done_items(path: Path):
    done = set()
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                raw = line.strip()
                if raw:
                    done.add(int(json.loads(raw)['item_idx']))
    return done
def main():
    a = args()
    if a.output_dir is None:
        a.output_dir = a.source_dir/'llm_relation_type_semantics_gpt54_batch10'
    ensure_dir(a.output_dir, a.keep_output, a.resume, a.reset_output)
    if a.api_mode == 'responses':
        api_url = responses_url(a.base_url)
        build_payload = lambda uprompt: payload_responses(a.model, a.reasoning_effort, uprompt, a.max_output_tokens)
    else:
        api_url = chat_completions_url(a.base_url)
        build_payload = lambda uprompt: payload_chat_completions(a.model, uprompt, a.max_output_tokens)
    relation_filter = rel_filter(a.relation_names)
    key = '' if a.dry_run else os.environ.get(a.api_key_env, '')
    if not a.dry_run and not key:
        raise RuntimeError(f'Missing API key in environment variable {a.api_key_env}')

    log('loading business metadata')
    meta = load_meta(a.source_dir)
    log(f'business items={len(meta):,}')

    log('selecting latest reviews from different users')
    reviews = load_reviews(a.source_dir, a.reviews_per_item)
    log(f'items with {a.reviews_per_item} distinct-user reviews={len(reviews):,}')

    relations = load_relation_catalog(a.source_dir, relation_filter)
    relation_names = {r.relation_name for r in relations}
    log(f'loaded relation types={len(relations):,}')

    write_json(a.output_dir/'run_config.json', {
        'source_dir': str(a.source_dir),
        'output_dir': str(a.output_dir),
        'base_url': a.base_url,
        'api_mode': a.api_mode,
        'request_url': api_url,
        'model': a.model,
        'reasoning_effort': a.reasoning_effort,
        'api_key_env': a.api_key_env,
        'reviews_per_item': a.reviews_per_item,
        'batch_size': a.batch_size,
        'max_items': a.max_items,
        'max_output_tokens': a.max_output_tokens,
        'dry_run': a.dry_run,
        'resume': a.resume,
        'reset_output': a.reset_output,
        'relation_names': sorted(relation_names),
        'prompt': PROMPT,
    })

    res_path = a.output_dir/'item_relation_semantics.jsonl'
    err_path = a.output_dir/'errors.jsonl'
    preview = a.output_dir/'dry_run_preview.json'
    batch_status = a.output_dir/'batch_status.jsonl'

    done = load_done_items(res_path)
    if done:
        log(f'resume detected, already finished items={len(done):,}')

    items = sorted(set(meta) & set(reviews))
    if a.max_items is not None:
        items = items[:a.max_items]
    todo = [i for i in items if i not in done]
    batches = list(chunked(todo, a.batch_size))
    log(f'candidate items={len(items):,} pending={len(todo):,} batches={len(batches):,} batch_size={a.batch_size}')

    processed = 0
    emitted = 0
    errors = 0
    done_batches = 0
    for bno, item_ids in enumerate(batches, 1):
        batch_rows = [(meta[item_idx], reviews[item_idx]) for item_idx in item_ids]
        up = batch_prompt(batch_rows, relations)
        body = build_payload(up)
        if a.dry_run:
            write_json(preview, {
                'batch_number': bno,
                'batch_size': len(batch_rows),
                'relation_catalog': [r.__dict__ for r in relations],
                'items': [
                    {
                        'item_idx': m.item_idx,
                        'raw_business_id': m.raw_business_id,
                        'name': m.name,
                        'city': m.city,
                        'state': m.state,
                        'selected_reviews': [r.__dict__ for r in rs],
                    }
                    for m, rs in batch_rows
                ],
                'request_url': api_url,
                'request_payload': body,
            })
            log(f'dry-run preview written to {preview}')
            return
        try:
            resp = request_with_retry(api_url, key, body, a.timeout, a.retry_count, a.retry_sleep)
            parsed = parse_json_text(output_text(resp))
            norm = validate_batch(parsed, batch_rows, relation_names)
            for m, rs in batch_rows:
                row = norm[m.item_idx]
                append_jsonl(res_path, {
                    'item_idx': m.item_idx,
                    'raw_business_id': m.raw_business_id,
                    'name': m.name,
                    'city': m.city,
                    'state': m.state,
                    'selected_review_ids': [r.review_id for r in rs],
                    'relation_type_count': len(relations),
                    'kept_relations': row['kept_relations'],
                    'model': a.model,
                    'reasoning_effort': a.reasoning_effort,
                    'request_url': api_url,
                    'response_id': resp.get('id'),
                    'batch_number': bno,
                    'batch_size': len(batch_rows),
                })
                processed += 1
                emitted += len(row['kept_relations'])
            done_batches += 1
            append_jsonl(batch_status, {'batch_number': bno, 'status': 'success', 'batch_size': len(batch_rows), 'item_ids': [m.item_idx for m, _ in batch_rows], 'response_id': resp.get('id')})
        except Exception as e:
            errors += len(batch_rows)
            append_jsonl(batch_status, {'batch_number': bno, 'status': 'error', 'batch_size': len(batch_rows), 'item_ids': [m.item_idx for m, _ in batch_rows], 'error': str(e)})
            append_jsonl(err_path, {'batch_number': bno, 'batch_size': len(batch_rows), 'item_ids': [m.item_idx for m, _ in batch_rows], 'raw_business_ids': [m.raw_business_id for m, _ in batch_rows], 'error': str(e)})
        log(f'progress batch {bno:,}/{len(batches):,} | items_done={processed:,} relations={emitted:,} errors={errors:,}')

    stats = rebuild_tsv(res_path, a.output_dir/'item_relation_semantics.tsv')
    write_json(a.output_dir/'run_summary.json', {
        'source_dir': str(a.source_dir),
        'output_dir': str(a.output_dir),
        'request_url': api_url,
        'model': a.model,
        'reasoning_effort': a.reasoning_effort,
        'candidate_items': len(items),
        'pending_items': len(todo),
        'batch_size': a.batch_size,
        'total_batches': len(batches),
        'processed_batches': done_batches,
        'processed_now': processed,
        'error_count': errors,
        'items_with_outputs': stats['items'],
        'semantic_relations': stats['semantic_relations'],
        'reviews_per_item': a.reviews_per_item,
        'relation_type_count': len(relations),
        'relation_names': [r.relation_name for r in relations],
        'notes': 'This output is relation-type level. It does not validate or reference concrete KG tail entities.'
    })
    log(f"finished | items_with_outputs={stats['items']:,} semantic_relations={stats['semantic_relations']:,} errors={errors:,}")

if __name__ == '__main__':
    main()
