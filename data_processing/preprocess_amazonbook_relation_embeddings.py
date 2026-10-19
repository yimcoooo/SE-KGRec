#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate relation-level semantic embeddings for the amazonbook_200k dataset.

Processing rules:
1. Use only semantic_text with confidence > threshold (default threshold = 0.9).
2. Emit one item_idx-aligned embedding tensor per relation_name.
3. The overall embedding is the mean of all high-confidence relation embeddings of each item.
4. Encode the raw semantic_text only (no template prefix) to reduce matching noise.

Default input:
- amazonbook_200k/llm_embedding/relation_semantics.jsonl
- sup-simcse-roberta-large

Default output:
- amazonbook_200k/llm_embedding/relation_embeddings
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "amazonbook_200k"
DEFAULT_LLM_DIR = DEFAULT_DATASET_DIR / "llm_embedding"
DEFAULT_INPUT_PATH = DEFAULT_LLM_DIR / "relation_semantics.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_LLM_DIR / "relation_embeddings"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "sup-simcse-roberta-large"
DEFAULT_THRESHOLD = 0.9


@dataclass
class ParseStats:
    total_records: int = 0
    total_relations: int = 0
    kept_relations: int = 0
    dropped_below_or_equal_threshold: int = 0
    dropped_missing_relation_name: int = 0
    dropped_missing_semantic_text: int = 0
    duplicate_item_relation_pairs: int = 0
    items_with_any_kept_relation: int = 0
    items_without_kept_relation: int = 0


@dataclass
class EncodingStats:
    max_item_id: int = 0
    item_vocab_size: int = 0
    relation_count_from_list: int = 0
    relation_count_from_jsonl: int = 0
    relation_count_output: int = 0
    overall_item_count: int = 0


def clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sanitize_filename(name: str) -> str:
    cleaned = clean_text(name)
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or "unknown_relation"


def parse_confidence(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_max_item_id(path: Path) -> int:
    max_item_id = -1
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            remap_id = int(parts[1])
            max_item_id = max(max_item_id, remap_id)
    if max_item_id < 0:
        raise ValueError(f"No valid item ids found in {path}")
    return max_item_id


def load_relation_order(path: Path) -> List[str]:
    relation_names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 1:
                continue
            relation_name = clean_text(parts[0])
            if relation_name:
                relation_names.append(relation_name)
    return relation_names


def should_replace(old_entry: dict, new_entry: dict) -> bool:
    old_conf = parse_confidence(old_entry.get("confidence"))
    new_conf = parse_confidence(new_entry.get("confidence"))
    if new_conf != old_conf:
        return new_conf > old_conf
    return len(clean_text(new_entry.get("semantic_text"))) > len(clean_text(old_entry.get("semantic_text")))


def load_relation_semantics(path: Path, threshold: float) -> Tuple[Dict[int, Dict[str, dict]], ParseStats]:
    kept_by_item: Dict[int, Dict[str, dict]] = {}
    stats = ParseStats()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stats.total_records += 1

            item_idx = int(obj["item_idx"])
            relation_map = kept_by_item.setdefault(item_idx, {})
            kept_before = len(relation_map)

            for relation in obj.get("kept_relations", []):
                stats.total_relations += 1
                relation_name = clean_text(relation.get("relation_name"))
                semantic_text = clean_text(relation.get("semantic_text"))
                confidence = parse_confidence(relation.get("confidence"))

                if not relation_name:
                    stats.dropped_missing_relation_name += 1
                    continue
                if not semantic_text:
                    stats.dropped_missing_semantic_text += 1
                    continue
                if confidence <= threshold:
                    stats.dropped_below_or_equal_threshold += 1
                    continue

                candidate = {
                    "relation_name": relation_name,
                    "semantic_text": semantic_text,
                    "confidence": confidence,
                    "evidence_review_ids": relation.get("evidence_review_ids") or [],
                }
                existing = relation_map.get(relation_name)
                if existing is not None:
                    stats.duplicate_item_relation_pairs += 1
                    if should_replace(existing, candidate):
                        relation_map[relation_name] = candidate
                else:
                    relation_map[relation_name] = candidate
                    stats.kept_relations += 1

            if len(relation_map) > kept_before:
                stats.items_with_any_kept_relation += 1
            elif not relation_map:
                stats.items_without_kept_relation += 1

    return kept_by_item, stats


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower().strip()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


def load_model(model_path: Path, device: torch.device, use_fp16: bool) -> Tuple[AutoTokenizer, AutoModel, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if device.type == "cuda" and use_fp16:
        model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Could not read hidden_size from the model configuration.")
    return tokenizer, model, hidden_size


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


def encode_text_map(
    text_map: Dict[int, str],
    vocab_size: int,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    normalize: bool,
    hidden_size: int,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
    encoded_mask = torch.zeros(vocab_size, dtype=torch.bool)
    ids = sorted(idx for idx, text in text_map.items() if 0 <= idx < vocab_size and clean_text(text))

    if not ids:
        return embeddings, encoded_mask

    for start in tqdm(range(0, len(ids), batch_size), desc=desc):
        batch_ids = ids[start : start + batch_size]
        batch_texts = [text_map[idx] for idx in batch_ids]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True)
            batch_embeddings = getattr(outputs, "pooler_output", None)
            if batch_embeddings is None:
                batch_embeddings = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if normalize:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            batch_embeddings = batch_embeddings.detach().float().cpu()

        embeddings[batch_ids] = batch_embeddings
        encoded_mask[batch_ids] = True

    return embeddings, encoded_mask


def build_output_relation_order(relation_names_from_list: List[str], semantics: Dict[int, Dict[str, dict]]) -> List[str]:
    relation_set_from_jsonl = {relation_name for item_relations in semantics.values() for relation_name in item_relations.keys()}
    extras = sorted(relation_set_from_jsonl.difference(relation_names_from_list))
    return list(relation_names_from_list) + extras


def build_metadata(
    dataset_dir: Path,
    input_jsonl: Path,
    output_dir: Path,
    model_path: Path,
    device: torch.device,
    hidden_size: int,
    threshold: float,
    normalize: bool,
    batch_size: int,
    max_length: int,
    relation_names_from_list: List[str],
    relation_order: List[str],
    relation_item_counts: Dict[str, int],
    parse_stats: ParseStats,
    encoding_stats: EncodingStats,
    relation_files: Dict[str, str],
) -> dict:
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_dir": str(dataset_dir),
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "device": str(device),
        "hidden_size": hidden_size,
        "confidence_threshold": float(threshold),
        "threshold_rule": "confidence > threshold",
        "normalize": bool(normalize),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "relation_names_from_relation_list": relation_names_from_list,
        "relation_order": relation_order,
        "relation_item_counts": relation_item_counts,
        "parse_stats": asdict(parse_stats),
        "encoding_stats": asdict(encoding_stats),
        "files": {
            "relation_embeddings": relation_files,
            "item_overall_embeddings": str(output_dir / "item_overall_embeddings.pt"),
            "item_relation_availability": str(output_dir / "item_relation_availability.pt"),
            "embedding_metadata": str(output_dir / "embedding_metadata.json"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate relation-aware semantic embeddings for SE-KGRec/amazonbook_200k.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--normalize", action="store_true", default=True, help="Apply L2 normalisation before saving.")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable half-precision inference on CUDA (default).")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.dataset_dir.resolve()
    input_jsonl = args.input_jsonl.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    item_list_path = dataset_dir / "item_list.txt"
    relation_list_path = dataset_dir / "relation_list.txt"

    for required_path in [dataset_dir, input_jsonl, model_path, item_list_path, relation_list_path]:
        if not required_path.exists():
            raise FileNotFoundError(str(required_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[1/6] loading item/relation metadata from {dataset_dir}")
    max_item_id = load_max_item_id(item_list_path)
    relation_names_from_list = load_relation_order(relation_list_path)
    item_vocab_size = max_item_id + 1

    print(f"[2/6] loading relation semantics from {input_jsonl}")
    semantics, parse_stats = load_relation_semantics(input_jsonl, threshold=float(args.confidence_threshold))
    relation_order = build_output_relation_order(relation_names_from_list, semantics)

    print(f"[3/6] loading model from {model_path} on {device}")
    tokenizer, model, hidden_size = load_model(model_path, device=device, use_fp16=args.fp16)

    print("[4/6] encoding per-relation semantic texts")
    relation_availability = torch.zeros((item_vocab_size, len(relation_order)), dtype=torch.bool)
    overall_sum = torch.zeros((item_vocab_size, hidden_size), dtype=torch.float32)
    overall_counts = torch.zeros(item_vocab_size, dtype=torch.float32)
    relation_files: Dict[str, str] = {}
    relation_item_counts: Dict[str, int] = {}

    for relation_idx, relation_name in enumerate(relation_order):
        text_map = {
            item_idx: relation_info[relation_name]["semantic_text"]
            for item_idx, relation_info in semantics.items()
            if relation_name in relation_info
        }
        relation_embeddings, relation_mask = encode_text_map(
            text_map=text_map,
            vocab_size=item_vocab_size,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            normalize=args.normalize,
            hidden_size=hidden_size,
            desc=f"relation:{relation_name}",
        )

        relation_availability[:, relation_idx] = relation_mask
        overall_sum += relation_embeddings
        overall_counts += relation_mask.to(torch.float32)
        relation_item_counts[relation_name] = int(relation_mask.sum().item())

        relation_file_path = output_dir / f"item_{sanitize_filename(relation_name)}_embeddings.pt"
        torch.save(relation_embeddings, relation_file_path)
        relation_files[relation_name] = str(relation_file_path)

    print("[5/6] building overall item embeddings")
    overall_embeddings = torch.zeros((item_vocab_size, hidden_size), dtype=torch.float32)
    valid_rows = overall_counts > 0
    overall_embeddings[valid_rows] = overall_sum[valid_rows] / overall_counts[valid_rows].unsqueeze(-1)
    if args.normalize and valid_rows.any():
        overall_embeddings[valid_rows] = F.normalize(overall_embeddings[valid_rows], p=2, dim=-1)

    print(f"[6/6] saving tensors to {output_dir}")
    torch.save(overall_embeddings, output_dir / "item_overall_embeddings.pt")
    torch.save(relation_availability, output_dir / "item_relation_availability.pt")

    encoding_stats = EncodingStats(
        max_item_id=max_item_id,
        item_vocab_size=item_vocab_size,
        relation_count_from_list=len(relation_names_from_list),
        relation_count_from_jsonl=len({relation for relations in semantics.values() for relation in relations.keys()}),
        relation_count_output=len(relation_order),
        overall_item_count=int(valid_rows.sum().item()),
    )
    metadata = build_metadata(
        dataset_dir=dataset_dir,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        model_path=model_path,
        device=device,
        hidden_size=hidden_size,
        threshold=float(args.confidence_threshold),
        normalize=bool(args.normalize),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        relation_names_from_list=relation_names_from_list,
        relation_order=relation_order,
        relation_item_counts=relation_item_counts,
        parse_stats=parse_stats,
        encoding_stats=encoding_stats,
        relation_files=relation_files,
    )
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"hidden_size={hidden_size}")
    print(f"item_vocab_size={item_vocab_size}")
    print(f"relation_count={len(relation_order)}")
    print(f"overall_item_count={int(valid_rows.sum().item())}")


if __name__ == "__main__":
    main()
