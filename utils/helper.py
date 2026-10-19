import os
import re
from typing import Any
from math import log
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader, random_split


_KG_COMPLETION_COMPONENTS = ("npmi", "semantic", "cosine", "fusion", "x_style", "x_sim", "x_overall")
_X_COMPLETION_COMPONENTS = ("x_style", "x_sim", "x_overall")
_KG_COMPLETION_MODE_ORDER = ("npmi", "semantic", "cosine", "fusion", "x_style", "x_sim", "x_overall")



def _has_item_embeds(item_embeds):
    if item_embeds is None:
        return False
    if isinstance(item_embeds, list):
        return len(item_embeds) > 0
    item_embeds = np.asarray(item_embeds)
    return item_embeds.size > 0



def _has_relation_embeddings(relation_item_embeds, relation_order=None):
    if relation_item_embeds is None:
        return False
    if isinstance(relation_item_embeds, dict):
        if len(relation_item_embeds) == 0:
            return False
        if relation_order is None:
            return True
        return len(relation_order) > 0
    if isinstance(relation_item_embeds, (list, tuple)):
        return len(relation_item_embeds) > 0
    relation_item_embeds = np.asarray(relation_item_embeds)
    return relation_item_embeds.size > 0



def _normalize_relation_order(relation_item_embeds, relation_order=None):
    if relation_order is not None:
        relation_order = [str(name) for name in relation_order if str(name).strip() != ""]
        if len(relation_order) > 0:
            return relation_order
    if isinstance(relation_item_embeds, dict):
        return [str(name) for name in relation_item_embeds.keys()]
    if isinstance(relation_item_embeds, (list, tuple)):
        return [str(idx) for idx in range(len(relation_item_embeds))]
    relation_item_embeds = np.asarray(relation_item_embeds)
    if relation_item_embeds.ndim >= 3:
        return [str(idx) for idx in range(relation_item_embeds.shape[0])]
    return []



def normalize_kg_completion_mode(mode):
    if mode is None:
        mode = "npmi+cosine"

    raw_parts = [part.strip().lower() for part in str(mode).split("+") if part.strip()]
    if len(raw_parts) == 0:
        raise ValueError("kg_completion_mode can not be empty")

    alias_map = {
        "cos": "cosine",
        "graph": "cosine",
        "style": "x_style",
        "similar": "x_sim",
        "overall": "x_overall",
        "combined": "x_overall",
    }

    component_set = set()
    for part in raw_parts:
        normalized_part = alias_map.get(part, part)
        if normalized_part not in _KG_COMPLETION_COMPONENTS:
            raise ValueError(
                "invalid kg_completion_mode component: {} (supported: {})".format(
                    part, "+".join(_KG_COMPLETION_COMPONENTS)
                )
            )
        component_set.add(normalized_part)

    if ("x_style" in component_set and "x_sim" in component_set) or ("fusion" in component_set):
        component_set.add("fusion")

    normalized_mode_components = []
    for component in _KG_COMPLETION_MODE_ORDER:
        if component not in component_set:
            continue
        if component in _X_COMPLETION_COMPONENTS and "fusion" in component_set:
            continue
        normalized_mode_components.append(component)

    normalized_mode = "+".join(normalized_mode_components)
    return normalized_mode, component_set


def _ordered_x_components(completion_components):
    explicit_x_components = [component for component in _X_COMPLETION_COMPONENTS if component in completion_components]
    if len(explicit_x_components) > 0:
        return explicit_x_components
    if "fusion" in completion_components:
        return ["x_style", "x_sim"]
    return []


def _resolve_x_branch_config(
    x_component,
    style_item_embeds,
    similar_item_embeds,
    overall_item_embeds,
    style_item_availability,
    similar_item_availability,
    overall_item_availability,
    style_struct_threshold,
    style_semantic_threshold,
    similar_struct_threshold,
    similar_semantic_threshold,
    overall_struct_threshold,
    overall_semantic_threshold,
):
    if x_component == "x_style":
        return {
            "component": "x_style",
            "embed_name": "style item",
            "semantic_item_embeds": style_item_embeds,
            "semantic_item_availability": style_item_availability,
            "availability_name": "style availability",
            "struct_threshold": style_struct_threshold,
            "semantic_threshold": style_semantic_threshold,
        }
    if x_component == "x_sim":
        return {
            "component": "x_sim",
            "embed_name": "similar item",
            "semantic_item_embeds": similar_item_embeds,
            "semantic_item_availability": similar_item_availability,
            "availability_name": "similar availability",
            "struct_threshold": similar_struct_threshold,
            "semantic_threshold": similar_semantic_threshold,
        }
    if x_component == "x_overall":
        return {
            "component": "x_overall",
            "embed_name": "overall item",
            "semantic_item_embeds": overall_item_embeds,
            "semantic_item_availability": overall_item_availability,
            "availability_name": "overall availability",
            "struct_threshold": overall_struct_threshold,
            "semantic_threshold": overall_semantic_threshold,
        }
    raise ValueError("unsupported x component: {}".format(x_component))


def _build_empty_sparse_mat(n_items):
    return sp.coo_matrix((n_items, n_items), dtype=np.float32)


def _limit_sparse_neighbors_topk(sparse_mat, topk):
    if topk is None or int(topk) <= 0:
        return sparse_mat.tocoo()

    sparse_mat = sparse_mat.tocsr()
    n_rows = sparse_mat.shape[0]
    rows = []
    cols = []
    data = []
    topk = int(topk)

    for row_idx in range(n_rows):
        start = sparse_mat.indptr[row_idx]
        end = sparse_mat.indptr[row_idx + 1]
        row_cols = sparse_mat.indices[start:end]
        row_data = sparse_mat.data[start:end]
        if row_data.size == 0:
            continue
        if row_data.size > topk:
            top_idx = np.argpartition(-row_data, topk - 1)[:topk]
            order = np.argsort(-row_data[top_idx])
            top_idx = top_idx[order]
            row_cols = row_cols[top_idx]
            row_data = row_data[top_idx]
        rows.append(np.full(row_data.shape[0], row_idx, dtype=np.int64))
        cols.append(row_cols.astype(np.int64, copy=False))
        data.append(row_data.astype(np.float32, copy=False))

    if len(data) == 0:
        return _build_empty_sparse_mat(sparse_mat.shape[0])

    return sp.coo_matrix(
        (
            np.concatenate(data, axis=0),
            (np.concatenate(rows, axis=0), np.concatenate(cols, axis=0)),
        ),
        shape=sparse_mat.shape,
        dtype=np.float32,
    )


def _align_embedding_rows(item_embeds, n_items, embed_name):
    if not _has_item_embeds(item_embeds):
        return None

    item_embeds = np.asarray(item_embeds, dtype=np.float32)
    if item_embeds.ndim != 2:
        print("warning: {} embeddings should be a 2D matrix, skip this branch".format(embed_name))
        return None

    cur_rows, embed_dim = item_embeds.shape
    if cur_rows > n_items:
        print("warning: {} embedding rows {} > n_items {}, truncate to n_items".format(embed_name, cur_rows, n_items))
        item_embeds = item_embeds[:n_items]
    elif cur_rows < n_items:
        print("warning: {} embedding rows {} < n_items {}, pad zeros to n_items".format(embed_name, cur_rows, n_items))
        pad_mat = np.zeros((n_items - cur_rows, embed_dim), dtype=np.float32)
        item_embeds = np.concatenate([item_embeds, pad_mat], axis=0)

    return item_embeds.astype(np.float32, copy=False)


def _align_item_availability(item_availability, n_items, availability_name):
    if item_availability is None:
        return None
    if isinstance(item_availability, list):
        if len(item_availability) == 0:
            return None
        item_availability = np.asarray(item_availability)

    item_availability = np.asarray(item_availability).reshape(-1)
    if item_availability.size == 0:
        return None

    cur_rows = item_availability.shape[0]
    if cur_rows > n_items:
        print("warning: {} rows {} > n_items {}, truncate to n_items".format(availability_name, cur_rows, n_items))
        item_availability = item_availability[:n_items]
    elif cur_rows < n_items:
        print("warning: {} rows {} < n_items {}, pad False to n_items".format(availability_name, cur_rows, n_items))
        pad_mask = np.zeros((n_items - cur_rows,), dtype=bool)
        item_availability = np.concatenate([item_availability.astype(bool, copy=False), pad_mask], axis=0)

    return item_availability.astype(bool, copy=False)


def _filter_sparse_mat_by_item_availability(sparse_mat, item_availability, availability_name="item availability"):
    sparse_mat = sparse_mat.tocoo()
    if sparse_mat.nnz == 0:
        return sparse_mat

    item_availability = _align_item_availability(item_availability, sparse_mat.shape[0], availability_name)
    if item_availability is None:
        return sparse_mat

    valid_mask = item_availability[sparse_mat.row] & item_availability[sparse_mat.col]
    if not np.any(valid_mask):
        return _build_empty_sparse_mat(sparse_mat.shape[0]).tocoo()

    return sp.coo_matrix(
        (
            sparse_mat.data[valid_mask].astype(np.float32, copy=False),
            (
                sparse_mat.row[valid_mask].astype(np.int64, copy=False),
                sparse_mat.col[valid_mask].astype(np.int64, copy=False),
            ),
        ),
        shape=sparse_mat.shape,
        dtype=np.float32,
    )


def _count_available_items(item_availability, n_items, availability_name):
    item_availability = _align_item_availability(item_availability, n_items, availability_name)
    if item_availability is None:
        return 0
    return int(np.sum(item_availability))


def _build_npmi_sparse_mat(pmi_dict, n_items, pmi_threshold):
    rows = []
    cols = []
    data = []
    for item_pair in tqdm(pmi_dict.keys()):
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        npmi = pmi_dict[item_pair][1]
        if npmi >= pmi_threshold:
            rows.append(item_i)
            cols.append(item_j)
            data.append(npmi)
    if len(data) == 0:
        return sp.coo_matrix((n_items, n_items), dtype=np.float32)
    return sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


def _build_cosine_sparse_mat(item_embeds, cos_threshold, chunk_size=256):
    item_embeds = np.asarray(item_embeds, dtype=np.float32)
    if item_embeds.ndim != 2 or item_embeds.shape[0] == 0:
        return sp.coo_matrix((0, 0), dtype=np.float32)

    item_norm = np.linalg.norm(item_embeds, axis=1, keepdims=True)
    item_norm = np.clip(item_norm, a_min=1e-12, a_max=None)
    norm_embeds = item_embeds / item_norm

    rows = []
    cols = []
    data = []
    n_items = norm_embeds.shape[0]
    chunk_size = max(1, min(int(chunk_size), n_items))

    for start in tqdm(range(0, n_items, chunk_size), desc="calc cosine"):
        end = min(start + chunk_size, n_items)
        sim_block = np.matmul(norm_embeds[start:end], norm_embeds.T)
        block_mask = sim_block >= cos_threshold
        if not np.any(block_mask):
            continue
        block_rows, block_cols = np.where(block_mask)
        rows.append(block_rows + start)
        cols.append(block_cols)
        data.append(sim_block[block_mask].astype(np.float32, copy=False))

    if len(data) == 0:
        return sp.coo_matrix((n_items, n_items), dtype=np.float32)

    return sp.coo_matrix(
        (
            np.concatenate(data, axis=0),
            (np.concatenate(rows, axis=0), np.concatenate(cols, axis=0)),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


def _build_thresholded_cosine_sparse_mat(
    item_embeds,
    n_items,
    cos_threshold,
    chunk_size=256,
    embed_name="item",
    item_availability=None,
    availability_name="item availability",
):
    aligned_embeds = _align_embedding_rows(item_embeds, n_items, embed_name)
    if aligned_embeds is None:
        return _build_empty_sparse_mat(n_items)
    sparse_mat = _build_cosine_sparse_mat(aligned_embeds, cos_threshold, chunk_size=chunk_size)
    return _filter_sparse_mat_by_item_availability(
        sparse_mat,
        item_availability=item_availability,
        availability_name=availability_name,
    )


def _fuse_sparse_similarity_mats(
    struct_sparse_mat,
    semantic_sparse_mat,
    n_items,
    semantic_available_items=0,
    fusion_alpha=0.7,
):
    if struct_sparse_mat is None:
        struct_sparse_mat = _build_empty_sparse_mat(n_items)
    if semantic_sparse_mat is None:
        semantic_sparse_mat = _build_empty_sparse_mat(n_items)

    struct_sparse_mat = struct_sparse_mat.tocsr()
    semantic_sparse_mat = semantic_sparse_mat.tocsr()
    # When semantic similarity is missing for an edge, fall back to full structural score (weight=1.0).
    semantic_indicator = semantic_sparse_mat.copy()
    semantic_indicator.data = np.ones_like(semantic_indicator.data, dtype=np.float32)
    struct_overlap = struct_sparse_mat.multiply(semantic_indicator)
    struct_only = struct_sparse_mat - struct_overlap
    fused_sum = (
        struct_overlap.multiply(np.float32(fusion_alpha))
        + semantic_sparse_mat.multiply(np.float32(1.0 - fusion_alpha))
        + struct_only
    )
    fused_sum = fused_sum.tocoo()
    if fused_sum.nnz == 0:
        return _build_empty_sparse_mat(n_items).tocoo(), {
            "struct_pairs": int(struct_sparse_mat.nnz),
            "semantic_pairs": int(semantic_sparse_mat.nnz),
            "fused_pairs": 0,
            "x_pairs": 0,
            "semantic_available_items": int(semantic_available_items),
            "fusion_alpha": float(fusion_alpha),
        }

    fused_data = np.clip(fused_sum.data.astype(np.float32, copy=False), a_min=0.0, a_max=None)
    valid_mask = (fused_data > 0) & (fused_sum.row != fused_sum.col)
    if not np.any(valid_mask):
        return _build_empty_sparse_mat(n_items).tocoo(), {
            "struct_pairs": int(struct_sparse_mat.nnz),
            "semantic_pairs": int(semantic_sparse_mat.nnz),
            "fused_pairs": 0,
            "x_pairs": 0,
            "semantic_available_items": int(semantic_available_items),
            "fusion_alpha": float(fusion_alpha),
        }

    fused_mat = sp.coo_matrix(
        (
            fused_data[valid_mask],
            (
                fused_sum.row[valid_mask].astype(np.int64, copy=False),
                fused_sum.col[valid_mask].astype(np.int64, copy=False),
            ),
        ),
        shape=(n_items, n_items),
        dtype=np.float32,
    )
    return fused_mat, {
        "struct_pairs": int(struct_sparse_mat.nnz),
        "semantic_pairs": int(semantic_sparse_mat.nnz),
        "fused_pairs": int(fused_mat.nnz),
        "x_pairs": int(fused_mat.nnz),
        "semantic_available_items": int(semantic_available_items),
        "fusion_alpha": float(fusion_alpha),
    }



def _build_x_sparse_mat(
    item_embeds,
    semantic_item_embeds,
    semantic_item_availability,
    n_items,
    struct_threshold,
    semantic_threshold,
    chunk_size=256,
    semantic_embed_name="semantic item",
    fusion_alpha=0.7,
):
    semantic_item_embeds = _align_embedding_rows(semantic_item_embeds, n_items, semantic_embed_name)
    if semantic_item_embeds is None:
        raise ValueError("x fusion is enabled, but {} embeddings are unavailable".format(semantic_embed_name))

    graph_item_embeds = _align_embedding_rows(item_embeds, n_items, "graph item")
    semantic_available_items = _count_available_items(
        semantic_item_availability,
        n_items,
        "{} availability".format(semantic_embed_name),
    )
    if graph_item_embeds is None:
        print("skip fusion branch: graph item embeddings unavailable")
        return _build_empty_sparse_mat(n_items).tocoo(), {
            "struct_pairs": 0,
            "semantic_pairs": 0,
            "fused_pairs": 0,
            "x_pairs": 0,
            "semantic_available_items": int(semantic_available_items),
            "fusion_alpha": float(fusion_alpha),
        }

    struct_sparse_mat = _build_cosine_sparse_mat(graph_item_embeds, struct_threshold, chunk_size=chunk_size).tocoo()
    semantic_sparse_mat = _build_thresholded_cosine_sparse_mat(
        semantic_item_embeds,
        n_items,
        semantic_threshold,
        chunk_size=chunk_size,
        embed_name=semantic_embed_name,
        item_availability=semantic_item_availability,
        availability_name="{} availability".format(semantic_embed_name),
    ).tocoo()
    return _fuse_sparse_similarity_mats(
        struct_sparse_mat=struct_sparse_mat,
        semantic_sparse_mat=semantic_sparse_mat,
        n_items=n_items,
        semantic_available_items=semantic_available_items,
        fusion_alpha=fusion_alpha,
    )


def _build_dense_npmi_mat(pmi_dict, n_items, pmi_threshold=None):
    npmi_mat = np.zeros((n_items, n_items), dtype=np.float32)
    pair_count = 0
    if pmi_dict is None:
        return npmi_mat, pair_count

    for item_pair, stat_list in tqdm(pmi_dict.items(), desc="build npmi mat"):
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        npmi = float(stat_list[1])
        if (pmi_threshold is None) or (npmi >= float(pmi_threshold)):
            npmi_mat[item_i, item_j] = np.float32(npmi)
            pair_count += 1
    return npmi_mat, pair_count


def _build_dense_cosine_mat(item_embeds, n_items, chunk_size=256, embed_name="dense cosine"):
    item_embeds = _align_embedding_rows(item_embeds, n_items, embed_name)
    if item_embeds is None:
        return np.zeros((n_items, n_items), dtype=np.float32)

    item_norm = np.linalg.norm(item_embeds, axis=1, keepdims=True)
    item_norm = np.clip(item_norm, a_min=1e-12, a_max=None)
    norm_embeds = item_embeds / item_norm

    cosine_mat = np.zeros((n_items, n_items), dtype=np.float32)
    chunk_size = max(1, min(int(chunk_size), n_items))
    for start in tqdm(range(0, n_items, chunk_size), desc="calc dense cosine"):
        end = min(start + chunk_size, n_items)
        cosine_mat[start:end] = np.matmul(norm_embeds[start:end], norm_embeds.T).astype(np.float32, copy=False)
    return cosine_mat


def _build_relation_specific_score_mat(
    npmi_mat,
    struct_cos_mat,
    semantic_cos_mat,
    struct_threshold,
    semantic_threshold,
    lambda_weight,
):
    npmi_mat = np.asarray(npmi_mat, dtype=np.float32)
    struct_cos_mat = np.asarray(struct_cos_mat, dtype=np.float32)
    semantic_cos_mat = np.asarray(semantic_cos_mat, dtype=np.float32)

    valid_mask = (
        (struct_cos_mat >= float(struct_threshold))
        & (semantic_cos_mat >= float(semantic_threshold))
        & (struct_cos_mat > 0)
        & (semantic_cos_mat > 0)
    )
    x_rel = np.zeros_like(struct_cos_mat, dtype=np.float32)
    x_rel[valid_mask] = np.sqrt(struct_cos_mat[valid_mask] * semantic_cos_mat[valid_mask]).astype(np.float32, copy=False)

    score_mat = np.float32(lambda_weight) * npmi_mat + np.float32(1.0 - lambda_weight) * x_rel
    score_mat[~valid_mask] = 0.0
    np.fill_diagonal(x_rel, 0.0)
    np.fill_diagonal(score_mat, 0.0)
    return score_mat.astype(np.float32, copy=False), x_rel.astype(np.float32, copy=False), valid_mask


def _topk_indices_from_scores(score_row, topk):
    score_row = np.asarray(score_row, dtype=np.float32)
    positive_idx = np.where(score_row > 0)[0]
    if positive_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    if topk is None or int(topk) <= 0 or positive_idx.size <= int(topk):
        return positive_idx[np.argsort(-score_row[positive_idx])].astype(np.int64, copy=False)

    topk = int(topk)
    local_idx = np.argpartition(-score_row[positive_idx], topk - 1)[:topk]
    ranked_local_idx = local_idx[np.argsort(-score_row[positive_idx][local_idx])]
    return positive_idx[ranked_local_idx].astype(np.int64, copy=False)


def _build_relation_tail_lookup(kg_dict, relation_id, n_items):
    relation_tail_lookup = {}
    for head in range(n_items):
        tails = [int(tail) for rel, tail in kg_dict.get(int(head), []) if int(rel) == int(relation_id)]
        if len(tails) > 0:
            relation_tail_lookup[int(head)] = sorted(set(tails))
    return relation_tail_lookup


def _build_npmi_lookup(pmi_dict, pmi_threshold=None):
    npmi_lookup = defaultdict(dict)
    pair_count = 0
    if pmi_dict is None:
        return npmi_lookup, pair_count
    for item_pair, stat_list in pmi_dict.items():
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        npmi = float(stat_list[1])
        if (pmi_threshold is not None) and (npmi < float(pmi_threshold)):
            continue
        npmi_lookup[item_i][item_j] = npmi
        pair_count += 1
    return npmi_lookup, pair_count



def _resolve_relation_item_embeds(relation_item_embeds, relation_name, relation_order=None):
    if relation_item_embeds is None:
        return None
    relation_name = str(relation_name)
    if isinstance(relation_item_embeds, dict):
        return relation_item_embeds.get(relation_name)

    normalized_order = _normalize_relation_order(relation_item_embeds, relation_order)
    if relation_name not in normalized_order:
        return None
    relation_idx = normalized_order.index(relation_name)

    if isinstance(relation_item_embeds, (list, tuple)):
        return relation_item_embeds[relation_idx] if relation_idx < len(relation_item_embeds) else None

    relation_item_embeds = np.asarray(relation_item_embeds)
    if relation_item_embeds.ndim >= 3 and relation_idx < relation_item_embeds.shape[0]:
        return relation_item_embeds[relation_idx]
    return None



def _resolve_relation_item_availability(relation_item_availability, relation_name, relation_order=None, n_items=None):
    if relation_item_availability is None:
        return None
    relation_name = str(relation_name)
    if isinstance(relation_item_availability, dict):
        return relation_item_availability.get(relation_name)

    normalized_order = _normalize_relation_order(None, relation_order)
    if relation_name not in normalized_order:
        return None
    relation_idx = normalized_order.index(relation_name)

    if isinstance(relation_item_availability, (list, tuple)):
        return relation_item_availability[relation_idx] if relation_idx < len(relation_item_availability) else None

    relation_item_availability = np.asarray(relation_item_availability)
    if relation_item_availability.ndim == 1:
        return relation_item_availability
    if relation_item_availability.ndim == 2:
        if n_items is not None and relation_item_availability.shape[0] == int(n_items) and relation_idx < relation_item_availability.shape[1]:
            return relation_item_availability[:, relation_idx]
        if n_items is not None and relation_item_availability.shape[1] == int(n_items) and relation_idx < relation_item_availability.shape[0]:
            return relation_item_availability[relation_idx]
        if relation_idx < relation_item_availability.shape[0]:
            candidate = np.asarray(relation_item_availability[relation_idx]).reshape(-1)
            if (n_items is None) or (candidate.shape[0] == int(n_items)):
                return candidate
        if relation_idx < relation_item_availability.shape[1]:
            candidate = np.asarray(relation_item_availability[:, relation_idx]).reshape(-1)
            if (n_items is None) or (candidate.shape[0] == int(n_items)):
                return candidate
        return None
    if relation_item_availability.ndim >= 3 and relation_idx < relation_item_availability.shape[0]:
        return relation_item_availability[relation_idx]
    return None



def _resolve_relation_id(relation_name, relation_index, relation_name_to_id=None):
    relation_name = str(relation_name)
    if isinstance(relation_name_to_id, dict) and relation_name in relation_name_to_id:
        return int(relation_name_to_id[relation_name])
    if re.fullmatch(r"-?\d+", relation_name):
        return int(relation_name)
    return int(relation_index) + 1



def _build_multi_relation_fusion_sparse_mat(
    item_embeds,
    relation_item_embeds,
    relation_item_availability,
    relation_order,
    n_items,
    struct_threshold,
    semantic_threshold,
    chunk_size=256,
    fusion_alpha=0.7,
):
    relation_order = _normalize_relation_order(relation_item_embeds, relation_order)
    pair_stats = {
        "struct_pairs": 0,
        "semantic_pairs": 0,
        "fused_pairs": 0,
        "x_pairs": 0,
        "semantic_available_items": 0,
        "relation_count": int(len(relation_order)),
        "relations_with_embeddings": 0,
        "relations_with_pairs": 0,
        "fusion_alpha": float(fusion_alpha),
    }
    branch_stats = {}
    if len(relation_order) == 0:
        return _build_empty_sparse_mat(n_items).tocoo(), pair_stats, branch_stats

    graph_item_embeds = _align_embedding_rows(item_embeds, n_items, "graph item")
    if graph_item_embeds is None:
        print("skip multi-relation fusion branch: graph item embeddings unavailable")
        return _build_empty_sparse_mat(n_items).tocoo(), pair_stats, branch_stats

    struct_sparse_mat = _build_cosine_sparse_mat(graph_item_embeds, struct_threshold, chunk_size=chunk_size).tocoo()
    pair_stats["struct_pairs"] = int(struct_sparse_mat.nnz)
    merged_fusion_mat = _build_empty_sparse_mat(n_items).tocsr()

    for relation_idx, relation_name in enumerate(relation_order):
        relation_embed = _resolve_relation_item_embeds(relation_item_embeds, relation_name, relation_order)
        relation_embed = _align_embedding_rows(relation_embed, n_items, "relation {} item".format(relation_name))
        relation_availability = _resolve_relation_item_availability(
            relation_item_availability,
            relation_name,
            relation_order=relation_order,
            n_items=n_items,
        )
        semantic_available_items = _count_available_items(
            relation_availability,
            n_items,
            "relation {} availability".format(relation_name),
        )
        has_semantic_embed = relation_embed is not None
        if has_semantic_embed:
            semantic_sparse_mat = _build_thresholded_cosine_sparse_mat(
                relation_embed,
                n_items,
                semantic_threshold,
                chunk_size=chunk_size,
                embed_name="relation {} item".format(relation_name),
                item_availability=relation_availability,
                availability_name="relation {} availability".format(relation_name),
            ).tocoo()
        else:
            semantic_available_items = 0
            semantic_sparse_mat = _build_empty_sparse_mat(n_items).tocoo()

        fused_mat, fused_stats = _fuse_sparse_similarity_mats(
            struct_sparse_mat=struct_sparse_mat,
            semantic_sparse_mat=semantic_sparse_mat,
            n_items=n_items,
            semantic_available_items=semantic_available_items,
            fusion_alpha=fusion_alpha,
        )
        merged_fusion_mat = merged_fusion_mat + fused_mat.tocsr()
        branch_stats[str(relation_name)] = {
            "relation_name": str(relation_name),
            "relation_index": int(relation_idx),
            "semantic_fallback_to_struct": bool(not has_semantic_embed),
            **fused_stats,
        }
        pair_stats["semantic_pairs"] += int(fused_stats["semantic_pairs"])
        pair_stats["semantic_available_items"] += int(fused_stats["semantic_available_items"])
        if has_semantic_embed:
            pair_stats["relations_with_embeddings"] += 1
        if int(fused_stats["fused_pairs"]) > 0:
            pair_stats["relations_with_pairs"] += 1

    merged_fusion_mat = merged_fusion_mat.tocoo()
    pair_stats["fused_pairs"] = int(merged_fusion_mat.nnz)
    pair_stats["x_pairs"] = int(merged_fusion_mat.nnz)
    return merged_fusion_mat, pair_stats, branch_stats


def _generate_relation_candidates_from_sparse_scores(
    kg_dict,
    npmi_lookup,
    struct_sparse_mat,
    semantic_sparse_mat,
    relation_id,
    topk,
    branch_name,
    lambda_weight,
    n_items,
):
    relation_triplets = _build_relation_tail_lookup(kg_dict, relation_id, n_items)
    struct_sparse_mat = struct_sparse_mat.tocsr()
    semantic_sparse_mat = semantic_sparse_mat.tocsr()

    candidate_triples = []
    pair_count = 0
    valid_pair_count = 0
    heads_without_relation = 0
    for head in tqdm(range(n_items), desc="generate {} candidates".format(branch_name)):
        if len(relation_triplets.get(int(head), [])) == 0:
            heads_without_relation += 1

        struct_start = struct_sparse_mat.indptr[head]
        struct_end = struct_sparse_mat.indptr[head + 1]
        sem_start = semantic_sparse_mat.indptr[head]
        sem_end = semantic_sparse_mat.indptr[head + 1]
        if struct_start == struct_end or sem_start == sem_end:
            continue

        struct_cols = struct_sparse_mat.indices[struct_start:struct_end]
        struct_vals = struct_sparse_mat.data[struct_start:struct_end]
        sem_cols = semantic_sparse_mat.indices[sem_start:sem_end]
        sem_vals = semantic_sparse_mat.data[sem_start:sem_end]
        sem_map = {int(col): float(val) for col, val in zip(sem_cols.tolist(), sem_vals.tolist())}

        candidate_neighbors = []
        candidate_scores = []
        for neighbor, struct_val in zip(struct_cols.tolist(), struct_vals.tolist()):
            neighbor = int(neighbor)
            if neighbor == int(head):
                continue
            semantic_val = sem_map.get(neighbor)
            if semantic_val is None:
                continue
            valid_pair_count += 1
            x_rel = float(np.sqrt(max(float(struct_val), 0.0) * max(float(semantic_val), 0.0)))
            if x_rel <= 0:
                continue
            npmi_score = float(npmi_lookup.get(int(head), {}).get(neighbor, 0.0))
            fused_score = float(lambda_weight) * npmi_score + float(1.0 - lambda_weight) * x_rel
            if fused_score <= 0:
                continue
            candidate_neighbors.append(neighbor)
            candidate_scores.append(fused_score)

        if len(candidate_neighbors) == 0:
            continue
        pair_count += len(candidate_neighbors)
        neighbor_idx = _topk_indices_from_scores(np.asarray(candidate_scores, dtype=np.float32), topk)
        if neighbor_idx.size == 0:
            continue

        existing_tails = set(int(tail) for rel, tail in kg_dict.get(int(head), []) if int(rel) == int(relation_id))
        for idx in neighbor_idx.tolist():
            neighbor = int(candidate_neighbors[int(idx)])
            for tail in relation_triplets.get(neighbor, []):
                tail = int(tail)
                if tail in existing_tails:
                    continue
                candidate_triples.append([int(head), int(relation_id), tail])

    if len(candidate_triples) == 0:
        return np.empty((0, 3), dtype=np.int64), {
            "branch": branch_name,
            "relation_id": int(relation_id),
            "pair_count": int(pair_count),
            "candidate_triples": 0,
            "heads_without_relation": int(heads_without_relation),
            "valid_pair_count": int(valid_pair_count),
            "x_pair_count": int(pair_count),
            "heads_with_relation": int(len(relation_triplets)),
        }

    candidate_triples = np.unique(np.asarray(candidate_triples, dtype=np.int64), axis=0)
    return candidate_triples, {
        "branch": branch_name,
        "relation_id": int(relation_id),
        "pair_count": int(pair_count),
        "candidate_triples": int(candidate_triples.shape[0]),
        "heads_without_relation": int(heads_without_relation),
        "valid_pair_count": int(valid_pair_count),
        "x_pair_count": int(pair_count),
        "heads_with_relation": int(len(relation_triplets)),
    }


def _generate_relation_candidates_from_sparse_single_score(
    kg_dict,
    npmi_lookup,
    similarity_sparse_mat,
    relation_id,
    topk,
    branch_name,
    pmi_weight,
    similarity_weight,
    n_items,
):
    relation_triplets = _build_relation_tail_lookup(kg_dict, relation_id, n_items)
    similarity_sparse_mat = similarity_sparse_mat.tocsr()

    candidate_triples = []
    pair_count = 0
    valid_pair_count = 0
    heads_without_relation = 0
    for head in tqdm(range(n_items), desc="generate {} candidates".format(branch_name)):
        if len(relation_triplets.get(int(head), [])) == 0:
            heads_without_relation += 1

        row_start = similarity_sparse_mat.indptr[head]
        row_end = similarity_sparse_mat.indptr[head + 1]
        if row_start == row_end:
            continue

        row_cols = similarity_sparse_mat.indices[row_start:row_end]
        row_vals = similarity_sparse_mat.data[row_start:row_end]

        candidate_neighbors = []
        candidate_scores = []
        for neighbor, sim_val in zip(row_cols.tolist(), row_vals.tolist()):
            neighbor = int(neighbor)
            if neighbor == int(head):
                continue
            valid_pair_count += 1
            npmi_score = float(npmi_lookup.get(int(head), {}).get(neighbor, 0.0))
            fused_score = float(pmi_weight) * npmi_score + float(similarity_weight) * float(sim_val)
            if fused_score <= 0:
                continue
            candidate_neighbors.append(neighbor)
            candidate_scores.append(fused_score)

        if len(candidate_neighbors) == 0:
            continue
        pair_count += len(candidate_neighbors)
        neighbor_idx = _topk_indices_from_scores(np.asarray(candidate_scores, dtype=np.float32), topk)
        if neighbor_idx.size == 0:
            continue

        existing_tails = set(int(tail) for rel, tail in kg_dict.get(int(head), []) if int(rel) == int(relation_id))
        for idx in neighbor_idx.tolist():
            neighbor = int(candidate_neighbors[int(idx)])
            for tail in relation_triplets.get(neighbor, []):
                tail = int(tail)
                if tail in existing_tails:
                    continue
                candidate_triples.append([int(head), int(relation_id), tail])

    if len(candidate_triples) == 0:
        return np.empty((0, 3), dtype=np.int64), {
            "branch": branch_name,
            "relation_id": int(relation_id),
            "pair_count": int(pair_count),
            "candidate_triples": 0,
            "heads_without_relation": int(heads_without_relation),
            "valid_pair_count": int(valid_pair_count),
            "score_pair_count": int(pair_count),
            "heads_with_relation": int(len(relation_triplets)),
        }

    candidate_triples = np.unique(np.asarray(candidate_triples, dtype=np.int64), axis=0)
    return candidate_triples, {
        "branch": branch_name,
        "relation_id": int(relation_id),
        "pair_count": int(pair_count),
        "candidate_triples": int(candidate_triples.shape[0]),
        "heads_without_relation": int(heads_without_relation),
        "valid_pair_count": int(valid_pair_count),
        "score_pair_count": int(pair_count),
        "heads_with_relation": int(len(relation_triplets)),
    }


def _generate_relation_specific_candidates(kg_dict, score_mat, relation_id, topk, branch_name, relation_triplets):
    candidate_triples = []
    pair_count = 0
    heads_without_relation = 0
    n_items = int(score_mat.shape[0])

    for head in tqdm(range(n_items), desc="generate {} candidates".format(branch_name)):
        if len(relation_triplets.get(int(head), [])) == 0:
            heads_without_relation += 1
        existing_tails = set(int(tail) for rel, tail in kg_dict.get(int(head), []) if int(rel) == int(relation_id))
        neighbor_idx = _topk_indices_from_scores(score_mat[head], topk)
        if neighbor_idx.size == 0:
            continue
        pair_count += int(neighbor_idx.size)
        for neighbor in neighbor_idx.tolist():
            for tail in relation_triplets.get(int(neighbor), []):
                if int(tail) in existing_tails:
                    continue
                candidate_triples.append([int(head), int(relation_id), int(tail)])

    if len(candidate_triples) == 0:
        return np.empty((0, 3), dtype=np.int64), {
            "branch": branch_name,
            "relation_id": int(relation_id),
            "pair_count": int(pair_count),
            "candidate_triples": 0,
            "heads_without_relation": int(heads_without_relation),
        }

    candidate_triples = np.unique(np.asarray(candidate_triples, dtype=np.int64), axis=0)
    return candidate_triples, {
        "branch": branch_name,
        "relation_id": int(relation_id),
        "pair_count": int(pair_count),
        "candidate_triples": int(candidate_triples.shape[0]),
        "heads_without_relation": int(heads_without_relation),
    }


def _filter_candidates_with_kgc(candidate_kg, kgc_scores, score_threshold=0.0, topk_per_head=0, selection='scored', rng=None):
    candidate_kg = np.asarray(candidate_kg, dtype=np.int64)
    if candidate_kg.size == 0:
        return candidate_kg.reshape(0, 3), {
            "before_total": 0,
            "after_total": 0,
            "before_by_relation": {},
            "after_by_relation": {},
            "retention_by_relation": {},
        }

    kgc_scores = np.asarray(kgc_scores, dtype=np.float32).reshape(-1)
    if candidate_kg.shape[0] != kgc_scores.shape[0]:
        raise ValueError("candidate_kg and kgc_scores size mismatch: {} vs {}".format(candidate_kg.shape[0], kgc_scores.shape[0]))

    before_by_relation = {}
    after_by_relation = {}
    kept_indices = []
    for rel in np.unique(candidate_kg[:, 1]).tolist():
        before_by_relation[int(rel)] = int(np.sum(candidate_kg[:, 1] == int(rel)))

    for head in np.unique(candidate_kg[:, 0]).tolist():
        head_indices = np.where(candidate_kg[:, 0] == int(head))[0]
        if head_indices.size == 0:
            continue
        head_scores = kgc_scores[head_indices]
        if float(score_threshold) > 0:
            pass_mask = head_scores >= float(score_threshold)
            head_indices = head_indices[pass_mask]
        if head_indices.size == 0:
            continue
        if int(topk_per_head) > 0 and head_indices.size > int(topk_per_head):
            if str(selection) == 'random':
                # random-injection ablation (same number of edges as scored selection)
                _picker = rng if rng is not None else np.random
                head_indices = _picker.choice(head_indices, size=int(topk_per_head), replace=False)
            else:
                head_scores = kgc_scores[head_indices]
                local_top_idx = np.argpartition(-head_scores, int(topk_per_head) - 1)[: int(topk_per_head)]
                head_indices = head_indices[local_top_idx]
        head_indices = head_indices[np.argsort(-kgc_scores[head_indices])]
        kept_indices.append(head_indices)

    if len(kept_indices) == 0:
        filtered_kg = np.empty((0, 3), dtype=np.int64)
    else:
        filtered_kg = np.unique(candidate_kg[np.concatenate(kept_indices, axis=0)], axis=0)

    for rel in before_by_relation.keys():
        after_by_relation[int(rel)] = int(np.sum(filtered_kg[:, 1] == int(rel))) if filtered_kg.shape[0] > 0 else 0

    retention_by_relation = {}
    for rel, before_count in before_by_relation.items():
        retention_by_relation[int(rel)] = 0.0 if before_count == 0 else float(after_by_relation[int(rel)]) / float(before_count)

    return filtered_kg, {
        "before_total": int(candidate_kg.shape[0]),
        "after_total": int(filtered_kg.shape[0]),
        "before_by_relation": before_by_relation,
        "after_by_relation": after_by_relation,
        "retention_by_relation": retention_by_relation,
    }


def _legacy_generate_candi_kg(
    pmi_dict,
    n_items,
    kg_dict,
    item_embeds=[],
    semantic_item_embeds=None,
    kg_completion_mode="npmi+cosine",
    use_x_fusion=False,
    pmi_threshold=0.5,
    cos_threshold=0.9,
    semantic_threshold=0.8,
    pmi_weight=1.0,
    cos_weight=1.0,
    semantic_weight=1.0,
    topk_neighbors=0,
    style_item_embeds=None,
    similar_item_embeds=None,
    overall_item_embeds=None,
    style_item_availability=None,
    similar_item_availability=None,
    overall_item_availability=None,
    style_struct_threshold=0.90,
    style_semantic_threshold=0.92,
    similar_struct_threshold=0.95,
    similar_semantic_threshold=0.95,
    overall_struct_threshold=0.90,
    overall_semantic_threshold=0.80,
    relation_item_embeds=None,
    relation_item_availability=None,
    relation_name_to_id=None,
    relation_order=None,
    relation_struct_threshold=0.90,
    relation_semantic_threshold=0.80,
    relation_lambda=0.7,
    relation_topk=3,
    score_chunk_size=256,
    fusion_alpha=0.7,
):
    normalized_mode, completion_components = normalize_kg_completion_mode(kg_completion_mode)
    print("legacy kg completion mode: {} use_x_fusion={}".format(normalized_mode, bool(use_x_fusion)))

    merge_mat = _build_empty_sparse_mat(n_items).tocsr()
    pair_stats = {
        "npmi_pairs": 0,
        "cosine_pairs": 0,
        "semantic_pairs": 0,
        "x_pairs": 0,
        "fusion_pairs": 0,
        "style_available_items": 0,
        "similar_available_items": 0,
        "overall_available_items": 0,
        "relation_count": 0,
        "relations_with_embeddings": 0,
        "relations_with_pairs": 0,
        "merged_pairs": 0,
        "fusion_alpha": float(fusion_alpha),
    }
    branch_stats = {}
    x_components = _ordered_x_components(completion_components)
    similarity_source = "cosine"
    similarity_label = "cosine_mat"
    semantic_branch_enabled = ("semantic" in completion_components) and (not bool(use_x_fusion))
    graph_item_embeds = _align_embedding_rows(item_embeds, n_items, "graph item")
    relation_order = _normalize_relation_order(relation_item_embeds, relation_order)
    multi_relation_fusion_enabled = (
        bool(use_x_fusion)
        and ("fusion" in completion_components)
        and _has_relation_embeddings(relation_item_embeds, relation_order)
        and len(relation_order) > 0
    )

    if "npmi" in completion_components:
        if pmi_dict is None:
            npmi_mat = _build_empty_sparse_mat(n_items).tocoo()
            print("skip npmi branch: pmi_dict is None")
        else:
            npmi_mat = _build_npmi_sparse_mat(pmi_dict, n_items, pmi_threshold).tocoo()
            print("pmi num pair: {}".format(int(npmi_mat.nnz)))
            merge_mat = merge_mat + npmi_mat.multiply(np.float32(pmi_weight))
        pair_stats["npmi_pairs"] = int(npmi_mat.nnz)
    else:
        npmi_mat = _build_empty_sparse_mat(n_items).tocoo()
        print("skip npmi branch")

    if multi_relation_fusion_enabled:
        fusion_mat, fusion_pair_stats, relation_branch_stats = _build_multi_relation_fusion_sparse_mat(
            item_embeds=item_embeds,
            relation_item_embeds=relation_item_embeds,
            relation_item_availability=relation_item_availability,
            relation_order=relation_order,
            n_items=n_items,
            struct_threshold=relation_struct_threshold,
            semantic_threshold=relation_semantic_threshold,
            chunk_size=score_chunk_size,
            fusion_alpha=fusion_alpha,
        )
        merge_mat = merge_mat + fusion_mat.multiply(np.float32(cos_weight))
        similarity_source = "multi_relation_fusion"
        similarity_label = "fusion_mat"
        pair_stats["cosine_pairs"] = int(fusion_pair_stats["struct_pairs"])
        pair_stats["semantic_pairs"] = int(fusion_pair_stats["semantic_pairs"])
        pair_stats["fusion_pairs"] = int(fusion_pair_stats["fused_pairs"])
        pair_stats["x_pairs"] = int(fusion_pair_stats["x_pairs"])
        pair_stats["relation_count"] = int(fusion_pair_stats["relation_count"])
        pair_stats["relations_with_embeddings"] = int(fusion_pair_stats["relations_with_embeddings"])
        pair_stats["relations_with_pairs"] = int(fusion_pair_stats["relations_with_pairs"])
        branch_stats["relation_fusion"] = relation_branch_stats
        print(
            "legacy fusion branch: relations={} fused_pairs={} struct_pairs={} semantic_pairs={} fusion_alpha={}".format(
                int(fusion_pair_stats["relation_count"]),
                int(fusion_pair_stats["fused_pairs"]),
                int(fusion_pair_stats["struct_pairs"]),
                int(fusion_pair_stats["semantic_pairs"]),
                float(fusion_pair_stats["fusion_alpha"]),
            )
        )
    elif bool(use_x_fusion):
        if len(x_components) == 0:
            raise ValueError("use_x_fusion=True requires kg_completion_mode to include fusion/x_style/x_sim/x_overall")

        fusion_blocks = []
        x_stats_list = []
        for x_component in x_components:
            x_branch_config = _resolve_x_branch_config(
                x_component,
                style_item_embeds=style_item_embeds,
                similar_item_embeds=similar_item_embeds,
                overall_item_embeds=overall_item_embeds,
                style_item_availability=style_item_availability,
                similar_item_availability=similar_item_availability,
                overall_item_availability=overall_item_availability,
                style_struct_threshold=style_struct_threshold,
                style_semantic_threshold=style_semantic_threshold,
                similar_struct_threshold=similar_struct_threshold,
                similar_semantic_threshold=similar_semantic_threshold,
                overall_struct_threshold=overall_struct_threshold,
                overall_semantic_threshold=overall_semantic_threshold,
            )

            fusion_mat, fusion_stats = _build_x_sparse_mat(
                item_embeds=item_embeds,
                semantic_item_embeds=x_branch_config["semantic_item_embeds"],
                semantic_item_availability=x_branch_config["semantic_item_availability"],
                n_items=n_items,
                struct_threshold=x_branch_config["struct_threshold"],
                semantic_threshold=x_branch_config["semantic_threshold"],
                chunk_size=score_chunk_size,
                semantic_embed_name=x_branch_config["embed_name"],
                fusion_alpha=fusion_alpha,
            )
            fusion_blocks.append(fusion_mat.tocsr())
            x_stats_list.append((x_branch_config, fusion_stats))
            print(
                "legacy fusion branch: {} -> fusion_mat, available_items={}, struct_pairs={}, semantic_pairs={}, fused_pairs={}, fusion_alpha={}".format(
                    x_branch_config["component"],
                    int(fusion_stats["semantic_available_items"]),
                    int(fusion_stats["struct_pairs"]),
                    int(fusion_stats["semantic_pairs"]),
                    int(fusion_stats["fused_pairs"]),
                    float(fusion_stats["fusion_alpha"]),
                )
            )

        merged_fusion_mat = _build_empty_sparse_mat(n_items).tocsr()
        for fusion_block in fusion_blocks:
            merged_fusion_mat = merged_fusion_mat + fusion_block
        merged_fusion_mat = merged_fusion_mat.tocoo()

        pair_stats["x_pairs"] = int(merged_fusion_mat.nnz)
        pair_stats["fusion_pairs"] = int(merged_fusion_mat.nnz)
        if len(x_stats_list) > 0:
            pair_stats["cosine_pairs"] = max(int(x_stats["struct_pairs"]) for _, x_stats in x_stats_list)
            pair_stats["semantic_pairs"] = int(sum(int(x_stats["semantic_pairs"]) for _, x_stats in x_stats_list))
        for x_branch_config, x_stats in x_stats_list:
            if x_branch_config["component"] == "x_style":
                pair_stats["style_available_items"] = int(x_stats["semantic_available_items"])
            elif x_branch_config["component"] == "x_sim":
                pair_stats["similar_available_items"] = int(x_stats["semantic_available_items"])
            elif x_branch_config["component"] == "x_overall":
                pair_stats["overall_available_items"] = int(x_stats["semantic_available_items"])
            branch_stats[x_branch_config["component"]] = dict(x_stats)
        merge_mat = merge_mat + merged_fusion_mat.multiply(np.float32(cos_weight))
        similarity_source = "fusion"
        similarity_label = "fusion_mat"
        if "cosine" in completion_components:
            print("legacy mode with use_x_fusion=True: cosine branch is replaced by fusion_mat")
    elif "cosine" in completion_components:
        if graph_item_embeds is None:
            embed_mat = _build_empty_sparse_mat(n_items).tocoo()
            print("skip cosine branch: item embeddings unavailable")
        else:
            embed_mat = _build_cosine_sparse_mat(graph_item_embeds, cos_threshold).tocoo()
            print("cos num pair: {}".format(int(embed_mat.nnz)))
            merge_mat = merge_mat + embed_mat.multiply(np.float32(cos_weight))
        pair_stats["cosine_pairs"] = int(embed_mat.nnz)
    else:
        embed_mat = _build_empty_sparse_mat(n_items).tocoo()
        print("skip cosine branch")
        if len(x_components) > 0:
            print("legacy mode ignores fusion branch because use_x_fusion=False; final similarity term remains cosine_mat")

    if semantic_branch_enabled:
        semantic_item_embeds = _align_embedding_rows(semantic_item_embeds, n_items, "semantic item")
        if semantic_item_embeds is None:
            raise ValueError("semantic branch is enabled, but semantic_item_embeds is empty")
        semantic_mat = _build_cosine_sparse_mat(semantic_item_embeds, semantic_threshold).tocoo()
        print("semantic num pair: {}".format(int(semantic_mat.nnz)))
        merge_mat = merge_mat + semantic_mat.multiply(np.float32(semantic_weight))
        pair_stats["semantic_pairs"] = int(semantic_mat.nnz)
    elif "semantic" in completion_components:
        semantic_mat = _build_empty_sparse_mat(n_items).tocoo()
        print("ignore standalone semantic branch because use_x_fusion=True and semantics are already folded into fusion_mat")
    else:
        semantic_mat = _build_empty_sparse_mat(n_items).tocoo()
        print("skip semantic branch")

    merge_mat = merge_mat.tocoo()
    if int(topk_neighbors) > 0:
        before_topk_nnz = merge_mat.nnz
        merge_mat = _limit_sparse_neighbors_topk(merge_mat, topk_neighbors).tocoo()
        print("topk neighbor limit: {} -> {} pairs (topk={})".format(before_topk_nnz, merge_mat.nnz, int(topk_neighbors)))

    pair_stats["merged_pairs"] = int(merge_mat.nnz)
    merge_terms = []
    if "npmi" in completion_components:
        merge_terms.append("npmi_mat * pmi_weight")
    if similarity_label == "fusion_mat":
        merge_terms.append("fusion_mat * cos_weight, where fusion = alpha*struct + (1-alpha)*semantic")
    elif "cosine" in completion_components:
        merge_terms.append("cosine_mat * cos_weight")
    if semantic_branch_enabled:
        merge_terms.append("semantic_mat * semantic_weight")
    merge_formula = "merge_mat = " + " + ".join(merge_terms) if len(merge_terms) > 0 else "merge_mat = 0"

    print("legacy similarity source: {}".format(similarity_label))
    print("legacy merge formula: {}".format(merge_formula))
    if merge_mat.nnz == 0:
        print("generate candi kg: 0")
        return np.empty((0, 3), dtype=np.int64), {
            "generator": "legacy_full_transfer",
            "use_relation_aware": False,
            "use_x_fusion": bool(use_x_fusion),
            "mode": normalized_mode,
            "baseline_compatible": True,
            "similarity_source": similarity_source,
            "merge_formula": merge_formula,
            "pair_stats": pair_stats,
            "branch_stats": branch_stats,
            "relation_stats": {},
        }

    row_data = merge_mat.row
    col_data = merge_mat.col
    item_touples = np.concatenate([np.expand_dims(row_data, axis=-1), np.expand_dims(col_data, axis=-1)], axis=-1)
    item_neb_dict = defaultdict(set)
    for item_1, item_2 in item_touples:
        item_neb_dict[int(item_1)].add(int(item_2))
        if int(topk_neighbors) <= 0:
            item_neb_dict[int(item_2)].add(int(item_1))

    new_kg_dict = defaultdict(list)
    for k, v in tqdm(item_neb_dict.items(), desc="process item_touples..."):
        temp = []
        for item in v:
            kg_seq = kg_dict[item]
            for r_t in kg_seq:
                if r_t not in kg_dict[k]:
                    temp.append(r_t)
        new_kg_dict[int(k)] = temp

    candi_kg = []
    for k, v in new_kg_dict.items():
        for r, t in v:
            candi_kg.append([int(k), int(r), int(t)])
    if len(candi_kg) == 0:
        return np.empty((0, 3), dtype=np.int64), {
            "generator": "legacy_full_transfer",
            "use_relation_aware": False,
            "use_x_fusion": bool(use_x_fusion),
            "mode": normalized_mode,
            "baseline_compatible": True,
            "similarity_source": similarity_source,
            "merge_formula": merge_formula,
            "pair_stats": pair_stats,
            "branch_stats": branch_stats,
            "relation_stats": {},
        }
    candi_kg = np.unique(np.asarray(candi_kg, dtype=np.int64), axis=0)
    relation_stats = {}
    for rel in np.unique(candi_kg[:, 1]).tolist():
        relation_stats[int(rel)] = int(np.sum(candi_kg[:, 1] == int(rel)))
    print("generate candi kg: {}".format(int(candi_kg.shape[0])))
    return candi_kg, {
        "generator": "legacy_full_transfer",
        "use_relation_aware": False,
        "use_x_fusion": bool(use_x_fusion),
        "mode": normalized_mode,
        "baseline_compatible": True,
        "similarity_source": similarity_source,
        "merge_formula": merge_formula,
        "pair_stats": pair_stats,
        "branch_stats": branch_stats,
        "relation_stats": relation_stats,
    }


def _relation_aware_generate_candi_kg(
    pmi_dict,
    n_items,
    kg_dict,
    item_embeds=[],
    kg_completion_mode="npmi+cosine",
    use_x_fusion=False,
    pmi_threshold=0.5,
    cos_threshold=0.9,
    pmi_weight=1.0,
    cos_weight=1.0,
    style_item_embeds=None,
    similar_item_embeds=None,
    style_item_availability=None,
    similar_item_availability=None,
    style_relation_id=1,
    similar_relation_id=2,
    style_struct_threshold=0.90,
    style_semantic_threshold=0.92,
    similar_struct_threshold=0.95,
    similar_semantic_threshold=0.95,
    style_lambda=0.7,
    similar_lambda=0.8,
    style_topk=3,
    similar_topk=1,
    relation_item_embeds=None,
    relation_item_availability=None,
    relation_name_to_id=None,
    relation_order=None,
    relation_struct_threshold=0.90,
    relation_semantic_threshold=0.80,
    relation_lambda=0.7,
    relation_topk=3,
    score_chunk_size=256,
    fusion_alpha=0.7,
):
    normalized_mode, completion_components = normalize_kg_completion_mode(kg_completion_mode)
    print("relation-aware kg completion mode: {} use_x_fusion={}".format(normalized_mode, bool(use_x_fusion)))
    x_components = _ordered_x_components(completion_components)
    relation_order = _normalize_relation_order(relation_item_embeds, relation_order)
    multi_relation_fusion_enabled = (
        bool(use_x_fusion)
        and ("fusion" in completion_components)
        and _has_relation_embeddings(relation_item_embeds, relation_order)
        and len(relation_order) > 0
    )
    if (not multi_relation_fusion_enabled) and (style_relation_id is None or similar_relation_id is None):
        raise ValueError(
            "relation-aware cosine mode requires explicit style/similar relation ids; got style_relation_id={} similar_relation_id={}. "
            "Disable relation-aware for pure npmi+cosine on multi-relation datasets or provide valid relation ids.".format(
                style_relation_id,
                similar_relation_id,
            )
        )

    pair_stats = {
        "npmi_pairs": 0,
        "cosine_pairs": 0,
        "semantic_pairs": 0,
        "x_style_pairs": 0,
        "x_sim_pairs": 0,
        "fusion_pairs": 0,
        "style_available_items": 0,
        "similar_available_items": 0,
        "relation_count": int(len(relation_order)),
        "relations_with_embeddings": 0,
        "relations_with_pairs": 0,
        "fusion_alpha": float(fusion_alpha),
    }
    branch_stats = {}
    relation_stats = {}
    similarity_source = "cosine"
    merge_formula = None

    npmi_lookup = defaultdict(dict)
    if "npmi" in completion_components:
        lookup_threshold = None if bool(use_x_fusion) else pmi_threshold
        npmi_lookup, npmi_pair_count = _build_npmi_lookup(pmi_dict, pmi_threshold=lookup_threshold)
        pair_stats["npmi_pairs"] = int(npmi_pair_count)
        if pmi_dict is None:
            print("relation-aware mode: pmi_dict is None, npmi branch contributes zeros")
        else:
            print("relation-aware npmi num pair: {}".format(pair_stats["npmi_pairs"]))
    else:
        print("skip npmi branch in relation-aware mode")

    candidate_blocks = []
    if multi_relation_fusion_enabled:
        similarity_source = "multi_relation_fusion"
        merge_formula = "relation_score = npmi_mat * relation_lambda + fusion_mat * (1-relation_lambda)"
        struct_sparse_mat = _build_thresholded_cosine_sparse_mat(
            item_embeds,
            n_items,
            relation_struct_threshold,
            chunk_size=score_chunk_size,
            embed_name="graph item",
        ).tocoo()
        pair_stats["cosine_pairs"] = int(struct_sparse_mat.nnz)
        print(
            "relation-aware fusion branch: relations={} struct_pairs={} semantic_threshold={} topk={} relation_lambda={}".format(
                len(relation_order),
                int(struct_sparse_mat.nnz),
                float(relation_semantic_threshold),
                int(relation_topk),
                float(relation_lambda),
            )
        )

        for relation_idx, relation_name in enumerate(relation_order):
            relation_id = _resolve_relation_id(relation_name, relation_idx, relation_name_to_id=relation_name_to_id)
            relation_embed = _resolve_relation_item_embeds(relation_item_embeds, relation_name, relation_order)
            relation_embed = _align_embedding_rows(relation_embed, n_items, "relation {} item".format(relation_name))
            relation_availability = _resolve_relation_item_availability(
                relation_item_availability,
                relation_name,
                relation_order=relation_order,
                n_items=n_items,
            )
            semantic_available_items = _count_available_items(
                relation_availability,
                n_items,
                "relation {} availability".format(relation_name),
            )
            has_semantic_embed = relation_embed is not None
            if has_semantic_embed:
                semantic_sparse_mat = _build_thresholded_cosine_sparse_mat(
                    relation_embed,
                    n_items,
                    relation_semantic_threshold,
                    chunk_size=score_chunk_size,
                    embed_name="relation {} item".format(relation_name),
                    item_availability=relation_availability,
                    availability_name="relation {} availability".format(relation_name),
                ).tocoo()
            else:
                semantic_available_items = 0
                semantic_sparse_mat = _build_empty_sparse_mat(n_items).tocoo()

            fused_mat, fused_stats = _fuse_sparse_similarity_mats(
                struct_sparse_mat=struct_sparse_mat,
                semantic_sparse_mat=semantic_sparse_mat,
                n_items=n_items,
                semantic_available_items=semantic_available_items,
                fusion_alpha=fusion_alpha,
            )
            relation_candidates, relation_candidate_stats = _generate_relation_candidates_from_sparse_single_score(
                kg_dict=kg_dict,
                npmi_lookup=npmi_lookup,
                similarity_sparse_mat=fused_mat,
                relation_id=relation_id,
                topk=relation_topk,
                branch_name="relation:{}".format(relation_name),
                pmi_weight=relation_lambda if "npmi" in completion_components else 0.0,
                similarity_weight=float(1.0 - relation_lambda),
                n_items=n_items,
            )
            branch_stats[str(relation_name)] = {
                **fused_stats,
                **relation_candidate_stats,
                "relation_name": str(relation_name),
                "semantic_fallback_to_struct": bool(not has_semantic_embed),
            }
            relation_stats[int(relation_id)] = int(relation_candidates.shape[0])
            candidate_blocks.append(relation_candidates)
            pair_stats["semantic_pairs"] += int(fused_stats["semantic_pairs"])
            pair_stats["fusion_pairs"] += int(fused_stats["fused_pairs"])
            if has_semantic_embed:
                pair_stats["relations_with_embeddings"] += 1
            if int(fused_stats["fused_pairs"]) > 0:
                pair_stats["relations_with_pairs"] += 1
            print(
                "relation-aware fusion branch {}: relation_id={} available_items={} semantic_pairs={} fused_pairs={} candidate_triples={} semantic_fallback_to_struct={}".format(
                    relation_name,
                    int(relation_id),
                    int(semantic_available_items),
                    int(fused_stats["semantic_pairs"]),
                    int(fused_stats["fused_pairs"]),
                    int(relation_candidate_stats["candidate_triples"]),
                    bool(not has_semantic_embed),
                )
            )
    elif bool(use_x_fusion):
        if len(x_components) == 0:
            raise ValueError("relation-aware mode with fusion requires kg_completion_mode to include fusion/x_style/x_sim")
        similarity_source = "fusion"
        merge_formula = "relation_score = npmi_mat * lambda + fusion branch * (1-lambda)"

        if "x_style" in completion_components:
            style_struct_sparse = _build_thresholded_cosine_sparse_mat(item_embeds, n_items, style_struct_threshold, chunk_size=score_chunk_size, embed_name="graph item")
            style_sem_sparse = _build_thresholded_cosine_sparse_mat(
                style_item_embeds,
                n_items,
                style_semantic_threshold,
                chunk_size=score_chunk_size,
                embed_name="style item",
                item_availability=style_item_availability,
                availability_name="style availability",
            )
            pair_stats["style_available_items"] = _count_available_items(style_item_availability, n_items, "style availability")
            pair_stats["cosine_pairs"] = max(pair_stats["cosine_pairs"], int(style_struct_sparse.nnz))
            style_candidates, style_stats = _generate_relation_candidates_from_sparse_scores(
                kg_dict=kg_dict,
                npmi_lookup=npmi_lookup,
                struct_sparse_mat=style_struct_sparse,
                semantic_sparse_mat=style_sem_sparse,
                relation_id=style_relation_id,
                topk=style_topk,
                branch_name="style fusion",
                lambda_weight=style_lambda,
                n_items=n_items,
            )
            pair_stats["x_style_pairs"] = int(style_stats["x_pair_count"])
            pair_stats["fusion_pairs"] += int(style_stats["x_pair_count"])
            style_stats["semantic_available_items"] = int(pair_stats["style_available_items"])
            branch_stats["style"] = style_stats
            relation_stats[int(style_relation_id)] = int(style_candidates.shape[0])
            candidate_blocks.append(style_candidates)
            print("style fusion branch pairs: available_items={} valid={} fused_pairs={} candidate_triples={}".format(pair_stats["style_available_items"], style_stats["valid_pair_count"], style_stats["x_pair_count"], style_stats["candidate_triples"]))
        else:
            print("skip style fusion branch")

        if "x_sim" in completion_components:
            similar_struct_sparse = _build_thresholded_cosine_sparse_mat(item_embeds, n_items, similar_struct_threshold, chunk_size=score_chunk_size, embed_name="graph item")
            similar_sem_sparse = _build_thresholded_cosine_sparse_mat(
                similar_item_embeds,
                n_items,
                similar_semantic_threshold,
                chunk_size=score_chunk_size,
                embed_name="similar item",
                item_availability=similar_item_availability,
                availability_name="similar availability",
            )
            pair_stats["similar_available_items"] = _count_available_items(similar_item_availability, n_items, "similar availability")
            pair_stats["cosine_pairs"] = max(pair_stats["cosine_pairs"], int(similar_struct_sparse.nnz))
            similar_candidates, similar_stats = _generate_relation_candidates_from_sparse_scores(
                kg_dict=kg_dict,
                npmi_lookup=npmi_lookup,
                struct_sparse_mat=similar_struct_sparse,
                semantic_sparse_mat=similar_sem_sparse,
                relation_id=similar_relation_id,
                topk=similar_topk,
                branch_name="similar fusion",
                lambda_weight=similar_lambda,
                n_items=n_items,
            )
            pair_stats["x_sim_pairs"] = int(similar_stats["x_pair_count"])
            pair_stats["fusion_pairs"] += int(similar_stats["x_pair_count"])
            similar_stats["semantic_available_items"] = int(pair_stats["similar_available_items"])
            branch_stats["similar"] = similar_stats
            relation_stats[int(similar_relation_id)] = int(similar_candidates.shape[0])
            candidate_blocks.append(similar_candidates)
            print("similar fusion branch pairs: available_items={} valid={} fused_pairs={} candidate_triples={}".format(pair_stats["similar_available_items"], similar_stats["valid_pair_count"], similar_stats["x_pair_count"], similar_stats["candidate_triples"]))
        else:
            print("skip similar fusion branch")
    else:
        if len(x_components) > 0:
            raise ValueError("relation-aware cosine mode does not support fusion/x_style/x_sim in kg_completion_mode; set use_x_fusion=True")
        if "cosine" not in completion_components:
            raise ValueError("relation-aware mode without fusion requires kg_completion_mode=npmi+cosine (or cosine-only)")

        cosine_sparse_mat = _build_thresholded_cosine_sparse_mat(item_embeds, n_items, cos_threshold, chunk_size=score_chunk_size, embed_name="graph item")
        pair_stats["cosine_pairs"] = int(cosine_sparse_mat.nnz)
        merge_formula = "relation_score = npmi_mat * pmi_weight + cosine_mat * cos_weight"
        print("relation-aware cosine branch pairs: cosine_pairs={}".format(pair_stats["cosine_pairs"]))

        style_candidates, style_stats = _generate_relation_candidates_from_sparse_single_score(
            kg_dict=kg_dict,
            npmi_lookup=npmi_lookup,
            similarity_sparse_mat=cosine_sparse_mat,
            relation_id=style_relation_id,
            topk=style_topk,
            branch_name="style cosine",
            pmi_weight=pmi_weight if "npmi" in completion_components else 0.0,
            similarity_weight=cos_weight,
            n_items=n_items,
        )
        branch_stats["style"] = style_stats
        relation_stats[int(style_relation_id)] = int(style_candidates.shape[0])
        candidate_blocks.append(style_candidates)
        print("style cosine branch pairs: valid={} candidate_triples={}".format(style_stats["valid_pair_count"], style_stats["candidate_triples"]))

        similar_candidates, similar_stats = _generate_relation_candidates_from_sparse_single_score(
            kg_dict=kg_dict,
            npmi_lookup=npmi_lookup,
            similarity_sparse_mat=cosine_sparse_mat,
            relation_id=similar_relation_id,
            topk=similar_topk,
            branch_name="similar cosine",
            pmi_weight=pmi_weight if "npmi" in completion_components else 0.0,
            similarity_weight=cos_weight,
            n_items=n_items,
        )
        branch_stats["similar"] = similar_stats
        relation_stats[int(similar_relation_id)] = int(similar_candidates.shape[0])
        candidate_blocks.append(similar_candidates)
        print("similar cosine branch pairs: valid={} candidate_triples={}".format(similar_stats["valid_pair_count"], similar_stats["candidate_triples"]))

    non_empty_blocks = [block for block in candidate_blocks if block.shape[0] > 0]
    if len(non_empty_blocks) == 0:
        print("generate candi kg: 0")
        return np.empty((0, 3), dtype=np.int64), {
            "generator": "relation_aware",
            "use_relation_aware": True,
            "use_x_fusion": bool(use_x_fusion),
            "mode": normalized_mode,
            "baseline_compatible": False,
            "similarity_source": similarity_source,
            "merge_formula": merge_formula,
            "pair_stats": pair_stats,
            "branch_stats": branch_stats,
            "relation_stats": relation_stats,
        }

    candi_kg = np.unique(np.concatenate(non_empty_blocks, axis=0), axis=0)
    print("generate candi kg: {}".format(int(candi_kg.shape[0])))
    return candi_kg, {
        "generator": "relation_aware",
        "use_relation_aware": True,
        "use_x_fusion": bool(use_x_fusion),
        "mode": normalized_mode,
        "baseline_compatible": False,
        "similarity_source": similarity_source,
        "merge_formula": merge_formula,
        "pair_stats": pair_stats,
        "branch_stats": branch_stats,
        "relation_stats": relation_stats,
    }


def _generate_candi_kg(
    pmi_dict,
    n_items,
    kg_dict,
    item_embeds=[],
    semantic_item_embeds=None,
    kg_completion_mode="npmi+cosine",
    use_relation_aware=False,
    use_x_fusion=False,
    pmi_threshold=0.5,
    cos_threshold=0.9,
    semantic_threshold=0.8,
    pmi_weight=1.0,
    cos_weight=1.0,
    semantic_weight=1.0,
    topk_neighbors=0,
    style_item_embeds=None,
    similar_item_embeds=None,
    overall_item_embeds=None,
    style_item_availability=None,
    similar_item_availability=None,
    overall_item_availability=None,
    style_relation_id=1,
    similar_relation_id=2,
    style_struct_threshold=0.90,
    style_semantic_threshold=0.92,
    similar_struct_threshold=0.95,
    similar_semantic_threshold=0.95,
    overall_struct_threshold=0.90,
    overall_semantic_threshold=0.80,
    style_lambda=0.7,
    similar_lambda=0.8,
    style_topk=3,
    similar_topk=1,
    relation_item_embeds=None,
    relation_item_availability=None,
    relation_name_to_id=None,
    relation_order=None,
    relation_struct_threshold=0.90,
    relation_semantic_threshold=0.80,
    relation_lambda=0.7,
    relation_topk=3,
    score_chunk_size=256,
    fusion_alpha=0.7,
):
    normalized_mode, completion_components = normalize_kg_completion_mode(kg_completion_mode)
    print(
        "[kg completion] mode={} use_relation_aware={} use_x_fusion={}".format(
            normalized_mode,
            bool(use_relation_aware),
            bool(use_x_fusion),
        )
    )

    if bool(use_relation_aware):
        return _relation_aware_generate_candi_kg(
            pmi_dict=pmi_dict,
            n_items=n_items,
            kg_dict=kg_dict,
            item_embeds=item_embeds,
            kg_completion_mode=normalized_mode,
            use_x_fusion=use_x_fusion,
            pmi_threshold=pmi_threshold,
            cos_threshold=cos_threshold,
            pmi_weight=pmi_weight,
            cos_weight=cos_weight,
            style_item_embeds=style_item_embeds,
            similar_item_embeds=similar_item_embeds,
            style_item_availability=style_item_availability,
            similar_item_availability=similar_item_availability,
            style_relation_id=style_relation_id,
            similar_relation_id=similar_relation_id,
            style_struct_threshold=style_struct_threshold,
            style_semantic_threshold=style_semantic_threshold,
            similar_struct_threshold=similar_struct_threshold,
            similar_semantic_threshold=similar_semantic_threshold,
            style_lambda=style_lambda,
            similar_lambda=similar_lambda,
            style_topk=style_topk,
            similar_topk=similar_topk,
            relation_item_embeds=relation_item_embeds,
            relation_item_availability=relation_item_availability,
            relation_name_to_id=relation_name_to_id,
            relation_order=relation_order,
            relation_struct_threshold=relation_struct_threshold,
            relation_semantic_threshold=relation_semantic_threshold,
            relation_lambda=relation_lambda,
            relation_topk=relation_topk,
            score_chunk_size=score_chunk_size,
            fusion_alpha=fusion_alpha,
        )

    return _legacy_generate_candi_kg(
        pmi_dict=pmi_dict,
        n_items=n_items,
        kg_dict=kg_dict,
        item_embeds=item_embeds,
        semantic_item_embeds=semantic_item_embeds,
        kg_completion_mode=normalized_mode,
        use_x_fusion=use_x_fusion,
        pmi_threshold=pmi_threshold,
        cos_threshold=cos_threshold,
        semantic_threshold=semantic_threshold,
        pmi_weight=pmi_weight,
        cos_weight=cos_weight,
        semantic_weight=semantic_weight,
        topk_neighbors=topk_neighbors,
        style_item_embeds=style_item_embeds,
        similar_item_embeds=similar_item_embeds,
        overall_item_embeds=overall_item_embeds,
        style_item_availability=style_item_availability,
        similar_item_availability=similar_item_availability,
        overall_item_availability=overall_item_availability,
        style_struct_threshold=style_struct_threshold,
        style_semantic_threshold=style_semantic_threshold,
        similar_struct_threshold=similar_struct_threshold,
        similar_semantic_threshold=similar_semantic_threshold,
        overall_struct_threshold=overall_struct_threshold,
        overall_semantic_threshold=overall_semantic_threshold,
        relation_item_embeds=relation_item_embeds,
        relation_item_availability=relation_item_availability,
        relation_name_to_id=relation_name_to_id,
        relation_order=relation_order,
        relation_struct_threshold=relation_struct_threshold,
        relation_semantic_threshold=relation_semantic_threshold,
        relation_lambda=relation_lambda,
        relation_topk=relation_topk,
        score_chunk_size=score_chunk_size,
        fusion_alpha=fusion_alpha,
    )


def _cal_npmi(train_user_set):
    item_freq = {}
    for user, item_seq in train_user_set.items():
        appeared = set()
        for item in item_seq:
            if item in appeared:
                print(1)
                continue
            if item in item_freq:
                item_freq[item] += 1
            else:
                item_freq[item] = 1
            appeared.add(item)
            
    num_windows = len(train_user_set)
    item_pair_count = {}
    for user, item_seq in tqdm(train_user_set.items()):
        appeared = set()
        for i in range(1,len(item_seq)):
            for j in range(0,i):
                item_i = item_seq[i]
                item_j = item_seq[j]
                if item_i == item_j:
                    continue
                item_pair_str = str(item_i) + "," +str(item_j)
                if item_pair_str in appeared:
                    continue
                if item_pair_str in item_pair_count:
                    item_pair_count[item_pair_str] += 1
                else:
                    item_pair_count[item_pair_str] = 1
                appeared.add(item_pair_str)
                #reverse pair
                item_pair_str = str(item_j) + "," +str(item_i)
                if item_pair_str in appeared:
                    continue
                if item_pair_str in item_pair_count:
                    item_pair_count[item_pair_str] += 1
                else:
                    item_pair_count[item_pair_str] = 1
                appeared.add(item_pair_str)

    tmp_max_npmi = 0
    tmp_min_npmi = 0
    tmp_max_pmi = 0
    tmp_min_pmi = 0    
    temp_item_i_max = None
    temp_item_j_max = None
    temp_i_j_max = None

    temp_item_i_min = None
    temp_item_j_min = None
    temp_i_j_min = None
    
    item_pair_pmi = defaultdict(list)
    print("calculating npmi...")
    for item_pair in tqdm(item_pair_count):
        item_i, item_j = [int(x) for x in item_pair.split(",")]
        pair_count = item_pair_count[item_pair]
        item_i_freq, item_j_freq = item_freq[item_i], item_freq[item_j]
        pmi = log(
        (1.0 * pair_count / num_windows)
        / (1.0 * item_i_freq * item_j_freq / (num_windows * num_windows))
        )
        npmi = (
            log(1.0 * item_i_freq * item_j_freq / (num_windows * num_windows))
            / log(1.0 * pair_count / num_windows)
            - 1
        )
        item_pair_pmi[item_pair] = [pmi, npmi, pair_count,item_i_freq,item_j_freq]

        if npmi > tmp_max_npmi:
            tmp_max_npmi = npmi
            temp_item_i_max = item_i_freq
            temp_item_j_max = item_j_freq
            temp_i_j_max = pair_count
        if npmi < tmp_min_npmi:
            tmp_min_npmi = npmi
            temp_item_i_min = item_i_freq
            temp_item_j_min = item_j_freq
            temp_i_j_min = pair_count

        if pmi > tmp_max_pmi:
            tmp_max_pmi = pmi
        if pmi < tmp_min_pmi:
            tmp_min_pmi = pmi
    print("max_pmi:", tmp_max_pmi, "min_pmi:", tmp_min_pmi)
    print("max_npmi:", tmp_max_npmi, "min_npmi:", tmp_min_npmi)
    print("temp_item_i_max: ",temp_item_i_max, "temp_item_j_max: ",temp_item_j_max, "temp_i_j_max: ",temp_i_j_max)
    print("temp_item_i_min: ",temp_item_i_min, "temp_item_j_min: ",temp_item_j_min, "temp_i_j_min: ",temp_i_j_min)
    return item_pair_pmi



def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

class MyDataset(Dataset[Any]):
    def __init__(self, train_cf_pairs, train_user_set, n_items, num_neg_sample):
        self.train_cf_pairs = train_cf_pairs
        self.train_user_set = train_user_set
        self.n_items = n_items
        self.num_neg_sample = num_neg_sample

    def __len__(self):
        return len(self.train_cf_pairs)

    def __getitem__(self, idx):
        ui = self.train_cf_pairs[idx]

        u = int(ui[0])
        each_negs = list()
        neg_item = np.random.randint(low=0, high=self.n_items, size=self.num_neg_sample)
        neg_item = np.asarray(neg_item).reshape(-1).tolist()
        if len(set(neg_item) & set(self.train_user_set[u])) == 0:
            each_negs += neg_item
        else:
            neg_item = list(set(neg_item) - set(self.train_user_set[u]))
            each_negs += neg_item
            while len(each_negs) < self.num_neg_sample:
                n1 = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if n1 not in self.train_user_set[u]:
                    each_negs += [n1]


        return [ui[0], ui[1], each_negs]
