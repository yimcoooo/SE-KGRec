# -*- coding: utf-8 -*-
"""SE-KGRec."""
import os
import json

import sys
import math
import copy
import random
import torch
import itertools
import numpy as np
import pickle as pkl
from datetime import datetime
from math import log
from tqdm import tqdm
from time import time 
import multiprocessing
import scipy.sparse as sp
from collections import Counter, defaultdict

from utils.parser import parse_args
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from utils.data_loader import load_data
from sklearn.metrics.pairwise import cosine_similarity

from modules.EDKG import Recommender
from modules.pcgrad import PCGrad

from utils.evaluate import test
from utils.eval_stratified import evaluate_split, item_pop_buckets, user_act_buckets, ITEM_POP_NAMES
import utils.helper as helper_utils


DATASET_CONFIGS = {
    'last-fm': {
        'style_relation_id': 1,
        'similar_relation_id': 2,
        'default_use_x_fusion': False,
    },
    'lastfm_selfbulid': {
        'style_relation_id': 1,
        'similar_relation_id': 2,
        'default_use_x_fusion': True,
    },
    'yelp': {
        'style_relation_id': None,
        'similar_relation_id': None,
        'default_use_x_fusion': False,
    },
    'yelp_50k': {
        'style_relation_id': None,
        'similar_relation_id': None,
        'default_use_x_fusion': False,
    },
    'yelp_200k': {
        'style_relation_id': None,
        'similar_relation_id': None,
        'default_use_x_fusion': False,
    },
    'amazonbook': {
        'style_relation_id': None,
        'similar_relation_id': None,
        'default_use_x_fusion': False,
    },
    'amazonbook_200k': {
        'style_relation_id': None,
        'similar_relation_id': None,
        'default_use_x_fusion': False,
    },
}


cores = multiprocessing.cpu_count()
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_data(train_entity_pairs, train_user_set):

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]

    return feed_dict

def _sample_kgc_negative_triplets(triplets, rng):
    triplets = np.asarray(triplets, dtype=np.int64)
    if triplets.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    neg_triplets = np.empty_like(triplets, dtype=np.int64)
    for idx, (h, r, t) in enumerate(triplets):
        branch = rng.random()
        while True:
            if branch < 0.33:
                new_rel = int(rng.integers(1, n_relations))
                if new_rel != r:
                    neg_triplets[idx] = [h, new_rel, t]
                    break
            elif branch > 0.66:
                new_head = int(rng.integers(0, n_entities))
                if new_head != h:
                    neg_triplets[idx] = [new_head, r, t]
                    break
            else:
                new_t = int(rng.integers(n_items, n_entities))
                if new_t != t:
                    neg_triplets[idx] = [h, r, new_t]
                    break
    return neg_triplets


def _build_rel_index(triplets):
    """Build per-relation indices of real tail/head distributions (used for type-constrained hard negatives)."""
    rel_tails = defaultdict(set)
    rel_heads = defaultdict(set)
    for h, r, t in np.asarray(triplets, dtype=np.int64):
        rel_tails[int(r)].add(int(t))
        rel_heads[int(r)].add(int(h))
    rel_tails = {k: np.fromiter(v, dtype=np.int64) for k, v in rel_tails.items()}
    rel_heads = {k: np.fromiter(v, dtype=np.int64) for k, v in rel_heads.items()}
    return rel_tails, rel_heads


def _sample_kgc_negative_triplets_hard(triplets, rng, rel_tails, rel_heads, pos_set=None):
    """Hard negatives (unused when disabled):
    tails/heads are sampled from the real distribution of each relation (type-consistent, hence hard),
    instead of uniformly from the whole space (easy); false negatives that coincide with real KG triples are rejected.
    """
    triplets = np.asarray(triplets, dtype=np.int64)
    if triplets.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if pos_set is None:
        pos_set = set(map(tuple, triplets.tolist()))

    neg_triplets = np.empty_like(triplets, dtype=np.int64)
    for idx, (h, r, t) in enumerate(triplets):
        h = int(h); r = int(r); t = int(t)
        _assigned = False
        for _try in range(64):
            u = rng.random()
            if u < 0.34:
                cand = rel_tails.get(r)
                nt = int(cand[rng.integers(0, cand.size)]) if (cand is not None and cand.size > 0) else int(rng.integers(n_items, n_entities))
                if nt != t and (h, r, nt) not in pos_set:
                    neg_triplets[idx] = [h, r, nt]; _assigned = True; break
            elif u < 0.67:
                cand = rel_heads.get(r)
                nh = int(cand[rng.integers(0, cand.size)]) if (cand is not None and cand.size > 0) else int(rng.integers(0, n_entities))
                if nh != h and (nh, r, t) not in pos_set:
                    neg_triplets[idx] = [nh, r, t]; _assigned = True; break
            else:
                nr = int(rng.integers(1, n_relations))
                if nr != r and (h, nr, t) not in pos_set:
                    neg_triplets[idx] = [h, nr, t]; _assigned = True; break
        if not _assigned:
            neg_triplets[idx] = [h, r, int(rng.integers(n_items, n_entities))]
    return neg_triplets


def _get_KGC_neg_data(triplets, neg_times=4, seed=None, hard=False):
    triplets = np.asarray(triplets, dtype=np.int64)
    if triplets.size == 0 or neg_times <= 0:
        return np.empty((0, 3), dtype=np.int64)

    base_seed = int(args.seed if seed is None else seed)
    _rel_tails = _rel_heads = _pos_set = None
    if hard:
        _rel_tails, _rel_heads = _build_rel_index(triplets)
        _pos_set = set(map(tuple, triplets.tolist()))
    all_neg = []
    for round_idx in range(int(neg_times)):
        rng = np.random.default_rng(base_seed + round_idx)
        if hard:
            neg_triplets = _sample_kgc_negative_triplets_hard(triplets, rng, _rel_tails, _rel_heads, _pos_set)
        else:
            neg_triplets = _sample_kgc_negative_triplets(triplets, rng)
        all_neg.append(neg_triplets)
    return np.unique(np.concatenate(all_neg, axis=0), axis=0)


def _normalize_kg_mask(kg_mask):
    if kg_mask is None:
        return None
    if isinstance(kg_mask, list):
        if len(kg_mask) == 0:
            return None
        kg_mask = np.asarray(kg_mask)
    if torch.is_tensor(kg_mask):
        kg_mask = kg_mask.detach().cpu().numpy()
    kg_mask = np.asarray(kg_mask).reshape(-1)
    if kg_mask.size == 0:
        return None
    return kg_mask


def train_kgr_model(model,kgr_optimizer, triplets, kg_mask=None,epochs=2,threshold=0.5,neg_times=4,conf_dump_path=None):
    # neg_kg_data = torch.LongTensor(neg_kg_data[index])
    kg_mask = _normalize_kg_mask(kg_mask)
    if torch.is_tensor(triplets):
        triplets = triplets.detach().cpu().numpy()
    triplets = np.asarray(triplets)
    if kg_mask is not None:
        triplets = triplets[kg_mask > 0]

    kgr_batch = 1024
    kgr_seed = int(args.seed)
    kgr_rng = np.random.default_rng(kgr_seed)

    if kg_mask is None:
        pos_label = np.ones((triplets.shape[0],1))
    else:
        pos_label = kg_mask[kg_mask > 0].reshape(-1, 1)
    pos_data = np.concatenate([triplets, pos_label.astype(np.float32, copy=False)], axis=-1)
    if len(pos_data) > 1:
        pos_data = pos_data[kgr_rng.permutation(len(pos_data))]

    if len(pos_data) <= 1:
        pos_train = pos_data
        pos_valid = pos_data
        pos_valid_num = len(pos_data)
    else:
        pos_valid_num = max(1, int(len(pos_data) * 0.05))
        pos_valid_num = min(pos_valid_num, len(pos_data) - 1)
        pos_train = pos_data[:-pos_valid_num]
        pos_valid = pos_data[-pos_valid_num:]

    # kgr_optimizer = torch.optim.AdamW(kgr_model.parameters(), lr=0.001)
    model.train()
    
    kgr_train_data = None
    kgr_valid_data = None
    kgr_train_hr = None
    kgr_train_label = None
    kgr_valid_hr = None
    kgr_valid_label = None
    kgr_iter = 0
    kgr_val_iter = 0
    _last_acc = None
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        if epoch % 4 == 0:
            kgc_neg_data = _get_KGC_neg_data(triplets, neg_times=int(neg_times), seed=kgr_seed + epoch * max(1, int(neg_times)),
                                             hard=bool(getattr(args, 'kgc_hard_neg', False)))
            neg_label = np.zeros((kgc_neg_data.shape[0], 1), dtype=np.float32)
            neg_data = np.concatenate([kgc_neg_data, neg_label], axis=-1)
            
            neg_train = neg_data[:-pos_valid_num]
            neg_valid = neg_data[-pos_valid_num:]
            kgr_train_data = np.concatenate([pos_train, neg_train], axis=0)
            if len(kgr_train_data) > 1:
                kgr_train_data = kgr_train_data[kgr_rng.permutation(len(kgr_train_data))]
            pos_valid_hard = pos_valid.copy()
            pos_valid_hard[:, -1] = 1.0
            neg_valid_hard = neg_valid.copy()
            neg_valid_hard[:, -1] = 0.0
            kgr_valid_data = np.concatenate([pos_valid_hard, neg_valid_hard], axis=0)

            kgr_train_hr = torch.as_tensor(kgr_train_data[:, :3], dtype=torch.long)
            kgr_train_label = torch.as_tensor(kgr_train_data[:, 3], dtype=torch.float32)
            kgr_valid_hr = torch.as_tensor(kgr_valid_data[:, :3], dtype=torch.long)
            kgr_valid_label = torch.as_tensor(kgr_valid_data[:, 3], dtype=torch.float32)

            kgr_iter = math.ceil(len(kgr_train_hr) / kgr_batch)
            kgr_val_iter = math.ceil(len(kgr_valid_hr) / kgr_batch)

        if kgr_train_hr is None or kgr_valid_hr is None or kgr_train_label is None or kgr_valid_label is None:
            continue
        batch_kg = dict()
        total_kg_loss = 0
        for i in range(kgr_iter):
            batch_kg["hr_pair"] = kgr_train_hr[i * kgr_batch:(i + 1) * kgr_batch].to(device)
            batch_kg["label"] = kgr_train_label[i * kgr_batch:(i + 1) * kgr_batch].to(device)

            kgr_batch_loss = model(batch_kg,mode="kgc")
            total_kg_loss += kgr_batch_loss.item()

            kgr_optimizer.zero_grad()
            kgr_batch_loss.backward()
            kgr_optimizer.step()

        model.eval()
        pred_label = []
        true_label = []
        _conf_scores, _conf_hr, _conf_label = [], [], []
        for i in range(kgr_val_iter):
            batch_kg["hr_pair"] = kgr_valid_hr[i * kgr_batch:(i + 1) * kgr_batch].to(device)
            batch_kg["label"] = kgr_valid_label[i * kgr_batch:(i + 1) * kgr_batch].to(device)
            _raw = model.gcn.kgc(batch_kg,eval=True)
            if conf_dump_path is not None:
                _conf_scores.append(_raw.detach().cpu().numpy().reshape(-1))
                _conf_hr.append(batch_kg["hr_pair"].detach().cpu().numpy())
                _conf_label.append(batch_kg["label"].detach().cpu().numpy().reshape(-1))
            pre = (_raw >= threshold).squeeze(-1).float()
            pre = pre.detach().cpu().numpy()
            label = batch_kg["label"].detach().cpu().numpy()
            pred_label.append(pre)
            true_label.append(label)
        if conf_dump_path is not None and len(_conf_scores) > 0:
            try:
                np.savez_compressed(conf_dump_path,
                                    hr=np.concatenate(_conf_hr, 0),
                                    label=np.concatenate(_conf_label, 0).astype(np.float32),
                                    score=np.concatenate(_conf_scores, 0).astype(np.float32))
            except Exception as _e:
                print('[conf dump] failed: {}'.format(_e))
            
        pred_label = np.concatenate(pred_label,axis=0)
        true_label = np.concatenate(true_label,axis=0)
        true_label = (true_label >= 0.5).astype(np.float32)
        # pred_label_d = pred_label[pred_label>0]
        # true_label_d = true_label[pred_label>0]
        
        acc = accuracy_score(pred_label, true_label)
        _last_acc = float(acc)
        print('[kgc] kgr valid acc={:.4f} (neg_times={})'.format(_last_acc, int(neg_times)))
    return _last_acc

        
def _build_sem_item_edges(item_embeds, n_items, relation_id=1, topk=5, threshold=0.5, chunk=512):
    """Build optional semantic item-item edges (i, r, j) from cosine neighbours of LLM item embeddings."""
    if item_embeds is None:
        return None
    emb = np.asarray(item_embeds, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] < n_items:
        return None
    emb = emb[:n_items]
    nrm = np.linalg.norm(emb, axis=1)
    keep = nrm > 1e-6
    X = emb / np.maximum(nrm.reshape(-1, 1), 1e-9)
    topk = int(max(1, int(topk)))
    edges = []
    for s in range(0, n_items, int(chunk)):
        e = min(s + int(chunk), n_items)
        sim = X[s:e] @ X.T
        for bi in range(e - s):
            i = s + bi
            if not keep[i]:
                continue
            row = sim[bi]
            k = min(topk, n_items - 1)
            idx = np.argpartition(-row, k)[:k]
            for j in idx:
                j = int(j)
                if j != i and keep[j] and float(row[j]) >= float(threshold):
                    edges.append((int(i), int(relation_id), int(j)))
    if not edges:
        return None
    return np.unique(np.asarray(edges, dtype=np.int64), axis=0)


def _process_kg_attr(canditate_kg,tripltes,kg_mask=None):
    
    canditate_kg = np.unique(canditate_kg,axis=0)
    kg_mask = _normalize_kg_mask(kg_mask)
    if kg_mask is not None:
        tripltes = tripltes[kg_mask != 0]
        
    # attr_triplets = tripltes[tripltes[:,0]>n_items-1]
    attr_set = set(np.unique(canditate_kg[:,-1]))
    out_attr_kg = []
    for tirp in tqdm(tripltes):
        if tirp[0] in attr_set:
            out_attr_kg.append(tirp)
    out_attr_kg = np.asarray(out_attr_kg)
    if out_attr_kg.shape[0] > 0:
        all_candi_kg = np.concatenate([canditate_kg,out_attr_kg],axis=0)
    else:
        all_candi_kg = canditate_kg
    return np.unique(all_candi_kg,axis=0)


def _log_cuda_memory(stage, device, extra_msg=''):
    if device.type != 'cuda' or (not torch.cuda.is_available()):
        return
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    msg = '[cuda memory] {} allocated={:.2f}GB reserved={:.2f}GB max_allocated={:.2f}GB max_reserved={:.2f}GB'.format(
        stage, allocated, reserved, max_allocated, max_reserved
    )
    if extra_msg:
        msg = msg + ' ' + str(extra_msg)
    print(msg)


def _load_embedding_array(embed_path):
    embed_array = torch.load(embed_path, map_location='cpu')
    if torch.is_tensor(embed_array):
        embed_array = embed_array.detach().cpu().numpy()
    else:
        embed_array = np.asarray(embed_array)
    return embed_array.astype(np.float32, copy=False)


def _load_layer_availability_array(mask_path):
    mask_array = torch.load(mask_path, map_location='cpu')
    if torch.is_tensor(mask_array):
        mask_array = mask_array.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_array)
    mask_array = np.asarray(mask_array)
    if mask_array.ndim != 2 or mask_array.shape[1] < 3:
        raise ValueError('artist_layer_availability should have shape [num_items, 3], got {}'.format(mask_array.shape))
    return mask_array.astype(bool, copy=False)


def _load_relation_embeddings_bundle(metadata_path):
    metadata_path = os.path.abspath(metadata_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError('relation embedding metadata file not found: {}'.format(metadata_path))
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    metadata_dir = os.path.dirname(metadata_path)

    def _resolve_metadata_file_path(path_value):
        path_value = str(path_value).strip()
        if path_value == '':
            return ''
        if os.path.isabs(path_value):
            return os.path.abspath(path_value)
        return os.path.abspath(os.path.join(metadata_dir, path_value))

    relation_order = [str(name) for name in metadata.get('relation_order', [])]
    relation_files = dict(metadata.get('files', {}).get('relation_embeddings', {}))
    if len(relation_order) == 0 or len(relation_files) == 0:
        raise ValueError('invalid relation embedding metadata: missing relation_order or relation_embeddings files in {}'.format(metadata_path))

    availability_path = metadata.get('files', {}).get('item_relation_availability', '')
    overall_path = metadata.get('files', {}).get('item_overall_embeddings', '')
    relation_embeddings = {}
    for relation_name in relation_order:
        relation_path = relation_files.get(relation_name)
        if relation_path is None:
            continue
        relation_embeddings[str(relation_name)] = _load_embedding_array(_resolve_metadata_file_path(relation_path))

    relation_availability = None
    if str(availability_path).strip() != '':
        availability_path = _resolve_metadata_file_path(availability_path)
        relation_availability = torch.load(availability_path, map_location='cpu')
        if torch.is_tensor(relation_availability):
            relation_availability = relation_availability.detach().cpu().numpy()
        relation_availability = np.asarray(relation_availability).astype(bool, copy=False)

    overall_embeddings = None
    if str(overall_path).strip() != '':
        overall_embeddings = _load_embedding_array(_resolve_metadata_file_path(overall_path))

    return {
        'metadata_path': metadata_path,
        'metadata': metadata,
        'relation_order': relation_order,
        'relation_embeddings': relation_embeddings,
        'relation_availability': relation_availability,
        'overall_embeddings': overall_embeddings,
    }


def _resolve_default_embed_path(args, file_name):
    return os.path.abspath(os.path.join(args.data_path, args.dataset, 'llm_embedding', file_name))


def _get_dataset_config(dataset_name):
    dataset_key = str(dataset_name).strip()
    return DATASET_CONFIGS.get(
        dataset_key,
        {
            'style_relation_id': None,
            'similar_relation_id': None,
            'default_use_x_fusion': False,
        },
    )


def _resolve_dataset_path(data_path, dataset_name, relative_path=''):
    dataset_root = os.path.abspath(os.path.join(data_path, dataset_name))
    relative_path = str(relative_path).strip()
    if relative_path == '':
        return dataset_root
    return os.path.abspath(os.path.join(dataset_root, relative_path))


def _looks_like_dataset_relative_metadata_path(path_value):
    normalized_path = str(path_value).strip().replace('\\', '/').strip('/')
    if normalized_path == '':
        return False
    if normalized_path == 'llm_embedding/relation_embeddings/embedding_metadata.json':
        return True
    return normalized_path.endswith('/llm_embedding/relation_embeddings/embedding_metadata.json')


def _resolve_relation_embedding_metadata_path(args):
    raw_metadata_path = str(getattr(args, 'relation_embedding_metadata_path', '')).strip()
    raw_relation_embedding_dir = str(getattr(args, 'relation_embedding_dir', '')).strip()
    dataset_root = _resolve_dataset_path(args.data_path, args.dataset)
    dataset_metadata_path = _resolve_dataset_path(
        args.data_path,
        args.dataset,
        'llm_embedding/relation_embeddings/embedding_metadata.json',
    )

    if _looks_like_dataset_relative_metadata_path(raw_metadata_path):
        return dataset_metadata_path

    if raw_metadata_path != '':
        if os.path.isabs(raw_metadata_path):
            normalized_abs_path = os.path.abspath(raw_metadata_path)
            if os.path.exists(normalized_abs_path):
                try:
                    common_root = os.path.commonpath([normalized_abs_path, dataset_root])
                except ValueError:
                    common_root = ''
                if common_root != dataset_root:
                    rel_to_data_path = os.path.relpath(normalized_abs_path, os.path.abspath(args.data_path))
                    if _looks_like_dataset_relative_metadata_path(rel_to_data_path):
                        return dataset_metadata_path
            return normalized_abs_path
        return _resolve_dataset_path(args.data_path, args.dataset, raw_metadata_path)

    if raw_relation_embedding_dir != '':
        candidate_metadata_path = os.path.join(raw_relation_embedding_dir, 'embedding_metadata.json')
        if os.path.exists(candidate_metadata_path):
            return os.path.abspath(candidate_metadata_path)

    return ''



def _load_relation_name_to_actual_id(data_path, dataset_name):
    relation_list_path = os.path.join(data_path, dataset_name, 'relation_list.txt')
    if not os.path.exists(relation_list_path):
        return {}
    relation_name_to_id = {}
    with open(relation_list_path, 'r', encoding='utf-8') as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('\t', ' ').split()
            if len(parts) < 2:
                continue
            relation_name = str(parts[0]).strip()
            remap_id = int(parts[1])
            relation_name_to_id[relation_name] = remap_id + 1
    return relation_name_to_id


def _ordered_x_components(kg_completion_components):
    explicit_x_components = [component for component in ("x_style", "x_sim", "x_overall") if component in kg_completion_components]
    if len(explicit_x_components) > 0:
        return explicit_x_components
    if "fusion" in kg_completion_components:
        return ["x_style", "x_sim"]
    return []


def _normalize_candidate_generator_mode(mode):
    raw_mode = str(mode).strip().lower()
    if raw_mode == '':
        raw_mode = 'auto'
    supported_modes = {'auto', 'relation_aware', 'legacy_full_transfer'}
    if raw_mode not in supported_modes:
        raise ValueError('invalid kg_candidate_generator: {} (supported: {})'.format(raw_mode, sorted(supported_modes)))
    return raw_mode


def _kg_completion_uses_semantic_mode(kg_completion_components):
    semantic_components = {"semantic", "fusion", "x_style", "x_sim", "x_overall"}
    return len(semantic_components.intersection(set(kg_completion_components))) > 0


def _is_baseline_npmi_cosine_mode(kg_completion_components):
    return set(kg_completion_components) == {"npmi", "cosine"}



def _resolve_relation_metadata_path(args, kg_completion_components):
    explicit_metadata_path = str(getattr(args, 'relation_embedding_metadata_path', '')).strip()
    if explicit_metadata_path != '':
        return _resolve_dataset_path(args.data_path, args.dataset, explicit_metadata_path), True

    if not _kg_completion_uses_semantic_mode(kg_completion_components):
        return '', False

    relation_embedding_dir = str(getattr(args, 'relation_embedding_dir', '')).strip()
    if relation_embedding_dir != '':
        relation_embedding_dir = _resolve_dataset_path(args.data_path, args.dataset, relation_embedding_dir)
    else:
        relation_embedding_dir = _resolve_dataset_path(args.data_path, args.dataset, 'llm_embedding/relation_embeddings')

    default_relation_metadata_path = os.path.join(relation_embedding_dir, 'embedding_metadata.json')
    if os.path.exists(default_relation_metadata_path):
        return os.path.abspath(default_relation_metadata_path), False
    return '', False



def _validate_kg_completion_config(args, kg_completion_components):
    x_components = _ordered_x_components(kg_completion_components)
    relation_embedding_dir = str(getattr(args, 'relation_embedding_dir', '')).strip()
    relation_metadata_path = str(getattr(args, 'relation_embedding_metadata_path', '')).strip()
    overall_embed_path = str(getattr(args, 'overall_embed_path', '')).strip()
    has_multi_relation_semantics = (relation_embedding_dir != '') or (relation_metadata_path != '')
    has_overall_semantics = overall_embed_path != ''
    candidate_generator_mode = _normalize_candidate_generator_mode(getattr(args, 'kg_candidate_generator', 'auto'))
    semantic_mode_enabled = _kg_completion_uses_semantic_mode(kg_completion_components)

    effective_use_relation_aware = bool(args.use_relation_aware)
    if candidate_generator_mode == 'relation_aware':
        effective_use_relation_aware = True
    elif candidate_generator_mode == 'legacy_full_transfer':
        effective_use_relation_aware = False
    elif semantic_mode_enabled:
        effective_use_relation_aware = True

    effective_use_x_fusion = bool(args.use_x_fusion)
    if semantic_mode_enabled:
        effective_use_x_fusion = True

    baseline_npmi_cosine_mode = _is_baseline_npmi_cosine_mode(kg_completion_components)
    if baseline_npmi_cosine_mode:
        effective_use_relation_aware = False
        effective_use_x_fusion = False

    if semantic_mode_enabled and (not has_multi_relation_semantics):
        raise ValueError("semantic kg completion mode requires relation embedding metadata or relation embedding dir")

    if effective_use_relation_aware and (not effective_use_x_fusion):
        if len(x_components) > 0:
            raise ValueError("relation-aware + cosine run should not include fusion/x_style/x_sim/x_overall in kg_completion_mode")
        unsupported_components = set(kg_completion_components) - {"npmi", "cosine"}
        if len(unsupported_components) > 0:
            raise ValueError("relation-aware + cosine only supports npmi and cosine, got unsupported components: {}".format(sorted(unsupported_components)))
        if "cosine" not in kg_completion_components:
            raise ValueError("relation-aware without x fusion requires kg_completion_mode to include cosine")
    if effective_use_x_fusion and len(x_components) == 0 and (not has_multi_relation_semantics) and (not has_overall_semantics):
        raise ValueError("use_x_fusion=True requires fusion/x_style/x_sim/x_overall or auto-discovered relation embeddings")

    args.use_relation_aware = effective_use_relation_aware
    args.use_x_fusion = effective_use_x_fusion
    if semantic_mode_enabled:
        args.kg_candidate_generator = 'relation_aware'
    elif baseline_npmi_cosine_mode:
        args.kg_candidate_generator = 'legacy_full_transfer'
    return x_components


def _score_candidate_triples_with_kgc(model, candidate_triples, device, batch_size=131072):
    candidate_triples = np.asarray(candidate_triples, dtype=np.int64)
    if candidate_triples.size == 0:
        return np.empty((0,), dtype=np.float32)

    batch_size = max(1, int(batch_size))
    total_candidates = int(candidate_triples.shape[0])
    total_batches = math.ceil(total_candidates / batch_size)
    print('[kgc scoring] candidate_triples={} batch_size={} total_batches={}'.format(
        total_candidates, batch_size, total_batches
    ))

    model.eval()
    all_scores = np.empty((total_candidates,), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, total_candidates, batch_size), total=total_batches, desc='score candidate kg'):
            end = min(start + batch_size, total_candidates)
            batch_np = candidate_triples[start:end]
            batch_tensor = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            batch_scores = model.gcn.kgc(batch_tensor, eval=True, cf_train=True)
            all_scores[start:end] = batch_scores.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    return all_scores


def _summarize_scores(score_array):
    score_array = np.asarray(score_array, dtype=np.float32).reshape(-1)
    if score_array.size == 0:
        return {'mean': 0.0, 'p50': 0.0, 'p90': 0.0}
    return {
        'mean': float(np.mean(score_array)),
        'p50': float(np.percentile(score_array, 50)),
        'p90': float(np.percentile(score_array, 90)),
    }


def _format_relation_stats(stats_dict):
    if not stats_dict:
        return '{}'
    return '{' + ', '.join(['{}:{}'.format(int(k), int(v)) for k, v in sorted(stats_dict.items(), key=lambda kv: kv[0])]) + '}'


def _append_log_line(result_file_path, message):
    with open(result_file_path, 'a+', encoding='utf-8') as f:
        f.write(str(message) + '\n')


def _announce_result_file_path(result_file_path, stage='training record'):
    result_file_abs_path = os.path.abspath(result_file_path)
    print('{} will be saved to: {}'.format(stage, result_file_abs_path))
    return result_file_abs_path



def _set_global_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False


def _maybe_warn_determinism_env():
    cublas_workspace_config = str(os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')).strip()
    if cublas_workspace_config == '':
        print('[determinism] CUBLAS_WORKSPACE_CONFIG is not set. For stricter CUDA determinism, export it in shell before launch, e.g. CUBLAS_WORKSPACE_CONFIG=:4096:8')
    else:
        print('[determinism] CUBLAS_WORKSPACE_CONFIG={}'.format(cublas_workspace_config))


def _format_debug_value(value):
    if isinstance(value, float):
        return '{:.6f}'.format(value)
    return str(value)


def _emit_epoch0_batch_debug(epoch, batch_idx, batch_loss, batch_mmd_loss, snapshot, result_file_path=None):
    parts = [
        '[debug epoch0 batch]',
        'epoch={}'.format(epoch),
        'batch_idx={}'.format(batch_idx),
        'loss={}'.format(_format_debug_value(float(batch_loss.detach().cpu().item()))),
        'mmd={}'.format(_format_debug_value(float(batch_mmd_loss.detach().cpu().item()))),
    ]
    for key in (
        'cf_loss',
        'mmd_loss',
        'user_norm_mean',
        'pos_item_norm_mean',
        'all_item_norm_mean',
        'embedding_abs_mean',
        'embedding_abs_max',
        'kg_weight_mean',
        'kg_weight_min',
        'kg_weight_max',
        'n_edge_count',
    ):
        if key in snapshot:
            parts.append('{}={}'.format(key, _format_debug_value(snapshot[key])))
    message = ' '.join(parts)
    print(message)
    if result_file_path is not None:
        _append_log_line(result_file_path, message)

    


if __name__ == '__main__':
    """read args"""
    global args, device, train_user_set,kg_dict, item_lists_dict, ent_lists_dict
    args = parse_args()
    args.kg_candidate_generator = _normalize_candidate_generator_mode(getattr(args, 'kg_candidate_generator', 'auto'))
    if args.kg_candidate_generator == 'relation_aware':
        args.use_relation_aware = True
    elif args.kg_candidate_generator == 'legacy_full_transfer':
        args.use_relation_aware = False
        args.use_x_fusion = False
    dataset_cfg = _get_dataset_config(args.dataset)
    llm_embedding_dir = _resolve_dataset_path(args.data_path, args.dataset, 'llm_embedding')
    default_relation_embedding_dir = os.path.join(llm_embedding_dir, 'relation_embeddings')
    if str(getattr(args, 'relation_embedding_dir', '')).strip() == '' and os.path.isdir(default_relation_embedding_dir):
        args.relation_embedding_dir = os.path.abspath(default_relation_embedding_dir)
    args.relation_embedding_metadata_path = _resolve_relation_embedding_metadata_path(args)
    has_custom_embedding_path = any(
        str(getattr(args, attr_name, '')).strip() != ''
        for attr_name in (
            'semantic_embed_path',
            'style_embed_path',
            'similar_embed_path',
            'overall_embed_path',
            'layer_availability_path',
            'relation_embedding_dir',
            'relation_embedding_metadata_path',
        )
    )
    if (not has_custom_embedding_path) and (not os.path.isdir(llm_embedding_dir)):
        args.use_x_fusion = False
    _set_global_seed(args.seed)
    _maybe_warn_determinism_env()

    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"

    os.makedirs("./result", exist_ok=True)
    semantic_threshold_tag = 'semthr{}'.format(str(getattr(args, 'relation_semantic_threshold', getattr(args, 'style_semantic_threshold', 'na'))).replace('.', 'p'))
    result_file_path = os.path.join("./result", f"{args.dataset}_sekgrec_seed{args.seed}_{semantic_threshold_tag}_{run_id}.txt")

    model_ckpt_path = os.path.join(args.out_dir, f"model_{args.dataset}_{semantic_threshold_tag}_{run_id}.ckpt")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph, item_rel_mask, triplets, kg_dict = load_data(args)
    # item_pmi_dict = pkl.load(open("item_pair_pmi.pkl","rb"))
    item_pmi_dict = None
    if args.do_kgr:
        # ===== Leakage control (npmi_leak, off by default; identical to the old behaviour) =====
        #   default: NPMI uses train_user_set only;
        #   leak arm: NPMI uses the merged train+val+test interactions and does not write the PMI cache
        #             (so the shared <dataset>item_pair_pmi.pkl is not overwritten).
        _npmi_user_set = user_dict['train_user_set']
        _npmi_leak = bool(getattr(args, 'npmi_leak', False))
        if _npmi_leak:
            _merged = defaultdict(list)
            for _k in ('train_user_set', 'val_user_set', 'test_user_set'):
                for _u, _its in (user_dict.get(_k) or {}).items():
                    _merged[_u].extend(list(_its))
            _npmi_user_set = _merged
            print('[npmi_leak] LEAK ARM: NPMI computed over train+val+test ({} users)'.format(len(_merged)))
        item_pmi_dict = helper_utils._cal_npmi(_npmi_user_set)
        if not _npmi_leak:
            pkl.dump(item_pmi_dict, open(args.dataset + "item_pair_pmi.pkl", "wb"))
    
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']

    # ===== cold-item mask (cold_kg_only, off by default; degree counted on the training split) =====
    cold_item_mask = None
    if bool(getattr(args, 'cold_kg_only', False)):
        _train_pairs = np.asarray(train_cf, dtype=np.int64)
        _item_deg_train = np.bincount(_train_pairs[:, 1], minlength=int(n_items)).astype(np.int64)
        cold_item_mask = (_item_deg_train[:int(n_items)] == 0)
        print('[cold_kg_only] n_cold_items={} / n_items={} (train degree==0)'.format(
            int(cold_item_mask.sum()), int(n_items)))

    # ===== oracle injection: held-out true edges are used as an upper bound =====
    _holdout_edges = None
    if bool(getattr(args, 'inject_oracle_holdout', False)):
        _ho_frac = float(getattr(args, 'kg_holdout_frac', 0.0) or 0.0)
        _ho_path = os.path.join(args.data_path, args.dataset,
                                'kg_holdout_frac{}_seed{}.txt'.format(str(_ho_frac).replace('.', 'p'), int(args.seed)))
        if os.path.exists(_ho_path):
            _holdout_edges = np.asarray(np.loadtxt(_ho_path, dtype=np.int64), dtype=np.int64)
            if _holdout_edges.ndim == 1:
                _holdout_edges = _holdout_edges.reshape(1, -1)
            print('[oracle] loaded held-out TRUE edges n={} from {}'.format(int(_holdout_edges.shape[0]), _ho_path))
        else:
            print('[oracle] WARNING: holdout file not found: {}'.format(_ho_path))

    # ===== runtime context (extra statistics / validation / strata / cold start) =====
    Ks_eval = eval(args.Ks) if isinstance(args.Ks, str) else args.Ks
    log_extras = bool(getattr(args, 'log_extras', True))
    extras_dir = os.path.join("./result", "extras", run_id)
    if log_extras:
        os.makedirs(extras_dir, exist_ok=True)
    use_val_split = bool(getattr(args, 'use_val_split', True)) and ('val_user_set' in user_dict)
    item_bucket = user_bucket = cold_items = item_deg = None
    if log_extras:
        item_bucket, item_deg = item_pop_buckets(train_user_set, n_items)
        user_bucket = user_act_buckets(train_user_set, n_users)
        cold_items = set(int(i) for i in np.where(item_deg <= 1)[0].tolist())
        print('[rev extras] item_bucket={} cold_items={}'.format(
            {ITEM_POP_NAMES[i]: int(np.sum(item_bucket == i)) for i in range(len(ITEM_POP_NAMES))},
            len(cold_items)))
    # Injected-edge record file
    inject_fh = None
    if log_extras:
        inject_fh = open(os.path.join(extras_dir, "inject.tsv"), "w", encoding="utf-8")
        inject_fh.write("epoch\thead\trel\ttail\thead_deg\ttail_deg\n")
    injection_round_stats = []
    best_val_recall = -1.0
    best_kgc_acc = -1.0
    best_kgc_state = None
    last_ret = None
    best_test_extras = None
    prev_gate_val = None

    args.kg_completion_mode, kg_completion_components = helper_utils.normalize_kg_completion_mode(args.kg_completion_mode)
    resolved_relation_metadata_path, relation_metadata_explicitly_provided = _resolve_relation_metadata_path(args, kg_completion_components)
    args.relation_embedding_metadata_path = resolved_relation_metadata_path
    if _is_baseline_npmi_cosine_mode(kg_completion_components):
        args.use_relation_aware = False
        args.use_x_fusion = False
        args.kg_candidate_generator = 'legacy_full_transfer'

    x_components = _validate_kg_completion_config(args, kg_completion_components)
    print(
        '[kg completion config] mode={} generator={} use_relation_aware={} use_x_fusion={} x_components={} relation_semantic_threshold={} relation_struct_threshold={} relation_topk={} relation_lambda={}'.format(
            args.kg_completion_mode,
            args.kg_candidate_generator,
            bool(args.use_relation_aware),
            bool(args.use_x_fusion),
            x_components,
            args.relation_semantic_threshold,
            args.relation_struct_threshold,
            args.relation_topk,
            args.relation_lambda,
        )
    )

    semantic_item_embeds = None
    style_item_embeds = None
    similar_item_embeds = None
    overall_item_embeds = None
    style_item_availability = None
    similar_item_availability = None
    overall_item_availability = None
    relation_embedding_bundle = None
    relation_name_to_id = _load_relation_name_to_actual_id(args.data_path, args.dataset)

    relation_metadata_path = str(getattr(args, 'relation_embedding_metadata_path', '')).strip()
    if relation_metadata_path != '':
        relation_embedding_bundle = _load_relation_embeddings_bundle(relation_metadata_path)
        args.relation_embedding_metadata_path = relation_embedding_bundle['metadata_path']
        args.relation_embedding_dir = os.path.dirname(relation_embedding_bundle['metadata_path'])
        print(
            'load relation embedding bundle: {} relation_count={}'.format(
                relation_embedding_bundle['metadata_path'],
                len(relation_embedding_bundle['relation_embeddings']),
            )
        )
        if relation_metadata_explicitly_provided:
            print('use explicit relation embedding metadata path: {}'.format(args.relation_embedding_metadata_path))
        else:
            print('auto-discover relation embedding metadata for dataset {}: {}'.format(args.dataset, args.relation_embedding_metadata_path))
        if relation_embedding_bundle['overall_embeddings'] is not None:
            semantic_item_embeds = relation_embedding_bundle['overall_embeddings']
            overall_item_embeds = relation_embedding_bundle['overall_embeddings']
            if relation_embedding_bundle['relation_availability'] is not None:
                overall_item_availability = np.any(relation_embedding_bundle['relation_availability'], axis=1).astype(bool, copy=False)
            print('load overall relation embeddings: {}'.format(semantic_item_embeds.shape))

        if dataset_cfg['style_relation_id'] is None and len(relation_embedding_bundle['relation_embeddings']) > 0:
            if args.kg_candidate_generator == 'legacy_full_transfer':
                print('keep legacy_full_transfer mode for multi-relation dataset {}'.format(args.dataset))
            elif bool(args.use_relation_aware):
                print('keep relation-aware mode for multi-relation dataset {}'.format(args.dataset))
            else:
                print('keep relation-aware disabled for multi-relation dataset {} under mode={} generator={}'.format(
                    args.dataset,
                    args.kg_completion_mode,
                    args.kg_candidate_generator,
                ))

    if semantic_item_embeds is None and ('semantic' in kg_completion_components) and (not bool(args.use_x_fusion)):
        semantic_embed_path = str(getattr(args, 'semantic_embed_path', '')).strip()
        if semantic_embed_path == '':
            if relation_embedding_bundle is not None and relation_embedding_bundle['overall_embeddings'] is not None:
                semantic_item_embeds = relation_embedding_bundle['overall_embeddings']
                print('reuse overall relation embeddings as semantic embeddings: {}'.format(semantic_item_embeds.shape))
            else:
                semantic_embed_path = _resolve_default_embed_path(args, 'artist_combined_embeddings.pt')
        if semantic_item_embeds is None:
            semantic_embed_path = os.path.abspath(semantic_embed_path)
            if not os.path.exists(semantic_embed_path):
                raise FileNotFoundError('semantic embedding file not found: {}'.format(semantic_embed_path))
            semantic_item_embeds = _load_embedding_array(semantic_embed_path)
            args.semantic_embed_path = semantic_embed_path
            print('load semantic embeddings: {} {}'.format(semantic_embed_path, semantic_item_embeds.shape))
    else:
        print('skip combined semantic embeddings: use_x_fusion={} kg completion mode = {}'.format(args.use_x_fusion, args.kg_completion_mode))

    if relation_embedding_bundle is None and bool(args.use_x_fusion) and any(component in x_components for component in ('x_style', 'x_sim')):
        layer_availability_path = str(getattr(args, 'layer_availability_path', '')).strip()
        if layer_availability_path == '':
            layer_availability_path = _resolve_default_embed_path(args, 'artist_layer_availability.pt')
        layer_availability_path = os.path.abspath(layer_availability_path)
        if not os.path.exists(layer_availability_path):
            raise FileNotFoundError('artist layer availability file not found: {}'.format(layer_availability_path))
        layer_availability = _load_layer_availability_array(layer_availability_path)
        style_item_availability = layer_availability[:, 1]
        similar_item_availability = layer_availability[:, 2]
        args.layer_availability_path = layer_availability_path
        print(
            'load artist layer availability: {} shape={} style_available={} similar_available={}'.format(
                layer_availability_path,
                layer_availability.shape,
                int(np.sum(style_item_availability)),
                int(np.sum(similar_item_availability)),
            )
        )
    else:
        print('skip artist layer availability: use_x_fusion={} relation_embedding_bundle={} kg completion mode = {}'.format(args.use_x_fusion, relation_embedding_bundle is not None, args.kg_completion_mode))

    if relation_embedding_bundle is None and bool(args.use_x_fusion) and ('x_style' in kg_completion_components):
        style_embed_path = str(getattr(args, 'style_embed_path', '')).strip()
        if style_embed_path == '':
            style_embed_path = _resolve_default_embed_path(args, 'artist_style_embeddings.pt')
        style_embed_path = os.path.abspath(style_embed_path)
        if not os.path.exists(style_embed_path):
            raise FileNotFoundError('style embedding file not found: {}'.format(style_embed_path))
        style_item_embeds = _load_embedding_array(style_embed_path)
        args.style_embed_path = style_embed_path
        print('load style embeddings: {} {}'.format(style_embed_path, style_item_embeds.shape))
    else:
        print('skip style embeddings: use_x_fusion={} relation_embedding_bundle={} kg completion mode = {}'.format(args.use_x_fusion, relation_embedding_bundle is not None, args.kg_completion_mode))

    if relation_embedding_bundle is None and bool(args.use_x_fusion) and ('x_sim' in kg_completion_components):
        similar_embed_path = str(getattr(args, 'similar_embed_path', '')).strip()
        if similar_embed_path == '':
            similar_embed_path = _resolve_default_embed_path(args, 'artist_similar_artist_embeddings.pt')
        similar_embed_path = os.path.abspath(similar_embed_path)
        if not os.path.exists(similar_embed_path):
            raise FileNotFoundError('similar embedding file not found: {}'.format(similar_embed_path))
        similar_item_embeds = _load_embedding_array(similar_embed_path)
        args.similar_embed_path = similar_embed_path
        print('load similar embeddings: {} {}'.format(similar_embed_path, similar_item_embeds.shape))
    else:
        print('skip similar embeddings: use_x_fusion={} relation_embedding_bundle={} kg completion mode = {}'.format(args.use_x_fusion, relation_embedding_bundle is not None, args.kg_completion_mode))

    if relation_embedding_bundle is None and bool(args.use_x_fusion) and ('x_overall' in kg_completion_components):
        overall_embed_path = str(getattr(args, 'overall_embed_path', '')).strip()
        if overall_embed_path == '':
            relation_dir = str(getattr(args, 'relation_embedding_dir', '')).strip()
            relation_overall_path = ''
            if relation_dir != '':
                candidate_relation_overall_path = os.path.abspath(os.path.join(relation_dir, 'item_overall_embeddings.pt'))
                if os.path.exists(candidate_relation_overall_path):
                    relation_overall_path = candidate_relation_overall_path
            if relation_overall_path != '':
                overall_embed_path = relation_overall_path
            else:
                overall_embed_path = _resolve_default_embed_path(args, 'artist_combined_embeddings.pt')
        overall_embed_path = os.path.abspath(overall_embed_path)
        if not os.path.exists(overall_embed_path):
            raise FileNotFoundError('overall embedding file not found: {}'.format(overall_embed_path))
        overall_item_embeds = _load_embedding_array(overall_embed_path)
        args.overall_embed_path = overall_embed_path
        print('load overall embeddings: {} {}'.format(overall_embed_path, overall_item_embeds.shape))
        if overall_item_availability is None and overall_item_embeds is not None:
            overall_item_availability = np.linalg.norm(np.asarray(overall_item_embeds, dtype=np.float32), axis=1) > 0
    else:
        print('skip overall embeddings: use_x_fusion={} relation_embedding_bundle={} kg completion mode = {}'.format(args.use_x_fusion, relation_embedding_bundle is not None, args.kg_completion_mode))

    style_relation_id = dataset_cfg['style_relation_id']
    similar_relation_id = dataset_cfg['similar_relation_id']



    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    """define model"""
    # ===== Optional LLM semantic item vectors for cold-item initialisation (--cold_sem_init, off by default) =====
    _sem_item_embeds = None
    if bool(getattr(args, 'cold_sem_init', False)):
        _sem_path = os.path.join(args.data_path, args.dataset, 'llm_embedding',
                                 'relation_embeddings', 'item_overall_embeddings.pt')
        if os.path.exists(_sem_path):
            _sem_item_embeds = torch.load(_sem_path, map_location='cpu').float().numpy()
            print('[sem_init] loaded LLM item embeddings {} from {}'.format(_sem_item_embeds.shape, _sem_path))
        else:
            print('[sem_init] WARNING: not found {}'.format(_sem_path))
    model = Recommender(n_params, args, graph, ui_sparse_graph,item_rel_mask,
                        sem_item_embeds=_sem_item_embeds,
                        cold_item_mask=cold_item_mask).to(device)

    # ===== optional semantic item-item edges (off by default) =====
    _sem_edges = None
    if bool(getattr(args, 'inject_semantic_item_edges', False)):
        _sem_src = overall_item_embeds if overall_item_embeds is not None else semantic_item_embeds
        _sem_edges = _build_sem_item_edges(_sem_src, n_items,
                                           relation_id=int(getattr(args, 'sem_edge_relation_id', 1)),
                                           topk=int(getattr(args, 'sem_edge_topk', 5)),
                                           threshold=float(getattr(args, 'sem_edge_threshold', 0.5)))
        if _sem_edges is not None:
            _sem_edges = np.unique(np.concatenate([_sem_edges, _sem_edges[:, [2, 1, 0]]], axis=0), axis=0)  # symmetrize
            print('[sem-edge] built {} semantic item-item edges (topk={} thr={} target={})'.format(
                int(_sem_edges.shape[0]), int(getattr(args, 'sem_edge_topk', 5)),
                float(getattr(args, 'sem_edge_threshold', 0.5)), str(getattr(args, 'sem_edge_target', 'ch2'))))
        else:
            print('[sem-edge] WARNING: no semantic edges built')
    if _sem_edges is not None and str(getattr(args, 'sem_edge_target', 'ch2')) == 'ch2':
        _n_added = model._append_original_knowledge(_sem_edges)
        print('[sem-edge] appended {} edges into ch2 graph (now {} edges)'.format(
            _n_added, int(model.edge_index.shape[1])))
    """define optimizer"""
    optimizer_base = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = PCGrad(optimizer_base)
    
    # ===== scorer optimizer scope =====
    _kgr_lr = float(getattr(args, 'kgr_lr', 0.001))
    if str(getattr(args, 'kgr_optimizer_scope', 'kgc')) == 'kgc':
        kgr_optimizer = torch.optim.AdamW(model.gcn.kgc.parameters(), lr=_kgr_lr)
        print('[kgc] kgr_optimizer scope = gcn.kgc only, lr={}'.format(_kgr_lr))
    else:
        kgr_optimizer = torch.optim.AdamW(model.parameters(), lr=_kgr_lr)
        print('[kgc] kgr_optimizer scope = ALL model params, lr={}'.format(_kgr_lr))  
    
    cur_best = 0
    stopping_step = 0
    should_stop = False
    best_metric = {"recall":0.0, "ndcg":0.0, "precision":0.0, "hit_ratio":0.0}
    best_epoch  = {"recall":0, "ndcg":0, "precision":0, "hit_ratio":0}
    result_file_abs_path = _announce_result_file_path(result_file_path)
    print('semantic threshold tag: {} style_semantic_threshold={} similar_semantic_threshold={} semantic_threshold_list={}'.format(
        semantic_threshold_tag,
        args.style_semantic_threshold,
        args.similar_semantic_threshold,
        getattr(args, 'semantic_threshold_list', ''),
    ))
    if args.save:

        print('model checkpoint will be saved to: {}'.format(os.path.abspath(model_ckpt_path)))
    else:
        print('model checkpoint saving is disabled, enable it with --save True')
    args_log_dict = dict(sorted(vars(args).items(), key=lambda kv: kv[0]))
    args_log_dict.pop('inject_append', None)  # the main method always uses append; hidden from the logged args
    with open(result_file_abs_path, 'w', encoding='utf-8') as f:
        f.write('args: ' + str(args_log_dict) + '\n')
        f.write('run_id: ' + str(run_id) + '\n\n')
    _log_cuda_memory('before training loop', device)
    
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)
    cl_batch = 512
    cl_iter = math.ceil(n_items / cl_batch)
    item_embs = []
    KG_mask = []
    model.eval()
    try:
        with torch.no_grad():
            init_item_embs, init_kg_mask = model.generate(for_kgc=True)
        item_embs = init_item_embs.detach().cpu().numpy()
        KG_mask = init_kg_mask.detach().cpu().numpy()
        print('[kg completion warm start] initialize structural item embeddings with model.generate(for_kgc=True) {}'.format(item_embs.shape))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        item_embs = model.all_embed[model.n_users:model.n_users + n_items].detach().cpu().numpy()
        KG_mask = []
        print('[kg completion warm start] model.generate(for_kgc=True) OOM, fall back to raw item embeddings for epoch 0 fusion {}'.format(item_embs.shape))
    model.train()
    _log_cuda_memory('after initial generate', device)
    
    all_feed_data = None
    epoch = 0
    for epoch in range(int(args.epoch)):
        torch.cuda.empty_cache()
        _log_cuda_memory('epoch {} start'.format(epoch), device)
        if epoch % 10 == 0 or epoch==0:
        # if epoch == 100 or epoch == 0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}
            all_feed_data['pos_index'] = torch.LongTensor(index)

        if args.do_kgr and (epoch % int(args.kgr_every) == 0):
            candi_kg, candi_meta = helper_utils._generate_candi_kg(
                item_pmi_dict,
                n_items,
                kg_dict,
                item_embeds=item_embs,
                semantic_item_embeds=semantic_item_embeds,
                style_item_embeds=style_item_embeds,
                similar_item_embeds=similar_item_embeds,
                overall_item_embeds=overall_item_embeds,
                style_item_availability=style_item_availability,
                similar_item_availability=similar_item_availability,
                overall_item_availability=overall_item_availability,
                relation_item_embeds=(None if relation_embedding_bundle is None else relation_embedding_bundle['relation_embeddings']),
                relation_item_availability=(None if relation_embedding_bundle is None else relation_embedding_bundle['relation_availability']),
                relation_name_to_id=relation_name_to_id,
                relation_order=(None if relation_embedding_bundle is None else relation_embedding_bundle['relation_order']),
                kg_completion_mode=args.kg_completion_mode,
                use_relation_aware=args.use_relation_aware,
                use_x_fusion=args.use_x_fusion,
                pmi_threshold=args.kgr_npmi_threshold,
                cos_threshold=args.kgr_cos_threshold,
                semantic_threshold=args.kgr_semantic_threshold,
                pmi_weight=args.kgr_npmi_weight,
                cos_weight=args.kgr_cos_weight,
                semantic_weight=args.kgr_semantic_weight,
                topk_neighbors=args.kgr_topk_neighbors,
                style_relation_id=style_relation_id,
                similar_relation_id=similar_relation_id,
                style_struct_threshold=args.style_struct_threshold,
                style_semantic_threshold=args.style_semantic_threshold,
                similar_struct_threshold=args.similar_struct_threshold,
                similar_semantic_threshold=args.similar_semantic_threshold,
                overall_struct_threshold=args.overall_struct_threshold,
                overall_semantic_threshold=args.overall_semantic_threshold,
                style_lambda=args.style_lambda,
                similar_lambda=args.similar_lambda,
                style_topk=args.style_topk,
                similar_topk=args.similar_topk,
                relation_struct_threshold=args.relation_struct_threshold,
                relation_semantic_threshold=args.relation_semantic_threshold,
                relation_lambda=args.relation_lambda,
                relation_topk=args.relation_topk,
                fusion_alpha=args.fusion_alpha,
            )

            raw_candi_kg_edge_num = int(candi_kg.shape[0])
            if bool(getattr(args, 'disable_aux_edges', False)):
                all_candi_kg = np.unique(np.asarray(candi_kg, dtype=np.int64), axis=0)
            else:
                all_candi_kg = _process_kg_attr(candi_kg,triplets)
            candi_kg_edge_num = int(all_candi_kg.shape[0])
            print(
                '[kg completion path] generator={} use_relation_aware={} use_x_fusion={} similarity_source={}'.format(
                    candi_meta.get('generator'),
                    candi_meta.get('use_relation_aware'),
                    candi_meta.get('use_x_fusion'),
                    candi_meta.get('similarity_source'),
                )
            )
            if candi_meta.get('merge_formula'):
                print('[kg completion path] merge_formula={}'.format(candi_meta.get('merge_formula')))


            print('[candidate stats] pair_stats={}'.format(candi_meta.get('pair_stats', {})))
            print('[candidate stats] relation_counts_before_attr={}'.format(_format_relation_stats(candi_meta.get('relation_stats', {}))))
            print('[candidate stats] raw_candidate_edges={} after_attr_edges={}'.format(raw_candi_kg_edge_num, candi_kg_edge_num))
            if 'branch_stats' in candi_meta:
                print('[candidate stats] branch_stats={}'.format(candi_meta['branch_stats']))
            kgr_epoch = int(args.kgr_epoch0) if epoch == 0 else int(args.kgr_epoch)
            _freeze_kgc = bool(getattr(args, 'kgc_freeze_after_first', True))
            if (not _freeze_kgc) or (epoch == 0):
                _conf_dump = os.path.join(extras_dir, 'confscores.npz') if log_extras else None
                _acc = train_kgr_model(model, kgr_optimizer, triplets, kg_mask=KG_mask, epochs=kgr_epoch,
                                neg_times=int(args.kgr_neg_times), conf_dump_path=_conf_dump)
                if bool(getattr(args, 'kgc_keep_best', False)):
                    if (best_kgc_state is None) or (_acc is not None and float(_acc) > best_kgc_acc):
                        best_kgc_acc = (-1.0 if _acc is None else float(_acc))
                        best_kgc_state = copy.deepcopy(model.gcn.kgc.state_dict())
                        print('[kgc] saved best scorer snapshot acc={:.4f}'.format(best_kgc_acc))
                    else:
                        model.gcn.kgc.load_state_dict(best_kgc_state)
                        print('[kgc] restored best scorer snapshot acc={:.4f}'.format(best_kgc_acc))
            else:
                print('[kgc] scorer frozen after first round; skip retraining at epoch {}'.format(epoch))
            print('[candidate stats] before_kgc_total={} by_relation={}'.format(candi_kg_edge_num, _format_relation_stats({int(rel): int(np.sum(all_candi_kg[:, 1] == rel)) for rel in np.unique(all_candi_kg[:, 1]).tolist()}) if all_candi_kg.shape[0] > 0 else '{}'))

            filtered_candi_kg = all_candi_kg
            kgc_filter_stats = {
                'before_total': int(all_candi_kg.shape[0]),
                'after_total': int(all_candi_kg.shape[0]),
                'before_by_relation': {int(rel): int(np.sum(all_candi_kg[:, 1] == rel)) for rel in np.unique(all_candi_kg[:, 1]).tolist()} if all_candi_kg.shape[0] > 0 else {},
                'after_by_relation': {int(rel): int(np.sum(all_candi_kg[:, 1] == rel)) for rel in np.unique(all_candi_kg[:, 1]).tolist()} if all_candi_kg.shape[0] > 0 else {},
                'retention_by_relation': {int(rel): 1.0 for rel in np.unique(all_candi_kg[:, 1]).tolist()} if all_candi_kg.shape[0] > 0 else {},
            }
            kgc_score_summary = {'mean': 0.0, 'p50': 0.0, 'p90': 0.0}

            if all_candi_kg.shape[0] > 0:
                kgc_scores = _score_candidate_triples_with_kgc(
                    model,
                    all_candi_kg,
                    device,
                    batch_size=int(getattr(args, 'kgc_score_batch_size', 131072)),
                )
                kgc_score_summary = _summarize_scores(kgc_scores)
                filtered_candi_kg, kgc_filter_stats = helper_utils._filter_candidates_with_kgc(
                    all_candi_kg,
                    kgc_scores,
                    score_threshold=args.kgc_filter_threshold,
                    topk_per_head=args.kgc_filter_topk,
                    selection=str(getattr(args, 'inject_selection', 'scored')),
                    rng=np.random.default_rng(int(args.seed) + int(epoch)),
                )
            print('[kgc filter] score_summary={} threshold={} topk={}'.format(kgc_score_summary, args.kgc_filter_threshold, args.kgc_filter_topk))
            print('[kgc filter] before_total={} after_total={}'.format(kgc_filter_stats['before_total'], kgc_filter_stats['after_total']))
            print('[kgc filter] before_by_relation={}'.format(_format_relation_stats(kgc_filter_stats['before_by_relation'])))
            print('[kgc filter] after_by_relation={}'.format(_format_relation_stats(kgc_filter_stats['after_by_relation'])))
            print('[kgc filter] retention_by_relation={}'.format(kgc_filter_stats['retention_by_relation']))

            _append_log_line(result_file_path, '[epoch {}] candidate_pair_stats={}'.format(epoch, candi_meta.get('pair_stats', {})))
            _append_log_line(result_file_path, '[epoch {}] candidate_relation_counts={}'.format(epoch, candi_meta.get('relation_stats', {})))
            _append_log_line(result_file_path, '[epoch {}] candidate_branch_stats={}'.format(epoch, candi_meta.get('branch_stats', {})))
            _append_log_line(
                result_file_path,
                '[epoch {}] candidate_path={}'.format(
                    epoch,
                    {
                        'generator': candi_meta.get('generator'),
                        'use_relation_aware': candi_meta.get('use_relation_aware'),
                        'use_x_fusion': candi_meta.get('use_x_fusion'),
                        'similarity_source': candi_meta.get('similarity_source'),
                        'merge_formula': candi_meta.get('merge_formula'),
                        'raw_candidate_edges': raw_candi_kg_edge_num,
                        'after_attr_edges': candi_kg_edge_num,
                    },
                ),
            )
            _append_log_line(result_file_path, '[epoch {}] kgc_filter_stats={}'.format(epoch, kgc_filter_stats))
            _append_log_line(result_file_path, '[epoch {}] kgc_score_summary={}'.format(epoch, kgc_score_summary))

            if _holdout_edges is not None and int(_holdout_edges.shape[0]) > 0:
                # oracle: replace the selected set with held-out true edges (precision/recall = 100%)
                filtered_candi_kg = _holdout_edges
                print('[oracle] inject held-out TRUE edges n={}'.format(int(_holdout_edges.shape[0])))
            elif _sem_edges is not None and str(getattr(args, 'sem_edge_target', 'ch2')) == 'ch3':
                # optional: replace the selected set with semantic item-item edges (target=ch3)
                filtered_candi_kg = _sem_edges
                print('[sem-edge] inject {} semantic edges into ch3'.format(int(_sem_edges.shape[0])))
            if filtered_candi_kg.shape[0] == 0:
                print('[kgc filter] no candidate triples kept after filtering, skip _update_knowledge and keep previous injected KG')
                _append_log_line(result_file_path, '[epoch {}] kgc filter produced empty candidate set; skip update'.format(epoch))
            else:
                _new_edges = filtered_candi_kg
                # append mode: channel 3 = original KG union injected edges (deduplicated)
                _update_edges = _new_edges
                if bool(getattr(args, 'inject_append', False)):
                    _update_edges = np.unique(np.concatenate(
                        [np.asarray(triplets, dtype=np.int64), np.asarray(_new_edges, dtype=np.int64)], axis=0), axis=0)
                    print('[inject append] channel3 edges = orig_kg({}) + injected({}) -> {}'.format(
                        int(np.asarray(triplets).shape[0]), int(np.asarray(_new_edges).shape[0]), int(_update_edges.shape[0])))
                _accepted = True
                # ===== Validation-gated injection =====
                if (bool(getattr(args, 'val_gated_injection', False)) and use_val_split
                        and len(user_dict.get('val_user_set', {})) > 0):
                    _prev = model.gcn.get_injected_state()
                    model.gcn._update_knowledge(torch.LongTensor(_update_edges).to(device))
                    model.eval()
                    with torch.no_grad():
                        _vr, _ = evaluate_split(model, user_dict, n_params, 'val', Ks_eval, device,
                                                batch_size=int(args.test_batch_size), test_flag=args.test_flag,
                                                eval_users=int(getattr(args, 'eval_users', 0)), progress=False)
                    model.train()
                    _vndcg = float(np.asarray(_vr['ndcg'])[0])
                    if prev_gate_val is None or _vndcg > prev_gate_val + float(getattr(args, 'val_gate_min_delta', 0.0)):
                        prev_gate_val = _vndcg
                        _accepted = True
                    else:
                        model.gcn.set_injected_state(_prev)
                        _accepted = False
                    print('[val gate] epoch {} val_ndcg={:.6f} accepted={}'.format(epoch, _vndcg, _accepted))
                    _append_log_line(result_file_path,
                                     '[epoch {}] val_gate val_ndcg={:.6f} accepted={}'.format(epoch, _vndcg, _accepted))
                else:
                    model.gcn._update_knowledge(torch.LongTensor(_update_edges).to(device))

                # ===== Injected-edge statistics (written to file only) =====
                if log_extras and inject_fh is not None:
                    for _row in _new_edges:
                        _h, _r, _t = int(_row[0]), int(_row[1]), int(_row[2])
                        _hd = int(item_deg[_h]) if (item_deg is not None and 0 <= _h < len(item_deg)) else -1
                        _td = int(item_deg[_t]) if (item_deg is not None and 0 <= _t < len(item_deg)) else -1
                        inject_fh.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, _h, _r, _t, _hd, _td))
                    inject_fh.flush()
                    _heads = _new_edges[:, 0]
                    _uniq, _cnt = np.unique(_heads, return_counts=True)
                    _kg = int(args.kgc_filter_topk)
                    _full = int(np.sum(_cnt >= _kg)) if _kg > 0 else 0
                    injection_round_stats.append({
                        'epoch': int(epoch), 'n_edges': int(_new_edges.shape[0]),
                        'n_heads': int(_uniq.shape[0]), 'quota_topk': _kg,
                        'fill_rate_full': float(_full / max(1, int(_uniq.shape[0]))),
                        'mean_head_count': float(np.mean(_cnt)) if _cnt.size else 0.0,
                        'accepted': bool(_accepted),
                    })
            _log_cuda_memory('epoch {} after kg update'.format(epoch), device, extra_msg='candidate_edges_before={} candidate_edges_after={}'.format(candi_kg_edge_num, kgc_filter_stats['after_total']))
            
        """training"""
        model.train()
        loss = 0
        train_s_t = time()

        if all_feed_data is None:
            continue
        for i in tqdm(range(iter)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch = dict()
            batch['pos_index'] = all_feed_data['pos_index'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)



            batch_loss, batch_mmd_loss = model(batch)
            loss_list  = [batch_mmd_loss, batch_loss]
            optimizer.pc_backward(loss_list)
            optimizer.step()
            loss += batch_loss.item()

            if epoch == 0 and int(getattr(args, 'debug_epoch0_batches', 0)) > 0 and i < int(args.debug_epoch0_batches):
                snapshot = model.debug_epoch0_snapshot(batch)
                _emit_epoch0_batch_debug(epoch, i, batch_loss, batch_mmd_loss, snapshot, result_file_path=result_file_path)


            if int(getattr(args, 'smoke_steps', 0)) > 0 and (i + 1) >= int(args.smoke_steps):
                break

        # if epoch > 8:
        #     for param_group in optimizer.param_groups:
        #             param_group['lr'] = 0.0001

        train_e_t = time()
        # scheduler.step()                                         
        item_embs, KG_mask = model.generate(for_kgc=True)
        item_embs = item_embs.detach().cpu().numpy()
        KG_mask = KG_mask.detach().cpu().numpy()
        kg_edge_num = None
        if model.gcn.n_edge_index is not None:
            kg_edge_num = int(model.gcn.n_edge_index.shape[1])
        _log_cuda_memory('epoch {} after generate'.format(epoch), device, extra_msg='stored_kg_edges={}'.format(kg_edge_num))
        # if epoch > 4 :
        """testing"""
        if bool(getattr(args, 'skip_test', False)):
            continue
        model.eval()
        test_s_t = time()
        _monitor = None
        with torch.no_grad():
            if use_val_split and len(user_dict.get('val_user_set', {})) > 0:
                _val_res, _ = evaluate_split(model, user_dict, n_params, 'val', Ks_eval, device,
                                             batch_size=int(args.test_batch_size), test_flag=args.test_flag,
                                             eval_users=int(getattr(args, 'eval_users', 0)), progress=False)
                _monitor = float(np.asarray(_val_res['recall'])[0])
        _improved = (not use_val_split) or (_monitor is not None and _monitor > best_val_recall)
        ret = None
        if _improved:
            with torch.no_grad():
                ret, _extras = evaluate_split(
                    model, user_dict, n_params, 'test', Ks_eval, device,
                    batch_size=int(args.test_batch_size), test_flag=args.test_flag,
                    eval_users=int(getattr(args, 'eval_users', 0)),
                    collect_extras=bool(log_extras), item_bucket=item_bucket,
                    user_bucket=user_bucket, cold_items=cold_items, progress=True)
            for k in best_metric.keys():
                metric_value = ret[k]
                if isinstance(metric_value, np.ndarray):
                    metric_value = metric_value.tolist()
                if isinstance(metric_value, (list, tuple)):
                    metric_value = metric_value[0]
                cur_metric = float(metric_value)
                if cur_metric > float(best_metric[k]):
                    best_metric[k] = cur_metric
                    best_epoch[k] = epoch
            if _monitor is not None:
                best_val_recall = _monitor
            last_ret = ret
            best_test_extras = _extras
        test_e_t = time()
        _row_ret = ret if ret is not None else (last_ret if last_ret is not None else
                                                {'recall': 0.0, 'ndcg': 0.0, 'precision': 0.0, 'hit_ratio': 0.0})
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
        train_res.add_row(
            [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, _row_ret['recall'], _row_ret['ndcg'], _row_ret['precision'], _row_ret['hit_ratio']]
        )

        with open(result_file_abs_path, 'a+', encoding='utf-8') as f:
            f.write(str(best_metric)+ '\n')
            f.write(str(best_epoch)+ '\n')
            f.write(str(train_res) + '\n')
            if _monitor is not None:
                f.write("'val_recall': {:.6f}\n".format(_monitor))
            f.write('\n')

        # *********************************************************
        recall_value = _monitor if _monitor is not None else (ret['recall'][0] if isinstance(ret['recall'], (list, tuple, np.ndarray)) else ret['recall'])
        cur_best, stopping_step, should_stop = helper_utils.early_stopping(recall_value, cur_best,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=20)
        
        if should_stop:
            break
        """save model"""
        if recall_value == cur_best and args.save:
            torch.save(model.state_dict(), model_ckpt_path)
        _log_cuda_memory('epoch {} end'.format(epoch), device)
              
    # ===== output extra statistics =====
    if log_extras:
        try:
            if inject_fh is not None:
                inject_fh.close()
            with open(os.path.join(extras_dir, "quota_rounds.json"), "w", encoding="utf-8") as f:
                json.dump(injection_round_stats, f)
            if best_test_extras is not None:
                _pu_r = best_test_extras.get('per_user_recall20', {})
                _pu_n = best_test_extras.get('per_user_ndcg20', {})
                _u = np.array(sorted(_pu_r.keys()), dtype=np.int64)
                np.savez_compressed(os.path.join(extras_dir, "per_user.npz"), uid=_u,
                                    recall20=np.array([_pu_r[int(x)] for x in _u], dtype=np.float32),
                                    ndcg20=np.array([_pu_n.get(int(x), np.nan) for x in _u], dtype=np.float32))
                with open(os.path.join(extras_dir, "strata.json"), "w", encoding="utf-8") as f:
                    json.dump({'item_pop': best_test_extras.get('strata_item', {}),
                               'user_act': best_test_extras.get('strata_user', {})}, f, ensure_ascii=False)
                with open(os.path.join(extras_dir, "coldstart.json"), "w", encoding="utf-8") as f:
                    json.dump(best_test_extras.get('coldstart', {}), f)
                _tk = best_test_extras.get('topk_items', {})
                np.savez_compressed(os.path.join(extras_dir, "topk.npz"),
                                    **{str(int(k)): np.asarray(v) for k, v in _tk.items()})
            print('[rev extras] written to {}'.format(os.path.abspath(extras_dir)))
        except Exception as _e:
            print('[rev extras] failed to write extras: {}'.format(_e))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))
    _announce_result_file_path(result_file_abs_path, stage='training record')