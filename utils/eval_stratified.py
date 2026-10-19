# -*- coding: utf-8 -*-
"""
eval_stratified.py — stratified and cold-start evaluation utilities (self-contained).

evaluate_split() provides:
  - Recall/NDCG/... on the given split ('val' / 'test'), with the same definitions as evaluate.test
  - optional extras (training-neutral): per-user metrics, item-popularity strata, user-activity strata,
    cold-start strata, and top-K lists (for overlap diagnostics).
  - not used in training and does not consume training randomness.

ranklist / get_performance are reused from utils.evaluate to keep metric definitions consistent.
"""
import numpy as np
import torch
import heapq
from tqdm import tqdm

from utils.evaluate import get_performance

# Stratum boundaries
ITEM_POP_EDGES = [5, 20, 50]      # [1,5) [5,20) [20,50) [50,inf)
ITEM_POP_NAMES = ['pop_1_5', 'pop_5_20', 'pop_20_50', 'pop_50_inf']
USER_ACT_EDGES = [10, 30, 100]    # [0,10) [10,30) [30,100) [100,inf)
USER_ACT_NAMES = ['act_0_10', 'act_10_30', 'act_30_100', 'act_100_inf']


def _bucket(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def item_pop_buckets(train_user_set, n_items):
    """Item-popularity buckets by training-set interaction count. Returns (bucket_id[n_items], deg[n_items])."""
    deg = np.zeros(n_items, dtype=np.int64)
    for u, items in train_user_set.items():
        for i in items:
            if 0 <= int(i) < n_items:
                deg[int(i)] += 1
    bucket = np.array([_bucket(int(d), ITEM_POP_EDGES) for d in deg], dtype=np.int64)
    return bucket, deg


def user_act_buckets(train_user_set, n_users):
    """User-activity buckets by training-set interaction count. Returns bucket_id[n_users]."""
    act = np.zeros(n_users, dtype=np.int64)
    for u, items in train_user_set.items():
        u = int(u)
        if 0 <= u < n_users:
            act[u] = len(items)
    bucket = np.array([_bucket(int(a), USER_ACT_EDGES) for a in act], dtype=np.int64)
    return bucket


def evaluate_split(model, user_dict, n_params, split, Ks, device, batch_size=2048,
                   test_flag='part', eval_users=0, collect_extras=False,
                   item_bucket=None, user_bucket=None, cold_items=None,
                   overlap_k=20, progress=True):
    """
    Returns (result, extras).
      result: {'recall': np.ndarray(len(Ks)), 'ndcg':..., 'precision':..., 'hit_ratio':...}
      extras: None when collect_extras=False; otherwise contains
              per_user_recall20/per_user_ndcg20、strata_item/user、coldstart、topk_items
    """
    n_items = int(n_params['n_items'])
    n_users = int(n_params['n_users'])
    train_user_set = user_dict['train_user_set']
    user_set = user_dict.get(split + '_user_set')
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)), 'hit_ratio': np.zeros(len(Ks))}
    if user_set is None or len(user_set) == 0:
        return result, ({} if collect_extras else None)

    Kmax = int(max(Ks))
    k20_idx = 0
    for idx, k in enumerate(Ks):
        if int(k) == 20:
            k20_idx = idx
            break
    all_items = np.arange(n_items)

    users = list(user_set.keys())
    if eval_users and eval_users > 0:
        users = users[:int(eval_users)]
    n_eval = len(users)

    entity_gcn_emb, user_gcn_emb = model.generate()

    extras = None
    if collect_extras:
        extras = {
            'per_user_recall20': {}, 'per_user_ndcg20': {},
            'strata_item': {}, 'strata_user': {}, 'coldstart': {},
            'topk_items': {},
        }

    it = tqdm(range(0, n_eval, batch_size), desc='eval[{}]'.format(split)) if progress \
        else range(0, n_eval, batch_size)
    for start in it:
        uids = users[start:start + batch_size]
        if len(uids) == 0:
            continue
        u_batch = torch.LongTensor(np.array(uids)).to(device)
        u_g = user_gcn_emb[u_batch]
        # score all items
        rate = np.zeros((len(uids), n_items), dtype=np.float32)
        for ib in range(0, n_items, batch_size):
            ie = min(ib + batch_size, n_items)
            i_batch = torch.LongTensor(np.arange(ib, ie)).to(device)
            i_g = entity_gcn_emb[i_batch]
            rate[:, ib:ie] = model.rating(u_g, i_g).numpy()

        for bi, u in enumerate(uids):
            u = int(u)
            pos = user_set[u]
            training_items = train_user_set[u] if u in train_user_set else []
            test_items = np.setdiff1d(all_items, np.asarray(training_items, dtype=np.int64),
                                      assume_unique=False)
            rating = rate[bi]
            sc = rating[test_items]
            if sc.shape[0] > Kmax:
                top_idx = np.argpartition(-sc, Kmax)[:Kmax]
            else:
                top_idx = np.arange(sc.shape[0])
            top_idx = top_idx[np.argsort(-sc[top_idx])]
            top_items = test_items[top_idx]
            pos_set = set(int(x) for x in pos)
            r = [1 if int(t) in pos_set else 0 for t in top_items]

            perf = get_performance(list(pos_set), r, 0.0, Ks)
            for key in result:
                result[key] += np.asarray(perf[key], dtype=np.float64) / n_eval

            if collect_extras:
                extras['per_user_recall20'][u] = float(np.asarray(perf['recall'])[k20_idx])
                extras['per_user_ndcg20'][u] = float(np.asarray(perf['ndcg'])[k20_idx])
                top20 = top_items[:Kmax]
                # item-popularity strata
                if item_bucket is not None:
                    pos_arr = np.asarray([int(x) for x in pos], dtype=np.int64)
                    for b in range(len(ITEM_POP_NAMES)):
                        p_in = pos_arr[item_bucket[pos_arr] == b]
                        if p_in.shape[0] == 0:
                            continue
                        hits = sum(1 for t in top20 if item_bucket[int(t)] == b and int(t) in pos_set)
                        rec = hits / float(p_in.shape[0])
                        extras['strata_item'].setdefault(ITEM_POP_NAMES[b], []).append(rec)
                # user-activity strata
                if user_bucket is not None and u < len(user_bucket):
                    ub = int(user_bucket[u])
                    if ub < len(USER_ACT_NAMES):
                        extras['strata_user'].setdefault(USER_ACT_NAMES[ub], []).append(
                            float(np.asarray(perf['recall'])[k20_idx]))
                # cold start
                if cold_items is not None:
                    pos_cold = [int(x) for x in pos if int(x) in cold_items]
                    if len(pos_cold) > 0:
                        s = set(pos_cold)
                        hits = sum(1 for t in top20 if int(t) in s)
                        extras['coldstart'].setdefault('recall20', []).append(hits / float(len(pos_cold)))
                # top-K lists for overlap diagnostics (ids only to limit size)
                extras['topk_items'][u] = top_items[:overlap_k].astype(np.int32)

    if collect_extras:
        # aggregate to scalars
        extras['strata_item'] = {k: float(np.mean(v)) for k, v in extras['strata_item'].items()}
        extras['strata_user'] = {k: float(np.mean(v)) for k, v in extras['strata_user'].items()}
        cs = extras['coldstart'].get('recall20', [])
        extras['coldstart'] = {'recall20': float(np.mean(cs)) if cs else None,
                               'n_users': int(len(cs))}
    return result, extras
