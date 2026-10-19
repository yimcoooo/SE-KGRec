# -*- coding: utf-8 -*-
"""SE-KGRec."""

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _manual_sparse_sum(src, index, dim_size):
    if src.dim() == 1:
        out = torch.zeros((dim_size,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out
    out = torch.zeros((dim_size, src.size(-1)), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items, n_entity, n_relation):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = 128

        # self.LN = nn.LayerNorm(64)
        self.activation = nn.LeakyReLU()

    def cosin_smi(self,a,b):
        if len(a.shape) != 3:
            a = a.unsqueeze(1)
            b = b.unsqueeze(1)
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-5)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-5)
        return torch.matmul(a_norm,b_norm.transpose(1,2))

    def _half_mask(self,a,b):
        ab_sig = torch.sigmoid(torch.rand(a.shape)).to(a.device)
        ab_mask = torch.bernoulli(ab_sig)
        ab_mask_rev = (ab_mask<1).float()
        mask_a = a * ab_mask
        mask_b = b * ab_mask_rev
        return mask_a, mask_b


    def forward(self, user_emb, item_emb,interact_mat):
        
        mat_row = interact_mat._indices()[0, :] 
        mat_col = interact_mat._indices()[1, :]  
        mat_val = interact_mat._values()         
 
        user_item_mat = torch.sparse_coo_tensor(torch.cat([mat_row, mat_col]).view(2, -1), mat_val,
                                                 size=[self.n_users, self.n_items])   #(114737,30040) --> (n_users,n_items)
        item_user_mat = torch.sparse_coo_tensor(torch.cat([mat_col, mat_row]).view(2, -1), mat_val,
                                                 size=[self.n_items, self.n_users]) 
        user_agg_cf = torch.sparse.mm(user_item_mat, item_emb)
        item_agg_cf = torch.sparse.mm(item_user_mat, user_emb)

        return user_agg_cf, item_agg_cf


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                  n_items, n_entities, n_relations, interact_mat,
                  device, node_dropout_rate=0.5, mess_dropout_rate=0.1,
                  inject_append: bool = True, channel3_empty: bool = False,
                  ch3_tie_tail: bool = False, cold_kg_only: bool = False,
                  cold_item_mask=None, cold_sem_init: bool = False,
                  sem_item_embeds=None):
        super(GraphConv, self).__init__()
        self.channel = channel
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.inject_append = bool(inject_append)      # concatenation is performed in main.py
        self.channel3_empty = bool(channel3_empty)
        self.ch3_tie_tail = bool(ch3_tie_tail)        # ch3 reuses ch2 entity representations for tail terms
        # cold items (zero training degree) are represented by KG propagation only
        self.cold_kg_only = bool(cold_kg_only)
        if self.cold_kg_only and cold_item_mask is not None:
            self.register_buffer('cold_item_mask',
                                 torch.as_tensor(np.asarray(cold_item_mask), dtype=torch.bool))
        else:
            self.cold_item_mask = None
        # optional: cold-item identity channel projected from LLM semantic vectors
        self.cold_sem_init = bool(cold_sem_init) and (sem_item_embeds is not None)
        if self.cold_sem_init:
            _se = torch.as_tensor(np.asarray(sem_item_embeds), dtype=torch.float32)
            self.register_buffer('sem_emb', _se)
            self.sem_proj = nn.Linear(int(_se.shape[1]), int(channel))
        else:
            self.sem_emb = None
            self.sem_proj = None
        self.n_edge_index = None
        self.n_edge_type = None
        self.act_func = nn.LeakyReLU()

        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  
        self.relation_weight = nn.Parameter(relation_weight)  
        n_relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  
        self.n_relation_weight = nn.Parameter(n_relation_weight)  
        
        self.kgc = KGC(n_items=self.n_items,num_ent=self.n_entity,num_rel=self.n_relations,dim=channel)
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, n_entity=n_entities, n_relation=n_relations).to(self.device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _update_knowledge(self,two_hpo_kg):
        self.n_edge_index = two_hpo_kg[:,[0,-1]].transpose(1,0)
        self.n_edge_type = two_hpo_kg[:,1]

    def get_injected_state(self):
        # state snapshot used for validation-gated rollback
        return (None if self.n_edge_index is None else self.n_edge_index.detach().clone(),
                None if self.n_edge_type is None else self.n_edge_type.detach().clone())

    def set_injected_state(self, state):
        idx, typ = state
        self.n_edge_index = idx
        self.n_edge_type = typ
        
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _edge_sampling_01(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        m = np.random.choice([0, 1], size=n_edges, p=[0.0, 1.0])
        return m

    def _sparse_dropout(self, x, rate=0.5):
        if rate <= 0.0:
            return x
        keep_rate = 1.0 - float(rate)
        if keep_rate <= 0.0:
            empty_idx = x._indices()[:, :0]
            empty_val = x._values()[:0]
            return torch.sparse_coo_tensor(empty_idx, empty_val, x.shape, device=x.device).coalesce()

        nnz = x._nnz()
        keep_count = int(round(nnz * keep_rate))
        keep_count = max(0, min(nnz, keep_count))
        if keep_count == nnz:
            return x.coalesce()
        if keep_count == 0:
            empty_idx = x._indices()[:, :0]
            empty_val = x._values()[:0]
            return torch.sparse_coo_tensor(empty_idx, empty_val, x.shape, device=x.device).coalesce()

        perm = torch.arange(nnz, device=x.device)
        keep_indices = perm[:keep_count]
        i = x._indices()[:, keep_indices]
        v = x._values()[keep_indices]

        out = torch.sparse_coo_tensor(i, v, x.shape, device=x.device)
        return out.coalesce()

    def KG_forward(self, entity_emb, edge_index, edge_type,
                   relation_weight, edge_weight, tail_table=None):
        
        n_entities = entity_emb.shape[0] 
        head, tail = edge_index 

        head_emb = entity_emb[head]
        # optional: tail/neighbour representations may come from the other entity table
        tail_emb = (entity_emb if tail_table is None else tail_table)[tail]
        rel_emb  = relation_weight[edge_type]
        
        KG_score = edge_weight
        
        neb_kg_emb = (tail_emb + rel_emb) * KG_score
        entity_agg = _manual_sparse_sum(src=neb_kg_emb, index=head, dim_size=n_entities)
        score_agg = _manual_sparse_sum(src=KG_score, index=head, dim_size=n_entities)
        entity_agg = entity_agg / (score_agg+1e-9)

        score_mask = (score_agg < 1).float()
        # entity_agg = entity_agg + score_mask * entity_emb
        return entity_agg, score_mask


    
    def split_kg(self,edge_index,edge_type,kg_mask_size=512):
        # topk_mask = np.zeros(edge_index.shape[0], dtype=bool)
        # topk_mask[topk_egde_id] = True
        # add another group of random mask
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=kg_mask_size, replace=False)
        random_mask = np.zeros(edge_index.shape[1], dtype=bool)
        random_mask[random_indices] = True

        mask_edge_index = edge_index[:, random_mask]
        mask_edge_type = edge_type[random_mask]
        
        retain_edge_index = edge_index[:,~random_mask]
        retain_edge_type = edge_type[~random_mask]
        
        return retain_edge_index, retain_edge_type, mask_edge_index, mask_edge_type
    
    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        return mf_loss
    
    def forward(self, all_embed, all_embed_cf, edge_index, edge_type, n_edge_index, n_edge_type, 
                interact_mat, mess_dropout=True, node_dropout=False):
        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        user_embeds = all_embed[:self.n_users]
        item_embed = all_embed[self.n_users:self.n_users + self.n_items][:, :self.channel]
        entity_emb = all_embed[self.n_users:][:, self.channel:self.channel * 2]
        n_entity_emb = all_embed[self.n_users:][:, self.channel * 2:]

        if n_edge_index is None or n_edge_type is None:
            n_edge_index = self.n_edge_index
            n_edge_type = self.n_edge_type
        if n_edge_index is None or n_edge_type is None:
            if self.channel3_empty:
                # empty-knowledge baseline: channel 3 uses an empty graph when nothing is injected
                n_edge_index = edge_index.new_empty((2, 0))
                n_edge_type = edge_type.new_empty((0,))
            else:
                n_edge_index = edge_index
                n_edge_type = edge_type
        if self.n_edge_index is None or self.n_edge_type is None:
            self.n_edge_index = n_edge_index
            self.n_edge_type = n_edge_type

        kg_weight = torch.ones(edge_type.shape[0], 1, device=item_embed.device, dtype=item_embed.dtype)
        n_kg_weight = torch.ones(n_edge_type.shape[0], 1, device=item_embed.device, dtype=item_embed.dtype)
        mmd_loss = torch.zeros((), device=item_embed.device, dtype=item_embed.dtype)

        # ===== cold-item representation: KG propagation only =====
        #   Cold items (zero training degree) have two problems: (1) their own embedding is untrained
        #   (random vector), and (2) in the CF softmax they only appear as negatives. Their
        #   representation is therefore built from KG neighbours only: (a) zero the ID channel;
        #   (b) exclude the layer-0 (self) term from ch2/ch3 and accumulate hops 1..L only.
        _cold = None
        if self.cold_kg_only and (self.cold_item_mask is not None):
            _cold = self.cold_item_mask[:self.n_items]
            # (a) ID channel: unchanged for warm items; zeroed for cold items (or LLM-projected with cold_sem_init)
            _keep = (~_cold).to(item_embed.dtype).unsqueeze(-1)
            item_embed = item_embed * _keep
            if self.cold_sem_init and (self.sem_emb is not None):
                _n = int(item_embed.shape[0])
                _sem = self.sem_proj(self.sem_emb[:_n].to(item_embed.device))
                item_embed = item_embed + _sem.to(item_embed.dtype) * (1.0 - _keep)

        entity_emb_res = entity_emb[:self.n_items]
        n_entity_emb_res = n_entity_emb[:self.n_items]
        if _cold is not None:
            # (b) cold items: residual excludes layer 0 (self); only hops 1..L are accumulated
            _keep_self = (~_cold).to(entity_emb_res.dtype).unsqueeze(-1)
            entity_emb_res = entity_emb_res * _keep_self
            n_entity_emb_res = n_entity_emb_res * _keep_self
        _ch2_tbl = entity_emb  # ch2 entity representations, reused for ch3 tail terms
        _tail_tbl = _ch2_tbl if self.ch3_tie_tail else None
        for i in range(len(self.convs)):
            entity_emb, _ = self.KG_forward(entity_emb, edge_index, edge_type, self.relation_weight, kg_weight)
            entity_emb_res = entity_emb_res + F.normalize(entity_emb[:self.n_items])
            n_entity_emb, _ = self.KG_forward(
                n_entity_emb,
                n_edge_index,
                n_edge_type,
                self.n_relation_weight,
                n_kg_weight,
                tail_table=_tail_tbl,
            )
            n_entity_emb_res = n_entity_emb_res + F.normalize(n_entity_emb[:self.n_items])

        item_embeds = torch.cat([item_embed, entity_emb_res, n_entity_emb_res], dim=-1)

        user_embed_res = user_embeds
        item_embed_res = item_embeds
        # Note (cold_kg_only): for cold items, item_embeds contains no untrained self vector
        #   (ID channel = 0 and no layer-0 term in ch2/ch3); the collaborative term item_agg_cf
        #   is identically zero for items without training interactions, so no extra handling is needed.
        for i in range(len(self.convs) - 1):
            user_embeds, item_embeds = self.convs[i](user_embeds, item_embeds, self.interact_mat)
            item_embed_res = item_embed_res + F.normalize(item_embeds)
            user_embed_res = user_embed_res + F.normalize(user_embeds)

        return user_embed_res, item_embed_res, kg_weight, mmd_loss

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ui_sp_graph,item_rel_mask, sem_item_embeds=None,
                 cold_item_mask=None):
        super(Recommender, self).__init__()
        self.args_config = args_config

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample
        
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.sem_item_embeds = sem_item_embeds

        # ===== cold-item mask =====
        self.cold_kg_only = bool(getattr(args_config, 'cold_kg_only', False))
        if self.cold_kg_only and cold_item_mask is not None:
            self.register_buffer('cold_item_mask',
                                 torch.as_tensor(np.asarray(cold_item_mask), dtype=torch.bool))
        else:
            self.cold_item_mask = None
        self.item_rel_mask = torch.FloatTensor(item_rel_mask).to(self.device)
        self.ui_sp_graph = ui_sp_graph

        self.edge_index, self.edge_type = self._get_edges(graph)
        
        self.n_edge_index = None
        self.n_edge_type = None

        self.cet_loss = nn.CrossEntropyLoss(label_smoothing=0.85)
        # self.ranking_loss = nn.MarginRankingLoss(margin=1)
        self._init_weight()
        self._init_loss_function()

        self.gcn = self._init_model()

        
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size * 3))
        self.all_embed = nn.Parameter(self.all_embed)

        # self.all_embed_cf = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
        # self.all_embed_cf = nn.Parameter(self.all_embed_cf)
        self.all_embed_cf = None
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.ui_sp_graph).to(self.device)   

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)


    def _init_model(self):
        gcn = GraphConv(channel=self.emb_size,
                          n_hops=self.context_hops,
                          n_users=self.n_users,
                          n_items=self.n_items,
                          n_entities=self.n_entities,
                          n_relations=self.n_relations,
                          interact_mat=self.interact_mat,
                          device=self.device,
                          node_dropout_rate=self.node_dropout_rate,
                          mess_dropout_rate=self.mess_dropout_rate,
                          inject_append=bool(getattr(self.args_config, 'inject_append', False)),
                          channel3_empty=bool(getattr(self.args_config, 'inject_channel3_empty', False)),
                          ch3_tie_tail=bool(getattr(self.args_config, 'ch3_tie_tail', False)),
                          cold_kg_only=bool(getattr(self.args_config, 'cold_kg_only', False)),
                          cold_item_mask=self.cold_item_mask,
                          cold_sem_init=bool(getattr(self.args_config, 'cold_sem_init', False)),
                          sem_item_embeds=self.sem_item_embeds)
        return gcn

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  
        index = graph_tensor[:, :-1] 
        type = graph_tensor[:, -1] 
        return index.t().long().to(self.device), type.long().to(self.device)

    def _append_original_knowledge(self, extra_edges):
        """Append extra edges (e.g. semantic item-item edges) to the original-graph channel (called once before training)."""
        if extra_edges is None:
            return 0
        extra_edges = np.asarray(extra_edges)
        if extra_edges.size == 0:
            return 0
        _idx = torch.as_tensor(extra_edges[:, [0, 2]].T, dtype=self.edge_index.dtype, device=self.edge_index.device)
        _typ = torch.as_tensor(extra_edges[:, 1], dtype=self.edge_type.dtype, device=self.edge_type.device)
        self.edge_index = torch.cat([self.edge_index, _idx], dim=1)
        self.edge_type = torch.cat([self.edge_type, _typ], dim=0)
        return int(extra_edges.shape[0])



    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def L2_norm(self,hidden,k=1):
        hidden_norm = torch.norm(hidden,p=2,dim=-1,keepdim=True)
        out_hidden = k * torch.div(hidden,hidden_norm+1e-8)
        return out_hidden

    def _contrastive_loss(self,user_kg,n_user_kg,pos_user_id, tau=1,small_batch = 256):
        if small_batch:
            small_id = torch.randint(low=0, high=pos_user_id.shape[0], size=(small_batch,)).to(user_kg.device)
            pos_user_id = pos_user_id[small_id]
            
        pos_user_kge = user_kg[pos_user_id]
        pos_n_user_kge = n_user_kg[pos_user_id]

        pos = torch.eye(pos_user_kge.shape[0]).to(pos_user_kge.device)
        z1_norm = torch.norm(pos_user_kge, dim=-1, keepdim=True)
        z2_norm = torch.norm(pos_n_user_kge, dim=-1, keepdim=True)
        dot_numerator = torch.mm(pos_user_kge, pos_n_user_kge.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-5) / tau)
        
        smi_sum = torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-5

        sim_matrix = sim_matrix/smi_sum
        assert sim_matrix.size(0) == sim_matrix.size(1)
        cl_loss = -torch.log(sim_matrix.mul(pos).sum(dim=-1)).mean()

        return cl_loss


    def _cf_loss(self, score, pos_item):
        """CF cross-entropy loss. With cold_kg_only, cold items are removed from the softmax negatives."""
        if self.cold_kg_only and (self.cold_item_mask is not None):
            return self._masked_cet_loss(score, pos_item)
        return self.cet_loss(score, pos_item)

    def _masked_cet_loss(self, score, pos_item):
        """Equivalent to self.cet_loss (= nn.CrossEntropyLoss(label_smoothing=0.85)), but cold items
        are removed from the softmax denominator/negatives. Label smoothing is applied only over
        the effective (non-cold) classes (C_eff = C - n_cold)."""
        _smooth_eps = 0.85
        n_cols = int(score.size(1))
        _mask = self.cold_item_mask[:n_cols].to(score.device)          # (C,) bool, True = cold item
        _valid = (~_mask)
        # cold-item logits -> -1e9: softmax probability ~ 0 (i.e. excluded from the negative pool)
        _logits = score.masked_fill(_mask.unsqueeze(0), -1e9)
        _logp = F.log_softmax(_logits, dim=-1)
        _nll = -_logp.gather(1, pos_item.view(-1, 1)).squeeze(1)
        _c_eff = _valid.to(score.dtype).sum().clamp_min(1.0)
        _neg_logp_valid = torch.where(_valid.unsqueeze(0), -_logp, torch.zeros_like(_logp))
        _smooth = _neg_logp_valid.sum(dim=1) / _c_eff
        _loss = (1.0 - _smooth_eps) * _nll + _smooth_eps * _smooth
        return _loss.mean()

    def gcn_forword(self, user, pos_item):
        user_all_emb, item_all_emb, kg_weight,mmd_loss  = self.gcn(self.all_embed,
                                                            self.all_embed_cf,
                                                            self.edge_index,
                                                            self.edge_type,
                                                            self.n_edge_index,
                                                            self.n_edge_type,
                                                            self.interact_mat,
                                                            mess_dropout=self.mess_dropout,
                                                            node_dropout=self.node_dropout)

        user_emb = user_all_emb[user]
        score = torch.matmul(user_emb, item_all_emb.transpose(1,0))
        cet_loss = self._cf_loss(score, pos_item)
        

        pos_emb = item_all_emb[pos_item]
        regularizer = (torch.norm(user_emb) ** 2
                       + torch.norm(pos_emb) ** 2)
 
        return  cet_loss, mmd_loss 

    def debug_epoch0_snapshot(self, batch=None):
        if batch is None:
            raise ValueError("batch data must not be empty")
        with torch.no_grad():
            user_all_emb, item_all_emb, kg_weight, mmd_loss = self.gcn(
                self.all_embed,
                self.all_embed_cf,
                self.edge_index,
                self.edge_type,
                self.n_edge_index,
                self.n_edge_type,
                self.interact_mat,
                mess_dropout=False,
                node_dropout=False,
            )
            users = batch['users']
            pos_items = batch['pos_items']
            user_emb = user_all_emb[users]
            pos_emb = item_all_emb[pos_items]
            score = torch.matmul(user_emb, item_all_emb.transpose(1, 0))
            cf_loss = self._cf_loss(score, pos_items)
            snapshot = {
                'cf_loss': float(cf_loss.detach().cpu().item()),
                'mmd_loss': float(mmd_loss.detach().cpu().item()) if torch.is_tensor(mmd_loss) else float(mmd_loss),
                'user_norm_mean': float(torch.norm(user_emb, dim=1).mean().detach().cpu().item()),
                'pos_item_norm_mean': float(torch.norm(pos_emb, dim=1).mean().detach().cpu().item()),
                'all_item_norm_mean': float(torch.norm(item_all_emb, dim=1).mean().detach().cpu().item()),
                'embedding_abs_mean': float(item_all_emb.abs().mean().detach().cpu().item()),
                'embedding_abs_max': float(item_all_emb.abs().max().detach().cpu().item()),
                'kg_weight_mean': float(kg_weight.mean().detach().cpu().item()) if torch.is_tensor(kg_weight) else float(kg_weight),
                'kg_weight_min': float(kg_weight.min().detach().cpu().item()) if torch.is_tensor(kg_weight) else float(kg_weight),
                'kg_weight_max': float(kg_weight.max().detach().cpu().item()) if torch.is_tensor(kg_weight) else float(kg_weight),
                'n_edge_count': int(self.gcn.n_edge_index.shape[1]) if self.gcn.n_edge_index is not None else int(self.edge_index.shape[1]),
            }
            return snapshot

    


    def forward(self,batch=None,mode="cf"):
        if batch is None:
            raise ValueError("batch data must not be empty")
        if mode == "cf":
            user = batch['users']                                                                          
            pos_item = batch['pos_items']                                                                  

            loss_network = self.gcn_forword(user, pos_item)
            return loss_network
        
        else:
            kgc_loss = self.gcn.kgc(batch)
            return kgc_loss


    def generate(self,for_kgc=False):
        # user_emb = self.all_embed[:self.n_users, :]
        # entity_emb = self.all_embed[self.n_users:, :]
        user_all_emb, item_all_emb,KG_weight,kg_loss = self.gcn( self.all_embed,
                                                            self.all_embed_cf,
                                                            self.edge_index,
                                                            self.edge_type,
                                                            self.n_edge_index,
                                                            self.n_edge_type,
                                                            self.interact_mat,
                                                            mess_dropout=False,
                                                            node_dropout=False,
                                                            )

        item_pred_emb = item_all_emb 
        user_pred_emb = user_all_emb 
        if for_kgc:
            return item_pred_emb, KG_weight
        else:
            return item_pred_emb, user_pred_emb


    def rating(self, u_g_embeddings, i_g_embeddings,type="bpr"):
        if type == "bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        else:
            return torch.cosine_similarity(u_g_embeddings[:, :self.emb_size].unsqueeze(1),
                                           i_g_embeddings[:, :self.emb_size].unsqueeze(0), dim=2).detach().cpu() + \
                   torch.cosine_similarity(u_g_embeddings[:, self.emb_size:].unsqueeze(1),
                                           i_g_embeddings[:, self.emb_size:].unsqueeze(0), dim=2).detach().cpu()


    def create_contrastive_loss(self, u_e, pos_e, neg_e,loss_weight):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)

        ui_pos_loss1 = torch.relu(1 - torch.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = (ui_pos_loss1 + ui_neg_loss1)

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss
    
    
    
    
    
class KGC(nn.Module):
    def __init__(self, n_items,num_ent, num_rel, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(KGC, self).__init__()
        self.n_items = n_items
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.linear_1 = nn.Linear(self.dim * 2,512)
        # self.linear_2 = nn.Linear(1024,1024)
        # self.linear_2 = nn.Linear(1024,512)
        self.linear_2 = nn.Linear(512,256)
        # self.ln = nn.LayerNorm(256)
        self.linear_pre = nn.Linear(256,1)
                

        self.ent_embeddings = nn.Embedding(self.num_ent,self.dim)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)

        self.loss_F = nn.MarginRankingLoss(self.margin if self.margin is not None else 1.0, reduction="mean")
        # self.bce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.bce_loss = nn.BCELoss(reduction="mean")
    def __parameter_init(self,normalize=False):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        if normalize:
            self.normalization_rel_embedding()
            self.normalization_ent_embedding()

    def normalization_ent_embedding(self):
        norm = self.ent_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embeddings.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embeddings.weight.data.copy_(torch.from_numpy(norm))
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)
    
    def _distance(self, h, t, r,neg=False):
        if neg:
            score = (h + r).unsqueeze(1) - t
            score = torch.norm(score, p=self.p_norm, dim=-1).mean(dim=1)
        else:
            score = (h + r) - t
            score = torch.norm(score, p=self.p_norm, dim=1)
        return score

    def forward(self, data, eval=False,rate=0.5,cf_train=False):

        if eval:
            if cf_train:
                batch_triple = data
            else:
                batch_triple = data["hr_pair"]
            
            batch_h = batch_triple[:,0]
            batch_r = batch_triple[:,1]
            batch_t = batch_triple[:,2]
            # batch_label = batch_triple[:,-1]

            h = self.ent_embeddings(batch_h)
            r = self.rel_embeddings(batch_r)
            t = self.ent_embeddings(batch_t)
            
            # score = torch.sigmoid(torch.matmul(h,(r * t)))
            h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
            h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
            h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
            score = torch.sigmoid(self.linear_pre(h_r_t_emb))
            # return (score>=rate).squeeze(-1).float()
            return score
        
        batch_triple = data["hr_pair"]
        batch_label = None
        if isinstance(data, dict) and ("label" in data):
            batch_label = data["label"]
        
        batch_h = batch_triple[:,0].long()
        batch_r = batch_triple[:,1].long()
        batch_t = batch_triple[:,2].long()
        if batch_label is None:
            batch_label = batch_triple[:,-1]

        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
        h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
        h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
        score = torch.sigmoid(self.linear_pre(h_r_t_emb))

        loss = self.bce_loss(score.squeeze(-1), batch_label.float())

        return loss
