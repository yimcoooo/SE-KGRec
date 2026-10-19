"""Command-line options for SE-KGRec."""
import argparse
from pathlib import Path


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def _normalize_dataset_name(dataset_name):
    dataset_name = str(dataset_name).strip()
    if dataset_name == '':
        raise argparse.ArgumentTypeError('dataset name cannot be empty')
    normalized = Path(dataset_name).as_posix().strip('/')
    if normalized in {'', '.', '..'}:
        raise argparse.ArgumentTypeError('dataset name must point to a valid dataset directory')
    if any(part in {'', '.', '..'} for part in normalized.split('/')):
        raise argparse.ArgumentTypeError('dataset name must not contain empty path segments or parent traversal')
    return normalized


def _normalize_optional_path(path_value):
    path_value = str(path_value).strip()
    if path_value == '':
        return ''
    return Path(path_value).as_posix()


def parse_args():
    parser = argparse.ArgumentParser(description="SE-KGRec")

    # ===== dataset ===== #
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=_normalize_dataset_name,
        default="amazonbook",
        help="Dataset directory name under data_path，Any dataset that follows the expected file layout is supported (e.g. yelp_50k).",
    )
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
 
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1998, help='random seed for reproducible experiments')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--smoke_steps', type=int, default=0, help='Maximum number of training batches per epoch (0 = unlimited)')
    parser.add_argument('--skip_test', type=_str2bool, default=False, help='Skip the per-epoch test pass (for smoke runs)')
    parser.add_argument('--debug_epoch0_batches', type=int, default=0, help='Print a minimal debug summary for the first N batches of epoch 0 (0 = off)')

    parser.add_argument('--dim', type=int, default=100, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate') #0.0005-->yelp 2018  

    parser.add_argument('--do_kgr', type=_str2bool, default=True, help='Enable the KGR/KGC training and knowledge-update pipeline')
    parser.add_argument('--kgr_every', type=int, default=3, help='Run the KGR/KGC pipeline every N epochs')
    parser.add_argument('--kgr_epoch0', type=int, default=10, help='Number of KGR training epochs at epoch 0 (larger by default)')
    parser.add_argument('--kgr_epoch', type=int, default=1, help='Number of KGR training epochs at later update rounds')
    parser.add_argument('--kgr_neg_times', type=int, default=4, help='Number of negative-sampling rounds for KGR (smaller is faster)')

    parser.add_argument('--kg_candidate_generator', type=str, default='auto', help='Candidate KG generator mode: [auto, relation_aware, legacy_full_transfer]')
    parser.add_argument('--kg_completion_mode', type=str, default='npmi+fusion', help='Information sources for candidate completion. Relation-aware use is controlled separately by --use_relation_aware. Supported: npmi+fusion, npmi+cosine, npmi+semantic and their combinations; legacy spellings (npmi+x_style, npmi+x_sim, npmi+x_overall, npmi+x_style+x_sim, style, similar, overall) are also accepted.')
    parser.add_argument('--use_relation_aware', type=_str2bool, default=False, help='Enable relation-aware candidate generation. Default off, which uses the legacy full-transfer mechanism.')

    parser.add_argument('--use_x_fusion', type=_str2bool, default=True, help='Enable the fused structural/semantic matrix (alpha*struct + (1-alpha)*semantic). Default on.')
    parser.add_argument('--relation_embedding_dir', type=str, default='', help='Directory of multi-relation semantic embeddings. Leave empty for automatic discovery.')
    parser.add_argument('--relation_embedding_metadata_path', type=_normalize_optional_path, default='', help='Metadata JSON for multi-relation semantic embeddings. Leave empty unless an explicit path is needed; it is otherwise resolved per dataset in semantic mode.')

    parser.add_argument('--semantic_embed_path', type=str, default='', help='Legacy generic semantic item embeddings. Leave empty to use data/{dataset}/llm_embedding/artist_combined_embeddings.pt')
    parser.add_argument('--style_embed_path', type=str, default='', help='Style semantic embeddings. Leave empty to use data/{dataset}/llm_embedding/artist_style_embeddings.pt')
    parser.add_argument('--similar_embed_path', type=str, default='', help='Similar-item semantic embeddings. Leave empty to use data/{dataset}/llm_embedding/artist_similar_artist_embeddings.pt')
    parser.add_argument('--overall_embed_path', type=str, default='', help='Overall semantic embeddings. Leave empty to reuse relation_embeddings/item_overall_embeddings.pt or data/{dataset}/llm_embedding/artist_combined_embeddings.pt')
    parser.add_argument('--layer_availability_path', type=str, default='', help='Path to artist_layer_availability.pt. Leave empty to use data/{dataset}/llm_embedding/artist_layer_availability.pt')
    parser.add_argument('--fusion_alpha', type=float, default=0.8, help='Fusion coefficient alpha: alpha*struct + (1-alpha)*semantic')
    parser.add_argument('--style_struct_threshold', type=float, default=0.90, help='Structural similarity threshold for the style branch')
    parser.add_argument('--style_semantic_threshold', type=float, default=0.8, help='Semantic similarity threshold for the style branch')
    parser.add_argument('--similar_struct_threshold', type=float, default=0.95, help='Structural similarity threshold for the similar branch')
    parser.add_argument('--similar_semantic_threshold', type=float, default=0.8, help='Semantic similarity threshold for the similar branch')
    parser.add_argument('--overall_struct_threshold', type=float, default=0.90, help='Structural similarity threshold for the overall branch')
    parser.add_argument('--overall_semantic_threshold', type=float, default=0.7, help='Semantic similarity threshold for the overall branch')

    parser.add_argument('--style_lambda', type=float, default=0.7, help='NPMI weight in the style branch')
    parser.add_argument('--similar_lambda', type=float, default=0.8, help='NPMI weight in the similar branch')
    parser.add_argument('--style_topk', type=int, default=3, help='Maximum number of candidate neighbours per head in the style branch')
    parser.add_argument('--similar_topk', type=int, default=1, help='Maximum number of candidate neighbours per head in the similar branch')
    parser.add_argument('--relation_struct_threshold', type=float, default=0.90, help='Structural similarity threshold for the multi-relation fusion branch')
    parser.add_argument('--relation_semantic_threshold', type=float, default=0.7, help='Semantic similarity threshold for the multi-relation fusion branch')
    parser.add_argument('--relation_lambda', type=float, default=0.7, help='NPMI weight in the multi-relation relation-aware branch')
    parser.add_argument('--relation_topk', type=int, default=3, help='Maximum number of candidate neighbours per relation per head')


    parser.add_argument('--kgc_filter_topk', type=int, default=5, help='Maximum number of edges kept per head by the scorer (0 = unlimited); can be combined with --kgc_filter_topk_list')
    parser.add_argument('--kgc_filter_topk_list', type=str, default='', help='Comma-separated list of top-k values; when set, a full training run is performed for each value')
    parser.add_argument('--kgc_filter_threshold', type=float, default=0.0, help='Score threshold for edge selection (0 = select by top-k only)')
    parser.add_argument('--kgc_score_batch_size', type=int, default=131072, help='Batch size for scoring candidate edges; larger values shorten selection time on large candidate sets')

    # Main-model injection defaults to append; ablation switches below remain optional.
    parser.add_argument('--inject_selection', type=str, default='scored', choices=['scored', 'random'],
                        help='Injection selection: scored = top-Kg by confidence (default); random = uniformly random (ablation, same number of edges)')
    parser.add_argument('--disable_aux_edges', type=_str2bool, default=False,
                        help='Disable auxiliary propagation edges (_process_kg_attr). Default off, i.e. auxiliary edges are kept.')
    parser.add_argument('--inject_append', type=_str2bool, default=True,
                        help=argparse.SUPPRESS)  # the main method always uses append; kept for backward compatibility and hidden from --help
    parser.add_argument('--inject_channel3_empty', type=_str2bool, default=False,
                        help='Use an empty graph for channel 3 when nothing is injected: False = fall back to the original KG (default); True = empty-knowledge baseline')
    parser.add_argument('--ch3_tie_tail', type=_str2bool, default=False,
                        help='Reuse ch2 entity representations for the tail terms of channel 3 (fixes undertrained attribute entities); default False')
    parser.add_argument('--kg_holdout_frac', type=float, default=0.0,
                        help='Diagnostic (0 = off): randomly hold out this fraction of true KG edges from training and candidate generation, to estimate the oracle upper bound')
    parser.add_argument('--inject_oracle_holdout', type=_str2bool, default=False,
                        help='Diagnostic (off by default): inject the held-out true edges instead of the selected set (precision/recall = 100%%); requires --kg_holdout_frac > 0')
    parser.add_argument('--inject_semantic_item_edges', type=_str2bool, default=False,
                        help='Optional: inject semantic item-item edges (nearest neighbours in LLM item_overall_embeddings), i.e. information absent from both the KG and the interaction graph')
    parser.add_argument('--sem_edge_topk', type=int, default=5, help='Maximum number of semantic neighbours per item')
    parser.add_argument('--sem_edge_threshold', type=float, default=0.5, help='Cosine similarity threshold for semantic edges')
    parser.add_argument('--sem_edge_relation_id', type=int, default=1, help='Relation id used for semantic edges (must be < n_relations)')
    parser.add_argument('--sem_edge_target', type=str, default='ch2', choices=['ch2', 'ch3'],
                        help='Add semantic edges to channel 2 (original graph, default) or channel 3')
    parser.add_argument('--npmi_leak', type=_str2bool, default=False,
                        help='Leakage control (default off): True = compute NPMI on train+val+test and skip the PMI cache; '
                             'False = training interactions only (default)')

    # ===== cold_kg_only (off by default) =====
    parser.add_argument('--cold_kg_only', type=_str2bool, default=False,
                        help='Represent cold items (zero training degree) by KG propagation only: zero the ID channel, '
                             'exclude the layer-0 (self) term from entity_emb_res/n_entity_emb_res and accumulate hops 1..L, '
                             'and remove cold items from the softmax negatives. Default False = previous behaviour.')

    # ===== cold_sem_init (off by default) =====
    parser.add_argument('--cold_sem_init', type=_str2bool, default=False,
                        help='Used with --cold_kg_only: initialise the identity channel of cold items from their LLM semantic vector '
                             '(linearly projected to the channel dimension) instead of zeroing it. '
                             'Default False.')

    # ===== KGC scorer training control =====
    parser.add_argument('--kgr_optimizer_scope', type=str, default='kgc', choices=['kgc', 'all'],
                        help='Parameter scope for the KGR optimizer: kgc = confidence network only (default); all = all parameters (legacy)')
    parser.add_argument('--kgc_freeze_after_first', type=_str2bool, default=True,
                        help='Train the confidence network only in the first KGR round, then freeze it (default True; False = retrain every round)')
    parser.add_argument('--kgr_lr', type=float, default=0.001,
                        help='Learning rate of the KGR/KGC scorer (default 0.001; 1e-4 may be more stable)')
    parser.add_argument('--kgc_keep_best', type=_str2bool, default=False,
                        help='Keep the scorer snapshot with the best validation accuracy')
    parser.add_argument('--kgc_hard_neg', type=_str2bool, default=False,
                        help='Use type-constrained hard negatives for the scorer (tails/heads drawn from the real per-relation distribution, false negatives removed); default False = random easy negatives')

    # ===== validation-gated injection =====
    parser.add_argument('--val_gated_injection', type=_str2bool, default=False,
                        help='Enable validation-gated injection; default off')
    parser.add_argument('--val_gate_min_delta', type=float, default=0.0,
                        help='Minimum validation improvement required to accept an injection round')

    # ===== data split & instrumentation =====
    parser.add_argument('--use_val_split', type=_str2bool, default=True,
                        help='Use val.txt for early stopping (default True; False = legacy test-based early stopping)')
    parser.add_argument('--log_extras', type=_str2bool, default=True,
                        help='Write extra statistics (injected edges, quota, strata, cold start, per-user metrics, confidence, timing)')

    parser.add_argument('--kgr_npmi_threshold', type=float, default=0.5, help='NPMI branch threshold')
    parser.add_argument('--kgr_cos_threshold', type=float, default=0.95, help='Cosine-similarity threshold for the graph-structure item-embedding branch')
    parser.add_argument('--kgr_semantic_threshold', type=float, default=0.8, help='Cosine-similarity threshold for the semantic-embedding branch')
    parser.add_argument('--kgr_npmi_weight', type=float, default=1.0, help='NPMI branch weight')
    parser.add_argument('--kgr_cos_weight', type=float, default=1.0, help='Graph-structure cosine branch weight')
    parser.add_argument('--kgr_semantic_weight', type=float, default=1.0, help='Semantic cosine branch weight')
    parser.add_argument('--kgr_topk_neighbors', type=int, default=0, help='Maximum number of similar neighbours per item during candidate generation (0 = unlimited)')
    parser.add_argument("--inverse_r", type=_str2bool, default=False, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=_str2bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=1, help="ratio of node dropout") #yelp-->0.75 last-fm-->0.3
    parser.add_argument("--mess_dropout", type=_str2bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=_str2bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=100 , help="hidden channels for model")
    parser.add_argument("--cuda", type=_str2bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--num_workers', type=int, default=0, help='Number of PyTorch DataLoader workers (0 = single process)')
    parser.add_argument('--eval_users', type=int, default=0, help='Maximum number of users to evaluate (0 = all users; useful for quick checks on large datasets)')
    parser.add_argument('--filter_kg_head_items', type=_str2bool, default=False, help='Keep only KG edges whose head is an item/artist (for ablations and compatibility)')
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
 
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')

    parser.add_argument('--num_neg_sample', type=int, default=1, help='the number of negative sample')
    parser.add_argument('--margin', type=float, default=0.2, help='the margin of contrastive_loss')
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, contrastive_loss]")

    # ===== save model ===== #
    parser.add_argument("--save", type=_str2bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    return parser.parse_args()

