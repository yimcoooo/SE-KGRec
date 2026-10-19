# SE-KGRec

Research implementation of semantic-enhanced knowledge-graph recommendation. The main model generates relation-aware candidate triples, selects them with a KG confidence scorer, and propagates over both the original KG and the updated KG. The updated graph uses original triples union selected triples (`--inject_append true`, the default). KG propagation uses unit edge weights; this implementation does not train an edge-deletion network.

The project began from the EditKG research codebase. EditKG remains a separately evaluated baseline and is not implemented by the SE-KGRec training entry point. Please cite the original EditKG paper when discussing that baseline.

## Data

No real dataset or embedding tensors are included in this release. See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the expected local file layout and processing methods. The reported results require the prepared Amazonbook, YELP 20-core, and Goodreads inputs plus their relation-level semantic embeddings.

## Main-model mode

The command-line parser defaults to `--inject_append true`. To reproduce the previous replacement-graph ablation only, pass `--inject_append false` explicitly. The main-model runner also specifies append explicitly. Learned KG edge deletion and its command-line options have been removed from this runtime.

Third-party baseline implementations are not included; they are evaluated as external baselines in the paper.

## Usage

```bash
pip install -r requirements.txt
bash run_amazonbook.sh    # or run_yelp.sh / run_goodreads.sh
```

Each runner expects `data/<dataset>/` with `train.txt`, `val.txt`, `test.txt`, `kg_final.txt`, `relation_list.txt`, and the relation-level embeddings under `llm_embedding/relation_embeddings/`. Results are written to `result/` and checkpoints to `model_para/`.
