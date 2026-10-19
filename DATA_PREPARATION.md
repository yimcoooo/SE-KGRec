# Data preparation for SE-KGRec

This code release does not include real datasets, raw reviews, processed interactions, knowledge-graph triples, semantic JSONL records, embedding tensors, caches, or experiment outputs. Obtain source data from the original providers under their terms and run the preparation code locally.

## Datasets used in the paper

- Amazonbook: `amazonbook_200k`
- YELP: `yelp_20-core`
- Goodreads: `goodreads`

The model reads `data/<dataset>/` (pass `--data_path data/`). Each dataset directory must contain `train.txt`, `val.txt`, `test.txt`, `kg_final.txt`, and `relation_list.txt`. Relation-aware semantic completion also requires `llm_embedding/relation_embeddings/embedding_metadata.json` and the tensor files referenced by that metadata. The optional mapping files (`item_list.txt`, `entity_list.txt`, `user_list.txt`) are used by preparation and interpretation scripts.

## Existing processing methods

1. Build an aligned dataset with the dataset-specific processing scripts: `reduce_kgat_dataset.py` (Amazonbook), `build_yelp_aligned_dataset.py` (YELP), or `build_goodreads_aligned_dataset.py` (Goodreads). These scripts require their respective original source files; their source files are not distributed here.
2. Where an aligned build directory is produced, convert it to the model input format with `data_processing/make_pipeline_dataset.py --build-dir <aligned-build> --name <dataset>`.
3. Generate the fixed user-wise train/validation/test partition with `data_processing/make_split.py --datasets amazonbook_200k yelp_20-core goodreads`. This script reads the aligned files in `data/<dataset>/` and writes `train.txt`, `val.txt`, and `test.txt` into the same dataset directory.
4. To use the relation-aware semantic method, generate relation-level semantic descriptions and convert them with the dataset-specific `preprocess_*_relation_embeddings.py` scripts. The text generation step may require a separately configured LLM service and an encoder. Keep credentials in environment variables, never in the repository.

## Notes on the preparation scripts

The scripts in `data_processing/` were used to prepare the inputs for the reported experiments. Every input and output location is exposed as a command-line argument and defaults to a relative path under `data/`. Dataset-specific steps still depend on the original files distributed by the dataset providers, and the semantic-extraction step depends on a separately configured LLM service and text encoder. Because the raw data and the semantic outputs are not redistributed, the repository alone cannot reproduce the reported scores.

## Repository contents

This repository contains the model code, the processing and evaluation utilities, and the three dataset runners. Processed dataset directories under `data/`, raw JSON and review text, semantic JSONL outputs, `.pt` tensors, `.pkl` caches, logs, and result directories are not included. Model checkpoints are written to `model_para/` and experiment outputs to `result/`; both are excluded from version control.
