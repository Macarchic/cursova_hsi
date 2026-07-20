# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`SCMT/` is a standalone PyTorch implementation of the **SCMT (Spectral Channel Mixing Transformer / TC-Former)** model for hyperspectral image (HSI) classification, described in the article this coursework is based on (see the parent `../CLAUDE.md`). It is a self-contained training pipeline: config JSON → data loading + PCA + patch extraction → model → train → evaluate → JSON results. Source comments are largely in Chinese.

## How to run

The entry point is `src/workflow.py`, run **from inside `src/`**:

```bash
cd src
python workflow.py
```

`workflow.py` iterates over `include_path` and trains a model per config. As checked in, `include_path` is `'scmtformer.json'` repeated 6×, so a single run trains the Indian Pines config six times.

Dependencies: `torch`, `einops`, `numpy`, `scipy` (`scipy.io.loadmat`), `scikit-learn`.

## Portability gotchas (READ FIRST — the code will not run as-is on this machine)

`src/utils.py` contains **hardcoded Windows paths** and a non-default CUDA device that must be fixed for this macOS checkout:

- `config_path_prefix = "D:\python\SCMT\src\params_use"` — where `workflow.py` loads config JSONs from. Point it at the local `params_use/` dir.
- `HSIRecoder.to_file(...)` **ignores its `path` argument** and writes results to a hardcoded `D:/python/SCMT/src/res/{data_file}_test_{timestamp}.json`. Fix this to write into the local `res/`.
- `device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')` — note **`cuda:1`**, not `cuda:0`. Falls back to CPU when CUDA is unavailable (fine on macOS).
- Only `data/Indian/Indian_10_1_split.mat` is present. PaviaU (and the other datasets referenced by configs and `evaluation.py` target-name lists) are **absent** — running `scmtformer1.json` will fail on missing data.

## Architecture & data flow

```
workflow.py  →  load config JSON  →  HSIDataLoader (PCA.py)  →  SCMTTrainer (trainer.py)
                                                                     ↓
                                     recorder (utils.py)  ←  HSIEvaluation (evaluation.py)
```

Pipeline steps (in `PCA.py` `HSIDataLoader`):
`.mat` file (keys `input`, `TR`, `TE`; `labels = TR + TE`) → per-band normalization (`max_min` default) → PCA with whitening (`pca` components) → truncate to `spectral_size` bands → zero-pad spatial dims → per-pixel patch extraction split by the **predefined TR/TE masks** (not a random split) → random flip/transpose augmentation → four `DataLoader`s: `train`, `unlabel`, `test`, `all`.

Model input shape is `(batch, spectral, w, h)`. `SCMT.encoder_block` runs a Conv2d spectral embedder → rearrange to token sequence → `former` (stacked time-mix / tiny-attn / channel-mix blocks) → takes the **center-token** feature → `classifier_mlp`.

## Key files

- `src/workflow.py` — entry point; `run_all()` loops configs, `train_by_param()` runs one config end-to-end.
- `src/PCA.py` — **the active data loader** (`HSIDataLoader`, `DataSetIter`, `applyPCA`). Imported by `workflow.py`.
- `src/models/SCMT.py` — model: `TimeMixFormer` + `ChannelMixFormer` + `TinyAttn` (RWKV-style), stacked in `former`, wrapped by `SCMT`. Also `Init` / `_weights_init`.
- `src/trainer.py` — `SCMTTrainer` (Adam, CrossEntropy, grad clip 15) and `get_trainer1(params)` dispatcher.
- `src/evaluation.py` — `HSIEvaluation`: OA / AA / Kappa / per-class accuracy (all ×100) via sklearn; hardcoded per-dataset class-name lists.
- `src/utils.py` — `device`, `config_path_prefix`, `AvgrageMeter`, `HSIRecoder`, and the global `recorder` singleton. **Holds the hardcoded paths above.**
- `src/params_use/scmtformer.json` — Indian Pines config. `scmtformer1.json` — PaviaU config (defined but not referenced by `include_path`).
- `src/res/*.json` — result outputs.

## Two data loaders — use the right one

There are two near-identical loaders. **`src/PCA.py` is authoritative** (predefined TR/TE split, tensors moved to `device` inside the dataset) and is what `workflow.py` imports. `src/data_provider/data_provider.py` is a **legacy/unused variant** (random `train_test_split`, hardcoded `../data/Indian/Honghu.mat`, debug prints, CPU tensors) — do not wire it into the pipeline.

## Config & output format

Each config JSON has three sections plus `uniq_name`:
- `data.*` — consumed by `HSIDataLoader`, `SCMT`, `HSIEvaluation` (e.g. `data_sign`, `data_file`, `patch_size`, `pca`, `spectral_size`, `num_classes`, `batch_size`).
- `net.*` — consumed by `SCMT`/`former`/mixers/`TinyAttn` and `get_trainer1` (`trainer` must be `"scmtformer"`; `depth`, `dim`, `heads`, `kernal`, `n_embd`, `ctx_len`, `tiny_attn`, …).
- `train.*` — consumed by `SCMTTrainer` (`epochs`, `lr`, `weight_decay`).

Result JSON keys: `epoch_loss` (`{type, index, value}` per-epoch loss series), `param` (verbatim copy of the config), `eval` (metrics dict: `classification` report string, `oa`, `aa`, `kappa`, `each_acc`, `confusion` — accuracies as percentages). Filenames: `{data_file}_test_{YYYY-MM-DD_HH-MM-SS}.json`.

## Gotchas

- **In-training test never fires**: it is gated on `(epoch+1) % 201 == 0`, but configs use `epochs=100`, so `train_oa/aa/kappa` are never recorded — only `final_eval` produces metrics.
- **Hardcoded model dims** (128 / 64 / 256) inside `SCMT.py`; some `nn.Linear` layers are created at runtime inside `forward` to absorb shape mismatches, so `pca`/`spectral_size`/`patch_size` (→ `ctx_len ≈ patch_size²`) must line up with these. Config `ctx_len` values do not exactly match `patch_size²`.
- **Global mutable `recorder` singleton** in `utils.py` is shared across modules and reset per config by `workflow.train_by_param`.
- **Labels are 1-indexed in the `.mat` files**; `DataSetIter` subtracts 1 (`label - 1`) to make them 0-indexed for training.
