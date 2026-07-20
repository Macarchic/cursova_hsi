"""
Train improved_paper TC-Former (base SCMT + toggleable improved layers).

Every enhancement over the base SCMT model can be switched off individually via
the ablation flags below, so any single new layer/trick can be isolated.

Usage:
    python -m improved_paper.train --dataset IP --paper_mode
    python -m improved_paper.train --dataset PU --seeds 0 1 2 --paper_mode

Ablation examples:
    # disable one layer
    python -m improved_paper.train --dataset IP --paper_mode --no-se-block
    # legacy SCMT mixing + plain center token
    python -m improved_paper.train --dataset IP --paper_mode --mixing scmt --center plain
    # fully base SCMT (all additions off)
    python -m improved_paper.train --dataset IP --paper_mode \
        --mixing scmt --center plain --no-se-block --no-pos-encoding \
        --no-bidirectional --no-fps --no-augmentation --no-outlier-removal \
        --label-smoothing 0 --no-cosine
"""

import argparse
import json
import dataclasses
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from improved_paper.config import CONFIGS
from improved_paper.preprocessing import load_dataset, apply_pca, remove_pca_outliers
from improved_paper.dataset import get_dataloaders
from improved_paper.utils import compute_metrics, evaluate, get_run_dir, get_seed_dir, detect_accelerator
from improved_paper.model import TCFormerLit, EpochLogger


_PAPER_FILE = Path(__file__).parent / 'paper_targets.json'
with open(_PAPER_FILE) as _f:
    PAPER_TARGETS: dict = json.load(_f)


def parse_args():
    p = argparse.ArgumentParser(description='Train improved_paper TC-Former (SCMT + toggleable improved layers)')
    p.add_argument('--dataset',   required=True, choices=['IP', 'PU', 'WHHH'])
    p.add_argument('--epochs',    type=int,   default=None)
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--patience',  type=int,   default=None)
    p.add_argument('--seeds',     type=int,   nargs='+', default=[42],
                   help='One or more random seeds, e.g. --seeds 0 1 2 3 4')
    p.add_argument('--log_every', type=int,   default=1)
    p.add_argument('--results',   default='results')
    p.add_argument('--data_path', default='data')
    p.add_argument('--paper_mode', action='store_true',
                   help='No val split — test set used as eval during training')

    # ── Ablation toggles (point-wise disable of each new layer/trick) ──────────
    g = p.add_argument_group('ablation toggles')
    g.add_argument('--mixing', choices=['improved', 'scmt'], default=None,
                   help="mixing stack: 'improved' RWKV blocks (default) or 'scmt' original former")
    g.add_argument('--center', choices=['ring', 'plain'], default=None,
                   help="center aggregation: 'ring' MultiRing (default) or 'plain' SCMT center token")
    g.add_argument('--multiscale-stem', dest='multiscale_stem', action='store_true', default=None,
                   help='use the multi-scale conv stem instead of a single Conv2d')
    g.add_argument('--no-se-block',       dest='se_block',       action='store_false', default=None)
    g.add_argument('--no-pos-encoding',   dest='pos_encoding',   action='store_false', default=None)
    g.add_argument('--no-bidirectional',  dest='bidirectional',  action='store_false', default=None)
    g.add_argument('--no-fps',            dest='fps',            action='store_false', default=None)
    g.add_argument('--no-augmentation',   dest='augmentation',   action='store_false', default=None)
    g.add_argument('--no-outlier-removal', dest='outlier_removal', action='store_false', default=None)
    g.add_argument('--label-smoothing',   dest='label_smoothing', type=float, default=None)
    g.add_argument('--no-cosine',         dest='cosine',         action='store_false', default=None)
    return p.parse_args()


def apply_overrides(cfg, args):
    """Apply CLI overrides onto the dataclass config (None = leave default)."""
    if args.epochs   is not None: cfg.epochs   = args.epochs
    if args.lr       is not None: cfg.lr       = args.lr
    if args.patience is not None: cfg.patience = args.patience
    cfg.data_path = args.data_path
    if args.paper_mode:
        cfg.num_val_per_class = 0

    if args.mixing          is not None: cfg.mixing_impl           = args.mixing
    if args.center          is not None: cfg.center_attn           = args.center
    if args.multiscale_stem is not None: cfg.use_multiscale_stem   = args.multiscale_stem
    if args.se_block        is not None: cfg.use_se_block          = args.se_block
    if args.pos_encoding    is not None: cfg.use_pos_encoding      = args.pos_encoding
    if args.bidirectional   is not None: cfg.use_bidirectional_wkv = args.bidirectional
    if args.fps             is not None: cfg.use_fps               = args.fps
    if args.augmentation    is not None: cfg.use_augmentation      = args.augmentation
    if args.outlier_removal is not None: cfg.remove_pca_outliers   = args.outlier_removal
    if args.label_smoothing is not None: cfg.label_smoothing       = args.label_smoothing
    if args.cosine          is not None: cfg.use_cosine_schedule   = args.cosine
    return cfg


def train_one_seed(seed, run_dir, args, cfg, hsi_pca, labels):
    L.seed_everything(seed, workers=True)
    seed_dir = get_seed_dir(run_dir, seed)
    print(f'\n── Seed {seed} → {seed_dir} ──')

    train_loader, val_loader, test_loader = get_dataloaders(hsi_pca, labels, cfg)

    if args.paper_mode:
        val_loader = test_loader
        print('paper_mode: val split disabled — test set used as eval during training')

    lit = TCFormerLit(cfg)
    accelerator, _ = detect_accelerator()

    checkpoint_cb = ModelCheckpoint(
        dirpath=seed_dir / 'checkpoints',
        monitor='val_OA',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:03d}-{val_OA:.4f}',
        verbose=False,
    )
    callbacks = [
        checkpoint_cb,
        EarlyStopping(monitor='val_OA', mode='max', patience=cfg.patience, verbose=False),
        EpochLogger(args.log_every),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(seed_dir), name='', version=''),
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    mode_label = 'paper_mode (test as eval)' if args.paper_mode else f'patience={cfg.patience}'
    print(f'Training {args.dataset} | max {cfg.epochs} epochs | {mode_label}')
    trainer.fit(lit, train_loader, val_loader)

    best_oa = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else 0.0
    print(f'Best val_OA={best_oa:.4f}  →  {checkpoint_cb.best_model_path}')

    print('Loading best checkpoint for test evaluation...')
    ckpt = torch.load(checkpoint_cb.best_model_path, map_location='cpu', weights_only=False)
    lit.load_state_dict(ckpt['state_dict'])
    _, dev = detect_accelerator()
    lit.model.to(dev)

    preds, trues = evaluate(lit.model, test_loader)
    m = compute_metrics(preds, trues, cfg.num_classes)
    print(f'[seed={seed}] OA={m["OA"]*100:.2f}%  AA={m["AA"]*100:.2f}%  Kappa={m["Kappa"]:.4f}')

    paper = PAPER_TARGETS.get(args.dataset, {})
    p_cls = paper.get('per_class', {})

    def vs(our, paper_val):
        return f'{our}-{paper_val}'

    our_oa    = round(m['OA'] * 100, 2)
    our_aa    = round(m['AA'] * 100, 2)
    our_kappa = round(m['Kappa'], 4)

    per_class_vs = {}
    for i, acc in enumerate(m['per_class']):
        key = f'class_{i+1}'
        our_val = round(acc * 100, 1)
        pap_val = p_cls.get(key, {}).get('mean', None)
        per_class_vs[key] = vs(our_val, pap_val) if pap_val is not None else str(our_val)

    with open(seed_dir / 'test_metrics.json', 'w') as f:
        json.dump({
            'seed':      seed,
            'dataset':   args.dataset,
            'mode':      'paper_mode' if args.paper_mode else 'normal',
            'OA':        vs(our_oa,    paper.get('OA',    {}).get('mean')),
            'AA':        vs(our_aa,    paper.get('AA',    {}).get('mean')),
            'Kappa':     vs(our_kappa, paper.get('Kappa', {}).get('mean')),
            'per_class': per_class_vs,
        }, f, indent=2, ensure_ascii=False)

    return {'OA': m['OA'], 'AA': m['AA'], 'Kappa': m['Kappa']}


def aggregate(results, seeds, dataset, paper_mode):
    oas    = [r['OA']    for r in results]
    aas    = [r['AA']    for r in results]
    kappas = [r['Kappa'] for r in results]

    paper = PAPER_TARGETS.get(dataset, {})

    def vs_agg(our_mean, our_std, paper_key):
        p = paper.get(paper_key, {})
        p_str = f'{p["mean"]} ± {p["std"]}' if p else '?'
        return f'{our_mean} ± {our_std} - {p_str}'

    our_oa    = round(float(np.mean(oas)) * 100, 2)
    our_oa_s  = round(float(np.std(oas))  * 100, 2)
    our_aa    = round(float(np.mean(aas)) * 100, 2)
    our_aa_s  = round(float(np.std(aas))  * 100, 2)
    our_k     = round(float(np.mean(kappas)), 4)
    our_k_s   = round(float(np.std(kappas)),  4)

    return {
        'dataset': dataset,
        'mode':    'paper_mode' if paper_mode else 'normal',
        'seeds':   seeds,
        'OA':    vs_agg(our_oa,  our_oa_s, 'OA'),
        'AA':    vs_agg(our_aa,  our_aa_s, 'AA'),
        'Kappa': vs_agg(our_k,   our_k_s,  'Kappa'),
        'per_seed': {
            f'seed_{s}': {'OA': round(r['OA'] * 100, 2), 'AA': round(r['AA'] * 100, 2), 'Kappa': round(r['Kappa'], 4)}
            for s, r in zip(seeds, results)
        },
    }


def main():
    args = parse_args()

    cfg = CONFIGS[args.dataset]
    cfg = apply_overrides(cfg, args)

    suffix  = '_paper' if args.paper_mode else ''
    run_dir = get_run_dir(f'improved_paper{suffix}_{args.dataset}', results_root=args.results)
    print(f'\nRun directory: {run_dir}')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, ensure_ascii=False)

    print(f'Dataset: {args.dataset}  Seeds: {args.seeds}')
    print(f'Toggles: mixing={cfg.mixing_impl} center={cfg.center_attn} '
          f'multiscale={cfg.use_multiscale_stem} se={cfg.use_se_block} pos={cfg.use_pos_encoding} '
          f'biwkv={cfg.use_bidirectional_wkv} fps={cfg.use_fps} aug={cfg.use_augmentation} '
          f'outliers={cfg.remove_pca_outliers} ls={cfg.label_smoothing} cosine={cfg.use_cosine_schedule}')

    hsi, labels = load_dataset(cfg.dataset, cfg.data_path)
    hsi_pca, _  = apply_pca(hsi, cfg.pca_components)
    if cfg.remove_pca_outliers:
        labels = remove_pca_outliers(labels, hsi_pca, cfg.outlier_std)

    seed_results = []
    for seed in args.seeds:
        cfg.seed = seed
        seed_results.append(train_one_seed(seed, run_dir, args, cfg, hsi_pca, labels))

    agg = aggregate(seed_results, args.seeds, args.dataset, args.paper_mode)
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    n = len(args.seeds)
    print(f'\n=== Aggregated [{args.dataset}] — improved_paper ({n} seed(s)) ===')
    print(f'OA    : {agg["OA"]}')
    print(f'AA    : {agg["AA"]}')
    print(f'Kappa : {agg["Kappa"]}')
    print(f'\nResults saved to: {run_dir}')


if __name__ == '__main__':
    main()
