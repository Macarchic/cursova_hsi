"""
CLI entry point for training HSI models.

Usage:
    python -m src.train --model baseline --dataset IP
    python -m src.train --model baseline --dataset IP --paper_mode
    python -m src.train --model baseline --dataset IP --seeds 0 1 2 3 4 --paper_mode
    python -m src.train --model improved --dataset IP --seeds 0 1 2 3 4 --paper_mode
"""

import argparse
import json
import dataclasses
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from src.preprocessing import load_dataset, apply_pca
from src.dataset import get_dataloaders
from src.utils import compute_metrics, evaluate, get_run_dir, get_seed_dir, detect_accelerator
from src.models import get_model, get_config, MODEL_REGISTRY
from src.models.base_lit import EpochLogger


PAPER_TARGETS = {
    'IP':   dict(OA=90.01, AA=94.36, Kappa=0.887),
    'PU':   dict(OA=94.60, AA=95.20, Kappa=0.930),
    'WHHH': dict(OA=91.56, AA=90.54, Kappa=0.894),
}


def parse_args():
    p = argparse.ArgumentParser(description='Train an HSI model')
    p.add_argument('--model',     required=True, help='Model name (e.g. baseline, improved)')
    p.add_argument('--dataset',   required=True, help='Dataset name: IP | PU | WHHH')
    p.add_argument('--epochs',    type=int,   default=None)
    p.add_argument('--lr',        type=float, default=None)
    p.add_argument('--patience',  type=int,   default=None)
    p.add_argument('--seeds',     type=int,   nargs='+', default=[42],
                   help='One or more random seeds, e.g. --seeds 0 1 2 3 4')
    p.add_argument('--log_every', type=int,   default=1,  help='Print interval (epochs)')
    p.add_argument('--results',   default='results', help='Root folder for run outputs')
    p.add_argument('--data_path', default='data',    help='Root folder for .mat data files')
    p.add_argument('--paper_mode', action='store_true',
                   help='Replicate paper setup: no val split, test set used as eval during training')
    return p.parse_args()


def aggregate_metrics(results, seeds, model, dataset, paper_mode):
    oas    = [r['OA']    for r in results]
    aas    = [r['AA']    for r in results]
    kappas = [r['Kappa'] for r in results]
    paper  = PAPER_TARGETS.get(dataset, {})
    return {
        'model':   model,
        'dataset': dataset,
        'mode':    'paper_mode' if paper_mode else 'normal',
        'seeds':   seeds,
        'OA':    {'mean': round(float(np.mean(oas)) * 100, 2), 'std': round(float(np.std(oas)) * 100, 2)},
        'AA':    {'mean': round(float(np.mean(aas)) * 100, 2), 'std': round(float(np.std(aas)) * 100, 2)},
        'Kappa': {'mean': round(float(np.mean(kappas)), 4),    'std': round(float(np.std(kappas)), 4)},
        'paper': paper,
        'per_seed': {
            f'seed_{s}': {
                'OA':    round(r['OA'] * 100, 2),
                'AA':    round(r['AA'] * 100, 2),
                'Kappa': round(r['Kappa'], 4),
            }
            for s, r in zip(seeds, results)
        },
    }


def train_one_seed(seed, run_dir, args, cfg, hsi_pca, labels):
    seed_dir = get_seed_dir(run_dir, seed)
    print(f'\n── Seed {seed} → {seed_dir} ──')

    registry_entry = MODEL_REGISTRY[args.model]
    if 'get_dataloaders' in registry_entry:
        train_loader, val_loader, test_loader = registry_entry['get_dataloaders'](hsi_pca, labels, cfg)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(hsi_pca, labels, cfg)

    if args.paper_mode:
        val_loader = test_loader
        print('paper_mode: val split disabled — test set used as eval during training')

    lit_model = get_model(args.model, cfg)

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

    csv_logger = CSVLogger(save_dir=str(seed_dir), name='', version='')

    mode_label = 'paper_mode (test as eval)' if args.paper_mode else f'patience={cfg.patience}'
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=csv_logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    print(f'Training {args.dataset} | max {cfg.epochs} epochs | {mode_label}')
    trainer.fit(lit_model, train_loader, val_loader)

    best_oa = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else 0.0
    print(f'Best val_OA={best_oa:.4f}  →  {checkpoint_cb.best_model_path}')

    print('Loading best checkpoint for test evaluation...')
    ckpt = torch.load(checkpoint_cb.best_model_path, map_location='cpu', weights_only=False)
    lit_model.load_state_dict(ckpt['state_dict'])
    _, dev = detect_accelerator()
    lit_model.model.to(dev)

    preds, trues = evaluate(lit_model.model, test_loader)
    m = compute_metrics(preds, trues, cfg.num_classes)

    mode_tag = 'paper_mode' if args.paper_mode else 'normal'
    print(f'[seed={seed}] OA={m["OA"]*100:.2f}%  AA={m["AA"]*100:.2f}%  Kappa={m["Kappa"]:.4f}')

    seed_out = {
        'seed': seed,
        'mode': mode_tag,
        'OA':    round(m['OA'] * 100, 2),
        'AA':    round(m['AA'] * 100, 2),
        'Kappa': round(m['Kappa'], 4),
        'per_class': {f'class_{i+1}': round(a * 100, 1) for i, a in enumerate(m['per_class'])},
    }
    with open(seed_dir / 'test_metrics.json', 'w') as f:
        json.dump(seed_out, f, indent=2)

    return {'OA': m['OA'], 'AA': m['AA'], 'Kappa': m['Kappa']}


def main():
    args = parse_args()

    cfg = get_config(args.model, args.dataset)
    if args.epochs   is not None: cfg.epochs   = args.epochs
    if args.lr       is not None: cfg.lr       = args.lr
    if args.patience is not None: cfg.patience = args.patience
    if args.data_path:            cfg.data_path = args.data_path
    if args.paper_mode:           cfg.num_val_per_class = 0

    suffix = '_paper' if args.paper_mode else ''
    run_dir = get_run_dir(f'{args.model}{suffix}', args.dataset, results_root=args.results)
    print(f'\nRun directory: {run_dir}')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    n_seeds = len(args.seeds)
    print(f'Model: {args.model}  Dataset: {args.dataset}  Seeds: {args.seeds}')

    hsi, labels = load_dataset(cfg.dataset, cfg.data_path)
    hsi_pca, _  = apply_pca(hsi, cfg.pca_components)

    seed_results = []
    for seed in args.seeds:
        cfg.seed = seed
        torch.manual_seed(seed)
        result = train_one_seed(seed, run_dir, args, cfg, hsi_pca, labels)
        seed_results.append(result)

    agg = aggregate_metrics(seed_results, args.seeds, args.model, args.dataset, args.paper_mode)
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(agg, f, indent=2)

    paper = PAPER_TARGETS.get(args.dataset)
    print(f'\n=== Aggregated Results [{args.dataset}] — {args.model} ({n_seeds} seed(s)) ===')
    print(f'OA    : {agg["OA"]["mean"]:.2f}% ± {agg["OA"]["std"]:.2f}%')
    print(f'AA    : {agg["AA"]["mean"]:.2f}% ± {agg["AA"]["std"]:.2f}%')
    print(f'Kappa : {agg["Kappa"]["mean"]:.4f} ± {agg["Kappa"]["std"]:.4f}')
    if paper:
        print(f'\nPaper target: OA {paper["OA"]:.2f}%  AA {paper["AA"]:.2f}%  Kappa {paper["Kappa"]:.3f}')
        print(f'Gap vs paper: OA {agg["OA"]["mean"] - paper["OA"]:+.2f}%')
    print(f'\nResults saved to: {run_dir}')


if __name__ == '__main__':
    main()
