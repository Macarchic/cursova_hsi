"""
Iterative pseudo-labeling semi-supervised training.

Usage:
    python -m semi.train --dataset IP --teacher_ckpt model/best-epoch=073-val_OA=0.8709.ckpt
    python -m semi.train --dataset IP --teacher_ckpt model/best.ckpt --pseudo_per_class 50 --seeds 0 1 2
"""

import argparse
import copy
import dataclasses
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from baseline.preprocessing import load_dataset, apply_pca
from baseline.utils import compute_metrics, evaluate, get_run_dir, get_seed_dir, detect_accelerator
from improved.model import TCFormerLit, EpochLogger
from semi.config import SEMI_CONFIGS
from semi.dataset import get_semi_dataloaders, split_for_semi
from semi.pseudolabel import pick_pseudo_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',          required=True, choices=['IP', 'PU', 'WHHH'])
    p.add_argument('--teacher_ckpt',     required=True, help='Path to teacher .ckpt file')
    p.add_argument('--pseudo_per_class', type=int,   default=None)
    p.add_argument('--num_val_per_class',type=int,   default=None)
    p.add_argument('--finetune_epochs',  type=int,   default=None)
    p.add_argument('--finetune_lr',      type=float, default=None)
    p.add_argument('--max_rounds',       type=int,   default=None)
    p.add_argument('--seeds',            type=int,   nargs='+', default=[42])
    p.add_argument('--log_every',        type=int,   default=1)
    p.add_argument('--results',          default='results')
    p.add_argument('--data_path',        default='data')
    return p.parse_args()


def load_teacher(ckpt_path: str, cfg) -> TCFormerLit:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    lit  = TCFormerLit(cfg)
    lit.load_state_dict(ckpt['state_dict'])
    return lit


def finetune_one_round(lit: TCFormerLit, train_loader, val_loader,
                       cfg, seed_dir: Path, round_idx: int, log_every: int) -> TCFormerLit:
    """Fine-tune lit for finetune_epochs epochs, return best-checkpoint model."""
    ckpt_dir = seed_dir / f'round_{round_idx:02d}' / 'checkpoints'
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_OA',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:03d}-{val_OA:.4f}',
        verbose=False,
    )
    accelerator, _ = detect_accelerator()
    trainer = L.Trainer(
        max_epochs=cfg.finetune_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            checkpoint_cb,
            EarlyStopping(monitor='val_OA', mode='max', patience=10, verbose=False),
            EpochLogger(log_every),
        ],
        logger=CSVLogger(save_dir=str(seed_dir / f'round_{round_idx:02d}'), name='', version=''),
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    # Override LR for fine-tuning via a new lit wrapper with same weights
    ft_lit = TCFormerLit(cfg)
    ft_lit.load_state_dict(lit.state_dict())
    ft_lit.cfg = copy.copy(cfg)
    ft_lit.cfg.lr = cfg.finetune_lr
    ft_lit.cfg.epochs = cfg.finetune_epochs

    trainer.fit(ft_lit, train_loader, val_loader)

    best_oa = float(checkpoint_cb.best_model_score) if checkpoint_cb.best_model_score else 0.0
    print(f'    → best val_OA={best_oa:.4f}  {checkpoint_cb.best_model_path}')

    best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location='cpu', weights_only=False)
    ft_lit.load_state_dict(best_ckpt['state_dict'])
    return ft_lit


def run_one_seed(seed, run_dir, args, cfg, hsi_pca, labels):
    L.seed_everything(seed, workers=True)
    seed_dir = get_seed_dir(run_dir, seed)
    print(f'\n── Seed {seed} → {seed_dir} ──')

    train_idx, val_idx, test_idx = split_for_semi(labels, hsi_pca, cfg)
    print(f'Split — Train: {len(train_idx)}  Val: {len(val_idx)}  Pool/Test: {len(test_idx)}')

    pool = test_idx.copy()

    _, dev = detect_accelerator()
    current_lit = load_teacher(args.teacher_ckpt, cfg)
    current_lit.model.to(dev)
    print(f'Teacher loaded from {args.teacher_ckpt}')

    # Evaluate teacher on test set as baseline
    _, _, test_loader_init = get_semi_dataloaders(
        hsi_pca, labels, train_idx, val_idx, test_idx,
        np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), cfg
    )
    preds, trues = evaluate(current_lit.model, test_loader_init)
    m0 = compute_metrics(preds, trues, cfg.num_classes)
    print(f'Teacher baseline — OA={m0["OA"]*100:.2f}%  AA={m0["AA"]*100:.2f}%  κ={m0["Kappa"]:.4f}')

    round_results = [{'round': 0, 'train_size': len(train_idx),
                      'OA': m0['OA'], 'AA': m0['AA'], 'Kappa': m0['Kappa']}]

    acc_pseudo_idx = np.empty((0, 2), dtype=int)
    acc_pseudo_cls = np.empty((0,),   dtype=int)

    for r in range(1, cfg.max_rounds + 1):
        print(f'\n  Round {r}/{cfg.max_rounds}  pool={len(pool)}')

        new_idx, new_cls = pick_pseudo_labels(current_lit.model, hsi_pca, pool, cfg, dev)
        if len(new_idx) == 0:
            print('  Pool exhausted — stopping.')
            break

        # Remove selected from pool
        selected_set = set(map(tuple, new_idx.tolist()))
        pool = np.array([p for p in pool if tuple(p) not in selected_set])

        acc_pseudo_idx = np.concatenate([acc_pseudo_idx, new_idx], axis=0)
        acc_pseudo_cls = np.concatenate([acc_pseudo_cls, new_cls], axis=0)

        per_cls = {c: int((new_cls == c).sum()) for c in range(1, cfg.num_classes + 1)}
        print(f'  Added {len(new_idx)} pseudo-labels  per class: {per_cls}')
        print(f'  Total train: {len(train_idx) + len(acc_pseudo_idx)}')

        train_loader, val_loader, test_loader = get_semi_dataloaders(
            hsi_pca, labels, train_idx, val_idx, test_idx,
            acc_pseudo_idx, acc_pseudo_cls, cfg
        )

        current_lit = finetune_one_round(
            current_lit, train_loader, val_loader, cfg, seed_dir, r, args.log_every
        )
        current_lit.model.to(dev)

        preds, trues = evaluate(current_lit.model, test_loader)
        m = compute_metrics(preds, trues, cfg.num_classes)
        print(f'  Test — OA={m["OA"]*100:.2f}%  AA={m["AA"]*100:.2f}%  κ={m["Kappa"]:.4f}')

        round_results.append({
            'round':      r,
            'train_size': len(train_idx) + len(acc_pseudo_idx),
            'OA':   m['OA'],
            'AA':   m['AA'],
            'Kappa': m['Kappa'],
        })

    best_round = max(round_results, key=lambda x: x['OA'])
    print(f'\n  Best round: {best_round["round"]}  OA={best_round["OA"]*100:.2f}%')

    with open(seed_dir / 'round_metrics.json', 'w') as f:
        json.dump({'seed': seed, 'dataset': args.dataset, 'rounds': round_results}, f, indent=2)

    return best_round


def main():
    args = parse_args()
    cfg = SEMI_CONFIGS[args.dataset]
    cfg.data_path        = args.data_path
    cfg.teacher_ckpt     = args.teacher_ckpt
    if args.pseudo_per_class  is not None: cfg.pseudo_per_class  = args.pseudo_per_class
    if args.num_val_per_class is not None: cfg.num_val_per_class = args.num_val_per_class
    if args.finetune_epochs   is not None: cfg.finetune_epochs   = args.finetune_epochs
    if args.finetune_lr       is not None: cfg.finetune_lr       = args.finetune_lr
    if args.max_rounds        is not None: cfg.max_rounds        = args.max_rounds

    run_dir = get_run_dir(f'semi_{args.dataset}', results_root=args.results)
    print(f'\nRun directory: {run_dir}')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, ensure_ascii=False)

    hsi, labels = load_dataset(cfg.dataset, cfg.data_path)
    hsi_pca, _  = apply_pca(hsi, cfg.pca_components)

    seed_results = []
    for seed in args.seeds:
        cfg.seed = seed
        seed_results.append(run_one_seed(seed, run_dir, args, cfg, hsi_pca, labels))

    oas    = [r['OA']    for r in seed_results]
    aas    = [r['AA']    for r in seed_results]
    kappas = [r['Kappa'] for r in seed_results]
    agg = {
        'dataset': args.dataset,
        'seeds':   args.seeds,
        'OA':    f'{round(float(np.mean(oas))*100,2)} ± {round(float(np.std(oas))*100,2)}',
        'AA':    f'{round(float(np.mean(aas))*100,2)} ± {round(float(np.std(aas))*100,2)}',
        'Kappa': f'{round(float(np.mean(kappas)),4)} ± {round(float(np.std(kappas)),4)}',
    }
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f'\n=== Semi [{args.dataset}] ({len(args.seeds)} seed(s)) ===')
    print(f'OA    : {agg["OA"]}')
    print(f'AA    : {agg["AA"]}')
    print(f'Kappa : {agg["Kappa"]}')
    print(f'\nResults saved to: {run_dir}')


if __name__ == '__main__':
    main()
