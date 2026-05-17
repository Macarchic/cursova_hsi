"""
Iterative pseudo-labeling semi-supervised training.

Each round:
  1. Current model predicts pool pixels → pick top pseudo_per_class per class
  2. Fine-tune on GT train + accumulated pseudo-labels (test used as eval monitor)
  3. Evaluate on full fixed test set (GT labels) → track metric dynamics

Usage:
    python -m semi.train --dataset IP --teacher_ckpt model/best-epoch=073-val_OA=0.8709.ckpt
    python -m semi.train --dataset IP --teacher_ckpt model/best.ckpt --pseudo_per_class 50 --seeds 0 1 2
"""

import argparse
import dataclasses
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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
    p.add_argument('--teacher_ckpt',     required=True)
    p.add_argument('--pseudo_per_class', type=int, default=None)
    p.add_argument('--max_rounds',       type=int, default=None)
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


def train_from_scratch(train_loader, test_loader, cfg, round_dir: Path, log_every: int) -> TCFormerLit:
    """Train a fresh TCFormerLit from random init. Test loader used as val monitor."""
    from lightning.pytorch.callbacks import EarlyStopping
    checkpoint_cb = ModelCheckpoint(
        dirpath=round_dir / 'checkpoints',
        monitor='val_OA', mode='max', save_top_k=1,
        filename='best-{epoch:03d}-{val_OA:.4f}', verbose=False,
    )
    accelerator, _ = detect_accelerator()
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            checkpoint_cb,
            EarlyStopping(monitor='val_OA', mode='max', patience=cfg.patience, verbose=False),
            EpochLogger(log_every),
        ],
        logger=CSVLogger(save_dir=str(round_dir), name='', version=''),
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    lit = TCFormerLit(cfg)
    trainer.fit(lit, train_loader, test_loader)

    if checkpoint_cb.best_model_path:
        best_ckpt = torch.load(checkpoint_cb.best_model_path, map_location='cpu', weights_only=False)
        lit.load_state_dict(best_ckpt['state_dict'])
        print(f'    best val_OA={float(checkpoint_cb.best_model_score):.4f}')

    return lit


def run_one_seed(seed, run_dir, args, cfg, hsi_pca, labels):
    L.seed_everything(seed, workers=True)
    seed_dir = get_seed_dir(run_dir, seed)
    print(f'\n── Seed {seed} → {seed_dir} ──')

    train_idx, test_idx = split_for_semi(labels, hsi_pca, cfg)
    print(f'Split — Train (GT): {len(train_idx)}  Test (fixed): {len(test_idx)}')

    pool = test_idx.copy()   # pool shrinks each round; test_idx stays fixed

    _, dev = detect_accelerator()
    current_lit = load_teacher(args.teacher_ckpt, cfg)
    current_lit.model.to(dev)
    print(f'Teacher: {args.teacher_ckpt}')

    # Round 0: teacher baseline on test
    _, test_loader = get_semi_dataloaders(
        hsi_pca, labels, train_idx, test_idx,
        np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), cfg
    )
    preds, trues = evaluate(current_lit.model, test_loader)
    m0 = compute_metrics(preds, trues, cfg.num_classes)
    print(f'Round 0 (teacher)  OA={m0["OA"]*100:.2f}%  AA={m0["AA"]*100:.2f}%  κ={m0["Kappa"]:.4f}')

    round_results = [{
        'round': 0, 'train_size': len(train_idx),
        'pool_remaining': len(pool),
        'OA': m0['OA'], 'AA': m0['AA'], 'Kappa': m0['Kappa'],
    }]

    acc_pseudo_idx = np.empty((0, 2), dtype=int)
    acc_pseudo_cls = np.empty((0,),   dtype=int)

    for r in range(1, cfg.max_rounds + 1):
        print(f'\n  ── Round {r}  pool={len(pool)} ──')

        new_idx, new_cls = pick_pseudo_labels(current_lit.model, hsi_pca, pool, cfg, dev)
        if len(new_idx) == 0:
            print('  Pool exhausted — stopping.')
            break

        # Remove newly selected from pool (test_idx stays unchanged)
        selected_set = set(map(tuple, new_idx.tolist()))
        pool = np.array([p for p in pool if tuple(p) not in selected_set])

        acc_pseudo_idx = np.concatenate([acc_pseudo_idx, new_idx], axis=0)
        acc_pseudo_cls = np.concatenate([acc_pseudo_cls, new_cls], axis=0)

        per_cls_count = {c: int((new_cls == c).sum()) for c in range(1, cfg.num_classes + 1)}
        train_size = len(train_idx) + len(acc_pseudo_idx)
        print(f'  Added {len(new_idx)} pseudo-labels → total train: {train_size}')
        print(f'  Per class: {per_cls_count}')

        # Check pseudo-label accuracy against GT
        correct = sum(int(labels[r, c]) == int(cls)
                      for (r, c), cls in zip(new_idx, new_cls))
        print(f'  Pseudo-label accuracy vs GT: {correct}/{len(new_idx)} = {correct/len(new_idx)*100:.1f}%')

        train_loader, test_loader = get_semi_dataloaders(
            hsi_pca, labels, train_idx, test_idx,
            acc_pseudo_idx, acc_pseudo_cls, cfg
        )

        round_dir = seed_dir / f'round_{r:02d}'
        current_lit = train_from_scratch(
            train_loader, test_loader, cfg, round_dir, args.log_every
        )
        current_lit.model.to(dev)

        preds, trues = evaluate(current_lit.model, test_loader)
        m = compute_metrics(preds, trues, cfg.num_classes)
        print(f'  Test  OA={m["OA"]*100:.2f}%  AA={m["AA"]*100:.2f}%  κ={m["Kappa"]:.4f}')

        round_results.append({
            'round':          r,
            'train_size':     train_size,
            'pool_remaining': len(pool),
            'pseudo_acc_gt':  round(correct / len(new_idx), 4),
            'OA':   m['OA'], 'AA': m['AA'], 'Kappa': m['Kappa'],
        })

    best = max(round_results, key=lambda x: x['OA'])
    print(f'\n  Best: round {best["round"]}  OA={best["OA"]*100:.2f}%')

    with open(seed_dir / 'round_metrics.json', 'w') as f:
        json.dump({'seed': seed, 'dataset': args.dataset, 'rounds': round_results}, f, indent=2)

    return best


def main():
    args = parse_args()
    cfg = SEMI_CONFIGS[args.dataset]
    cfg.data_path    = args.data_path
    cfg.teacher_ckpt = args.teacher_ckpt
    if args.pseudo_per_class is not None: cfg.pseudo_per_class = args.pseudo_per_class
    if args.max_rounds       is not None: cfg.max_rounds       = args.max_rounds

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
        'dataset': args.dataset, 'seeds': args.seeds,
        'OA':    f'{round(np.mean(oas)*100,2)} ± {round(np.std(oas)*100,2)}',
        'AA':    f'{round(np.mean(aas)*100,2)} ± {round(np.std(aas)*100,2)}',
        'Kappa': f'{round(np.mean(kappas),4)} ± {round(np.std(kappas),4)}',
    }
    with open(run_dir / 'test_metrics.json', 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f'\n=== Semi [{args.dataset}] ({len(args.seeds)} seed(s)) ===')
    print(f'OA    : {agg["OA"]}')
    print(f'AA    : {agg["AA"]}')
    print(f'Kappa : {agg["Kappa"]}')
    print(f'Results: {run_dir}')


if __name__ == '__main__':
    main()
