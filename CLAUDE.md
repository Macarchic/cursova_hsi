# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Курсова робота на основі статті **"Spectral Channel Mixing Transformer with Spectral-Center Attention for Hyperspectral Image Classification"** (TC-Former). Завдання:
1. Завантажити і дослідити датасети (EDA: розподіл класів, мода, граничні значення, кількість лейблів тощо)
2. Відтворити модель TC-Former із статті
3. Підготувати дані та навчити модель
4. Покращити метрики (напрямок — semi-supervised: consistency regularization, pseudo-labeling, contrastive pretraining)

## Article Summary

**Проблема:** HSI-класифікація з довгими спектральними векторами; стандартний Transformer має квадратичну складність self-attention.

**Рішення — TC-Former** (три послідовних блоки після PCA + Conv2D stem):
- **TimeMixFormer** — моделює залежності вздовж спектральної послідовності через RWKV-подібний WKV-оператор + Tiny Attention (замість квадратичного self-attention)
- **HyperMixFormer** — двопрохідне змішування каналів з gated WKV + Mish-активацією + Tiny Attention
- **Center Attention** — центральний піксель як Query, сусідні — як Key/Value; найважливіший блок (ablation: −3.14% OA на IP, −3.42% PU, −7.80% WHHH при видаленні)
- Фінальний MLP-класифікатор

**Пайплайн:** PCA (зменшення каналів) → Conv2D+BN+ReLU (локальні ознаки) → TimeMixFormer → HyperMixFormer → Center Attention → MLP head

## Datasets

### Indian Pines (IP)
- Сенсор: AVIRIS; 145×145 пікселів, 200 спектральних каналів (після видалення водяних смуг — 200→залишок)
- 16 класів земного покриву (сільськогосподарські культури, ліс тощо)
- Стандартний train/test split: **10 зразків на клас** для навчання
- Джерело: `http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes`
- Файли: `Indian_pines_corrected.mat` (дані) + `Indian_pines_gt.mat` (мітки)

### Pavia University (PU)
- Сенсор: ROSIS; 610×340 пікселів, 103 канали
- 9 класів (асфальт, луки, гравій, дерева тощо)
- **10 міток на клас** для навчання
- Файли: `PaviaU.mat` + `PaviaU_gt.mat`

### WHU-Hi-HongHu (WHHH)
- Сенсор: Headwall Nano-Hyperspec; 940×475 пікселів, 270 каналів
- 22 класи (складніший, найбільший датасет)
- **10 зразків на клас** для навчання
- Джерело: `http://irsip.whu.edu.cn/resources/WHU_Hi_Dataset.php`
- Файли: `WHU_Hi_HongHu.mat` + `WHU_Hi_HongHu_gt.mat`

### Де завантажити
Всі три датасети доступні на сторінці Hyperspectral Remote Sensing Scenes (University of the Basque Country) або через офіційні посилання в статті.

## Hyperparameters (з таблиці конфігурації статті)

| Parameter | IP | PU | WHHH |
|---|---|---|---|
| Patch size | 15 | 24 | 30 |
| PCA channels | 150 | 20 | 135 |
| Kernel size (Conv stem) | 9 | 17 | 19 |
| Batch size | 200 | 100 | 20 |
| Learning rate | 0.0005 | 0.0005 | 0.0005 |

> **Увага:** у тексті статті та в таблиці є суперечності (наприклад, у тексті patch size 15/20/30 і PCA 100/30/150, у таблиці інші значення). Орієнтуватись на **таблицю конфігурації**, а не на текстові описи.

## Metrics

- **OA** (Overall Accuracy) — основна метрика порівняння
- **AA** (Average Accuracy)
- **Kappa coefficient**
- Per-class accuracy, F1, confusion matrix, ROC curves

Baseline results із статті (supervised, 10 samples/class):
- IP: OA 90.01%, AA 88.35%, Kappa 0.889, F1 0.91
- PU: OA 94.60%, AA 93.91%, Kappa 0.930, F1 0.95
- WHHH: OA 91.56%, AA 88.23%, Kappa 0.904, F1 0.91

## Directions for Improvement

Найперспективніший напрямок — **semi-supervised TC-Former**:
- Consistency regularization між різними аугментаціями одного патча (unlabeled пікселі)
- Pseudo-labeling із порогом впевненості (лише для high-confidence predictions)
- Contrastive pretraining на нерозмічених спектральних даних
- Class-balanced loss для рідкісних класів (особливо для WHHH з 22 класами)
- Ці підходи логічні саме тут, бо labeled data = 10 зразків на клас — дефіцит розмітки очевидний

## Project Structure

```
cursova_hsi/
├── CLAUDE.md
├── article_summary.docx   # Конспект статті
├── data/                  # Сюди кладемо .mat файли датасетів
├── data_explor.ipynb      # EDA: дослідження датасетів
└── model.ipynb            # Реалізація TC-Former та навчання
```

## Environment

Рекомендований стек:
- Python 3.10+
- `torch` (PyTorch), `numpy`, `scipy` (для читання `.mat`), `sklearn`, `matplotlib`, `seaborn`
- Читання mat-файлів: `scipy.io.loadmat()` або `h5py` для новіших форматів HDF5

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn einops
```

Запуск ноутбуків: `jupyter notebook` або `jupyter lab`
