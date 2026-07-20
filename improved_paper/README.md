# improved_paper

Самодостатній пакет: **базова модель SCMT** + **нові шари з `improved/`**, де **кожен новий шар/трюк можна точечно вимкнути** (повний ablation аж до чистого SCMT).

За замовчуванням конфіг відтворює перевірену *improved*-модель. Вимкнувши всі перемикачі, отримуєте поведінку оригінального SCMT.

## Запуск

```bash
# з кореня репозиторію
python -m improved_paper.train --dataset IP --paper_mode
python -m improved_paper.train --dataset PU --seeds 0 1 2 --paper_mode
```

Дані очікуються в `data/` (сирий формат): `data/IP/Indian_pines_corrected.mat`+`IPgt.npy`, `data/Pavia/PaviaU*.mat`. Результати — у `results/improved_paper[_paper]_<DS>_v<N>/` (`config.json` зі знімком усіх перемикачів + `test_metrics.json`).

Залежності: `pip install -r improved_paper/requirements.txt` (потрібен `lightning`).

## Точкові перемикачі (ablation)

Кожен додаток над базовим SCMT керується прапорцем у `config.py` і CLI-опцією в `train.py`:

| Компонент | Прапорець конфіга | CLI (вимкнути/змінити) | Дефолт |
|---|---|---|---|
| Multi-scale conv stem | `use_multiscale_stem` | `--multiscale-stem` (увімкнути) | `False` |
| Squeeze-Excitation | `use_se_block` | `--no-se-block` | `True` |
| 2D sin-cos pos. encoding | `use_pos_encoding` | `--no-pos-encoding` | `True` |
| Mixing-реалізація | `mixing_impl` | `--mixing {improved,scmt}` | `improved` |
| Bidirectional WKV¹ | `use_bidirectional_wkv` | `--no-bidirectional` | `True` |
| Center attention | `center_attn` | `--center {ring,plain}` | `ring` |
| Label smoothing | `label_smoothing` | `--label-smoothing FLOAT` | `0.05` |
| Cosine LR schedule | `use_cosine_schedule` | `--no-cosine` | `True` |
| FPS-семплінг train | `use_fps` | `--no-fps` | `True` |
| Аугментації | `use_augmentation` | `--no-augmentation` | `True` |
| PCA-outlier removal | `remove_pca_outliers` | `--no-outlier-removal` | `True` |

¹ `use_bidirectional_wkv` діє лише при `mixing_impl='improved'` (WKV — це improved-реалізація; при `'scmt'` ігнорується).

## Режими даних / швидкість

| Прапорець | Що робить |
|---|---|
| `--scmt-split` | Тренування на **фіксованому спліті авторів SCMT** (`Indian_10_1_split.mat`: `input` 220 каналів, маски `TR`/`TE`) + SCMT-препроцесинг (max-min норма + `PCA(whiten=True)`). Дає дані 1:1 з авторами для чесного порівняння. **Лише IP.** Шлях: `--scmt-split-path` (дефолт `SCMT/data/Indian/Indian_10_1_split.mat`). |
| `--final-eval-only` | SCMT-стиль: **без валідації щоепохи**, тест міряється **один раз у кінці** на моделі фінальної епохи. Швидко (немає eval на ~10k патчів щоепохи). |

Максимальна парність **даних** з авторами SCMT:
```bash
python -m improved_paper.train --dataset IP --scmt-split --final-eval-only
```
(той самий спліт + той самий препроцесинг + фінальна модель на тесті).

## Запуск повністю як SCMT (`--emulate-scmt`)

Щоб `improved_paper` поводився як оригінальний SCMT (дані + модель + гіперпараметри) — одна команда:
```bash
python -m improved_paper.train --dataset IP --emulate-scmt
```
Вона вмикає: `--scmt-split` + `--final-eval-only`, базову SCMT-модель (`mixing=scmt`, `center=plain`, усі improved-шари вимкнені), без трюків (`aug/fps/outliers` off, `label_smoothing=0`, без cosine), і SCMT-гіперпараметри `weight_decay=0`, `grad_clip=15`. Окремі прапорці після `--emulate-scmt` мають пріоритет (напр. `--emulate-scmt --center ring` увімкне лише ring-attention поверх SCMT-бази).

Розгорнутий еквівалент:
```bash
python -m improved_paper.train --dataset IP --scmt-split --final-eval-only \
  --mixing scmt --center plain --no-se-block --no-pos-encoding --no-bidirectional \
  --no-augmentation --label-smoothing 0 --no-cosine --weight-decay 0 --grad-clip 15
```

Додаткові оверайди: `--weight-decay FLOAT`, `--grad-clip FLOAT`.

> Залишкова відмінність: оригінальний SCMT робив легкі геометричні аугментації (random flip/transpose); у `--emulate-scmt` аугментації вимкнено (навмисно, для чистоти). Також `mixing=scmt` відтворює оригінальний код разом із його баґом (невивчений runtime-Linear) — це очікувано, бо ми емулюємо саме оригінал.

## Крайні конфігурації

```bash
# Повний improved (дефолт)
python -m improved_paper.train --dataset IP --paper_mode

# Вимкнути один шар (приклад: без SE)
python -m improved_paper.train --dataset IP --paper_mode --no-se-block

# Чистий базовий SCMT (усі додатки вимкнені)
python -m improved_paper.train --dataset IP --paper_mode \
    --mixing scmt --center plain --no-se-block --no-pos-encoding \
    --no-bidirectional --no-fps --no-augmentation --no-outlier-removal \
    --label-smoothing 0 --no-cosine
```

## Структура

```
improved_paper/
├── config.py          # Config (усі перемикачі) + CONFIGS (IP/PU/WHHH)
├── layers_scmt.py     # оригінальні модулі SCMT (former, TimeMixFormer, ChannelMixFormer, TinyAttn)
├── layers_improved.py # нові шари (SE, PosEnc, BiWKV, MultiScaleStem, MultiRing center, RWKV-блоки)
├── model.py           # TCFormer (flag-композиція обох гілок) + TCFormerLit (Lightning)
├── dataset.py         # FPS-split, аугментації, HSIPatchDataset
├── preprocessing.py   # load_dataset, apply_pca, PCA-outlier removal
├── utils.py           # метрики, evaluate, run-dir helpers, device
├── train.py           # CLI із точковими ablation-прапорцями
└── paper_targets.json # орієнтири зі статті (OA/AA/Kappa/per-class)
```

## Обмеження
- `mixing_impl='scmt'` — це оригінальний код SCMT з його особливостями (хардкод dims 128/64/256, runtime-Linear усередині forward). Рекомендований (і дефолтний) шлях — `improved`.
- WHHH недоступний локально (немає датасету в `data/`); IP і PU доступні.
