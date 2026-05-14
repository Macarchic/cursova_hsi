# TC-Former: Можливі покращення

## 1. Підбір гіперпараметрів

Автори статті підбирали параметри емпірично. Є простір для покращення:

| Параметр | Поточне (IP) | Що спробувати |
|---|---|---|
| `hidden_dim` | 64 | 96, 128 — більша ємність моделі |
| `depth` | 2 | 3 — глибша обробка послідовності |
| `num_heads` | 1 | 2, 4 — multi-head у TinyAttention |
| `dropout` | 0.1 | 0.2, 0.3 — при перенавчанні |
| `lr` | 5e-4 | 1e-3 з warm-up, або cosine з restarts |
| `patch_size` | 15 | 11, 13 — менший контекст іноді краще |
| `pca_components` | 150 | 30–100 — 150 майже надлишково для IP |
| `weight_decay` | 1e-4 | 1e-3 — сильніша регуляризація |

**Найперспективніший:** зменшити `pca_components` для IP (зараз 1.000 explained variance — це перебір).  
**Найризикованіший:** збільшити `depth` без збільшення `hidden_dim` — зростає кількість параметрів, але мало train-даних.

---

## 2. Оптимізація CNN для інференсу

Поточний стан: Conv2D запускається окремо на кожному патчі (правильно під час тренування, бо ваги змінюються).  
Для **inference** можна зробити один прохід по повному зображенню:

```python
# Замість patch loop:
with torch.no_grad():
    full_feat = model.stem(full_image_tensor)  # (1, 64, H, W) — один раз
    for (r, c) in coords:
        feat_patch = full_feat[:, :, r:r+P, c:c+P]  # slice, без перерахунку
        out = model.forward_from_stem(feat_patch)
```

Економить ~N_pixels повторних Conv2D forward passes. Для IP: ~10K разів → в рази швидший inference.

---

## 3. Заміна PCA на learnable spectral tokenizer

PCA — фіксована лінійна проекція, не адаптується до задачі класифікації.

```python
# Замість sklearn PCA:
class SpectralTokenizer(nn.Module):
    def __init__(self, in_bands, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_bands, out_dim, bias=False)
        # Ініціалізація PCA-компонентами для стабільного старту:
        # self.proj.weight.data = torch.from_numpy(pca.components_)

    def forward(self, x):  # x: (B, C, H, W)
        return self.proj(x.permute(0,2,3,1)).permute(0,3,1,2)
```

**Плюси:** навчається разом з моделлю, адаптується до класів, немає окремого preprocessing кроку.  
**Мінуси:** +C×D параметрів (200×64=12800), складніше навчити з 10 зразків/клас.  
**Компроміс:** ініціалізувати PCA-компонентами і дозволити дотягуватися градієнтами.

---

## 4. Bidirectional WKV (двонаправлена обробка)

Поточний WKV — causal: токен №112 (центр) бачить через WKV тільки токени 0–111.  
Нижня половина патча (113–224) не доступна йому напряму.

```python
class BidirectionalWKV(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wkv_fwd = WKVOperator(dim)  # зліва направо
        self.wkv_bwd = WKVOperator(dim)  # справа наліво
        self.merge   = nn.Linear(2*dim, dim, bias=False)

    def forward(self, k, v):
        fwd = self.wkv_fwd(k, v)
        bwd = self.wkv_bwd(k.flip(1), v.flip(1)).flip(1)
        return self.merge(torch.cat([fwd, bwd], dim=-1))
```

**Ефект:** центральний токен бачить весь патч через WKV, а не тільки половину.  
**Вартість:** +2× параметрів WKV, але WKV невеликий (~128 params).

---

## 5. 2D позиційне кодування для просторових токенів

Зараз токени 0–224 не знають свою (row, col) позицію в патчі. Порядок залежить від raster scan, а WKV — causal по цьому порядку (не по просторовій відстані від центру).

```python
class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        # Будує 2D sin-cos encoding і реєструє як buffer (не навчається)
        P = patch_size
        pe = torch.zeros(P*P, dim)
        for i in range(P):
            for j in range(P):
                idx = i*P + j
                pe[idx, 0::4] = torch.sin(torch.tensor(i / 10000 ** (torch.arange(dim//4)*4/dim)))
                pe[idx, 1::4] = torch.cos(torch.tensor(i / 10000 ** (torch.arange(dim//4)*4/dim)))
                pe[idx, 2::4] = torch.sin(torch.tensor(j / 10000 ** (torch.arange(dim//4)*4/dim)))
                pe[idx, 3::4] = torch.cos(torch.tensor(j / 10000 ** (torch.arange(dim//4)*4/dim)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe
```

Додати після stem flatten, перед TimeMixFormer.

---

## 6. Аугментація спектральних даних

З 10 зразків/клас перенавчання майже гарантоване. Аугментація критична.

```python
class SpectralAugmentation:
    def spectral_jitter(self, x, sigma=0.01):
        # Гаусівський шум до спектрального вектора
        return x + torch.randn_like(x) * sigma

    def band_dropout(self, x, p=0.1):
        # Випадково обнуляємо деякі PCA-компоненти
        mask = torch.bernoulli(torch.ones(x.shape[1]) * (1-p))
        return x * mask.view(1,-1,1,1)

    def spectral_mixup(self, x1, x2, alpha=0.2):
        # Mixup у спектральному просторі (не в класах)
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1-lam) * x2
```

Просторова аугментація (flip, rotate) теж корисна — HSI не має просторової орієнтації.

---

## 7. Label Smoothing + Class-Balanced Loss

**Label smoothing** — вирішує проблему overconfidence при малих даних.

**Проблема:** CrossEntropyLoss каже моделі "видай probability=1.0 для правильного класу". З 10 зразків на клас модель може буквально запам'ятати тренувальний набір і виводити probability≈0.99 з повною "впевненістю" — навіть для незнайомих тестових зразків. Модель не знає того що не знає.

**Рішення:** замінити ціль з `[0,0,1,0,...,0]` на `[ε/K, ε/K, 1-ε+ε/K, ε/K,...,ε/K]`. Модель ніколи не може повністю задовольнити цю ціль → не може ідеально переобнавчитися. Gradient залишається навіть коли "правильна відповідь" вже є.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # одна строчка, безплатно
```

**Class-balanced sampling** — класи з малою кількістю зразків (наприклад Alfalfa: 54 пікселі) недопредставлені. Виправити через `WeightedRandomSampler`:

```python
class_counts = np.bincount(train_labels)
weights = 1.0 / class_counts[train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
```

---

## 9. Semi-supervised: Consistency Regularization (Mean Teacher)

**Нелейбловані дані у нас є:** `labels==0` — це 10,776 пікселів в IP (145×145 − 10,249 labeled).

```
IP: 21,025 пікселів загалом
    labeled:   10,249  (з ground truth)
    unlabeled: 10,776  ← ці використовує Mean Teacher
```

Mean Teacher **не присвоює їм класи** (на відміну від pseudo-labeling). Він каже лише: "два різних аугментованих погляди на той самий нелейблований патч мають давати схожий prediction". Правильний клас знати не потрібно — тільки consistency.

Дві версії моделі:
- **Student** — навчається градієнтами
- **Teacher** — exponential moving average ваг student (повільно наздоганяє student, стабільніший)

```python
# EMA update teacher після кожного кроку
for tp, sp in zip(teacher.parameters(), student.parameters()):
    tp.data = 0.999 * tp.data + 0.001 * sp.data

# Loss: supervised на labeled + consistency на unlabeled
sup_loss     = ce_loss(student(x_labeled), y)
student_pred = student(augment(x_unlabeled))   # аугментований вхід
teacher_pred = teacher(x_unlabeled).detach()   # чистий вхід, без градієнта
cons_loss    = kl_div(student_pred, teacher_pred)
loss = sup_loss + λ * cons_loss
```

**Ефект:** teacher дає стабільні "soft targets" для нелейблованих пікселів. Student вчиться: (а) правильно класифікувати labeled, (б) бути стійким до аугментацій на unlabeled.

---

## 10. Contrastive Pretraining (SimCLR-style) — складно, ризиковано

**SimCLR** (Simple Contrastive Learning of Representations) — навчання без міток через порівняння:

```
Патч A → aug1 → embedding z1 ─┐
                                ├─ Loss: z1 і z2 мають бути БЛИЗЬКИМИ
Патч A → aug2 → embedding z2 ─┘

Патч B → aug1 → embedding z3 ── z1 і z3 мають бути ДАЛЕКИМИ
```

Ідея: без жодних міток модель вчиться "різні аугментації одного патча = одне і те ж, різні патчі = різні речі". Це змушує шукати справжні спектральні патерни, а не шум. Після pretrain → fine-tune з 10 мітками/клас.

```python
x1, x2 = augment(patch), augment(patch)
z1 = projector(backbone(x1))  # backbone = TCFormer без head
z2 = projector(backbone(x2))
loss = NT_Xent(z1, z2, temperature=0.5)  # NT-Xent = normalized temperature cross-entropy
```

**Складність:** окремий pretrain цикл, projection head, потім fine-tune — три етапи замість одного.  
**Ризик при малих даних:** якщо аугментації занадто сильні, модель вчиться ігнорувати корисні ознаки.

---

## Пріоритет впровадження

**Правило:** при малих даних (10 зразків/клас) додаткова складність шкодить якщо збільшує кількість параметрів. Корисна складність — та що вбудовує знання (positional encoding) або збільшує ефективну кількість даних (аугментація, Mean Teacher).

| Покращення | Очікуваний приріст OA | Складність | При малих даних |
|---|---|---|---|
| Label smoothing (п.7) | +0.5–1% | Мінімальна | Тільки краще |
| Spectral augmentation (п.6) | +1–3% | Низька | Збільшує ефективно дані |
| Class-balanced sampling (п.7) | +1–2% | Низька | Не додає параметрів |
| 2D positional encoding (п.5) | +0.5–1.5% | Середня | Buffer, не параметри |
| Bidirectional WKV (п.4) | +0.5–2% | Середня | Краща структура, мін. параметри |
| Mean Teacher (п.9) | +3–6% | Висока | Використовує 10K unlabeled |
| Learnable tokenizer (п.3) | +1–3% | Середня | +12K параметрів, ризик |
| Contrastive pretraining (п.10) | +2–5% | Висока | Складний pipeline, ризик |
| CNN inference optimization (п.2) | 0% OA | Середня | Тільки швидкість inference |

**Рекомендований порядок:** 7 → 6 → 5 → 4 → 9
