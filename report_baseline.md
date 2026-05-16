# Baseline Model: опис та відповідність статті

## Загальна архітектура (за статтею)

TC-Former складається з послідовних компонентів:

```
Вхід (H×W×C)
  → PCA (C → pca_components)
  → Conv2D + BN + ReLU  (stem)
  → TimeMixFormer × depth
  → HyperMixFormer × depth
  → CenterAttention
  → MLPHead → клас
```

---

## Компоненти та їх реалізація

### 1. WKV Operator (`WKVOperator`)

**Стаття:** лінійна альтернатива self-attention на основі RWKV-4, з каузальним decay і learnable параметрами `w` (decay) та `u` (current-token bias). Замінює квадратичну складність на лінійну вздовж послідовності.

**Реалізація:** повністю векторизована (без Python-циклу). Ключова деталь — exp-маскування некаузальних позицій через `-inf` перед `exp()`, щоб уникнути `NaN` на MPS/CUDA:

```python
exp_safe = exp.masked_fill(~causal.unsqueeze(-1), float('-inf'))
```

**Відповідність:** ✅ формула збігається з RWKV-4 (рівняння 5 зі статті).

---

### 2. TimeMixing (`TimeMixing`)

**Стаття:** time shift між поточним і попереднім токеном через learnable `μ`-параметри, три проєкції (R, K, V) з learnable scalar weights, sigmoid gate на R, WKV-агрегація.

**Реалізація:**
- `_shift(x)` — зміщення на 1 позицію (попередній токен)
- Learnable `mu_r`, `mu_k`, `mu_v` для інтерполяції поточний/попередній
- Learnable scalar `W_r`, `W_k`, `W_v` (broadcasting по `(1,1,dim)`)
- `W_o` — фінальна лінійна проєкція

**Відповідність:** ✅ повністю відповідає описаному в статті механізму.

---

### 3. HyperMixing (`HyperMixing`)

**Стаття:** аналогічний до TimeMixing, але призначений для змішування каналів. Використовує gated WKV, Mish-активацію, shared-weight повторний прохід.

Ключова нотаційна відмінність від TimeMixing:
- TimeMixing (eq. 2-4): `W_r ⊙ (...)` — element-wise scaling (learnable вектор)
- HyperMixing (eq. 8-9): `W'_r · (...)` — лінійна проєкція (матриця `dim×dim`)

**Реалізація:**
- Time shift і mu-параметри тільки для R і K (без окремого V-проєктора)
- `W_r`, `W_k` — `nn.Linear(dim, dim, bias=False)` (лінійна проєкція, eq. 8-9)
- `v_prime = self.wkv(k, x)` — вхід `x` безпосередньо як V у WKV
- Gate: `σ(r) ⊙ W_h(Mish(k) ⊙ v_prime)` (eq. 10)

**Відповідність:** ✅

---

### 4. TinyAttention (`TinyAttention`)

**Стаття:** легкий multi-head self-attention (мала кількість голів, спільні проєкції) для збереження часткового глобального контексту без великої обчислювальної ціни.

**Реалізація:** стандартний scaled dot-product attention з `qkv`-проєкцією, `num_heads` голів, вихідна `proj`.

**Відповідність:** ✅

---

### 5. TimeMixFormerBlock (`TimeMixFormerBlock`)

**Стаття:** два послідовних TimeMixing з residual + LayerNorm, після чого TinyAttention.

**Реалізація:**
```python
x = x + self.tm1(self.norm1(x))
x = x + self.tm2(self.norm2(x))
x = x + self.attn(self.norm3(x))
```

**Відповідність:** ✅

---

### 6. HyperMixFormerBlock (`HyperMixFormerBlock`)

**Стаття:** HyperMixing викликається двічі зі **спільними вагами** + TinyAttention.

**Реалізація:** один інстанс `self.hm` викликається двічі через різні `LayerNorm`:
```python
x = x + self.hm(self.norm1(x))
x = x + self.hm(self.norm2(x))
x = x + self.attn(self.norm3(x))
```

**Відповідність:** ✅ weight sharing реалізований правильно.

---

### 7. CenterAttention (`CenterAttention`)

**Стаття:** центральний піксель патча — Query; всі токени — Key/Value. За ablation study — найважливіший блок: без нього OA падає на 3.14% (IP), 3.42% (PU), 7.80% (WHHH).

**Реалізація:**
```python
self.center_idx = (patch_size // 2) * patch_size + (patch_size // 2)
```
Center як Q, всі `T` токенів як K/V → scaled dot-product → вихідний вектор + residual від центрального токена.

**Відповідність:** ✅ індекс центру правильний (для непарного patch_size).

---

### 8. MLPHead (`MLPHead`)

**Стаття:** фінальний MLP-класифікатор, деталі внутрішньої структури не специфіковані.

**Реалізація:** `Linear(D → 4D) → BN → Dropout → ReLU → Linear(4D → num_classes)`.

**Відповідність:** ✅ стандартний bottleneck, обґрунтований вибір.

---

### 9. Загальний пайплайн (`TCFormer`)

```python
x = self.stem(x)                        # Conv2D+BN+ReLU
x = x.flatten(2).transpose(1, 2)        # (B, T, D), T = P²
for block in self.time_mix_formers:
    x = block(x)
for block in self.hyper_mix_formers:
    x = block(x)
x = self.center_attn(x)                 # (B, D)
return self.head(x)                     # (B, num_classes)
```

**Відповідність:** ✅ порядок блоків відповідає статті.

---

## Відхилення та примітки

| # | Аспект | Стаття | Реалізація | Статус |
|---|--------|--------|------------|--------|
| 1 | Patch size PU | 24 (таблиця) | 25 | ⚠️ навмисне: формула `2*(p//2)+1` вимагає непарного числа для наявності центрального пікселя |
| 2 | Patch size WHHH | 30 (таблиця) | 31 | ⚠️ та сама причина |
| 3 | Позиційне кодування | не згадується | відсутнє в baseline | ✅ додано в improved |
| 4 | Напрямок WKV | однонаправлений (каузальний) | однонаправлений | ✅ |
| 5 | LR / batch size | суперечності між текстом і таблицею | використано значення з таблиці (lr=5e-4) | ✅ |
| 6 | W'_r, W'_k у HyperMixing | `·` — лінійна проєкція `nn.Linear` | виправлено з `nn.Parameter` на `nn.Linear(dim, dim)` | ✅ |

---

## Висновок

Реалізація baseline відповідає архітектурі TC-Former зі статті. Усі ключові компоненти — WKV, time shift, gated HyperMixing, TinyAttention, CenterAttention — реалізовані коректно. Єдине відхилення у patch size (24→25, 30→31) є навмисним і технічно обґрунтованим: патч повинен мати центральний піксель, тому розмір має бути непарним.
