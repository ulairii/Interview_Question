# Scaled Dot-Product Attention 与 Mask

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

导航：[总览](00_总览.md) · [SDPA 与 Mask](01_SDPA与Mask.md) · [Self-Attention](02_SelfAttention.md) · [Cross-Attention](03_CrossAttention.md) · [Multi-Head](04_MultiHead.md) · [KV Cache](05_KVCache.md) · [追问与验证](06_追问与验证.md)

面试时**一步步加**，不要一上来就写 mask 和 dropout：

| 步 | 什么时候写 | 加了什么 |
|---|---|---|
| 第 1 步 | 一开始 | 只有公式 `softmax(q kᵀ/√dk) v`，3 行 |
| 第 2 步 | 面试官问"decoder 怎么防止看到未来 / batch 里长度不同怎么办" | `mask` 参数 |
| 第 3 步 | 只有被问到才写，通常口头说一句即可 | 注意力权重上的 dropout |

下面每一步的代码都是**完整可运行**的（含 import 和自检），直接复制即可，不依赖其他文件。

## 第 1 步：最简版（先写这个）

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    # q:(...,T,dk)  k:(...,S,dk)  v:(...,S,dv)
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (...,T,S) 每个 query 与每个 key 的相似度
    w = F.softmax(scores, dim=-1)                          # 沿 key 维归一化，每行和为 1
    return w @ v                                           # (...,T,dv) 对 value 加权平均


# 自检：和 PyTorch 自带实现一致
q, k, v = torch.randn(2, 5, 8), torch.randn(2, 7, 8), torch.randn(2, 7, 4)
print(torch.allclose(scaled_dot_product_attention(q, k, v),
                     F.scaled_dot_product_attention(q, k, v), atol=1e-5))   # True
```

| 行 | 做了什么 | shape |
|---|---|---|
| `q @ kᵀ` | 每个 query 与每个 key 做点积 → 相似度 | (T,dk)@(dk,S) → (T,S) |
| `/ √dk` | 控制方差，避免 softmax 饱和 | 同上 |
| `softmax(dim=-1)` | **沿 S 归一化**：每个 query 的权重和为 1 | (T,S) |
| `w @ v` | 用权重对 value 加权平均 | (T,S)@(S,dv) → (T,dv) |

softmax 的 `dim=-1` 是 key 维，**不是** query 维。写成 `dim=-2` 是常见 bug，不报错但结果错。

**为什么除以 √dk**：q、k 元素独立、零均值、单位方差 ⇒ `q·k` 的方差为 `dk`。不缩放的话，dk 大时点积绝对值很大，softmax 趋于 one-hot，梯度接近 0。除以 `√dk` 把方差拉回 1。

## 第 2 步：加 mask（被问到 causal / padding 时再写）

约定：**bool，True = 可以看**。在 softmax **之前**把不能看的位置设成 -inf。

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    # mask: bool，可广播到 (...,T,S)，True=可见
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (...,T,S)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))  # 不可见位置 → -inf，softmax 后为 0
    w = F.softmax(scores, dim=-1)
    return w @ v


def causal_mask(T, S, device=None):
    # 第 i 个 query 只能看 key 0..S-T+i（T=S 时就是标准下三角）
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


# 自检 1：causal 和 PyTorch 一致
q, k, v = torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8)
print(torch.allclose(scaled_dot_product_attention(q, k, v, causal_mask(6, 6)),
                     F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5))  # True

# 自检 2：padding mask —— 改动被屏蔽的 key，输出不变
lengths = torch.tensor([6, 4])                               # 第 2 条只有 4 个真实 token
pad = lengths[:, None] > torch.arange(6)[None, :]            # (B,S) True=真实 token
mask = pad[:, None, :]                                       # (B,1,S) 广播到 (B,T,S)
k2, v2 = k.clone(), v.clone(); k2[1, 4:] = 100; v2[1, 4:] = 100
print(torch.allclose(scaled_dot_product_attention(q, k, v, mask),
                     scaled_dot_product_attention(q, k2, v2, mask)))           # True
```

**Causal mask**（T=S=4）：

```
        key0 key1 key2 key3
query0   1    0    0    0
query1   1    1    0    0
query2   1    1    1    0
query3   1    1    1    1
```

第 i 个 token 只能看 ≤ i 的位置，防止"偷看未来"。

**`diagonal=S-T` 的含义**：有 KV cache 时，T 个新 query 排在 S 个 key 的末尾。第 i 个新 query 的位置是 `S-T+i`，能看到 key `0..S-T+i`。

| 场景 | T | S | 结果 |
|---|---|---|---|
| 训练 / 整段前向 | n | n | 标准下三角 |
| 解码一步 | 1 | S | 整行 True |
| 带 cache 的 prefill 续写 | 3 | 10 | 前 7 列全 True，后 3 列下三角 |

**Padding mask**：batch 内序列长度不同，补 0 的位置不应被看到，形状 (B,S) → 加维度广播。多头时广播成 `(B,1,1,S)`。

两种 mask 同时用：`mask = pad_mask & causal_mask`。

## 第 3 步：加 dropout（一般口头说即可）

dropout 加在 **softmax 之后的注意力权重**上，只在训练时生效。面试时通常说一句"训练时可以在权重上加 dropout"就够了，被要求写才写：

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = F.softmax(scores, dim=-1)
    w = F.dropout(w, dropout_p, training)                  # 只在 training=True 时随机置 0 并放大
    return w @ v, w                                        # 顺便返回权重，便于可视化/调试


# 自检：eval 模式下 dropout 不起作用，结果和第 2 步一样
q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
out, w = scaled_dot_product_attention(q, k, v, dropout_p=0.1, training=False)
print(torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5),  # True
      torch.allclose(w.sum(-1), torch.ones(2, 5)))                               # True：每行和为 1
```

## 边界与坑

- **整行全 -inf → softmax 得 NaN**（比如某个 query 对应的全是 padding）。解决：用一个很大的负数代替 `-inf`，或对 NaN 行置 0。
- mask 约定：这里统一 bool，True = 保留；`nn.MultiheadAttention` 的 `key_padding_mask` 约定相反（True = 屏蔽），要看文档。
- 在 softmax **之前**加 mask，不是之后。
- dropout 作用在注意力权重上（softmax 之后）。

## NumPy 版（不能用 torch 时）

和上面一样分三步写。NumPy 没有现成的 softmax，要自己写，并且**减去每行最大值**防止 `exp` 溢出（这是常被追问的点）。

### 第 1 步：最简版

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)                # 减最大值：数值稳定，结果不变
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v):
    # q:(...,T,dk)  k:(...,S,dk)  v:(...,S,dv)
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])   # (...,T,S)
    return softmax(scores, axis=-1) @ v                           # (...,T,dv)


# 自检：和逐元素 for 循环的写法一致
def naive(q, k, v):                                        # 单个样本 q:(T,dk) k:(S,dk) v:(S,dv)
    T, S = q.shape[0], k.shape[0]
    out = np.zeros((T, v.shape[1]))
    for i in range(T):
        s = np.array([q[i] @ k[j] for j in range(S)]) / np.sqrt(q.shape[1])
        w = np.exp(s - s.max()); w /= w.sum()
        out[i] = sum(w[j] * v[j] for j in range(S))
    return out

rng = np.random.default_rng(0)
q, k, v = rng.normal(size=(5, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 4))
print(np.allclose(scaled_dot_product_attention(q, k, v), naive(q, k, v)))   # True
```

### 第 2 步：加 mask

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    # mask: bool，可广播到 (...,T,S)，True=可见
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)           # 不可见 → -inf
    return softmax(scores, axis=-1) @ v


def causal_mask(T, S):
    return np.tril(np.ones((T, S), dtype=bool), k=S - T)   # 第 i 个 query 能看 key 0..S-T+i


# 自检 1：causal —— 第 0 个 query 只能看第 0 个 key，输出就是 v[0]
rng = np.random.default_rng(0)
q, k, v = rng.normal(size=(6, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
out = scaled_dot_product_attention(q, k, v, causal_mask(6, 6))
print(np.allclose(out[0], v[0]))                           # True

# 自检 2：padding —— 改动被屏蔽的 key/value，输出不变
mask = np.arange(6) < 4                                    # (S,) 只有前 4 个是真实 token
k2, v2 = k.copy(), v.copy(); k2[4:] = 100; v2[4:] = 100
print(np.allclose(scaled_dot_product_attention(q, k, v, mask),
                  scaled_dot_product_attention(q, k2, v2, mask)))   # True
```

`np.where(mask, scores, -np.inf)` 对应 torch 的 `masked_fill(~mask, -inf)`。整行都被屏蔽时，`softmax` 里会出现 `-inf - (-inf) = nan`，和 torch 版一样要单独保护。

### 第 3 步：加 dropout

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False, rng=None):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    if training and dropout_p > 0:                         # inverted dropout：保留的放大 1/(1-p)
        rng = rng or np.random.default_rng()
        w = w * (rng.random(w.shape) >= dropout_p) / (1 - dropout_p)
    return w @ v, w


# 自检：eval 模式下权重每行和为 1；训练模式下期望不变
rng = np.random.default_rng(0)
q, k, v = rng.normal(size=(5, 8)), rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
_, w = scaled_dot_product_attention(q, k, v)
print(np.allclose(w.sum(-1), 1))                                              # True
avg = np.mean([scaled_dot_product_attention(q, k, v, dropout_p=0.2, training=True, rng=rng)[1]
               for _ in range(4000)], axis=0)
print(np.allclose(avg, w, atol=0.02))                                         # True
```


<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, S: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular mask aligned to the END of the keys (works with a KV cache).

    Returns:
        (T, S) bool; query i (the i-th of the last T positions) sees keys 0..S-T+i.
    """
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout_p: float = 0.0, training: bool = False
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T / sqrt(dk)) v

    Args:
        q: (..., T, dk) queries.
        k: (..., S, dk) keys.
        v: (..., S, dv) values.
        mask: bool, broadcastable to (..., T, S); True = keep. None = no mask.
    Returns:
        out: (..., T, dv); weights: (..., T, S).
    """
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5        # (..., T, S)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = F.softmax(scores, dim=-1)
    w = F.dropout(w, dropout_p, training)
    return w @ v, w                                             # (..., T, dv), (..., T, S)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



