# Self-Attention（自注意力）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

导航：[总览](00_总览.md) · [SDPA 与 Mask](01_SDPA与Mask.md) · [Self-Attention](02_SelfAttention.md) · [Cross-Attention](03_CrossAttention.md) · [Multi-Head](04_MultiHead.md) · [KV Cache](05_KVCache.md) · [追问与验证](06_追问与验证.md)

## 定义

Q、K、V **都由同一个序列 x 投影得到**，序列里每个位置都去"查"同一序列里的所有位置。

## 第 1 步：不带 mask（encoder 用的双向版本）

完整可运行，不依赖其他文件：

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (B,T,T)
    return F.softmax(scores, dim=-1) @ v                    # (B,T,dv)


def self_attention(x, Wq, Wk, Wv):
    # x:(B,T,D)  Wq,Wk:(D,dk)  Wv:(D,dv)
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # Q、K、V 都来自同一个 x
    return scaled_dot_product_attention(q, k, v)           # (B,T,dv)


# 自检
B, T, D, dk = 2, 6, 16, 8
x = torch.randn(B, T, D)
Wq, Wk, Wv = torch.randn(D, dk), torch.randn(D, dk), torch.randn(D, dk)
ref = F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv)
print(torch.allclose(self_attention(x, Wq, Wk, Wv), ref, atol=1e-5))   # True
```

## 第 2 步：加 causal mask（decoder / GPT 用的单向版本）

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))  # True=可见
    return F.softmax(scores, dim=-1) @ v


def causal_mask(T, S, device=None):
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


def self_attention(x, Wq, Wk, Wv, causal=False):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    T = x.size(1)
    mask = causal_mask(T, T, x.device) if causal else None
    return scaled_dot_product_attention(q, k, v, mask)


# 自检
B, T, D, dk = 2, 6, 16, 8
x = torch.randn(B, T, D)
Wq, Wk, Wv = torch.randn(D, dk), torch.randn(D, dk), torch.randn(D, dk)
ref = F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=True)
print(torch.allclose(self_attention(x, Wq, Wk, Wv, causal=True), ref, atol=1e-5))   # True
# 第 0 个位置只能看自己 → 输出就是它自己的 v
print(torch.allclose(self_attention(x, Wq, Wk, Wv, causal=True)[:, 0], (x @ Wv)[:, 0], atol=1e-5))  # True
```

## 数据流（T = S）

```
x (B,T,D) ──@Wq──> Q (B,T,dk) ┐
          ──@Wk──> K (B,T,dk) ├─> scores (B,T,T) ─mask─softmax─> W (B,T,T) ─@V─> out (B,T,dv)
          ──@Wv──> V (B,T,dv) ┘
```

## 两种用法

| | Encoder 自注意力 | Decoder 自注意力 |
|---|---|---|
| mask | 只有 padding mask | **causal** + padding |
| 每个 token 能看到 | 整个序列（双向） | 只能看自己和之前（单向） |
| 代表 | BERT、ViT | GPT、LLaMA |

## 性质

- 对 token 顺序**置换等变**：不加位置编码就不知道顺序 → 需要位置编码（正弦 / 可学习 / RoPE）。
- 复杂度：时间 O(T²·d)，显存 O(T²)（score 矩阵）。序列越长越贵。
- 每个位置一步就能与任意位置交互（路径长度 O(1)），这是它相比 RNN 的优势。

## 手算小例子（检查理解）

T=2，dk=1，`q=[1,0]ᵀ`（两个位置的 q 分别是 1 和 0），`k=[1,0]ᵀ`，`v=[10,20]ᵀ`，不缩放：

scores = q kᵀ = [[1,0],[0,0]]；
softmax 行：第 0 行 `[e/(e+1), 1/(e+1)] ≈ [0.73, 0.27]`，第 1 行 `[0.5, 0.5]`；
out = [0.73·10+0.27·20, 0.5·10+0.5·20] ≈ [12.7, 15]。

## 常见追问

- self-attn 与 cross-attn 的代码差别？→ 只差 K/V 是否也来自 x，见 [Cross-Attention](03_CrossAttention.md)。
- 为什么 Q、K 要用不同的投影而不是直接 `x xᵀ`？→ 允许"我要找什么"与"我是什么"处于不同子空间；也让相似度矩阵不必对称。

## NumPy 版（不能用 torch 时）

完整可运行，不依赖其他文件：

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)                # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)           # True=可见
    return softmax(scores, axis=-1) @ v


def causal_mask(T, S):
    return np.tril(np.ones((T, S), dtype=bool), k=S - T)


def self_attention(x, Wq, Wk, Wv, causal=False):
    # x:(B,T,D)  Wq,Wk:(D,dk)  Wv:(D,dv)
    q, k, v = x @ Wq, x @ Wk, x @ Wv                       # Q、K、V 都来自 x
    T = x.shape[1]
    mask = causal_mask(T, T) if causal else None
    return scaled_dot_product_attention(q, k, v, mask)     # (B,T,dv)


# 自检
rng = np.random.default_rng(0)
B, T, D, dk = 2, 6, 16, 8
x = rng.normal(size=(B, T, D))
Wq, Wk, Wv = rng.normal(size=(D, dk)), rng.normal(size=(D, dk)), rng.normal(size=(D, dk))
out = self_attention(x, Wq, Wk, Wv, causal=True)
print(out.shape)                                           # (2, 6, 8)
print(np.allclose(out[:, 0], (x @ Wv)[:, 0]))              # True：第 0 个位置只能看自己
# causal：改动最后一个 token，不影响前面位置的输出
x2 = x.copy(); x2[:, -1] = 100
print(np.allclose(self_attention(x2, Wq, Wk, Wv, causal=True)[:, :-1], out[:, :-1]))   # True
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


def attention(x_q: torch.Tensor, x_kv: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
              Wv: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Single-head attention. Self-attention if x_kv is x_q; cross-attention otherwise.

    Args:
        x_q: (B, T, Dq) the sequence that asks (decoder side).
        x_kv: (B, S, Dkv) the sequence that is looked up (encoder side / same as x_q).
        Wq: (Dq, dk), Wk: (Dkv, dk), Wv: (Dkv, dv) projection matrices.
        mask: bool broadcastable to (B, T, S), or None.
    Returns:
        (B, T, dv).
    """
    q, k, v = x_q @ Wq, x_kv @ Wk, x_kv @ Wv                    # (B,T,dk) (B,S,dk) (B,S,dv)
    return scaled_dot_product_attention(q, k, v, mask)[0]


def self_attention(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor,
                   causal: bool = False) -> torch.Tensor:
    """x: (B, T, D); W*: (D, d) -> (B, T, d). Queries, keys, values all come from x."""
    mask = causal_mask(x.size(1), x.size(1), x.device) if causal else None
    return attention(x, x, Wq, Wk, Wv, mask)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



