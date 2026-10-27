# KV Cache（自回归解码加速）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

导航：[总览](00_总览.md) · [SDPA 与 Mask](01_SDPA与Mask.md) · [Self-Attention](02_SelfAttention.md) · [Cross-Attention](03_CrossAttention.md) · [Multi-Head](04_MultiHead.md) · [KV Cache](05_KVCache.md) · [追问与验证](06_追问与验证.md)

## 动机

自回归生成时，第 t 步只需要新 token 的 query，但要对**所有历史 token 的 K、V** 做注意力。历史 token 的 K、V 不会变 → **算一次存起来**，之后每步只算新 token 的 K、V 并追加。

| | 无 cache | 有 cache |
|---|---|---|
| 第 t 步重算 | 全部 t 个 token 的 Q、K、V | 只算 1 个 token 的 Q、K、V |
| 注意力计算 | O(t²) | O(t)（1×t 的 score） |
| 生成 n 个 token 总计 | O(n³)（逐步重算整段） | O(n²) |
| 代价 | 无 | 显存 `2·L·B·H·S·dk` |

## 第 1 步：只做自注意力的 cache（先写这个）

完整可运行，不依赖其他文件。核心只有两行 `torch.cat`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))  # True=可见
    return F.softmax(scores, dim=-1) @ v


def causal_mask(T, S, device=None):
    # 新的 T 个 query 排在 S 个 key 末尾；T=1 时整行可见
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h, self.dk = num_heads, d_model // num_heads
        self.q_proj, self.k_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v_proj, self.out_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)

    def _split(self, x):                                   # (B,L,C) -> (B,H,L,dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(x)), self._split(self.v_proj(x))
        if kv_cache is not None:                           # 把历史 K/V 拼在前面（序列维 dim=2）
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = scaled_dot_product_attention(q, k, v, causal_mask(T, k.size(2), x.device))
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k, v)                  # 返回新的 cache


# 自检：逐 token 带 cache 解码 == 整段 causal 前向
torch.manual_seed(0)
attn = CausalSelfAttention(32, 4).eval()
x = torch.randn(2, 7, 32)
full, _ = attn(x)                                          # 整段一次算
cache, steps = None, []
for t in range(7):                                         # 每步只喂 1 个 token
    y, cache = attn(x[:, t:t + 1], kv_cache=cache)
    steps.append(y)
print(torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5))   # True
print(cache[0].shape)                                      # (2,4,7,8)：缓存长度随生成增长
```

解码步 shape 变化（T=1）：

| 步 | x | q | 缓存 k | scores | 输出 |
|---|---|---|---|---|---|
| 1 | (B,1,C) | (B,H,1,dk) | (B,H,1,dk) | (B,H,1,1) | (B,1,C) |
| 2 | (B,1,C) | (B,H,1,dk) | (B,H,2,dk) | (B,H,1,2) | (B,1,C) |
| t | (B,1,C) | (B,H,1,dk) | (B,H,t,dk) | (B,H,1,t) | (B,1,C) |

- 每步 **query 只有 1 个**，causal mask 是 (1,S) 整行 True；`causal_mask(T, S)` 的 `diagonal=S-T` 自动正确，不必特殊处理。
- 缓存要在 **投影之后**（存 K、V，不是存 x）。
- 缓存沿 **序列维 dim=2** 拼接，不是 batch 或 head 维。

## 第 2 步：加上交叉注意力的 cache（被问到 encoder-decoder 时）

context（encoder 输出）在整个解码过程中不变，K/V **第一步算好，之后直接复用，不追加**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return F.softmax(scores, dim=-1) @ v


def causal_mask(T, S, device=None):
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h, self.dk = num_heads, d_model // num_heads
        self.q_proj, self.k_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v_proj, self.out_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x, context=None, mask=None, is_causal=False, kv_cache=None):
        B, T, C = x.shape
        q = self._split(self.q_proj(x))
        if context is not None and kv_cache is not None:   # 交叉注意力：直接复用 context 的 K/V
            k, v = kv_cache
        else:
            src = x if context is None else context
            k, v = self._split(self.k_proj(src)), self._split(self.v_proj(src))
            if kv_cache is not None:                       # 自注意力：追加新 token 的 K/V
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
        if is_causal:
            cm = causal_mask(T, k.size(2), x.device)
            mask = cm if mask is None else mask & cm
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k, v)


# 自检：交叉注意力复用 cache 的结果 == 每步重算
torch.manual_seed(0)
cross = MultiHeadAttention(32, 4).eval()
x, enc = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
_, ctx_cache = cross(x[:, :1], context=enc)                # 第 1 步算出 context 的 K/V
same = all(torch.allclose(cross(x[:, t:t + 1], context=enc, kv_cache=ctx_cache)[0],
                          cross(x[:, t:t + 1], context=enc)[0], atol=1e-6) for t in range(5))
print(same, ctx_cache[0].shape)                            # True (2,4,9,8)：长度固定为 S
```

| | 自注意力 cache | 交叉注意力 cache |
|---|---|---|
| 内容 | 已生成 token 的 K/V | context 的 K/V |
| 每步更新 | **追加**新 token | **不变** |
| 长度 | 随生成增长 | 固定 = S |

## 推理循环（口述即可）

```python
cache = None
for t in range(max_new_tokens):
    y, cache = attn(x_t, kv_cache=cache)    # x_t: (B,1,C)
    x_t = next_token_embedding(y)
```

首步常把整个 prompt 一次性送入（prefill，T=n，cache=None），之后每步 T=1。

## 追问

| 追问 | 回答 |
|---|---|
| 显存怎么估？ | `2 × 层数 × batch × 头数(KV) × 序列长度 × dk × 字节数`；长上下文下 cache 往往比权重还大 |
| 怎么减小？ | GQA/MQA、量化 KV（int8/fp8）、滑动窗口、PagedAttention（分页管理避免碎片）|
| 训练时用吗？ | 不用，训练是整段并行 + causal mask |
| 为什么不缓存 Q？ | 历史 token 的 Q 之后不会再被用到 |
| 与结果的等价性？ | 数学上完全等价于整段重算（上面第 1 步的自检就是验证这个）|

## NumPy 版（不能用 torch 时）

把权重存在一个普通类里，cache 用 `np.concatenate` 沿序列维拼接：

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ v


def causal_mask(T, S):
    return np.tril(np.ones((T, S), dtype=bool), k=S - T)   # T=1 时整行可见


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng):
        self.h, self.dk = num_heads, d_model // num_heads
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
                                              for _ in range(4))

    def _split(self, x):                                   # (B,L,C) -> (B,H,L,dk)
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.dk).transpose(0, 2, 1, 3)

    def __call__(self, x, context=None, is_causal=False, kv_cache=None):
        B, T, C = x.shape
        q = self._split(x @ self.Wq)
        if context is not None and kv_cache is not None:   # 交叉注意力：复用 context 的 K/V
            k, v = kv_cache
        else:
            src = x if context is None else context
            k, v = self._split(src @ self.Wk), self._split(src @ self.Wv)
            if kv_cache is not None:                       # 自注意力：沿序列维(axis=2)追加
                k = np.concatenate([kv_cache[0], k], axis=2)
                v = np.concatenate([kv_cache[1], v], axis=2)
        mask = causal_mask(T, k.shape[2]) if is_causal else None
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return out @ self.Wo, (k, v)


# 自检 1：自注意力，逐 token 带 cache == 整段 causal 前向
rng = np.random.default_rng(0)
attn = MultiHeadAttention(32, 4, rng)
x = rng.normal(size=(2, 7, 32))
full, _ = attn(x, is_causal=True)
cache, steps = None, []
for t in range(7):
    y, cache = attn(x[:, t:t + 1], is_causal=True, kv_cache=cache)
    steps.append(y)
print(np.allclose(full, np.concatenate(steps, axis=1)), cache[0].shape)   # True (2, 4, 7, 8)

# 自检 2：交叉注意力，复用 cache == 每步重算
enc = rng.normal(size=(2, 9, 32))
_, ctx_cache = attn(x[:, :1], context=enc)
print(all(np.allclose(attn(x[:, t:t + 1], context=enc, kv_cache=ctx_cache)[0],
                      attn(x[:, t:t + 1], context=enc)[0]) for t in range(7)))   # True
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


KVCache = Tuple[torch.Tensor, torch.Tensor]   # (k, v), each (B, H, S, dk)


class MultiHeadAttention(nn.Module):
    """Module version: self-attention (context=None) or cross-attention, with KV cache."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dk = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> (B, H, L, dk)."""
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        """
        Args:
            x: (B, T, C) queries' source (T=1 when decoding step by step).
            context: (B, S, C) keys/values' source for cross-attention; None = self-attention.
            mask: bool broadcastable to (B, 1, T, S); True = keep (e.g. padding mask).
            is_causal: add the causal mask (self-attention decoding).
            kv_cache: self-attn: previous (k, v) to append to; cross-attn: the (k, v) of
                the context computed on the first step, reused as-is.
        Returns:
            y: (B, T, C); cache: (k, v), each (B, H, S_total, dk).
        """
        B, T, C = x.shape
        q = self._split(self.q_proj(x))                         # (B, H, T, dk)
        if context is not None and kv_cache is not None:        # cross-attn: reuse cached context K/V
            k, v = kv_cache
        else:
            src = x if context is None else context
            k, v = self._split(self.k_proj(src)), self._split(self.v_proj(src))  # (B, H, S, dk)
            if kv_cache is not None:                            # self-attn: append new K/V
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
        if is_causal:
            cm = causal_mask(T, k.size(2), x.device)            # (T, S)
            mask = cm if mask is None else mask & cm
        out, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout_p, self.training)
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)
        return self.out_proj(out), (k, v)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



