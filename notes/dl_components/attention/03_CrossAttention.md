# Cross-Attention（交叉注意力）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

导航：[总览](00_总览.md) · [SDPA 与 Mask](01_SDPA与Mask.md) · [Self-Attention](02_SelfAttention.md) · [Cross-Attention](03_CrossAttention.md) · [Multi-Head](04_MultiHead.md) · [KV Cache](05_KVCache.md) · [追问与验证](06_追问与验证.md)

## 定义

**Q 来自序列 A（x），K、V 来自另一个序列 B（context）**。A 的每个位置去 B 里检索信息。T 与 S 可以不同。

## 第 1 步：不带 mask

完整可运行，不依赖其他文件：

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (B,T,S)
    return F.softmax(scores, dim=-1) @ v                    # (B,T,dv)


def cross_attention(x, context, Wq, Wk, Wv):
    # x:(B,T,Dq)  context:(B,S,Dkv)  Wq:(Dq,dk)  Wk:(Dkv,dk)  Wv:(Dkv,dv)
    q = x @ Wq                                             # Q 来自 x
    k, v = context @ Wk, context @ Wv                      # K、V 来自 context
    return scaled_dot_product_attention(q, k, v)           # (B,T,dv)：长度跟 query 走


# 自检
B, T, S, Dq, Dkv, dk = 2, 5, 9, 16, 12, 8                  # T≠S，两边维度也可以不同
x, ctx = torch.randn(B, T, Dq), torch.randn(B, S, Dkv)
Wq, Wk, Wv = torch.randn(Dq, dk), torch.randn(Dkv, dk), torch.randn(Dkv, dk)
ref = F.scaled_dot_product_attention(x @ Wq, ctx @ Wk, ctx @ Wv)
print(torch.allclose(cross_attention(x, ctx, Wq, Wk, Wv), ref, atol=1e-5))   # True
```

## 第 2 步：加 context 的 padding mask

cross-attention **不需要 causal mask**，只需要屏蔽 context 里补齐的 padding：

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))  # True=可见
    return F.softmax(scores, dim=-1) @ v


def cross_attention(x, context, Wq, Wk, Wv, context_mask=None):
    # context_mask:(B,S) bool，True=真实 token
    mask = context_mask[:, None, :] if context_mask is not None else None   # (B,1,S) 广播到 (B,T,S)
    return scaled_dot_product_attention(x @ Wq, context @ Wk, context @ Wv, mask)


# 自检：改动 context 里被屏蔽的 padding，输出不变
B, T, S, D, dk = 2, 5, 9, 16, 8
x, ctx = torch.randn(B, T, D), torch.randn(B, S, D)
Wq, Wk, Wv = torch.randn(D, dk), torch.randn(D, dk), torch.randn(D, dk)
lengths = torch.tensor([9, 6])
ctx_mask = lengths[:, None] > torch.arange(S)[None, :]      # (B,S)
ctx2 = ctx.clone(); ctx2[1, 6:] = 100
print(torch.allclose(cross_attention(x, ctx, Wq, Wk, Wv, ctx_mask),
                     cross_attention(x, ctx2, Wq, Wk, Wv, ctx_mask)))      # True
# context = x 时就是 self-attention
print(torch.allclose(cross_attention(x, x, Wq, Wk, Wv),
                     F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv), atol=1e-5))  # True
```

## 数据流（T ≠ S）

```
x       (B,T,Dq)  ──@Wq──> Q (B,T,dk) ┐
context (B,S,Dkv) ──@Wk──> K (B,S,dk) ├─> scores (B,T,S) ─mask─softmax─> W (B,T,S) ─@V─> out (B,T,dv)
                  ──@Wv──> V (B,S,dv) ┘
```

输出长度 = **query 的长度 T**；信息量来自 context。

## 与 Self-Attention 对比

| | Self | Cross |
|---|---|---|
| K、V 来源 | 与 Q 相同 | 另一个序列 |
| 注意力矩阵 | (T,T) | (T,S) |
| causal mask | decoder 里需要 | **不需要**，query 可看整个 context |
| padding mask | 作用在自己的序列 | 作用在 **context** 的 padding 上 |
| Wk/Wv 输入维 | = Q 的维度 | = context 的维度（可不同于 Q） |

## 用在哪里

| 场景 | Q（提问方） | K/V（资料库） |
|---|---|---|
| Transformer 翻译 decoder | 已生成的目标语言 token | encoder 输出的源语言表示 |
| Stable Diffusion / 文生图 | 图像 latent 位置 | 文本 embedding |
| DETR | 可学习的 object query | 图像特征图 |
| 多模态 LLM（Flamingo 等） | 语言 token | 视觉 token |
| Perceiver | 少量 latent | 大量输入 |

## 常见追问

- **mask 该加在哪？** 只屏蔽 context 里的 padding；query 的 padding 位置输出被丢弃即可。
- **为什么 cross 的 KV 可以缓存？** context 在整个解码过程中不变，K/V 只需算一次，见 [KV Cache](05_KVCache.md)。
- **T、S 不同会不会影响 softmax 归一化？** 归一化沿 S，每个 query 对整个 context 的权重和为 1。
- **encoder 输出维度和 decoder 不同怎么办？** Wk、Wv 的输入维度设成 context 维度即可（上面自检里 Dq≠Dkv）。

## NumPy 版（不能用 torch 时）

完整可运行，不依赖其他文件：

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


def cross_attention(x, context, Wq, Wk, Wv, context_mask=None):
    # x:(B,T,Dq)  context:(B,S,Dkv)  context_mask:(B,S) bool，True=真实 token
    mask = context_mask[:, None, :] if context_mask is not None else None   # (B,1,S)
    return scaled_dot_product_attention(x @ Wq, context @ Wk, context @ Wv, mask)


# 自检
rng = np.random.default_rng(0)
B, T, S, Dq, Dkv, dk = 2, 5, 9, 16, 12, 8
x, ctx = rng.normal(size=(B, T, Dq)), rng.normal(size=(B, S, Dkv))
Wq, Wk, Wv = rng.normal(size=(Dq, dk)), rng.normal(size=(Dkv, dk)), rng.normal(size=(Dkv, dk))
ctx_mask = np.array([9, 6])[:, None] > np.arange(S)[None, :]            # (B,S)
out = cross_attention(x, ctx, Wq, Wk, Wv, ctx_mask)
print(out.shape)                                                         # (2, 5, 8)：长度跟 query 走
ctx2 = ctx.copy(); ctx2[1, 6:] = 100                                     # 改 padding 位置
print(np.allclose(cross_attention(x, ctx2, Wq, Wk, Wv, ctx_mask), out))  # True
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


def cross_attention(x: torch.Tensor, context: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                    Wv: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Queries from x, keys/values from context (e.g. decoder attends to encoder output).

    Args:
        x: (B, T, Dq); context: (B, S, Dkv) -- T and S may differ.
        context_mask: (B, S) bool, True = real token, False = padding; or None.
    Returns:
        (B, T, dv). No causal mask: every query may see the whole context.
    """
    mask = context_mask[:, None, :] if context_mask is not None else None  # (B, 1, S)
    return attention(x, context, Wq, Wk, Wv, mask)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



