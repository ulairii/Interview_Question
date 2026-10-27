# Multi-Head Attention（多头注意力）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍

导航：[总览](00_总览.md) · [SDPA 与 Mask](01_SDPA与Mask.md) · [Self-Attention](02_SelfAttention.md) · [Cross-Attention](03_CrossAttention.md) · [Multi-Head](04_MultiHead.md) · [KV Cache](05_KVCache.md) · [追问与验证](06_追问与验证.md)

## 思路

把 `C` 维切成 `H` 个头、每头 `dk = C/H`，**各头独立做一次 SDPA**，再把结果拼回 `C` 维，最后过输出投影 `Wo`。总计算量与单头（维度 C）相同。

为什么：不同头可以在不同子空间关注不同关系（位置、语法、共指……），表达力更强。

## 第 1 步：函数版，不带 mask（先写这个）

完整可运行，不依赖其他文件：

```python
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5   # (B,H,T,S)
    return F.softmax(scores, dim=-1) @ v                    # (B,H,T,dk)


def multi_head_attention(x_q, x_kv, Wq, Wk, Wv, Wo, num_heads):
    # x_q:(B,T,C)  x_kv:(B,S,C)  W*:(C,C)
    B, T, C = x_q.shape
    S, H = x_kv.size(1), num_heads
    q = (x_q  @ Wq).view(B, T, H, C // H).transpose(1, 2)  # (B,H,T,dk)
    k = (x_kv @ Wk).view(B, S, H, C // H).transpose(1, 2)  # (B,H,S,dk)
    v = (x_kv @ Wv).view(B, S, H, C // H).transpose(1, 2)  # (B,H,S,dk)
    out = scaled_dot_product_attention(q, k, v)            # (B,H,T,dk) 各头并行
    out = out.transpose(1, 2).contiguous().view(B, T, C)   # (B,T,C) 拼接各头
    return out @ Wo


# 自检：和 PyTorch 的 nn.MultiheadAttention 一致
B, T, C, H = 2, 6, 32, 4
x = torch.randn(B, T, C)
mha = torch.nn.MultiheadAttention(C, H, bias=False, batch_first=True)
Wq, Wk, Wv = (w.T for w in mha.in_proj_weight.chunk(3))  # PyTorch 存的是 (out,in)，转置成 (in,out)
Wo = mha.out_proj.weight.T
ref, _ = mha(x, x, x, need_weights=False)
print(torch.allclose(multi_head_attention(x, x, Wq, Wk, Wv, Wo, H), ref, atol=1e-5))   # True
```

- `x_kv` 传 `x` → 多头**自**注意力；`x_kv` 传 `context` → 多头**交叉**注意力。

## 第 2 步：加 mask

各头共享同一个 mask，形状广播成 `(B,1,T,S)`：

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


def multi_head_attention(x_q, x_kv, Wq, Wk, Wv, Wo, num_heads, mask=None):
    # mask: bool，可广播到 (B,1,T,S)，各头共享
    B, T, C = x_q.shape
    S, H = x_kv.size(1), num_heads
    q = (x_q  @ Wq).view(B, T, H, C // H).transpose(1, 2)
    k = (x_kv @ Wk).view(B, S, H, C // H).transpose(1, 2)
    v = (x_kv @ Wv).view(B, S, H, C // H).transpose(1, 2)
    out = scaled_dot_product_attention(q, k, v, mask)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return out @ Wo


# 自检：causal 多头自注意力和 PyTorch 一致
B, T, C, H = 2, 6, 32, 4
x = torch.randn(B, T, C)
mha = torch.nn.MultiheadAttention(C, H, bias=False, batch_first=True)
Wq, Wk, Wv = (w.T for w in mha.in_proj_weight.chunk(3))
Wo = mha.out_proj.weight.T
ref, _ = mha(x, x, x, attn_mask=~causal_mask(T, T), need_weights=False)   # PyTorch 约定：True=屏蔽
print(torch.allclose(multi_head_attention(x, x, Wq, Wk, Wv, Wo, H, causal_mask(T, T)), ref, atol=1e-5))  # True
```

## shape 变换逐步看（B=2, T=6, C=32, H=4, dk=8）

| 步骤 | 操作 | shape |
|---|---|---|
| 1 | `x @ Wq` | (2, 6, 32) |
| 2 | `.view(B,T,H,dk)` **把最后一维切成 H×dk** | (2, 6, 4, 8) |
| 3 | `.transpose(1,2)` 把头换到 batch 旁 | (2, 4, 6, 8) |
| 4 | SDPA（各头并行，批量 matmul） | (2, 4, 6, 8)，权重 (2,4,6,S) |
| 5 | `.transpose(1,2)` 换回 | (2, 6, 4, 8) |
| 6 | `.contiguous().view(B,T,C)` **拼接头** | (2, 6, 32) |
| 7 | `@ Wo` 混合各头 | (2, 6, 32) |

要点：
- 第 2 步 `view` 要求最后一维连续，`C` 必须能被 `H` 整除。
- 第 5→6 步 `transpose` 后内存不连续，**必须 `.contiguous()` 再 `.view`**（或直接 `.reshape`）。
- 不要写成 `view(B, H, T, dk)` 直接重排 `(B,T,C)`——那会把序列和通道混起来，是静默错误。

## 第 3 步：Module 版（被要求写 nn.Module 时）

完整可运行：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=False):
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    w = F.dropout(F.softmax(scores, dim=-1), dropout_p, training)
    return w @ v


def causal_mask(T, S, device=None):
    return torch.ones(T, S, dtype=torch.bool, device=device).tril(diagonal=S - T)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dk = num_heads, d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _split(self, x):                                   # (B,L,C) -> (B,H,L,dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)

    def forward(self, x, context=None, mask=None, is_causal=False):
        # context=None → 自注意力；传入 context → 交叉注意力
        B, T, C = x.shape
        src = x if context is None else context
        q, k, v = self._split(self.q_proj(x)), self._split(self.k_proj(src)), self._split(self.v_proj(src))
        if is_causal:
            cm = causal_mask(T, k.size(2), x.device)
            mask = cm if mask is None else mask & cm
        out = scaled_dot_product_attention(q, k, v, mask, self.dropout_p, self.training)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# 自检：自注意力 causal、交叉注意力 padding 都能跑通，shape 正确
B, T, S, C, H = 2, 6, 9, 32, 4
mha = MultiHeadAttention(C, H).eval()
x, enc = torch.randn(B, T, C), torch.randn(B, S, C)
pad = (torch.tensor([9, 5])[:, None] > torch.arange(S))[:, None, None, :]   # (B,1,1,S)
print(mha(x, is_causal=True).shape, mha(x, context=enc, mask=pad).shape)    # (2,6,32) (2,6,32)
```

用法：
```python
y = mha(x, is_causal=True)                        # decoder 自注意力
y = mha(x, context=enc_out, mask=pad_mask)        # 交叉注意力，pad_mask:(B,1,1,S)
```

## 融合 QKV 投影（工程常见写法，口述即可）

```python
qkv = self.qkv(x).reshape(B, T, 3, H, dk).permute(2, 0, 3, 1, 4)   # (3,B,H,T,dk)
q, k, v = qkv[0], qkv[1], qkv[2]
```

只对自注意力有效（Q、K、V 同输入）。交叉注意力的 K/V 输入不同于 Q，需要分开投影，所以上面的 Module 用三个独立 Linear。

## 变体（口述）

| 变体 | 做法 | 好处 |
|---|---|---|
| MQA | 所有 query 头共享 **1 组** K/V | KV cache 缩 H 倍 |
| GQA | query 头分 G 组，每组共享 K/V | 质量与缓存的折中（LLaMA-2/3） |
| MLA | K/V 压缩到低秩 latent | 进一步压缩缓存（DeepSeek） |

## 参数量

`Wq, Wk, Wv, Wo` 各 `C×C`，共 `4C²`（不含 bias），与头数 H 无关。

## NumPy 版（不能用 torch 时）

NumPy 没有 `view` 和 `contiguous`，用 `reshape` + `transpose` 即可：`reshape` 会在需要时自动拷贝，不存在"内存不连续"的报错。

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])   # (B,H,T,S)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores, axis=-1) @ v


def causal_mask(T, S):
    return np.tril(np.ones((T, S), dtype=bool), k=S - T)


def multi_head_attention(x_q, x_kv, Wq, Wk, Wv, Wo, num_heads, mask=None):
    # x_q:(B,T,C)  x_kv:(B,S,C)  W*:(C,C)  mask 可广播到 (B,1,T,S)
    B, T, C = x_q.shape
    S, H = x_kv.shape[1], num_heads
    dk = C // H
    q = (x_q  @ Wq).reshape(B, T, H, dk).transpose(0, 2, 1, 3)  # (B,H,T,dk)
    k = (x_kv @ Wk).reshape(B, S, H, dk).transpose(0, 2, 1, 3)  # (B,H,S,dk)
    v = (x_kv @ Wv).reshape(B, S, H, dk).transpose(0, 2, 1, 3)  # (B,H,S,dk)
    out = scaled_dot_product_attention(q, k, v, mask)            # (B,H,T,dk)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)             # (B,T,C) 拼接各头
    return out @ Wo


# 自检：多头 == 每个头单独算一遍再拼接（for 循环版）
rng = np.random.default_rng(0)
B, T, C, H = 2, 6, 32, 4
dk = C // H
x = rng.normal(size=(B, T, C))
Wq, Wk, Wv, Wo = (rng.normal(size=(C, C)) / np.sqrt(C) for _ in range(4))
heads = []
for h in range(H):                                       # 第 h 个头用 W 的第 h 段列
    sl = slice(h * dk, (h + 1) * dk)
    heads.append(scaled_dot_product_attention(x @ Wq[:, sl], x @ Wk[:, sl], x @ Wv[:, sl],
                                              causal_mask(T, T)))
loop = np.concatenate(heads, axis=-1) @ Wo
print(np.allclose(multi_head_attention(x, x, Wq, Wk, Wv, Wo, H, causal_mask(T, T)), loop))   # True
```

自检的思路就是多头的定义本身：第 h 个头只用 `Wq/Wk/Wv` 的第 h 段列，各头算完沿最后一维拼起来，再乘 `Wo`。

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


def multi_head_attention(x_q: torch.Tensor, x_kv: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor, num_heads: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Functional multi-head attention (self if x_kv is x_q, cross otherwise).

    Args:
        x_q: (B, T, C); x_kv: (B, S, C).
        Wq, Wk, Wv, Wo: (C, C) projections, C divisible by num_heads.
        mask: bool broadcastable to (B, 1, T, S) (shared across heads), or None.
    Returns:
        (B, T, C).
    """
    B, T, C = x_q.shape
    S, H = x_kv.size(1), num_heads
    q = (x_q @ Wq).view(B, T, H, C // H).transpose(1, 2)        # (B, H, T, dk)
    k = (x_kv @ Wk).view(B, S, H, C // H).transpose(1, 2)       # (B, H, S, dk)
    v = (x_kv @ Wv).view(B, S, H, C // H).transpose(1, 2)       # (B, H, S, dk)
    out, _ = scaled_dot_product_attention(q, k, v, mask)        # (B, H, T, dk)
    out = out.transpose(1, 2).contiguous().view(B, T, C)        # (B, T, C), heads concatenated
    return out @ Wo
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



