# BatchNorm vs LayerNorm（vs RMSNorm）

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


| | BatchNorm | LayerNorm | RMSNorm |
|---|---|---|---|
| 统计维度 | 沿 batch（1D: dim 0；2D: dim (0,2,3)），逐通道 | 沿特征维 dim=-1，逐样本 | 同 LN，但不减均值 |
| train / eval | 训练用 batch 统计 + EMA 更新 running；评估用 running | 无差异 | 无差异 |
| 小 batch / 变长序列 | 差 | 稳 | 稳 |

```python
if self.training:
    mean = x.mean(0, keepdim=True)
    var  = x.var(0, keepdim=True, unbiased=False)       # 归一化用有偏方差
    with torch.no_grad():                               # running_var 用无偏方差（与 PyTorch 一致）
        running_mean.mul_(1-m).add_(m*mean)
        running_var.mul_(1-m).add_(m*x.var(0, keepdim=True, unbiased=True))
else:
    mean, var = running_mean, running_var
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

- `running_*` 不是 Parameter（不求梯度），但要进 `state_dict` → `register_buffer`。
- RMSNorm：`x · γ / sqrt(mean(x²)+ε)`，省去均值计算，LLM 常用，更快且效果相当。
- 与 `torch.nn.BatchNorm1d/LayerNorm` 逐项对照通过（含 running_var）。

<!-- ANSWER-START -->
## ✅ 标准答案

> 先自己写，再点开对照。代码来自 [code/ml_coding.py](../../code/ml_coding.py)，已通过 `code/test_ml_coding.py` 与 `practice/run.py` 的判分。

<details>
<summary>点击展开标准答案</summary>

```python
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    def __init__(self, n: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(1, n))
        self.beta = nn.Parameter(torch.zeros(1, n))
        self.register_buffer('running_mean', torch.zeros(1, n))
        self.register_buffer('running_var', torch.ones(1, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C) -> (B, C)."""
        if self.training:
            mean = x.mean(0, keepdim=True)                      # (1, C)
            var_biased = x.var(0, keepdim=True, unbiased=False) # (1, C) used to normalise
            var_unbiased = x.var(0, keepdim=True, unbiased=True)  # (1, C) used for running stats
            with torch.no_grad():                               # buffers: no gradient
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var_unbiased)
            var = var_biased
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)         # (B, C)
        return self.gamma * x_hat + self.beta


class LayerNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma, self.beta = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n) -> (..., n); normalises over the last dim."""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps, self.gamma = eps, nn.Parameter(torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n) -> (..., n)."""
        return self.gamma * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



