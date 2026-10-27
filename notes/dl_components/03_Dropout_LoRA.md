# Dropout / LoRA

- [ ] 读懂　- [ ] 默写通过　- [ ] 能答追问　- [ ] 面试前再过一遍


**Inverted Dropout**：训练时 `x * mask / (1-p)`，评估时恒等。缩放放在训练期，推理零额外开销，且保持期望不变。

**LoRA**：`y = W₀x + (α/r)·B(Ax)`，`A∈R^{r×din}` 随机、`B∈R^{dout×r}` **全 0**。

- 为什么不都为 0：A、B 梯度互相依赖（`∂L/∂B ∝ A`，`∂L/∂A ∝ Bᵀ`），都为 0 则梯度全 0，永远不更新。
- 为什么不都随机：初始 `ΔW≠0` 会扰动预训练模型。
- 只有 A 随机、B=0：起点 `ΔW=0`，且 B 能立刻得到非零梯度。
- 推理合并：`W = W₀ + (α/r)·BA`，零延迟（代码里 `merge()`，已验证合并前后输出一致）。
- 参数量：`r(din+dout)` vs `din·dout`。

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


class InvertedDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: any shape -> same shape. Identity in eval mode."""
        if not self.training or self.p == 0:
            return x
        keep = 1.0 - self.p
        return x * (torch.rand_like(x) < keep).to(x.dtype) / keep


class LoRALinear(nn.Module):
    def __init__(self, din: int, dout: int, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = nn.Linear(din, dout)
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, din) / r ** 0.5)   # random
        self.B = nn.Parameter(torch.zeros(dout, r))             # zeros => delta W = 0 at start
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., din) -> (..., dout)."""
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the low-rank update into the base weight (zero inference overhead)."""
        self.base.weight += self.scale * self.B @ self.A        # (dout, din)
```

</details>
<!-- ANSWER-END -->

---

## ✏️ 我的笔记

> 在这里写：卡壳的地方、自己的口述版回答、面试官的追问



